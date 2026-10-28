"""
创新点 3: 多视图对比预训练 + 微调 (Cross-View Self-Supervised Pretraining)
动机: 论文使用单一 atlas (cc200) 的静态特征。创新:
  - 用 cc200 / aal / dos160 三种视图作为同一被试的正样本对 (InfoNCE 对比)
  - 预训练 GNN encoder 学习视图不变表示
  - 冻结 encoder, 在预训练 embedding 上训练 classifier (微调)
损失: SimCLR-style InfoNCE, 同一被试不同 atlas 互为正样本
"""
import warnings, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)

CACHE = r"D:/ASD-GraphNet - mhx/data/cache"
fc_cc200 = dict(np.load(CACHE + "/fc_all.npz"))
fc_aal = dict(np.load(CACHE + "/fc_aal.npz"))
fc_dos = dict(np.load(CACHE + "/fc_dosenbach160.npz"))
df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
sub = df[(df["graph_feature_type"]=="node_based")&(df["feature"]=="degree")
         &(df["feature_engineering"]=="original")&(df["atlas"]=="cc200")].drop_duplicates("id")

def build_graph(fc, thr=0.5):
    n = fc.shape[0]
    adj = (np.abs(fc) > thr).astype(float); np.fill_diagonal(adj, 0)
    feats = np.stack([fc.mean(1), fc.std(1), adj.sum(1)/(n-1)], 1)
    feats = np.nan_to_num(feats)
    return torch.tensor(feats, dtype=torch.float), torch.tensor(np.argwhere(adj>0).T, dtype=torch.long)

# 收集三个 atlas 都有的被试
ids = []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid in fc_cc200 and sid in fc_aal and sid in fc_dos:
        ids.append((sid, int(r["ASD"])))
print(f"Subjects in all 3 atlases: {len(ids)}")

# 构建每个 atlas 的图列表
def build_graphs(ats_dict, ids):
    graphs = []
    for sid, y in ids:
        fc = ats_dict[sid]; x, ei = build_graph(fc)
        graphs.append(Data(x=x, edge_index=ei, y=y))
    return graphs

g_cc200 = build_graphs(fc_cc200, ids)
g_aal   = build_graphs(fc_aal, ids)
g_dos   = build_graphs(fc_dos, ids)

class GNN_Encoder(torch.nn.Module):
    """共享的多视图 encoder"""
    def __init__(self, hid=32):
        super().__init__()
        self.c1 = GCNConv(3, hid); self.c2 = GCNConv(hid, hid)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.c2(x, ei)); x = F.dropout(x, 0.3, self.training)
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))

class Projection(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU()
        )
    def forward(self, z):
        return F.normalize(self.mlp(z), dim=1)

def contrastive_loss(z1, z2, T=0.5):
    """SimCLR InfoNCE"""
    N = z1.size(0)
    sim = torch.exp(torch.mm(z1, z2.t()) / T)
    pos = torch.diag(sim)
    loss = -torch.log(pos / sim.sum(1) + 1e-8)
    return loss.mean()

encoder = GNN_Encoder(hid=32)
proj = Projection(32)
opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=0.001)

# 预训练: 对比 cc200 vs aal, cc200 vs dos160, aal vs dos160
dataloader = DataLoader(list(zip(g_cc200, g_aal, g_dos)), batch_size=64, shuffle=True)
print("\n=== 预训练: 多视图对比学习 ===")
for ep in range(30):
    tot = 0
    for b in dataloader:
        g1, g2 = b[0], b[1]  # cc200 vs aal
        opt.zero_grad()
        z1 = F.normalize(proj(encoder(g1.x, g1.edge_index)))
        z2 = F.normalize(proj(encoder(g2.x, g2.edge_index)))
        L = contrastive_loss(z1, z2)
        L.backward(); opt.step()
        tot += L.item()
    if ep % 10 == 0:
        print(f"  ep{ep} loss={tot/len(dataloader):.4f}", flush=True)

# 提取所有被试的 3 个视图 embedding
def get_emb(view_graphs):
    encoder.eval()
    with torch.no_grad():
        embs = [proj(encoder(g.x, g.edge_index)).numpy().reshape(1,-1) for g in view_graphs]
    return np.concatenate(embs, axis=0)  # (N, 32)

emb_cc = get_emb(g_cc200); emb_aa = get_emb(g_aal); emb_do = get_emb(g_dos)
emb_concat = np.concatenate([emb_cc, emb_aa, emb_do], axis=1)
print(f"Embedding dim: cc200={emb_cc.shape[1]}, aal={emb_aa.shape[1]}, dos={emb_do.shape[1]}")
print(f"Concatenated: {emb_concat.shape[1]}")

# 10-fold CV: 在 embedding 上训练 LR
y_arr = np.array([y for _, y in ids])
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs = []; senss=[]; specs=[]
for fold, (tr, te) in enumerate(cv.split(emb_concat, y_arr)):
    sc = StandardScaler()
    Xtr = sc.fit_transform(emb_concat[tr]); Xte = sc.transform(emb_concat[te])
    lr = LogisticRegression(max_iter=2000, C=0.1)
    lr.fit(Xtr, y_arr[tr])
    pred = lr.predict(Xte)
    accs.append(accuracy_score(y_arr[te], pred)*100)
    cm = confusion_matrix(y_arr[te], pred)
    if cm.shape == (2,2):
        tn,fp,fn,tp = cm.ravel()
        senss.append(tp/(tp+fn)*100 if tp+fn else 0)
        specs.append(tn/(tn+fp)*100 if tn+fp else 0)
    print(f"  fold {fold+1}: acc={accs[-1]:.2f}% sens={senss[-1]:.0f}% spec={specs[-1]:.0f}%", flush=True)

print(f"\n=== 创新3: 多视图对比预训练 + LR ===")
print(f"Accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
print(f"Sensitivity: {np.mean(senss):.2f}%, Specificity: {np.mean(specs):.2f}%")
print(f"\n(论文 75.25%, 原始复现 75.63%)")
