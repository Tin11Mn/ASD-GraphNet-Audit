"""
创新点 3 (修正版): 多视图自监督预训练 + 跨站点迁移
- 预训练: SimCLR 风格对比学习 (cc200/aal/dos160 互为正样本对)
- 评估1: 10-fold CV (同站点分布)
- 评估2: Leave-One-Site-Out 跨站点迁移 (核心检验)
  对比: 预训练 encoder vs 随机初始化 encoder
"""
import warnings, ast, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
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
    return Data(x=torch.tensor(feats, dtype=torch.float),
                edge_index=torch.tensor(np.argwhere(adj>0).T, dtype=torch.long))

ids, ys, site_ids = [], [], []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid in fc_cc200 and sid in fc_aal and sid in fc_dos:
        ids.append(sid); ys.append(int(r["ASD"])); site_ids.append(r["SITE_ID"])
ys = np.array(ys); site_ids = np.array(site_ids)
print(f"Subjects: {len(ids)}, sites: {len(set(site_ids))}")

g_cc = [build_graph(fc_cc200[s]) for s in ids]
g_aa = [build_graph(fc_aal[s]) for s in ids]
g_do = [build_graph(fc_dos[s]) for s in ids]

class Encoder(torch.nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.c1 = GCNConv(3, hid); self.c2 = GCNConv(hid, hid)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.c2(x, ei)); x = F.dropout(x, 0.3, self.training)
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))

from torch_geometric.nn import GCNConv, global_mean_pool

class Proj(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d,d), torch.nn.ReLU(), torch.nn.Linear(d,d))
    def forward(self, z):
        return F.normalize(self.mlp(z), dim=1)

def info_nce(z1, z2, T=0.5):
    sim = torch.exp(torch.mm(z1, z2.t())/T)
    return -torch.log(torch.diag(sim)/sim.sum(1)).mean()

def batches(views, idx, bs=64, shuffle=True):
    """返回 [(Batch_view1, Batch_view2, Batch_view3, labels)]"""
    out = []
    idx = np.array(idx)
    if shuffle: np.random.shuffle(idx)
    for s in range(0, len(idx), bs):
        sel = idx[s:s+bs]
        bl = [Batch.from_data_list([views[v][i] for i in sel]) for v in range(3)]
        out.append((bl[0], bl[1], bl[2], torch.tensor(ys[sel]).long()))
    return out

def pretrain(epochs=30, hid=32, lr=1e-3):
    enc = Encoder(hid); proj = Proj(hid)
    opt = torch.optim.Adam(list(enc.parameters())+list(proj.parameters()), lr=lr)
    views = [g_cc, g_aa, g_do]
    for ep in range(epochs):
        enc.train(); proj.train()
        tot, nb = 0, 0
        for b0, b1, b2, yy in batches(views, np.arange(len(ids))):
            opt.zero_grad()
            z0 = proj(enc(b0.x, b0.edge_index))
            z1 = proj(enc(b1.x, b1.edge_index))
            z2 = proj(enc(b2.x, b2.edge_index))
            L = info_nce(z0, z1) + info_nce(z0, z2) + info_nce(z1, z2)
            L.backward(); opt.step()
            tot += L.item(); nb += 1
        if ep % 10 == 0:
            print(f"  ep{ep} loss={tot/nb:.4f}", flush=True)
    return enc, proj

def extract(enc, proj, views, idx):
    enc.eval(); proj.eval()
    out = []
    for v in views:
        with torch.no_grad():
            embs = [proj(enc(g.x, g.edge_index)).numpy().reshape(1,-1) for g in v]
        out.append(np.concatenate(embs, 0))
    return np.concatenate(out, axis=1)

print("\n=== 预训练 (多视图对比学习) ===")
enc, proj = pretrain(epochs=30)

E_pretrained = extract(enc, proj, [g_cc, g_aa, g_do], np.arange(len(ids)))
E_random = extract(Encoder(32), Proj(32), [g_cc, g_aa, g_do], np.arange(len(ids)))
print(f"Emb dim: {E_pretrained.shape[1]}")

def cv_eval(E, n_splits=10):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(E, ys):
        sc = StandardScaler(); Xtr = sc.fit_transform(E[tr]); Xte = sc.transform(E[te])
        lr = LogisticRegression(max_iter=2000, C=0.1); lr.fit(Xtr, ys[tr])
        accs.append(accuracy_score(ys[te], lr.predict(Xte))*100)
    return np.mean(accs), np.std(accs)

def loso_eval(E):
    """Leave-One-Site-Out: 训练 16 站点, 测试留出站点"""
    accs, details = [], []
    for s in sorted(set(site_ids)):
        te = np.where(site_ids == s)[0]
        tr = np.where(site_ids != s)[0]
        if len(te) < 10: continue
        sc = StandardScaler(); Xtr = sc.fit_transform(E[tr]); Xte = sc.transform(E[te])
        lr = LogisticRegression(max_iter=2000, C=0.1); lr.fit(Xtr, ys[tr])
        a = accuracy_score(ys[te], lr.predict(Xte))*100
        accs.append(a); details.append((s, len(te), a))
    return np.mean(accs), details

print("\n=== 评估1: 10-fold CV (同站点) ===")
a1, s1 = cv_eval(E_pretrained); print(f"预训练 embedding: {a1:.2f}% +/- {s1:.2f}%")
a2, s2 = cv_eval(E_random);    print(f"随机初始化:      {a2:.2f}% +/- {s2:.2f}%")

print("\n=== 评估2: Leave-One-Site-Out 跨站点迁移 ===")
a3, d3 = loso_eval(E_pretrained)
print(f"预训练 embedding: {a3:.2f}%")
for s,n,a in d3: print(f"    {s:10s} n={n:3d} acc={a:.1f}%")
a4, d4 = loso_eval(E_random)
print(f"随机初始化:      {a4:.2f}%")

# 基线: 论文的 degree 特征, 同样做 LOSO
print("\n=== 基线: 论文 degree 特征 LOSO ===")
deg_df = df[(df["graph_feature_type"]=="node_based")&(df["feature"]=="degree")
            &(df["feature_engineering"]=="original")&(df["atlas"]=="cc200")].drop_duplicates("id")
deg_map = {str(r["id"]): ast.literal_eval(r["features_value"]) for _, r in deg_df.iterrows()}
E_deg = np.array([deg_map[i] for i in ids], dtype=float)
a5, d5 = loso_eval(E_deg)
print(f"论文 degree 特征 LOSO: {a5:.2f}%")
a6, s6 = cv_eval(E_deg)
print(f"论文 degree 特征 10-fold CV: {a6:.2f}% +/- {s6:.2f}%")
