"""
创新点 1 (最终版): GNN 从真实 FC 预测 ASD
关键修正:
  - 特征: 用 FC 的 row features (mean, std, max) + 节点 degree centrality + 原始对角线值
  - Graph batch 训练, 用 DataLoader 加速
  - LR baseline (FC flatten) 先对比
"""
import warnings, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)

CACHE = r"D:/ASD-GraphNet - mhx/data/cache/fc_all.npz"
fc_data = dict(np.load(CACHE))
df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
sub = df[(df["graph_feature_type"]=="node_based")&(df["feature"]=="degree")&(df["feature_engineering"]=="original")&(df["atlas"]=="cc200")].drop_duplicates("id")

THR = 0.5
graphs, y_list = [], []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid not in fc_data: continue
    fc = fc_data[sid].copy()
    if not np.all(np.isfinite(fc)):
        fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 1.0)
    adj = (np.abs(fc) > THR).astype(float); np.fill_diagonal(adj, 0)
    if adj.sum() < 20: continue
    # 稀疏化: 只保留最强的 K 条边
    K = 1000
    if adj.sum() > K:
        flat = np.argwhere(adj > 0)
        vals = np.array([np.abs(fc[i, j]) for i, j in flat])
        keep = flat[np.argsort(vals)[::-1][:K]]
        adj2 = np.zeros_like(adj)
        adj2[keep[:, 0], keep[:, 1]] = 1
        adj = adj2
    # 节点特征: 每个 ROI 的 7 个统计量
    feats = np.stack([fc.mean(1), fc.std(1), fc.max(1), fc.min(1), adj.sum(1)/199, np.diag(fc), np.abs(fc).mean(1)], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    # 特征标准化 (全局)
    x = torch.tensor(feats, dtype=torch.float)
    ei = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)
    graphs.append(Data(x=x, edge_index=ei, y=torch.tensor([int(r["ASD"])], dtype=torch.long)))
    y_list.append(int(r["ASD"]))
print(f"Graphs: {len(graphs)} ASD={sum(y_list)} TDC={len(y_list)-sum(y_list)}")

# 特征全局标准化
all_x = torch.cat([g.x for g in graphs], dim=0)
mu, sd = all_x.mean(0), all_x.std(0) + 1e-8
for g in graphs:
    g.x = (g.x - mu) / sd

class GCNNet(torch.nn.Module):
    def __init__(self, in_dim=7, hid=64):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.classifier = torch.nn.Linear(hid * 3, 2)
    def forward(self, x, ei, batch):
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.c2(x, ei)); x = F.dropout(x, 0.3, self.training)
        pooled = torch.cat([global_add_pool(x,batch), global_mean_pool(x,batch), global_max_pool(x,batch)], dim=1)
        return self.classifier(pooled)

torch.set_num_threads(4)

def run_cv(gnn=True, epochs=50):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, senss, specs = [], [], []
    for fold, (tr, te) in enumerate(cv.split(graphs, y_list)):
        tr_g = [graphs[i] for i in tr]; te_g = [graphs[i] for i in te]
        yt = np.array(y_list)[te]
        tr_loader = DataLoader(tr_g, batch_size=32, shuffle=True)
        te_loader = DataLoader(te_g, batch_size=64, shuffle=False)
        
        model = GCNNet()
        ytr_arr = np.array(y_list)[tr]
        w = torch.tensor([len(ytr_arr)/(2*max((ytr_arr==0).sum(),1)),
                          len(ytr_arr)/(2*max((ytr_arr==1).sum(),1))], dtype=torch.float)
        crit = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        model.train()
        for ep in range(epochs):
            for b in tr_loader:
                opt.zero_grad()
                out = model(b.x, b.edge_index, b.batch)
                loss = crit(out, b.y)
                loss.backward(); opt.step()
        
        model.eval()
        preds = []
        with torch.no_grad():
            for b in te_loader:
                out = model(b.x, b.edge_index, b.batch)
                preds.extend(out.argmax(1).tolist())
        
        acc = accuracy_score(yt, preds) * 100
        cm = confusion_matrix(yt, preds)
        if cm.shape == (2,2):
            tn,fp,fn,tp = cm.ravel()
            senss.append(tp/(tp+fn)*100 if tp+fn else 0)
            specs.append(tn/(tn+fp)*100 if tn+fp else 0)
        accs.append(acc)
        print(f"  GCN fold {fold+1}: {acc:.2f}% sens={senss[-1]:.0f}%", flush=True)
    return np.mean(accs), np.std(accs), np.mean(senss), np.mean(specs)

print("\n=== GCN (GNN) ===")
a, s, se, sp = run_cv(gnn=True, epochs=40)
print(f"GCN: acc={a:.2f}% +/- {s:.2f}%, sens={se:.1f}%, spec={sp:.1f}%")
print(f"\n(论文 75.25%, 原始复现 75.63%)")
