"""
创新点 1 (最终): 融合 GNN 与 LR
- LR on flattened FC (baseline 69.69%)
- GNN on real FC → extract graph embedding (200-node, 1-layer)
- Concat GNN embedding + LR features → 集成模型
- 10-fold CV
"""
import warnings, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)

fc_data = dict(np.load(r"D:/ASD-GraphNet - mhx/data/cache/fc_all.npz"))
df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
sub = df[(df["graph_feature_type"]=="node_based")&(df["feature"]=="degree")
         &(df["feature_engineering"]=="original")&(df["atlas"]=="cc200")].drop_duplicates("id")

IU = np.triu_indices(200, k=1)
graphs, y_list, X_flat = [], [], []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid not in fc_data: continue
    fc = np.nan_to_num(fc_data[sid], nan=0.0); np.fill_diagonal(fc, 1.0)
    adj = (np.abs(fc)>0.5).astype(float); np.fill_diagonal(adj,0)
    K = 1000
    if adj.sum() > K:
        flat = np.argwhere(adj>0)
        vals = np.array([np.abs(fc[i,j]) for i,j in flat])
        keep = flat[np.argsort(vals)[::-1][:K]]
        adj[keep[:,0], keep[:,1]] = 1
    feats = np.stack([fc.mean(1), fc.std(1), fc.max(1), fc.min(1),
                      adj.sum(1)/199, np.diag(fc), np.abs(fc).mean(1)],1)
    feats = np.nan_to_num(feats)
    graphs.append(Data(x=torch.tensor(feats,dtype=torch.float),
                       edge_index=torch.tensor(np.argwhere(adj>0).T,dtype=torch.long),
                       y=torch.tensor([int(r["ASD"])],dtype=torch.long)))
    X_flat.append(fc[IU])
    y_list.append(int(r["ASD"]))
y_list = np.array(y_list); X_flat = np.array(X_flat)

class GCNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(7, 16)
        self.fc = torch.nn.Linear(16, 8)
    def forward(self, x, ei, b):
        x = F.relu(self.conv(x, ei))
        return F.relu(self.fc(global_mean_pool(x, b)))

# LR 基线
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs_lr, accs_gnn, accs_gnn_lr = [], [], []
for fold, (tr, te) in enumerate(cv.split(X_flat, y_list)):
    sc = StandardScaler(); Xtr = sc.fit_transform(X_flat[tr]); Xte = sc.transform(X_flat[te])
    lr = LogisticRegression(max_iter=2000, C=0.1)
    lr.fit(Xtr, y_list[tr])
    accs_lr.append(accuracy_score(y_list[te], lr.predict(Xte))*100)
    
    # GNN: 全图一起训练 (batch_size=全部)
    all_Xtr = np.stack([g.x.numpy() for g in [graphs[i] for i in tr]])
    all_ytr = y_list[tr]
    all_Xte = np.stack([g.x.numpy() for g in [graphs[i] for i in te]])
    all_yte = y_list[te]
    # 预处理 FC 特征 (LR 部分) + GNN 输出 concat
    # 简单策略: 用 pre-trained GNN encoder (shared across folds for fair eval? no.)
    # 简化: 训练 GNN 只用于提取 embedding，然后用 LR on concatenation
    # Step 1: train GNN on tr
    tr_g = [graphs[i] for i in tr]
    m = GCNNet(); opt = torch.optim.Adam(m.parameters(), lr=0.005); crit = torch.nn.CrossEntropyLoss()
    loader = DataLoader(tr_g, batch_size=32, shuffle=True)
    for ep in range(20):
        for b in loader:
            opt.zero_grad(); out = crit(m(b.x, b.edge_index, b.batch), b.y); out.backward(); opt.step()
    m.eval()
    # 提取所有被试的 embedding
    with torch.no_grad():
        emb_tr = [m(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long)).numpy() for g in tr_g]
        te_g = [graphs[i] for i in te]
        emb_te = [m(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long)).numpy() for g in te_g]
    
    emb_tr = np.concatenate(emb_tr); emb_te = np.concatenate(emb_te)
    # LR on concatenated features
    sc2 = StandardScaler(); Xc_tr = sc2.fit_transform(np.concatenate([Xtr, emb_tr], axis=1))
    Xc_te = sc2.transform(np.concatenate([Xte, emb_te], axis=1))
    lr2 = LogisticRegression(max_iter=2000, C=0.1)
    lr2.fit(Xc_tr, y_list[tr])
    accs_gnn_lr.append(accuracy_score(y_list[te], lr2.predict(Xc_te))*100)
    
    # GNN only (直接预测)
    pred_g = [m(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long)).argmax(1).numpy().item() for g in te_g]
    accs_gnn.append(accuracy_score(y_list[te], pred_g)*100)
    print(f"Fold {fold+1}: LR={accs_lr[-1]:.1f}% GNN={accs_gnn[-1]:.1f}% LR+GNN={accs_gnn_lr[-1]:.1f}%", flush=True)

print(f"\n=== 创新1 结果 ===")
print(f"LR (FC flatten):      {np.mean(accs_lr):.2f}% +/- {np.std(accs_lr):.2f}%")
print(f"GCN only:             {np.mean(accs_gnn):.2f}% +/- {np.std(accs_gnn):.2f}%")
print(f"LR + GNN embedding:   {np.mean(accs_gnn_lr):.2f}% +/- {np.std(accs_gnn_lr):.2f}%")
print(f"(论文 75.25%, 原始复现 75.63%)")
