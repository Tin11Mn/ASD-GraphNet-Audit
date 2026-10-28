"""
创新点 1 (修正版): 端到端 GNN
修正:
  - class_weight 平衡 (ASD/TDC 1:1 但 fold 内仍可能偏)
  - sum pooling + concat(mean, max, sum)
  - DataLoader 批量训练 (快)
  - 节点特征加入 degree centrality
"""
import warnings, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)

CACHE = r"D:/ASD-GraphNet - mhx/data/cache/fc_all.npz"
DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"

fc_data = dict(np.load(CACHE))
df = pd.read_csv(DATA_PATH)
sub = df[(df["graph_feature_type"]=="node_based") & (df["feature"]=="degree")
         & (df["feature_engineering"]=="original") & (df["atlas"]=="cc200")].drop_duplicates("id")

THR = 0.3
graphs, labels = [], []
for _, row in sub.iterrows():
    sid = str(row["id"])
    if sid not in fc_data: continue
    fc = fc_data[sid]
    adj = (np.abs(fc) > THR).astype(float); np.fill_diagonal(adj, 0)
    if adj.sum() < 10: continue
    deg = adj.sum(1) / (adj.shape[0]-1)          # degree centrality
    feats = np.stack([fc.mean(1), fc.std(1), fc.max(1), deg], axis=1)
    x = torch.tensor(feats, dtype=torch.float)
    ei = torch.tensor(np.argwhere(adj>0).T, dtype=torch.long)
    graphs.append(Data(x=x, edge_index=ei, y=torch.tensor([int(row["ASD"])], dtype=torch.long)))
    labels.append(int(row["ASD"]))
print(f"Graphs: {len(graphs)}")

class GCNNet(torch.nn.Module):
    def __init__(self, in_dim=4, hid=32):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.fc = torch.nn.Linear(hid*3, 2)
    def forward(self, x, ei, batch):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.2, self.training)
        x = F.relu(self.c2(x, ei))
        pooled = torch.cat([global_add_pool(x,batch), global_mean_pool(x,batch), global_max_pool(x,batch)], 1)
        return self.fc(pooled)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs, senss, specs, f1s = [], [], [], []
for fold, (tr, te) in enumerate(cv.split(graphs, labels)):
    tr_g = [graphs[i] for i in tr]; te_g = [graphs[i] for i in te]
    tr_loader = DataLoader(tr_g, batch_size=32, shuffle=True)
    te_loader = DataLoader(te_g, batch_size=64, shuffle=False)
    yt = np.array([labels[i] for i in te])
    model = GCNNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    # class weight
    ytr = np.array([labels[i] for i in tr])
    w = torch.tensor([len(ytr)/(2*max((ytr==0).sum(),1)), len(ytr)/(2*max((ytr==1).sum(),1))], dtype=torch.float)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for ep in range(60):
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
    accs.append(accuracy_score(yt, preds)*100)
    cm = confusion_matrix(yt, preds)
    if cm.shape==(2,2):
        tn,fp,fn,tp = cm.ravel()
        senss.append(tp/(tp+fn)*100 if tp+fn else 0)
        specs.append(tn/(tn+fp)*100 if tn+fp else 0)
    print(f"Fold {fold+1}: acc={accs[-1]:.2f}% sens={senss[-1]:.0f}%", flush=True)

print(f"\n=== GNN (GCN) 10-fold CV ===")
print(f"Accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
print(f"Sensitivity: {np.mean(senss):.2f}%, Specificity: {np.mean(specs):.2f}%")
print(f"(论文 75.25%, 原始复现 75.63%)")
