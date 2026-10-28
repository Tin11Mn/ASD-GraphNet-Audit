"""
创新点 1: 端到端 GNN (GCN/GAT) 从真实 FC 预测 ASD
使用缓存的 FC 矩阵 (1016 被试, cc200)。
节点特征: FC 行的统计量 (mean/std/max) 或 degree centrality
10-fold CV, 按站点分层
"""
import warnings, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings("ignore")

CACHE = r"D:/ASD-GraphNet - mhx/data/cache/fc_all.npz"
DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"

fc_data = dict(np.load(CACHE))
df = pd.read_csv(DATA_PATH)
# 每个被试取一行 (cc200, node_based, degree, original)
sub = df[(df["graph_feature_type"]=="node_based") & (df["feature"]=="degree")
         & (df["feature_engineering"]=="original") & (df["atlas"]=="cc200")]
sub = sub.drop_duplicates("id")

# 构建图
THR = 0.3
graphs, labels, sites = [], [], []
for _, row in sub.iterrows():
    sid = str(row["id"])
    if sid not in fc_data:
        continue
    fc = fc_data[sid]
    adj = (np.abs(fc) > THR).astype(float)
    np.fill_diagonal(adj, 0)
    if adj.sum() < 10:
        continue
    # 节点特征: 每行 FC 的统计量
    feats = np.stack([fc.mean(1), fc.std(1), fc.max(1)], axis=1)  # (200, 3)
    x = torch.tensor(feats, dtype=torch.float)
    ei = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)
    graphs.append(Data(x=x, edge_index=ei, y=torch.tensor([int(row["ASD"])], dtype=torch.long)))
    labels.append(int(row["ASD"]))
    sites.append(row["SITE_ID"])

print(f"Graphs: {len(graphs)}, ASD={sum(labels)}, TDC={len(labels)-sum(labels)}")

class GNN(torch.nn.Module):
    def __init__(self, kind="gcn", in_dim=3, hid=64):
        super().__init__()
        Conv = GCNConv if kind == "gcn" else GATConv
        self.c1 = Conv(in_dim, hid)
        self.c2 = Conv(hid, hid)
        self.fc = torch.nn.Linear(hid, 2)
    def forward(self, x, ei, batch):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, batch)
        return self.fc(x)

def run_cv(kind="gcn", epochs=20, seed=42):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs, senss, specs = [], [], []
    for tr, te in cv.split(graphs, labels):
        tr_g = [graphs[i] for i in tr]; te_g = [graphs[i] for i in te]
        model = GNN(kind)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        crit = torch.nn.CrossEntropyLoss()
        model.train()
        for ep in range(epochs):
            for g in tr_g:
                opt.zero_grad()
                b = torch.zeros(g.x.size(0), dtype=torch.long)
                out = model(g.x, g.edge_index, b)
                loss = crit(out, g.y)
                loss.backward(); opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for g in te_g:
                b = torch.zeros(g.x.size(0), dtype=torch.long)
                out = model(g.x, g.edge_index, b)
                preds.append(out.argmax(1).item())
        yt = np.array([labels[i] for i in te])
        accs.append(accuracy_score(yt, preds)*100)
        cm = confusion_matrix(yt, preds)
        if cm.shape == (2,2):
            tn,fp,fn,tp = cm.ravel()
            senss.append(tp/(tp+fn)*100 if tp+fn else 0)
            specs.append(tn/(tn+fp)*100 if tn+fp else 0)
    return np.mean(accs), np.std(accs), np.mean(senss), np.mean(specs)

for kind in ["gcn"]:
    a, s, se, sp = run_cv(kind)
    print(f"{kind.upper()}: acc={a:.2f}% +/- {s:.2f}%, sens={se:.1f}%, spec={sp:.1f}%")

print("\n(论文 75.25%, 原始复现 75.63%)")
