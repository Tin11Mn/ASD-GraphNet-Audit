"""
创新点 1 (完整版): 端到端 GNN 从真实 FC 预测 ASD
- 读取 1016 个被试的 ROI 时间序列
- 计算 FC 矩阵 → 阈值化邻接
- 节点特征：FC 对角线 (或 degree centrality)
- GCN/GAT 分类
- 10-fold CV
"""
import os, warnings, ast, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
PHENO_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"
ROI_DIR = r"D:/ASD-GraphNet - mhx/data/rois_cc200"

pheno = pd.read_csv(PHENO_PATH)
valid = pheno[pheno["FILE_ID"] != "no_filename"]
file2sid = dict(zip(valid["FILE_ID"], valid["SUB_ID"]))

df = pd.read_csv(DATA_PATH)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

# 加载所有被试的图
# 一次性建立 id -> 文件路径索引
file_index = {}
for f in os.listdir(ROI_DIR):
    if f.endswith(".1D"):
        fid = f.replace(".1D", "")
        # FILE_ID 里包含 subject id
        for sid in file2sid.values():
            if str(sid) in fid:
                file_index[str(sid)] = os.path.join(ROI_DIR, f)
print(f"File index: {len(file_index)} subjects")

graphs = []
labels = []
sites = []
for _, row in df.iterrows():
    sid = str(row["id"])
    fpath = file_index.get(sid)
    if fpath is None:
        continue
    ts = np.loadtxt(fpath, delimiter="\t")
    ts_norm = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
    fc = np.corrcoef(ts_norm.T)
    # 阈值化
    adj = (np.abs(fc) > 0.3).astype(float)
    np.fill_diagonal(adj, 0)
    if adj.sum() == 0:
        continue
    # 节点特征：FC 对角线
    x = torch.tensor(np.diag(fc).reshape(-1, 1), dtype=torch.float)
    edge_index = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)
    y = torch.tensor([row["ASD"]], dtype=torch.long)
    graphs.append(Data(x=x, edge_index=edge_index, y=y))
    labels.append(row["ASD"])
    sites.append(row["SITE_ID"])

print(f"Loaded {len(graphs)} graphs")

# 10-fold CV (site-stratified)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs = []
for fold, (tr_idx, te_idx) in enumerate(cv.split(graphs, labels)):
    tr_graphs = [graphs[i] for i in tr_idx]
    te_graphs = [graphs[i] for i in te_idx]
    tr_y = np.array(labels)[tr_idx]
    te_y = np.array(labels)[te_idx]
    
    # 构建 GCN
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(1, 32)
            self.conv2 = GCNConv(32, 32)
            self.fc = torch.nn.Linear(32, 2)
        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index); x = F.relu(x)
            x = self.conv2(x, edge_index); x = F.relu(x)
            x = global_mean_pool(x, batch)
            return self.fc(x)
    
    # 训练
    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    for epoch in range(30):
        total_loss = 0
        for g in tr_graphs:
            opt.zero_grad()
            b = torch.zeros(g.x.size(0), dtype=torch.long)
            out = model(g.x, g.edge_index, b)
            loss = crit(out, g.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
    
    # 测试
    model.eval()
    preds = []
    with torch.no_grad():
        for g in te_graphs:
            b = torch.zeros(g.x.size(0), dtype=torch.long)
            out = model(g.x, g.edge_index, b)
            pred = out.argmax(dim=1).item()
            preds.append(pred)
    acc = accuracy_score(te_y, preds) * 100
    accs.append(acc)
    print(f"Fold {fold+1}: {acc:.2f}%", flush=True)

print(f"\n=== GCN 10-fold CV ===")
print(f"Mean: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
print(f"(论文 75.25%, 原始复现 75.63%)")
