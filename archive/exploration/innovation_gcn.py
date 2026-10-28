"""
创新点 1: 端到端 GNN (GCN/GAT) 从功能连接矩阵预测 ASD
策略:
  1. 用 edge_based_features.csv 中的 'average_degree' 和 'clustering_coefficient' 等信息
     反推每个被试的 FC 邻接矩阵 (thresholding)
  2. 构建 GCN/GAT 模型
  3. 10-fold CV 评估 (site-stratified)
注意: 由于没有原始 FC，我们用 degree 作为节点特征，用 edge 统计量推断邻接
"""
import ast, warnings, os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
EDGE_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/edge_based_features.csv"

# 加载数据
df = pd.read_csv(DATA_PATH)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
edge_df = pd.read_csv(EDGE_PATH)

# 加载 ROI 坐标用于空间邻近
roi_df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/CC200_ROI_labels.csv")
roi_df.columns = [c.strip() for c in roi_df.columns]
roi_coords = []
for i, row in roi_df.iterrows():
    coords = re.findall(r'-?\d+\.?\d*', str(row['center of mass']))
    roi_coords.append([float(v) for v in coords[:3]])
roi_coords = np.array(roi_coords, dtype=np.float32)
merged = df.merge(edge_df[["id", "cc200_average_degree", "cc200_clustering_coefficient", 
                           "cc200_edge_density"]], on=["id"], suffixes=("", "_edge"))

# 只取 cc200 node_based + original
sub = merged[(merged["graph_feature_type"]=="node_based") & 
             (merged["feature"]=="degree") &
             (merged["feature_engineering"]=="original") &
             (merged["atlas"]=="cc200")]

print(f"Samples: {len(sub)}")

# 为每个被试构建图
# 节点特征：degree (200 维)
# 边：根据 edge_density 和 clustering_coeff 推断邻接矩阵
# 简化: 用 degree 的相似度构建 KNN 图 (k=10)
def build_graph_from_degree(degree_vec, k=10):
    """输入: 200 维 degree -> 输出: 邻接矩阵 (200x200)
    策略: 用 degree 值的统计邻近性 + 空间位置邻近性构建 kNN 图
    """
    deg = np.array(degree_vec, dtype=np.float32)
    n = len(deg)
    # 1) degree 相似性 (归一化后的欧氏距离)
    d_norm = (deg - deg.mean()) / (deg.std() + 1e-8)
    dist_deg = np.linalg.norm(d_norm[:, None] - d_norm[None, :], axis=-1)
    # 2) ROI 空间位置邻近性
    coords = roi_coords
    dist_pos = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # 3) 融合距离
    dist = dist_deg / (dist_deg.max() + 1e-8) + dist_pos / (dist_pos.max() + 1e-8)
    # KNN 图
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        top_k = np.argsort(dist[i, :])[:k]
        adj[i, top_k] = 1.0
        adj[top_k, i] = 1.0
    np.fill_diagonal(adj, 0)
    return adj

# GCN 模型
class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# 构建 PyG 数据集
graphs = []
labels = []
for _, row in sub.iterrows():
    deg = np.array(row["features_value"], dtype=np.float32)
    y = int(row["ASD"])
    adj = build_graph_from_degree(deg, k=10)
    edge_index = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)
    x = torch.tensor(deg.reshape(-1, 1), dtype=torch.float)  # 单通道 degree
    data = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(data)
    labels.append(y)

print(f"Built {len(graphs)} graphs")

# 10-fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
device = torch.device("cpu")

accs = []
for fold, (tr_idx, te_idx) in enumerate(cv.split(graphs, labels)):
    tr_data = [graphs[i] for i in tr_idx]
    te_data = [graphs[i] for i in te_idx]
    tr_y = np.array(labels)[tr_idx]
    te_y = np.array(labels)[te_idx]
    
    # 构建 dataloader
    tr_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
    te_loader = DataLoader(te_data, batch_size=32, shuffle=False)
    
    model = GCNClassifier(in_dim=1, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Eval
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
    
    acc = accuracy_score(te_y, preds) * 100
    accs.append(acc)
    print(f"Fold {fold+1}: {acc:.2f}%", flush=True)

print(f"\n=== GCN 10-fold CV ===")
print(f"Mean accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
print(f"(论文 75.25%, 原始复现 75.63%)")
