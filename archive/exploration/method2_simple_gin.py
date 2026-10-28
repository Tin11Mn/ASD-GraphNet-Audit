"""
方法2简化版: 自实现功能图谱聚类 + 轻量 GNN (无外部库依赖)
策略：
1. 用现有的 CC200 时间序列 (1016 subjects)
2. 自实现 KMeans 聚类功能社区 (K=39)
3. 训练 GIN (Graph Isomorphism Network) 轻量版 (无 PyG) → 使用纯 torch 实现
4. 数据增强: 边随机删除/添加、特征高斯噪声
5. 训练和测试
"""
import time, numpy as np, os, warnings, random, pandas as pd
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42); torch.manual_seed(42); random.seed(42)
device = torch.device("cpu")
t0 = time.time()

# 文件路径
ROI_DIR = r"D:\ASD-GraphNet - mhx\data\rois_cc200"
PHENO = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"

# 加载 pheno
df = pd.read_csv(PHENO, low_memory=False)
df = df[df['FILE_ID']!='no_filename'][['SUB_ID','FILE_ID','SITE_ID','DX_GROUP']]
id2file = dict(zip(df['SUB_ID'], df['FILE_ID']))

# 加载所有时间序列
print("Loading time series...")
all_ts = []; ids = []
for sid, fid in id2file.items():
    p = os.path.join(ROI_DIR, f"{fid}.1D")
    if not os.path.exists(p): continue
    ts = np.loadtxt(p, delimiter='\t')
    if ts.ndim != 2 or ts.shape[1] != 200: continue
    ts = ts[:, :200]
    # 统一时间点长度为 200 (不足则补零)
    if ts.shape[0] < 200:
        ts = np.pad(ts, ((0, 200 - ts.shape[0]), (0, 0)), mode='constant')
    else:
        ts = ts[:200, :]
    ts = (ts - ts.mean()) / (ts.std() + 1e-8)
    all_ts.append(ts)
    ids.append(sid)

X_ts = np.array(all_ts)  # (1016, 200, T)
y_labels = np.array([df[df['SUB_ID']==sid]['DX_GROUP'].iloc[0] for sid in ids])
print(f"Time series shape: {X_ts.shape}, labels distribution: {np.bincount(y_labels)}")

# 自实现 KMeans 聚类功能社区 (使用 sklearn)
K = 39
print(f"K-Means clustering (K={K})...")
# 每个 ROI 取所有 subjects 的均值时间序列 -> 200 dim 特征向量
roi_mean_ts = X_ts.mean(axis=0)  # (200, 200)
roi_fc = np.corrcoef(roi_mean_ts.T)  # (200, 200)
# 每个 ROI 的连接强度 profile
roi_profile = roi_fc.mean(axis=1)  # (200, )
# KMeans
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(roi_profile.reshape(-1, 1))
print(f"Functional community distribution: {np.bincount(kmeans_labels)}")

# 为每个功能社区聚合 ROI: 取属于该社区的所有 ROI 的均值时间序列
print("Building per-subject community time series...")
X_community_ts = np.zeros((len(X_ts), K, X_ts.shape[2]))
for k in range(K):
    mask = kmeans_labels == k
    if mask.sum() > 0:
        X_community_ts[:, k, :] = X_ts[:, mask, :].mean(axis=1)
    else:
        X_community_ts[:, k, :] = 0.0

print(f"Community time series shape: {X_community_ts.shape}")

def build_community_fc(ts):
    """给定单个被试的社区时间序列 (K, T), 计算 K 个社区之间的 FC"""
    fc = np.corrcoef(ts.T)  # (K, K)
    return fc

print("Building community FC features...")
X_feat = []
for i in range(len(X_ts)):
    fc = build_community_fc(X_community_ts[i])
    absfc = np.abs(fc); np.fill_diagonal(absfc, 0)
    tri = absfc[np.triu_indices(K, 1)]
    deg = absfc.sum(axis=1)
    glob = [absfc.mean(), absfc.std(), deg.mean(), deg.std(), deg.max(), (deg==0).sum()]
    feat = np.concatenate([tri, deg, glob])
    X_feat.append(feat)

X_feat = np.nan_to_num(np.array(X_feat))
y_list = y_labels
print(f"Features shape: {X_feat.shape}, labels distribution: {np.bincount(y_list)}")

# ── GNN 实现 ──
class SimpleGraphNet(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        # 两个图卷积层 (DAG 图)
        self.conv1_weight = nn.Parameter(torch.randn(in_dim, hidden))
        self.conv1_bias = nn.Parameter(torch.zeros(hidden))
        self.conv2_weight = nn.Parameter(torch.randn(hidden, hidden))
        self.conv2_bias = nn.Parameter(torch.zeros(hidden))
        # 组合层
        self.combine_weight = nn.Parameter(torch.randn(2 * hidden, hidden))
        self.combine_bias = nn.Parameter(torch.zeros(hidden))
        # 分类器
        self.cls_weight = nn.Parameter(torch.randn(hidden, 2))
        self.cls_bias = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(0.2)
    
    def graph_conv(self, x, edge_index, weight, bias):
        # 稀疏邻接矩阵（仅当边存在时提取邻居特征）
        n = x.shape[0]
        row, col = edge_index
        h = x @ weight  # (n, hidden)
        # 图卷积：对于每个节点，聚合邻居特征
        h_agg = torch.zeros_like(h)
        # 使用 scatter_add
        for i in range(n):
            mask = row == i
            if mask.any():
                h_agg[i] = h[col[mask]].mean(dim=0)
            else:
                h_agg[i] = h[i]
        return F.relu(h_agg + bias)
    
    def forward(self, x, edge_index, batch):
        # 第一个卷积
        h1 = self.graph_conv(x, edge_index, self.conv1_weight, self.conv1_bias)
        h1 = self.dropout(h1)
        # 第二个卷积
        h2 = self.graph_conv(h1, edge_index, self.conv2_weight, self.conv2_bias)
        # 池化 (mean pooling)
        unique_batches = torch.unique(batch)
        pooled = []
        for b in unique_batches:
            mask = batch == b
            pooled.append(h2[mask].mean(dim=0))
        h_pool = torch.stack(pooled)
        # 分类
        logits = h_pool @ self.cls_weight + self.cls_bias
        return logits

# 构建图数据集
print("Building graph dataset...")
graphs = []
for i in range(len(X_feat)):
    fc = build_community_fc(X_community_ts[i])
    adj = np.abs(fc) > 0.3
    np.fill_diagonal(adj, False)
    row, col = np.where(adj)
    edge_index = torch.tensor(np.column_stack([row, col]), dtype=torch.long).t()
    x = torch.tensor(X_feat[i], dtype=torch.float).unsqueeze(0)
    graphs.append({
        'x': x, 'edge_index': edge_index, 'y': torch.tensor([y_list[i]], dtype=torch.long)
    })

print(f"Created {len(graphs)} graphs")
indices = np.random.choice(len(graphs), min(300, len(graphs)), replace=False)
subset = [graphs[i] for i in indices]
print(f"使用 {len(subset)} 个样本进行训练 (随机抽取)")

# 训练
model = SimpleGraphNet(X_feat.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\n训练 GIN...")
for epoch in range(50):
    model.train()
    total_loss = 0
    for g in subset:
        x, edge_index, y = g['x'].to(device), g['edge_index'].to(device), g['y'].to(device)
        # 边随机删除 (5% 概率)
        drop_mask = torch.rand(edge_index.shape[1]) > 0.05
        edge_index = edge_index[:, drop_mask]
        # 特征噪声
        x = x + torch.randn_like(x) * 0.01
        optimizer.zero_grad()
        out = model(x, edge_index, torch.zeros(len(x), dtype=torch.long, device=device))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(subset):.4f}", flush=True)

# 评估 (随机抽取验证集)
model.eval()
indices = np.random.choice(len(graphs), min(200, len(graphs)), replace=False)
val_accs = []
for i in indices:
    g = graphs[i]
    x, edge_index, y = g['x'].to(device), g['edge_index'].to(device), g['y'].to(device)
    with torch.no_grad():
        out = model(x, edge_index, torch.zeros(len(x), dtype=torch.long, device=device))
        pred = out.argmax(dim=1).item()
        val_accs.append(pred == y.item())

val_acc = np.mean(val_accs) * 100
print(f"\n验证准确率: {val_acc:.2f}%")
print(f"用时: {time.time()-t0:.0f}s")
