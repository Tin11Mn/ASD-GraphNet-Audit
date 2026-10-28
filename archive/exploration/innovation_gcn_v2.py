"""
创新点 1: 端到端 GNN (GCN/GAT) 从真实 FC 直接预测 ASD
使用下载的 1016 个被试的 ROI 时间序列 (cc200)。
思路:
  1. 读取每个被试的 200 个 ROI 平均时间序列
  2. 计算 Pearson 相关相关矩阵 → FC 矩阵
  3. 根据阈值阻断 → 邻接矩阵
  4. 用 PyG GCN/GAT 进行 ASD 二分类
  5. 10-fold CV 按站点 stratified
"""
import ast, warnings, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings("ignore")

DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
PHENO_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"
DOWNLOAD_DIR = r"D:/ASD-GraphNet - mhx/data/rois_cc200"

# 加载 meta info
pheno = pd.read_csv(PHENO_PATH)
valid = pheno[pheno["FILE_ID"] != "no_filename"]
file2subject = dict(zip(valid["FILE_ID"], valid["SUB_ID"]))

# 加载特征
df = pd.read_csv(DATA_PATH)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

# 用度 degree 作为节点特征
def compute_fc_and_adj(subject_id, roi_dir=DOWNLOAD_DIR):
    """读取时间序列 → 计算 FC 矩阵 → 阈值化得到邻接矩阵"""
    # 找到该被试的 ROI 文件
    fid = None
    for f in os.listdir(roi_dir):
        if subject_id in f:
            fid = f.replace(".1D", "")
            break
    if fid is None:
        return None, None
    
    # 读取时间序列: 200×T
    ts = np.loadtxt(os.path.join(roi_dir, fid + ".1D"), delimiter="\t")
    # 归一化 (z-score per ROI)
    ts_norm = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
    # FC 矩阵: 200×200 Pearson 相关
    fc = np.corrcoef(ts_norm.T)
    # 阈值化 (保留 top 20% 或固定阈值 0.3)
    thr = 0.3
    adj = (np.abs(fc) > thr).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return fc, adj

# PyG GCN
class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# 主循环
device = torch.device("cpu")
sites = ["NYU", "UM", "UCLA", "USM", "LEUVEN"]  # 5 大站点用于快速演示
subject_ids = []
for site in sites:
    sdf = df[df["SITE_ID"] == site]
    for _, row in sdf.iterrows():
        subject_ids.append(row["id"])

accs_fold = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 快速演示改 10 再算
for fold, (tr_idx, te_idx) in enumerate(cv.split(subject_ids, [1]*len(subject_ids))):
    tr_ids = [subject_ids[i] for i in tr_idx]
    te_ids = [subject_ids[i] for i in te_idx]
    
    graphs_tr = []
    y_tr = []
    for sid in tr_ids:
        fc, adj = compute_fc_and_adj(sid)
        if fc is None:
            continue
        n = adj.shape[0]
        # 特征: 度 centrality (或用 FC 对角线 / 度)
        x = torch.tensor(np.diag(fc).reshape(-1, 1), dtype=torch.float)
        # 边索引: 阈值化矩阵的非零项
        adj_norm = adj > 0.3
        edge_index = torch.tensor(np.argwhere(adj_norm > 0).T, dtype=torch.long)
        # degree 加权: x = degree (可能是每个 ROI 的度 centrality 也行，但这里用度 centrality 或原始度)
        # 简化: 用 diag FC 作为特征，GCN 学习聚合
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([1], dtype=torch.long))  # placeholder
        graphs_tr.append(data)
        y_tr.append(1)  # placeholder，后面改
        
    # 实际要把所有被试图进来构建 graph list
    # 这里先做个简化版: 对每个被试单独跑 1 轮，算准确率
    break  # 先跑一个被试测流程

print("见控制台输出")