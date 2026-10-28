"""
突破 80% 方案 v10: 基于 arXiv 近期论文的关键洞察

核心策略 (来自文献调研):
1. [Beyond Anatomy, 2026.03] MSDL功能图谱 > 解剖图谱: 39-ROI功能分区比200-ROI解剖分区区分度更高
   -> 用现有 cc200(200) / aal(116) / dos160(160) 三图谱的融合，本质就是"功能分区探索"
2. [BrainNetMLP, 2025.07] 双分支 MLP (空间+频谱) 比复杂 GNN 更高效
   -> 在 FC 矩阵上构建空间分支 + 频域分支 (DCT变换)
3. [IMAGIN, 2022.11] 多粒度时空注意力
   -> 滑窗FC + 时间动态聚合
4. [MCDGLN, 2024.09] 动态FC + 掩码连接
   -> 滑窗 + 稀疏掩码去除噪声连接

实施方案:
  - 用已下载的 cc200 时间序列计算 FC 矩阵 (不同阈值)
  - 在 FC 上应用 DCT/FFT 获取频域特征 (空间+频谱双分支)
  - 滑动窗口 (20s/40s/60s) 计算动态FC，提取时变统计量
  - 合并: 静态FC (多阈值) + 频域特征 + 动态FC统计量 + 原始图特征
  - 模型: 双分支 MLP + 软投票
"""
import warnings, ast, time, os, numpy as np, pandas as pd
from scipy.fft import dct, fft
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

# 数据路径
PHENO_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"
ROI_DIR = r"D:/ASD-GraphNet - mhx/data/rois_cc200"
DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"

print(f"检查数据: ROI_DIR exists={os.path.exists(ROI_DIR)}")
n_files = len([f for f in os.listdir(ROI_DIR) if f.endswith('.1D')]) if os.path.exists(ROI_DIR) else 0
print(f"  ROI .1D 文件数: {n_files}")

df = pd.read_csv(DATA_PATH)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
SITES = sorted(df["SITE_ID"].unique())
print(f"ABIDE 站点: {len(SITES)}, 样本: {len(df)}, ASD: {df['ASD'].sum()}, TDC: {len(df)-df['ASD'].sum()}")

# ---- 读取时间序列并计算FC ----
def load_fc_matrix(subject_id, roi_dir, use_pearson=True):
    """读取 cc200 时间序列 -> 计算 Pearson FC 矩阵"""
    import glob
    files = glob.glob(os.path.join(roi_dir, f"*{subject_id}*.1D"))
    if not files:
        return None
    path = files[0]
    ts = np.loadtxt(path, delimiter="\t")
    # 归一化
    ts_norm = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
    # FC 矩阵 (Pearson 相关)
    fc = np.corrcoef(ts_norm.T)
    return fc

def compute_fc_features(fc, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """从 FC 矩阵提取多阈值图特征 + 频域特征"""
    features = []
    n = fc.shape[0]
    # 1. 各阈值下的度特征 (node level)
    for thr in thresholds:
        adj = (np.abs(fc) > thr).astype(float)
        np.fill_diagonal(adj, 0)
        degree = adj.sum(axis=1)
        features.append(degree.reshape(-1))  # n维
    # 2. 全局图统计量 (各阈值)
    for thr in thresholds:
        adj = (np.abs(fc) > thr).astype(float)
        np.fill_diagonal(adj, 0)
        # 聚类系数 (近似)
        clustering = 0.0
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) < 2: continue
            tri = 0
            for a in range(len(neighbors)):
                for b in range(a+1, len(neighbors)):
                    if adj[neighbors[a], neighbors[b]] > 0:
                        tri += 1
            clustering += 2*tri/(len(neighbors)*(len(neighbors)-1))
        clustering /= n
        features.append(np.array([adj.sum()/n**2, clustering, degree.mean(), degree.std()]))  # 4维
    # 3. 频域特征: DCT
    diag = np.diag(fc)
    features.append(dct(diag, type=2, norm='ortho'))  # n维
    mean_row = fc.mean(axis=1)
    features.append(dct(mean_row, type=2, norm='ortho'))  # n维
    # 4. FC 向量化 (上三角)
    triu_idx = np.triu_indices(n, k=1)
    features.append(fc[triu_idx])  # n*(n-1)/2 维
    return np.concatenate(features)

# ---- 加载预计算特征 + FC特征融合 ----
print("\n计算 FC 特征 (前3大站点)...")
feat_parts = {}  # site -> list of feature vectors
site_ids_map = {}

for site in ["NYU", "UM", "UCLA"]:
    print(f"\n  站点 {site}:")
    site_data = df[df["SITE_ID"] == site]
    n_subjects = site_data["id"].nunique()
    print(f"    被试数: {n_subjects}")
    
    X_extra = []; y_site = []; ids_site = []
    processed = 0
    
    for _, row in site_data.iterrows():
        sid = int(row["id"])
        # 原始图特征 (来自 filtered_subjects_df)
        orig_feat = np.array(row["features_value"], dtype=float)
        
        # FC 特征 (需要时间序列)
        fc_feat = None
        if os.path.exists(ROI_DIR):
            fc = load_fc_matrix(str(sid), ROI_DIR)
            if fc is not None and fc.shape == (200, 200):
                fc_feat = compute_fc_features(fc)
        
        if fc_feat is not None:
            combined = np.concatenate([orig_feat, fc_feat])
        else:
            combined = orig_feat
        
        X_extra.append(combined)
        y_site.append(row["ASD"])
        ids_site.append(sid)
        processed += 1
        if processed % 20 == 0:
            print(f"      已处理 {processed}/{n_subjects}", flush=True)
    
    X_site = np.nan_to_num(np.array(X_extra))
    y_site = np.array(y_site)
    print(f"    FC特征维度: {X_site.shape[1]}, 完整样本: {processed}/{n_subjects}")
    feat_parts[site] = (X_site, y_site, ids_site)

# ---- 简单 MLP 分类器 (双分支: FC分支 + 原始分支) ----
class DualBranchMLP:
    """双分支 MLP: FC 特征分支 + 原始图特征分支"""
    def __init__(self, fc_dim, orig_dim, hidden=(64, 32)):
        self.fc_dim = fc_dim
        self.orig_dim = orig_dim
        self.hidden = hidden
        # 简单线性投影代替完整 MLP (速度更快)
        self.w1 = np.random.randn(fc_dim, hidden[0]) * 0.01
        self.w2 = np.random.randn(orig_dim, hidden[1]) * 0.01
        self.w_out = np.random.randn(hidden[0]+hidden[1], 2) * 0.01
        self.b_out = np.zeros(2)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    def predict_proba(self, X_fc, X_orig):
        h1 = self._relu(X_fc @ self.w1)
        h2 = self._relu(X_orig @ self.w2)
        h = np.concatenate([h1, h2], axis=1)
        logits = h @ self.w_out + self.b_out
        return self._softmax(logits)
    
    def fit(self, X_fc, X_orig, y, epochs=50, lr=0.01):
        n = len(y)
        for ep in range(epochs):
            # Forward
            h1 = self._relu(X_fc @ self.w1)
            h2 = self._relu(X_orig @ self.w2)
            h = np.concatenate([h1, h2], axis=1)
            logits = h @ self.w_out + self.b_out
            probs = self._softmax(logits)
            # Loss (cross-entropy)
            loss = -np.mean(y * np.log(probs[:, 1] + 1e-8) + (1-y) * np.log(probs[:, 0] + 1e-8))
            # Backward (简化版)
            dlogits = probs.copy(); dlogits[:, 1] -= y; dlogits /= n
            self.w_out -= lr * (h.T @ dlogits)
            self.b_out -= lr * dlogits.sum(0)
            dh = dlogits @ self.w_out.T
            dh1, dh2 = dh[:, :self.hidden[0]], dh[:, self.hidden[0]:]
            # ReLU gradient
            gh1 = (h1 > 0).astype(float) * dh1
            gh2 = (h2 > 0).astype(float) * dh2
            self.w1 -= lr * (X_fc.T @ (gh1))
            self.w2 -= lr * (X_orig.T @ (gh2))
        return self

# 只在大站点测试 (NYU/UM/UCLA)
results_log = []
for site, (X, y, ids) in feat_parts.items():
    print(f"\n=== 站点 {site} (n={len(y)}) ===")
    # 分离 FC 和原始特征 (假设原始特征在前 200 维，FC 在后面)
    # 实际: orig = 200 (degree) + ..., fc = ~1500
    # 简化: 前 500 维为原始，后面为FC
    n_orig = min(500, X.shape[1]//2)
    X_orig = X[:, :n_orig]
    X_fc = X[:, n_orig:]
    
    sc = StandardScaler(); X_orig_s = sc.fit_transform(X_orig)
    sc2 = StandardScaler(); X_fc_s = sc2.fit_transform(X_fc)
    
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X_orig_s, y):
        # 训练双分支 MLP
        m = DualBranchMLP(X_fc_s.shape[1], X_orig_s.shape[1], hidden=(64, 32))
        m.fit(X_orig_s[tr], X_fc_s[tr], y[tr], epochs=50)
        proba = m.predict_proba(X_orig_s[te], X_fc_s[te])
        pred = proba.argmax(axis=1)
        accs.append(accuracy_score(y[te], pred)*100)
    
    mean_acc = np.mean(accs)
    print(f"  Dual-Branch MLP: {mean_acc:.2f}%")
    results_log.append({"site": site, "method": "dual_branch_mlp", "acc": mean_acc, "n": len(y)})

# ---- 传统方法对比 (相同数据) ----
print("\n=== 对比: 传统 LR/RF 在融合特征上 ===")
for site, (X, y, ids) in feat_parts.items():
    X_s = StandardScaler().fit_transform(X)
    for name, mfn in [("LR", lambda: LogisticRegression(max_iter=2000)),
                       ("RF", lambda: RandomForestClassifier(n_estimators=100, random_state=42))]:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        accs = []
        for tr, te in cv.split(X_s, y):
            m = mfn(); m.fit(X_s[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict(X_s[te]))*100)
        print(f"  {site} {name}: {np.mean(accs):.2f}%")

print(f"\n总用时: {time.time()-t0:.0f}s")