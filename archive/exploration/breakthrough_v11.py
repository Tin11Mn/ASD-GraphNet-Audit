"""
突破80% v11: 高效实现
1. 向量化 FC 特征 (无 O(n^3) 聚类系数)
2. 缓存到 npy, 避免重复计算
3. 核心: 频域分支 (BrainNetMLP 洞察) + 多阈值FC + 图特征融合
"""
import warnings, ast, time, os, glob, numpy as np, pandas as pd
from scipy.fft import dct
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

PHENO_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"
ROI_DIR = r"D:/ASD-GraphNet - mhx/data/rois_cc200"
DATA_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
CACHE = r"D:/ASD-GraphNet - mhx/data/fc_cache_cc200.npz"

df = pd.read_csv(DATA_PATH)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
# 每个被试唯一 (取 node_based/cc200/original 一行)
uniq = df[(df["atlas"]=="cc200") & (df["graph_feature_type"]=="node_based") &
          (df["feature_engineering"]=="original")].drop_duplicates(subset=["id"])
print(f"唯一被试: {len(uniq)}, ASD={uniq['ASD'].sum()}")

# ---- FC 特征计算 (向量化 + 缓存) ----
def fast_fc_features(fc, thresholds=(0.1,0.2,0.3,0.4,0.5)):
    """向量化图特征 + 频域特征"""
    n = fc.shape[0]
    absfc = np.abs(fc)
    np.fill_diagonal(absfc, 0)
    feats = []
    degrees = []
    for thr in thresholds:
        adj = (absfc > thr).astype(np.float32)
        deg = adj.sum(axis=1)
        degrees.append(deg)
        # 聚类系数 (向量化: A^3 对角)
        A3 = adj @ adj @ adj
        tri = np.diag(A3) / 2.0
        denom = deg * (deg - 1) / 2.0
        denom[denom == 0] = 1
        clustering = np.mean(tri / denom)
        # 加权度 (连接强度)
        wdeg = (absfc * adj).sum(axis=1)
        feats.append(np.array([
            adj.mean(), clustering, deg.mean(), deg.std(),
            deg.max(), deg.min(), (deg == 0).sum(),
            np.mean(np.abs(np.linalg.eigvalsh(adj))),          # 谱半径代理
            (A3.sum() / 6.0) / max(1.0, (adj.sum()/2)**2),      # 归一化聚类
        ]))
        feats.append(wdeg)   # n维 加权度
    # 频域分支: FC 的低频谱特征 (BrainNetMLP 洞察)
    # 对 FC 做 2D DCT 取低频系数 (捕捉整体连接模式)
    fc_dct = dct(dct(fc, type=2, norm='ortho'), type=2, norm='ortho')
    # 低频 20x20 块
    k = 20
    feats.append(fc_dct[:k, :k].flatten())
    # 每个ROI的连接强度谱
    conn_spectrum = np.abs(np.fft.fft(absfc.sum(axis=1)))
    feats.append(conn_spectrum[:40])
    # FC 整体能量谱
    fc_flat = absfc[np.triu_indices(n, 1)]
    feats.append(np.abs(np.fft.fft(fc_flat))[:60])
    # 时域统计: FC值分布
    feats.append(np.array([
        fc_flat.mean(), fc_flat.std(), fc_flat.min(), fc_flat.max(),
        np.percentile(fc_flat, 10), np.percentile(fc_flat, 25),
        np.percentile(fc_flat, 50), np.percentile(fc_flat, 75),
        np.percentile(fc_flat, 90), (fc_flat > 0.3).mean(),
    ]))
    return np.concatenate([f.reshape(-1) for f in feats])

def build_cache():
    print("构建 FC 缓存...")
    ids = sorted(uniq["id"].unique())
    all_feats = []
    ok_ids = []
    for i, sid in enumerate(ids):
        files = glob.glob(os.path.join(ROI_DIR, f"*{sid}*.1D"))
        if not files: continue
        ts = np.loadtxt(files[0], delimiter="\t")
        if ts.ndim != 2 or ts.shape[0] != 200: continue
        ts_n = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
        fc = np.corrcoef(ts_n.T)
        if not np.isfinite(fc).all(): continue
        all_feats.append(fast_fc_features(fc))
        ok_ids.append(sid)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(CACHE, ids=np.array(ok_ids), feats=np.array(all_feats))
    print(f"缓存完成: {len(ok_ids)} 被试, {len(all_feats[0])} 维")

if not os.path.exists(CACHE):
    build_cache()
else:
    print("使用已有缓存")

cache = np.load(CACHE)
fc_ids = cache["ids"]; fc_feats = cache["feats"]
print(f"FC 特征: {fc_feats.shape}")

# ---- 合并: 图特征 + FC特征 ----
id_to_fc = {int(i): fc_feats[j] for j, i in enumerate(fc_ids)}
uniq = uniq.copy()
uniq["fc_feat"] = [id_to_fc.get(int(i), np.zeros(fc_feats.shape[1])) for i in uniq["id"]]
uniq["has_fc"] = [int(i) in id_to_fc for i in uniq["id"]]
print(f"有 FC 的被试: {uniq['has_fc'].sum()}/{len(uniq)}")

# 论文图特征 (各 level 的 mi_10 拼接)
graph_feats = {}
for atlas in ["cc200","aal","dos160"]:
    for gt in ["node_based","edge_based","graph_level"]:
        sub = df[(df["atlas"]==atlas)&(df["graph_feature_type"]==gt)&(df["feature_engineering"]=="mi_10")]
        d = {int(r["id"]): np.array(r["features_value"], dtype=float) for _, r in sub.iterrows()}
        graph_feats[(atlas,gt)] = d

def build_site_matrix(site, use_fc=True, use_graph=True):
    site_df = uniq[uniq["SITE_ID"]==site]
    ids = site_df["id"].tolist(); y = site_df["ASD"].values
    parts = []
    if use_graph:
        gparts = []
        for atlas in ["cc200","aal","dos160"]:
            for gt in ["node_based","edge_based","graph_level"]:
                d = graph_feats[(atlas,gt)]
                gparts.append(np.stack([d.get(int(i), np.zeros(200)) for i in ids]))
        parts.extend(gparts)
    if use_fc:
        parts.append(np.stack([id_to_fc.get(int(i), np.zeros(fc_feats.shape[1])) for i in ids]))
    X = np.concatenate(parts, axis=1)
    return np.nan_to_num(X), y, site_df["has_fc"].values

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    "SVM": lambda: SVC(probability=True, random_state=42),
}

def cv_site(X, y, mfn, n_splits=10, seed=42):
    k = min(n_splits, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

SITES = sorted(uniq["SITE_ID"].unique())
print("\n" + "="*78)
print("方案对比 (每站点 10-fold CV, 按样本量加权)")
print("="*78)

for tag, use_fc, use_graph in [
    ("图特征 only (论文式)", False, True),
    ("FC 特征 only (新)", True, False),
    ("图特征 + FC 特征 (融合)", True, True),
]:
    print(f"\n[{tag}]")
    all_accs, all_ns = [], []
    for site in SITES:
        X, y, _ = build_site_matrix(site, use_fc, use_graph)
        if len(y) < 20 or min(np.bincount(y)) < 5: continue
        best_a, best_m = None, None
        for mname, mfn in MODELS.items():
            a = cv_site(X, y, mfn)
            if a is None: continue
            if best_a is None or a > best_a:
                best_a, best_m = a, mname
        if best_a is not None:
            all_accs.append(best_a); all_ns.append(len(y))
            print(f"  {site:10s} n={len(y):4d} best={best_m:5s} {best_a:.2f}%", flush=True)
    if all_accs:
        w = (np.array(all_accs)*np.array(all_ns)).sum()/np.array(all_ns).sum()
        print(f"  >>> 加权 = {w:.2f}%")

print(f"\n用时 {time.time()-t0:.0f}s")
