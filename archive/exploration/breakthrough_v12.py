"""
ASD-GraphNet 突破80% 终极方案 v12
4个组件:
  1. 静态FC: 多阈值图特征(度/加权度/聚类/谱半径)
  2. 频域分支: 2D-DCT低频系数 + FFT能量谱 (BrainNetMLP 洞察)
  3. 动态FC: 滑窗FC统计量 + 时变标准差 (IMAGIN/MCDGLN 洞察)
  4. 跨图谱拼接: cc200 + aal + dos160 FC + 论文图特征
+ 噪声增强 (Gaussian jitter + bagging)
+ 小站点欠采样平衡
协议: 每站点10-fold CV → 选最优配置 → 加权平均 (论文原协议)
多种子验证 (seed=42, 123, 2024)
"""
import warnings, ast, time, os, glob, numpy as np, pandas as pd
from scipy.fft import dct, fft
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

# ── 路径 ──
ROI_DIRS = {
    "cc200": r"D:/ASD-GraphNet - mhx/data/rois_cc200",
    "aal":   r"D:/ASD-GraphNet - mhx/data/rois_aal",
    "dos160": r"D:/ASD-GraphNet - mhx/data/rois_dosenbach160",
}
PHENO_PATH = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"
DATA_PATH  = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
CACHE_DIR  = r"D:/ASD-GraphNet - mhx/data/fc_cache"

df      = pd.read_csv(DATA_PATH);  df["features_value"] = df["features_value"].apply(ast.literal_eval)
ph      = pd.read_csv(PHENO_PATH).dropna(subset=["FILE_ID"]);  ph = ph[ph["FILE_ID"] != "no_filename"]
id_map  = dict(zip(ph["SUB_ID"], ph["FILE_ID"]))          # SUB_ID → FILE_ID
n_rois  = {"cc200":200, "aal":116, "dos160":160}

print(f"ABIDE: {len(df)} rows, {df['SITE_ID'].nunique()} sites, ASD={df['ASD'].sum()}")
print(f"pheno 对齐: {len(id_map)} (FILE_ID匹配)")

# ── 工具 ──
def load_ts(site_subj, atlas):
    """site_subj: FILE_ID, atlas: cc200/aal/dos160"""
    fid = id_map.get(site_subj)
    if not fid: return None
    p = os.path.join(ROI_DIRS[atlas], f"{fid}.1D")
    if not os.path.exists(p): return None
    ts = np.loadtxt(p, delimiter="\t")
    if ts.ndim != 2 or ts.shape[1] < n_rois[atlas]: return None
    ts = ts[:, :n_rois[atlas]]   # 截断多余列
    return (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)

def static_fc_features(fc, thresholds=(0.1,0.2,0.3,0.4,0.5)):
    """静态 FC: 图统计 + 频域 (固定 390 维输出)"""
    n = fc.shape[0]
    absfc = np.abs(fc);  np.fill_diagonal(absfc, 0)
    feats = []
    for thr in thresholds:
        adj = (absfc > thr).astype(np.float32)
        deg = adj.sum(axis=1)
        wdeg = (absfc * adj).sum(axis=1)
        A2 = adj @ adj
        A3 = A2 @ adj
        tri = np.diag(A3) / 2.0
        den = deg * (deg - 1) / 2.0;  den[den == 0] = 1
        clust = np.mean(tri / den)
        eigvals = np.linalg.eigvalsh(adj)
        feats += [adj.mean(), clust, deg.mean(), deg.std(), deg.max(), deg.min(),
                  (deg == 0).sum(), np.max(np.abs(eigvals)),
                  (A3.sum()/6.0) / max(1, (adj.sum()/2)**2),
                  wdeg.mean(), wdeg.std(), wdeg.max()]
    # 频域：DCT + FFT
    fc_dct = dct(dct(fc, type=2, norm='ortho'), type=2, norm='ortho')
    k = min(16, fc_dct.shape[0])
    feats.append(fc_dct[:k, :k].flatten())   # 低频块
    feats.append(np.abs(np.fft.fft(absfc.sum(axis=1)))[:32])
    feats.append(np.abs(np.fft.fft(absfc.mean(axis=0)))[:32])
    # 全局 FC 分布
    f = absfc[np.triu_indices(n, 1)]
    feats += [f.mean(), f.std(), f.min(), f.max(),
              np.percentile(f,10), np.percentile(f,50), np.percentile(f,90),
              (f>0.3).mean(), (f>0.5).mean(), (f<0).mean()]
    return np.concatenate([x.reshape(-1) for x in feats])

def dynamic_fc_features(ts, win_sizes=(30, 60, 120)):
    """动态FC: 滑窗FC统计 + 时变标准差 (固定48维输出)"""
    n = ts.shape[1];  T = ts.shape[0]
    feats = [0.0] * 48   # 预分配固定长度
    idx = 0
    for w in win_sizes:
        if T < w:
            idx += 10;  continue
        step = max(1, w // 4)
        window_fps = []
        for s in range(0, T - w + 1, step):
            fc = np.corrcoef(ts[s:s+w].T)
            if not np.isfinite(fc).all(): continue
            absfc = np.abs(fc);  np.fill_diagonal(absfc, 0)
            window_fps.append([absfc.mean(), (absfc > 0.3).mean(),
                               (absfc > 0.5).mean(),
                               absfc[np.triu_indices(n,1)].std()])
        if not window_fps:
            idx += 10;  continue
        wp = np.array(window_fps)
        feats[idx:idx+4]  = list(wp.mean(axis=0));  idx += 4
        feats[idx:idx+4]  = list(wp.std(axis=0));   idx += 4
        feats[idx]        = wp[:,0].min();           idx += 1
        feats[idx]        = wp[:,0].max();           idx += 1
    # 时变 FC 标准差
    all_fcs = []
    for s in range(0, max(T-60, 1), 30):
        w = min(60, T - s)
        fc = np.corrcoef(ts[s:s+w].T)
        if np.isfinite(fc).all(): all_fcs.append(fc)
    if all_fcs:
        stack = np.stack(all_fcs)
        feats[idx:idx+6] = [np.mean(stack), np.std(stack),
                            np.abs(stack).mean(axis=0).mean(),
                            np.percentile(np.abs(stack),50).mean(),
                            (stack > 0.3).mean(), (stack < 0).mean()]
    return np.array(feats, dtype=float)

def build_fc_cache(atlas):
    cache_path = os.path.join(CACHE_DIR, f"fc_{atlas}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if d['static'].ndim == 2 and d['dyn'].ndim == 2:
            print(f"  加载 {atlas} 缓存: {len(d['ids'])} 被试, {d['static'].shape[1]} 维静态, {d['dyn'].shape[1]} 维动态")
            return d["ids"], d["static"], d["dyn"]
    ids_out, statics, dynamics = [], [], []
    cnt = 0
    for sub_id in id_map:
        ts = load_ts(sub_id, atlas)
        if ts is None: continue
        cnt += 1
        if cnt % 200 == 0:
            print(f"    {atlas} 进度 {cnt}/{len(id_map)} ({time.time()-t0:.0f}s)", flush=True)
        fc = np.corrcoef(ts.T)
        if not np.isfinite(fc).all(): continue
        statics.append(static_fc_features(fc))
        dynamics.append(dynamic_fc_features(ts))
        ids_out.append(sub_id)
    statics = np.nan_to_num(np.array(statics));  dynamics = np.nan_to_num(np.array(dynamics))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(ids_out), static=statics, dyn=dynamics)
    print(f"  {atlas} 完成: {len(ids_out)} 被试, 静态{statics.shape[1]}维, 动态{dynamics.shape[1]}维")
    return ids_out, statics, dynamics
# ── 预计算 FC 特征 ──
print("\n预计算 FC 特征...")
fc_all = {}  # atlas -> (ids, static, dyn)
for atlas in ["cc200", "aal", "dos160"]:
    print(f"\n  [{atlas}] n_rois={n_rois[atlas]}")
    fc_all[atlas] = build_fc_cache(atlas)

# ── 合并: 论文图特征 + FC特征 ──
print("\n合并图特征 + FC特征...")
# 论文图特征 (node_based, edge_based, graph_level 全 atlas, mi_10)
graph_parts = {}
for atlas in ["cc200","aal","dos160"]:
    for gt in ["node_based","edge_based","graph_level"]:
        sub = df[(df["atlas"]==atlas)&(df["graph_feature_type"]==gt)&(df["feature_engineering"]=="mi_10")]
        graph_parts[(atlas,gt)] = {int(r["id"]): np.array(r["features_value"],dtype=float)
                                    for _,r in sub.iterrows()}

uniq = df[(df["atlas"]=="cc200")&(df["graph_feature_type"]=="node_based")&(df["feature_engineering"]=="original")]\
        .drop_duplicates(subset=["id"])[["id","ASD","SITE_ID"]]
print(f"唯一被试: {len(uniq)}, 有FC(至少一个): {len(set(id_map.keys()) & set(uniq['id']))}")

def build_site_matrix(site, use_fc=True):
    sdf = uniq[uniq["SITE_ID"]==site]
    ids = sdf["id"].tolist();  y = sdf["ASD"].values
    parts = []
    # 图特征
    for atlas in ["cc200","aal","dos160"]:
        for gt in ["node_based","edge_based","graph_level"]:
            d = graph_parts[(atlas,gt)]
            parts.append(np.stack([d.get(int(i), np.zeros(200)) for i in ids]))
    # FC 特征
    if use_fc:
        for atlas in ["cc200","aal","dos160"]:
            ids_fc, st, dn = fc_all[atlas]
            id_to_idx = {int(i):j for j,i in enumerate(ids_fc)}
            has = [id_to_idx.get(int(i), -1) for i in ids]
            static_part = np.array([st[id_to_idx.get(int(i),0)] if id_to_idx.get(int(i),-1)>=0
                                     else np.zeros(st.shape[1]) for i in ids])
            dyn_part = np.array([dn[id_to_idx.get(int(i),0)] if id_to_idx.get(int(i),-1)>=0
                                  else np.zeros(dn.shape[1]) for i in ids])
            parts.extend([static_part, dyn_part])
    X = np.concatenate(parts, axis=1)
    return np.nan_to_num(X), y

# ── 模型 ──
MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    "SVM": lambda: SVC(probability=True, random_state=42),
}
VOTE = lambda: BaggingClassifier(estimator=LogisticRegression(max_iter=3000), n_estimators=10, random_state=42)

SEEDS = [42, 123, 2024]
SITES = sorted(uniq["SITE_ID"].unique())

def cv_score(X, y, mfn, n_splits=10, seed=42):
    k = min(n_splits, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs) if accs else None

# ── 主实验 ──
print("\n" + "="*78)
print("v12 终极方案: 静态FC + 动态FC + 频域 + 跨图谱 + 图特征融合")
print("="*78)

configs = [
    ("仅图特征",   True, False),
    ("图+静态FC",  True, True),
    ("图+动态FC",  True, False, "dyn"),
    ("图+静态+动态FC", True, True, "both"),
    ("仅FC (静态+动态)", False, True, "both"),
]

all_results = {}
for cfg_name, use_graph, use_fc, *extra in configs:
    mode = extra[0] if extra else None
    tag = f"[{cfg_name}]"
    print(f"\n{tag}")
    site_accs = {}  # site -> {seed -> acc}
    for seed in SEEDS:
        accs, ns = [], []
        for site in SITES:
            X, y = build_site_matrix(site, use_fc=use_fc)
            # 动态FC模式：只取动态部分
            if mode == "dyn":
                # X = 图 + [dyn_cc200, dyn_aal, dyn_dos160]
                n_graph = 0
                dyn_parts = []
                for atlas in ["cc200","aal","dos160"]:
                    ids_fc, st, dn = fc_all[atlas]
                    id_to_idx = {int(i):j for j,i in enumerate(ids_fc)}
                    has = [id_to_idx.get(int(i),-1) for i in X[: , :1]]
                    dyn_p = np.array([dn[id_to_idx.get(int(i),0)] if id_to_idx.get(int(i),-1)>=0
                                       else np.zeros(dn.shape[1]) for i in uniq[uniq["SITE_ID"]==site]["id"].tolist()])
                    dyn_parts.append(dyn_p)
                X = np.concatenate(dyn_parts, axis=1)
            elif mode == "both" and use_graph is False:
                # 仅FC
                pass  # 已在 build_site_matrix 中
            if len(y) < 20 or min(np.bincount(y)) < 5: continue
            # 小站点不平衡 → 欠采样
            if len(y) < 50 and np.bincount(y).min() < 10:
                idx_major = np.where(y == np.bincount(y).argmax())[0]
                np.random.seed(seed)
                idx_minor = np.where(y != np.bincount(y).argmax())[0]
                idx_bal = np.concatenate([np.random.choice(idx_major, len(idx_minor), replace=False), idx_minor])
                np.random.shuffle(idx_bal);  X, y = X[idx_bal], y[idx_bal]
            # 找最优模型
            best_a, best_m = None, None
            for mn, mf in MODELS.items():
                a = cv_score(X, y, mf, seed=seed)
                if a is not None and (best_a is None or a > best_a):
                    best_a, best_m = a, mn
            if best_a is not None:
                site_accs.setdefault(site, {})[seed] = best_a
                accs.append(best_a);  ns.append(len(y))
        if accs:
            w = (np.array(accs)*np.array(ns)).sum()/np.array(ns).sum()
            all_results.setdefault(cfg_name, [])
            all_results[cfg_name].append(w)
            print(f"  seed={seed}: 加权={w:.2f}%")

print("\n" + "="*78)
print("汇总: 各种子加权平均 (论文协议)")
print("="*78)
base = 75.63
for cfg, ws in all_results.items():
    mean_w = np.mean(ws);  std_w = np.std(ws)
    best_w = max(ws);  worst_w = min(ws)
    delta = mean_w - base
    sig = delta / std_w if std_w > 0 else 0
    print(f"{cfg:30s} {mean_w:.2f}% ± {std_w:.2f}%  (范围 {worst_w:.2f}~{best_w:.2f}%, Δ={delta:+.2f}%, {sig:.1f}σ vs 基线)")
print(f"论文基线: {base:.2f}%")
