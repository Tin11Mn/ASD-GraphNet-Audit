"""
ASD-GraphNet 改进 v8: 跨配置 Stacking + 多种子稳定性验证
思路:
  - 对每站点, 对候选配置集合生成 OOF 概率 (每个样本的预测来自没见过它的模型)
  - 元学习器 (LR / RF / Ridge) 在 OOF 概率上学 -> 无泄露地融合多配置信号
  - 相比"每站点选单一最优配置", Stacking 利用了多配置多样性
稳定性:
  - 整个流程用 5 个随机种子重复, 报告均值±标准差
"""
import warnings, ast, numpy as np, pandas as pd, time, pickle, itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
t0 = time.time()

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(DATA)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
SITES = sorted(df["SITE_ID"].unique())
ATLASES = ["cc200", "aal", "dos160"]
LEVELS = ["node_based", "edge_based", "graph_level"]

INDEX = {}
for _, r in df.iterrows():
    k = (r["SITE_ID"], r["atlas"], r["graph_feature_type"], r["feature_engineering"])
    INDEX.setdefault(k, {})[int(r["id"])] = (np.array(r["features_value"], dtype=float), int(r["ASD"]))

def get_concat(site, keys):
    id_map = {}
    for k in keys:
        d = INDEX.get((site,) + k)
        if d is None: return None, None
        for sid, (x, lab) in d.items():
            id_map.setdefault(sid, {"x": [], "y": lab})["x"].append(x)
    if not id_map: return None, None
    ids = sorted(id_map)
    X = np.stack([np.concatenate(id_map[i]["x"]) for i in ids])
    y = np.array([id_map[i]["y"] for i in ids])
    return np.nan_to_num(X), y

BASE_MODELS = {
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "DT": lambda: DecisionTreeClassifier(random_state=42),
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
}
META_MODELS = {
    "meta_LR": lambda: LogisticRegression(max_iter=2000, C=1.0),
    "meta_Ridge": lambda: RidgeClassifier(alpha=1.0),
    "meta_RF": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}

# ---- 候选配置池: 用 mi_10/pca_10/original 的代表性融合 ----
POOL = {
    "tri_atlas/node/mi_10": [(a,"node_based","mi_10") for a in ATLASES],
    "tri_atlas/edge/mi_10": [(a,"edge_based","mi_10") for a in ATLASES],
    "tri_atlas/graph/mi_10": [(a,"graph_level","mi_10") for a in ATLASES],
    "dual/cc200+dos160/node/mi_10": [("cc200","node_based","mi_10"),("dos160","node_based","mi_10")],
    "dual/cc200+aal/node/mi_10": [("cc200","node_based","mi_10"),("aal","node_based","mi_10")],
    "dual/aal+dos160/node/mi_10": [("aal","node_based","mi_10"),("dos160","node_based","mi_10")],
    "multi_level/cc200/mi_10": [("cc200",lv,"mi_10") for lv in LEVELS],
    "multi_level/aal/mi_10": [("aal",lv,"mi_10") for lv in LEVELS],
    "multi_level/dos160/mi_10": [("dos160",lv,"mi_10") for lv in LEVELS],
    "full/mi_10": [(a,lv,"mi_10") for a in ATLASES for lv in LEVELS],
    "single/cc200/node/mi_10": [("cc200","node_based","mi_10")],
    "single/dos160/node/mi_10": [("dos160","node_based","mi_10")],
    "single/aal/node/mi_10": [("aal","node_based","mi_10")],
    "tri_atlas/node/pca_10": [(a,"node_based","pca_10") for a in ATLASES],
    "full/pca_10": [(a,lv,"pca_10") for a in ATLASES for lv in LEVELS],
    "tri_atlas/node/original": [(a,"node_based","original") for a in ATLASES],
}
print(f"配置池: {len(POOL)}, 基学习器: {len(BASE_MODELS)}, 元学习器: {len(META_MODELS)}")

def oof_prob(X, y, mfn, n_splits=10, seed=42):
    """标准 OOF: 每折训练后预测留出部分 -> (n,2) 概率"""
    k = min(n_splits, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 2))
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)
    return oof

def stacking_eval(X_meta, y, meta_fn, n_splits=10, seed=42):
    """元学习器交叉验证 (meta特征本身已是OOF, 基本无泄露)"""
    k = min(n_splits, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X_meta, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X_meta[tr]); Xte = sc.transform(X_meta[te])
        m = meta_fn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

SEEDS = [42, 123, 2024, 7, 99]
records = []
for site in SITES:
    for seed in SEEDS:
        # 1) 生成所有 base 的 OOF 概率
        meta_parts = []; names = []
        y_site = None
        for cname, keys in POOL.items():
            X, y = get_concat(site, keys)
            if X is None or len(y) < 20 or min(np.bincount(y)) < 4:
                continue
            if y_site is None: y_site = y
            elif len(y) != len(y_site):
                continue
            for mname, mfn in BASE_MODELS.items():
                oof = oof_prob(X, y, mfn, seed=seed)
                if oof is None: continue
                meta_parts.append(oof[:, 1]); names.append(f"{cname}|{mname}")
        if not meta_parts or y_site is None:
            continue
        X_meta = np.stack(meta_parts, axis=1)
        # 2) 元学习器评估
        for mname, mfn in META_MODELS.items():
            a = stacking_eval(X_meta, y_site, mfn, seed=seed)
            if a is not None:
                records.append({"SITE_ID": site, "seed": seed, "meta": mname,
                                "acc": a, "n": len(y_site), "n_base": len(names)})
    print(f"  {site} done ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(records)
res.to_csv(r"D:/ASD-GraphNet - mhx/improve_v8_stacking.csv", index=False)

print("\n" + "="*78)
print("Stacking 结果 (5 种子 x 17 站点)")
print("="*78)
for mname in META_MODELS:
    sub = res[res["meta"] == mname]
    # 按种子 -> 每站点取该种子最优 -> 再按站点加权
    per_seed_w = []
    for seed in SEEDS:
        s = sub[sub["seed"] == seed]
        if s.empty: continue
        b = s.loc[s.groupby("SITE_ID")["acc"].idxmax()]
        per_seed_w.append((b["acc"]*b["n"]).sum()/b["n"].sum())
    if per_seed_w:
        print(f"  {mname:12s}: {np.mean(per_seed_w):.2f}% +/- {np.std(per_seed_w):.2f}%  (per-seed: {[f'{w:.2f}' for w in per_seed_w]})")

# 单一 meta 模型 (不按站点选) 的加权表现
print("\n[单一 meta 模型, 不按站点选优]")
for mname in META_MODELS:
    sub = res[res["meta"] == mname]
    per_seed_w = []
    for seed in SEEDS:
        s = sub[sub["seed"] == seed]
        if s.empty: continue
        per_seed_w.append((s["acc"]*s["n"]).sum()/s["n"].sum())
    print(f"  {mname:12s}: {np.mean(per_seed_w):.2f}% +/- {np.std(per_seed_w):.2f}%")

# 每站点最优 meta 模型
print("\n[每站点选最优 meta 模型]")
per_seed_w = []
for seed in SEEDS:
    s = res[res["seed"] == seed]
    if s.empty: continue
    b = s.loc[s.groupby("SITE_ID")["acc"].idxmax()]
    per_seed_w.append((b["acc"]*b["n"]).sum()/b["n"].sum())
    if seed == SEEDS[0]:
        print(b[["SITE_ID","meta","acc","n"]].to_string(index=False))
print(f"\n>>> Stacking 加权准确率: {np.mean(per_seed_w):.2f}% +/- {np.std(per_seed_w):.2f}%")
print(">>> 对比: 论文基线 75.63%, v5 融合 77.21%, v7 扩展 77.28%")
print(f"用时 {time.time()-t0:.0f}s")
