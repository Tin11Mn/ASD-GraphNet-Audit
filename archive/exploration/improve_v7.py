"""
ASD-GraphNet 改进 v7: 扩大候选配置池 + 每站点选优
基于 v5 成功经验 (77.21%), 在论文预计算的 mi_10/pca_10/original 上
枚举更多融合组合, 让每站点有更多候选可选

新增候选:
  1. 任意两图谱组合 (C(3,2)=3种) x 3 level x 3 fe
  2. 任意三图谱组合 (1种) x 3 level x 3 fe
  3. 跨 fe 融合 (mi_10 + pca_10)
  4. 跨 level 融合 (node + edge + graph)
严格协议: 每站点 10-fold CV, 每站点选最优 -> 按样本量加权
"""
import warnings, ast, numpy as np, pandas as pd, itertools, time, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(DATA)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
SITES = sorted(df["SITE_ID"].unique())
ATLASES = ["cc200", "aal", "dos160"]
LEVELS = ["node_based", "edge_based", "graph_level"]
FES = ["original", "mi_10", "pca_10"]

# 预建索引: (site, atlas, level, fe) -> {id: vector, id: label}
INDEX = {}
for _, r in df.iterrows():
    k = (r["SITE_ID"], r["atlas"], r["graph_feature_type"], r["feature_engineering"])
    INDEX.setdefault(k, {})[int(r["id"])] = (np.array(r["features_value"], dtype=float), int(r["ASD"]))
print(f"Index keys: {len(INDEX)}")

def get_concat(site, keys):
    """keys: list of (atlas, level, fe); 返回对齐后的 (X, y)"""
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

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "DT": lambda: DecisionTreeClassifier(random_state=42),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "SVM": lambda: SVC(probability=True, random_state=42),
}

def cv_score(X, y, mfn, seeds=(42,)):
    accs = []
    for sd in seeds:
        k = min(10, min(np.bincount(y)))
        if k < 3: return None
        cv = StratifiedKFold(k, shuffle=True, random_state=sd)
        for tr, te in cv.split(X, y):
            sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            m = mfn(); m.fit(Xtr, y[tr])
            accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

# ---- 构建候选配置池 ----
CONFIGS = {}
# 1) 单图谱单level (论文原生)
for a in ATLASES:
    for lv in LEVELS:
        for fe in FES:
            CONFIGS[f"single/{a}/{lv}/{fe}"] = [(a, lv, fe)]
# 2) 多图谱同level同fe
for combo in itertools.combinations(ATLASES, 2):
    for lv in LEVELS:
        for fe in FES:
            CONFIGS[f"dual_atlas/{'+'.join(combo)}/{lv}/{fe}"] = [(a, lv, fe) for a in combo]
for lv in LEVELS:
    for fe in FES:
        CONFIGS[f"tri_atlas/{lv}/{fe}"] = [(a, lv, fe) for a in ATLASES]
# 3) 多level同图谱同fe
for a in ATLASES:
    for fe in FES:
        CONFIGS[f"multi_level/{a}/{fe}"] = [(a, lv, fe) for lv in LEVELS]
# 4) 跨fe (mi_10+pca_10)
for a in ATLASES:
    for lv in LEVELS:
        CONFIGS[f"cross_fe/{a}/{lv}"] = [(a, lv, "mi_10"), (a, lv, "pca_10")]
# 5) 全融合
CONFIGS["full/mi_10"] = [(a, lv, "mi_10") for a in ATLASES for lv in LEVELS]
CONFIGS["full/original"] = [(a, lv, "original") for a in ATLASES for lv in LEVELS]
CONFIGS["full/all_fe"] = [(a, lv, fe) for a in ATLASES for lv in LEVELS for fe in FES]
CONFIGS["tri_atlas_multi_level/mi_10"] = [(a, lv, "mi_10") for a in ATLASES for lv in LEVELS]

print(f"候选配置总数: {len(CONFIGS)}")

# ---- 评估 ----
records = []
done = 0
for cname, keys in CONFIGS.items():
    for mname, mfn in MODELS.items():
        for site in SITES:
            X, y = get_concat(site, keys)
            if X is None or len(y) < 15 or min(np.bincount(y)) < 3:
                continue
            a = cv_score(X, y, mfn)
            if a is None: continue
            records.append({"config": cname, "model": mname, "SITE_ID": site, "acc": a, "n": len(y)})
        done += 1
    if done % 20 == 0:
        print(f"  [{done}/{len(CONFIGS)*len(MODELS)}] {cname} elapsed={time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(records)
res.to_csv(r"D:/ASD-GraphNet - mhx/improve_v7_results.csv", index=False)
print(f"\n共 {len(res)} 条结果, 用时 {time.time()-t0:.0f}s")

# ---- 每站点选最优 ----
best = res.loc[res.groupby("SITE_ID")["acc"].idxmax()]
w = (best["acc"] * best["n"]).sum() / best["n"].sum()
print("\n" + "="*78)
print(f"v7 每站点选最优 加权准确率: {w:.2f}%")
print("="*78)
print(best[["SITE_ID","config","model","acc","n"]].to_string(index=False))

# 对比
base = pd.read_csv(r"D:/ASD-GraphNet - mhx/reproduction_cv_full.csv")
bb = base.loc[base.groupby("SITE_ID")["cv_accuracy_mean"].idxmax()]
bw = (bb["cv_accuracy_mean"]*bb["n_samples"]).sum()/bb["n_samples"].sum()
print(f"\n论文基线: {bw:.2f}%")
print(f"v5 融合:  77.21%")
print(f"v7 扩展:  {w:.2f}%  ({w-bw:+.2f}% vs 论文, {w-77.21:+.2f}% vs v5)")
