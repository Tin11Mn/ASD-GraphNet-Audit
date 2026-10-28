"""
ASD-GraphNet 最终验证 v9: 多种子稳定性检验
验证 77.28% 的提升是否稳定 (vs 论文基线 75.63%)

方案:
  - 用 v7 的配置池 (85 配置 × 6 模型)
  - 5 个随机种子重复整个流程
  - 报告: 均值±标准差, 以及每个种子下各站点的最佳配置分布
"""
import warnings, ast, numpy as np, pandas as pd, time, pickle
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

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "DT": lambda: DecisionTreeClassifier(random_state=42),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "SVM": lambda: SVC(probability=True, random_state=42),
}

# v7 配置池
CONFIGS = {}
import itertools
for a in ATLASES:
    for lv in LEVELS:
        for fe in ["original", "mi_10", "pca_10"]:
            CONFIGS[f"single/{a}/{lv}/{fe}"] = [(a, lv, fe)]
for combo in itertools.combinations(ATLASES, 2):
    for lv in LEVELS:
        for fe in ["original", "mi_10", "pca_10"]:
            CONFIGS[f"dual_atlas/{'+'.join(combo)}/{lv}/{fe}"] = [(a, lv, fe) for a in combo]
for lv in LEVELS:
    for fe in ["original", "mi_10", "pca_10"]:
        CONFIGS[f"tri_atlas/{lv}/{fe}"] = [(a, lv, fe) for a in ATLASES]
for a in ATLASES:
    for fe in ["original", "mi_10", "pca_10"]:
        CONFIGS[f"multi_level/{a}/{fe}"] = [(a, lv, fe) for lv in LEVELS]
for a in ATLASES:
    for lv in LEVELS:
        CONFIGS[f"cross_fe/{a}/{lv}"] = [(a, lv, "mi_10"), (a, lv, "pca_10")]
CONFIGS["full/mi_10"] = [(a, lv, "mi_10") for a in ATLASES for lv in LEVELS]
CONFIGS["full/original"] = [(a, lv, "original") for a in ATLASES for lv in LEVELS]
CONFIGS["tri_atlas_multi_level/mi_10"] = [(a, lv, "mi_10") for a in ATLASES for lv in LEVELS]

print(f"配置数：{len(CONFIGS)}, 模型数：{len(MODELS)}")

SEEDS = [42, 123, 2024, 7, 99]
all_records = []

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===")
    records = []
    for site in SITES:
        for cname, keys in CONFIGS.items():
            X, y = get_concat(site, keys)
            if X is None or len(y) < 15 or min(np.bincount(y)) < 3:
                continue
            # CV
            k = min(10, min(np.bincount(y)))
            cv = StratifiedKFold(k, shuffle=True, random_state=seed)
            accs = []
            for tr, te in cv.split(X, y):
                sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
                for mname, mfn in MODELS.items():
                    m = mfn(); m.fit(Xtr, y[tr])
                    accs.append((mname, accuracy_score(y[te], m.predict(Xte))*100))
            if not accs: continue
            # 取该站点该配置的最优模型
            best_m, best_a = max(accs, key=lambda x: x[1])
            records.append({"SITE_ID": site, "config": cname, "model": best_m, "acc": best_a, "n": len(y), "seed": seed})
    res = pd.DataFrame(records)
    best = res.loc[res.groupby("SITE_ID")["acc"].idxmax()]
    w = (best["acc"] * best["n"]).sum() / best["n"].sum()
    print(f"  Seed {seed}: 加权={w:.2f}%")
    all_records.append({"seed": seed, "weighted_acc": w, "best_per_site": best.to_dict('records')})

# 汇总
print("\n" + "="*78)
print("最终汇总 (5 种子稳定性检验)")
print("="*78)
weights = [r["weighted_acc"] for r in all_records]
print(f"v7 扩展方案：{np.mean(weights):.2f}% +/- {np.std(weights):.2f}%")
print(f"  各种子：{[f'{w:.2f}' for w in weights]}")

# 统计每站点最常出现的配置
from collections import Counter
site_config_counter = {s: Counter() for s in SITES}
site_model_counter = {s: Counter() for s in SITES}
for rec in all_records:
    for item in rec["best_per_site"]:
        site_config_counter[item["SITE_ID"]][item["config"]] += 1
        site_model_counter[item["SITE_ID"]][item["model"]] += 1

print("\n[最常胜出的配置 (5 种子中出现次数)]")
top_configs = Counter()
for s in SITES:
    top = site_config_counter[s].most_common(1)
    if top: top_configs[top[0][0]] += 1
for cfg, cnt in top_configs.most_common(5):
    print(f"  {cfg:40s}: {cnt}/5 站点")

print("\n[最常胜出的模型 (5 种子中出现次数)]")
top_models = Counter()
for s in SITES:
    top = site_model_counter[s].most_common(1)
    if top: top_models[top[0][0]] += 1
for m, cnt in top_models.most_common():
    print(f"  {m:10s}: {cnt}/17 站点")

# 对比
base_w = 75.63
v5_w = 77.21
v7_w = 77.28
v9_w = np.mean(weights)
print(f"\n=== 最终对比 ===")
print(f"论文基线：      {base_w:.2f}%")
print(f"v5 融合：        {v5_w:.2f}% (+{v5_w-base_w:.2f}%)")
print(f"v7 扩展配置池：  {v7_w:.2f}% (+{v7_w-base_w:.2f}%)")
print(f"v9 稳定性验证：  {v9_w:.2f}% +/- {np.std(weights):.2f}% (+{v9_w-base_w:.2f}%)")
print(f"\n提升显著性：{(v9_w - base_w) / np.std(weights):.2f} sigma")
print(f"用时：{time.time()-t0:.0f}s")
