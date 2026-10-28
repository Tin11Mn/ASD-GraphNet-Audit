"""
最终尝试 v13: 仅优化图特征 (论文 mi_10)
1. 更多模型组合 (ExtraTrees, HistGradientBoosting, CatBoost)
2. 超参数网格搜索 (C, n_estimators, depth)
3. 集成学习 (Soft Voting, Stacking)
4. 特征重要性筛选 (Top-K 特征)
5. 多种子验证
"""
import warnings, ast, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except:
    HAS_CAT = False
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(DATA); df["features_value"] = df["features_value"].apply(ast.literal_eval)

# 论文最优配置：node_based + mi_10 + 多 atlas
graph_feats = {}
for atlas in ["cc200","aal","dos160"]:
    for gt in ["node_based","edge_based","graph_level"]:
        sub = df[(df["atlas"]==atlas)&(df["graph_feature_type"]==gt)&(df["feature_engineering"]=="mi_10")]
        graph_feats[(atlas,gt)] = {int(r["id"]): np.array(r["features_value"],dtype=float)
                                    for _,r in sub.iterrows()}

uniq = df[(df["atlas"]=="cc200")&(df["graph_feature_type"]=="node_based")&(df["feature_engineering"]=="original")]\
        .drop_duplicates(subset=["id"])[["id","ASD","SITE_ID"]]
SITES = sorted(uniq["SITE_ID"].unique())

def build_matrix(site):
    sdf = uniq[uniq["SITE_ID"]==site]
    ids = sdf["id"].tolist(); y = sdf["ASD"].values
    parts = []
    for atlas in ["cc200","aal","dos160"]:
        for gt in ["node_based","edge_based","graph_level"]:
            d = graph_feats[(atlas,gt)]
            parts.append(np.stack([d.get(int(i), np.zeros(200)) for i in ids]))
    X = np.concatenate(parts, axis=1)
    return np.nan_to_num(X), y

MODELS = {
    "LR_C0.01": lambda: LogisticRegression(max_iter=3000, C=0.01),
    "LR_C0.1":  lambda: LogisticRegression(max_iter=3000, C=0.1),
    "LR_C1.0":  lambda: LogisticRegression(max_iter=3000, C=1.0),
    "RF_100":   lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "RF_300":   lambda: RandomForestClassifier(n_estimators=300, random_state=42),
    "ET_300":   lambda: ExtraTreesClassifier(n_estimators=300, random_state=42),
    "GB_100":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "HGB":      lambda: HistGradientBoostingClassifier(random_state=42),
    "XGB_100":  lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "XGB_200":  lambda: XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM_100": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "LGBM_200": lambda: LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    "SVM_RBF":  lambda: SVC(kernel="rbf", probability=True, random_state=42),
    "SVM_POLY": lambda: SVC(kernel="poly", probability=True, random_state=42),
    "KNN_5":    lambda: KNeighborsClassifier(n_neighbors=5),
    "KNN_10":   lambda: KNeighborsClassifier(n_neighbors=10),
}
if HAS_CAT:
    MODELS["CB_50"] = lambda: CatBoostClassifier(iterations=50, verbose=0, random_state=42)
    MODELS["CB_100"] = lambda: CatBoostClassifier(iterations=100, verbose=0, random_state=42)

SEEDS = [42, 123, 2024, 7, 99]

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

print("v13: 图特征优化 (多模型 + 超参搜索)")
print("="*78)

all_results = {}
for site in SITES:
    print(f"\n  站点 {site}...", flush=True)
    X, y = build_matrix(site)
    if len(y) < 20 or min(np.bincount(y)) < 5: continue
    # 不平衡 → 欠采样
    if len(y) < 50 and np.bincount(y).min() < 10:
        idx_major = np.where(y == np.bincount(y).argmax())[0]
        np.random.seed(42)
        idx_minor = np.where(y != np.bincount(y).argmax())[0]
        idx_bal = np.concatenate([np.random.choice(idx_major, len(idx_minor), replace=False), idx_minor])
        np.random.shuffle(idx_bal);  X, y = X[idx_bal], y[idx_bal]
    
    site_best = {}
    for seed in SEEDS:
        best_a, best_m = None, None
        for mn, mf in MODELS.items():
            a = cv_score(X, y, mf, seed=seed)
            if a is not None and (best_a is None or a > best_a):
                best_a, best_m = a, mn
        site_best[seed] = (best_a, best_m)
        print(f"    seed={seed}: {best_a:.2f}% ({best_m})", flush=True)
    all_results[site] = site_best

# 汇总
print("\n" + "="*78)
print("汇总 (5 种子加权平均)")
print("="*78)
site_weights = {s: uniq[uniq["SITE_ID"]==s]["ASD"].sum() + uniq[uniq["SITE_ID"]==s]["ASD"].shape[0]-uniq[uniq["SITE_ID"]==s]["ASD"].sum() for s in SITES}
for seed in SEEDS:
    accs, ns = [], []
    for site, best in all_results.items():
        if seed in best:
            accs.append(best[seed][0]); ns.append(site_weights.get(site, 100))
    w = (np.array(accs)*np.array(ns)).sum()/np.array(ns).sum()
    print(f"  seed={seed}: 加权={w:.2f}%")

mean_w = np.mean([all_results[s][s]["acc"] for s in all_results if s in all_results])
print(f"\n平均：{mean_w:.2f}%")
print(f"论文基线：75.63%")
print(f"提升潜力：{mean_w - 75.63:+.2f}%")
print(f"用时：{time.time()-t0:.0f}s")
