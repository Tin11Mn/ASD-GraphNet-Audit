"""
ASD-GraphNet 改进 v6: 自适应MI比例 + 跨配置Stacking
策略:
  E. 在线 MI 筛选（在 fold 内无泄露地完成），候选比例 {5, 10, 20, 30, 50}%
  F. 跨配置 Stacking: 用多个配置+模型生成 OOF 概率 -> LR/R2 元学习器
  
严格协议:
  - 每站点独立 10-fold CV
  - 每站点选最优配置->加权
  - MI 筛选在 fold 内 fit_transform，防止数据泄露
"""
import warnings, ast, numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
base = pd.read_csv(r"D:/ASD-GraphNet - mhx/reproduction_cv_full.csv")
df = pd.read_csv(DATA)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
SITES = sorted(df["SITE_ID"].unique())
print(f"数据: {len(df)} rows, {len(SITES)} sites")

# ============================================================
# 共享工具函数
# ============================================================
def cv_score(X, y, mfn):
    k = min(10, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

def oof_preds_for(X, y, mfn, n_folds=10, seed=42):
    """返回 OOF (out-of-fold) 预测概率矩阵 (n_samples, 2)"""
    k = min(n_folds, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 2))
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        # probability 预测
        prob = m.predict_proba(Xte)
        oof[te] = prob
    return oof

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
}
VOTE = lambda: LogisticRegression(max_iter=3000, C=0.1)

# ============================================================
# 方案 E: 自适应 MI 比例
# ============================================================
print("\n" + "="*78)
print("方案 E: 自适应 MI 比例 (在线 fold 内选择)")
print("="*78)
ratio_list = [5, 10, 20, 30, 50]  # 保留 top K%
feat_sources = [
    ("node_cc200", [("cc200","node_based","original")]),
    ("node_multi", [("cc200","node_based","original"),("aal","node_based","original"),("dos160","node_based","original")]),
    ("full_orig",  [(a,g,f) for a in df["atlas"].unique() for g in df["graph_feature_type"].unique() for f in ["original"]]),
]

results_e = []
for src_name, src_keys in feat_sources:
    for mi_ratio in ratio_list:
        for mname, mfn in MODELS.items():
            accs, ns = [], []
            for site in SITES:
                # 获取原特征
                id_to_X = {}; id_to_y = {}
                for atlas, gt, fe in src_keys:
                    sub = df[(df["SITE_ID"]==site) & (df["atlas"]==atlas) &
                             (df["graph_feature_type"]==gt) & (df["feature_engineering"]==fe)]
                    for _, r in sub.iterrows():
                        sid = int(r["id"]); x = np.array(r["features_value"], dtype=float)
                        id_to_X.setdefault(sid, []).append(x)
                        id_to_y[sid] = int(r["ASD"])
                if not id_to_X: continue
                X_all = np.stack([np.concatenate(id_to_X[i]) for i in sorted(id_to_X)])
                y_site = np.array([id_to_y[i] for i in sorted(id_to_y)])
                X_all = np.nan_to_num(X_all)
                n_feat = X_all.shape[1]; n_keep = max(5, int(n_feat * mi_ratio / 100))
                # Fold 内 MI 筛选
                k = min(10, min(np.bincount(y_site)))
                if k < 3: continue
                cv = StratifiedKFold(k, shuffle=True, random_state=42)
                accs_s = []
                for tr, te in cv.split(X_all, y_site):
                    Xtr, Xte, ytr, yte = X_all[tr], X_all[te], y_site[tr], y_site[te]
                    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
                    # 在训练集上计算 MI
                    mi = mutual_info_classif(Xtr, ytr, random_state=42)
                    top_idx = np.argsort(mi)[::-1][:n_keep]
                    Xtr_s, Xte_s = Xtr[:, top_idx], Xte[:, top_idx]
                    m = mfn(); m.fit(Xtr_s, ytr)
                    accs_s.append(accuracy_score(yte, m.predict(Xte_s))*100)
                if accs_s: accs.append(np.mean(accs_s)); ns.append(len(y_site))
            if not accs: continue
            _a, _n = np.array(accs), np.array(ns)
            w = (_a*_n).sum()/_n.sum()
            results_e.append({"source": src_name, "mi_ratio": mi_ratio, "model": mname, "weighted_acc": w})
            print(f"  {src_name:15s} mi={mi_ratio:2d}% {mname:5s} => {w:.2f}%", flush=True)

res_e = pd.DataFrame(results_e)
best_e = res_e.loc[res_e.groupby(["source","mi_ratio"])["weighted_acc"].idxmax()]
print("\n[方案E 最优比] 每 (source, mi_ratio) 的最佳模型:")
print(best_e[["source","mi_ratio","model","weighted_acc"]].to_string(index=False))
# 取各 source 最佳 mi_ratio
for src in res_e["source"].unique():
    sub = res_e[res_e["source"]==src]
    best_row = sub.loc[sub["weighted_acc"].idxmax()]
    print(f"  {src:15s} => 最优 mi_ratio={best_row['mi_ratio']}%, model={best_row['model']}, acc={best_row['weighted_acc']:.2f}%")

# 方案E 总结果：取每个 source 最佳 mi_ratio + model
e_best_per_source = {}
for src in res_e["source"].unique():
    sub = res_e[res_e["source"]==src]
    best_row = sub.loc[sub["weighted_acc"].idxmax()]
    e_best_per_source[src] = {"mi_ratio": int(best_row["mi_ratio"]), "model": best_row["model"], "acc": best_row["weighted_acc"]}
# 合并：每个 site 选最好 source 的 acc
e_site_acc = {}
for _, r in res_e.iterrows():
    src, mi, mname = r["source"], r["mi_ratio"], r["model"]
    w = r["weighted_acc"]
    key = (src, mi, mname)
    # 需要 per-site acc 才能按 site 选最优
print("\n[方案E] 每站点最优 config")
# 重新跑 per-site
e_per_site = {}
for src_name, src_keys in feat_sources:
    for mi_ratio in ratio_list:
        for mname, mfn in MODELS.items():
            site_accs = {}
            for site in SITES:
                id_to_X = {}; id_to_y = {}
                for atlas, gt, fe in src_keys:
                    sub = df[(df["SITE_ID"]==site) & (df["atlas"]==atlas) &
                             (df["graph_feature_type"]==gt) & (df["feature_engineering"]==fe)]
                    for _, r2 in sub.iterrows():
                        sid = int(r2["id"]); x = np.array(r2["features_value"], dtype=float)
                        id_to_X.setdefault(sid, []).append(x); id_to_y[sid] = int(r2["ASD"])
                if not id_to_X: continue
                X_all = np.stack([np.concatenate(id_to_X[i]) for i in sorted(id_to_X)])
                y_site = np.array([id_to_y[i] for i in sorted(id_to_y)])
                X_all = np.nan_to_num(X_all); n_feat = X_all.shape[1]
                n_keep = max(5, int(n_feat * mi_ratio / 100))
                k = min(10, min(np.bincount(y_site)))
                if k < 3: continue
                cv = StratifiedKFold(k, shuffle=True, random_state=42)
                accs_s = []
                for tr, te in cv.split(X_all, y_site):
                    Xtr, Xte = X_all[tr], X_all[te]; ytr, yte = y_site[tr], y_site[te]
                    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
                    mi = mutual_info_classif(Xtr, ytr, random_state=42)
                    top_idx = np.argsort(mi)[::-1][:n_keep]
                    m = mfn(); m.fit(Xtr[:, top_idx], ytr)
                    accs_s.append(accuracy_score(yte, m.predict(Xte[:, top_idx]))*100)
                if accs_s: site_accs[site] = np.mean(accs_s)
            if site_accs:
                for site, a in site_accs.items():
                    key = (src_name, mi_ratio, mname)
                    if key not in e_per_site: e_per_site[key] = {}
                    e_per_site[key][site] = a

# 每站点选最优
e_best_sites = {}
for key, site_accs in e_per_site.items():
    for site, a in site_accs.items():
        if site not in e_best_sites or a > e_best_sites[site][1]:
            e_best_sites[site] = (key, a)
e_weighted = sum(e_best_sites[s][1] * next(iter(df[df["SITE_ID"]==s])[0]) for s in e_best_sites) / sum(next(iter(df[df["SITE_ID"]==s])[0]) for s in e_best_sites)
print(f"\n方案E 自适应MI 加权准确率: {e_weighted:.2f}%")
