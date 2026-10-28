"""
ASD-GraphNet 改进 V5: 多图谱特征融合
改进思路 (方法论层面，非调参):
1. 将同一被试在 4 个图谱 (aal/cc200/dos160) 下的 node degree 特征拼接成多视图特征
2. 在每个 CV fold 内部做 MI 特征筛选 (无泄露)
3. 用 LR + RF 两个模型评估
"""
import ast, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

data_path = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(data_path)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

# 只取 node_based/degree/original, 按 SUB_ID 组织
core = df[
    (df["graph_feature_type"] == "node_based") &
    (df["feature"] == "degree") &
    (df["feature_engineering"] == "original")
]
print(f"Core rows: {len(core)}")

sites = sorted(core["SITE_ID"].unique())
result_rows = []

def eval_model(X, y, model_name, k_mi=None):
    """10-fold CV, fold 内 MI 筛选 + 标准化"""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, senss, specs = [], [], []
    for tr, te in cv.split(X, y):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        # fold 内 MI 筛选
        if k_mi is not None and X_tr.shape[1] > k_mi:
            mi = mutual_info_classif(X_tr, y_tr, random_state=42)
            top = np.argsort(mi)[::-1][:k_mi]
            X_tr, X_te = X_tr[:, top], X_te[:, top]
        if model_name == "LR":
            m = LogisticRegression(max_iter=2000, random_state=42)
        elif model_name == "RF":
            m = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            m = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        cm = confusion_matrix(y_te, pred)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            senss.append(tp/(tp+fn) if tp+fn else 0)
            specs.append(tn/(tn+fp) if tn+fp else 0)
    return np.mean(accs)*100, np.mean(senss)*100, np.mean(specs)*100

for site in sites:
    site_data = core[core["SITE_ID"] == site]
    # 以被试为单位组织: 每个 id 收集各 atlas 特征
    id_col = "id"
    atlas_list = ["aal", "cc200", "dos160"]
    by_subj = {}
    subj_labels = {}
    for atlas in atlas_list:
        a = site_data[site_data["atlas"] == atlas]
        for _, row in a.iterrows():
            sid = row[id_col]
            by_subj.setdefault(sid, {})[atlas] = np.array(row["features_value"], dtype=float)
            subj_labels[sid] = row["ASD"]

    # 只保留 3 个图谱都有数据的被试
    fused_subs = [s for s in by_subj if all(k in by_subj[s] for k in atlas_list)]
    if len(fused_subs) < 20:
        print(f"{site}: only {len(fused_subs)} complete subjects, skip")
        continue

    X_fused = np.stack([
        np.concatenate([by_subj[s][a] for a in atlas_list]) for s in fused_subs
    ])
    y_fused = np.array([subj_labels[s] for s in fused_subs])

    # 基线: 单独 cc200
    X_cc200 = np.stack([by_subj[s]["cc200"] for s in fused_subs])

    for model in ["LR", "RF", "XGB"]:
        # 基线 (cc200 单图谱, MI-10)
        acc_b, sens_b, spec_b = eval_model(X_cc200, y_fused, model, k_mi=10)
        # 融合 (3 图谱拼接, MI-30, 因为特征×3)
        acc_f, sens_f, spec_f = eval_model(X_fused, y_fused, model, k_mi=30)
        result_rows.append({
            "site": site, "model": model, "n": len(fused_subs),
            "baseline_cc200_mi10": acc_b, "fused_mi30": acc_f,
            "diff": acc_f - acc_b,
            "sens_f": sens_f, "spec_f": spec_f,
        })
        print(f"{site} [{model}] n={len(fused_subs)}: baseline={acc_b:.2f}% -> fused={acc_f:.2f}% ({acc_f-acc_b:+.2f}%)", flush=True)

results = pd.DataFrame(result_rows)
print("\n=== 总体平均 (各站点简单平均) ===")
print(f"Baseline avg: {results['baseline_cc200_mi10'].mean():.2f}%")
print(f"Fused avg:    {results['fused_mi30'].mean():.2f}%")
print(f"Diff:         {results['diff'].mean():+.2f}%")

# 加权平均
w = results.groupby("site")["n"].first()
best_fused = results.loc[results.groupby("site")["fused_mi30"].idxmax()].set_index("site")
best_base = results.loc[results.groupby("site")["baseline_cc200_mi10"].idxmax()].set_index("site")
wf = (best_fused["fused_mi30"] * w).sum() / w.sum()
wb = (best_base["baseline_cc200_mi10"] * w).sum() / w.sum()
print(f"\n加权平均 (best model per site): baseline={wb:.2f}% -> fused={wf:.2f}% ({wf-wb:+.2f}%)")
print(f"(论文: 75.25%, 原始复现: 75.63%)")
results.to_csv(r"D:/ASD-GraphNet - mhx/improved_v5_multialias_fusion.csv", index=False)
