"""
ASD-GraphNet 改进版 V3
改进点:
1. RandomizedSearchCV 调参 RF (max_depth, min_samples_leaf, n_estimators)
2. class_weight='balanced' 代替 SMOTE
3. 保持 node_based/degree/original 主配置
"""
import ast, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform
warnings.filterwarnings("ignore")

data_path = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(data_path)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

# 只测试核心配置: node_based/degree/original
subset_df = df[
    (df["graph_feature_type"] == "node_based") &
    (df["feature"] == "degree")
]
print(f"Filtered to node_based/degree: {len(subset_df)} rows")

sites = sorted(subset_df["SITE_ID"].unique())
result_rows = []

for site in sites:
    print(f"\n=== Site: {site} ===", flush=True)
    for atlas in sorted(subset_df["atlas"].unique()):
        for fe in ["original", "mi_10", "pca_10"]:
            data = subset_df[
                (subset_df["SITE_ID"] == site) &
                (subset_df["atlas"] == atlas) &
                (subset_df["feature_engineering"] == fe)
            ]
            if data.empty or len(data) < 20:
                continue
            X = np.stack(data["features_value"].values)
            y = data["ASD"].values
            n, p = X.shape
            
            # 标准化
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            # RF with hyperparameter tuning (small grid for speed)
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_dist = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2", None],
            }
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                rf, param_dist, n_iter=15, cv=3,
                scoring="accuracy", n_jobs=1, random_state=42, refit=True
            )
            search.fit(X_s, y)
            best_rf = search.best_estimator_
            
            # 10-fold CV with best RF
            accs = []
            for tr_idx, te_idx in cv.split(X_s, y):
                X_tr, X_te = X_s[tr_idx], X_s[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                best_rf.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, best_rf.predict(X_te)))
            
            acc = np.mean(accs) * 100
            result_rows.append({
                "site": site, "atlas": atlas, "fe": fe,
                "model": "RF(tuned)",
                "cv_accuracy": acc,
                "best_params": str(search.best_params_),
                "n_samples": n
            })
            print(f"  {atlas}/{fe}: {acc:.2f}% (best: {search.best_params_})", flush=True)

results_df = pd.DataFrame(result_rows)
best_per_site = results_df.loc[results_df.groupby("site")["cv_accuracy"].idxmax()]
print("\n=== Best per site ===")
print(best_per_site[["site", "atlas", "fe", "cv_accuracy", "n_samples"]].to_string(index=False))

site_sizes = best_per_site["n_samples"]
weighted_acc = (best_per_site["cv_accuracy"] * site_sizes).sum() / site_sizes.sum()
print(f"\nWeighted Accuracy: {weighted_acc:.2f}% (paper: 75.25%, original repro: 75.63%)")
print(f"Improvement over original: {weighted_acc - 75.63:+.2f}%")
