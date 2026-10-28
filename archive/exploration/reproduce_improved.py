"""
ASD-GraphNet 改进版复现 (修复版)
改进点: 
1. 标准化 (StandardScaler)
2. PCA 在 train split 后 fit-transform (避免数据泄露)
3. 更合理的特征数选择
"""
import ast, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

data_path = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(data_path)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

cases = [
    {"graph_feature_type": "node_based", "feature": "degree"},
    {"graph_feature_type": "edge_based", "feature": "average_degree"},
    {"graph_feature_type": "graph_level", "feature": "average_degree"},
]

models = {
    "LR": LogisticRegression(max_iter=1000, random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGB": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

sites = sorted(df["SITE_ID"].unique())
result_rows = []
t0 = time.time()

for site in sites:
    print(f"\n=== Site: {site} ===", flush=True)
    for atlas in sorted(df["atlas"].unique()):
        for case in cases:
            for fe in ["original", "mi_10", "pca_10"]:
                subset = df[
                    (df["SITE_ID"] == site) &
                    (df["atlas"] == atlas) &
                    (df["graph_feature_type"] == case["graph_feature_type"]) &
                    (df["feature"] == case["feature"]) &
                    (df["feature_engineering"] == fe)
                ]
                if subset.empty:
                    continue
                X = np.stack(subset["features_value"].values)
                y = subset["ASD"].values
                n, p = X.shape
                
                for name, model in models.items():
                    try:
                        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                        accs = []
                        
                        for tr_idx, te_idx in cv.split(X, y):
                            X_tr, X_te = X[tr_idx], X[te_idx]
                            y_tr, y_te = y[tr_idx], y[te_idx]
                            
                            # 标准化
                            scaler = StandardScaler()
                            X_tr_s = scaler.fit_transform(X_tr)
                            X_te_s = scaler.transform(X_te)
                            
                            # PCA (pca_10 使用 10 个主成分)
                            if fe == "pca_10" and p > 10:
                                pca = PCA(n_components=10)
                                X_tr_s = pca.fit_transform(X_tr_s)
                                X_te_s = pca.transform(X_te_s)
                            # 原始特征不降维
                            elif fe == "original" and p > 50:
                                # 原始特征过多时，只取前 50 个
                                X_tr_s = X_tr_s[:, :50]
                                X_te_s = X_te_s[:, :50]
                            
                            model.fit(X_tr_s, y_tr)
                            pred = model.predict(X_te_s)
                            accs.append(accuracy_score(y_te, pred))
                        
                        acc = np.mean(accs) * 100
                        result_rows.append({
                            "SITE_ID": site, "atlas": atlas,
                            "graph_feature_type": case["graph_feature_type"],
                            "feature": case["feature"],
                            "feature_engineering": fe,
                            "model": name,
                            "cv_accuracy": acc,
                            "n_samples": n
                        })
                        print(f"  {site}/{atlas}/{case['graph_feature_type']}/{fe}/{name}: {acc:.2f}%", flush=True)
                        
                    except Exception as e:
                        pass  # 跳过失败的配置

results_df = pd.DataFrame(result_rows)
print(f"\n=== Summary ===", flush=True)
print(f"Total results: {len(results_df)}", flush=True)

# 每个站点的最佳配置
best_per_site = results_df.loc[results_df.groupby("SITE_ID")["cv_accuracy"].idxmax()]
print("\nBest per site:")
print(best_per_site[["SITE_ID", "atlas", "graph_feature_type", "feature_engineering", "model", "cv_accuracy"]].to_string(index=False), flush=True)

# 加权平均
site_sizes = best_per_site["n_samples"]
weighted_acc = (best_per_site["cv_accuracy"] * site_sizes).sum() / site_sizes.sum()
print(f"\nWeighted Accuracy: {weighted_acc:.2f}% (paper: 75.25%)", flush=True)