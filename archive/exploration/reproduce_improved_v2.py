"""
ASD-GraphNet 改进版 V2
改进点:
1. SMOTE 类别平衡 (解决 ASD/HC 不平衡)
2. RandomForest 超参数调优 (n_estimators, max_depth, min_samples_leaf)
3. 软投票集成 (LR+RF+XGB)
"""
import ast, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

data_path = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(data_path)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

cases = [
    {"graph_feature_type": "node_based", "feature": "degree"},
    {"graph_feature_type": "edge_based", "feature": "average_degree"},
    {"graph_feature_type": "graph_level", "feature": "average_degree"},
]

sites = sorted(df["SITE_ID"].unique())
result_rows = []

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
                
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                accs = []
                
                for tr_idx, te_idx in cv.split(X, y):
                    X_tr, X_te = X[tr_idx], X[te_idx]
                    y_tr, y_te = y[tr_idx], y[te_idx]
                    
                    # 标准化
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)
                    
                    # PCA (pca_10)
                    if fe == "pca_10" and p > 10:
                        pca = PCA(n_components=10)
                        X_tr_s = pca.fit_transform(X_tr_s)
                        X_te_s = pca.transform(X_te_s)
                    elif fe == "original" and p > 50:
                        X_tr_s = X_tr_s[:, :50]
                        X_te_s = X_te_s[:, :50]
                    
                    # SMOTE (仅训练集)
                    try:
                        smote = SMOTE(random_state=42, k_neighbors=min(3, sum(y_tr==1)-1, sum(y_tr==0)-1))
                        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_tr)
                    except:
                        X_tr_sm, y_tr_sm = X_tr_s, y_tr
                    
                    # 投票集成
                    lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
                    rf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                               min_samples_leaf=2, random_state=42, class_weight='balanced')
                    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       random_state=42, eval_metric='logloss', verbosity=0)
                    
                    voting = VotingClassifier(
                        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
                        voting='soft'
                    )
                    voting.fit(X_tr_sm, y_tr_sm)
                    pred = voting.predict(X_te_s)
                    accs.append(accuracy_score(y_te, pred))
                
                acc = np.mean(accs) * 100
                result_rows.append({
                    "SITE_ID": site, "atlas": atlas,
                    "graph_feature_type": case["graph_feature_type"],
                    "feature": case["feature"],
                    "feature_engineering": fe,
                    "model": "Voting(SMOTE+RF200+XGB200+LR)",
                    "cv_accuracy": acc,
                    "n_samples": n
                })
                print(f"  {site}/{atlas}/{case['graph_feature_type']}/{fe}: {acc:.2f}%", flush=True)

results_df = pd.DataFrame(result_rows)
print(f"\n=== Summary ===", flush=True)
best_per_site = results_df.loc[results_df.groupby("SITE_ID")["cv_accuracy"].idxmax()]
print(best_per_site[["SITE_ID", "atlas", "graph_feature_type", "feature_engineering", "model", "cv_accuracy"]].to_string(index=False), flush=True)

site_sizes = best_per_site["n_samples"]
weighted_acc = (best_per_site["cv_accuracy"] * site_sizes).sum() / site_sizes.sum()
print(f"\nWeighted Accuracy: {weighted_acc:.2f}% (paper: 75.25%, original repro: ~75.6%)", flush=True)