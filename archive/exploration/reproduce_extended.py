"""
ASD-GraphNet 完整复现 - 多配置测试
目标：通过测试不同特征工程策略，验证能否达到论文 75.25%
"""
import ast
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

import os
os.environ["OMP_NUM_THREADS"] = "1"

BASE = r"D:/ASD-GraphNet - mhx"
DATA_PATH = BASE + r"/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
OUT_DIR = BASE + r"/reproduction_output"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    t0 = time.time()
    subjects_df = pd.read_csv(DATA_PATH)
    subjects_df["features_value"] = subjects_df["features_value"].apply(ast.literal_eval)
    
    print(f"Loaded data: {subjects_df.shape}")
    
    # 选择大样本站点和所有配置
    sites = ["NYU", "UM", "UCLA", "USM", "LEUVEN", "YALE", "PITT"]  # 7 个站点
    cases = [
        ("node_based", "degree"),
        ("edge_based", "average_degree"),
        ("graph_level", "average_degree"),
    ]
    feature_engs = ["original", "mi_10", "pca_10"]
    atlases = ["cc200", "aal", "dos160", "multi"]
    models = [
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("XGB", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(probability=True, random_state=42)),
    ]
    
    results = []
    count = 0
    total = len(sites) * len(atlases) * len(cases) * len(feature_engs) * len(models)
    
    for site in sites:
        for atlas in atlases:
            for gft, feat in cases:
                for fe in feature_engs:
                    subset = subjects_df[
                        (subjects_df["SITE_ID"] == site)
                        & (subjects_df["atlas"] == atlas)
                        & (subjects_df["graph_feature_type"] == gft)
                        & (subjects_df["feature"] == feat)
                        & (subjects_df["feature_engineering"] == fe)
                    ]
                    if subset.empty or len(subset) < 20:
                        continue
                    
                    X = np.stack(subset["features_value"].values)
                    y = subset["ASD"].values
                    
                    for name, model in models:
                        try:
                            scores = cross_val_score(model, X, y, cv=10, scoring="accuracy", n_jobs=1)
                            y_pred = cross_val_predict(model, X, y, cv=10, n_jobs=1)
                            acc = scores.mean() * 100
                            
                            cm = confusion_matrix(y, y_pred)
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                                spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
                            else:
                                sens = spec = 0
                            
                            results.append({
                                "site": site, "atlas": atlas,
                                "graph_feature_type": gft, "feature": feat,
                                "feature_engineering": fe, "model": name,
                                "accuracy": acc, "sensitivity": sens, "specificity": spec,
                                "n_samples": len(y)
                            })
                            count += 1
                            
                            if count % 50 == 0:
                                elapsed = time.time() - t0
                                remaining = elapsed / count * (total - count)
                                print(f"[{count}/{total}] elapsed={elapsed:.0f}s remaining~{remaining:.0f}s")
                        except Exception as e:
                            continue
    
    results_df = pd.DataFrame(results)
    out_path = OUT_DIR + "/full_reproduction_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} results -> {out_path}")
    
    # 分析结果
    print("\n=== Analysis ===")
    
    # Best per site
    best_per_site = results_df.loc[results_df.groupby("site")["accuracy"].idxmax()]
    print("\nBest config per site:")
    print(best_per_site[["site", "atlas", "graph_feature_type", "feature_engineering", "model", "accuracy"]].to_string(index=False))
    
    # Weighted average
    site_counts = results_df["n_samples"].groupby(results_df["site"]).first()
    weighted_acc = (best_per_site["accuracy"] * best_per_site["n_samples"]).sum() / best_per_site["n_samples"].sum()
    print(f"\nWeighted avg accuracy (7 sites): {weighted_acc:.2f}%")
    print(f"Target (paper): 75.25%")
    
    # By feature engineering
    print("\n=== By Feature Engineering ===")
    for fe in feature_engs:
        sub = results_df[results_df["feature_engineering"] == fe]
        best = sub.loc[sub.groupby("site")["accuracy"].idxmax()]
        w = (best["accuracy"] * best["n_samples"]).sum() / best["n_samples"].sum()
        print(f"{fe}: weighted acc = {w:.2f}%")
    
    # Top configs
    print("\n=== Top 10 Configs ===")
    print(results_df.nlargest(10, "accuracy")[["site", "atlas", "feature_engineering", "model", "accuracy"]].to_string(index=False))
    
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()
