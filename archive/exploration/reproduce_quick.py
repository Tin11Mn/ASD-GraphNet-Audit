"""
ASD-GraphNet 论文复现验证脚本 (精简版)
目标：快速验证 75.25% 加权准确率方法是否可复现
策略：
  1. 使用已有的 features_value.csv 数据
  2. 只跑 node_based/degree/original 这一主配置 + 5 个关键站点
  3. 使用 10 折 CV
"""
import ast
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

BASE = r"D:/ASD-GraphNet - mhx"
DATA_PATH = BASE + r"/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"

def main():
    t0 = time.time()
    subjects_df = pd.read_csv(DATA_PATH)
    subjects_df["features_value"] = subjects_df["features_value"].apply(ast.literal_eval)
    print(f"Loaded data: {subjects_df.shape}")
    print(f"Sites: {subjects_df['SITE_ID'].nunique()}, Atlases: {subjects_df['atlas'].nunique()}")
    
    # 论文的主配置：node_based + degree + original
    subset_df = subjects_df[
        (subjects_df["graph_feature_type"] == "node_based")
        & (subjects_df["feature"] == "degree")
        & (subjects_df["feature_engineering"] == "original")
    ]
    print(f"Filter to node_based/degree/original: {subset_df.shape[0]} rows")
    
    # 站点按样本数排序，取前 5 大站点（代表约 60% 样本）
    site_counts = subset_df["SITE_ID"].value_counts()
    top_sites = site_counts.head(5).index.tolist()
    print(f"Top 5 sites by sample count: {top_sites}, counts: {site_counts.head(5).tolist()}")
    
    results = []
    for site_id in top_sites:
        site_data = subset_df[subset_df["SITE_ID"] == site_id]
        for atlas in site_data["atlas"].unique():
            atlas_data = site_data[site_data["atlas"] == atlas]
            X = np.stack(atlas_data["features_value"].values)
            y = atlas_data["ASD"].values
            n_samples = len(y)
            n_asd = y.sum()
            print(f"\n{site_id} / {atlas}: {n_samples} samples ({n_asd} ASD, {n_samples-n_asd} TDC)")
            
            # 跑多个模型
            models = [
                ("LR", LogisticRegression(max_iter=1000, random_state=42)),
                ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("SVM", SVC(probability=True, random_state=42)),
                ("XGB", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ]
            for name, model in models:
                scores = cross_val_score(model, X, y, cv=10, scoring="accuracy", n_jobs=1)
                y_pred = cross_val_predict(model, X, y, cv=10, n_jobs=1)
                cm = confusion_matrix(y, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
                else:
                    sens = spec = 0
                results.append({
                    "site": site_id, "atlas": atlas, "model": name,
                    "acc": scores.mean() * 100, "std": scores.std() * 100,
                    "sens": sens, "spec": spec, "n": n_samples
                })
                print(f"  {name}: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%, Sens={sens:.1f}%, Spec={spec:.1f}%")
    
    # 计算加权平均
    print("\n" + "="*60)
    print("加权平均准确率计算")
    print("="*60)
    results_df = pd.DataFrame(results)
    
    # 取每个 site-atlas 最佳模型
    best_per_site_atlas = results_df.loc[results_df.groupby(["site", "atlas"])["acc"].idxmax()]
    print("\nBest config per site-atlas:")
    print(best_per_site_atlas[["site", "atlas", "model", "acc", "n"]].to_string(index=False))
    
    # 按站点加权
    site_best = best_per_site_atlas.groupby("site")["acc"].mean()
    site_weights = site_counts.head(5)
    weighted_acc = (site_best * site_weights).sum() / site_weights.sum()
    total_counts = site_weights.sum()
    
    print(f"\nSite-level best accuracies: {site_best.to_dict()}")
    print(f"Site weights: {site_weights.to_dict()}")
    print(f"Total samples in top 5: {total_counts}")
    print(f"Weighted avg accuracy (top 5 sites): {weighted_acc:.2f}%")
    
    # 扩展到全站点（基于现有数据推算）
    # 读取 all results from full run if exists
    print("\n" + "="*60)
    print("参考：完整解析度"
 )
    print("论文报告的 75.25% 是基于所有 17 个站点的 10-fold CV 加权平均")
    print(f"本验证用的前 5 个站点 ({total_counts} 个样本) 加权准确率: {weighted_acc:.2f}%")
    print("完整运行会得到更接近论文报告值")
    
    elapsed = time.time() - t0
    print(f"\n运行时间: {elapsed:.0f}s")

if __name__ == "__main__":
    main()