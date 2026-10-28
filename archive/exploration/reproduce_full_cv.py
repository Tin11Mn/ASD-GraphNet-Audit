"""
ASD-GraphNet 完整复现脚本 (faithful to experiments_pipeline_classifiers_revised.ipynb)
目标: 复现论文 75.25% 10-fold CV 加权平均准确率
"""
import ast, os, time, sys, json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

BASE = r"D:/ASD-GraphNet - mhx"
DATA_PATH = BASE + r"/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
OUT_PATH = BASE + r"/reproduction_cv_full.csv"
PROGRESS_PATH = BASE + r"/progress.json"
N_JOBS = 1  # 保持稳定

def save_progress(idx, total, results_count, last_print):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"idx": idx, "total": total, "results_count": results_count, "last_print": last_print}, f)

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"idx": 0, "total": 0, "results_count": 0, "last_print": ""}

def main():
    t0 = time.time()

    subjects_df = pd.read_csv(DATA_PATH)
    subjects_df["features_value"] = subjects_df["features_value"].apply(ast.literal_eval)
    print(f"Loaded data: {subjects_df.shape}")

    cases = [
        {"graph_feature_type": "node_based", "feature": "degree"},
        {"graph_feature_type": "edge_based", "feature": "average_degree"},
        {"graph_feature_type": "graph_level", "feature": "average_degree"},
    ]

    # 7 个分类器，与论文一致
    models = [
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
        ("DT", DecisionTreeClassifier(random_state=42)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("XGB", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)),
        ("LGBM", LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        ("SVM", SVC(probability=True, random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ]

    sites = sorted(subjects_df["SITE_ID"].unique())
    atlases = sorted(subjects_df["atlas"].unique())
    feature_engs = sorted(subjects_df["feature_engineering"].unique())

    # 枚举所有组合
    combos = []
    for site_id in sites:
        for atlas in atlases:
            for case in cases:
                for fe in feature_engs:
                    subset = subjects_df[
                        (subjects_df["SITE_ID"] == site_id)
                        & (subjects_df["atlas"] == atlas)
                        & (subjects_df["graph_feature_type"] == case["graph_feature_type"])
                        & (subjects_df["feature"] == case["feature"])
                        & (subjects_df["feature_engineering"] == fe)
                    ]
                    if subset.empty:
                        continue
                    X = np.stack(subset["features_value"].values)
                    y = subset["ASD"].values
                    if len(y) < 10:
                        continue
                    combos.append((site_id, atlas, case, fe, X, y))

    total = len(combos) * len(models)
    print(f"Total configs: {len(combos)}, Total model fits: {total}")

    # 恢复进度
    progress = load_progress()
    start_combo_idx = progress.get("idx", 0)
    results = []
    if os.path.exists(OUT_PATH):
        # 检查之前是否保存过部分结果
        old = pd.read_csv(OUT_PATH)
        results = old.to_dict("records")
        print(f"Loaded {len(results)} previous results")

    combo_idx = 0
    for site_id, atlas, case, fe, X, y in combos:
        if combo_idx < start_combo_idx:
            combo_idx += 1
            continue

        for model_name, model in models:
            try:
                cv_scores = cross_val_score(model, X, y, cv=10, scoring="accuracy", n_jobs=N_JOBS)
                y_pred = cross_val_predict(model, X, y, cv=10, n_jobs=N_JOBS)
            except Exception as e:
                print(f"  ERROR {site_id},{atlas},{case['graph_feature_type']},{fe},{model_name}: {e}")
                continue

            cm = confusion_matrix(y, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) != 0 else 0
            specificity = (tn / (tn + fp)) * 100 if (tn + fp) != 0 else 0
            precision = precision_score(y, y_pred, average="weighted", zero_division=0) * 100
            recall = recall_score(y, y_pred, average="weighted", zero_division=0) * 100
            f1 = f1_score(y, y_pred, average="weighted", zero_division=0) * 100

            results.append({
                "SITE_ID": site_id,
                "atlas": atlas,
                "graph_feature_type": case["graph_feature_type"],
                "feature": case["feature"],
                "feature_engineering": fe,
                "model": model_name,
                "cv_accuracy_mean": cv_scores.mean() * 100,
                "cv_accuracy_std": cv_scores.std() * 100,
                "precision": precision,
                "recall": recall,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": f1,
                "n_samples": len(y),
            })

        # 保存进度
        combo_idx += 1
        if combo_idx % 20 == 0 or combo_idx == len(combos):
            pd.DataFrame(results).to_csv(OUT_PATH, index=False)
            save_progress(combo_idx, len(combos), len(results), f"{combo_idx}/{len(combos)}")
            elapsed = time.time() - t0
            rate = combo_idx / elapsed if elapsed > 0 else 0
            remaining = (len(combos) - combo_idx) / rate if rate > 0 else 0
            print(f"[{combo_idx}/{len(combos)}] elapsed={elapsed:.0f}s remaining~{remaining:.0f}s results={len(results)}", flush=True)

    results_df = pd.DataFrame(results)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(results_df)} results -> {OUT_PATH}")

    # ---- 论文的加权平均计算 ----
    print("\n=== 加权平均准确率计算 (与论文一致) ===")

    best_config_per_site = results_df.loc[
        results_df.groupby("SITE_ID")["cv_accuracy_mean"].idxmax()
    ]
    print("\nBest config per site:")
    print(best_config_per_site[["SITE_ID", "atlas", "graph_feature_type", "feature_engineering", "model", "cv_accuracy_mean", "n_samples"]].to_string(index=False))

    # site_sizes = site_counts / (unique_graph_feature_type * unique_feature_engineering * unique_atlas)
    unique_graph_feature_type = subjects_df["graph_feature_type"].nunique()
    unique_feature_engineering = subjects_df["feature_engineering"].nunique()
    unique_atlas = subjects_df["atlas"].nunique()
    product_of_unique_configs = (
        unique_graph_feature_type * unique_feature_engineering * unique_atlas
    )
    site_id_counts = subjects_df["SITE_ID"].value_counts()
    site_sizes = (site_id_counts / product_of_unique_configs).astype(int)

    best_config_per_site = best_config_per_site.set_index("SITE_ID").join(site_sizes)
    total = site_sizes.sum()

    weighted_avg_accuracy = (
        best_config_per_site["cv_accuracy_mean"]
        * best_config_per_site.index.map(site_sizes)
    ).sum() / total
    weighted_precision = (
        best_config_per_site["precision"] * best_config_per_site.index.map(site_sizes)
    ).sum() / total
    weighted_recall = (
        best_config_per_site["recall"] * best_config_per_site.index.map(site_sizes)
    ).sum() / total
    weighted_sensitivity = (
        best_config_per_site["sensitivity"] * best_config_per_site.index.map(site_sizes)
    ).sum() / total
    weighted_specificity = (
        best_config_per_site["specificity"] * best_config_per_site.index.map(site_sizes)
    ).sum() / total
    weighted_f1 = (
        best_config_per_site["f1_score"] * best_config_per_site.index.map(site_sizes)
    ).sum() / total

    print("\n========================================")
    print("       REPRODUCTION RESULTS")
    print("========================================")
    print(f"Weighted Average Accuracy: {weighted_avg_accuracy:.2f}%   (paper: {75.25}%)")
    print(f"Weighted Precision:        {weighted_precision:.4f}    (paper: {75.3347})")
    print(f"Weighted Recall:           {weighted_recall:.4f}    (paper: {75.0725})")
    print(f"Weighted Sensitivity:      {weighted_sensitivity:.4f}    (paper: {71.2001})")
    print(f"Weighted Specificity:      {weighted_specificity:.4f}    (paper: {76.4010})")
    print(f"Weighted F1 Score:         {weighted_f1:.4f}    (paper: {74.6501})")

    acc_diff = abs(weighted_avg_accuracy - 75.25)
    print(f"\nAccuracy difference: {acc_diff:.2f}%")
    if acc_diff < 1.0:
        print("RESULT: FULLY REPRODUCED")
    elif acc_diff < 5.0:
        print("RESULT: PARTIALLY REPRODUCED")
    else:
        print("RESULT: NOT FULLY REPRODUCED")

    best_config_per_site.to_csv(BASE + r"/best_config_per_site.csv")
    print(f"\nSaved best_config_per_site.csv")
    print(f"Total runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()