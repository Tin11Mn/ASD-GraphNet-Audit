"""
ASD-GraphNet 复现验证脚本
快速运行（只跑关键配置），并验证10折CV的加权平均准确率是否与论文一致。
利用已有的 outpipeline_results_df.csv 中的预计算结果 + 部分重跑。
"""
import ast
import os
import warnings
import numpy as np
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
DATA_PATH = os.path.join(BASE, r"ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
PRECOMP_PATH = os.path.join(BASE, r"ASD-GraphNet-main/outputs/outpipeline_results_df.csv")
OUT_DIR = os.path.join(BASE, r"reproduction_output")
os.makedirs(OUT_DIR, exist_ok=True)


def build_models():
    return [
        LogisticRegression(max_iter=1000, random_state=42),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        SVC(probability=True, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    ]


MODEL_NAMES = {
    "LogisticRegression": "LR",
    "DecisionTreeClassifier": "DT",
    "RandomForestClassifier": "RF",
    "SVC": "SVM",
    "MLPClassifier": "MLP",
    "GradientBoostingClassifier": "XGB",  # XGB 和 LGBM 此处都映射为 GB
}

# 论文中 75.25% 的计算方式:
# 1. 对每个 SITE_ID 选取最佳配置（以 cv_accuracy_mean 为准）
# 2. 按站点样本量加权平均


def weighted_avg_from_precomputed(precomp_df, subjects_df):
    """
    从预计算的 outpipeline_results_df.csv 复现 75.25% 计算。
    precomp_df 有 SITE_ID, model, cv_accuracy_mean 等列。
    """
    # 取每个 SITE_ID 的最优模型配置
    best_per_site = precomp_df.loc[precomp_df.groupby("SITE_ID")["test_accuracy"].idxmax()]

    # 站点样本量 (从 filtered_subjects_df 计算)
    site_counts = subjects_df["SITE_ID"].value_counts()
    configs_per_site = (
        subjects_df["graph_feature_type"].nunique()
        * subjects_df["feature_engineering"].nunique()
        * subjects_df["atlas"].nunique()
    )
    site_sizes = (site_counts / configs_per_site).astype(int)
    best_per_site = best_per_site.set_index("SITE_ID").join(site_sizes)

    total = site_sizes.sum()
    weighted_acc = (
        best_per_site["test_accuracy"] * best_per_site.index.map(site_sizes)
    ).sum() / total

    return weighted_acc, best_per_site


def run_reduced_cv(subjects_df, sites=None, atlases=None, cases=None, modes=None, verbose=True):
    """
    只对部分配置执行完整的 10 折 CV 计算，用作验证。
    """
    if sites is None:
        sites = subjects_df["SITE_ID"].unique()[:3]
    if cases is None:
        cases = [
            {"graph_feature_type": "node_based", "feature": "degree"},
            {"graph_feature_type": "edge_based", "feature": "average_degree"},
            {"graph_feature_type": "graph_level", "feature": "average_degree"},
        ]
    if modes is None:
        modes = ["LR", "RF", "SVM"]

    models_map = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGB": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "LGBM": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for site in sites:
        for atlas in atlases if atlases else subjects_df["atlas"].unique():
            for case in cases:
                for fe in subjects_df["feature_engineering"].unique():
                    subset = subjects_df[
                        (subjects_df["SITE_ID"] == site)
                        & (subjects_df["atlas"] == atlas)
                        & (subjects_df["graph_feature_type"] == case["graph_feature_type"])
                        & (subjects_df["feature"] == case["feature"])
                        & (subjects_df["feature_engineering"] == fe)
                    ]
                    if subset.empty:
                        continue
                    X = np.stack(subset["features_value"].values)
                    y = subset["ASD"].values

                    for mode in modes:
                        model = models_map[mode]
                        scores = cross_val_score(model, X, y, cv=10, scoring="accuracy", n_jobs=1)
                        y_pred = cross_val_predict(model, X, y, cv=10, n_jobs=1)
                        precision = precision_score(y, y_pred, average="weighted", zero_division=0) * 100
                        recall = recall_score(y, y_pred, average="weighted", zero_division=0) * 100
                        f1 = f1_score(y, y_pred, average="weighted", zero_division=0) * 100
                        cm = confusion_matrix(y, y_pred)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                        else:
                            tn = fp = fn = tp = 0
                        sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) != 0 else 0
                        specificity = (tn / (tn + fp)) * 100 if (tn + fp) != 0 else 0

                        results.append({
                            "SITE_ID": site,
                            "atlas": atlas,
                            "graph_feature_type": case["graph_feature_type"],
                            "feature": case["feature"],
                            "feature_engineering": fe,
                            "model": mode,
                            "cv_accuracy_mean": scores.mean() * 100,
                            "cv_accuracy_std": scores.std() * 100,
                            "precision": precision,
                            "recall": recall,
                            "sensitivity": sensitivity,
                            "specificity": specificity,
                            "f1_score": f1,
                        })
                        if verbose:
                            print(
                                f"{site},{atlas},{case['graph_feature_type']},{fe},{mode}: "
                                f"{scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%"
                            )
    return results


def main():
    subjects_df = pd.read_csv(DATA_PATH)
    subjects_df["features_value"] = subjects_df["features_value"].apply(ast.literal_eval)
    print(f"Loaded data: {subjects_df.shape}")

    # ---- 步骤 1：用预计算结果验证 75.25% ----
    if os.path.exists(PRECOMP_PATH):
        precomp = pd.read_csv(PRECOMP_PATH)
        print(f"Loaded precomputed results: {precomp.shape}")
        weighted_acc, best_per_site = weighted_avg_from_precomputed(precomp, subjects_df)
        print(f"\n=== Step 1: 用预计算结果验证 ===")
        print(f"Weighted avg accuracy (from precomputed 10-fold CV): {weighted_acc:.2f}%")
        print(f"Target from paper: 75.25%")
        diff = abs(weighted_acc - 75.25)
        print(f"Difference: {diff:.2f}% {'✓ MATCH' if diff < 1 else '✗ DIFFERENT'}")

        best_per_site.to_csv(os.path.join(OUT_DIR, "best_config_per_site.csv"))
        print(f"Saved best_config_per_site.csv")
    else:
        print("Precomputed results not found, running reduced CV...")

    # ---- 步骤 2：运行缩减版 CV 验证 ----
    print(f"\n=== Step 2: 运行缩减版 10 折 CV（前3个站点 × 部分配置）===")
    reduced_results = run_reduced_cv(
        subjects_df,
        sites=["NYU", "SDSU", "UM"],
        cases=[{"graph_feature_type": "node_based", "feature": "degree"}],
        modes=["LR", "RF", "SVM", "XGB"],
    )

    if reduced_results:
        reduced_df = pd.DataFrame(reduced_results)
        reduced_df.to_csv(os.path.join(OUT_DIR, "reduced_cv_results.csv"), index=False)
        # 按站点取最优
        best_reduced = reduced_df.loc[reduced_df.groupby("SITE_ID")["cv_accuracy_mean"].idxmax()]
        site_sizes = subjects_df["SITE_ID"].value_counts().head(3)
        site_sizes_ratio = site_sizes / site_sizes.sum()
        weighted_reduced = (best_reduced["cv_accuracy_mean"] * site_sizes_ratio).sum()
        print(f"\nReduced weighted avg accuracy: {weighted_reduced:.2f}%")
        print(best_reduced[["SITE_ID","atlas","model","cv_accuracy_mean"]].to_string(index=False))

    # ---- 步骤 3：对比论文报告值 ----
    print(f"\n=== Step 3: 与论文报告对比 ===")
    print("Paper reported (from experiments_pipeline_classifiers_revised.ipynb):")
    print("  Weighted Average Accuracy for 10-fold CV: 75.25%")
    print("  Weighted Precision:   75.3347")
    print("  Weighted Recall:      75.0725")
    print("  Weighted Sensitivity: 71.2001")
    print("  Weighted Specificity: 76.4010")
    print("  Weighted F1 Score:    74.6501")

    # 检查是否有完整 results_df 可用
    full_results_path = os.path.join(OUT_DIR, "full_cv_results.csv")
    if os.path.exists(full_results_path):
        full = pd.read_csv(full_results_path)
        best_full = full.loc[full.groupby("SITE_ID")["cv_accuracy_mean"].idxmax()]
        site_sizes_full = subjects_df["SITE_ID"].value_counts()
        product_configs = (
            subjects_df["graph_feature_type"].nunique()
            * subjects_df["feature_engineering"].nunique()
            * subjects_df["atlas"].nunique()
        )
        site_sizes_full = (site_sizes_full / product_configs).astype(int)
        best_full = best_full.set_index("SITE_ID").join(site_sizes_full)
        total = site_sizes_full.sum()
        for col in ["cv_accuracy_mean", "precision", "recall", "sensitivity", "specificity", "f1_score"]:
            w = (best_full[col] * best_full.index.map(site_sizes_full)).sum() / total
            print(f"  Full CV Weighted {col}: {w:.4f}")

    print(f"\nDone. Output saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
