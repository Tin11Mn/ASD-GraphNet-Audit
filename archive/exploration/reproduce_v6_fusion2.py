"""
ASD-GraphNet 改进 V6: 多图谱融合 (使用论文预处理的 mi_10/pca_10 特征)
设计: 对比"单图谱最优"vs"多图谱拼接融合",两者都用论文预处理好的特征
"""
import ast, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

data_path = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(data_path)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

core = df[(df["graph_feature_type"] == "node_based") & (df["feature"] == "degree")]
sites = sorted(core["SITE_ID"].unique())

def build_models():
    return {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGB": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
        "LGBM": None,  # skip LGBM for fused features (dim mismatch)
        "SVM": None,
    }

def cv_eval(X, y, model):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
        model.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, model.predict(X_te)))
    return np.mean(accs) * 100

results = []
for site in sites:
    site_data = core[core["SITE_ID"] == site]
    for fe in ["mi_10", "pca_10"]:
        # 收集每个 atlas 下同一被试的特征
        by_subj = {}
        labels = {}
        for atlas in ["aal", "cc200", "dos160"]:
            a = site_data[(site_data["atlas"] == atlas) & (site_data["feature_engineering"] == fe)]
            for _, row in a.iterrows():
                sid = row["id"]
                by_subj.setdefault(sid, {})[atlas] = np.array(row["features_value"], dtype=float)
                labels[sid] = row["ASD"]
        subs = [s for s in by_subj if all(k in by_subj[s] for k in ["aal","cc200","dos160"])]
        if len(subs) < 20:
            continue
        y = np.array([labels[s] for s in subs])
        X_fused = np.stack([np.concatenate([by_subj[s][a] for a in ["aal","cc200","dos160"]]) for s in subs])
        
        for atlas in ["aal", "cc200", "dos160"]:
            X_single = np.stack([by_subj[s][atlas] for s in subs])
            for mname, m in build_models().items():
                if m is None: continue
                acc_s = cv_eval(X_single, y, m)
                acc_f = cv_eval(X_fused, y, m)
                results.append({
                    "site": site, "fe": fe, "atlas": atlas, "model": mname,
                    "single": acc_s, "fused": acc_f, "diff": acc_f - acc_s,
                    "n": len(subs)
                })
        # 打印该站点的最佳
        site_res = [r for r in results if r["site"] == site and r["fe"] == fe]
        if site_res:
            bs = max(site_res, key=lambda r: r["single"])
            bf = max(site_res, key=lambda r: r["fused"])
            print(f"{site}/{fe} n={len(subs)}: best-single={bs['single']:.2f}% ({bs['atlas']}/{bs['model']})  best-fused={bf['fused']:.2f}% ({bf['model']})", flush=True)

res = pd.DataFrame(results)
print(f"\n=== 汇总 ===")
print(f"简单平均: single={res['single'].mean():.2f}%  fused={res['fused'].mean():.2f}%  diff={res['diff'].mean():+.2f}%")

# 加权: 每站点取最优配置
w = res.groupby("site")["n"].first()
bs = res.loc[res.groupby("site")["single"].idxmax()].set_index("site")
bf = res.loc[res.groupby("site")["fused"].idxmax()].set_index("site")
wbs = (bs["single"] * w).sum() / w.sum()
wbf = (bf["fused"] * w).sum() / w.sum()
print(f"\n加权平均 (best per site):")
print(f"  单图谱最优: {wbs:.2f}%")
print(f"  多图谱融合: {wbf:.2f}%  ({wbf-wbs:+.2f}%)")
print(f"  (论文 75.25%, 原始复现 75.63%)")
res.to_csv(r"D:/ASD-GraphNet - mhx/improved_v6_fusion_mi_pca.csv", index=False)
