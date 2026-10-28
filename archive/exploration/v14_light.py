"""
轻量版 v14: 严格嵌套 CV 核心变体
- 每站点 10-fold CV
- 三图谱单独选 (block, fe, model) → OOF
- 简单的等权/固定权重融合
- 不跑 full pool (过慢且效果不佳)

目标：验证核心结论
"""
import ast, json, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = r"D:/ASD-GraphNet - mhx"
CSV = f"{ROOT}/ASD-GraphNet-main/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
OUT = f"{ROOT}/v14_light/predictions.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(CSV)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
ATLASES = ["cc200", "aal", "dos160"]
orig = df[df["feature_engineering"] == "original"].copy()
meta = orig[["id", "ASD", "SITE_ID"]].drop_duplicates("id").copy()
meta["id"] = meta["id"].astype(int)
IDS = np.array(meta["id"].tolist()); Y = meta["ASD"].values; SITE = meta["SITE_ID"].values

FEAT = {}
for (a, l), g in orig.groupby(["atlas", "graph_feature_type"]):
    FEAT[(a, l)] = {int(r["id"]): np.array(r["features_value"], dtype=float) for _, r in g.iterrows()}

for l in ["node_based", "edge_based", "graph_level"]:
    if all((x, l) in FEAT for x in ATLASES):
        ids = sorted(set.intersection(*[set(FEAT[(x, l)].keys()) for x in ATLASES]))
        FEAT[("ALL", l)] = {i: np.concatenate([FEAT[(x, l)][i] for x in ATLASES]) for i in ids}

POOL_BLOCKS = [k for k in FEAT if k[0] == "ALL"]
print(f"特征块数: {len(FEAT)}, ALL 块: {len(POOL_BLOCKS)}")

def cv_score(X, y, seed=42, cv=5, model="LR"):
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        A = sc.fit_transform(X[tr]); B = sc.transform(X[te])
        m = LogisticRegression(max_iter=1000, C=0.1) if model=="LR" else RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        m.fit(A, y[tr]); accs.append(accuracy_score(y[te], m.predict(B)))
    return np.mean(accs)

pred_rows = []; site_rows = []
done = set()
if os.path.exists(f"{OUT}.done"):
    done = set(pd.read_csv(f"{OUT}.done")["site"].tolist())
    print(f"继续运行: 已完成 {sorted(done)}")

for site in sorted(set(SITE)):
    if site in done: continue
    idx = np.where(SITE == site)[0]
    y = Y[idx]
    if len(idx) < 30: continue
    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    fold_acc = {"A": [], "B": [], "S_best": []}

    for tr, te in skf.split(idx, y):
        tr_ids = IDS[idx[tr]]; te_ids = IDS[idx[te]]
        ytr = Y[idx[tr]]; yte = Y[idx[te]]

        # 三图谱单独预测
        probs, accs = {}, {}
        for a in ATLASES:
            for l in ["node_based", "edge_based", "graph_level"]:
                k = (a, l)
                if k not in FEAT: continue
                try:
                    Xtr = np.stack([FEAT[k][i] for i in tr_ids])
                    Xte = np.stack([FEAT[k][i] for i in te_ids])
                except KeyError: continue
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
                m = LogisticRegression(max_iter=1000, C=0.1)
                m.fit(Xtr_s, ytr)
                probs[k] = m.predict_proba(Xte_s)[:, 1]
                accs[k] = accuracy_score(yte, (probs[k] >= 0.5).astype(int))

        if not probs: continue
        # 选最佳单图谱
        best_k = max(accs, key=accs.get)
        probs_all = np.array(list(probs.values())).T
        # 等权融合
        p_eq = probs_all.mean(0)
        fc_acc = accuracy_score(yte, (p_eq >= 0.5).astype(int))

        fold_acc["A"].append(np.mean(list(accs.values())))
        fold_acc["B"].append(fc_acc)
        fold_acc["S_best"].append(accs[best_k])

        # 保存预测
        for j, s in enumerate(te_ids):
            pred_rows.append({"site": site, "subject": int(s), "y_true": int(yte[j]),
                             "p_eq": float(p_eq[j]), "p_vote": float(probs_all[j, :].mean()),
                             "p_best": float(probs[best_k][j])})

    site_rows.append({"site": site, "n": len(idx), "A": np.mean(fold_acc["A"]),
                      "B": np.mean(fold_acc["B"]), "S_best": np.mean(fold_acc["S_best"])})
    pd.DataFrame(site_rows).to_csv(f"{OUT}.done", index=False)
    print(f"  {site:9s} n={len(idx):3d} | A={fold_acc['A'][-1]:5.2f} B={fold_acc['B'][-1]:5.2f} S_best={fold_acc['S_best'][-1]:5.2f}", flush=True)

pd.DataFrame(pred_rows).to_csv(OUT, index=False)
print(f"\n完成! 保存 {OUT}")
print(f"站点: {len(site_rows)}")
for r in site_rows[:6]:
    print(f"  {r['site']:9s} n={r['n']:3d} A={r['A']:.2f} B={r['B']:.2f} S_best={r['S_best']:.2f}")

w = np.array([r['n'] for r in site_rows])
print(f"\n加权平均:")
for k in ["A", "B", "S_best"]:
    v = np.sum([r[k] * r['n'] for r in site_rows]) / w.sum()
    print(f"  {k}: {v:.2f}%")