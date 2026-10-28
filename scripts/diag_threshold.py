"""
诊断: 决策阈值问题
- 模型 ROC AUC vs 固定阈值准确率
- 用训练折 OOF 学阈值 vs 固定 0.5
"""
import ast, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = r"D:/ASD-GraphNet - mhx"
df = pd.read_csv(f"{ROOT}/ASD-GraphNet-main/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
df["features_value"] = df["features_value"].apply(ast.literal_eval)
orig = df[df["feature_engineering"] == "original"].copy()
meta = orig[["id", "ASD", "SITE_ID"]].drop_duplicates("id").copy()
meta["id"] = meta["id"].astype(int)
IDS = np.array(meta["id"].tolist()); Y = meta["ASD"].values; SITE = meta["SITE_ID"].values
FEAT = {}
for (a, l), g in orig.groupby(["atlas", "graph_feature_type"]):
    FEAT[(a, l)] = {int(r["id"]): np.array(r["features_value"], dtype=float) for _, r in g.iterrows()}

def best_threshold(y, p, grid=None):
    """在 (y,p) 上选使 balanced_accuracy 最大的阈值"""
    grid = grid if grid is not None else np.unique(np.round(p, 3))
    best, bt = -1, 0.5
    for t in grid:
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best: best, bt = ba, t
    return bt, best

print("=" * 88)
print("诊断: 决策阈值对准确率的影响 (cc200/node_based/original + LR)")
print("=" * 88)
print(f"{'site':9s} {'n':>4s} {'AUC':>7s} {'acc@0.5':>9s} {'bal@0.5':>9s} "
      f"{'acc@thr*':>9s} {'bal@thr*':>9s} {'thr*':>6s} {'|prob|':>7s}")
res = []
for site in sorted(set(SITE)):
    idx = np.where(SITE == site)[0]; y = Y[idx]
    if len(idx) < 25: continue
    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    a05, b05, athr, bthr, aucs, thrs, probs_all = [], [], [], [], [], [], []
    for tr, te in skf.split(idx, y):
        tr_i, te_i = idx[tr], idx[te]
        tr_ids, te_ids = IDS[tr_i], IDS[te_i]
        ytr, yte = Y[tr_i], Y[te_i]
        Xtr = np.stack([FEAT[("cc200","node_based")][i] for i in tr_ids])
        Xte = np.stack([FEAT[("cc200","node_based")][i] for i in te_ids])
        sc = StandardScaler(); A = sc.fit_transform(Xtr); B = sc.transform(Xte)
        # 内层 OOF (训练折内)
        skf2 = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros(len(ytr))
        for itr, ite in skf2.split(A, ytr):
            m = LogisticRegression(max_iter=1000, C=0.1); m.fit(A[itr], ytr[itr])
            oof[ite] = m.predict_proba(A[ite])[:, 1]
        t_star, _ = best_threshold(ytr, oof)
        m = LogisticRegression(max_iter=1000, C=0.1); m.fit(A, ytr)
        pt = m.predict_proba(B)[:, 1]
        a05.append(accuracy_score(yte, (pt >= .5).astype(int)))
        b05.append(balanced_accuracy_score(yte, (pt >= .5).astype(int)))
        athr.append(accuracy_score(yte, (pt >= t_star).astype(int)))
        bthr.append(balanced_accuracy_score(yte, (pt >= t_star).astype(int)))
        aucs.append(roc_auc_score(yte, pt)); thrs.append(t_star); probs_all += list(pt)
    r = {"site": site, "n": len(idx), "auc": np.mean(aucs), "a05": np.mean(a05)*100,
         "b05": np.mean(b05)*100, "athr": np.mean(athr)*100, "bthr": np.mean(bthr)*100,
         "thr": np.mean(thrs), "pmed": np.median(probs_all)}
    res.append(r)
    print(f"{site:9s} {r['n']:4d} {r['auc']:7.3f} {r['a05']:9.2f} {r['b05']:9.2f} "
          f"{r['athr']:9.2f} {r['bthr']:9.2f} {r['thr']:6.3f} {r['pmed']:7.3f}")

R = pd.DataFrame(res); w = R["n"].values
print(f"\n加权平均: AUC={R['auc'].dot(w)/w.sum():.3f}  "
      f"acc@0.5={R['a05'].dot(w)/w.sum():.2f}%  acc@thr*={R['athr'].dot(w)/w.sum():.2f}%  "
      f"bal@0.5={R['b05'].dot(w)/w.sum():.2f}%  bal@thr*={R['bthr'].dot(w)/w.sum():.2f}%")
R.to_csv(f"{ROOT}/v15_out/threshold_diagnostic.csv", index=False)
