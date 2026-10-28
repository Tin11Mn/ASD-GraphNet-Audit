"""
快速实验 v16 (简化版): 优化权重学习 + Stacking

3 个方案:
  E1  L2 正则化权重 : Ridge 线性模型 (alpha=1.0) -> 权重
  E2  权重置信度加权: 根据 OOF 置信度给高置信度样本更大权重
  E3  Ridge Stacking : OOF 概率 -> Ridge 分类器预测

目标: 验证哪个改进对 58% 基线有显著增益
"""
import ast, os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

ROOT = r"D:/ASD-GraphNet - mhx"
CSVP = f"{ROOT}/ASD-GraphNet-main/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
OUTDIR = f"{ROOT}/v16_out"; os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(CSVP); df["features_value"] = df["features_value"].apply(ast.literal_eval)
ATLASES = ["cc200", "aal", "dos160"]
LVLS = ["node_based", "edge_based", "graph_level"]

orig = df[df["feature_engineering"] == "original"].copy()
FEAT = {}
for (a, l), g in orig.groupby(["atlas", "graph_feature_type"]):
    FEAT[(a, l)] = {int(r["id"]): np.array(r["features_value"], dtype=float) for _, r in g.iterrows()}
meta = orig[["id", "ASD", "SITE_ID"]].drop_duplicates("id").copy()
meta["id"] = meta["id"].astype(int)
IDS = np.array(meta["id"].tolist()); Y = meta["ASD"].values; SITE = meta["SITE_ID"].values
print(f"被试 {len(IDS)}  ASD={Y.sum()}  站点 {len(set(SITE))}")

def make_model(name, seed):
    if name == "LR":  return LogisticRegression(max_iter=1500, C=0.1, random_state=seed)
    if name == "RF":  return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if name == "XGB": return XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss", verbosity=0, tree_method="hist")
    raise ValueError(name)
MODELS = ["LR", "RF", "XGB"]

def fit_fe(Xtr, ytr, Xte, mode, seed=42):
    if mode == "original": return Xtr, Xte
    k = max(1, int(round(Xtr.shape[1] * (int(mode.split("_")[1])/100.0))))
    if mode.startswith("mi"):
        k = min(k, Xtr.shape[1])
        mi = mutual_info_classif(Xtr, ytr, random_state=seed, n_neighbors=max(1, min(3, len(ytr)//4)))
        top = np.argsort(mi)[::-1][:k]
        return Xtr[:, top], Xte[:, top]
    k = min(k, Xtr.shape[1], Xtr.shape[0])
    p = PCA(n_components=k, random_state=seed)
    return p.fit_transform(Xtr), p.transform(Xte)

def auc_safe(y, p):
    if len(np.unique(y)) < 2: return 0.5
    try: return roc_auc_score(y, p)
    except: return 0.5

def inner_cv(Xtr, ytr, Xte, seed):
    k = min(3, int(np.bincount(ytr, minlength=2).min()))
    if k < 2: return None
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    oofs = {mn: np.zeros(len(ytr)) for mn in MODELS}
    tps = {mn: np.zeros(len(Xte)) for mn in MODELS}
    cnt = np.zeros(len(ytr))
    sc_all = StandardScaler()
    Xtr_s = sc_all.fit_transform(Xtr); Xte_s = sc_all.transform(Xte)
    for itr, ite in skf.split(Xtr_s, ytr):
        for mn in MODELS:
            m = make_model(mn, seed); m.fit(Xtr_s[itr], ytr[itr])
            oofs[mn][ite] += m.predict_proba(Xtr_s[ite])[:, 1]; cnt[ite] += 1
    for mn in MODELS:
        m = make_model(mn, seed); m.fit(Xtr_s, ytr)
        tps[mn] = m.predict_proba(Xte_s)[:, 1]
    if not (cnt > 0).all(): return None
    best = min(MODELS, key=lambda mn: -auc_safe(ytr, oofs[mn]/cnt))
    return best, oofs[best]/cnt, tps[best]

def learn_thr(y, p):
    grid = np.unique(np.round(p, 2))
    best, bt = -1, 0.5
    for t in grid:
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best: best, bt = ba, t
    return bt

# ══════════════════ 主循环 ══════════════════
OUTER, SEED = 10, 42
site_data = []

print("\n开始 v16 实验...")
print("=" * 84)

for site in sorted(set(SITE)):
    idx = np.where(SITE == site)[0]; y = Y[idx]
    if len(idx) < 25: continue
    skf = StratifiedKFold(OUTER, shuffle=True, random_state=SEED)
    accs = {"baseline": [], "E1_ridge": [], "E2_weighted": [], "E3_stack": []}
    
    for tr, te in skf.split(idx, y):
        tr_i, te_i = idx[tr], idx[te]
        tr_ids, te_ids = IDS[tr_i], IDS[te_i]
        ytr, yte = Y[tr_i], Y[te_i]
        
        # 三图谱 OOF
        base = {}
        for a in ATLASES:
            for (blk, l) in [(a, l) for l in LVLS if (a, l) in FEAT]:
                for fe in ["original", "mi_10"]:
                    try:
                        Xtr = np.stack([FEAT[(a, l)][i] for i in tr_ids])
                        Xte = np.stack([FEAT[(a, l)][i] for i in te_ids])
                    except KeyError: continue
                    A, B = fit_fe(Xtr, ytr, Xte, fe, SEED)
                    r = inner_cv(A, ytr, B, SEED)
                    if r is None: continue
                    mn, oof, pt = r
                    a_auc = auc_safe(ytr, oof)
                    if a not in base or a_auc > base[a][0]:
                        base[a] = (a_auc, blk, fe, mn, oof, pt)
        
        if len(base) < 2: continue
        A3 = np.array([base[a][4] for a in ATLASES])
        T3 = np.array([base[a][5] for a in ATLASES])
        
        # Baseline: 负对数似然优化权重
        def nll(w, P, y):
            q = (w[0]*P[0] + w[1]*P[1] + w[2]*P[2])
            q = np.clip(q, 1e-6, 1-1e-6)
            return -np.mean(y * np.log(q) + (1-y)*np.log(1-q))
        from scipy.optimize import minimize
        res = minimize(lambda w: nll(w, A3, ytr), np.ones(3)/3, method="Nelder-Mead",
                      options={"maxiter": 200, "xatol": 1e-3})
        w_bl = res.x / res.x.sum()
        p_bl = (w_bl[:, None] * T3).sum(0)
        t_bl = learn_thr(ytr, A3.mean(0))
        accs["baseline"].append(accuracy_score(yte, (p_bl >= t_bl).astype(int)) * 100)
        
        # E1: L2 正则化 Ridge
        ridge = RidgeClassifier(alpha=1.0, fit_intercept=False)
        ridge.fit(A3.T, ytr)
        w_e1 = ridge.coef_.ravel()
        w_e1 = np.exp(w_e1 - w_e1.max()); w_e1 /= w_e1.sum()
        p_e1 = (w_e1[:, None] * T3).sum(0)
        t_e1 = learn_thr(ytr, A3.mean(0))
        accs["E1_ridge"].append(accuracy_score(yte, (p_e1 >= t_e1).astype(int)) * 100)
        
        # E2: OOF 置信度加权 (高置信度样本更大)
        conf = np.abs(2 * A3 - 1)  # (3, n_tr)
        sample_weights = conf.max(0)  # 每样本置信度
        w_e2 = w_bl.copy()  # 先用 baseline 权重
        # 用置信度加权的 OOF 重新拟合权重
        res2 = minimize(lambda w: -np.mean(sample_weights *
                    (ytr * np.log((w[0]*A3[0]+w[1]*A3[1]+w[2]*A3[2]).clip(1e-6,1-1e-6)) +
                     (1-ytr)*np.log(1-(w[0]*A3[0]+w[1]*A3[1]+w[2]*A3[2]).clip(1e-6,1-1e-6)))),
                    np.ones(3)/3, method="Nelder-Mead", options={"maxiter": 200})
        w_e2 = np.exp(res2.x - res2.x.max()); w_e2 /= w_e2.sum()
        p_e2 = (w_e2[:, None] * T3).sum(0)
        t_e2 = learn_thr(ytr, A3.mean(0))
        accs["E2_weighted"].append(accuracy_score(yte, (p_e2 >= t_e2).astype(int)) * 100)
        
        # E3: Ridge Stacking
        ridge_stack = RidgeClassifier(alpha=1.0)
        ridge_stack.fit(A3.T, ytr)
        d_e3 = ridge_stack.decision_function(T3.T)
        # sigmoid 转概率
        p_e3 = 1 / (1 + np.exp(-d_e3))
        t_e3 = learn_thr(ytr, A3.mean(0))
        accs["E3_stack"].append(accuracy_score(yte, (p_e3 >= t_e3).astype(int)) * 100)
    
    row = {"site": site, "n": len(idx)}
    for k, v in accs.items(): row[k] = np.mean(v) if v else np.nan
    site_data.append(row)
    print(f"  {site:9s} n={len(idx):3d} | base={row['baseline']:5.2f} "
          f"E1={row['E1_ridge']:5.2f} E2={row['E2_weighted']:5.2f} "
          f"E3={row['E3_stack']:5.2f}", flush=True)

sr = pd.DataFrame(site_data)
sr.to_csv(f"{OUTDIR}/results.csv", index=False)

print("\n" + "=" * 84)
print("v16 汇总")
w = sr["n"].values
for c in ["baseline", "E1_ridge", "E2_weighted", "E3_stack"]:
    v = np.array(sr[c].values)
    m = np.nansum(v * w) / w.sum()
    print(f"  {c:14s} {m:+5.2f}%")
print(f"\n提升: E1={sr['E1_ridge'].mean()-sr['baseline'].mean():+.2f}pp, "
      f"E2={sr['E2_weighted'].mean()-sr['baseline'].mean():+.2f}pp, "
      f"E3={sr['E3_stack'].mean()-sr['baseline'].mean():+.2f}pp")
print(f"用时 {time.time()-t0:.0f}s")