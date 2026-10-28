"""
=============================================================================
严格无泄露实验 v15 (修复版) — 多图谱可靠性融合
=============================================================================

速度修复 (相对 v14):
  1. MI/PCA 拟合提升到「外层训练折」层面 —— 完全不接触外层测试标签,
     但内层 CV 不再重复拟合 MI (v14 每外折调用 ~90 次, 现在 ~15 次)
  2. pool 块的 12 个候选用 joblib 并行 (32 核)
  3. 内层 RF/XGB 用 n_estimators=50 (最终拟合用 100)
  4. 缓存 (block, fe) -> 已变换的 (train, test) 矩阵, 避免重复 StandardScaler

协议 (仍严格无泄露):
  外层 10-fold CV (每站点)  -> 测试折从不参与任何拟合
  外层训练折: 拟合 StandardScaler + MI/PCA (只用训练折标签)
  内层 5-fold (在已变换的训练折上): 只选模型, 得到 OOF 概率
  融合权重 / 门控网络: 只用训练折 OOF
  报告准确率: 只在外层测试折

输出: v15_out/
  predictions.csv   逐被试外层测试预测
  site_summary.csv  每站点汇总
"""
import ast, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

ROOT = r"D:/ASD-GraphNet - mhx"
CSVP = f"{ROOT}/ASD-GraphNet-main/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
OUTDIR = f"{ROOT}/v15_out"; os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════ 1. 数据 ══════════════════
df = pd.read_csv(CSVP)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
ATLASES = ["cc200", "aal", "dos160"]
LVLS = ["node_based", "edge_based", "graph_level"]

orig = df[df["feature_engineering"] == "original"].copy()
FEAT = {}
for (a, l), g in orig.groupby(["atlas", "graph_feature_type"]):
    FEAT[(a, l)] = {int(r["id"]): np.array(r["features_value"], dtype=float)
                    for _, r in g.iterrows()}
meta = orig[["id", "ASD", "SITE_ID"]].drop_duplicates("id").copy()
meta["id"] = meta["id"].astype(int)
meta = meta.sort_values("id").reset_index(drop=True)
IDS = np.array(meta["id"].tolist()); Y = meta["ASD"].values; SITE = meta["SITE_ID"].values
print(f"被试 {len(IDS)}  ASD={Y.sum()}  站点 {len(set(SITE))}")

# 跨图谱拼接块
for l in LVLS:
    if all((x, l) in FEAT for x in ATLASES):
        ids = sorted(set.intersection(*[set(FEAT[(x, l)].keys()) for x in ATLASES]))
        FEAT[("ALL", l)] = {i: np.concatenate([FEAT[(x, l)][i] for x in ATLASES]) for i in ids}
for a in ATLASES:
    for l in LVLS:
        duo = [x for x in ATLASES if x != a]
        if all((x, l) in FEAT for x in duo):
            ids = sorted(set.intersection(*[set(FEAT[(x, l)].keys()) for x in duo]))
            FEAT[("DUAL", a, l)] = {i: np.concatenate([FEAT[(x, l)][i] for x in duo]) for i in ids}

ATLAS_BLOCKS = {a: [(a, l) for l in LVLS if (a, l) in FEAT] for a in ATLASES}
POOL_BLOCKS = [k for k in FEAT if k[0] == "ALL" or (len(k) == 3 and k[0] == "DUAL")]
print(f"特征块 {len(FEAT)}  pool候选 {len(POOL_BLOCKS)}")

# ══════════════════ 2. 模型 ══════════════════
def make_model(name, seed, fast=False):
    n = 50 if fast else 100
    if name == "LR":   return LogisticRegression(max_iter=1500, C=0.1)
    if name == "DT":   return DecisionTreeClassifier(random_state=seed)
    if name == "RF":   return RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=1)
    if name == "XGB":  return XGBClassifier(n_estimators=n, random_state=seed,
                                           eval_metric="logloss", verbosity=0, tree_method="hist")
    if name == "LGBM": return LGBMClassifier(n_estimators=n, random_state=seed, verbose=-1, n_jobs=1)
    if name == "SVM":  return SVC(probability=True, random_state=seed)
    if name == "MLP":  return MLPClassifier(max_iter=300, random_state=seed)
    raise ValueError(name)
MODEL_NAMES = ["LR", "DT", "RF", "XGB", "LGBM"]

# ══════════════════ 3. 特征工程 (拟合在外层训练折) ══════════════════
def fit_fe(Xtr, ytr, Xte, mode, seed=42):
    if mode == "original": return Xtr, Xte
    pct = int(mode.split("_")[1]) / 100.0
    k = max(1, int(round(Xtr.shape[1] * pct)))
    if mode.startswith("mi"):
        k = min(k, Xtr.shape[1])
        mi = mutual_info_classif(Xtr, ytr, random_state=seed,
                                  n_neighbors=max(1, min(3, len(ytr) // 4)))
        top = np.argsort(mi)[::-1][:k]
        return Xtr[:, top], Xte[:, top]
    k = min(k, Xtr.shape[1], Xtr.shape[0])
    p = PCA(n_components=k, random_state=seed)
    return p.fit_transform(Xtr), p.transform(Xte)

def transform_block(block, fe, tr_ids, te_ids, ytr, seed=42):
    """在训练折上拟合 scaler + MI/PCA, 返回 (Xtr_t, Xte_t)"""
    Xtr = np.stack([FEAT[block][i] for i in tr_ids])
    Xte = np.stack([FEAT[block][i] for i in te_ids])
    sc = StandardScaler()
    A = sc.fit_transform(Xtr); B = sc.transform(Xte)
    return fit_fe(A, ytr, B, fe, seed)

# ══════════════════ 4. 内层: 只选模型 (在已变换的训练折上) ══════════════════
def inner_models(Xtr, y, Xte, seed, n_inner=5, models=None):
    """Xtr: 已变换的训练折特征; Xte: 已变换的测试折特征.
       内层 CV 只用于选模型, 返回 (best_model, oof_probs, test_probs)"""
    models = models or MODEL_NAMES
    k = min(n_inner, int(np.bincount(y, minlength=2).min()))
    if k < 2: return None
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    oofs = {mn: np.zeros(len(y)) for mn in models}
    cnt = np.zeros(len(y))
    for itr, ite in skf.split(Xtr, y):
        for mn in models:
            m = make_model(mn, seed, fast=True); m.fit(Xtr[itr], y[itr])
            oofs[mn][ite] += m.predict_proba(Xtr[ite])[:, 1]; cnt[ite] += 1
    if not (cnt > 0).all(): return None
    # 测试折概率: 用全训练折重拟合每个模型 (各模型的最终预测)
    tps = {}
    for mn in models:
        m = make_model(mn, seed, fast=False); m.fit(Xtr, y)
        tps[mn] = m.predict_proba(Xte)[:, 1]
    best = None
    for mn in models:
        a = auc_safe(y, oofs[mn] / cnt)
        if best is None or a > best[0]: best = (a, mn, oofs[mn] / cnt, tps[mn])
    return (best[1], best[2], best[3]) if best else None

def auc_safe(y, p):
    if len(np.unique(y)) < 2: return 0.5
    try: return roc_auc_score(y, p)
    except Exception: return 0.5

# ══════════════════ 5. 校准 / 权重 / 门控 ══════════════════
EPS = 1e-6
def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p)).reshape(-1, 1)
def platt_fit(oof, y):
    if len(np.unique(y)) < 2 or len(np.unique(oof)) < 2: return None
    lr = LogisticRegression(C=1e4, max_iter=1000); lr.fit(to_logit(oof), y); return lr
def platt_apply(c, p):
    return p if c is None else c.predict_proba(to_logit(p))[:, 1]
def softmax_w(pr):
    e = np.exp(pr - pr.max()); return e / e.sum()
def fit_fixed_weights(op, y):
    def obj(pr):
        w = softmax_w(pr); p = np.clip((w[:, None] * op).sum(0), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    best, bv = np.zeros(op.shape[0]), np.inf
    inits = [np.zeros(op.shape[0])]
    if op.shape[0] == 3:
        inits += [np.array([.5, -.5, 0.]), np.array([-.5, .5, 0.]), np.array([0., .5, -.5])]
    for init in inits:
        r = minimize(obj, init, method="Nelder-Mead",
                     options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
        if r.fun < bv: bv, best = r.fun, softmax_w(r.x)
    return best
def fit_rel_weights(op, y):
    nll = []
    for a in range(op.shape[0]):
        p = np.clip(op[a], EPS, 1 - EPS)
        nll.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return softmax_w(-np.array(nll) / max(np.std(nll), 1e-6))

class GateNet:
    """门控 MLP: [probs, confidences] -> softmax 权重. VJP 正确实现."""
    def __init__(self, n_atlas, hidden=8, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, .3, (n_atlas * 2, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, .3, (hidden, n_atlas)); self.b2 = np.zeros(n_atlas)
        self.n = n_atlas
    def _fwd(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        g = h @ self.W2 + self.b2; g = g - g.max(1, keepdims=True)
        e = np.exp(g); return e / e.sum(1, keepdims=True), h
    def fit(self, probs, y, epochs=400, lr=0.05):
        P = probs.T; X = np.concatenate([P, np.abs(2 * P - 1)], 1); n = len(y)
        for _ in range(epochs):
            w, h = self._fwd(X)
            Pmix = w * P
            pm = np.clip(Pmix.sum(1), EPS, 1 - EPS)
            dpm = (pm - y) / (pm * (1 - pm))
            dPmix = np.repeat(dpm[:, None], self.n, axis=1)
            dL_dw = dPmix * P
            dw = w * (dL_dw - (w * dL_dw).sum(1, keepdims=True))
            dW2 = h.T @ dw / n; db2 = dw.sum(0) / n
            dh = (dw @ self.W2.T) / n; dz = dh * (1 - h ** 2)
            self.W1 -= lr * (X.T @ dz / n); self.b1 -= lr * dz.sum(0) / n
            self.W2 -= lr * dW2; self.b2 -= lr * db2
        return self
    def predict_w(self, probs):
        P = probs.T; X = np.concatenate([P, np.abs(2 * P - 1)], 1)
        return self._fwd(X)[0].T

# ══════════════════ 6. pool 选优 (并行) ══════════════════
def pool_worker(block, fe, y, tr_ids, te_ids, seed):
    try:
        A, B = transform_block(block, fe, tr_ids, te_ids, y, seed)
    except Exception:
        return None
    r = inner_models(A, y, B, seed, n_inner=3, models=["LR", "RF"])
    if r is None: return None
    return (auc_safe(y, r[1]), block, fe, r[0], r[1], r[2])

def select_pool(y, tr_ids, te_ids, seed):
    jobs = [(b, fe) for b in POOL_BLOCKS for fe in ["original", "mi_10"]]
    res = Parallel(n_jobs=1, backend="threading")(
        delayed(pool_worker)(b, fe, y, tr_ids, te_ids, seed) for b, fe in jobs)
    res = [r for r in res if r is not None]
    if not res: return None
    return max(res, key=lambda r: r[0])

# ══════════════════ 7. 主循环 ══════════════════
import sys
OUTER, SEED = 10, 42
# 可选: 通过命令行指定本次要跑的站点批次, e.g. python clean_v15_core.py 0 1 2 3
BATCH = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
ALL_SITES = sorted(set(SITE))
RUN_SITES = [ALL_SITES[i] for i in BATCH] if BATCH else ALL_SITES
print(f"本次运行站点: {RUN_SITES}")

pred_rows, site_rows = [], []
done = set()
if os.path.exists(f"{OUTDIR}/site_summary.csv"):
    try:
        d = pd.read_csv(f"{OUTDIR}/site_summary.csv")
        done = set(d["site"].tolist())
        site_rows = d.to_dict("records")
        print(f"已完成站点: {sorted(done)}")
    except Exception: pass
# 读回已有预测 (保留)
if os.path.exists(f"{OUTDIR}/predictions.csv"):
    try:
        pred_rows = pd.read_csv(f"{OUTDIR}/predictions.csv").to_dict("records")
        print(f"已有预测 {len(pred_rows)} 行")
    except Exception: pass

for site in RUN_SITES:
    if site in done: continue
    idx = np.where(SITE == site)[0]; y = Y[idx]
    if len(idx) < 25 or np.bincount(y, minlength=2).min() < 6: continue
    skf = StratifiedKFold(OUTER, shuffle=True, random_state=SEED)
    FACC = {k: [] for k in ["A_eq", "B_fixed", "C_conf", "D_gate", "M_vote",
                            "S_cc200", "S_aal", "S_dos160", "S_pool", "S_best_atlas"]}
    t_site = time.time()
    for tr, te in skf.split(idx, y):
        tr_i, te_i = idx[tr], idx[te]
        tr_ids, te_ids = IDS[tr_i], IDS[te_i]
        ytr, yte = Y[tr_i], Y[te_i]

        # ── 三图谱: 选 (block, fe) 用 LR, 再选模型 ──
        base = {}
        for a in ATLASES:
            bestA = None
            for b in ATLAS_BLOCKS[a]:
                for fe in ["original", "mi_10"]:
                    try:
                        A, B = transform_block(b, fe, tr_ids, te_ids, ytr, SEED)
                    except Exception:
                        continue
                    r = inner_models(A, ytr, B, SEED, n_inner=3, models=["LR"])
                    if r is None: continue
                    a_auc = auc_safe(ytr, r[1])
                    if bestA is None or a_auc > bestA[0]: bestA = (a_auc, b, fe, A, B)
            if bestA is None: continue
            _, b, fe, A, B = bestA
            r = inner_models(A, ytr, B, SEED, n_inner=3)
            if r is None: continue
            mn, oof, pt = r
            # 不做 Platt 校准: 小样本折上 Platt 过度拟合, 把测试概率推到极端(0.9), 破坏0.5阈值
            base[a] = {"block": b, "fe": fe, "model": mn,
                       "oof": oof, "test": pt}
        if len(base) < 3: continue

        A3 = np.array([base[a]["oof"] for a in ATLASES])
        T3 = np.array([base[a]["test"] for a in ATLASES])

        # 从训练折 OOF 学习阈值 (平衡准确率最大化), 应用到测试折
        def learn_thresh(y, p):
            grid = np.unique(np.round(p, 2))
            best, bt = -1, 0.5
            for t in grid:
                ba = balanced_accuracy_score(y, (p >= t).astype(int))
                if ba > best: best, bt = ba, t
            return bt
        thresh = [learn_thresh(ytr, A3[i]) for i in range(3)]
        singles = {a: (T3[i] >= thresh[i]).astype(int) for i, a in enumerate(ATLASES)}

        # 从训练折 OOF 学习阈值 (平衡准确率最大化)
        def learn_thresh(y, p):
            grid = np.unique(np.round(p, 2))
            best, bt = -1, 0.5
            for t in grid:
                ba = balanced_accuracy_score(y, (p >= t).astype(int))
                if ba > best: best, bt = ba, t
            return bt
        thresh_vec = [learn_thresh(ytr, A3[i]) for i in range(3)]
        thresh_vec = np.array(thresh_vec)

        # 等权融合 (概率平均后也用训练折学的阈值)
        p_eq = T3.mean(0)
        thresh_eq = learn_thresh(ytr, A3.mean(0))
        pred_eq = (p_eq >= thresh_eq).astype(int)

        # 固定权重融合 (权重用 OOF 学, 预测用 0.5 阈值)
        w_fixed = fit_fixed_weights(A3, ytr)
        p_fixed = (w_fixed[:, None] * T3).sum(0)
        pred_fixed = (p_fixed >= .5).astype(int)

        # 置信度自适应
        w_rel = fit_rel_weights(A3, ytr)
        bg, bll = 0.0, np.inf
        for g in [0., .5, 1., 1.5, 2., 3., 4.]:
            w = w_rel[:, None] * np.exp(g * np.abs(2 * A3 - 1)); w = w / w.sum(0, keepdims=True)
            p = np.clip((w * A3).sum(0), EPS, 1 - EPS)
            ll = -np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p))
            if ll < bll: bll, bg = ll, g
        w_ct = w_rel[:, None] * np.exp(bg * np.abs(2 * T3 - 1)); w_ct = w_ct / w_ct.sum(0, keepdims=True)
        p_conf = (w_ct * T3).sum(0)
        pred_conf = (p_conf >= .5).astype(int)

        # 门控网络
        gate = GateNet(3, 8, SEED).fit(A3, ytr, epochs=400, lr=0.05)
        p_gate = (gate.predict_w(T3) * T3).sum(0)
        pred_gate = (p_gate >= .5).astype(int)

        # 多数投票 (用训练折学的阈值)
        p_vote = np.where((T3 >= thresh_vec[:, None]).astype(int).sum(0) >= 2, 1.0, 0.0)

        singles = {a: (T3[i] >= thresh_vec[i]).astype(int) for i, a in enumerate(ATLASES)}
        bi = int(np.argmax([auc_safe(ytr, A3[i]) for i in range(3)]))
        pred_best_atlas = singles[ATLASES[bi]]

        preds = {"A_eq": pred_eq, "B_fixed": pred_fixed,
                 "C_conf": pred_conf, "D_gate": pred_gate,
                 "M_vote": p_vote.astype(int),
                 "S_cc200": singles["cc200"], "S_aal": singles["aal"],
                 "S_dos160": singles["dos160"], "S_best_atlas": pred_best_atlas}
        bp = select_pool(ytr, tr_ids, te_ids, SEED)
        if bp is not None:
            p_pool = bp[5]
            thresh_pool = learn_thresh(ytr, bp[4])
            preds["S_pool"] = (p_pool >= thresh_pool).astype(int)
        for k, v in preds.items(): FACC[k].append(accuracy_score(yte, v) * 100)

        for j, s in enumerate(te_i):
            row = {"site": site, "subject": int(IDS[s]), "y_true": int(Y[s]),
                   "conflict": int(len({singles[a][j] for a in ATLASES}) > 1)}
            for k, v in preds.items(): row[k] = int(v[j])
            row["p_eq"] = float(p_eq[j]); row["p_fixed"] = float(p_fixed[j])
            row["p_conf"] = float(p_conf[j]); row["p_gate"] = float(p_gate[j])
            row["w_fx_cc200"] = float(w_fixed[0]); row["w_fx_aal"] = float(w_fixed[1])
            row["w_fx_dos160"] = float(w_fixed[2]); row["gamma"] = float(bg)
            pred_rows.append(row)

    row = {"site": site, "n": len(idx)}
    for k in FACC: row[k] = np.mean(FACC[k]) if FACC[k] else np.nan
    site_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(f"{OUTDIR}/predictions.csv", index=False)
    pd.DataFrame(site_rows).to_csv(f"{OUTDIR}/site_summary.csv", index=False)
    r = site_rows[-1]
    print(f"  {site:9s} n={len(idx):3d} | A={r['A_eq']:5.2f} B={r['B_fixed']:5.2f} "
          f"C={r['C_conf']:5.2f} D={r['D_gate']:5.2f} vote={r['M_vote']:5.2f} | "
          f"cc200={r['S_cc200']:5.2f} aal={r['S_aal']:5.2f} dos={r['S_dos160']:5.2f} | "
          f"pool={r.get('S_pool', float('nan')):5.2f} best={r['S_best_atlas']:5.2f} "
          f"[{time.time()-t_site:.0f}s]", flush=True)

sr = pd.DataFrame(site_rows)
w = sr["n"].values
print("\n" + "=" * 78); print("v15 严格嵌套 CV 汇总"); print("=" * 78)
for c in ["A_eq", "B_fixed", "C_conf", "D_gate", "M_vote", "S_cc200", "S_aal",
          "S_dos160", "S_best_atlas", "S_pool"]:
    if c in sr: print(f"  {c:12s} {np.nansum(sr[c].values*w)/w.sum():.2f}%")
print(f"\n用时 {time.time()-t0:.0f}s")
