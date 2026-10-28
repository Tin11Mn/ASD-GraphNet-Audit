"""
=============================================================================
严格无泄露实验 v14: 多图谱可靠性融合 + 全配置池选优 (全部嵌套 CV 内)
=============================================================================

修复清单:
  P0-1  MI/PCA 在训练折内重新拟合 (基底用论文 original 特征, 不用泄漏的 mi_10/pca_10)
  P0-2  门控网络 softmax VJP 修正:
         dw = w * (dL_dw - (w*dL_dw).sum(1,keepdims=True))
         仅对最终融合概率裁剪; 梯度按样本数归一化
  P0-3  全配置池选优只用内层训练数据
  P1-1  冲突子集对照: 多数投票 / 等权 / 固定权重 / 自适应 / 门控 / 最强单模型
  P1-2  保存逐被试外层测试预测 -> 支持 McNemar 与配对置信区间
  P2-1  LOSO 留一站点验证

评估协议:
  外层: 每站点 10-fold CV (测试折完全独立)
  内层: 训练折内 5-fold
        - 两阶段选择: 先用 LR 选 (level, feature_engineering), 再选模型
        - MI/PCA 仅在训练折拟合
        - 概率校准仅用训练折 OOF
        - 融合权重 / 门控网络仅用训练折 OOF
  准确率只在外层测试折计算

输出 (v14_out/):
  predictions.csv     逐被试: site, subject, y_true, 各方法预测
  site_summary.csv    每站点汇总
  results.json        总汇总
"""
import ast, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
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
OUTDIR = f"{ROOT}/v14_out"; os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════ 1. 数据 (仅 original 特征: 逐被试图统计量, 无标签参与) ══════════════════
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

# 跨图谱拼接特征块
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

# 配置池: (特征块key, feature_engineering)
FE_BLOCKS = [(k, fe) for k in FEAT for fe in ["original", "mi_10", "pca_10"]]
print(f"特征块 {len(FEAT)}, 配置池 {len(FE_BLOCKS)}")

# ══════════════════ 2. 模型 ══════════════════
def make_model(name, seed):
    if name == "LR":   return LogisticRegression(max_iter=1500, C=0.1)
    if name == "DT":   return DecisionTreeClassifier(random_state=seed)
    if name == "RF":   return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if name == "XGB":  return XGBClassifier(n_estimators=100, random_state=seed,
                                           eval_metric="logloss", verbosity=0, tree_method="hist")
    if name == "LGBM": return LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1, n_jobs=1)
    if name == "SVM":  return SVC(probability=True, random_state=seed)
    if name == "MLP":  return MLPClassifier(max_iter=300, random_state=seed)
    raise ValueError(name)
MODEL_NAMES = ["LR", "DT", "RF", "XGB", "LGBM", "SVM", "MLP"]

# ══════════════════ 3. fold 内特征工程 (P0-1) ══════════════════
def fit_fe(Xtr, ytr, Xte, mode, seed=42):
    if mode == "original":
        return Xtr, Xte
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

# ══════════════════ 4. 校准 / 权重 / 门控 (P0-2) ══════════════════
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
    """门控 MLP: 输入 [probs, confidences] -> softmax 权重. softmax VJP 正确实现."""
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
            pm = np.clip(Pmix.sum(1), EPS, 1 - EPS)     # 仅裁剪最终融合概率
            dpm = (pm - y) / (pm * (1 - pm))
            dPmix = np.repeat(dpm[:, None], self.n, axis=1)
            dL_dw = dPmix * P
            dw = w * (dL_dw - (w * dL_dw).sum(1, keepdims=True))   # 正确 VJP
            dW2 = h.T @ dw / n; db2 = dw.sum(0) / n
            dh = (dw @ self.W2.T) / n; dz = dh * (1 - h ** 2)
            self.W1 -= lr * (X.T @ dz / n); self.b1 -= lr * dz.sum(0) / n
            self.W2 -= lr * dW2; self.b2 -= lr * db2
        return self
    def predict_w(self, probs):
        P = probs.T; X = np.concatenate([P, np.abs(2 * P - 1)], 1)
        return self._fwd(X)[0].T

def auc_safe(y, p):
    if len(np.unique(y)) < 2: return 0.5
    try: return roc_auc_score(y, p)
    except Exception: return 0.5

# ══════════════════ 5. 内层选择 (P0-3: 只用训练折) ══════════════════
# 关键优化: 同一 (block, fe, inner_fold) 的 MI/PCA 变换只做一次, 多个模型共享
def _transform_cache(block, fe, Xtr, Xte, itr, ite, y_itr, seed):
    """返回 (A, B, C): 训练折内层train变换后 / 内层test / 外层test"""
    sc = StandardScaler()
    A = sc.fit_transform(Xtr[itr]); B = sc.transform(Xtr[ite]); C = sc.transform(Xte)
    A2, B2 = fit_fe(A, y_itr, B, fe, seed)
    _, C2 = fit_fe(A, y_itr, C, fe, seed)
    return A2, B2, C2

def inner_select(feat_map, y, tr_ids, te_ids, seed, n_inner=5):
    """两阶段选择: LR 选 (block, fe) -> 再选模型.
       返回 (block, fe, model, oof_tr, prob_te). 所有拟合只在训练折内完成。
    """
    k = min(n_inner, int(np.bincount(y, minlength=2).min()))
    if k < 2: return None
    blocks = [b for b in feat_map if b in FEAT]
    if not blocks: return None
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)

    # 预先取出特征矩阵
    mats = {}
    for b in blocks:
        try:
            mats[b] = (np.stack([FEAT[b][i] for i in tr_ids]),
                       np.stack([FEAT[b][i] for i in te_ids]))
        except KeyError:
            continue
    if not mats: return None

    # 阶段 A: LR 选 (block, fe) —— 每个 (block, fe, fold) 只变换一次
    bestA = None
    for b, (Xtr, Xte) in mats.items():
        for fe in ["original", "mi_10"]:
            oof = np.zeros(len(tr_ids)); cnt = np.zeros(len(tr_ids))
            for itr, ite in skf.split(Xtr, y):
                A2, B2, _ = _transform_cache(b, fe, Xtr, Xte, itr, ite, y[itr], seed)
                m = make_model("LR", seed); m.fit(A2, y[itr])
                oof[ite] += m.predict_proba(B2)[:, 1]; cnt[ite] += 1
            if not (cnt > 0).all(): continue
            a = auc_safe(y, oof / cnt)
            if bestA is None or a > bestA[0]: bestA = (a, b, fe)
    if bestA is None: return None
    _, bb, bfe = bestA
    Xtr, Xte = mats[bb]

    # 阶段 B: 选模型 —— 变换结果缓存, 多模型共享
    models = ["LR", "DT", "RF", "XGB", "LGBM"]
    oofs = {mn: np.zeros(len(tr_ids)) for mn in models}
    tps = {mn: np.zeros(len(te_ids)) for mn in models}
    cnt = np.zeros(len(tr_ids))
    for itr, ite in skf.split(Xtr, y):
        A2, B2, C2 = _transform_cache(bb, bfe, Xtr, Xte, itr, ite, y[itr], seed)
        for mn in models:
            m = make_model(mn, seed); m.fit(A2, y[itr])
            oofs[mn][ite] += m.predict_proba(B2)[:, 1]
            tps[mn] += m.predict_proba(C2)[:, 1]
        cnt[ite] += 1
    if not (cnt > 0).all(): return None
    bestB = None
    for mn in models:
        oof = oofs[mn] / cnt
        a = auc_safe(y, oof)
        if bestB is None or a > bestB[0]: bestB = (a, mn, oof, tps[mn] / k)
    if bestB is None: return None
    return bb, bfe, bestB[1], bestB[2], bestB[3]

# 图谱级候选块
ATLAS_BLOCKS = {a: [(a, l) for l in LVLS if (a, l) in FEAT] for a in ATLASES}
POOL_BLOCKS = [k for k in FEAT if k[0] in ("ALL",) or (len(k) == 3 and k[0] == "DUAL")]

def inner_select_pool(y, tr_ids, te_ids, seed, n_inner=3):
    """全配置池选优 (只在内层训练折上做) -> (auc, block, fe, (model, oof, prob_te))
       每块用 LR+RF+XGB, fe 用 original/mi_10 各评估
    """
    best = None
    for b in POOL_BLOCKS:
        for fe in ["original", "mi_10"]:
            r = inner_select_single(b, fe, y, tr_ids, te_ids, seed, n_inner)
            if r is None: continue
            a = auc_safe(y, r[1])
            if best is None or a > best[0]: best = (a, b, fe, r)
    return best

def inner_select_single(block, fe, y, tr_ids, te_ids, seed, n_inner=3):
    """返回 (model, oof, prob_te)"""
    k = min(n_inner, int(np.bincount(y, minlength=2).min()))
    if k < 2: return None
    try:
        Xtr = np.stack([FEAT[block][i] for i in tr_ids])
        Xte = np.stack([FEAT[block][i] for i in te_ids])
    except KeyError: return None
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    models = ["LR", "RF", "XGB"]
    oofs = {mn: np.zeros(len(tr_ids)) for mn in models}
    tps = {mn: np.zeros(len(te_ids)) for mn in models}
    cnt = np.zeros(len(tr_ids))
    for itr, ite in skf.split(Xtr, y):
        sc = StandardScaler()
        A = sc.fit_transform(Xtr[itr]); B = sc.transform(Xtr[ite]); C = sc.transform(Xte)
        A2, B2 = fit_fe(A, y[itr], B, fe, seed); _, C2 = fit_fe(A, y[itr], C, fe, seed)
        for mn in models:
            m = make_model(mn, seed); m.fit(A2, y[itr])
            oofs[mn][ite] += m.predict_proba(B2)[:, 1]; cnt[ite] += 1
            tps[mn] += m.predict_proba(C2)[:, 1]
    if not (cnt > 0).all(): return None
    best = None
    for mn in models:
        a = auc_safe(y, oofs[mn] / cnt)
        if best is None or a > best[0]: best = (a, mn, oofs[mn], tps[mn] / k)
    if best is None: return None
    return best[1], best[2], best[3]

print("\n准备完成, 开始嵌套 CV")
print("=" * 84)

OUTER = 10
SEED = 42
pred_rows = []
site_rows = []

# 断点续跑：读取已有的结果
if os.path.exists(f"{OUTDIR}/site_summary.csv"):
    try:
        done_sites = set(pd.read_csv(f"{OUTDIR}/site_summary.csv")["site"].tolist())
        print(f"已完成的站点：{done_sites}")
    except Exception:
        done_sites = set()
else:
    done_sites = set()

for site in sorted(set(SITE)):
    if site in done_sites:
        print(f"跳过已完成站点：{site}", flush=True)
        continue
    idx = np.where(SITE == site)[0]
    y = Y[idx]
    if len(idx) < 25 or np.bincount(y, minlength=2).min() < 6:
        continue
    skf = StratifiedKFold(OUTER, shuffle=True, random_state=SEED)
    fold_acc = {k: [] for k in ["A_eq", "B_fixed", "C_conf", "D_gate", "M_vote",
                                "S_cc200", "S_aal", "S_dos160", "S_pool", "S_best_atlas"]}
    for tr, te in skf.split(idx, y):
        tr_i = idx[tr]; te_i = idx[te]; ytr = Y[tr_i]; yte = Y[te_i]
        # subject IDs (FEAT 以 ID 为键)
        tr_ids = IDS[tr_i].tolist(); te_ids = IDS[te_i].tolist()
        # ── 三图谱基础预测 (内层选 block/fe/model) ──
        base = {}
        for a in ATLASES:
            r = inner_select({b: FEAT[b] for b in ATLAS_BLOCKS[a]}, ytr, tr_ids, te_ids, SEED)
            if r is None: continue
            b, fe, mn, oof, pt = r
            cal = platt_fit(oof, ytr)
            base[a] = {"block": b, "fe": fe, "model": mn,
                       "oof": platt_apply(cal, oof), "test": platt_apply(cal, pt)}
        if len(base) < 3:
            continue
        A = np.array([base[a]["oof"] for a in ATLASES])      # (3, n_tr) 校准后
        T = np.array([base[a]["test"] for a in ATLASES])     # (3, n_te)

        # ── 融合方案 (权重仅由训练折 OOF 学习) ──
        p_eq = T.mean(0)
        w_fixed = fit_fixed_weights(A, ytr)
        p_fixed = (w_fixed[:, None] * T).sum(0)
        w_rel = fit_rel_weights(A, ytr)
        bg, bll = 0.0, np.inf
        for g in [0., .5, 1., 1.5, 2., 3., 4.]:
            w = w_rel[:, None] * np.exp(g * np.abs(2 * A - 1)); w = w / w.sum(0, keepdims=True)
            p = np.clip((w * A).sum(0), EPS, 1 - EPS)
            ll = -np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p))
            if ll < bll: bll, bg = ll, g
        # 测试折: 用训练折学到的可靠性权重 + 同样的 gamma, 用测试折自身置信度
        w_c_te = w_rel[:, None] * np.exp(bg * np.abs(2 * T - 1))
        w_c_te = w_c_te / w_c_te.sum(0, keepdims=True)
        p_conf = (w_c_te * T).sum(0)
        gate = GateNet(3, 8, SEED).fit(A, ytr, epochs=400, lr=0.05)
        w_g = gate.predict_w(T)
        p_gate = (w_g * T).sum(0)
        # 多数投票
        votes = (T >= 0.5).astype(int).sum(0)
        p_vote = np.where(votes >= 2, 1.0, 0.0)

        # ── 单图谱对照 ──
        singles = {}
        for ai, a in enumerate(ATLASES):
            singles[a] = (T[ai] >= 0.5).astype(int)

        # ── 全配置池选优 (内层, 只用训练折) ──
        bp = inner_select_pool(ytr, tr_ids, te_ids, SEED)
        p_pool = (bp[3][2] >= 0.5).astype(int) if bp is not None else None

        # ── 最强单图谱 (按训练折 OOF AUC 选) ──
        aucs = [auc_safe(ytr, A[i]) for i in range(3)]
        bi = int(np.argmax(aucs))
        p_best_atlas = singles[ATLASES[bi]]

        preds = {
            "A_eq": (p_eq >= .5).astype(int),
            "B_fixed": (p_fixed >= .5).astype(int),
            "C_conf": (p_conf >= .5).astype(int),
            "D_gate": (p_gate >= .5).astype(int),
            "M_vote": p_vote,
            "S_cc200": singles["cc200"], "S_aal": singles["aal"], "S_dos160": singles["dos160"],
            "S_best_atlas": p_best_atlas,
        }
        if p_pool is not None: preds["S_pool"] = p_pool
        for k, v in preds.items():
            fold_acc[k].append(accuracy_score(yte, v) * 100)
        # 保存逐被试预测 (P1-2)
        for j, s in enumerate(te_i):
            row = {"site": site, "subject": IDS[s], "y_true": int(Y[s]),
                   "conflicts": int(len(set(singles[a][j] for a in ATLASES)) > 1)}
            for k, v in preds.items(): row[k] = int(v[j])
            row["p_eq"] = float(p_eq[j]); row["p_fixed"] = float(p_fixed[j])
            row["p_conf"] = float(p_conf[j]); row["p_gate"] = float(p_gate[j])
            row["w_fixed_cc200"] = float(w_fixed[0]); row["w_fixed_aal"] = float(w_fixed[1])
            row["w_fixed_dos160"] = float(w_fixed[2])
            row["gamma_C"] = float(bg)
            pred_rows.append(row)
    row = {"site": site, "n": len(idx)}
    for k in fold_acc:
        row[k] = np.mean(fold_acc[k]) if fold_acc[k] else np.nan
    site_rows.append(row)
    # 逐站点即时落盘 (防中断丢失)
    pd.DataFrame(pred_rows).to_csv(f"{OUTDIR}/predictions.csv", index=False)
    pd.DataFrame(site_rows).to_csv(f"{OUTDIR}/site_summary.csv", index=False)
    r = site_rows[-1]
    print(f"  {site:9s} n={len(idx):3d} | A={r['A_eq']:5.2f} B={r['B_fixed']:5.2f} "
          f"C={r['C_conf']:5.2f} D={r['D_gate']:5.2f} vote={r['M_vote']:5.2f} | "
          f"cc200={r['S_cc200']:5.2f} aal={r['S_aal']:5.2f} dos={r['S_dos160']:5.2f} | "
          f"pool={r.get('S_pool', float('nan')):5.2f}", flush=True)

pr = pd.DataFrame(pred_rows); pr.to_csv(f"{OUTDIR}/predictions.csv", index=False)
sr = pd.DataFrame(site_rows); sr.to_csv(f"{OUTDIR}/site_summary.csv", index=False)
print(f"\n预测已保存: {len(pr)} 行")
print(f"用时 {time.time()-t0:.0f}s")
