"""
多图谱自适应融合 (Reliability-Aware Multi-Atlas Fusion)

严格无泄露设计:
  外层: 每站点 10-fold CV (论文协议)
  内层: 训练折内部再做 5-fold -> OOF 概率
    - 图谱权重、置信度模型、门控网络 全部只用训练折的 OOF 概率学习
    - 概率校准 (Platt) 也只在训练折 OOF 上拟合
  测试折: 三个图谱模型各自预测 -> 校准 -> 用训练折学到的权重融合

融合方案:
  A  等权平均 (基线)
  B  固定权重 (训练折 OOF 上优化 log-loss)
  C  置信度自适应 (逐被试权重, 超参 gamma 在 OOF 上选)
  D  轻量门控网络 (numpy MLP, 逐被试权重)

关键对照:
  S  最优单图谱 (相同内外层协议下每站点选最好的单个图谱)
"""
import ast, warnings, time, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(DATA)
df["features_value"] = df["features_value"].apply(ast.literal_eval)

ATLASES = ["cc200", "aal", "dos160"]
LEVELS = ["node_based", "edge_based", "graph_level"]
FEAT_ENG = ["mi_10", "pca_10"]

# ── 载入每个 (atlas, level, feat_eng) 的特征矩阵 ──
X_by = {}
for atlas in ATLASES:
    for lvl in LEVELS:
        for fe in FEAT_ENG:
            sub = df[(df["atlas"] == atlas) & (df["graph_feature_type"] == lvl) &
                     (df["feature_engineering"] == fe)]
            if len(sub) == 0:
                continue
            d = {int(r["id"]): np.array(r["features_value"], dtype=float) for _, r in sub.iterrows()}
            dims = {len(v) for v in d.values()}
            if len(dims) != 1:
                continue
            X_by[(atlas, lvl, fe)] = d

meta = df[["id", "ASD", "SITE_ID"]].drop_duplicates(subset=["id"]).copy()
meta["id"] = meta["id"].astype(int)
meta = meta.sort_values("id").reset_index(drop=True)
IDS = meta["id"].tolist()
Y = meta["ASD"].values
SITE = meta["SITE_ID"].values
print(f"被试 {len(IDS)}, ASD={Y.sum()}, 站点 {len(set(SITE))}")
for k, v in X_by.items():
    print(f"  {k}: {len(v)} 被试, {len(next(iter(v.values())))} 维")

# ── 模型候选 ──
def make_model(name, seed):
    if name == "LR":
        return LogisticRegression(max_iter=1000, C=0.1)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss",
                             verbosity=0, tree_method="hist")
    raise ValueError(name)
MODEL_NAMES = ["LR", "RF", "XGB"]

# ── 校准 (Platt scaling) ──
EPS = 1e-6
def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)

def platt_fit(oof_p, y):
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(to_logit(oof_p), y)
    return lr

def platt_apply(cal, p):
    if cal is None:
        return p
    return cal.predict_proba(to_logit(p))[:, 1]

# ── 权重优化 (方案 B) ──
def softmax_w(params):
    e = np.exp(params - np.max(params))
    return e / e.sum()

def fit_fixed_weights(oof_probs, y):
    """在训练折 OOF 上优化 log-loss, 得到固定权重"""
    def obj(params):
        w = softmax_w(params)
        p = np.clip((w[:, None] * oof_probs).sum(axis=0), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    best, best_v = np.zeros(len(oof_probs)), np.inf
    for init in [np.zeros(len(oof_probs)), np.array([0.5, -0.5, 0.0]), np.array([-0.5, 0.5, 0.0])]:
        if len(init) != len(oof_probs):
            continue
        r = minimize(obj, init, method="Nelder-Mead",
                     options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
        if r.fun < best_v:
            best_v, best = r.fun, softmax_w(r.x)
    return best, best_v

def fit_reliability_weights(oof_probs, y):
    """可靠性权重: 每个图谱的 OOF 负对数似然 -> softmax"""
    nll = []
    for a in range(oof_probs.shape[0]):
        p = np.clip(oof_probs[a], EPS, 1 - EPS)
        nll.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    nll = np.array(nll)
    logits = -nll / max(nll.std(), 1e-6)
    return softmax_w(logits)

# ── 门控网络 (方案 D) ──
class GateNet:
    """6 -> 8 -> 3 门控 MLP, 输出逐被试权重"""
    def __init__(self, n_atlas, hidden=8, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, (n_atlas * 2, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, (hidden, n_atlas))
        self.b2 = np.zeros(n_atlas)
        self.n = n_atlas

    def _forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        g = h @ self.W2 + self.b2
        g = g - g.max(axis=1, keepdims=True)
        e = np.exp(g)
        w = e / e.sum(axis=1, keepdims=True)
        return w, h

    def fit(self, probs, y, base_w=None, epochs=300, lr=0.05, seed=42):
        """probs: (n_atlas, n) 训练折 OOF 概率"""
        P = probs.T                                   # (n, n_atlas)
        conf = np.abs(2 * P - 1)                      # (n, n_atlas)
        X = np.concatenate([P, conf], axis=1)
        n = len(y)
        for ep in range(epochs):
            w, h = self._forward(X)
            Pmix = np.clip(w * P, EPS, 1 - EPS)
            pmix = Pmix.sum(axis=1)
            dL_dpmix = (pmix - y) / (pmix * (1 - pmix))
            dL_dw = dL_dpmix[:, None] * P            # (n, n_atlas)
            # softmax jacobian
            dw = dL_dw * (w * (1 - w) - w * w)
            dW2 = h.T @ dw
            db2 = dw.sum(axis=0)
            dh = dw @ self.W2.T
            dz = dh * (1 - h ** 2)
            dW1 = X.T @ dz
            db1 = dz.sum(axis=0)
            self.W1 -= lr * dW1; self.b1 -= lr * db1
            self.W2 -= lr * dW2; self.b2 -= lr * db2
        return self

    def predict_weights(self, probs):
        P = probs.T
        conf = np.abs(2 * P - 1)
        X = np.concatenate([P, conf], axis=1)
        w, _ = self._forward(X)
        return w.T                                    # (n_atlas, n)

# ── 内层: 训练折内部 OOF + 测试折预测 ──
def inner_oof_and_test(Xtr, ytr, Xte, model, n_inner=5, seed=42):
    """返回 (oof_probs_train, test_probs_avg)"""
    n = len(ytr)
    oof = np.zeros(n)
    cnt = np.zeros(n)
    test_acc = np.zeros(len(Xte))
    k = min(n_inner, int(np.bincount(ytr, minlength=2).min()))
    if k < 2:
        return None, None
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for itr, ite in skf.split(Xtr, ytr):
        sc = StandardScaler()
        A = sc.fit_transform(Xtr[itr]); B = sc.transform(Xtr[ite]); C = sc.transform(Xte)
        m = model
        m.fit(A, ytr[itr])
        oof[ite] += m.predict_proba(B)[:, 1]
        cnt[ite] += 1
        test_acc += m.predict_proba(C)[:, 1]
    nz = cnt > 0
    oof[nz] /= cnt[nz]
    return oof, test_acc / k

# ── 主实验 ──
print("\n" + "=" * 84)
print("多图谱自适应融合 (论文协议: 每站点 10-fold CV -> 按样本量加权)")
print("=" * 84)

OUTER = 10
INNER = 5
SEEDS = [42, 123]

site_rows = []   # 每站点 × seed × (方案 -> acc)
weights_log = []
PARTIAL = r"D:/ASD-GraphNet - mhx/fusion_partial.csv"

done_sites = set()
if os.path.exists(PARTIAL):
    prev = pd.read_csv(PARTIAL)
    site_rows = prev.to_dict("records")
    done_sites = set(prev["site"].unique())
    print(f"续跑: 已完成 {sorted(done_sites)}")

for site in sorted(set(SITE)):
    if site in done_sites:
        continue
    idx = np.where(SITE == site)[0]
    y_site = Y[idx]
    if len(idx) < 25 or np.bincount(y_site, minlength=2).min() < 6:
        continue
    # 候选配置: (atlas, level, feat_eng) × model
    configs = [(k, m) for k in X_by.keys() for m in MODEL_NAMES]
    per_seed = {s: {sch: [] for sch in ["A", "B", "C", "D", "S_cc200", "S_aal", "S_dos160"]}
                for s in SEEDS}
    wsum = {s: np.zeros(3) for s in SEEDS}
    wcnt = {s: 0 for s in SEEDS}

    for seed in SEEDS:
        skf = StratifiedKFold(OUTER, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(idx, y_site)):
            tr_i, te_i = idx[tr], idx[te]
            ytr, yte = Y[tr_i], Y[te_i]

            # ── 对每个图谱/配置/模型计算 OOF + test 概率 ──
            # 结果缓存: (atlas) -> {model_key: (oof, test)}
            per_atlas = {a: {} for a in ATLASES}
            for (cfg_key, mname) in configs:
                atlas = cfg_key[0]
                feats = X_by[cfg_key]
                try:
                    A = np.stack([feats[IDS[i]] for i in tr_i])
                    B = np.stack([feats[IDS[i]] for i in te_i])
                except KeyError:
                    continue
                oof, tst = inner_oof_and_test(A, ytr, B, make_model(mname, seed),
                                              n_inner=INNER, seed=seed)
                if oof is None:
                    continue
                per_atlas[atlas][(cfg_key, mname)] = (oof, tst)

            # 每个图谱选最优配置 (用训练折 OOF 的 AUC)
            best_per_atlas = {}
            for atlas in ATLASES:
                best_auc, best_key = -1, None
                for key, (oof, tst) in per_atlas[atlas].items():
                    try:
                        auc = roc_auc_score(ytr, oof)
                    except Exception:
                        continue
                    if auc > best_auc:
                        best_auc, best_key = auc, key
                if best_key is not None:
                    best_per_atlas[atlas] = per_atlas[atlas][best_key] + (best_auc, best_key)

            if len(best_per_atlas) < 3:
                continue

            # ── 校准 ──
            cal_oof = np.zeros((3, len(tr_i)))
            cal_test = np.zeros((3, len(te_i)))
            for ai, atlas in enumerate(ATLASES):
                oof, tst, auc, key = best_per_atlas[atlas]
                c = platt_fit(oof, ytr)
                cal_oof[ai] = platt_apply(c, oof)
                cal_test[ai] = platt_apply(c, tst)

            # ── 方案 A: 等权 ──
            pA = cal_test.mean(axis=0)
            per_seed[seed]["A"].append(accuracy_score(yte, (pA >= 0.5).astype(int)) * 100)

            # ── 方案 B: 固定权重 (训练折 OOF 优化) ──
            wB, _ = fit_fixed_weights(cal_oof, ytr)
            pB = (wB[:, None] * cal_test).sum(axis=0)
            per_seed[seed]["B"].append(accuracy_score(yte, (pB >= 0.5).astype(int)) * 100)
            wsum[seed] += wB; wcnt[seed] += 1

            # ── 方案 C: 置信度自适应 ──
            # 基础可靠性权重 + 逐被试置信度指数加权, gamma 在 OOF 上选
            w_rel = fit_reliability_weights(cal_oof, ytr)
            best_gamma, best_ll = 0.0, np.inf
            for gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
                conf_tr = np.abs(2 * cal_oof - 1)
                wtr = w_rel[:, None] * np.exp(gamma * conf_tr)
                wtr = wtr / wtr.sum(axis=0, keepdims=True)
                ptr = np.clip((wtr * cal_oof).sum(axis=0), EPS, 1 - EPS)
                ll = -np.mean(ytr * np.log(ptr) + (1 - ytr) * np.log(1 - ptr))
                if ll < best_ll:
                    best_ll, best_gamma = ll, gamma
            conf_te = np.abs(2 * cal_test - 1)
            wte = w_rel[:, None] * np.exp(best_gamma * conf_te)
            wte = wte / wte.sum(axis=0, keepdims=True)
            pC = (wte * cal_test).sum(axis=0)
            per_seed[seed]["C"].append(accuracy_score(yte, (pC >= 0.5).astype(int)) * 100)

            # ── 方案 D: 门控网络 ──
            gate = GateNet(3, hidden=8, seed=seed).fit(cal_oof, ytr, epochs=300, lr=0.05)
            wd = gate.predict_weights(cal_test)
            pD = (wd * cal_test).sum(axis=0)
            per_seed[seed]["D"].append(accuracy_score(yte, (pD >= 0.5).astype(int)) * 100)

            # ── 对照: 最优单图谱 ──
            for ai, atlas in enumerate(ATLASES):
                pS = cal_test[ai]
                per_seed[seed][f"S_{atlas}"].append(
                    accuracy_score(yte, (pS >= 0.5).astype(int)) * 100)

    for seed in SEEDS:
        row = {"site": site, "seed": seed, "n": len(idx)}
        for sch, vals in per_seed[seed].items():
            row[sch] = np.mean(vals) if vals else np.nan
        row["w_cc200"], row["w_aal"], row["w_dos160"] = (
            wsum[seed] / max(wcnt[seed], 1) if wcnt[seed] else [np.nan]*3)
        site_rows.append(row)
    pd.DataFrame(site_rows).to_csv(PARTIAL, index=False)
    r = site_rows[-len(SEEDS)]
    print(f"  {site:9s} n={len(idx):3d} | A={r['A']:5.2f} B={r['B']:5.2f} C={r['C']:5.2f} "
          f"D={r['D']:5.2f} | 单图谱 cc200={r['S_cc200']:5.2f} aal={r['S_aal']:5.2f} "
          f"dos160={r['S_dos160']:5.2f}", flush=True)

res = pd.DataFrame(site_rows)
res.to_csv(r"D:/ASD-GraphNet - mhx/fusion_results.csv", index=False)

# ── 汇总 ──
print("\n" + "=" * 84)
print("汇总: 3 种子加权平均 (每站点先在种子上平均, 再按样本量加权)")
print("=" * 84)
site_avg = res.groupby(["site", "n"]).mean(numeric_only=True).reset_index()
wts = site_avg["n"].values
schemes = ["A", "B", "C", "D", "S_cc200", "S_aal", "S_dos160"]
summary = {}
for sch in schemes:
    w = np.nansum(site_avg[sch].values * wts) / wts.sum()
    summary[sch] = w
    print(f"  {sch:12s} {w:.2f}%")
best_single = max(summary["S_cc200"], summary["S_aal"], summary["S_dos160"])
best_fuse = max(summary["A"], summary["B"], summary["C"], summary["D"])
print(f"\n  最优单图谱 (对照): {best_single:.2f}%")
print(f"  最优融合方案:       {best_fuse:.2f}%")
print(f"  融合增益:           {best_fuse - best_single:+.2f}%")
print(f"  (论文基线 75.63% | v9 改进 77.59%)")

# 权重分析
print("\n  学习的平均权重 (方案 B, 固定权重):")
for a, c in zip(ATLASES, ["w_cc200", "w_aal", "w_dos160"]):
    vals = site_avg[c].dropna()
    print(f"    {a:8s} 均值={vals.mean():.3f}  范围=[{vals.min():.3f}, {vals.max():.3f}]")

# 显著性: 配对检验 (站点层面)
from scipy.stats import wilcoxon
print("\n  站点层面配对检验 (Wilcoxon signed-rank):")
for sch in ["A", "B", "C", "D"]:
    d = (site_avg[sch] - best_single).dropna()
    if len(d) > 5 and d.abs().sum() > 0:
        stat, p = wilcoxon(d)
        print(f"    {sch} vs 最优单图谱: Δ={d.mean():+.2f}%, p={p:.4f}")

print(f"\n用时 {time.time()-t0:.0f}s")
