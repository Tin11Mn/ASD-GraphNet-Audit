"""
ASD-GraphNet 真正提升方案 v5
核心策略: 在论文 mi_10 预处理基础上, 增加跨图谱/跨层次融合候选 + 更多模型 + 模型集成

论文协议 (严格):
  - 用 filtered_subjects_df.csv 的 features_value (已是论文预处理好的)
  - 每站点 10-fold CV (StratifiedKFold)
  - 每站点选最优配置 -> 按站点样本量加权平均
  - 论文权重: site_sizes = SITE_ID.value_counts() / (3 types * 3 fe * 4 atlases)
"""
import warnings, ast, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
np.random.seed(42)

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(DATA)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
SITES = sorted(df["SITE_ID"].unique())
n_configs = df["graph_feature_type"].nunique() * df["feature_engineering"].nunique() * df["atlas"].nunique()
print(f"数据: {len(df)} rows, {df['SITE_ID'].nunique()} sites, n_configs={n_configs}")
print(f"feature_engineering: {df['feature_engineering'].value_counts().to_dict()}")
print(f"graph_feature_type: {df['graph_feature_type'].value_counts().to_dict()}")
print(f"atlas: {df['atlas'].value_counts().to_dict()}")

sites_all = df["SITE_ID"].values
y_all = df["ASD"].values

# ---- 候选配置 ----
BASE_CONFIGS = []
for atlas in df["atlas"].unique():
    for gt in ["node_based", "edge_based", "graph_level"]:
        for fe in ["original", "mi_10", "pca_10"]:
            BASE_CONFIGS.append((atlas, gt, fe))

# 新增融合配置: (name, 列表 of (atlas, gt, fe))
FUSION_CONFIGS = [
    # 多图谱融合 (保持同一 gt+fe 跨所有 atlas)
    ("multi_atlas_node_mi",   [("cc200","node_based","mi_10"), ("aal","node_based","mi_10"), ("dos160","node_based","mi_10")]),
    ("multi_atlas_node_orig", [("cc200","node_based","original"), ("aal","node_based","original"), ("dos160","node_based","original")]),
    # 多图谱 + 多层次融合 (node + edge + graph 同 atlas)
    ("multi_atlas_multi_level_mi", 
     [("cc200","node_based","mi_10"), ("cc200","edge_based","mi_10"), ("cc200","graph_level","mi_10")]),
    ("multi_atlas_multi_level_orig",
     [("cc200","node_based","original"), ("cc200","edge_based","original"), ("cc200","graph_level","original")]),
    # 全融合
    ("full_fusion_mi",
     [(a,g,fe) for a in df["atlas"].unique() for g in df["graph_feature_type"].unique() for fe in ["mi_10"]]),
    ("full_fusion_orig",
     [(a,g,fe) for a in df["atlas"].unique() for g in df["graph_feature_type"].unique() for fe in ["original"]]),
]

print(f"\nBASE_CONFIGS: {len(BASE_CONFIGS)}, FUSION_CONFIGS: {len(FUSION_CONFIGS)}")

# ---- 模型 ----
MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "DT": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LGBM": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "SVM": lambda: SVC(probability=True, random_state=42),
}
def vote_fn():
    return VotingClassifier([
        ("lr", LogisticRegression(max_iter=3000, C=0.1)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)),
        ("svm", SVC(probability=True, random_state=42)),
    ], voting="soft")
MODELS["VOTE"] = vote_fn

def get_features(atlas, gt, fe, site):
    """从 filtered_subjects_df 中取特定站点的特征"""
    sub = df[(df["SITE_ID"]==site) & (df["atlas"]==atlas) & 
             (df["graph_feature_type"]==gt) & (df["feature_engineering"]==fe)]
    if sub.empty:
        return None, None
    X = np.stack(sub["features_value"].values).astype(float)
    y = sub["ASD"].values
    return np.nan_to_num(X), y

def cv_score(X, y, mfn):
    k = min(10, min(np.bincount(y)))
    if k < 3:
        return None
    cv = StratifiedKFold(k, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

def eval_config(conf_name, conf_keys, site):
    """对单站点某配置, 用多个模型取最大"""
    # 合并特征 (按被试 id 对齐)
    id_to_X = {}
    id_to_y = {}
    valid = True
    for atlas, gt, fe in conf_keys:
        X, y = get_features(atlas, gt, fe, site)
        if X is None:
            valid = False; break
        # 按 id 匹配 (每个站点内 id 应唯一)
        id_x = dict(zip(X.shape[0] and range(len(X)), X))  # 用位置索引
        # 简化: 直接 concat 同一站点的行 (假设每个被试一行, 且顺序一致)
        pass
    # 实际: 用 subject id 对齐
    merged_rows = []
    ids = []; ys = []
    for atlas, gt, fe in conf_keys:
        sub = df[(df["SITE_ID"]==site) & (df["atlas"]==atlas) & 
                 (df["graph_feature_type"]==gt) & (df["feature_engineering"]==fe)]
        if sub.empty:
            valid = False; break
        for _, row in sub.iterrows():
            sid = row["id"]; x = np.array(row["features_value"], dtype=float)
            merged_rows.append(x); ids.append(sid); ys.append(row["ASD"])
    if not valid: return None
    ids_unique = pd.Series(ids).drop_duplicates()
    y_uni = np.array([ys[ids.index(i)] for i in ids_unique])
    X_uni = np.stack([merged_rows[ids.index(i)] for i in ids_unique])
    X_uni = np.nan_to_num(X_uni)
    return X_uni, y_uni

def eval_fusion(fusion_key, site):
    _, keys = fusion_key
    Xs, ys, ids_list = [], [], []
    for atlas, gt, fe in keys:
        sub = df[(df["SITE_ID"]==site) & (df["atlas"]==atlas) & 
                 (df["graph_feature_type"]==gt) & (df["feature_engineering"]==fe)]
        if sub.empty: continue
        for _, row in sub.iterrows():
            Xs.append(np.array(row["features_value"], dtype=float))
            ids_list.append(row["id"]); ys.append(row["ASD"])
    if not Xs: return None, None
    X = np.stack(Xs)
    ids_s = np.array(ids_list); y = np.array(ys)
    # 去重 (同一被试可能多行? 不, 每个 atlas/gt/fe 各一行)
    # 每个被试只保留一个 X 向量 (但需按 id 去重, 因为融合 config 含同一被试多个 atlas)
    # 正确: 按 id 分组, 将每个被试的所有特征拼接
    df_sub = pd.DataFrame({"id": ids_s, "ASD": y, "features": X.tolist()})
    grouped = df_sub.groupby("id").agg({"ASD":"first", "features":"sum"}).reset_index()
    # features sum -> concat (应该 concat 不是 sum)
    X_concat = np.stack(grouped["features"].apply(np.array).values)
    y_g = grouped["ASD"].values
    return np.nan_to_num(X_concat), y_g

def eval_concat(fusion_key, site):
    """concat 而不是 sum"""
    ids_dict = {}; feats_dict = {}
    for atlas, gt, fe in fusion_key:
        sub = df[(df["SITE_ID"]==site) & (df["atlas"]==atlas) & 
                 (df["graph_feature_type"]==gt) & (df["feature_engineering"]==fe)]
        if sub.empty: continue
        for _, row in sub.iterrows():
            sid = row["id"]; x = np.array(row["features_value"], dtype=float)
            ids_dict.setdefault(sid, []).append(x)
    if not ids_dict: return None, None
    xs, ys = [], []
    for sid in sorted(ids_dict.keys()):
        combined = np.concatenate(ids_dict[sid])
        xs.append(combined)
        ys.append(next(r["ASD"] for _, r in df[(df["SITE_ID"]==site)&(df["id"]==sid)].iterrows()))
    return np.nan_to_num(np.array(xs)), np.array(ys)

print("\n" + "="*78)
print("论文协议评估: 每站点 10-fold CV, 每站点取最优配置->加权")
print("="*78)

results = []  # (config_name, model_name, site, acc, n)

# 先跑基线所有配置
print("\n[Phase 1] 基线 36 配置 x 7 模型 (17站点)")
for _, r in df.iterrows():
    sit = r["SITE_ID"]; atlas = r["atlas"]; gt = r["graph_feature_type"]
    fe = r["feature_engineering"]; sid = r["id"]
    # 用已有的 reproduction_cv_full.csv 跳过基线重跑
print("跳过低效的逐行遍历, 直接用已有的 base 结果")
base = pd.read_csv(r"D:/ASD-GraphNet - mhx/reproduction_cv_full.csv")
base["config_key"] = base.apply(lambda r: (r["atlas"], r["graph_feature_type"], r["feature_engineering"]), axis=1)

for (conf_name, conf_keys) in FUSION_CONFIGS:
    print(f"\n[FUSION: {conf_name}] 候选配置: {len(conf_keys)}")
    for mname, mfn in MODELS.items():
        accs, ns = [], []
        for site in SITES:
            X, y = eval_concat(conf_keys, site)
            if X is None or len(y) < 20 or min(np.bincount(y)) < 5:
                continue
            a = cv_score(X, y, mfn)
            if a is not None:
                accs.append(a); ns.append(len(y))
        if not accs: continue
        _a = np.array(accs, dtype=float); _n = np.array(ns, dtype=float)
        w = (_a*_n).sum()/_n.sum()
        print(f"  {conf_name:30s} {mname:5s}: 加权={w:.2f}%")
        for si, a, n_ in zip(SITES, accs, ns):
            results.append({"config": conf_name, "model": mname, "SITE_ID": si, "acc": a, "n": n_})

# 保存
import pickle
pickle.dump(results, open(r"D:/ASD-GraphNet - mhx/improve_v5_results.pkl", "wb"))
print(f"\n共保存 {len(results)} 条结果")

# 汇总
print("\n" + "="*78)
print("汇总: 每站点选最优 (含基线+融合)")
print("="*78)
# 需要把基线也加入比较
base_best = base.loc[base.groupby("SITE_ID")["cv_accuracy_mean"].idxmax()]
base_w = (base_best["cv_accuracy_mean"] * base_best["n_samples"]).sum() / base_best["n_samples"].sum()
print(f"论文基线 (仅原始4284): 加权={base_w:.2f}%")

# 融合结果
if results:
    res_df = pd.DataFrame(results)
    fusion_best = res_df.loc[res_df.groupby("SITE_ID")["acc"].idxmax()]
    w_f = (fusion_best["acc"]*fusion_best["n"]).sum()/fusion_best["n"].sum()
    print(f"融合改进:              加权={w_f:.2f}%")
    print(f"提升:                  {w_f - base_w:+.2f}%")
    print(fusion_best[["SITE_ID","config","model","acc","n"]].to_string(index=False))
