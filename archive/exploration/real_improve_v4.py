"""
真正提升方案 v4: 严格对齐论文协议
- 站点集合: 与 reproduction_cv_full.csv 完全一致 (17站点)
- 评估: 每站点 10-fold CV, 每站点选最优 (配置, 模型), 按样本量加权
- 改进方案作为额外候选配置参与每站点选优

新增候选配置 (在论文的 36 配置 x 7 模型 之外):
  + 层次融合 (node+edge+graph)
  + 多图谱融合
  + 全融合
  + 软投票集成
  + 站点标准化版本
若改进配置在多数站点被选中 -> 说明有真实提升
"""
import warnings, ast, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs"
node = pd.read_csv(f"{OUT}/node_based_features.csv")
edge = pd.read_csv(f"{OUT}/edge_based_features.csv")
graph = pd.read_csv(f"{OUT}/graph_level_features.csv")
key = ["id", "ASD", "SITE_ID"]
merged = node.merge(edge.drop(columns=["ASD","SITE_ID"]), on="id") \
             .merge(graph.drop(columns=["ASD","SITE_ID"]), on="id")

# 统一站点命名 (LEUVEN_1/2 -> LEUVEN, UCLA_1/2 -> UCLA, UM_1/2 -> UM)
def norm_site(s):
    if s.endswith("_1") or s.endswith("_2"):
        return s.rsplit("_", 1)[0]
    return s
for df_ in [node, edge, graph]:
    df_["SITE_ID"] = df_["SITE_ID"].map(norm_site)
merged = node.merge(edge.drop(columns=["ASD","SITE_ID"]), on="id") \
             .merge(graph.drop(columns=["ASD","SITE_ID"]), on="id")

# 对齐站点集合: 用 reproduction_cv_full.csv 的站点
base = pd.read_csv(r"D:/ASD-GraphNet - mhx/reproduction_cv_full.csv")
SITES = sorted(base["SITE_ID"].unique())
merged = merged[merged["SITE_ID"].isin(SITES)].reset_index(drop=True)
print(f"站点数: {merged['SITE_ID'].nunique()} (应为17), 样本: {len(merged)}")
print(f"每站点样本: {merged['SITE_ID'].value_counts().to_dict()}")

def cell_values(v):
    s = str(v).strip()
    if s.startswith("["):
        try: return [float(x) for x in ast.literal_eval(s)]
        except: return [0.0]
    if s.startswith("{"):
        try:
            d = ast.literal_eval(s)
            return [float(list(d.values())[0])]
        except: return [0.0]
    try: return [float(s)]
    except: return [0.0]

def build_matrix(cols):
    rows = []
    for ri in range(len(merged)):
        r = []
        for c in cols:
            r.extend(cell_values(merged[c].iloc[ri]))
        rows.append(r)
    X = np.array(rows, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ---- 特征配置 ----
CC200_G = ["cc200_density","cc200_diameter","cc200_radius","cc200_transitivity",
           "cc200_assortativity_coefficient","cc200_average_clustering"]
AAL_G   = ["aal_density","aal_diameter","aal_radius","aal_transitivity",
           "aal_assortativity_coefficient","aal_average_clustering"]
DOS_G   = ["dos160_density","dos160_diameter","dos160_radius","dos160_transitivity",
           "dos160_assortativity_coefficient","dos160_average_clustering"]
# edge 层的标量 (取多阈值向量的聚合: 均值/最大)
EDGE_COLS = ["cc200_edge_betweenness_centrality","cc200_edge_density",
             "cc200_clustering_coefficient","cc200_number_of_connected_components",
             "cc200_average_shortest_path_length",
             "aal_edge_betweenness_centrality","aal_edge_density",
             "aal_clustering_coefficient","aal_number_of_connected_components",
             "aal_average_shortest_path_length",
             "dos160_edge_betweenness_centrality","dos160_edge_density",
             "dos160_clustering_coefficient","dos160_number_of_connected_components",
             "dos160_average_shortest_path_length"]

def build_edge_agg(cols):
    """edge 列: list -> [mean, max, std] 三个聚合量 (恒定 3 维)"""
    rows = []
    for ri in range(len(merged)):
        r = []
        for c in cols:
            vals = np.array(cell_values(merged[c].iloc[ri]), dtype=float).ravel()
            if vals.size == 0:
                r.extend([0.0, 0.0, 0.0])
            elif vals.size == 1:
                r.extend([float(vals[0])] * 3)
            else:
                r.extend([float(vals.mean()), float(vals.max()), float(vals.std())])
        rows.append(r)
    return np.nan_to_num(np.array(rows, dtype=float), nan=0.0)

FEATURE_SETS = {
    "node_cc200": build_matrix(["cc200_degree"]),
    "node_multi": build_matrix(["cc200_degree","aal_degree","dos160_degree"]),
    "node_edge":  build_matrix(["cc200_degree","aal_degree","dos160_degree"]),
    "graph_all":  build_matrix(CC200_G + AAL_G + DOS_G),
    "edge_agg":   build_edge_agg(EDGE_COLS),
}
# 融合
FEATURE_SETS["fuse_node_graph"] = np.concatenate([FEATURE_SETS["node_multi"], FEATURE_SETS["graph_all"]], axis=1)
FEATURE_SETS["fuse_all"] = np.concatenate(
    [FEATURE_SETS["node_multi"], FEATURE_SETS["graph_all"], FEATURE_SETS["edge_agg"]], axis=1)
# 站点标准化版本
def site_z(X):
    Xz = X.copy()
    for s in SITES:
        idx = np.where(merged["SITE_ID"].values == s)[0]
        Xz[idx] = (X[idx]-X[idx].mean(0))/(X[idx].std(0)+1e-8)
    return np.nan_to_num(Xz, nan=0.0)
FEATURE_SETS["fuse_all_z"] = site_z(FEATURE_SETS["fuse_all"])
FEATURE_SETS["node_multi_z"] = site_z(FEATURE_SETS["node_multi"])

for k, v in FEATURE_SETS.items():
    print(f"  {k}: {v.shape}")

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": lambda: SVC(probability=True, random_state=42),
    "XGB": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
}
def vote_fn():
    return VotingClassifier([
        ("lr", LogisticRegression(max_iter=3000, C=0.1)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
    ], voting="soft")

y_all = merged["ASD"].values
sites_all = merged["SITE_ID"].values

def cv_one(X, y, model_fn):
    k = min(10, min(np.bincount(y)))
    if k < 3: return None
    cv = StratifiedKFold(k, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = model_fn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

print("\n" + "="*80)
print("每站点 10-fold CV (论文协议)")
print("="*80)
rows = []
for fs_name, X in FEATURE_SETS.items():
    for m_name, m_fn in list(MODELS.items()) + [("VOTE", vote_fn)]:
        accs, ns = [], []
        for s in SITES:
            idx = np.where(sites_all == s)[0]
            a = cv_one(X[idx], y_all[idx], m_fn)
            if a is not None:
                accs.append(a); ns.append(len(idx))
        rows.append({"feature_set": fs_name, "model": m_name,
                     "SITE_ID": s, "acc": a, "n": len(idx)})
        _a = np.array(accs, dtype=float); _n = np.array(ns, dtype=float)
        print(f"  {fs_name:16s} {m_name:5s}: 加权={((_a*_n).sum()/_n.sum()):.2f}%", flush=True)

res = pd.DataFrame(rows)
res.to_csv(r"D:/ASD-GraphNet - mhx/real_improve_v4_results.csv", index=False)

print("\n" + "="*80)
print("每站点选最优 -> 加权平均 (论文口径)")
print("="*80)
best = res.loc[res.groupby("SITE_ID")["acc"].idxmax()]
w = (best["acc"]*best["n"]).sum()/best["n"].sum()
print(best[["SITE_ID","feature_set","model","acc","n"]].to_string(index=False))
print(f"\n>>> 含改进配置的最佳加权准确率: {w:.2f}%")
print(">>> 论文基线 (仅原 4284 配置): 75.63%")

# 仅原配置 (node_multi 近似 node_based+degree 多图谱, 其余 graph/edge 对应原类型)
print("\n仅用论文原有类型 (node/graph/edge 各自的单图谱最佳):")
orig = res[res["feature_set"].isin(["node_cc200","graph_all","edge_agg"])]
b2 = orig.loc[orig.groupby("SITE_ID")["acc"].idxmax()]
w2 = (b2["acc"]*b2["n"]).sum()/b2["n"].sum()
print(f">>> 论文口径(近似) 加权准确率: {w2:.2f}%")
