"""
真正提升方案 v2: 正确解析三类特征 -> 层次融合 -> 论文协议评估
解析规则:
  node_based:  list -> np.array (ROI 维度)
  edge_based:  dict {threshold: value} -> 展开为多阈值向量 (阈值作为特征名)
  graph_level: 标量直接用; dict 展开
关键改进:
  1. 层次融合: node + edge + graph 同时使用
  2. 多图谱融合: cc200 + aal + dos160
  3. edge 多阈值信息 (论文只用了单一阈值)
  4. 软投票集成
"""
import warnings, ast, json, re, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
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
print(f"Merged: {merged.shape}, sites: {merged['SITE_ID'].nunique()}")

# 解析所有特征列 -> 构建特征矩阵
ATLASES = ["cc200", "aal", "dos160"]
LEVELS = {"node": [], "edge": [], "graph": []}
for c in merged.columns:
    if c in key: continue
    for a in ATLASES:
        if c.startswith(a + "_"):
            lvl = "node" if c in node.columns else ("edge" if c in edge.columns else "graph")
            LEVELS[lvl].append(c)
            break

def parse_cell(v):
    """单格 -> 一维数组"""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.array([float(v)])
    s = str(v).strip()
    if s.startswith("{"):
        try:
            d = ast.literal_eval(s)
            keys = sorted(d.keys(), key=lambda x: float(x))
            return np.array([float(d[k]) for k in keys])
        except: pass
    if s.startswith("["):
        try:
            return np.array([float(x) for x in ast.literal_eval(s)])
        except: pass
    try:
        return np.array([float(s)])
    except:
        return np.array([0.0])

def level_matrix(level):
    """构建某层次的 (n_samples, n_features) 矩阵"""
    cols = LEVELS[level]
    rows, names = [], []
    for ri in range(len(merged)):
        parts = []
        for c in cols:
            parts.append(parse_cell(merged[c].iloc[ri]))
        rows.append(np.concatenate(parts))
    X = np.vstack(rows)
    # 特征名
    for c in cols:
        arr = parse_cell(merged[c].iloc[0])
        names += [f"{c}_{i}" for i in range(len(arr))]
    return X, names

Xn, nn = level_matrix("node")
Xe, ne = level_matrix("edge")
Xg, ng = level_matrix("graph")
print(f"node: {Xn.shape}, edge: {Xe.shape}, graph: {Xg.shape}")

X_full = np.concatenate([Xn, Xe, Xg], axis=1)
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
y_all = merged["ASD"].values
sites_all = merged["SITE_ID"].values

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": lambda: SVC(probability=True, random_state=42),
}

def weighted_eval(X, model_fn, tag, sites=None, y=None):
    if y is None: y = y_all
    if sites is None: sites = sites_all
    accs, ns = [], []
    for s in sorted(set(sites)):
        idx = np.where(sites == s)[0]
        gX, gy = X[idx], y[idx]
        if len(np.unique(gy)) < 2 or len(gy) < 10:
            continue
        k = min(10, min(np.bincount(gy)))
        cv = StratifiedKFold(k, shuffle=True, random_state=42)
        a = []
        for tr, te in cv.split(gX, gy):
            sc = StandardScaler(); Xtr = sc.fit_transform(gX[tr]); Xte = sc.transform(gX[te])
            m = model_fn(); m.fit(Xtr, gy[tr])
            a.append(accuracy_score(gy[te], m.predict(Xte))*100)
        accs.append(np.mean(a)); ns.append(len(gy))
    accs = np.array(accs); ns = np.array(ns)
    w = (accs*ns).sum()/ns.sum()
    print(f"{tag:48s} 加权={w:.2f}%  宏平均={accs.mean():.2f}%  站点数={len(accs)}")
    return w, accs, ns

print("\n" + "="*78)
print("论文协议 (每站点 10-fold CV -> 按样本量加权)")
print("="*78)
print("\n[B0] 论文基线: node degree 单图谱 (cc200)")
w0,_,_ = weighted_eval(Xn[:, [i for i,n in enumerate(nn) if n.startswith("cc200_degree_")]], MODELS["LR"], "  cc200_degree + LR")

print("\n[B1] 层次融合 (node+edge+graph, 全部图谱)")
w1,_,_ = weighted_eval(X_full, MODELS["LR"], "  全层次 + LR")
weighted_eval(X_full, MODELS["RF"], "  全层次 + RF")
weighted_eval(X_full, MODELS["SVM"], "  全层次 + SVM")

print("\n[B2] 节点层多图谱 (node only, 3 atlas)")
w2,_,_ = weighted_eval(Xn, MODELS["LR"], "  node 层 (3图谱) + LR")
weighted_eval(Xn, MODELS["RF"], "  node 层 (3图谱) + RF")

print("\n[B3] edge+graph 层 (不含 node)")
Xe_g = np.nan_to_num(np.concatenate([Xe, Xg], axis=1), nan=0.0)
weighted_eval(Xe_g, MODELS["LR"], "  edge+graph 层 + LR")
weighted_eval(Xe_g, MODELS["RF"], "  edge+graph 层 + RF")

print("\n[B4] 全层次 + 软投票")
def vote():
    return VotingClassifier([("lr", LogisticRegression(max_iter=3000, C=0.1)),
                             ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                             ("svm", SVC(probability=True, random_state=42))], voting="soft")
w4,_,_ = weighted_eval(X_full, vote, "  全层次 + 软投票(LR+RF+SVM)")

print("\n[B5] 全层次 + 站点内标准化")
Xs = X_full.copy()
for s in sorted(set(sites_all)):
    idx = np.where(sites_all == s)[0]
    Xs[idx] = (X_full[idx] - X_full[idx].mean(0)) / (X_full[idx].std(0) + 1e-8)
w5,_,_ = weighted_eval(Xs, MODELS["LR"], "  全层次(站点标准化) + LR")
weighted_eval(Xs, MODELS["RF"], "  全层次(站点标准化) + RF")
weighted_eval(Xs, vote, "  全层次(站点标准化) + 软投票")

print("\n" + "="*78)
print(f"B0 论文基线:              {w0:.2f}%")
print(f"B1 全层次融合:             {w1:.2f}%")
print(f"B2 node 多图谱:            {w2:.2f}%")
print(f"B4 全层次+软投票:          {w4:.2f}%")
print(f"B5 全层次+站点标准化:      {w5:.2f}%")
print(f"\n(论文 75.25%, 原始复现 75.63%)")
