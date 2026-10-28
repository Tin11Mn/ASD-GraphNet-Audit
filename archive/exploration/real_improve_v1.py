"""
在论文协议下探索真正能提升的方案
论文协议: 每站点独立 10-fold CV -> 每站点选最优配置 -> 按样本量加权平均
基线: 75.63% (单层次单图谱)

本脚本对比:
  B0  论文基线: node_based/degree, 单 atlas, 单模型 (复现)
  B1  三层次特征融合 (node+edge+graph level, 同 atlas)
  B2  多图谱融合 (cc200+aal+dos160, 同层次)
  B3  全融合 (三层次 x 三图谱 = 所有标量特征)
  B4  B3 + 多模型软投票集成
  B5  B3 + 站点效应校正 (逐站点标准化)
"""
import warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
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

def feat_cols(df, prefixes=None):
    cs = [c for c in df.columns if c not in key]
    if prefixes:
        cs = [c for c in cs if any(c.startswith(p) for p in prefixes)]
    return cs

NODE_ALL = feat_cols(node)
EDGE_ALL = feat_cols(edge)
GRAPH_ALL = feat_cols(graph)
ALL_FEATS = NODE_ALL + EDGE_ALL + GRAPH_ALL
print(f"node feats={len(NODE_ALL)}, edge feats={len(EDGE_ALL)}, graph feats={len(GRAPH_ALL)}, total={len(ALL_FEATS)}")

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=2000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": lambda: SVC(probability=True, random_state=42),
}

def cv_site(df, feats, model_fn, n_splits=10):
    """单站点 10-fold CV, 返回平均准确率"""
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ASD"].values
    if len(np.unique(y)) < 2 or len(y) < n_splits:
        return None
    cv = StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y))), shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = model_fn(); m.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte))*100)
    return np.mean(accs)

def weighted_eval(feats, model_fn, tag, per_site_best=False):
    """对所有站点评估, 返回加权准确率"""
    accs, ns = [], []
    for site, g in merged.groupby("SITE_ID"):
        a = cv_site(g, feats, model_fn)
        if a is not None:
            accs.append(a); ns.append(len(g))
    accs = np.array(accs); ns = np.array(ns)
    w = (accs*ns).sum()/ns.sum()
    print(f"{tag:52s} 加权={w:.2f}%  宏平均={accs.mean():.2f}%  ({len(accs)}站点)")
    return w

print("\n" + "="*80)
print("论文协议: 每站点 10-fold CV, 按样本量加权")
print("="*80)

# B0: 论文基线 - node degree 单图谱单模型
print("\n[B0] 论文基线 (单层次/单图谱/单模型)")
w0 = weighted_eval(["cc200_degree"], MODELS["LR"], "  cc200_degree + LR")
weighted_eval(["cc200_degree"], MODELS["RF"], "  cc200_degree + RF")
weighted_eval(["cc200_degree"], MODELS["SVM"], "  cc200_degree + SVM")

# B1: 三层次融合 (同 atlas=cc200)
print("\n[B1] 三层次特征融合 (cc200: node+edge+graph)")
cc200_all = [c for c in ALL_FEATS if c.startswith("cc200")]
print(f"  cc200 全特征数: {len(cc200_all)}")
w1 = weighted_eval(cc200_all, MODELS["LR"], "  cc200 三层融合 + LR")
weighted_eval(cc200_all, MODELS["RF"], "  cc200 三层融合 + RF")

# B2: 多图谱融合 (node degree across 3 atlases)
print("\n[B2] 多图谱融合 (degree: cc200+aal+dos160)")
w2 = weighted_eval(["cc200_degree","aal_degree","dos160_degree"], MODELS["LR"], "  三图谱 degree + LR")
weighted_eval(["cc200_degree","aal_degree","dos160_degree"], MODELS["RF"], "  三图谱 degree + RF")

# B3: 全融合
print("\n[B3] 全融合 (三层次 x 三图谱)")
print(f"  总特征数: {len(ALL_FEATS)}")
w3 = weighted_eval(ALL_FEATS, MODELS["LR"], "  全融合 + LR")
weighted_eval(ALL_FEATS, MODELS["RF"], "  全融合 + RF")
weighted_eval(ALL_FEATS, MODELS["SVM"], "  全融合 + SVM")

# B4: 全融合 + 软投票
print("\n[B4] 全融合 + 软投票集成 (LR+RF+SVM)")
def vote_fn():
    return VotingClassifier([
        ("lr", LogisticRegression(max_iter=2000, C=0.1)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
    ], voting="soft")
w4 = weighted_eval(ALL_FEATS, vote_fn, "  全融合 + 软投票")

# B5: 全融合 + 逐站点标准化 (站点效应校正)
print("\n[B5] 全融合 + 站点内标准化 (站点效应校正)")
# 站点内 z-score 后再 CV (模拟 ComBat 效果)
merged_z = merged.copy()
for c in ALL_FEATS:
    merged_z[c] = merged.groupby("SITE_ID")[c].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
def weighted_eval_z(feats, model_fn, tag):
    accs, ns = [], []
    for site, g in merged_z.groupby("SITE_ID"):
        a = cv_site(g, feats, model_fn)
        if a is not None:
            accs.append(a); ns.append(len(g))
    accs = np.array(accs); ns = np.array(ns)
    w = (accs*ns).sum()/ns.sum()
    print(f"{tag:52s} 加权={w:.2f}%  宏平均={accs.mean():.2f}%")
    return w
w5 = weighted_eval_z(ALL_FEATS, MODELS["LR"], "  全融合(站点标准化) + LR")
weighted_eval_z(ALL_FEATS, vote_fn, "  全融合(站点标准化) + 软投票")

print("\n" + "="*80)
print("汇总")
print("="*80)
print(f"B0 论文基线 (cc200_degree+LR):        {w0:.2f}%")
print(f"B1 三层次融合 (cc200):                {w1:.2f}%")
print(f"B2 多图谱融合 (degree):               {w2:.2f}%")
print(f"B3 全融合:                            {w3:.2f}%")
print(f"B4 全融合+软投票:                     {w4:.2f}%")
print(f"B5 全融合+站点标准化:                 {w5:.2f}%")
print(f"\n(论文报告 75.25%, 本项目原始复现 75.63%)")
