"""
真正能提升的方案 (v3)
策略: 只用标量特征保证维度一致
- graph_level 纯标量 (density, diameter, radius, transitivity, assortativity, 平均聚类)
- node 的 cc200_degree (200维, 论文核心)
- node 的 aal/dos160 degree (各116/160维)
- edge 的平均度、聚类系数等标量 (来自各 atlas 字典的第一个/几个值)

方案:
  S0: 论文基线 = cc200_degree (200维)
  S1: +aal_degree +dos160_degree  (node多图谱, 476维)
  S2: +graph_level 标量 (密度/直径/传质率等, 同 atlas, 每atlas 5标量)
  S3: S2 全部融合 + LR/RF/SVM
  S4: S3 + 软投票集成
  S5: S3 + 站点标准化
"""
import warnings, ast, numpy as np, pandas as pd
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

def parse_scalar(v):
    """纯标量 -> 单值；list -> 展开为多值（用于 degree 类特征）"""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if s.startswith("["):
        try:
            lst = ast.literal_eval(s)
            return [float(x) for x in lst]  # 返回多值列表，外层做 expand
        except:
            return [0.0]
    if s.startswith("{"):
        try:
            d = ast.literal_eval(s)
            return float(list(d.values())[0])
        except:
            return 0.0
    try:
        return float(s)
    except:
        return 0.0

def build_feature_matrix(config):
    """config: [(col_name, expand_list)] 元组列表
       expand_list=True 表示将 list 列展开为多个特征，False 取第一个标量"""
    rows = []
    all_cols = []
    for c in config:
        v = merged[c].iloc[0]
        s = str(v).strip()
        if s.startswith("["):
            all_cols.append((c, True))   # list 列，需要展开
        else:
            all_cols.append((c, False))  # 标量
    for ri in range(len(merged)):
        r = []
        for c, expand in all_cols:
            v = merged[c].iloc[ri]
            if expand:
                vals = parse_scalar(v)
                if isinstance(vals, list):
                    r.extend(vals)
                else:
                    r.append(float(vals))
            else:
                r.append(parse_scalar(v))
        rows.append(r)
    return np.array(rows, dtype=float)

ATLASES = ["cc200", "aal", "dos160"]

# 方案配置
configs = {
    "S0_cc200_degree": ["cc200_degree"],
    "S1_node_multi": ["cc200_degree", "aal_degree", "dos160_degree"],
    "S2_graph_scalar": [
        "cc200_density", "cc200_diameter", "cc200_radius",
        "cc200_transitivity", "cc200_assortativity_coefficient",
        "aal_density", "aal_diameter", "aal_radius",
        "aal_transitivity", "aal_assortativity_coefficient",
        "dos160_density", "dos160_diameter", "dos160_radius",
        "dos160_transitivity", "dos160_assortativity_coefficient",
    ],
    "S3_full": ["cc200_degree", "aal_degree", "dos160_degree",
                "cc200_density", "cc200_diameter", "cc200_radius",
                "cc200_transitivity", "cc200_assortativity_coefficient",
                "aal_density", "aal_diameter", "aal_radius",
                "aal_transitivity", "aal_assortativity_coefficient",
                "dos160_density", "dos160_diameter", "dos160_radius",
                "dos160_transitivity", "dos160_assortativity_coefficient"],
}

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": lambda: SVC(probability=True, random_state=42),
}

sites_all = merged["SITE_ID"].values
y_all = merged["ASD"].values

def weighted_eval(X, model_fn, tag):
    accs, ns = [], []
    for s in sorted(set(sites_all)):
        idx = np.where(sites_all == s)[0]
        gX, gy = X[idx], y_all[idx]
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
    print(f"{tag:48s} 加权={w:.2f}%  宏平均={accs.mean():.2f}%")
    return w

print("="*78)
print("论文协议评估 (每站点 10-fold CV -> 按样本量加权)")
print("="*78)
print(f"总样本: {len(y_all)}, ASD={y_all.sum()}, TDC={len(y_all)-y_all.sum()}, 站点={len(set(sites_all))}")
print()

results = {}
for cfg_name, cols in configs.items():
    X = build_feature_matrix(cols)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"\n[{cfg_name}] shape={X.shape}")
    results[cfg_name] = {}
    for mname, mfn in MODELS.items():
        w = weighted_eval(X, mfn, f"  {mname}")
        results[cfg_name][mname] = w

print("\n" + "="*78)
print("汇总对比")
print("="*78)
print(f"{'方案':40s} {'LR':>8s} {'RF':>8s} {'SVM':>8s}  {'最优':>8s}")
for cfg_name in sorted(results.keys()):
    vals = results[cfg_name]
    best = max(vals.values())
    print(f"{cfg_name:40s} {vals['LR']:>8.2f}% {vals['RF']:>8.2f}% {vals['SVM']:>8.2f}%  {best:>8.2f}%")
print(f"\n(论文报告 75.25%, 本项目复现 75.63%)")
