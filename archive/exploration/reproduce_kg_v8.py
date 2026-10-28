"""
ASD-GraphNet 知识图谱增强 V8
知识图谱设计:
  节点: 200 个 CC200 ROI
  边:
    (1) 空间邻近边: ROI 质心距离 < 阈值 (解剖相邻)
    (2) 层级边: ROI -> 所属脑区 (Frontal/Parietal/.../Cerebellum/Subcortical)
    (3) 跨图谱别名边: AAL/Harvard-Oxford 指向同一解剖实体
  KG 衍生特征:
    (a) 区域层级聚合: 每个脑区的平均 degree (7 维)
    (b) KG 中心性加权: degree * KG_PageRank (200 维)
    (c) 区域密度比: 该 ROI degree / 所属区域平均 degree (200 维)
    (d) 跨图谱共识度: 多命名体系一致指向同一实体的 ROI 权重 (200 维)
"""
import ast, json, warnings, re
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

ROI_CSV = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/CC200_ROI_labels.csv"
FEAT_CSV = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"

# ---------- 构建知识图谱 ----------
roi_df = pd.read_csv(ROI_CSV)
roi_df.columns = [c.strip() for c in roi_df.columns]

def parse_json_map(x):
    """解析形如 [\"Name\": 0.8][\"Name2\": 0.2] 的字符串"""
    if not isinstance(x, str):
        return []
    out = []
    for m in re.finditer(r'\[\s*"([^"]+)"\s*:\s*([\d.]+)\s*\]', x):
        out.append((m.group(1), float(m.group(2))))
    return out

def region_of(name):
    n = name.lower()
    if 'frontal' in n: return 'Frontal'
    if 'parietal' in n: return 'Parietal'
    if 'temporal' in n: return 'Temporal'
    if 'occipital' in n: return 'Occipital'
    if 'cerebellum' in n or 'cerebel' in n: return 'Cerebellum'
    if 'thalamus' in n or 'caudate' in n or 'putamen' in n or 'accumbens' in n or 'pallidum' in n: return 'Subcortical'
    if 'insula' in n: return 'Insula'
    if 'cingulate' in n: return 'Limbic'
    return 'Other'

rois = []
for i, row in roi_df.iterrows():
    coords = re.findall(r'-?\d+\.?\d*', str(row['center of mass']))
    x, y, z = (float(v) for v in coords[:3])
    ho = parse_json_map(row.get('Harvard-Oxford', ''))
    aal = parse_json_map(row.get('AAL', ''))
    ez = parse_json_map(row.get('Eickhoff-Zilles', ''))
    tt = parse_json_map(row.get('Talairach-Tournoux', ''))
    primary = ho[0][0] if ho else (aal[0][0] if aal else f'ROI_{i+1}')
    rois.append({
        'id': i, 'x': x, 'y': y, 'z': z, 'volume': float(row['volume']),
        'name': primary, 'region': region_of(primary),
        'ho': ho, 'aal': aal, 'ez': ez, 'tt': tt,
    })

N_ROI = len(rois)
print(f"ROIs: {N_ROI}, regions: {sorted(set(r['region'] for r in rois))}")

# 构建 KG
KG = nx.Graph()
for r in rois:
    KG.add_node(r['id'], label=r['name'], region=r['region'],
                x=r['x'], y=r['y'], z=r['z'], volume=r['volume'])

REGIONS = sorted(set(r['region'] for r in rois))
# (1) 层级边: ROI -> 区域
for r in rois:
    KG.add_node(r['region'], type='region')
    KG.add_edge(r['id'], r['region'], rel='belongs_to')
# (2) 空间邻近边
coords = np.array([[r['x'], r['y'], r['z']] for r in rois])
dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
THRESH = 30.0
for i in range(N_ROI):
    for j in range(i + 1, N_ROI):
        if dist[i, j] < THRESH:
            KG.add_edge(i, j, rel='adjacent', weight=float(np.exp(-dist[i, j] / THRESH)))
# (3) 跨图谱别名边: 两个 ROI 若在 AAL 上第一标签相同 -> 连边
aal_primary = {}
for r in rois:
    if r['aal']:
        aal_primary.setdefault(r['aal'][0][0], []).append(r['id'])
for name, ids in aal_primary.items():
    if len(ids) > 1:
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                KG.add_edge(ids[a], ids[b], rel='alias', weight=0.8)

print(f"KG: nodes={KG.number_of_nodes()}, edges={KG.number_of_edges()}")
print(f"  spatial edges: {sum(1 for _,_,d in KG.edges(data=True) if d.get('rel')=='adjacent')}")
print(f"  alias edges:   {sum(1 for _,_,d in KG.edges(data=True) if d.get('rel')=='alias')}")

# KG 中心性
kg_pr = nx.pagerank(KG, weight='weight')
kg_deg = dict(KG.degree())
kg_cent = {n: kg_pr.get(n, 0.0) for n in range(N_ROI)}

# ROI -> 区域索引
region_idx = {rg: i for i, rg in enumerate(REGIONS)}
roi_region = np.array([region_idx[r['region']] for r in rois])

# 跨图谱共识度: 该 ROI 被多少个命名体系一致指向
consensus = []
for r in rois:
    sets = [set(n for n, _ in d) for d in [r['ho'], r['aal'], r['ez'], r['tt']] if d]
    if not sets:
        consensus.append(0.0)
    else:
        inter = set.intersection(*sets) if len(sets) > 1 else sets[0]
        consensus.append(len(inter) / len(sets))
consensus = np.array(consensus)

# ---------- 特征增强 ----------
def kg_features(degree_vec):
    """输入: 被试的 200 维 degree 特征 -> KG 衍生特征"""
    d = np.asarray(degree_vec, dtype=float)
    n_reg = len(REGIONS)
    # (a) 区域聚合: mean, std, max per region
    reg_feats = []
    for rg in range(n_reg):
        mask = roi_region == rg
        if mask.sum() == 0:
            reg_feats.extend([0.0, 0.0, 0.0])
        else:
            vals = d[mask]
            reg_feats.extend([vals.mean(), vals.std(), vals.max()])
    # (b) KG PageRank 加权
    pr_w = d * np.array([kg_cent[i] for i in range(N_ROI)])
    # (c) 区域密度比
    reg_mean = np.zeros(N_ROI)
    for rg in range(n_reg):
        mask = roi_region == rg
        if mask.sum() > 0:
            reg_mean[mask] = d[mask].mean()
    ratio = d - reg_mean
    # (d) 共识度加权
    cons_w = d * consensus
    return np.concatenate([
        np.asarray(reg_feats),        # 3*n_reg
        pr_w,                          # N_ROI
        ratio,                         # N_ROI
        cons_w,                        # N_ROI
    ])

KG_DIM = 3 * len(REGIONS) + 3 * N_ROI
print(f"KG feature dim: {KG_DIM}")

# ---------- 加载数据 ----------
df = pd.read_csv(FEAT_CSV)
df["features_value"] = df["features_value"].apply(ast.literal_eval)
core = df[(df["graph_feature_type"] == "node_based") & (df["feature"] == "degree")]

def build_models():
    return {
        "LR": lambda: LogisticRegression(max_iter=2000, random_state=42),
        "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "XGB": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
        "LGBM": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

def cv_eval(X, y, model_factory, n_pca=None):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, senss, specs = [], [], []
    for tr, te in cv.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        if n_pca and X_tr.shape[1] > n_pca and X_tr.shape[0] > n_pca:
            pca = PCA(n_components=n_pca, random_state=42)
            X_tr = pca.fit_transform(X_tr); X_te = pca.transform(X_te)
        model = model_factory()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        cm = confusion_matrix(y_te, pred)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            senss.append(tp/(tp+fn) if tp+fn else 0)
            specs.append(tn/(tn+fp) if tn+fp else 0)
    return np.mean(accs)*100, np.mean(senss)*100, np.mean(specs)*100

rows = []
sites = sorted(core["SITE_ID"].unique())
for site in sites:
    sd = core[core["SITE_ID"] == site]
    for atlas in ["cc200"]:
        for fe in ["original", "mi_10", "pca_10"]:
            d = sd[(sd["atlas"] == atlas) & (sd["feature_engineering"] == fe)]
            if d.empty or len(d) < 20:
                continue
            X_raw = np.stack(d["features_value"].values)
            y = d["ASD"].values
            # 只在 cc200 上做 KG 增强 (KG 基于 CC200 ROI)
            if X_raw.shape[1] != N_ROI:
                continue
            X_kg = np.stack([kg_features(f) for f in X_raw])
            X_comb = np.concatenate([X_raw, X_kg], axis=1)
            for mname, mf in build_models().items():
                for tag, X in [("raw", X_raw), ("kg", X_kg), ("raw+kg", X_comb)]:
                    npc = 100 if X.shape[1] > 100 else None
                    a, s, sp = cv_eval(X, y, mf, n_pca=npc)
                    rows.append({"site": site, "atlas": atlas, "fe": fe, "model": mname,
                                 "variant": tag, "acc": a, "sens": s, "spec": sp, "n": len(y)})
    rr = [r for r in rows if r["site"] == site]
    if rr:
        b = max(rr, key=lambda r: r["acc"])
        raw_best = max([r for r in rr if r["variant"] == "raw"], key=lambda r: r["acc"])
        print(f"{site}: raw={raw_best['acc']:.2f}% -> best={b['acc']:.2f}% ({b['variant']}/{b['fe']}/{b['model']})", flush=True)

res = pd.DataFrame(rows)
res.to_csv(r"D:/ASD-GraphNet - mhx/kg_results.csv", index=False)

print("\n=== 按 variant 平均 (每站点取最优) ===")
for tag in ["raw", "kg", "raw+kg"]:
    sub = res[res["variant"] == tag]
    best = sub.loc[sub.groupby("site")["acc"].idxmax()]
    w = best["n"]
    print(f"{tag:8s}: 简单平均={sub['acc'].mean():.2f}%  加权最优={((best['acc']*w).sum()/w.sum()):.2f}%")

print("\n=== 结论 ===")
w = res.groupby("site")["n"].first()
for tag in ["raw", "kg", "raw+kg"]:
    best = res[res["variant"] == tag].loc[
        res[res["variant"] == tag].groupby("site")["acc"].idxmax()].set_index("site")
    print(f"{tag:8s} 加权最优: {((best['acc']*w).sum()/w.sum()):.2f}%")
print("(参考: 论文 75.25%, 原始复现 75.63%)")
