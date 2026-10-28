"""
从原始 ROI 时间序列重建干净的图特征（无标签参与）

流程（严格对齐论文 utils/graph_features.py + graph_generation.py）:
  1. 读原始 ROI 时间序列 (.1D)
  2. Pearson 相关 -> 全连接加权图
  3. 阈值: 移除 |weight| <= threshold 的边
       node_based  : threshold = 0.49
       edge_based  : threshold = 0.50
       graph_level : threshold = 0.50
  4. 计算图特征 (networkx), 与论文 utils 完全一致
  5. multi atlas = 三图谱特征拼接(列表) / 平均(标量)

输出: clean_features.csv
  columns: id, ASD, SITE_ID, atlas, graph_feature_type, feature, features_value(list)
"""
import os, ast, warnings, time
import numpy as np, pandas as pd, networkx as nx
warnings.filterwarnings("ignore")
t0 = time.time()

ROOT = r"D:/ASD-GraphNet - mhx"
DATA = os.path.join(ROOT, "data")
PHENO = os.path.join(ROOT, "ASD-GraphNet-main", "ASD-GraphNet-main", "pheno_file.csv")
OUT = os.path.join(ROOT, "clean_features.csv")

ATLAS_DIRS = {"cc200": "rois_cc200", "aal": "rois_aal", "dos160": "rois_dosenbach160"}
THRESH = {"node_based": 0.49, "edge_based": 0.50, "graph_level": 0.50}

# ---------- 1. 读表型 ----------
ph = pd.read_csv(PHENO)
ph["FILE_ID"] = ph["FILE_ID"].astype(str).str.strip()
ph = ph[ph["FILE_ID"] != "no_filename"]
ph["ASD"] = ph["DX_GROUP"].map({1: 1, 2: 0})
ph = ph[["FILE_ID", "subject", "ASD", "SITE_ID"]].dropna(subset=["ASD"])
file2meta = ph.set_index("FILE_ID").to_dict("index")
print(f"表型有效被试: {len(ph)}")

# ---------- 2. 读时间序列 ----------
ts = {}   # FILE_ID -> {atlas: ndarray (T x n_roi)}
for atlas, d in ATLAS_DIRS.items():
    dirp = os.path.join(DATA, d)
    for fn in os.listdir(dirp):
        stem = fn[:-3] if fn.endswith(".1D") else fn.split(".")[0]
        try:
            arr = np.loadtxt(os.path.join(dirp, fn))
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] < 10:
            continue
        ts.setdefault(stem, {})[atlas] = arr

# 只保留有表型且三图谱齐全的被试
common = [s for s in ts if s in file2meta and len(ts[s]) == 3]
print(f"三图谱齐全且有表型的被试: {len(common)}")
meta = {s: file2meta[s] for s in common}
print(f"最终被试: {len(common)}  ASD={sum(m['ASD'] for m in meta.values()):.0f}  "
      f"站点={len(set(m['SITE_ID'] for m in meta.values()))}")

# ---------- 3. 图特征函数 (与论文 utils/graph_features.py 对齐) ----------
def build_graph(arr, thr):
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    G = nx.from_numpy_array(np.abs(corr))
    # 论文: 移除 |weight| <= threshold 的边
    drop = [(u, v) for u, v, d in G.edges(data=True) if abs(d["weight"]) <= thr]
    G.remove_edges_from(drop)
    return G

def node_degree(G):
    return [d for _, d in G.degree()]

def edge_features(G):
    comps = list(nx.connected_components(G))
    avg_deg = [sum(d for _, d in G.degree(c)) / len(c) for c in comps]
    return avg_deg

def graph_features(G):
    comps = list(nx.connected_components(G))
    avg_deg_conn = nx.average_degree_connectivity(G)
    dens = nx.density(G)
    diam = max((nx.diameter(G.subgraph(c)) for c in comps), default=0)
    rad = max((nx.radius(G.subgraph(c)) for c in comps), default=0)
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = 0.0
    trans = nx.transitivity(G)
    avg_clust = nx.average_clustering(G)
    # 论文 graph_level 用 'average_degree' 字段 = nx.average_degree_connectivity 的值(dict->list)
    return list(avg_deg_conn.values())

print("\n开始计算图特征...")
rows = []
for i, sid in enumerate(common):
    if i % 100 == 0:
        print(f"  {i}/{len(common)}  ({time.time()-t0:.0f}s)", flush=True)
    asd = int(meta[sid]["ASD"]); site = meta[sid]["SITE_ID"]
    per_atlas = {}
    for atlas in ATLAS_DIRS:
        arr = ts[sid][atlas]
        G = build_graph(arr, THRESH["node_based"])
        per_atlas[atlas] = {
            "node_based": node_degree(G),
        }
        Ge = build_graph(arr, THRESH["edge_based"])
        per_atlas[atlas]["edge_based"] = edge_features(Ge)
        Gg = build_graph(arr, THRESH["graph_level"])
        per_atlas[atlas]["graph_level"] = graph_features(Gg)
    # 单图谱行
    for atlas in ATLAS_DIRS:
        for gft in ["node_based", "edge_based", "graph_level"]:
            feat = "degree" if gft == "node_based" else "average_degree"
            rows.append({"id": sid, "ASD": asd, "SITE_ID": site, "atlas": atlas,
                         "graph_feature_type": gft, "feature": feat,
                         "features_value": per_atlas[atlas][gft]})
    # multi atlas 行 (拼接列表 / 平均标量)
    for gft in ["node_based", "edge_based", "graph_level"]:
        feat = "degree" if gft == "node_based" else "average_degree"
        vals = [per_atlas[a][gft] for a in ATLAS_DIRS]
        if all(isinstance(v, list) for v in vals):
            merged = [x for v in vals for x in v]
        else:
            merged = float(np.mean(vals))
        rows.append({"id": sid, "ASD": asd, "SITE_ID": site, "atlas": "multi",
                     "graph_feature_type": gft, "feature": feat,
                     "features_value": merged})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\n完成! 保存 {OUT}")
print(f"  行数: {len(df)}  被试: {df['id'].nunique()}  用时 {time.time()-t0:.0f}s")
print(f"  atlas: {sorted(df['atlas'].unique())}")
print(f"  每被试行数: {len(df)/df['id'].nunique():.1f}")
