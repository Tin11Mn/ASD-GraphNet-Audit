"""
验证论文 original 特征是否逐被试独立（无跨被试信息）
- node_based_features.csv: 每列 = 单被试的图统计量（长度 = ROI 数）
- edge_based_features.csv: 每列 = 单被试的连通分量平均度列表
- graph_level_features.csv: 每列 = 单被试的标量图特征
只要每列长度只取决于 atlas (ROI数) 而非其他被试, 就是无泄露的。
另外用 site_id_distribution.csv 核对站点数
"""
import pandas as pd, ast, numpy as np
OUT = r'D:/ASD-GraphNet - mhx/ASD-GraphNet-main/ASD-GraphNet-main/outputs'

nb = pd.read_csv(f'{OUT}/node_based_features.csv')
eb = pd.read_csv(f'{OUT}/edge_based_features.csv')
gb = pd.read_csv(f'{OUT}/graph_level_features.csv')

print('=== node_based 每列长度 (应只取决于 ROI 数) ===')
for c in nb.columns:
    if c in ('id', 'ASD', 'SITE_ID'): continue
    L = nb[c].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
    print(f'  {c:38s} len: min={L.min()} max={L.max()} uniq={sorted(L.unique())[:5]}')

print('\n=== edge_based 每列长度 (连通分量数, 因被试而异是正常的) ===')
for c in eb.columns:
    if c in ('id', 'ASD', 'SITE_ID'): continue
    if 'average_degree' in c or 'clustering' in c or 'shortest' in c or 'betweenness' in c or 'density' in c:
        L = eb[c].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
        print(f'  {c:44s} len: min={L.min()} max={L.max()} uniq={L.nunique()}')

print('\n=== graph_level 标量列 ===')
for c in gb.columns[:12]:
    if c in ('id', 'ASD', 'SITE_ID'): continue
    v = pd.to_numeric(gb[c], errors='coerce')
    print(f'  {c:36s} min={v.min():.4f} max={v.max():.4f}')

print('\n=== 被试/站点核对 ===')
print(f'  node_based: {len(nb)} 被试, {nb["SITE_ID"].nunique()} 站点')
print(f'  被试 id 样例: {nb["id"].head(5).tolist()}')
sd = pd.read_csv(f'{OUT}/site_stratification_analysis.csv')
print(f'\n  site_stratification_analysis.csv:')
print(sd.to_string())
