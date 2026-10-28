import ast, pandas as pd, numpy as np
df = pd.read_csv(r'D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv')
o = df[(df['feature_engineering'] == 'original') & (df['atlas'] == 'cc200')]
for gft in ['node_based', 'edge_based', 'graph_level']:
    s = o[o['graph_feature_type'] == gft]
    feats = list(s['feature'].unique())
    print(f'--- cc200 {gft} | features: {feats} | rows/subject: {len(s)/1035:.1f} ---')
    for feat in feats:
        ss = s[s['feature'] == feat]
        v = ast.literal_eval(ss['features_value'].iloc[0])
        print(f'   {feat:20s} dim={len(v):4d}  head={np.round(v[:5], 4)}')

print()
print('=== multi atlas 结构 ===')
m = df[(df['feature_engineering'] == 'original') & (df['atlas'] == 'multi')]
for gft in ['node_based']:
    s = m[m['graph_feature_type'] == gft]
    feats = list(s['feature'].unique())
    print(f'  {gft}: features={feats}')
    for feat in feats:
        ss = s[s['feature'] == feat]
        v = ast.literal_eval(ss['features_value'].iloc[0])
        print(f'   {feat:16s} dim={len(v):4d}')

print()
print('=== 被试数核对 ===')
meta = df[['id', 'ASD', 'SITE_ID']].drop_duplicates(subset=['id'])
print(f'filtered_subjects_df 唯一被试: {len(meta)}')
print(f'ASD 分布: {meta["ASD"].value_counts().to_dict()}')
print(f'站点数: {meta["SITE_ID"].nunique()}')
print()
print('按站点:')
print(meta['SITE_ID'].value_counts().sort_index().to_string())
