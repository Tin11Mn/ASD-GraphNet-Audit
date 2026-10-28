import pandas as pd, ast, numpy as np
cl = pd.read_csv(r'D:/ASD-GraphNet - mhx/clean_features.csv')
orig = pd.read_csv(r'D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv')

print('=== 站点对比 ===')
cs = cl.groupby('SITE_ID')['id'].nunique().sort_index()
os_ = orig[orig['feature_engineering'] == 'original'].groupby('SITE_ID')['id'].nunique().sort_index()
cmp = pd.DataFrame({'clean': cs, 'orig': os_}).fillna(0).astype(int)
cmp['diff'] = cmp['clean'] - cmp['orig']
print(cmp.to_string())
print(f"\nclean 站点数: {len(cs)}  orig 站点数: {len(os_)}")
print(f"clean 总计: {cs.sum()}  orig 总计: {os_.sum()}")

print('\n=== 被试交集 ===')
cids = set(cl['id']); oids = set(orig[orig['feature_engineering'] == 'original']['id'])
print(f"clean: {len(cids)}  orig: {len(oids)}  交集: {len(cids & oids)}")
print(f"仅 clean 有: {len(cids - oids)}  仅 orig 有: {len(oids - cids)}")

print('\n=== 维度对比 (验证重建正确性) ===')
cl['n'] = cl['features_value'].apply(lambda x: len(ast.literal_eval(x)))
orig['n'] = orig['features_value'].apply(lambda x: len(ast.literal_eval(x)))
a = cl[cl['feature_engineering' if 'feature_engineering' in cl.columns else 'atlas'].notna()].copy() if False else cl
print('--- clean ---')
print(a.groupby(['atlas', 'graph_feature_type'])['n'].agg(['min', 'max']).to_string())
print('--- orig ---')
print(orig[orig['feature_engineering'] == 'original'].groupby(['atlas', 'graph_feature_type'])['n'].agg(['min', 'max']).to_string())

# 逐被试数值比对 (cc200 node_based degree)
print('\n=== 数值比对: cc200 node_based degree (前3个共同被试) ===')
oc = orig[(orig['atlas'] == 'cc200') & (orig['graph_feature_type'] == 'node_based')].set_index('id')
cc = cl[(cl['atlas'] == 'cc200') & (cl['graph_feature_type'] == 'node_based')].set_index('id')
for sid in sorted(cids & oids)[:3]:
    v1 = np.array(ast.literal_eval(oc.loc[sid, 'features_value']), dtype=float)
    v2 = np.array(ast.literal_eval(cc.loc[sid, 'features_value']), dtype=float)
    same = len(v1) == len(v2) and np.allclose(v1, v2)
    print(f"  {sid}: orig_dim={len(v1)} clean_dim={len(v2)} 完全一致={same}")
    if len(v1) == len(v2) and not same:
        print(f"    max_diff={np.abs(v1-v2).max():.6f}  corr={np.corrcoef(v1,v2)[0,1]:.6f}")
