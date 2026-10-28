"""
汇总: 四个创新方向的最终对比
额外补充: 论文特征 (degree) 与多视图 FC 融合的最强组合
"""
import warnings, ast, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
np.random.seed(42)

CACHE = r"D:/ASD-GraphNet - mhx/data/cache"
fc_cc200 = dict(np.load(CACHE + "/fc_all.npz"))
fc_aal = dict(np.load(CACHE + "/fc_aal.npz"))
fc_dos = dict(np.load(CACHE + "/fc_dosenbach160.npz"))
df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
sub = df[(df["graph_feature_type"]=="node_based")&(df["feature"]=="degree")
         &(df["feature_engineering"]=="original")&(df["atlas"]=="cc200")].drop_duplicates("id")

def triu(fc):
    n = fc.shape[0]; iu = np.triu_indices(n, k=1)
    return np.nan_to_num(fc[iu], nan=0.0)

ids, y, V1, V2, V3, DEG = [], [], [], [], [], []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid not in fc_cc200 or sid not in fc_aal or sid not in fc_dos: continue
    ids.append(sid); y.append(int(r["ASD"]))
    V1.append(triu(fc_cc200[sid])); V2.append(triu(fc_aal[sid])); V3.append(triu(fc_dos[sid]))
    DEG.append(ast.literal_eval(r["features_value"]))
y = np.array(y); V1 = np.array(V1); V2 = np.array(V2); V3 = np.array(V3); DEG = np.array(DEG, dtype=float)
print(f"Samples: {len(y)} (ASD={y.sum()})")

cv = StratifiedKFold(10, shuffle=True, random_state=42)
def ev(X, tag):
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, C=0.1); lr.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], lr.predict(Xte))*100)
    print(f"{tag:42s} {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
    return np.mean(accs)

print("\n=== 汇总对比 (10-fold CV, LR 分类器) ===")
ev(DEG, "论文 degree 特征 (baseline)")
ev(V1, "cc200 FC 单视图")
ev(V2, "aal FC 单视图")
ev(V3, "dos160 FC 单视图")
ev(np.concatenate([V1,V2,V3],1), "三视图 FC concat")
ev(np.concatenate([DEG,V1,V2,V3],1), "degree + 三视图 FC concat")
print(f"\n(论文报告 75.25%, 本项目原始复现 75.63%)")
