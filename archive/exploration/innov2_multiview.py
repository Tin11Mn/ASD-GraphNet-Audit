"""
创新点 2: 多视图融合
- 视图1: cc200 FC (19900 维上三角)
- 视图2: aal FC (6670 维)
- 视图3: dos160 FC (12720 维)
对比:
  A) 单视图 LR
  B) 三视图 concat LR
  C) 三视图分别 LR → 概率平均 (late fusion)
  D) 三视图分别 LR → 堆叠 (stacking)
"""
import warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
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
    n = fc.shape[0]
    iu = np.triu_indices(n, k=1)
    return np.nan_to_num(fc[iu], nan=0.0)

ids, y, V1, V2, V3 = [], [], [], [], []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid not in fc_cc200 or sid not in fc_aal or sid not in fc_dos:
        continue
    ids.append(sid); y.append(int(r["ASD"]))
    V1.append(triu(fc_cc200[sid]))
    V2.append(triu(fc_aal[sid]))
    V3.append(triu(fc_dos[sid]))
y = np.array(y); V1 = np.array(V1); V2 = np.array(V2); V3 = np.array(V3)
print(f"Samples: {len(y)} (ASD={y.sum()})")
print(f"Views: cc200={V1.shape[1]}, aal={V2.shape[1]}, dos160={V3.shape[1]}")

cv = StratifiedKFold(10, shuffle=True, random_state=42)

def eval_lr(X):
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, C=0.1)
        lr.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], lr.predict(Xte))*100)
    return np.mean(accs), np.std(accs)

def eval_late_fusion(views):
    accs = []
    for tr, te in cv.split(V1, y):
        probs = np.zeros((len(te), 2))
        for V in views:
            sc = StandardScaler(); Xtr = sc.fit_transform(V[tr]); Xte = sc.transform(V[te])
            lr = LogisticRegression(max_iter=2000, C=0.1)
            lr.fit(Xtr, y[tr])
            probs += lr.predict_proba(Xte)
        pred = probs.argmax(1)
        accs.append(accuracy_score(y[te], pred)*100)
    return np.mean(accs), np.std(accs)

def eval_stacking(views):
    from sklearn.ensemble import RandomForestClassifier
    accs = []
    for tr, te in cv.split(V1, y):
        # 内层 CV 生成 meta 特征
        inner = StratifiedKFold(5, shuffle=True, random_state=0)
        meta_tr = np.zeros((len(tr), 2*len(views)))
        for itr, ite in inner.split(views[0][tr], y[tr]):
            for vi, V in enumerate(views):
                sc = StandardScaler(); Xtr = sc.fit_transform(V[tr][itr]); Xte = sc.transform(V[tr][ite])
                lr = LogisticRegression(max_iter=2000, C=0.1)
                lr.fit(Xtr, y[tr][itr])
                meta_tr[ite, 2*vi:2*vi+2] = lr.predict_proba(Xte)
        # 外层 test meta
        meta_te = np.zeros((len(te), 2*len(views)))
        for vi, V in enumerate(views):
            sc = StandardScaler(); Xtr = sc.fit_transform(V[tr]); Xte = sc.transform(V[te])
            lr = LogisticRegression(max_iter=2000, C=0.1)
            lr.fit(Xtr, y[tr])
            meta_te[:, 2*vi:2*vi+2] = lr.predict_proba(Xte)
        rf = RandomForestClassifier(n_estimators=200, random_state=0)
        rf.fit(meta_tr, y[tr])
        accs.append(accuracy_score(y[te], rf.predict(meta_te))*100)
    return np.mean(accs), np.std(accs)

print("\n=== 创新2: 多视图融合 ===")
a, s = eval_lr(V1); print(f"A) cc200 单视图 LR:      {a:.2f}% +/- {s:.2f}%")
a, s = eval_lr(V2); print(f"   aal 单视图 LR:        {a:.2f}% +/- {s:.2f}%")
a, s = eval_lr(V3); print(f"   dos160 单视图 LR:     {a:.2f}% +/- {s:.2f}%")
a, s = eval_lr(np.concatenate([V1, V2, V3], axis=1)); print(f"B) 三视图 concat LR:     {a:.2f}% +/- {s:.2f}%")
a, s = eval_late_fusion([V1, V2, V3]); print(f"C) 三视图 late fusion:   {a:.2f}% +/- {s:.2f}%")
a, s = eval_stacking([V1, V2, V3]); print(f"D) 三视图 stacking(RF):  {a:.2f}% +/- {s:.2f}%")
print(f"\n(论文 75.25%, 原始复现 75.63%)")
