import ast, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")

ROOT = r"D:/ASD-GraphNet - mhx"
df = pd.read_csv(f"{ROOT}/ASD-GraphNet-main/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
df["features_value"] = df["features_value"].apply(ast.literal_eval)
orig = df[df["feature_engineering"] == "original"].copy()
meta = orig[["id", "ASD", "SITE_ID"]].drop_duplicates("id").copy()
meta["id"] = meta["id"].astype(int)
IDS = np.array(meta["id"].tolist()); Y = meta["ASD"].values; SITE = meta["SITE_ID"].values
FEAT = {}
for (a, l), g in orig.groupby(["atlas", "graph_feature_type"]):
    FEAT[(a, l)] = {int(r["id"]): np.array(r["features_value"], dtype=float) for _, r in g.iterrows()}

site = "UM"   # 最大站点
idx = np.where(SITE == site)[0]
tr_ids = IDS[idx[:126]]; ytr = Y[idx[:126]]
y = ytr

def timeit(label, fn, reps=1):
    t = time.time()
    for _ in range(reps): r = fn()
    print(f"  {label:38s} {(time.time()-t)/reps*1000:8.1f} ms")
    return r

for blk in [("cc200","node_based"), ("cc200","edge_based"), ("cc200","graph_level")]:
    X = np.stack([FEAT[blk][i] for i in tr_ids])
    print(f"\n[{blk}] dim={X.shape[1]}")
    sc = StandardScaler(); A = sc.fit_transform(X)
    timeit("StandardScaler", lambda: StandardScaler().fit_transform(X))
    timeit("mutual_info_classif", lambda: mutual_info_classif(A, y, random_state=42, n_neighbors=3), reps=2)
    timeit("PCA(10)", lambda: PCA(n_components=10, random_state=42).fit_transform(A), reps=2)
    timeit("LR fit", lambda: LogisticRegression(max_iter=1000, C=0.1).fit(A, y), reps=3)
    timeit("RF(100) fit", lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(A, y), reps=3)
    timeit("RF(50) fit", lambda: RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1).fit(A, y), reps=3)
    from xgboost import XGBClassifier
    timeit("XGB(100) fit", lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0, tree_method="hist").fit(A, y), reps=3)
