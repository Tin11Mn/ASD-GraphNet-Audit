"""
ASD-GraphNet 知识图谱增强 V7
思路: 为每个 ROI 构建解剖学嵌入 (位置+功能网络+区域)，
    拼接到原始 degree 特征上形成 "节点+位置" 特征，
    让模型学习"哪些脑区/哪些位置的异常连改变变量更灵敏"
"""
import ast, json, warnings, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

# 读取 ROI 知识图谱
roi_df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/CC200_ROI_labels.csv")
roi_df.columns = [c.strip() for c in roi_df.columns]

# 解析 Harvard-Oxford 标签 (JSON 数组格式)
def parse_harvard_o(x):
    try:
        data = json.loads(x.replace('"', '\\"'))
        return list(data.items())
    except:
        return []

rois = []
for i, row in roi_df.iterrows():
    # 位置坐标 (xyz)
    coords = row['center of mass']
    x, y, z = [float(v) for v in re.findall(r'-?\d+\.?\d*', str(coords))]
    # Harvard-Oxford 解剖标签
    labels = parse_harvard_o(row.get('Harvard-Oxford', '[]'))
    primary_name = labels[0][0] if labels else f'ROI_{i+1}'
    rois.append({
        'id': i + 1,
        'x': x, 'y': y, 'z': z,
        'name': primary_name,
        'labels': labels
    })

print(f"Loaded {len(rois)} ROI embeddings")

# 加载特征数据
df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
df["features_value"] = df["features_value"].apply(ast.literal_eval)

# 为每个 ROI 构造 3D 位置编码 (归一化坐标 + 大脑区域)
def get_region_type(name):
    name = name.lower()
    if 'frontal' in name: return 1
    if 'parietal' in name: return 2  
    if 'temporal' in name: return 3
    if 'occipital' in name: return 4
    if 'cerebellum' in name: return 5
    if 'thalamus' in name: return 6
    if 'insula' in name: return 7
    return 0

roi_embeds = []
for r in rois:
    region = get_region_type(r['name'])
    enc = [r['x'], r['y'], r['z'], region]
    roi_embeds.append(enc)

roi_embeds = np.array(roi_embeds, dtype=float)
# 标准化位置坐标
roi_embeds[:, :3] = StandardScaler().fit_transform(roi_embeds[:, :3])

# 为每个被试构建增强特征: degree_features + position_features
def enhance_features(features_value, roi_embeds):
    """features_value: degree degree for each ROI"""
    feats = np.array(features_value, dtype=float)
    # 对每个 ROI, 将 degree 拼接到其位置编码上
    enhanced = np.concatenate([feats, roi_embeds], axis=1)
    return enhanced

# 实测
core = df[(df["graph_feature_type"] == "node_based") & (df["feature"] == "degree")]
site = "NYU"
site_data = core[core["SITE_ID"] == site]
# 按 atlas 选一个
a = site_data[site_data["atlas"] == "cc200"]
X = np.stack([enhance_features(f, roi_embeds) for f in a["features_value"]])
y = a["ASD"].values
print(f"Enhanced X shape: {X.shape} (original: {len(a)} subjects x 200 features)")

# 10-fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs = []
for tr, te in cv.split(X, y):
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    # 特征降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    accs.append(accuracy_score(y_te, rf.predict(X_te)))
    
print(f"NYU/cc200/node_based/original with ROI embedding + PCA: {np.mean(accs)*100:.2f}%")
print("(baseline: ~68.66%)")

# 保存结果
np.save(r"D:/ASD-GraphNet - mhx/roi_embeddings.npy", roi_embeds)
print(f"\nSaved ROI embeddings to roi_embeddings.npy")
print(f"Embedding dim per ROI: {roi_embeds.shape[1]} (coords:3 + region:1)")