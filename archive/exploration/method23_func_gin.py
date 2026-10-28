"""
方法2+3 完整实现: 功能图谱 + 批量 GNN (GIN) + 数据增强
节点 = 39 个功能社区, 节点特征 = 社区连接向量
评估: 每站点 5-fold CV, GIN vs LR/XGB (相同输入特征)
"""
import time, numpy as np, os, warnings, random, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42); random.seed(42)
device = torch.device("cpu"); torch.set_num_threads(8)
t0 = time.time()

ROI_DIR = r"D:\ASD-GraphNet - mhx\data\rois_cc200"
PHENO   = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv"
CACHE   = r"D:/ASD-GraphNet - mhx/data/func_atlas_cache.npz"

df = pd.read_csv(PHENO, low_memory=False)
df = df[df['FILE_ID'] != 'no_filename'][['SUB_ID','FILE_ID','SITE_ID','DX_GROUP']]
id2file = dict(zip(df['SUB_ID'], df['FILE_ID']))

# ── 1. 加载时间序列 ──
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    X_ts, ids, labels, kmeans_labels = d['X_ts'], d['ids'], d['labels'], d['kmeans_labels']
    print(f"加载缓存: {X_ts.shape}")
else:
    all_ts, ids = [], []
    for sid, fid in id2file.items():
        p = os.path.join(ROI_DIR, f"{fid}.1D")
        if not os.path.exists(p): continue
        ts = np.loadtxt(p, delimiter='\t')
        if ts.ndim != 2 or ts.shape[1] != 200: continue
        ts = ts[:, :200]
        if ts.shape[0] < 200:
            ts = np.pad(ts, ((0, 200-ts.shape[0]), (0,0)), mode='constant')
        else:
            ts = ts[:200, :]
        ts = (ts - ts.mean()) / (ts.std() + 1e-8)
        all_ts.append(ts); ids.append(sid)
    X_ts = np.array(all_ts)
    labels = np.array([df[df['SUB_ID']==s]['DX_GROUP'].iloc[0] for s in ids])
    # 功能分区: KMeans on ROI 连接 profile
    roi_mean = X_ts.mean(axis=0)                       # (200, T)
    roi_fc = np.corrcoef(roi_mean.T)                   # (200, 200)
    roi_profile = roi_fc.mean(axis=1).reshape(-1, 1)   # (200, 1)
    km = KMeans(n_clusters=39, n_init=10, random_state=42)
    kmeans_labels = km.fit_predict(roi_profile)
    np.savez_compressed(CACHE, X_ts=X_ts, ids=np.array(ids), labels=labels,
                        kmeans_labels=kmeans_labels)
    print(f"构建缓存: {X_ts.shape}, 社区分布 {np.bincount(kmeans_labels)}")

labels = (labels == 2).astype(int)   # DX_GROUP: 1=ASD, 2=TC
K = 39
site_of = np.array([df[df['SUB_ID']==s]['SITE_ID'].iloc[0] for s in ids])
print(f"样本 {len(labels)}, ASD={labels.sum()}, 站点 {len(set(site_of))}")

# ── 2. 构建图数据: 节点=社区, 节点特征=连接向量+度 ──
def subject_graph(i):
    ts = X_ts[i]                                    # (200, T)
    cts = np.stack([ts[kmeans_labels == k].mean(axis=0) if (kmeans_labels==k).sum()>0
                    else np.zeros(ts.shape[1]) for k in range(K)])   # (K, T)
    fc = np.corrcoef(cts.T)                         # (K, K)
    np.fill_diagonal(fc, 0.0)
    absfc = np.abs(fc)
    deg = absfc.sum(axis=1)
    # 节点特征: 自己的连接向量(39) + 度(1) + 标准化度(1)
    node_feat = np.concatenate([absfc, deg.reshape(-1,1),
                                (deg/ (deg.sum()+1e-8)).reshape(-1,1)], axis=1)  # (K, 41)
    # 边: |FC| > 阈值
    adj = absfc > 0.3
    np.fill_diagonal(adj, False)
    row, col = np.where(adj)
    edge_index = np.stack([row, col])
    # 全局图特征
    tri = absfc[np.triu_indices(K,1)]
    glob = np.array([absfc.mean(), absfc.std(), deg.mean(), deg.std(), deg.max(),
                     (deg==0).sum(), tri.mean(), tri.std(),
                     np.percentile(tri,90), np.percentile(tri,10)])
    return node_feat.astype(np.float32), edge_index.astype(np.int64), glob.astype(np.float32)

print("\n构建图...")
NODE_FEAT, EDGE_IDX, GLOB_FEAT = [], [], []
for i in range(len(labels)):
    nf, ei, gf = subject_graph(i)
    NODE_FEAT.append(nf); EDGE_IDX.append(ei); GLOB_FEAT.append(gf)
NODE_FEAT = np.stack(NODE_FEAT); GLOB_FEAT = np.stack(GLOB_FEAT)
print(f"节点特征 {NODE_FEAT.shape}, 全局特征 {GLOB_FEAT.shape}")

# ── 3. GIN 模型 ──
class GIN(nn.Module):
    def __init__(self, in_dim, g_dim, hidden=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.eps1 = nn.Parameter(torch.zeros(1))
        self.eps2 = nn.Parameter(torch.zeros(1))
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, hidden)
        self.glob_lin = nn.Linear(g_dim, hidden)
        self.out = nn.Linear(2*hidden, 2)
        self.drop = nn.Dropout(0.3)
    def forward(self, x, edge_index, batch, gfeat):
        h = F.relu(self.lin1(x))
        row, col = edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])
        cnt = torch.zeros(h.shape[0], 1, device=h.device)
        cnt.index_add_(0, row, torch.ones(row.shape[0], 1, device=h.device))
        agg = agg / (cnt + 1e-6)
        h = F.relu((1+self.eps1)*h + agg)
        h = self.drop(h)
        h2 = torch.zeros_like(h)
        h2.index_add_(0, row, h[col])
        cnt2 = torch.zeros(h.shape[0],1, device=h.device)
        cnt2.index_add_(0, row, torch.ones(row.shape[0],1, device=h.device))
        h2 = h2/(cnt2+1e-6)
        h = F.relu((1+self.eps2)*h + h2)
        # pool: mean + max
        n_graph = int(batch.max().item())+1
        means = torch.zeros(n_graph, h.shape[1], device=h.device)
        means.index_add_(0, batch, h)
        counts = torch.zeros(n_graph, device=h.device)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        means = means / counts.unsqueeze(1)
        maxs = torch.full((n_graph, h.shape[1]), -1e9, device=h.device)
        maxs = maxs.index_reduce_(0, batch, h, 'amax', include_self=True)
        glob = F.relu(self.glob_lin(gfeat))
        out = self.out(torch.cat([means, glob], dim=1))
        return out

def collate(idx_list):
    xs, eis, batch = [], [], []
    off = 0
    for b, i in enumerate(idx_list):
        xs.append(torch.tensor(NODE_FEAT[i]))
        ei = torch.tensor(EDGE_IDX[i]) + off
        eis.append(ei); off += NODE_FEAT[i].shape[0]
        batch.append(torch.full((NODE_FEAT[i].shape[0],), b, dtype=torch.long))
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(batch),
            torch.tensor(GLOB_FEAT[idx_list]))

def augment(node_x, edge_index, p_drop=0.1, noise=0.02):
    """边随机删除 + 特征高斯噪声"""
    m = torch.rand(edge_index.shape[1]) > p_drop
    ei = edge_index[:, m]
    x = node_x + torch.randn_like(node_x)*noise
    return x, ei

def train_gin(train_idx, test_idx, y, epochs=60, lr=0.003, p_drop=0.1):
    model = GIN(NODE_FEAT.shape[2], GLOB_FEAT.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    y_t = torch.tensor(y, dtype=torch.long)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        bs = 32
        for s in range(0, len(perm), bs):
            bi = perm[s:s+bs]
            x, ei, bt, gf = collate(bi)
            x, ei = augment(x, ei, p_drop, 0.02)
            x, ei, bt, gf = x.to(device), ei.to(device), bt.to(device), gf.to(device)
            opt.zero_grad()
            out = model(x, ei, bt, gf)
            loss = crit(out, y_t[bi].to(device))
            loss.backward(); opt.step()
    # 评估
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(test_idx), 32):
            bi = test_idx[s:s+32]
            x, ei, bt, gf = collate(bi)
            out = model(x, ei.to(device), bt.to(device), gf.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)

# ── 4. 每站点 5-fold CV ──
SITES = sorted(set(site_of))
print("\n" + "="*78)
print(f"方法2+3: 功能图谱(K=39) + GIN vs 传统模型  (每站点 5-fold CV)")
print("="*78)

results = {}
for site in SITES:
    idx_all = np.where(site_of == site)[0]
    if len(idx_all) < 25 or min(np.bincount(labels[idx_all], minlength=2)) < 6:
        continue
    gin_accs, lr_accs, rf_accs = [], [], []
    for seed in [42, 123]:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in cv.split(idx_all, labels[idx_all]):
            tr_i, te_i = idx_all[tr], idx_all[te]
            # GIN
            p = train_gin(tr_i, te_i, labels)
            gin_accs.append(accuracy_score(labels[te_i], p)*100)
            # 传统: 节点特征 + 全局特征 展平
            flat = np.stack([np.concatenate([NODE_FEAT[i].ravel(), GLOB_FEAT[i]])
                              for i in range(len(labels))])
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            sc = StandardScaler()
            Xtr = sc.fit_transform(flat[tr_i]); Xte = sc.transform(flat[te_i])
            m = LogisticRegression(max_iter=2000, C=0.1); m.fit(Xtr, labels[tr_i])
            lr_accs.append(accuracy_score(labels[te_i], m.predict(Xte))*100)
            m2 = RandomForestClassifier(n_estimators=200, random_state=42); m2.fit(Xtr, labels[tr_i])
            rf_accs.append(accuracy_score(labels[te_i], m2.predict(Xte))*100)
    results[site] = (np.mean(gin_accs), np.mean(lr_accs), np.mean(rf_accs), len(idx_all))
    print(f"  {site:10s} n={len(idx_all):4d}  GIN={np.mean(gin_accs):.2f}%  "
          f"LR={np.mean(lr_accs):.2f}%  RF={np.mean(rf_accs):.2f}%", flush=True)

# 加权汇总
tw = sum(v[3] for v in results.values())
w_gin = sum(v[0]*v[3] for v in results.values())/tw
w_lr  = sum(v[1]*v[3] for v in results.values())/tw
w_rf  = sum(v[2]*v[3] for v in results.values())/tw
print("\n" + "="*78)
print(f"加权平均 (n={tw}): GIN={w_gin:.2f}%  LR={w_lr:.2f}%  RF={w_rf:.2f}%")
print(f"论文基线 75.63%  |  v9 改进 77.59%")
print(f"用时 {time.time()-t0:.0f}s")

np.save(r"D:/ASD-GraphNet - mhx/method23_results.npy",
        np.array([(s, *v) for s, v in results.items()], dtype=object), allow_pickle=True)
