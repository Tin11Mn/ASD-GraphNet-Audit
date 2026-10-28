"""
创新点 4: 时序图网络 (Sliding-Window FC + GNN + LSTM)
- 对每个被试的时间序列做滑窗 (W=30, stride=15) -> 每个窗口计算 FC 矩阵
- 每个窗口视为一张图 -> GCN 编码 -> 窗口嵌入序列
- LSTM 建模窗口间时序动态 -> 图级分类
对比:
  A) 静态 FC + LR (baseline)
  B) 动态 FC 多窗口 + LR 概率聚合 (无 GNN/LSTM)
  C) GNN + LSTM (完整时序图网络)
"""
import warnings, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader as TorchDL
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)

CACHE = r"D:/ASD-GraphNet - mhx/data/cache"
ts_data = dict(np.load(CACHE + "/ts_all.npz"))
fc_static = dict(np.load(CACHE + "/fc_all.npz"))
df = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv")
sub = df[(df["graph_feature_type"]=="node_based")&(df["feature"]=="degree")
         &(df["feature_engineering"]=="original")&(df["atlas"]=="cc200")].drop_duplicates("id")

W, STRIDE, MAXW = 30, 15, 8

def window_fcs(ts):
    """ts: (200, T) -> list of FC (200,200)"""
    T = ts.shape[1]
    starts = list(range(0, max(1, T - W + 1), STRIDE))
    if len(starts) > MAXW:
        sel = np.linspace(0, len(starts)-1, MAXW).astype(int)
        starts = [starts[i] for i in sel]
    out = []
    for s in starts:
        seg = ts[:, s:s+W]
        if seg.shape[1] < 10:
            continue
        seg_n = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-8)
        fc = np.corrcoef(seg_n).astype(np.float32)
        fc = np.nan_to_num(fc, nan=0.0); np.fill_diagonal(fc, 1.0)
        out.append(fc)
    return out

# 构建数据
ids, y = [], []
win_fcs = []
for _, r in sub.iterrows():
    sid = str(r["id"])
    if sid not in ts_data: continue
    fcs = window_fcs(ts_data[sid])
    if len(fcs) < 2: continue
    ids.append(sid); y.append(int(r["ASD"])); win_fcs.append(fcs)
y = np.array(y)
print(f"Subjects with windows: {len(y)}, avg windows: {np.mean([len(f) for f in win_fcs]):.1f}")

# 静态 FC baseline
IU = np.triu_indices(200, k=1)
X_static = np.array([np.nan_to_num(fc_static[s][IU], nan=0.0) for s in ids])

# ---------- B) 动态 FC + LR 概率聚合 ----------
def eval_dynamic_lr():
    cv = StratifiedKFold(10, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X_static, y):
        # 训练: 把所有训练窗口当作独立样本
        Xw, yw = [], []
        for i in tr:
            for fc in win_fcs[i]:
                Xw.append(np.nan_to_num(fc[IU], nan=0.0)); yw.append(y[i])
        Xw = np.array(Xw); yw = np.array(yw)
        sc = StandardScaler(); Xw_s = sc.fit_transform(Xw)
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(Xw_s, yw)
        # 测试: 窗口概率聚合
        preds = []
        for i in te:
            probs = []
            for fc in win_fcs[i]:
                probs.append(lr.predict_proba(sc.transform(np.nan_to_num(fc[IU], nan=0.0).reshape(1,-1)))[0])
            probs = np.array(probs)
            preds.append(probs.mean(0).argmax())
        accs.append(accuracy_score(y[te], preds)*100)
    return np.mean(accs), np.std(accs)

# ---------- C) GNN + LSTM ----------
class TemporalGNN(torch.nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.conv = GCNConv(3, hid)
        self.lstm = torch.nn.LSTM(hid, hid, batch_first=True)
        self.fc = torch.nn.Linear(hid, 2)
    def forward(self, seq_x, seq_ei, seq_mask):
        """seq_x: (B, L, N, F), seq_ei: (B, L, 2, Emax), seq_mask: (B, L)"""
        B, L, N, Fd = seq_x.shape
        pooled_seq = torch.zeros(B, L, self.conv.out_channels)
        for b in range(B):
            for l in range(L):
                if seq_mask[b, l] < 0.5:
                    continue
                x = seq_x[b, l]
                ei = seq_ei[b, l, :, :int(seq_ei[b, l].sum(dim=1).max())]
                # 简化: 用固定边数
                ei = seq_ei[b, l, :, :800]
                h = F.relu(self.conv(x, ei))
                pooled_seq[b, l] = global_mean_pool(h, torch.zeros(N, dtype=torch.long)).view(-1)
        out, _ = self.lstm(pooled_seq)
        idx = (seq_mask.sum(1).long() - 1).clamp(min=0)
        last = out[torch.arange(B), idx]
        return self.fc(last)

def make_graph(fc, thr=0.5, K=800):
    adj = (np.abs(fc) > thr).astype(float); np.fill_diagonal(adj, 0)
    if adj.sum() > K:
        flat = np.argwhere(adj > 0)
        vals = np.array([np.abs(fc[i,j]) for i,j in flat])
        keep = flat[np.argsort(vals)[::-1][:K]]
        adj = np.zeros_like(adj); adj[keep[:,0], keep[:,1]] = 1
    elif adj.sum() < K:
        # 补齐到固定边数 (填充 (0,0) 自环, 无害)
        need = K - int(adj.sum())
        pad = np.zeros((need, 2), dtype=int)
        adj[pad[:,0], pad[:,1]] = 1
    feats = np.stack([fc.mean(1), fc.std(1), adj.sum(1)/199], 1)
    feats = np.nan_to_num(feats)
    return torch.tensor(feats, dtype=torch.float), torch.tensor(np.argwhere(adj>0).T, dtype=torch.long)

def eval_temporal_gnn(epochs=12):
    cv = StratifiedKFold(10, shuffle=True, random_state=42)
    accs = []
    # 预先构建所有窗口图
    all_seq_x, all_seq_ei = [], []
    for fcs in win_fcs:
        xs, eis = [], []
        for fc in fcs:
            x, ei = make_graph(fc)
            xs.append(x); eis.append(ei)
        all_seq_x.append(torch.stack(xs)); all_seq_ei.append(torch.stack(eis))
    for fold, (tr, te) in enumerate(cv.split(X_static, y)):
        model = TemporalGNN()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = torch.nn.CrossEntropyLoss()
        model.train()
        # padding 到统一长度
        Lmax = max(sx.size(0) for sx in all_seq_x)
        def pad(idx_list):
            sx = torch.zeros(len(idx_list), Lmax, 200, 3)
            se = torch.zeros(len(idx_list), Lmax, 2, 800)
            mk = torch.zeros(len(idx_list), Lmax)
            for b, i in enumerate(idx_list):
                L = all_seq_x[i].size(0)
                sx[b, :L] = all_seq_x[i]; mk[b, :L] = 1
                for l in range(L):
                    e = all_seq_ei[i][l]
                    se[b, l, :, :e.size(1)] = e
            return sx, se.long(), mk
        sx_tr, se_tr, mk_tr = pad(tr)
        ds = torch.utils.data.TensorDataset(sx_tr, se_tr, mk_tr, torch.tensor(y[tr]).long())
        loader = TorchDL(ds, batch_size=16, shuffle=True)
        for ep in range(epochs):
            for sx, se, mk, sy in loader:
                opt.zero_grad()
                out = model(sx, se, mk)
                loss = crit(out, sy)
                loss.backward(); opt.step()
        model.eval()
        sx_te, se_te, mk_te = pad(te)
        with torch.no_grad():
            out = model(sx_te, se_te, mk_te)
            preds = out.argmax(1).tolist()
        acc = accuracy_score(y[te], preds)*100
        accs.append(acc)
        print(f"  fold {fold+1}: {acc:.2f}%", flush=True)
    return np.mean(accs), np.std(accs)

print("\n=== 创新4: 时序图网络 ===")
cv = StratifiedKFold(10, shuffle=True, random_state=42)
accs = []
for tr, te in cv.split(X_static, y):
    sc = StandardScaler(); Xtr = sc.fit_transform(X_static[tr]); Xte = sc.transform(X_static[te])
    lr = LogisticRegression(max_iter=2000, C=0.1); lr.fit(Xtr, y[tr])
    accs.append(accuracy_score(y[te], lr.predict(Xte))*100)
print(f"A) 静态 FC + LR:            {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
a, s = eval_dynamic_lr(); print(f"B) 动态 FC(滑窗) + LR 聚合:  {a:.2f}% +/- {s:.2f}%")
a, s = eval_temporal_gnn(epochs=12); print(f"C) GNN + LSTM 时序图网络:    {a:.2f}% +/- {s:.2f}%")
print(f"\n(论文 75.25%, 原始复现 75.63%)")
