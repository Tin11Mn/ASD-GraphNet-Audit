"""
共享预处理: 计算并缓存 FC 矩阵与时间序列
输出: data/cache_fc.npz (subject_id -> 200x200 FC), data/cache_ts.npz (subject_id -> 200xT)
"""
import os, numpy as np, pandas as pd

ROI_DIR = r"D:/ASD-GraphNet - mhx/data/rois_cc200"
CACHE_DIR = r"D:/ASD-GraphNet - mhx/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

pheno = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv")
valid = pheno[pheno["FILE_ID"] != "no_filename"]
file2sid = dict(zip(valid["FILE_ID"], valid["SUB_ID"]))

# 建立 sid -> filepath 索引
sid2path = {}
for f in os.listdir(ROI_DIR):
    if not f.endswith(".1D"):
        continue
    fid = f[:-3]
    sid = file2sid.get(fid)
    if sid is not None:
        sid2path[str(sid)] = os.path.join(ROI_DIR, f)
print(f"Indexed {len(sid2path)} subject files")

fc_store, ts_store = {}, {}
for i, (sid, path) in enumerate(sid2path.items()):
    try:
        ts = np.loadtxt(path, delimiter="\t")  # (T, 200)
        if ts.ndim != 2 or ts.shape[1] != 200:
            continue
        ts_norm = (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-8)
        fc = np.corrcoef(ts_norm.T).astype(np.float32)  # (200, 200)
        np.fill_diagonal(fc, 1.0)
        fc_store[sid] = fc
        ts_store[sid] = ts.astype(np.float32).T  # 存成 (200, T)
    except Exception as e:
        continue
    if (i + 1) % 200 == 0:
        print(f"[{i+1}/{len(sid2path)}]", flush=True)

print(f"Computed FC for {len(fc_store)} subjects")

# 保存
np.savez_compressed(os.path.join(CACHE_DIR, "fc_all.npz"), **fc_store)
print(f"Saved FC cache -> {CACHE_DIR}/fc_all.npz")

# 时间序列单独存 (更大, 用 npz 不压缩以加快读取)
np.savez(os.path.join(CACHE_DIR, "ts_all.npz"), **ts_store)
print(f"Saved TS cache -> {CACHE_DIR}/ts_all.npz")

# 元信息
meta = pd.DataFrame([{"id": sid, "T": ts_store[sid].shape[1]} for sid in ts_store])
print("T range:", meta["T"].min(), "-", meta["T"].max())