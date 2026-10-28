"""
为 aal / dosenbach160 计算 FC 缓存
"""
import os, numpy as np, pandas as pd

CACHE_DIR = r"D:/ASD-GraphNet - mhx/data/cache"
pheno = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv")
valid = pheno[pheno["FILE_ID"] != "no_filename"]
file2sid = dict(zip(valid["FILE_ID"], valid["SUB_ID"]))

for atlas, n_roi in [("aal", 116), ("dosenbach160", 160)]:
    ROI_DIR = f"D:/ASD-GraphNet - mhx/data/rois_{atlas}"
    store = {}
    for f in os.listdir(ROI_DIR):
        if not f.endswith(".1D"):
            continue
        sid = file2sid.get(f[:-3])
        if sid is None:
            continue
        try:
            ts = np.loadtxt(os.path.join(ROI_DIR, f), delimiter="\t")
            if ts.ndim != 2:
                continue
            # dos160 文件首列是 ROI 序号，需去除
            if atlas == "dosenbach160" and ts.shape[1] == n_roi + 1:
                ts = ts[:, 1:]
            if ts.shape[1] != n_roi:
                continue
            ts_n = (ts - ts.mean(0, keepdims=True)) / (ts.std(0, keepdims=True) + 1e-8)
            fc = np.corrcoef(ts_n.T).astype(np.float32)
            np.fill_diagonal(fc, 1.0)
            store[str(sid)] = fc
        except:
            continue
    out = os.path.join(CACHE_DIR, f"fc_{atlas}.npz")
    np.savez_compressed(out, **store)
    print(f"{atlas}: {len(store)} subjects, expected {n_roi} ROIs -> {out}")
