"""从 ABIDE S3 下载 CPAC filt_global ROI 时间序列 (cc200)"""
import os
import pandas as pd
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR = r"D:/ASD-GraphNet - mhx/data/rois_cc200"
os.makedirs(OUT_DIR, exist_ok=True)

pheno = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv")
valid = pheno[(pheno["FILE_ID"] != "no_filename") & pheno["FILE_ID"].notna()]
print(f"Valid subjects with FILE_ID: {len(valid)}")

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/{}_rois_cc200.1D"

def download(row):
    fid = row["FILE_ID"]
    out = os.path.join(OUT_DIR, f"{fid}.1D")
    if os.path.exists(out):
        return (fid, "exists")
    try:
        urllib.request.urlretrieve(BASE.format(fid), out)
        return (fid, "ok")
    except Exception as e:
        return (fid, f"fail: {e}")

ok, fail = 0, 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(download, r) for _, r in valid.iterrows()]
    for i, f in enumerate(as_completed(futs)):
        fid, status = f.result()
        if status in ("ok", "exists"):
            ok += 1
        else:
            fail += 1
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(valid)}] ok={ok} fail={fail}", flush=True)

print(f"\nDone: ok={ok}, fail={fail}")