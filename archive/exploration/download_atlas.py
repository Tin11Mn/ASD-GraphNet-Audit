"""下载 aal, dos160 坐标系下的 ROI 时间序列"""
import os, pandas as pd, urllib.request

pheno = pd.read_csv(r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/pheno_file.csv")
valid = pheno[pheno["FILE_ID"] != "no_filename"]
file2sid = dict(zip(valid["FILE_ID"], valid["SUB_ID"]))

ATLASES = ["aal", "dosenbach160"]
TPL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_{}/{}_rois_{}.1D"

for atlas in ATLASES:
    OUT = f"D:/ASD-GraphNet - mhx/data/rois_{atlas}"
    os.makedirs(OUT, exist_ok=True)
    count = 0
    for fid in valid["FILE_ID"].unique():
        url = TPL.format(atlas, fid, atlas)
        out = os.path.join(OUT, f"{fid}.1D")
        if os.path.exists(out):
            count += 1
            continue
        try:
            urllib.request.urlretrieve(url, out)
            count += 1
        except:
            pass
        if count % 100 == 0:
            print(f"{atlas}: {count}", flush=True)
    print(f"{atlas} done: {count} files")