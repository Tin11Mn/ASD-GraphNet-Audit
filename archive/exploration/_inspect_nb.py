import json, re
p = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/experiments_pipeline_classifiers_revised.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]
print(f"total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if any(k in src for k in ["mutual_info", "mi_10", "SelectKBest", "PCA", "cross_val", "StratifiedKFold", "train_test_split", "weighted", "average"]):
        print(f"\n===== cell {i} ({c['cell_type']}) =====")
        print(src[:1500])
