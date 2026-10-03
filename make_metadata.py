"""Buat metadata.csv dari dataset_raw/<kelas>/*.jpg dengan split train/val/test (70/15/15)."""
import argparse, os, random
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="dataset_raw")
ap.add_argument("--out", default="metadata.csv")
ap.add_argument("--source", default="foto sendiri", help="sumber data, mis. 'foto sendiri' / URL dataset")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

random.seed(a.seed)
rows = []
for cls in sorted(d for d in os.listdir(a.root) if os.path.isdir(f"{a.root}/{d}")):
    files = sorted(f for f in os.listdir(f"{a.root}/{cls}") if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(files) < 50:
        print(f"[PERINGATAN] kelas '{cls}' hanya {len(files)} citra (minimal 50)")
    random.shuffle(files)
    n = len(files); n_tr = int(0.7 * n); n_va = int(0.15 * n)
    for i, f in enumerate(files):
        split = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
        rows.append(dict(filename=f"{cls}/{f}", label=cls, sumber=a.source, split=split))
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
