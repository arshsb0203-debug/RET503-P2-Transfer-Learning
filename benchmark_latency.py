"""Ukur latensi model (preprocessing + inferensi) per gambar di perangkat yang dipakai."""
import argparse, glob, time
import cv2, numpy as np, pandas as pd, torch
from common import build_model, preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/model_fine_tuning.pt")
ap.add_argument("--img_dir", default="dataset_raw")
ap.add_argument("--runs", type=int, default=100)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.ckpt, map_location=dev)
model = build_model(ck["arch"], len(ck["classes"]), "scratch").to(dev)
model.load_state_dict(ck["state"]); model.eval()

img = cv2.imread(glob.glob(f"{a.img_dir}/**/*.jp*g", recursive=True)[0])
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
ts = []
with torch.no_grad():
    for i in range(a.runs + 20):
        t0 = time.perf_counter()
        x = preprocess(img).unsqueeze(0).to(dev)
        model(x); sync()
        if i >= 20:  # 20 iterasi pertama = warm-up
            ts.append((time.perf_counter() - t0) * 1000)
ts = np.array(ts)
res = dict(model=ck["arch"], device=dev, mean_ms=ts.mean().round(2), p95_ms=np.percentile(ts, 95).round(2),
           fps=round(1000 / ts.mean(), 1))
print(res); pd.DataFrame([res]).to_csv("results/latensi.csv", index=False)
