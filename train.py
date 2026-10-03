"""Latih 3 mode: scratch, feature_extraction, fine_tuning. Hasil: tabel CSV + grafik akurasi per epoch."""
import argparse, os, time
import pandas as pd
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from common import ImageDataset, build_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mobilenet_v3_small",
                help="mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | resnet18 | resnet50")
ap.add_argument("--data", default="dataset_raw")
ap.add_argument("--meta", default="metadata.csv")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)
classes = sorted(pd.read_csv(a.meta)["label"].unique())
mk = lambda s, aug: DataLoader(ImageDataset(a.data, a.meta, s, classes, aug), batch_size=a.batch, shuffle=aug)
dl = {"train": mk("train", True), "val": mk("val", False), "test": mk("test", False)}


def run_epoch(model, loader, opt=None):
    model.train(opt is not None)
    correct = total = 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            if opt:
                loss = nn.functional.cross_entropy(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
            correct += (out.argmax(1) == y).sum().item(); total += len(y)
    return correct / max(total, 1)


LR = {"scratch": 1e-3, "feature_extraction": 1e-3, "fine_tuning": 1e-4}
summary, curves = [], {}
for mode in ["scratch", "feature_extraction", "fine_tuning"]:
    print(f"\n=== {mode} ({a.model}) ===")
    model = build_model(a.model, len(classes), mode).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR[mode])
    hist, best, t0 = [], -1, time.time()
    for ep in range(1, a.epochs + 1):
        tr, va = run_epoch(model, dl["train"], opt), run_epoch(model, dl["val"])
        hist.append((tr, va)); print(f"ep {ep:02d} train={tr:.3f} val={va:.3f}")
        if va > best:
            best = va
            torch.save({"state": model.state_dict(), "classes": classes, "arch": a.model, "mode": mode},
                       f"results/model_{mode}.pt")
    ckpt = torch.load(f"results/model_{mode}.pt", map_location=dev)
    model.load_state_dict(ckpt["state"])
    te = run_epoch(model, dl["test"])
    summary.append(dict(mode=mode, model=a.model, trainable_params=sum(p.numel() for p in params),
                        best_val_acc=round(best, 4), test_acc=round(te, 4),
                        train_time_s=round(time.time() - t0, 1)))
    curves[mode] = hist

pd.DataFrame(summary).to_csv("results/hasil_3_mode.csv", index=False)
print("\n", pd.DataFrame(summary).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for mode, h in curves.items():
    ax[0].plot(range(1, a.epochs + 1), [x[0] for x in h], label=mode)
    ax[1].plot(range(1, a.epochs + 1), [x[1] for x in h], label=mode)
for t, axis in zip(["Akurasi train", "Akurasi validasi"], ax):
    axis.set_title(t); axis.set_xlabel("Epoch"); axis.grid(alpha=.3)
ax[0].set_ylabel("Akurasi"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/akurasi_per_epoch.png", dpi=150)
