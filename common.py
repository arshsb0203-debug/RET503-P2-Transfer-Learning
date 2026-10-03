"""Preprocessing + dataset + model builder. HARUS dipakai sama persis saat training & deployment."""
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr, K=None, dist=None):
    """undistort (opsional) -> resize 224x224 -> BGR->RGB -> normalisasi. Return tensor CxHxW."""
    if K is not None and dist is not None:
        bgr = cv2.undistort(bgr, K, dist)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV membaca BGR!
    rgb = cv2.resize(rgb, (SIZE, SIZE))
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


class ImageDataset(Dataset):
    def __init__(self, root, metadata_csv, split, classes, augment=False):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root, self.classes, self.augment = root, classes, augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(f"{self.root}/{row['filename']}")
        if img is None:
            raise FileNotFoundError(row["filename"])
        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        return preprocess(img), self.classes.index(row["label"])


def _replace_head(model, name, n):
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n)
        return model.fc
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, n)
    return model.classifier[-1]


def build_model(name, n_classes, mode):
    """mode: scratch | feature_extraction | fine_tuning"""
    ctor = getattr(models, name)
    model = ctor(weights=None if mode == "scratch" else "DEFAULT")
    if mode == "feature_extraction":
        for p in model.parameters():
            p.requires_grad = False
    _replace_head(model, name, n_classes)  # head baru selalu trainable
    return model
