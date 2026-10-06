#!/usr/bin/env python3
"""Extract frozen backbone features for the whole FAW dataset (one GPU pass per
backbone). Frozen-feature probing is the standard DINOv2-style eval protocol and
makes repeated k-fold + significance testing tractable (heads train in seconds).

For each backbone -> features_<tag>.npz  {feat: (N,D), y: (N,), paths: (N,)}
Row order matches folds.csv exactly (so cv_eval can index by fold directly).
"""
import argparse, os, numpy as np, pandas as pd, torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
DS = str(Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve())
BACKBONES = {   # tag -> timm name
    "dinov2_vitb":   "vit_base_patch14_reg4_dinov2.lvd142m",
    "convnextv2_b":  "convnextv2_base.fcmae_ft_in22k_in1k",
    "resnet50":      "resnet50.a1_in1k",             # CNN baseline
    "eva02_b":       "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
}

class ImgDS(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(f"{DS}/{self.paths[i]}").convert("RGB")), i

def run(tag, name, paths, bs, workers, dev):
    print(f"[{tag}] {name}", flush=True)
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    print(f"   input cfg: {cfg['input_size']} norm={cfg['mean']}", flush=True)
    dl = DataLoader(ImgDS(paths, tf), batch_size=bs, num_workers=workers,
                    pin_memory=True)
    feats = [None] * len(paths)
    with torch.no_grad(), torch.autocast(dev.type, enabled=(dev.type == "cuda")):
        for x, idx in dl:
            f = model(x.to(dev)).float().cpu().numpy()
            for j, i in zip(f, idx.numpy()): feats[i] = j
    feat = np.stack(feats)
    np.savez(f"{DS}/features_{tag}.npz", feat=feat, y=Y, paths=np.array(paths))
    print(f"   saved features_{tag}.npz  {feat.shape}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["dinov2_vitb", "convnextv2_b", "resnet50"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    folds = pd.read_csv(f"{DS}/folds.csv")
    PATHS = folds["image_path"].tolist()
    Y = folds["injury_score"].values
    for tag in a.backbones:
        run(tag, BACKBONES[tag], PATHS, a.bs, a.workers, dev)
    print("DONE_FEATURES", flush=True)
