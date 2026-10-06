#!/usr/bin/env python3
"""Repeated stratified-group k-fold evaluation on frozen features.

Heads:
  logreg   - balanced multinomial logistic regression (standard linear probe)
  ordinal  - Frank & Hall ordinal decomposition (K-1 cumulative binary probes) -> ordinal-aware
  majority - predict train majority class (baseline for significance vs chance)

Ordinal metrics per (backbone, head, rep, fold): accuracy, macro-F1, MAE,
off-by-one accuracy, QWK (quadratic-weighted kappa). Plain accuracy is reported
but QWK/MAE are the headline for an ordered severity scale.

Outputs:
  per_fold_metrics.csv                         (one row per backbone*head*rep*fold)
  oof_<backbone>_<head>.csv                    (out-of-fold predictions for bootstrap CI)
"""
import argparse, glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, cohen_kappa_score

from pathlib import Path
DS = str(Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve())
K = None  # inferred from folds.csv (number of distinct fold ids)

def metrics(yt, yp):
    d = np.abs(yt - yp)
    return dict(accuracy=accuracy_score(yt, yp),
                macro_f1=f1_score(yt, yp, average="macro"),
                mae=mean_absolute_error(yt, yp),
                off_by_one=np.mean(d <= 1),
                within1=np.mean(d <= 1), within2=np.mean(d <= 2),
                qwk=cohen_kappa_score(yt, yp, labels=np.arange(1,10), weights="quadratic"))

def fit_logreg(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))

def fit_ordinal(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ranks = np.sort(np.unique(ytr))                 # ordered classes
    # P(y > rank_k) for k = 0..m-2
    P = np.zeros((Xte.shape[0], len(ranks) - 1))
    for k in range(len(ranks) - 1):
        bin_y = (ytr > ranks[k]).astype(int)
        if bin_y.min() == bin_y.max():              # degenerate fold
            P[:, k] = float(bin_y.max())
        else:
            c = LogisticRegression(class_weight="balanced", max_iter=3000).fit(Xtr, bin_y)
            P[:, k] = c.predict_proba(Xte)[:, 1]
    # class probs from cumulative (Frank & Hall)
    probs = np.zeros((Xte.shape[0], len(ranks)))
    probs[:, 0] = 1 - P[:, 0]
    for i in range(1, len(ranks) - 1): probs[:, i] = P[:, i - 1] - P[:, i]
    probs[:, -1] = P[:, -1]
    return ranks[np.argmax(np.clip(probs, 0, None), axis=1)]

def run_head(feat, y, folds, backbone, head, rows, oof_records):
    for rep in [c for c in folds.columns if c.startswith("fold_rep")]:
        fc = folds[rep].values
        oof = np.full(len(y), -1)
        for fold in range(K):
            te = fc == fold; tr = ~te
            if head == "majority":
                maj = np.bincount(y[tr]).argmax(); pred = np.full(te.sum(), maj)
            elif head == "logreg":
                pred = fit_logreg(feat[tr], y[tr], feat[te])
            else:
                pred = fit_ordinal(feat[tr], y[tr], feat[te])
            oof[te] = pred
            m = metrics(y[te], pred)
            rows.append(dict(backbone=backbone, head=head, rep=int(rep[-1]),
                             fold=fold, n_test=int(te.sum()), **m))
        for p, yt, im in zip(oof, y, folds["image_path"] if "image_path" in folds else range(len(y))):
            oof_records.append(dict(backbone=backbone, head=head, rep=int(rep[-1]),
                                    image_path=im, y_true=int(yt), y_pred=int(p)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["majority", "logreg", "ordinal"])
    a = ap.parse_args()
    folds = pd.read_csv(f"{DS}/folds.csv")
    K = int(folds["fold_rep0"].max()) + 1
    print(f"inferred K={K} folds", flush=True)
    y = folds["injury_score"].values
    feat_files = sorted(glob.glob(f"{DS}/features_*.npz"))
    assert feat_files, "no features_*.npz -- run extract_features.py (GPU) first"
    rows, oof_records = [], []
    for ff in feat_files:
        tag = os.path.basename(ff)[len("features_"):-4]
        d = np.load(ff, allow_pickle=True)
        idx = {str(path): i for i, path in enumerate(d["paths"])}
        selection = np.array([idx[path] for path in folds.image_path])
        feat = d["feat"][selection]
        assert np.array_equal(d["y"][selection], y), f"{tag}: label mismatch after path alignment"
        print(f"{tag}: feat {feat.shape}", flush=True)
        for head in a.heads:
            if head == "majority" and tag != os.path.basename(feat_files[0])[len('features_'):-4]:
                continue  # majority independent of backbone -> compute once
            run_head(feat, y, folds, tag, head, rows, oof_records)
    pd.DataFrame(rows).to_csv(f"{DS}/per_fold_metrics.csv", index=False)
    pd.DataFrame(oof_records).to_csv(f"{DS}/oof_predictions.csv", index=False)
    summ = (pd.DataFrame(rows).groupby(["backbone", "head"])
            [["qwk", "mae", "accuracy", "macro_f1", "off_by_one"]]
            .agg(["mean", "std"]).round(4))
    print("\n=== CV summary (mean/std over rep*fold) ===")
    print(summ)
    print("DONE_CV", flush=True)
