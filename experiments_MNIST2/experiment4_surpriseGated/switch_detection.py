"""Can a context switch be detected from the loss, and after how many images?

Input: the per-image loss (before each update) and position in the burst of the unreset
Meta-SGD on the short stream, curves/metasgd_h<h>_short.npz (online.py 4 metasgd). For a window
of w images:
    score_t = mean loss over the last w images (including t) - mean loss over the 50 images before
The first 60 images of the stream are skipped. Positives: images at position == w (the window holds
exactly the first w images of the new context). Negatives: images at position > 30.

Reports the AUC (rank formula, Mann-Whitney U, ties averaged) and the true and false positive
rates at the threshold that maximizes TPR - FPR, and writes switch_detection.csv.

Usage (from experiments_MNIST2):
    python experiment4_surpriseGated/switch_detection.py
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = (1, 3, 5, 10)
BEFORE, SKIP, NEGATIVE_FROM = 50, 60, 30


def scores(loss, w):
    """score_t for every t (NaN where the windows do not fit)."""
    c = np.r_[0.0, np.cumsum(loss)]
    t = np.arange(len(loss))
    out = np.full(len(loss), np.nan)
    ok = t - w + 1 - BEFORE >= 0
    t = t[ok]
    recent = (c[t + 1] - c[t + 1 - w]) / w
    before = (c[t + 1 - w] - c[t + 1 - w - BEFORE]) / BEFORE
    out[ok] = recent - before
    return out


def ranks(x):
    """1-based ranks, ties get their average rank."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=r)
    return (sums / counts)[inverse]


def auc(pos, neg):
    r = ranks(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold(pos, neg):
    """TPR and FPR at the threshold (score >= threshold) that maximizes TPR - FPR."""
    pos, neg = np.sort(pos), np.sort(neg)
    thresholds = np.unique(np.r_[pos, neg])
    tpr = 1 - np.searchsorted(pos, thresholds, side="left") / len(pos)
    fpr = 1 - np.searchsorted(neg, thresholds, side="left") / len(neg)
    i = np.argmax(tpr - fpr)
    return tpr[i], fpr[i]


def main():
    rows = []
    for h in (8, 4):
        d = np.load(os.path.join(HERE, "curves", "metasgd_h%d_short.npz" % h))
        loss, position = d["loss"], d["position"]
        keep = np.arange(len(loss)) >= SKIP
        for w in WINDOWS:
            s = scores(loss, w)
            valid = keep & np.isfinite(s)
            pos, neg = s[valid & (position == w)], s[valid & (position > NEGATIVE_FROM)]
            tpr, fpr = best_threshold(pos, neg)
            rows.append({"h": h, "w": w, "auc": "%.3f" % auc(pos, neg), "best_tpr": "%.3f" % tpr,
                         "best_fpr": "%.3f" % fpr})
            print("h=%d w=%2d: AUC %.3f, best TPR %.3f at FPR %.3f  (%d positives, %d negatives)"
                  % (h, w, auc(pos, neg), tpr, fpr, len(pos), len(neg)))
    path = os.path.join(HERE, "switch_detection.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["h", "w", "auc", "best_tpr", "best_fpr"])
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", path)


if __name__ == "__main__":
    main()
