"""Shared by svd.py and matched.py: sizes, ranks, parameter counts and results.csv merging."""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (8, 4)
RANKS = (0, 1, 2, 4, 8)
MATCHED_RANKS = (0, 1, 2, 4)  # r = 0: the mixed model of width h itself
COLUMNS = ["h", "r", "explained", "ceiling", "total_params", "matched_H", "matched_acc", "margin"]


def total_params(h, r):
    """S1 and the head (a whole 784 -> h -> 10 MLP) plus r directions of one first layer."""
    return 795 * h + 10 + r * 785 * h


def matched_width(h, r):
    """Hidden size H whose plain MLP (795 H + 10 parameters) is closest to total_params(h, r)."""
    return min(range(1, 1000), key=lambda H: abs(795 * H + 10 - total_params(h, r)))


def update_results(values):
    """Merge {(h, r): {column: value}} into results.csv and recompute the margin."""
    path = os.path.join(HERE, "results.csv")
    rows = {(h, r): {"h": h, "r": r, "total_params": total_params(h, r)} for h in SIZES for r in RANKS}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                rows[int(row["h"]), int(row["r"])].update({k: v for k, v in row.items() if v != ""})
    for key, v in values.items():
        rows[key].update(v)
    for row in rows.values():
        if row.get("ceiling") and row.get("matched_acc"):
            row["margin"] = "%.2f" % (float(row["ceiling"]) - float(row["matched_acc"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows[h, r] for h in SIZES for r in RANKS)
