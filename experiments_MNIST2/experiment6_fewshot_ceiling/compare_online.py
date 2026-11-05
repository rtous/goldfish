"""Experiment 6, part 1 comparisons (no training; reads saved results):
1. Online gain (online - frozen, pp) at burst positions 11-50, 51-200 and 201-1000, from the
   per-position curves of Meta-SGD (experiment 3) and dual_shared (5b), next to the oracle.
2. Equivalent width: the width of a frozen mixed model with the same accuracy, interpolated
   linearly between the mixed models of width 4, 8 (offline.py), 12, 16, 20, 24 and 40 (5d).
   A method's accuracy here is the mixed model's test accuracy plus its gain on the stream, so
   streams whose frozen accuracy differs from the test mean (short: 73.25 vs 74.05) compare fairly.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment6_fewshot_ceiling/compare_online.py
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WINDOWS = {"11-50": (11, 50), "51-200": (51, 200), "201-1000": (201, 1000)}
CURVES = [("Meta-SGD (3)", "experiment3_metaSGD", "metasgd", "short", ("11-50", "51-200")),
          ("Meta-SGD (3)", "experiment3_metaSGD", "metasgd", "L1000", ("11-50", "51-200", "201-1000")),
          ("dual_shared (5b)", "experiment5b", "dual_shared", "L1000", ("11-50", "51-200", "201-1000"))]
EXPERIMENTS = ["experiment1", "experiment2", "experiment3_metaSGD", "experiment4_surpriseGated",
               "experiment5b", "experiment5c"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mixed_widths():
    """{width: mixed test accuracy (%)}."""
    acc = {4: 64.78, 8: 74.05}
    for row in read(os.path.join(ROOT, "experiment5d_lowdim", "results.csv")):
        if row["matched_H"]:
            acc[int(row["matched_H"])] = float(row["matched_acc"])
    return dict(sorted(acc.items()))


def width_for(acc, widths):
    w, a = np.array(list(widths)), np.array(list(widths.values()))
    if acc < a[0]:
        return "< %d" % w[0]
    if acc > a[-1]:
        return "> %d" % w[-1]
    return "%.1f" % np.interp(acc, a, w)


def main():
    print("Online gain at burst positions (pp):")
    for h in (8, 4):
        for label, exp, config, stream, windows in CURVES:
            d = np.load(os.path.join(ROOT, exp, "curves", "%s_h%d_%s.npz" % (config, h, stream)))
            gains = []
            for wname in windows:
                lo, hi = WINDOWS[wname]
                m = (d["position"] >= lo) & (d["position"] <= hi)
                gains.append("%s: %+.2f" % (wname, (d["online"][m].mean() - d["frozen"][m].mean()) * 100))
            print("  h=%d %-17s %-6s %s" % (h, label, stream, "  ".join(gains)))

    widths = mixed_widths()
    print("\nMixed models:", ", ".join("%d: %.2f" % kv for kv in widths.items()))
    print("\nEquivalent width of each experiment's best config (by gain):")
    print("| experiment | config | h | stream | gain (pp) | accuracy (%) | equivalent width |")
    print("|---|---|---|---|---|---|---|")
    for exp in EXPERIMENTS:
        rows = read(os.path.join(ROOT, exp, "results.csv"))
        for h in (8, 4):
            for stream in ("short", "L1000"):
                cand = [r for r in rows if int(r["h"]) == h and r["stream"] == stream]
                best = max(cand, key=lambda r: float(r["gain"]))
                acc = widths[h] + float(best["gain"])
                print("| %s | %s | %d | %s | %+.2f | %.2f | %s |" % (exp, best["config"], h, stream,
                                                                    float(best["gain"]), acc, width_for(acc, widths)))
    print("\nEquivalent width of the part 1 oracle (all parameters, best cell):")
    for row in read(os.path.join(HERE, "results.csv")):
        if row["part"] == "1" and row["measure"] == "oracle_all":
            print("  h=%s N=%s acc %s -> width %s" % (row["budget"], row["N"], row["acc"],
                                                     width_for(float(row["acc"]), widths)))


if __name__ == "__main__":
    main()
