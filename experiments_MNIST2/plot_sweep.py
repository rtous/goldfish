"""Plots of a burst-length sweep (experiment 2): results against the burst length L.

Configurations are named <family>_L<L>_<h>_<variant> (with a "sequence" field "L<L>"). Figures,
saved to the experiment's plots/ folder as PNG and PDF:
  sweep_gain       online gain on bursts (bursts - frozen, pp) against L (log scale), one panel per
                   family and h, one line per variant; dashed: the per-context ceiling
                   (oracle - frozen)
  sweep_locality   the same for the locality effect (bursts - random)
  burst_curves     within-burst learning curves at one L (default the longest): online accuracy
                   against position in the burst, averaged over the bursts and smoothed (moving
                   average), for frozen, sgd and the best fast variant (the best mean gain on bursts
                   over h = 8, 4, 2 at that L), at h = 8 and 4. These runs are redone online (they
                   are deterministic, so they match results.csv), since results.csv keeps no
                   per-image results.

Usage:
    python3 plot_sweep.py experiment2
    python3 plot_sweep.py 2 --sizes 32 8 4 2 --curves-length 5000 --window 200
"""
import argparse
import contextlib
import csv
import io
import os
import re

import matplotlib
import numpy as np

from mnist_mlp import load_configurations, set_experiment
from plot_results import MUTED, OFFLINE_COLOR, SERIES_COLORS, TEXT, line

VARIANTS = ("sgd", "prior", "fastbias", "fastout", "fasthidden", "fastall")
NAME = re.compile(r"^(?P<family>[a-z]+)_L(?P<L>\d+)_(?P<h>\d+)_(?P<variant>[a-z]+)$")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read(exp_dir):
    """{(family, L, h, variant): row}."""
    out = {}
    with open(os.path.join(exp_dir, "results.csv")) as f:
        for r in csv.DictReader(f):
            m = NAME.match(r["configuration"])
            if m:
                out[(m["family"], int(m["L"]), int(m["h"]), m["variant"])] = r
    return out


def style(ax):
    ax.set_xscale("log")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)


def sweep_figure(plt, rows, sizes, metric, title):
    families = [f for f in ("rotated", "permuted") if any(k[0] == f for k in rows)]
    lengths = sorted({k[1] for k in rows})
    fig, axes = plt.subplots(len(families), len(sizes), figsize=(3.3 * len(sizes), 3.0 * len(families)),
                             squeeze=False, sharex=True)
    for i, fam in enumerate(families):
        for j, h in enumerate(sizes):
            ax = axes[i, j]
            present = [v for v in VARIANTS if any(key[3] == v for key in rows)]
            for k, v in enumerate(present):
                pts = [(L, metric(rows[(fam, L, h, v)], rows[(fam, L, h, "frozen")]))
                       for L in lengths if (fam, L, h, v) in rows]
                pts = [(L, y) for L, y in pts if y is not None]
                if pts:
                    line(ax, *zip(*pts), SERIES_COLORS[k], v)
            if metric is gain:
                ceil = [number(rows[(fam, L, h, "frozen")]["oracle_accuracy"]) -
                        number(rows[(fam, L, h, "frozen")]["offline_accuracy"]) for L in lengths]
                ax.axhline(np.mean(ceil), color=OFFLINE_COLOR, linestyle="--", linewidth=1.5,
                           label="per-context ceiling")
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.set_title("%s, h = %d" % (fam, h), fontsize=9, color=TEXT)
            style(ax)
            if i == len(families) - 1:
                ax.set_xlabel("burst length L (log scale)", color=MUTED, fontsize=8)
            if j == 0:
                ax.set_ylabel("pp", color=MUTED, fontsize=8)
            ax.set_xticks(lengths)
            ax.set_xticklabels([str(L) for L in lengths], fontsize=7)
            ax.minorticks_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(title, y=0.995, fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def gain(r, frozen):
    a, b = number(r["bursts_accuracy"]), number(frozen["offline_accuracy"])
    return None if a is None or b is None else a - b


def locality(r, frozen):
    a, b = number(r["bursts_accuracy"]), number(r["random_accuracy"])
    return None if a is None or b is None else a - b


def best_fast(rows, family, length):
    fast = [v for v in ("fastbias", "fastout", "fasthidden", "fastall") if (family, length, 8, v) in rows]
    means = {v: np.mean([gain(rows[(family, length, h, v)], rows[(family, length, h, "frozen")])
                         for h in (8, 4, 2)]) for v in fast}
    return max(means, key=means.get)


def curves_figure(plt, rows, length, window):
    import torch
    from online_mlp import Stream, run
    families = [f for f in ("rotated", "permuted") if (f, length, 8, "frozen") in rows]
    sizes = (8, 4)
    fig, axes = plt.subplots(len(families), len(sizes), figsize=(5.2 * len(sizes), 3.2 * len(families)),
                             squeeze=False, sharex=True)
    device = torch.device("cpu")
    for i, fam in enumerate(families):
        fast = best_fast(rows, fam, length)
        stream = None
        for j, h in enumerate(sizes):
            ax = axes[i, j]
            for k, (variant, color) in enumerate((("frozen", OFFLINE_COLOR), ("sgd", SERIES_COLORS[0]),
                                                  (fast, SERIES_COLORS[4]))):
                config = load_configurations(["%s_L%d_%d_%s" % (fam, length, h, variant)])[0]
                if stream is None:
                    stream = Stream(config, "bursts", device)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run(config, device, stream)
                correct = result["correct"].reshape(-1, length).mean(axis=0) * 100  # mean over bursts
                smooth = np.convolve(correct, np.ones(window) / window, mode="valid")
                x = np.arange(len(smooth)) + window // 2 + 1
                ax.plot(x, smooth, color=color, linewidth=1.8 if variant != "frozen" else 1.2,
                        linestyle="--" if variant == "frozen" else "-", label=variant)
                print("%s h=%d %s: mean %.2f%%" % (fam, h, variant, correct.mean()))
            oracle = number(rows[(fam, length, h, "frozen")]["oracle_accuracy"])
            ax.axhline(oracle, color=MUTED, linestyle=":", linewidth=1.2, label="per-context ceiling")
            ax.set_title("%s, h = %d, L = %d (best fast: %s)" % (fam, h, length, fast), fontsize=9, color=TEXT)
            ax.grid(True, color="#e4e3df", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)
            ax.set_ylabel("accuracy (%%), moving average of %d" % window, color=MUTED, fontsize=8)
            if i == len(families) - 1:
                ax.set_xlabel("position in the burst", color=MUTED, fontsize=8)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 4, 2])
    parser.add_argument("--curves-length", type=int, help="L of the within-burst curves (default: longest)")
    parser.add_argument("--window", type=int, default=200, help="moving-average window of the curves")
    parser.add_argument("--no-curves", action="store_true")
    args = parser.parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exp_dir = set_experiment(args.experiment)
    rows = read(exp_dir)
    plots = os.path.join(exp_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    figures = [("sweep_gain", sweep_figure(plt, rows, args.sizes, gain,
                                           "online gain on bursts (bursts - frozen) against burst length")),
               ("sweep_locality", sweep_figure(plt, rows, args.sizes, locality,
                                               "locality effect (bursts - random) against burst length"))]
    if not args.no_curves:
        length = args.curves_length or max(k[1] for k in rows)
        figures.append(("burst_curves", curves_figure(plt, rows, length, args.window)))
    for name, fig in figures:
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(plots, "%s.%s" % (name, ext)), dpi=150)
        print("Saved", os.path.join(plots, name + ".{png,pdf}"))


if __name__ == "__main__":
    main()
