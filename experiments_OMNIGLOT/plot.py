"""Figures for an experiment, from results.csv and curves/ (the best configuration of each method
by accuracy on the bursts stream; the frozen model dashed gray):
    plots/burst_curve.png/.pdf   one panel per architecture: accuracy by position in the burst
                                 (binned, log x)
    plots/occurrence.png/.pdf    one panel per architecture: accuracy by occurrence (how many times
                                 the class was already seen in the burst: 0 = first sighting)
Colors: sgd blue, dual orange, metasgd aqua (as in MNIST2), then the controls: label prior yellow,
alphabet mask magenta, output-bias sgd violet, cache green. Each line also has its own marker and
an end label, so identity never rests on color alone.

Usage:
    python plot.py 1
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data import experiment_dir, load_config  # noqa: E402

FAMILIES = [  # (key, label, color, marker); key = method, or "bias_sgd" for sgd on the output bias
    ("sgd", "sgd", "#2a78d6", "o"), ("dual", "dual", "#eb6834", "s"), ("metasgd", "Meta-SGD", "#1baf7a", "D"),
    ("prior", "label prior (2a)", "#eda100", "^"), ("mask", "alphabet mask (2b)", "#e87ba4", "v"),
    ("bias_sgd", "output-bias sgd (2c)", "#6250d6", "P"), ("cache", "cache (3)", "#008300", "X")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EDGES = np.array([1, 4, 11, 21, 51, 101, 201, 301, 441])
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def family(cfg):
    return "bias_sgd" if cfg.get("layers") == "out_bias" else cfg["method"]


def best_per_family(exp_dir, arch):
    config = load_config(exp_dir)
    with open(os.path.join(exp_dir, "results.csv")) as f:
        rows = {(r["config"], r["arch"]): r for r in csv.DictReader(f)}
    best = {}
    for cfg in config["configurations"]:
        r = rows.get((cfg["name"], arch))
        if r and r["bursts"] and (family(cfg) not in best or float(r["bursts"]) > best[family(cfg)][1]):
            best[family(cfg)] = (cfg["name"], float(r["bursts"]))
    return best


def panel(ax, exp_dir, arch, kind):
    best = best_per_family(exp_dir, arch)
    frozen_line = None
    ends = []
    for key, label, color, marker in FAMILIES:
        if key not in best:
            continue
        d = np.load(os.path.join(exp_dir, "curves", "%s_%s.npz" % (best[key][0], arch)))
        if kind == "position":
            xs = [np.sqrt(lo * (hi - 1)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
            bins = [(d["position"] >= lo) & (d["position"] < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        else:
            xs = list(range(8))
            bins = [d["occurrence"] == k for k in xs]
        ys = [d["online"][b].mean() * 100 for b in bins]
        if frozen_line is None:
            frozen_line = [d["frozen"][b].mean() * 100 for b in bins]
            ax.plot(xs, frozen_line, color=MUTED, lw=1.5, ls="--", zorder=1)
            ends.append((frozen_line[-1], "frozen", MUTED))
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5, mec="white", mew=1, zorder=3,
                label="%s (%s)" % (label, best[key][0]))
        ends.append((ys[-1], label, color))
    ends.sort()
    last = -1e9
    for y, label, _ in ends:  # end labels in ink, nudged apart
        y = max(y, last + 2.6)
        last = y
        ax.annotate(label, (xs[-1], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    if kind == "position":
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100, 300])
        ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
        ax.set_xlabel("position in the burst")
    else:
        ax.set_xticks(range(8))
        ax.set_xlabel("times the class was already seen in the burst")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("architecture %s" % arch, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    exp_dir = experiment_dir(sys.argv[1])
    archs = load_config(exp_dir)["archs"]
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    for kind, name in (("position", "burst_curve"), ("occurrence", "occurrence")):
        fig, axes = plt.subplots(1, len(archs), figsize=(5.2 * len(archs), 3.8), sharey=True)
        for ax, arch in zip(np.atleast_1d(axes), archs):
            panel(ax, exp_dir, arch, kind)
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.1, 0.95, 1))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(exp_dir, "plots", "%s.%s" % (name, ext)), dpi=160)
        plt.close(fig)
        print("Wrote %s" % os.path.join(exp_dir, "plots", name + ".png"))


if __name__ == "__main__":
    main()
