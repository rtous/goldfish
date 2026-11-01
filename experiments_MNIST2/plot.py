"""The same figures for any experiment, from results.csv and curves/:
    plots/summary.png/.pdf             one panel per (stream, h): gain over frozen and locality
                                       effect (pp) for each configuration
    plots/burst_curve_<stream>.png/.pdf  one panel per h: online accuracy by position within the
                                       burst (binned, log x), one line per configuration, frozen level dashed
--compare writes the same figures for the best configuration of each experiment (by mean gain
over h and streams) to experiments_MNIST2/plots/compare.png/.pdf and compare_burst_curve_<stream>.
Each method has one hue (sgd blue, dual orange, metasgd aqua); configurations of the same method
get lighter to darker shades, in configuration order.

Usage:
    python plot.py 1
    python plot.py --compare 1 2 3
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from data import HERE, experiment_dir, load_config

METHOD_COLOR = {"sgd": "#2a78d6", "dual": "#eb6834", "metasgd": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BIN_EDGES = {"short": np.array([1, 6, 11, 21, 51, 101, 151, 201]), "L1000": np.array([1, 6, 11, 21, 51, 101, 201, 301, 501, 701, 1001])}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def shade(color, t):
    """color mixed with white by t > 0, or with black by -t when t < 0 (0 = the color itself)."""
    return tuple(c + (1 - c) * t if t >= 0 else c * (1 + t) for c in to_rgb(color))


def load(exp_dir):
    config = load_config(exp_dir)
    with open(os.path.join(exp_dir, "results.csv")) as f:
        rows = {(r["config"], int(r["h"]), r["stream"]): r for r in csv.DictReader(f)}
    return config, rows


def experiment_series(exp_dir):
    """(label, color, exp_dir, config name, results rows) for each configuration."""
    config, rows = load(exp_dir)
    series, by_method = [], {}
    for cfg in config["configurations"]:
        by_method.setdefault(cfg["method"], []).append(cfg["name"])
    for cfg in config["configurations"]:
        names = by_method[cfg["method"]]
        t = 0.55 - 0.9 * names.index(cfg["name"]) / (len(names) - 1) if len(names) > 1 else 0
        series.append((cfg["name"], shade(METHOD_COLOR[cfg["method"]], t), exp_dir, cfg["name"], rows))
    return config, series


def value(rows, key, column):
    r = rows.get(key)
    return float(r[column]) if r and r[column] else None


def summary(series, sizes, streams, path):
    fig, axes = plt.subplots(len(streams), len(sizes), figsize=(3.4 * len(sizes), 2.9 * len(streams)), squeeze=False)
    x = np.arange(len(series))
    for i, stream in enumerate(streams):
        for j, h in enumerate(sizes):
            ax = axes[i, j]
            frozen = None
            for k, (label, color, _, name, rows) in enumerate(series):
                gain, loc = value(rows, (name, h, stream), "gain"), value(rows, (name, h, stream), "locality")
                frozen = frozen or value(rows, (name, h, stream), "frozen")
                if gain is None:
                    ax.text(k, 0, "diverged", rotation=90, ha="center", va="bottom", color=MUTED)
                    continue
                ax.bar(k - 0.2, gain, 0.38, color=color, edgecolor="white", linewidth=1)
                if loc is not None:
                    ax.bar(k + 0.2, loc, 0.38, color=shade(color, 0.6), edgecolor=color, hatch="////", linewidth=0.8)
            ax.axhline(0, color=INK, linewidth=0.8)
            ax.grid(True, axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_xticks(x, [s[0] for s in series], rotation=30, ha="right")
            ax.set_title("%s, h = %d (frozen %.2f%%)" % (stream, h, frozen or 0), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("pp")
    fig.legend(handles=[Patch(facecolor=MUTED, label="gain over frozen (bursts - frozen)"),
                        Patch(facecolor=shade(MUTED, 0.6), edgecolor=MUTED, hatch="////",
                              label="locality effect (bursts - shuffled)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def burst_curve(series, sizes, stream, path):
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes) + 1.8, 3.2), squeeze=False)
    edges = BIN_EDGES[stream]
    for ax, h in zip(axes[0], sizes):
        frozen = None
        for label, color, exp_dir, name, _ in series:
            f = os.path.join(exp_dir, "curves", "%s_h%d_%s.npz" % (name, h, stream))
            if not os.path.exists(f):
                continue
            d = np.load(f)
            which = np.digitize(d["position"], edges) - 1
            keep = [b for b in range(len(edges) - 1) if (which == b).any()]
            centers = [np.sqrt(edges[b] * (edges[b + 1] - 1)) for b in keep]
            ax.plot(centers, [d["online"][which == b].mean() * 100 for b in keep], color=color, linewidth=2,
                    marker="o", markersize=3, label=label)
            frozen = d["frozen"].mean() * 100
        if frozen is not None:
            ax.axhline(frozen, color=INK, linewidth=1, linestyle="--", label="frozen mixed model")
        ax.set_xscale("log")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_title("%s, h = %d" % (stream, h), fontsize=9, color=INK)
        ax.set_xlabel("position within the burst (log scale)")
        ax.set_ylabel("online accuracy (%)")
    axes[0, -1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save(fig, path)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path + "." + ext, dpi=150)
    plt.close(fig)
    print("Saved", path + ".png/.pdf")


def best(exp_dir):
    """The configuration with the highest mean gain over h and streams (diverged runs count -inf)."""
    config, series = experiment_series(exp_dir)
    def mean_gain(s):
        gains = [value(s[4], (s[3], h, st), "gain") for h in config["sizes"] for st in config["streams"]]
        return -np.inf if None in gains else np.mean(gains)
    s = max(series, key=mean_gain)
    method = next(c["method"] for c in config["configurations"] if c["name"] == s[3])
    return config, ("%s: %s" % (os.path.basename(exp_dir).split("_")[0], s[3]), METHOD_COLOR[method]) + s[2:]


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__.strip())
    if args[0] == "--compare":
        picked = [best(experiment_dir(a)) for a in args[1:]]
        config, series = picked[0][0], [p[1] for p in picked]
        out = os.path.join(HERE, "plots")
        summary(series, config["sizes"], config["streams"], os.path.join(out, "compare"))
        for stream in config["streams"]:
            burst_curve(series, config["sizes"], stream, os.path.join(out, "compare_burst_curve_" + stream))
        return
    exp_dir = experiment_dir(args[0])
    config, series = experiment_series(exp_dir)
    summary(series, config["sizes"], config["streams"], os.path.join(exp_dir, "plots", "summary"))
    for stream in config["streams"]:
        burst_curve(series, config["sizes"], stream, os.path.join(exp_dir, "plots", "burst_curve_" + stream))


if __name__ == "__main__":
    main()
