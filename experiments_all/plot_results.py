"""Plot results.csv (written by mnist_mlp.py and online_mlp.py) against model size.

Configurations are grouped into series by everything except the model size: model, training_params
and online_params (e.g. one series per online learning rate). Each series is one line, with one
point per configuration, at its size in KB (log scale).

Figures, saved to the experiment's plots/ folder as PNG and PDF:
  accuracy   one panel per series: frozen offline model, online on bursts, online on random
  locality   left:  online - frozen accuracy on bursts (percentage points), one line per series
             right: locality effect = (bursts - frozen) - (random - frozen) = bursts - random
Runs marked "diverged" (or missing) are left out of the lines.

Several experiments can be given to compare them in the same figures; their series are then
prefixed with the experiment name and the figures go to plots/compare_* (next to this script).

Usage:
    python3 plot_results.py experiment2                    # experiment2/plots/
    python3 plot_results.py 1 2                            # plots/compare_*
    python3 plot_results.py experiment2 --show             # also open the figures in a window
"""
import argparse
import csv
import json
import os

import matplotlib

from mnist_mlp import HERE, experiment_dir, list_experiments

# Categorical slots in fixed order (CVD-checked for adjacent lines), and recessive ink.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OFFLINE_COLOR = "#8a8985"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ABBREV = {"hidden_size": "h", "learning_rate": "lr", "width": "c", "label_smoothing": "ls",
          "epochs": "ep", "batch_size": "bs", "optimizer": "", "bn_mode": "bn",
          "label_prior": "prior"}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None  # empty or "diverged"


def read_rows(folder):
    path = os.path.join(folder, "results.csv")
    if not os.path.exists(path):
        raise SystemExit("No %s: run mnist_mlp.py and online_mlp.py first." % path)
    with open(path) as f:
        return list(csv.DictReader(f))


def short(params_text, skip=()):
    """'{"optimizer": "sgd", "learning_rate": 0.001}' -> 'sgd lr=0.001'."""
    try:
        params = json.loads(params_text)
    except ValueError:
        return params_text
    parts = []
    for k, v in params.items():
        if k in skip:
            continue
        key = ABBREV.get(k, k)
        if k == "fast":  # {"learning_rate", "decay", "layers"} -> fast=1/0.97(1 layer)
            n = len(v["layers"])
            parts.append("fast=%g/%g(%d layer%s)" % (v["learning_rate"], v["decay"], n, "s" * (n != 1)))
            continue
        parts.append(str(v) if key == "" else "%s=%g" % (key, v) if isinstance(v, (int, float)) else "%s=%s" % (key, v))
    return " ".join(parts)


def build_series(folders):
    """[(label, [row, ...] sorted by size)] in the order the series first appear in results.csv."""
    series = {}
    for folder in folders:
        # "experiment2_aggressive" -> "experiment2"
        prefix = os.path.basename(os.path.abspath(folder)).split("_")[0] + ": " if len(folders) > 1 else ""
        for r in read_rows(folder):
            key = (prefix, r["model"], r["training_params"], r["online_params"])
            series.setdefault(key, []).append(r)
    # Name each series by what tells it apart from the others.
    vary_model = len({k[1] for k in series}) > 1
    out = []
    for key, rows in series.items():
        prefix, model, training, online = key
        parts = [model] if vary_model else []
        if len({k[2] for k in series if k[1] == model}) > 1:  # training differs within this model
            parts.append("train " + short(training))
        parts.append("online " + (short(online) or "-"))
        out.append((prefix + ", ".join(parts), sorted(rows, key=lambda r: float(r["size_bytes"]))))
    return out


def points(rows, value):
    """(sizes in KB, values) for the rows where value(row) is a number."""
    xs, ys = [], []
    for r in rows:
        v = value(r)
        if v is not None:
            xs.append(float(r["size_bytes"]) / 1024)
            ys.append(v)
    return xs, ys


def difference(a, b):
    return lambda r: None if number(r[a]) is None or number(r[b]) is None else number(r[a]) - number(r[b])


def style(ax, ylabel):
    ax.set_xscale("log")
    ax.set_xlabel("model size (KB, log scale)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)


def line(ax, xs, ys, color, label, dashed=False):
    ax.plot(xs, ys, color=color, linewidth=2, linestyle="--" if dashed else "-", marker="o",
            markersize=5, markeredgecolor="white", markeredgewidth=1, label=label)


def plot_accuracy(plt, series):
    n = len(series)
    cols = n if n <= 3 else 2 if n == 4 else 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.6 * cols, 7), 3.6 * rows), squeeze=False,
                             sharex=True, sharey=True)
    for ax, (label, rs) in zip(axes.flat, series):
        line(ax, *points(rs, lambda r: number(r["offline_accuracy"])), OFFLINE_COLOR,
             "offline (frozen)", dashed=True)
        line(ax, *points(rs, lambda r: number(r["bursts_accuracy"])), SERIES_COLORS[0], "online, bursts")
        line(ax, *points(rs, lambda r: number(r["random_accuracy"])), SERIES_COLORS[1], "online, random")
        ax.set_title(label, fontsize=10, color=TEXT)
        style(ax, "accuracy (%)")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.6 * rows)))
    return fig


def plot_locality(plt, series):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for i, (label, rs) in enumerate(series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        line(left, *points(rs, difference("bursts_accuracy", "offline_accuracy")), color, label)
        line(right, *points(rs, difference("bursts_accuracy", "random_accuracy")), color, label)
    if len(series) > len(SERIES_COLORS):
        print("Warning: %d series share %d colors; plot fewer at once." % (len(series), len(SERIES_COLORS)))
    left.set_title("online gain on bursts (bursts - frozen)", fontsize=10, color=TEXT)
    right.set_title("locality effect (bursts - random)", fontsize=10, color=TEXT)
    for ax in (left, right):
        ax.axhline(0, color=MUTED, linewidth=1)
        style(ax, "accuracy difference (pp)")
    handles, labels = left.get_legend_handles_labels()
    ncol = 1 if max(len(l) for l in labels) > 40 else min(len(series), 4)
    legend_rows = (len(series) + ncol - 1) // ncol
    fig.set_size_inches(10, 4 + 0.25 * legend_rows)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)
    fig.tight_layout(rect=(0, 0.25 * legend_rows / (4 + 0.25 * legend_rows) + 0.02, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiments", nargs="+",
                        help="one experiment, or several to compare: %s (or their numbers)"
                        % ", ".join(list_experiments()))
    parser.add_argument("--show", action="store_true", help="also show the figures in a window")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folders = []
    for name in args.experiments:
        folder = experiment_dir(name)
        if folder not in folders:
            folders.append(folder)
    series = build_series(folders)
    prefix = "compare_" if len(folders) > 1 else ""
    plots_dir = os.path.join(HERE if len(folders) > 1 else folders[0], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for name, fig in (("accuracy", plot_accuracy(plt, series)), ("locality", plot_locality(plt, series))):
        for ext in ("png", "pdf"):
            path = os.path.join(plots_dir, "%s%s.%s" % (prefix, name, ext))
            fig.savefig(path, dpi=150)
        print("Saved", os.path.join(plots_dir, prefix + name + ".{png,pdf}"))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
