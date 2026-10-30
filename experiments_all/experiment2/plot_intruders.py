"""Experiment 2's extra figure: online accuracy on intruders against model size, one line per
variant, with the frozen model as a dashed reference. A second panel shows the regular (burst)
images, for the trade-off. Both on the bursts sequence, from experiment2/results.csv.

Saved to experiment2/plots/intruders.{png,pdf}.

Usage (from experiments_all/):
    python3 experiment2/plot_intruders.py
"""
import os
import sys

import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from plot_results import (MUTED, OFFLINE_COLOR, SERIES_COLORS, TEXT, build_series, line, number,  # noqa: E402
                          points, style)


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = build_series([HERE])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    panels = (("intruders", "intruder"), ("regular (burst) images", "regular"))
    for ax, (title, kind) in zip(axes, panels):
        # The frozen model is the same for every variant: take it from the first series.
        line(ax, *points(series[0][1], lambda r: number(r["offline_%s_accuracy" % kind])),
             OFFLINE_COLOR, "frozen (offline)", dashed=True)
        for i, (label, rows) in enumerate(series):
            line(ax, *points(rows, lambda r: number(r["bursts_%s_accuracy" % kind])),
                 SERIES_COLORS[i % len(SERIES_COLORS)], label)
        ax.set_title("online accuracy on %s, bursts sequence" % title, fontsize=10, color=TEXT)
        style(ax, "accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=MUTED)
    rows = (len(labels) + 1) // 2
    fig.tight_layout(rect=(0, 0.06 * rows, 1, 1))
    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "plots", "intruders." + ext), dpi=150)
    print("Saved", os.path.join(HERE, "plots", "intruders.{png,pdf}"))


if __name__ == "__main__":
    main()
