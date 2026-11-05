"""Figures for the paper, drawn from the saved results in ../experiments_LLM2_cache/ (no runs).

Usage (from the repo root or from paper/):
    venv/bin/python paper/make_figures.py
Writes paper/figures/*.pdf.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments_LLM2_cache")
FIG = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA, INK, MUTED, GRID, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
BIN_LABELS = ["same\nchunk", "earlier\n< 1k", "earlier\n1k-10k", "earlier\n10k-100k", "earlier\n$\\geq$ 100k",
              "never\nseen"]
BINS = ["same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
        "earlier chunk, >= 100k", "never seen"]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42})


def read(path):
    with open(os.path.join(EXP, path)) as f:
        return list(csv.DictReader(f))


def lr_text(v):
    """1e-4 style: 0.0003 -> "3e-4"."""
    mantissa, exponent = ("%.0e" % float(v)).split("e")
    return "%se%d" % (mantissa, int(exponent))


def zero(ax):
    ax.axhline(0, color=MUTED, lw=0.8)


def distance_and_position():
    """Test 1 (gain by distance and by position in the book), with test 3-4's runs on the same bins."""
    rows = {r["bin"]: r for r in read("test4_checks/results.csv")}
    pos = read("test1_distance/position.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(BINS))
    for key, label, color, ls in (("gain_online", "online, in order", BLUE, "-"),
                                  ("gain_shuffled", "online, chunks shuffled", BLUE, "--"),
                                  ("gain_unigram", "unigram cache (frozen model)", INK, ":"),
                                  ("gain_online_cache", "online + unigram cache", INK, "-")):
        a.plot(x, [float(rows[k][key]) for k in BINS], color=color, ls=ls, lw=1.6, marker="o", ms=3, label=label)
    zero(a)
    a.set_xticks(x, BIN_LABELS)
    a.set_ylabel("gain over frozen (nats/token)")
    a.set_title("(a) by distance to the previous occurrence", fontsize=8, loc="left", color=INK)
    a.legend(loc="upper right")
    xs = np.arange(len(pos)) + 0.5
    g, se = np.array([float(r["gain"]) for r in pos]), np.array([float(r["se"]) for r in pos])
    b.fill_between(xs, g - se, g + se, color=BLUE, alpha=0.18, lw=0)
    b.plot(xs, g, color=BLUE, lw=1.6, marker="o", ms=3)
    zero(b)
    b.set_xlim(0, len(pos))
    b.set_xlabel("position in the book (thousands of tokens)")
    b.set_title("(b) by position in the book (online, in order)", fontsize=8, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "distance_position.pdf"))


def eviction():
    """Test 2 and test 3: memory of A against the interlude, mean of the 3 A books."""
    rows = read("test3_baselines/eviction.csv")
    interludes = [0, 10000, 100000]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(interludes))
    for key, label, color in (("online", "online, all parameters", BLUE), ("no_output", "online, all but output", AQUA),
                              ("output", "online, output only (lr 1e-3)", ORANGE), ("unigram", "unigram cache", INK)):
        m = [np.mean([float(r["memory"]) for r in rows if r["method"] == key and int(r["I"]) == i])
             for i in interludes]
        ax.plot(x, m, color=color, lw=1.6, marker="o", ms=3, label=label)
    zero(ax)
    ax.set_xticks(x, ["0", "10k", "100k"])
    ax.set_xlabel("interlude between A1 and A2 (tokens)")
    ax.set_ylabel("memory of A (nats/token)")
    ax.set_ylim(top=0.056)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eviction.pdf"))


def layers_and_size():
    """Test 6: single-layer MLP runs (Pythia-14M) and locality against model size."""
    rows = read("test6/results.csv")

    def row(name, book):
        return next(r for r in rows if r["bin"] == name and r["book"] == book)

    near, mid, overall = row("earlier chunk, < 1k", "all"), row("earlier chunk, 1k-10k", "same book"), \
        row("all tokens", "all")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(1, 7)
    with open(os.path.join(EXP, "test6", "runtime.txt")) as f:  # chosen lr per run, parameter counts
        lines = f.read().splitlines()
    lr = json.loads(lines[0].split(" ", 2)[2])
    n_params = json.loads(lines[1].split(" ", 1)[1])
    lrs = [lr_text(lr["mlp_l%d" % i]) for i in x]
    for r, label, color in ((near, "earlier chunk, < 1k", BLUE), (mid, "earlier chunk, 1k-10k, same book", ORANGE),
                            (overall, "all tokens", INK)):
        a.plot(x, [float(r["mlp_l%d" % i]) for i in x], color=color, lw=1.6, marker="o", ms=3, label=label)
    zero(a)
    a.set_xticks(x, ["%d\nlr %s" % (i, lr) for i, lr in zip(x, lrs)])
    a.set_ylim(top=0.5)
    a.set_ylabel("gain over frozen (nats/token)")
    a.set_title("(a) Pythia-14M, one layer's MLP updated", fontsize=8, loc="left", color=INK)
    a.legend(loc="upper right")
    sizes = [n_params[s] / 1e6 for s in ("14m", "31m", "70m")]
    for r, key, label, color in ((near, "locality", "locality, earlier chunk < 1k", BLUE),
                                 (mid, "locality", "locality, 1k-10k, same book", ORANGE),
                                 (overall, "online", "overall gain, in order", INK)):
        b.plot(sizes, [float(r[key + s]) for s in ("14", "31", "70")], color=color, lw=1.6, marker="o", ms=3,
               label=label)
    zero(b)
    b.set_xscale("log")
    lr14 = float(min((r for r in read("lr_check.csv") if r["lr"] != "0"), key=lambda r: float(r["loss"]))["lr"])
    b.set_xticks(sizes, ["%s\nlr %s" % (n, lr_text(v)) for n, v in zip(("14M", "31M", "70M"),
                                                                      (lr14, lr["31m"], lr["70m"]))])
    b.minorticks_off()
    b.set_ylabel("nats/token")
    b.set_title("(b) full online learning, by model size", fontsize=8, loc="left", color=INK)
    b.set_ylim(top=0.4)
    b.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "layers_size.pdf"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    distance_and_position()
    eviction()
    layers_and_size()
