"""Figures for the paper, drawn from the saved results in ../experiments_LLM2_cache/ (no runs).

Usage (from the repo root or from paper/):
    venv/bin/python paper/make_figures.py
Writes paper/figures/*.pdf.
"""
import csv
import json
import os

import matplotlib
import matplotlib.ticker

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
    """Test 11: memory of A (gain on A2 - gain on a never-read control book's C2, same snapshot) and gain on
    A2 against the interlude, mean of 20 books, +- 1 SE."""
    rows = read("test11_probe_eviction/means.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for setting, label, color in (("full_1e-4", "all, lr 1e-4", BLUE), ("mlp4_3e-3", "MLP 4, lr 3e-3", ORANGE),
                                  ("dual", "slow + fast (MLP 2)", AQUA)):
        r = [x for x in rows if x["setting"] == setting and x["I"] != "pre"]
        I = np.array([int(x["I"]) for x in r], dtype=float)
        xs = np.where(I == 0, 300, I)
        for ax, key in ((a, "memory_A"), (b, "gain_A2")):
            y, s_ = np.array([float(x[key]) for x in r]), np.array([float(x[key + "_se"]) for x in r])
            ax.fill_between(xs, y - s_, y + s_, color=color, alpha=0.15, lw=0)
            ax.plot(xs, y, color=color, lw=1.6, marker="o", ms=3, label=label)
        if setting == "full_1e-4":
            b.plot(xs, [float(x["gain_C2"]) for x in r], color=color, lw=1.2, ls="--", label="all, lr 1e-4: control C2")
    for ax, title in ((a, "(a) memory of A = gain on A2 $-$ gain on C2"), (b, "(b) gain on A2 over the pretrained model")):
        zero(ax)
        ax.set_xscale("log")
        ax.set_xticks([300, 1024, 3072, 9984, 29952, 100096, 300032], ["0", "1k", "3k", "10k", "30k", "100k", "300k"])
        ax.minorticks_off()
        ax.set_xlabel("interlude after A1 (tokens of other books)")
        ax.set_title(title, fontsize=8, loc="left")
    a.set_ylabel("nats/token")
    a.legend(loc="upper right")
    b.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eviction.pdf"))


def rules_size():
    """Test 13: locality within 1k tokens and gain / WikiText forgetting against size, per update rule."""
    rows = read("test13_optimizers/results.csv")
    ex = read("test13_optimizers/exchange.csv")
    params = {"14m": 14.07, "31m": 30.49, "70m": 70.43}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for rule, label, color in (("sgd", "plain SGD", INK), ("adam", "Adam", BLUE), ("nsgd", "normalized SGD", ORANGE)):
        r = [x for x in rows if x["rule"] == rule and x["bin"] == "earlier chunk < 1k"]
        x = [params[v["size"]] for v in r]
        y, s_ = np.array([float(v["locality"]) for v in r]), np.array([float(v["locality_se"]) for v in r])
        a.fill_between(x, y - s_, y + s_, color=color, alpha=0.15, lw=0)
        a.plot(x, y, color=color, lw=1.6, marker="o", ms=3, label=label)
        e = [v for v in ex if v["rule"] == rule]
        b.plot([params[v["size"]] for v in e], [float(v["exchange_rate_K1"]) for v in e], color=color, lw=1.6,
               marker="o", ms=3, label=label)
    zero(a)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(list(params.values()), ["14M", "31M", "70M"])
        ax.minorticks_off()
        ax.set_xlabel("model size (parameters)")
    a.set_ylabel("locality, earlier chunk $<$ 1k")
    a.set_title("(a) locality within 1k tokens ($\\pm$ 1 SE)", fontsize=8, loc="left")
    b.set_yscale("log")
    b.set_yticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
    b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_ylabel("stream gain / WikiText forgetting")
    b.set_title("(b) exchange rate, books in order", fontsize=8, loc="left")
    a.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rules_size.pdf"))


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


def locality_levels():
    """Test 7 (settings across streams) and tests 8 / 10 (dual on K = 1, shuffled, long): gain over frozen 14M."""
    rows = read("test7_aggressive/results.csv")
    streams = ["long", "K1", "K5", "K25", "shuffled"]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    x = np.arange(len(streams))
    for key, label, color, ls in (("full_1e-4", "all parameters, lr 1e-4", BLUE, "-"),
                                  ("mlp4_3e-3", "MLP 4, lr 3e-3", AQUA, "-"), ("mlp4_1e-2", "MLP 4, lr 1e-2", ORANGE, "-"),
                                  ("fast4_d0.99", "MLP 4, lr 1e-2, decay 0.99", ORANGE, "--"),
                                  ("fast4_d0.9", "MLP 4, lr 1e-2, decay 0.9", ORANGE, ":")):
        g = [float(next(r["gain"] for r in rows if r["setting"] == key and r["stream"] == s)) for s in streams]
        ax.plot(x, g, color=color, ls=ls, lw=1.6, marker="o", ms=3, label=label)
    pick8 = json.loads(open(os.path.join(EXP, "test8_dual", "runtime.txt")).readline())["pick_val_K1"]
    r8 = {r["stream"]: float(r["gain_frozen"]) for r in read("test8_dual/results.csv")
          if r["phase"] == "test" and r["config"] == pick8}
    idx = [streams.index(s) for s in ("long", "K1", "shuffled")]
    ax.plot(idx, [r8[s] for s in ("long", "K1", "shuffled")], color=INK, lw=0, marker="D", ms=4,
            label="all parameters + fast part on MLP 4 (test 8)")
    zero(ax)
    ax.set_xticks(x, ["long\n(5 x 100k)", "K = 1", "K = 5", "K = 25", "shuffled"])
    ax.set_xlabel("stream, from strongest to weakest locality")
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "locality_levels.pdf"))


def very_long():
    """Test 10 part C: loss by position in 200k-token books, mean of 4 books, +- 1 SE over books."""
    pos = read("test10_followups/partC_position.csv")
    dual_key = next(k for k in pos[0] if k.startswith("14m dual") and not k.endswith(" se"))
    xs = np.arange(len(pos)) * 10 + 5
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for key, label, color, ls, band in (("frozen14m", "frozen 14M", INK, "-", False),
                                        ("14m full_1e-4", "14M online, lr 1e-4", BLUE, "-", True),
                                        (dual_key, "14M online + fast part (MLP 2)", ORANGE, "-", True),
                                        ("frozen31m", "frozen 31M", INK, "--", False),
                                        ("31m online lr3e-4", "31M online, lr 3e-4", AQUA, "-", True),
                                        ("frozen70m", "frozen 70M", INK, ":", False)):
        y = np.array([float(r[key]) for r in pos])
        if band:
            se = np.array([float(r[key + " se"]) for r in pos])
            ax.fill_between(xs, y - se, y + se, color=color, alpha=0.13, lw=0)
        ax.plot(xs, y, color=color, ls=ls, lw=1.4, marker="o" if band else None, ms=2.5, label=label)
    ax.set_xlim(0, len(pos) * 10)
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("loss (nats/token)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "very_long.pdf"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    distance_and_position()
    eviction()
    rules_size()
    layers_and_size()
    locality_levels()
    very_long()
