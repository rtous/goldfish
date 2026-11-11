"""Figures for the Pattern Recognition Letters version, drawn from the saved results in ../experiments/ (no runs).

A copy of ../paper/make_figures.py resized for the two-column PRL layout: two-panel figures span the page
(7 in), single-panel ones fit one column (3.45 in), and all text is at least 8 pt, the caption size.

Usage (from the repo root or from paper_PRL/):
    venv/bin/python paper_PRL/make_figures.py
Writes paper_PRL/figures/*.pdf.
"""
import csv
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")
FIG = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA, INK, MUTED, GRID, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
BIN_LABELS = ["same\nchunk", "earlier\n< 1k", "earlier\n1k-10k", "earlier\n10k-100k", "earlier\n$\\geq$ 100k",
              "never\nseen"]
BINS = ["same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
        "earlier chunk, >= 100k", "never seen"]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 3.0, "pdf.fonttype": 42})


def read(path):
    with open(os.path.join(EXP, path)) as f:
        return list(csv.DictReader(f))


def lr_text(v):
    """1e-4 style: 0.0003 -> "3e-4"."""
    mantissa, exponent = ("%.0e" % float(v)).split("e")
    return "%se%d" % (mantissa, int(exponent))


def save(fig, name):
    """Save figures/<name>.pdf with no creation date, so that rebuilding gives identical files."""
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def zero(ax):
    ax.axhline(0, color=MUTED, lw=0.8)


def distance_and_position():
    """E1: gain by distance to the previous occurrence and by position in the book."""
    rows = {r["bin"]: r for r in read("e1_locality/distance.csv")}
    pos = read("e1_locality/position.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(BINS))
    for key, label, color, ls in (("gain_in_order", "dyn. eval., in order", BLUE, "-"),
                                  ("gain_shuffled", "dyn. eval., chunks shuffled", BLUE, "--"),
                                  ("gain_unigram", "unigram cache (frozen model)", INK, ":")):
        a.plot(x, [float(rows[k][key]) for k in BINS], color=color, ls=ls, lw=1.6, marker="o", ms=3, label=label)
    zero(a)
    a.set_xticks(x, BIN_LABELS)
    a.set_ylabel("gain over frozen (nats/token)")
    a.set_title("(a) by distance to the previous occurrence", fontsize=8, loc="left", color=INK)
    a.legend(loc="upper right")
    xs = np.arange(len(pos)) + 0.5
    for key, label, ls in (("gain_in_order", "in order", "-"), ("gain_shuffled", "shuffled", "--")):
        g, se = np.array([float(r[key]) for r in pos]), np.array([float(r[key + "_se"]) for r in pos])
        b.fill_between(xs, g - se, g + se, color=BLUE, alpha=0.18, lw=0)
        b.plot(xs, g, color=BLUE, ls=ls, lw=1.6, marker="o", ms=3, label=label)
    zero(b)
    b.set_xlim(0, len(pos))
    b.set_xlabel("position in the book (thousands of tokens)")
    b.set_title("(b) by position in the book (dyn. eval.)", fontsize=8, loc="left", color=INK)
    b.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "distance_position")


def eviction():
    """E4: memory of A (gain on A2 - gain on a never-read control book's C2, same snapshot) and gain on A2
    against the interlude, mean of 20 books, +- 1 SE."""
    r = [x for x in read("e4_eviction/means.csv") if x["I"] != "pre"]
    I = np.array([int(x["I"]) for x in r], dtype=float)
    xs = np.where(I == 0, 300, I)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, key, label, color, ls in ((a, "memory_A", "memory of A", BLUE, "-"),
                                      (a, "memory_rare_in_A1", "rare tokens whose id is in A1", ORANGE, "-"),
                                      (b, "gain_A2", "gain on A2", BLUE, "-"),
                                      (b, "gain_C2", "gain on the control C2", BLUE, "--")):
        y, s_ = np.array([float(x[key]) for x in r]), np.array([float(x[key + "_se"]) for x in r])
        ax.fill_between(xs, y - s_, y + s_, color=color, alpha=0.15, lw=0)
        ax.plot(xs, y, color=color, ls=ls, lw=1.6, marker="o", ms=3, label=label)
    for ax, title in ((a, "(a) memory of A = gain on A2 $-$ gain on C2"), (b, "(b) gain over the pretrained model")):
        zero(ax)
        ax.set_xscale("log")
        ax.set_xticks([300, 1024, 3072, 9984, 29952, 100096, 300032], ["0", "1k", "3k", "10k", "30k", "100k", "300k"])
        ax.minorticks_off()
        ax.set_xlabel("interlude after A1 (tokens of other books)")
        ax.set_title(title, fontsize=8, loc="left")
        ax.legend(loc="upper right")
    a.set_ylabel("nats/token")
    fig.tight_layout()
    save(fig, "eviction")


PARAMS = {"14m": 14.07, "31m": 30.49, "70m": 70.43}  # millions


def rules_size():
    """E7: locality within 1k tokens and exchange rate against size, per update rule."""
    rows = read("e7_update_rules/results.csv")
    ex = read("e7_update_rules/exchange.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for rule, label, color in (("sgd", "plain SGD", INK), ("adam", "Adam", BLUE), ("nsgd", "normalized SGD", ORANGE)):
        r = [x for x in rows if x["rule"] == rule and x["bin"] == "earlier chunk < 1k"]
        x = [PARAMS[v["size"]] for v in r]
        y, s_ = np.array([float(v["locality"]) for v in r]), np.array([float(v["locality_se"]) for v in r])
        a.fill_between(x, y - s_, y + s_, color=color, alpha=0.15, lw=0)
        a.plot(x, y, color=color, lw=1.6, marker="o", ms=3, label=label)
        e = [v for v in ex if v["rule"] == rule]
        b.plot([PARAMS[v["size"]] for v in e], [float(v["exchange_rate_K1"]) for v in e], color=color, lw=1.6,
               marker="o", ms=3, label=label)
    zero(a)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(list(PARAMS.values()), ["14M", "31M", "70M"])
        ax.minorticks_off()
        ax.set_xlabel("model size (parameters)")
    a.set_ylabel("locality gain, earlier chunk $<$ 1k")
    a.set_title("(a) locality gain within 1k tokens ($\\pm$ 1 SE)", fontsize=8, loc="left")
    b.set_yscale("log")
    b.set_yticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
    b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_ylabel("stream gain /\ncollateral forgetting (WikiText)")
    b.set_title("(b) exchange rate, books in order", fontsize=8, loc="left")
    a.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "rules_size")


def layers():
    """E2: one layer's MLP of Pythia-14M updated (each at its own rate), against all parameters (dotted)."""
    e2 = {(r["subset"], r["bin"]): r for r in read("e2_where/results.csv")}
    fig, a = plt.subplots(figsize=(3.45, 2.7))
    x = np.arange(1, 7)
    lrs = [lr_text(e2["mlp@%d" % (i - 1), "all tokens"]["lr"]) for i in x]
    for b_, label, color in (("earlier chunk, < 1k", "earlier chunk, < 1k", BLUE),
                             ("earlier chunk, 1k-10k, same book", "earlier chunk, 1k-10k, same book", ORANGE),
                             ("all tokens", "all tokens", INK)):
        y = np.array([float(e2["mlp@%d" % (i - 1), b_]["gain"]) for i in x])
        se = np.array([float(e2["mlp@%d" % (i - 1), b_]["gain_se"]) for i in x])
        a.errorbar(x, y, yerr=se, color=color, lw=1.6, marker="o", ms=3, capsize=2, label=label)
        a.axhline(float(e2["all", b_]["gain"]), color=color, lw=0.8, ls=":")
    zero(a)
    a.set_xticks(x, ["%d\n%s" % (i, lr) for i, lr in zip(x, lrs)])
    a.set_ylim(top=0.32)
    a.set_xlabel("layer whose MLP is updated, and its lr")
    a.set_ylabel("gain (nats/token)")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
    fig.tight_layout()
    save(fig, "layers")


def tradeoff():
    """E2 and E7: locality within 1k tokens against WikiText forgetting (K = 1, last half), +- 1 SE on both
    axes. Marker by update rule, colour by size; MLPs only (plain SGD, 14M) as an open marker. Also writes
    tables/tradeoff_points.csv."""
    pts = []
    for r in read("e7_update_rules/results.csv"):
        if r["bin"] != "earlier chunk < 1k":
            continue
        e = next(x for x in read("e7_update_rules/exchange.csv") if x["size"] == r["size"] and x["rule"] == r["rule"])
        pts.append({"size": r["size"], "rule": r["rule"], "params": "all", "lr": r["lr"], "locality_1k": r["locality"],
                    "locality_1k_se": r["locality_se"], "wiki_forgetting": e["wiki_forgetting_K1"],
                    "wiki_forgetting_se": e["wiki_forgetting_K1_se"]})
    mvf = {(r["quantity"], r["stream"]): r for r in read("e2_where/mlp_vs_full.csv")}
    loc, wiki = mvf["locality", "earlier chunk, < 1k"], mvf["WikiText forgetting, last half", "K = 1"]
    pts.append({"size": "14m", "rule": "sgd", "params": "mlp", "lr": loc["lr_mlp"], "locality_1k": loc["mlp"],
                "locality_1k_se": loc["mlp_se"], "wiki_forgetting": wiki["mlp"], "wiki_forgetting_se": wiki["mlp_se"]})
    with open(os.path.join(HERE, "tables", "tradeoff_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0]))
        w.writeheader()
        w.writerows(pts)
    color = {"14m": BLUE, "31m": ORANGE, "70m": INK}
    marker = {"sgd": "o", "adam": "s", "nsgd": "^"}
    names = {"sgd": "plain SGD", "adam": "Adam", "nsgd": "normalized SGD"}
    fig, ax = plt.subplots(figsize=(3.45, 3.4))
    for p in pts:
        mlp = p["params"] == "mlp"
        ax.errorbar(float(p["locality_1k"]), float(p["wiki_forgetting"]), xerr=float(p["locality_1k_se"]),
                    yerr=float(p["wiki_forgetting_se"]), color=color[p["size"]], marker=marker[p["rule"]], ms=6,
                    mfc="white" if mlp else color[p["size"]], mew=1.4, lw=0.9, capsize=2, ls="none")
    handles = [plt.Line2D([], [], color=color[s_], marker="o", ls="none", label=s_.upper()) for s_ in color]
    handles += [plt.Line2D([], [], color=MUTED, marker=marker[r_], ls="none", label=names[r_]) for r_ in marker]
    handles.append(plt.Line2D([], [], color=BLUE, marker="o", mfc="white", ls="none", label="14M plain SGD, MLPs only"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8, handlelength=1.2,
              columnspacing=1.0)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("locality gain, earlier chunk $<$ 1k (nats/token)")
    ax.set_ylabel("forgetting on WikiText (nats/token)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.tight_layout()
    save(fig, "tradeoff")


def locality_levels():
    """E3: gain over frozen 14M (plain SGD, all parameters) against the locality of the stream, +- 1 SE."""
    rows = {r["stream"]: r for r in read("e3_locality_levels/results.csv")}
    streams = ["long", "main_K1", "main_K5", "main_K25", "main_shuffled"]
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    x = np.arange(len(streams))
    g = np.array([float(rows[s]["gain"]) for s in streams])
    se = np.array([float(rows[s]["gain_se"]) for s in streams])
    ax.fill_between(x, g - se, g + se, color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, g, color=BLUE, lw=1.6, marker="o", ms=3, label="all parameters, lr %s" % lr_text(rows["main_K1"]["lr"]))
    zero(ax)
    ax.set_xticks(x, ["long\n5 x 100k", "K = 1", "K = 5", "K = 25", "shuf-\nfled"])
    ax.set_xlabel("stream, from strongest to weakest locality")
    ax.set_ylabel("gain (nats/token)")
    fig.tight_layout()
    save(fig, "locality_levels")


def very_long():
    """E6: loss by position in 200k-token books, mean of 4 books, +- 1 SE over books."""
    pos = read("e6_long_books/position.csv")
    xs = np.arange(len(pos)) * 10 + 5
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for key, label, color, ls, band in (("frozen14m", "frozen 14M", INK, "-", False),
                                        ("online14m", "14M dyn. eval., lr 1e-4", BLUE, "-", True),
                                        ("frozen31m", "frozen 31M", INK, "--", False),
                                        ("online31m", "31M dyn. eval., lr 3e-4", AQUA, "-", True),
                                        ("frozen70m", "frozen 70M", INK, ":", False)):
        y = np.array([float(r[key]) for r in pos])
        if band:
            se = np.array([float(r[key + "_se"]) for r in pos])
            ax.fill_between(xs, y - se, y + se, color=color, alpha=0.13, lw=0)
        ax.plot(xs, y, color=color, ls=ls, lw=1.4, marker="o" if band else None, ms=2.5, label=label)
    ax.set_xlim(0, len(pos) * 10)
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("loss (nats/token)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    save(fig, "very_long")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    distance_and_position()
    eviction()
    layers()
    tradeoff()
    locality_levels()
