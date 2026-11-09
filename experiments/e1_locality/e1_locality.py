"""E1 (claims C1-C4): how far back does the online memory reach, and what is it?

Runs (registry, runs.py): Pythia-14M frozen and plain SGD on all parameters at its val-pick rate, on
the main set (25 books x 78 chunks), in book order (K = 1) and with the same chunks shuffled.
Baseline (no training): the frozen model mixed with a unigram cache (lib/baselines.py, lam 0.05,
half-life 1k tokens).

Tables (lib/metrics.py bins, 1000 most frequent ids excluded from the bins; SE over the 25 books):
    distance.csv   per distance bin (and same-book / earlier-book split of the earlier-chunk bins):
                   gain in order, gain shuffled, locality (in order - shuffled, paired), unigram cache gain
    position.csv   gain by position in the book (1k-token bins), books 1-24 (the books that follow a
                   switch), in order, shuffled and unigram cache
Usage (from experiments/):
    ../venv/bin/python e1_locality/e1_locality.py run      # the registry runs it needs, if missing
    ../venv/bin/python e1_locality/e1_locality.py report
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from baselines import unigram_losses  # noqa: E402
from data import chunk_set  # noqa: E402
from metrics import bin_masks, position_bins, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def keys():
    return {"frozen": runs.frozen_key("14m", "main_K1"), "in order": runs.find("14m", "sgd", "main_K1"),
            "shuffled": runs.find("14m", "sgd", "main_shuffled")}


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def run():
    for k in keys().values():
        if not os.path.exists(runs.path_of(k)):
            runs.run(next(s for s in runs.registry() if s["key"] == k))


def report():
    from plots import plt, style, BLUE, INK, MUTED
    k = keys()
    chunks = chunk_set("main")
    book, masks = bin_masks(chunks)
    frozen = runs.load(k["frozen"])["loss"].reshape(-1)
    order, shuf = runs.load(k["in order"])["loss"].reshape(-1), runs.load(k["shuffled"])["loss"].reshape(-1)
    uni = unigram_losses(frozen.astype(np.float32), chunks.reshape(-1)).astype(np.float64)
    g = {"in_order": frozen - order, "shuffled": frozen - shuf, "unigram": frozen - uni}
    rows = []
    for name, m in masks.items():
        m = m & np.isfinite(frozen)
        r = {"bin": name, "tokens": int(m.sum()), "frozen": "%.4f" % frozen[m].mean()}
        for key, x in list(g.items()) + [("locality", g["in_order"] - g["shuffled"]),
                                         ("in_order_minus_unigram", g["in_order"] - g["unigram"])]:
            s = summary(x, book, m)
            r.update({"gain_" + key if key in g else key: "%.4f" % s["mean"],
                      ("gain_" + key if key in g else key) + "_se": "%.4f" % s["se"],
                      ("gain_" + key if key in g else key) + "_under_2se": s["under_2se"]})
        rows.append(r)
        print({kk: v for kk, v in r.items() if not kk.endswith("under_2se")})
    write_csv("distance.csv", rows)

    n = chunks.shape[1] * chunks.shape[2]
    pos = {key: position_bins(x, book, n, books=list(range(1, 25))) for key, x in g.items()}
    prow = [{"position_in_book": "%dk-%dk" % (j, j + 1),
             **{kk: v for key in g for kk, v in (("gain_" + key, "%.4f" % pos[key][0][j]),
                                                 ("gain_" + key + "_se", "%.4f" % pos[key][1][j]))}}
            for j in range(len(pos["in_order"][0]))]
    write_csv("position.csv", prow)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for kk in k.values():
            f.write("%s %d s\n" % (kk, runs.seconds(kk)))
        f.write("unigram cache: seconds, no model\n")

    style()
    main_rows = [r for r in rows if r["bin"] in ("same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k",
                                                 "earlier chunk, 10k-100k", "earlier chunk, >= 100k", "never seen")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    x = np.arange(len(main_rows))
    for key, color, ls, label in (("in_order", BLUE, "-", "online, in order (K = 1)"),
                                  ("shuffled", BLUE, "--", "online, shuffled chunks"),
                                  ("unigram", INK, ":", "unigram cache on frozen")):
        y = np.array([float(r["gain_" + key]) for r in main_rows])
        s = np.array([float(r["gain_%s_se" % key]) for r in main_rows])
        ax.errorbar(x, y, yerr=2 * s, color=color, ls=ls, lw=1.8, marker="o", ms=4, capsize=2, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [r["bin"].replace(", ", "\n") for r in main_rows], fontsize=7)
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Gain by distance to the previous occurrence (bars +- 2 SE)", fontsize=8.5, color=INK, loc="left")
    ax = axes[1]
    xs = np.arange(len(prow)) + 0.5
    for key, color, ls, label in (("in_order", BLUE, "-", "in order"), ("shuffled", BLUE, "--", "shuffled"),
                                  ("unigram", INK, ":", "unigram cache")):
        y, s = pos[key]
        ax.fill_between(xs, y - s, y + s, color=color, alpha=0.15, lw=0)
        ax.plot(xs, y, color=color, ls=ls, lw=1.8, marker="o", ms=3, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(0, len(prow))
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Gain by position in the book, books 1-24 (band +- 1 SE)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_distance_position.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
