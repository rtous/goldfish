"""E6 (claim C9): on very long books, how much of a bigger model's advantage does online learning buy?

Runs (registry): very long set (the first 4 PG-19 test books with at least 200k tokens, cut to 200k,
781 chunks each, book after book): frozen Pythia-14M / 31M / 70M; 14M and 31M plain SGD on all
parameters at their val-pick rates.

Tables: results.csv (per run: loss, gain over frozen 14M with SE over the 4 books, share of the
frozen 14M - frozen 31M gap closed), pairs.csv (paired per-book differences between runs),
position.csv (loss per 10k-token position bin, mean of 4 books with SE, and the share of the
14M-31M gap closed by 14M online in each bin). Usage (from experiments/):
    ../venv/bin/python e6_long_books/e6_long_books.py run | report
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from metrics import position_bins, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
POS = 10000
LABELS = {"frozen14m": "frozen 14M", "frozen31m": "frozen 31M", "frozen70m": "frozen 70M", "online14m": "14M online",
          "online31m": "31M online"}
PAIRS = [("14M online - frozen 14M", "frozen14m", "online14m"), ("31M online - frozen 31M", "frozen31m", "online31m"),
         ("frozen 31M - 14M online", "online14m", "frozen31m"), ("31M online - 14M online", "online14m", "online31m"),
         ("31M online - frozen 70M", "frozen70m", "online31m")]


def keys():
    k = {"frozen%s" % s: runs.frozen_key(s, "verylong") for s in ("14m", "31m", "70m")}
    k.update(online14m=runs.find("14m", "sgd", "verylong"), online31m=runs.find("31m", "sgd", "verylong"))
    return k


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
    from plots import plt, style, BLUE, INK, AQUA
    k = keys()
    loss = {n: runs.load(v)["loss"] for n, v in k.items()}
    shape = loss["frozen14m"].shape
    book = np.repeat(np.arange(shape[0]), shape[1] * shape[2])
    allm = np.ones(book.shape, bool)
    flat = {n: v.reshape(-1) for n, v in loss.items()}
    f14, f31 = np.nanmean(flat["frozen14m"]), np.nanmean(flat["frozen31m"])
    rows = []
    for n, v in flat.items():
        g = summary(flat["frozen14m"] - v, book, allm)
        rows.append({"run": n, "label": LABELS[n], "key": k[n], "loss": "%.4f" % np.nanmean(v), "gain_over_frozen14m": "%.4f" % g["mean"],
                     "se": "%.4f" % g["se"], "gap31_closed": "%.4f" % ((f14 - np.nanmean(v)) / (f14 - f31))})
    write_csv("results.csv", rows)
    prs = []
    for name, base, other in PAIRS:
        d = summary(flat[base] - flat[other], book, allm)
        prs.append({"comparison": name, "mean": "%.4f" % d["mean"], "se": "%.4f" % d["se"], "books": d["books"],
                    "under_2se": d["under_2se"]})
    write_csv("pairs.csv", prs)
    n = shape[1] * shape[2]
    pb = {nm: position_bins(v, book, n, POS) for nm, v in flat.items()}
    # gap closed per bin: per-book (frozen14 - online14) / (frozen14 - frozen31), then mean over books
    def per_book_bins(x):
        xb = x.reshape(-1, n)
        return np.array([[np.nanmean(r[j * POS:(j + 1) * POS]) for j in range((n + POS - 1) // POS)] for r in xb])
    b14, b31, o14 = per_book_bins(flat["frozen14m"]), per_book_bins(flat["frozen31m"]), per_book_bins(flat["online14m"])
    gap = (b14 - o14).mean(0) / (b14 - b31).mean(0)
    pos = []
    for j in range(len(gap)):
        r = {"position": "%dk-%dk" % (j * POS // 1000, (j + 1) * POS // 1000)}
        for nm in flat:
            r[nm], r[nm + "_se"] = "%.4f" % pb[nm][0][j], "%.4f" % pb[nm][1][j]
        r["gap31_closed_by_online14m"] = "%.3f" % gap[j]
        pos.append(r)
    write_csv("position.csv", pos)
    for r in rows + prs:
        print(r)
    print("gap closed by bin:", " ".join(r["gap31_closed_by_online14m"] for r in pos))
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for v in k.values():
            f.write("%s %d s\n" % (v, runs.seconds(v)))

    style()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    xs = np.arange(len(pos)) * POS / 1000 + POS / 2000
    for nm, color, ls in (("frozen14m", INK, "-"), ("frozen31m", INK, "--"), ("frozen70m", INK, ":"),
                          ("online14m", BLUE, "-"), ("online31m", AQUA, "-")):
        y, s = pb[nm]
        if nm.startswith("online"):
            ax.fill_between(xs, y - s, y + s, color=color, alpha=0.15, lw=0)
        ax.plot(xs, y, color=color, ls=ls, lw=1.6, marker="o" if nm.startswith("online") else None, ms=2.5,
                label=LABELS[nm])
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("loss (nats/token)")
    ax.set_xlim(0, len(pos) * POS / 1000)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("4 books x 200k: loss by position, mean of 4 books (band +- 1 SE)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "loss_vs_position.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
