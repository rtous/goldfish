"""Test 2: does memory of a book get evicted by other books, and how fast?

A books: books 0, 1, 2 of data.py (a memoir, a novel, a travel book). A1 = first 10k tokens of
the book, A2 = the next 10k. B book for A number i: book 22 + i (not an A, not in the interlude),
B1 = its first 10k tokens. Interlude of I tokens: the first I tokens of books 3, 4, 5, ... (20k each)
concatenated, the same for every run; I in {0, 10k, 100k}.

Runs (online, lr from lr_check.csv, from the pretrained weights each time):
    A1 + interlude(I) + A2      and      B1 + interlude(I) + A2
Memory of A = loss(A2 | B1 run) - loss(A2 | A1 run), mean loss over A2's scored tokens. Both
streams have the same length before A2, so A2 is cut into the same chunks in both. Frozen loss on
A2 (same chunks) is the reference.

Writes test2_eviction/results.csv, losses.npz (per-token losses of every run, whole stream) and
plots/memory_vs_interlude.png. Usage: python test2_eviction.py   (about 4 min)
                                                 python test2_eviction.py plot   (plot only)
"""
import csv
import os
import time

import numpy as np

from data import HERE
from online import chosen_lr, setup, stream_losses
from plots import plt, style, BLUE, INK, MUTED

OUT = os.path.join(HERE, "test2_eviction")
A_BOOKS, B_FIRST, INTERLUDE_FIRST = (0, 1, 2), 22, 3
HALF = 10000
INTERLUDES = (0, 10000, 100000)


def main():
    model, init, info, d = setup()
    lr = chosen_lr()
    ids = d["ids"]
    interlude_pool = ids[INTERLUDE_FIRST:B_FIRST].reshape(-1)
    interlude_book = np.repeat(np.arange(INTERLUDE_FIRST, B_FIRST), ids.shape[1])
    rows, saved = [], {}
    start = time.time()
    for i, a in enumerate(A_BOOKS):
        b = B_FIRST + i
        a1, a2, b1 = ids[a, :HALF], ids[a, HALF:2 * HALF], ids[b, :HALF]
        for I in INTERLUDES:
            mid = interlude_pool[:I]
            seg_book = np.r_[np.full(HALF, -1), interlude_book[:I], np.full(HALF, a)]
            res = {}
            for run, first, first_book in (("A1", a1, a), ("B1", b1, b), ("frozen", a1, a)):
                stream = np.r_[first, mid, a2]
                loss = stream_losses(model, init, stream, 0 if run == "frozen" else lr)
                res[run] = np.nanmean(loss[-HALF:])
                book = seg_book.copy()
                book[:HALF] = first_book
                saved["a%d_I%d_%s" % (a, I, run)] = loss
                saved["a%d_I%d_%s_book" % (a, I, run)] = book
                saved["a%d_I%d_%s_token" % (a, I, run)] = stream
            rows.append({"A": a, "title": d["titles"][a], "B": b, "I": I, "frozen_A2": "%.4f" % res["frozen"],
                         "A2_after_A1": "%.4f" % res["A1"], "A2_after_B1": "%.4f" % res["B1"],
                         "memory": "%.4f" % (res["B1"] - res["A1"]), "gain_A1_run": "%.4f" % (res["frozen"] - res["A1"]),
                         "gain_B1_run": "%.4f" % (res["frozen"] - res["B1"])})
            print(rows[-1], "%.0f s" % (time.time() - start), flush=True)
    seconds = time.time() - start
    for I in INTERLUDES:
        r = [x for x in rows if x["I"] == I]
        mean = {k: "%.4f" % np.mean([float(x[k]) for x in r]) for k in
                ("frozen_A2", "A2_after_A1", "A2_after_B1", "memory", "gain_A1_run", "gain_B1_run")}
        rows.append({"A": "mean", "title": "", "B": "", "I": I, **mean})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(OUT, "losses.npz"), lr=lr, **saved)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("lr %g; %.0f s for all runs; %s\n" % (lr, seconds, info["device"]))
    print("%.0f s" % seconds)
    plot()


def plot():
    with open(os.path.join(OUT, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(INTERLUDES))
    for a in A_BOOKS:
        m = [float(r["memory"]) for r in rows if r["A"] == str(a)]
        ax.plot(x, m, color="#9a9893", lw=1, marker="o", ms=3)
        ax.annotate(next(r["title"] for r in rows if r["A"] == str(a)).split(" by ")[0][:28], (x[-1], m[-1]),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=MUTED)
    m = [float(r["memory"]) for r in rows if r["A"] == "mean"]
    ax.plot(x, m, color=BLUE, lw=2, marker="o", ms=5)
    ax.annotate("mean of 3 books", (x[0], m[0]), xytext=(8, 6), textcoords="offset points", va="bottom",
                ha="left", fontsize=7, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["%dk" % (I // 1000) for I in INTERLUDES])
    ax.set_xlim(-0.3, len(x) - 0.2 + 0.9)
    ax.set_xlabel("interlude between A1 and A2 (tokens of other books; x not to scale)")
    ax.set_ylabel("memory of A (nats/token on A2)")
    ax.set_title("Memory of A = loss(A2 | B1 run) - loss(A2 | A1 run)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "memory_vs_interlude.png"), dpi=160)


if __name__ == "__main__":
    import sys
    plot() if sys.argv[1:] == ["plot"] else main()
