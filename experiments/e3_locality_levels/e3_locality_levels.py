"""E3 (claim C6): does the gain fall steadily as the input's locality decreases?

Runs (registry): Pythia-14M plain SGD on all parameters at its val-pick rate, on five streams from
strongest to weakest locality: long (5 books x 100k tokens, book after book), and the main set's
25 x 78 chunks in book order (K = 1), round-robin in groups of 5 books (K = 5), round-robin across all
25 (K = 25), shuffled. The main streams hold the same chunks, so their gains are paired per token.

Table (results.csv): overall gain over frozen 14M with SE over books, and the paired difference to
the previous (more local) main stream. Usage (from experiments/):
    ../venv/bin/python e3_locality_levels/e3_locality_levels.py run | report
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from metrics import summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
STREAMS = [("long", "long (5 x 100k)"), ("main_K1", "K = 1"), ("main_K5", "K = 5"), ("main_K25", "K = 25"),
           ("main_shuffled", "shuffled")]


def keys():
    return {st: runs.find("14m", "sgd", st) for st, _ in STREAMS}


def run():
    for k in list(keys().values()) + [runs.frozen_key("14m", "main_K1"), runs.frozen_key("14m", "long")]:
        if not os.path.exists(runs.path_of(k)):
            runs.run(next(s for s in runs.registry() if s["key"] == k))


def report():
    from plots import plt, style, BLUE, INK, MUTED
    k = keys()
    rows, prev = [], None
    for st, label in STREAMS:
        frozen = runs.load(runs.frozen_key("14m", st))["loss"]
        loss = runs.load(k[st])["loss"]
        book = np.repeat(np.arange(frozen.shape[0]), frozen.shape[1] * frozen.shape[2])
        g = summary((frozen - loss).reshape(-1), book, np.ones(book.shape, bool))
        r = {"stream": st, "label": label, "lr": "%g" % runs.load_meta(k[st])["lr"], "tokens": g["tokens"],
             "frozen": "%.4f" % np.nanmean(frozen), "loss": "%.4f" % np.nanmean(loss), "gain": "%.4f" % g["mean"],
             "gain_se": "%.4f" % g["se"], "books": g["books"]}
        if prev is not None and st.startswith("main_") and prev[0].startswith("main_"):
            d = summary((prev[1] - loss).reshape(-1), book, np.ones(book.shape, bool))
            r.update({"drop_from_previous": "%.4f" % d["mean"], "drop_se": "%.4f" % d["se"],
                      "drop_under_2se": d["under_2se"]})
        prev = (st, loss)
        rows.append(r)
        print(r)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(kk for r in rows for kk in r)))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for st, _ in STREAMS:
            f.write("%s %d s\n" % (k[st], runs.seconds(k[st])))

    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(rows))
    y, s = np.array([float(r["gain"]) for r in rows]), np.array([float(r["gain_se"]) for r in rows])
    ax.errorbar(x, y, yerr=2 * s, color=BLUE, lw=2, marker="o", ms=4, capsize=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [r["label"].replace(" (", "\n(") for r in rows])
    ax.set_xlabel("stream, from strongest to weakest locality")
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.set_title("Pythia-14M, plain SGD, lr %s (bars +- 2 SE over books)" % rows[0]["lr"], fontsize=8.5, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_stream.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
