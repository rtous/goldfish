"""E2 (claim C5): which parameters hold the online memory?

Runs (registry): Pythia-14M, plain SGD, main set, K = 1, updating only one parameter subset, each at
its own val-pick rate (e0_lr_picks): all parameters (E1's run), output embedding only, all but the
output embedding, MLPs only, attention only, the MLP of a single layer (1 ... 6, 1 = first).

Table (results.csv, long format): per subset and E1 bin, gain over frozen 14M with SE over books, and
the paired difference to the full learner (subset - all parameters). Plot: single-layer MLPs next to
the full learner. Usage (from experiments/):
    ../venv/bin/python e2_where/e2_where.py run | report
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from data import chunk_set  # noqa: E402
from metrics import bin_masks, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SUBSETS = ["all", "output", "no_output", "mlp", "attention"] + ["mlp@%d" % i for i in range(6)]
LABELS = {"all": "all parameters", "output": "output only", "no_output": "all but output", "mlp": "MLP only",
          "attention": "attention only", **{"mlp@%d" % i: "MLP of layer %d" % (i + 1) for i in range(6)}}
TABLE_BINS = ["all tokens", "same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
              "earlier chunk, >= 100k", "never seen", "earlier chunk, 1k-10k, same book"]


def keys():
    return {p: runs.find("14m", "sgd", "main_K1", p) for p in SUBSETS}


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def run():
    for k in list(keys().values()) + [runs.frozen_key("14m", "main_K1")]:
        if not os.path.exists(runs.path_of(k)):
            runs.run(next(s for s in runs.registry() if s["key"] == k))


def n_params():
    from learner import setup, trained_params
    model, _, _ = setup()
    return {p: sum(t.numel() for t in trained_params(model, p)) for p in SUBSETS}


def report():
    from plots import plt, style, BLUE, ORANGE, INK, MUTED
    k = keys()
    book, masks = bin_masks(chunk_set("main"))
    frozen = runs.load(runs.frozen_key("14m", "main_K1"))["loss"].reshape(-1)
    loss = {p: runs.load(k[p])["loss"].reshape(-1) for p in SUBSETS}
    pk, npar = runs.picks(), n_params()
    rows = []
    for p in SUBSETS:
        for b in TABLE_BINS:
            m = masks[b] & np.isfinite(frozen)
            g, d = summary(frozen - loss[p], book, m), summary(loss["all"] - loss[p], book, m)
            rows.append({"subset": p, "label": LABELS[p], "lr": "%g" % pk["14m", "sgd", p], "params": npar[p], "bin": b,
                         "tokens": g["tokens"], "gain": "%.4f" % g["mean"], "gain_se": "%.4f" % g["se"],
                         "gain_under_2se": g["under_2se"], "minus_full": "%.4f" % d["mean"],
                         "minus_full_se": "%.4f" % d["se"], "minus_full_under_2se": d["under_2se"],
                         "share_of_full": "%.2f" % (g["mean"] / float(next(r["gain"] for r in rows if r["subset"] == "all"
                                                                           and r["bin"] == b)))
                         if p != "all" else "1.00"})
    write_csv("results.csv", rows)
    for r in rows:
        if r["bin"] in ("all tokens", "earlier chunk, < 1k", "earlier chunk, 1k-10k, same book"):
            print("%-16s lr %-7s %-34s gain %s +- %s  share %s" % (r["label"], r["lr"], r["bin"], r["gain"], r["gain_se"],
                                                                 r["share_of_full"]))
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for p in SUBSETS:
            f.write("%s %d s\n" % (k[p], runs.seconds(k[p])))

    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(1, 7)
    for b, color, label in (("earlier chunk, < 1k", BLUE, "earlier chunk, < 1k"),
                            ("earlier chunk, 1k-10k, same book", ORANGE, "earlier chunk, 1k-10k, same book"),
                            ("all tokens", INK, "all tokens")):
        r = {q["subset"]: q for q in rows if q["bin"] == b}
        y = np.array([float(r["mlp@%d" % (i - 1)]["gain"]) for i in x])
        s = np.array([float(r["mlp@%d" % (i - 1)]["gain_se"]) for i in x])
        ax.errorbar(x, y, yerr=2 * s, color=color, lw=1.8, marker="o", ms=4, capsize=2, label=label)
        ax.axhline(float(r["all"]["gain"]), color=color, lw=1, ls="--")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["%d\nlr %s" % (i, next(q["lr"] for q in rows if q["subset"] == "mlp@%d" % (i - 1))) for i in x])
    ax.set_xlabel("layer whose MLP learns online (1 = first); dashed: all parameters")
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pythia-14M, one layer's MLP learning online (bars +- 2 SE)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_layer.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
