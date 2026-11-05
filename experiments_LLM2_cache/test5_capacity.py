"""Test 5: does the cache depend on capacity?

All on test 1's stream, chunks (L = 256) and distance bins, per-token comparisons; gain = the
model's own frozen loss - run.

Part A (no runs): each "earlier chunk" bin split by whether the token's previous occurrence is in
the same book or in an earlier book, for test 1 (online, in order), test 4 (shuffled, MLP only)
and test 3 (unigram cache). Sub-bins with fewer than MIN_TOKENS tokens are skipped.
Part B: EleutherAI/pythia-70m (same tokenizer), frozen, online in order and online shuffled (test
4's chunk order), full online learning with plain SGD; lr picked from {3e-5, 1e-4, 3e-4} on the 2
check books.
Part C: Pythia-14M updating only the MLP of the last layer, and of the last 2 layers (lr 1e-4, in
order); test 4 has all 6 layers.

Writes test5_capacity/: results.csv (one row per bin and book split, a column per run), picks.csv,
losses.npz (per-token losses of the new runs, stream order; token ids, books and positions are in
test1_distance/losses.npz), runtime.txt and two plots. Usage:
    python test5_capacity.py          # everything (about 25 min)
    python test5_capacity.py report   # tables and plots from the saved losses
"""
import csv
import os
import sys
import time

import numpy as np

from data import HERE
from online import LR_BOOKS, setup, stream_losses, trained_params
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test1_distance import BINS, TOP, distance_bins
from test3_baselines import AQUA

OUT = os.path.join(HERE, "test5_capacity")
T1, T3, T4 = (os.path.join(HERE, t) for t in ("test1_distance", "test3_baselines", "test4_checks"))
BIG = "EleutherAI/pythia-70m"
LRS_BIG = (3e-5, 1e-4, 3e-4)
MLP_LAYERS = {"mlp_last1": "mlp@5", "mlp_last2": "mlp@4,5"}
LR_MLP = 1e-4
MIN_TOKENS = 500
# result columns: key -> (label, losses key, frozen key)
GAINS = {"online14": ("14M online, in order (test 1)", "online", "frozen"),
         "shuffled14": ("14M online, shuffled (test 4)", "shuffled", "frozen"),
         "mlp14": ("14M MLP, all 6 layers (test 4)", "mlp", "frozen"),
         "mlp_last2_14": ("14M MLP, last 2 layers", "mlp_last2", "frozen"),
         "mlp_last1_14": ("14M MLP, last layer", "mlp_last1", "frozen"),
         "unigram14": ("unigram cache on 14M (test 3)", "unigram", "frozen"),
         "online70": ("70M online, in order", "online70", "frozen70"),
         "shuffled70": ("70M online, shuffled", "shuffled70", "frozen70")}


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    t1 = np.load(os.path.join(T1, "losses.npz"))
    ids = t1["token"]
    order = np.load(os.path.join(T4, "losses.npz"))["order"]
    out, seconds = {}, {}

    model, init, info, d = setup()  # Part C, Pythia-14M
    params14 = {}
    for key, params in MLP_LAYERS.items():
        params14[key] = sum(p.numel() for p in trained_params(model, params))
        t = time.time()
        out[key] = stream_losses(model, init, ids, LR_MLP, params=params)
        seconds[key] = time.time() - t
        print("%s (%d parameters): %.0f s" % (key, params14[key], seconds[key]), flush=True)
    params14["mlp_all"] = sum(p.numel() for p in trained_params(model, "mlp"))
    del model, init

    model, init, info, d = setup(BIG)  # Part B
    check = np.concatenate([d["ids"][b] for b in LR_BOOKS])
    t = time.time()
    picks = [{"model": BIG, "lr": 0, "loss": "%.4f" % np.nanmean(stream_losses(model, init, check, 0))}]
    for lr in LRS_BIG:
        picks.append({"model": BIG, "lr": lr, "loss": "%.4f" % np.nanmean(stream_losses(model, init, check, lr))})
        print(picks[-1], flush=True)
    seconds["picks70"] = time.time() - t
    write_csv("picks.csv", picks)
    lr = float(min(picks[1:], key=lambda r: float(r["loss"]))["lr"])
    for key, kwargs in (("frozen70", {"lr": 0}), ("online70", {"lr": lr}), ("shuffled70", {"lr": lr, "order": order})):
        t = time.time()
        out[key] = stream_losses(model, init, ids, **kwargs)
        seconds[key] = time.time() - t
        print("%s: %.0f s" % (key, seconds[key]), flush=True)
    np.savez_compressed(os.path.join(OUT, "losses.npz"), lr70=lr, **out)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("70M lr %g, %d parameters; 14M MLP parameters %s; %s\n%s\n" % (
            lr, sum(p.numel() for p in model.parameters()), params14, info["device"],
            ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    report()


def report():
    d = dict(np.load(os.path.join(OUT, "losses.npz")))
    t1, t3, t4 = (np.load(os.path.join(t, "losses.npz")) for t in (T1, T3, T4))
    d.update(frozen=t1["frozen"], online=t1["online"], unigram=t3["unigram"], shuffled=t4["shuffled"], mlp=t4["mlp"])
    ids, book = t1["token"], t1["book"]
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    same_book = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    scored = np.isfinite(d["frozen"])
    bins = distance_bins(ids)

    rows = []
    for k, name in [(None, "all tokens")] + list(enumerate(BINS)):
        base = scored if k is None else scored & (bins == k)
        splits = [("all", base)]
        if name.startswith("earlier chunk"):
            splits += [("same book", base & same_book), ("earlier book", base & ~same_book)]
        for split, m in splits:
            if m.sum() < MIN_TOKENS:
                continue
            r = {"bin": name, "book": split, "tokens": int(m.sum()), "frozen14": "%.4f" % d["frozen"][m].mean(),
                 "frozen70": "%.4f" % d["frozen70"][m].mean()}
            for key, (_, run, frozen) in GAINS.items():
                r[key] = "%.4f" % (d[frozen][m].mean() - d[run][m].mean())
            r["locality14"] = "%.4f" % (float(r["online14"]) - float(r["shuffled14"]))
            r["locality70"] = "%.4f" % (float(r["online70"]) - float(r["shuffled70"]))
            rows.append(r)
            print({k: v for k, v in r.items() if k not in ("frozen14", "frozen70")})
    write_csv("results.csv", rows)

    main_rows = [r for r in rows if r["book"] == "all" and r["bin"] != "all tokens"]
    x = np.arange(len(main_rows))
    labels = [r["bin"].replace(", ", "\n") for r in main_rows]
    style()
    for fname, title, runs in (
            ("gain_vs_distance_size.png", "Pythia-14M and 70M, full online learning, in order and shuffled",
             [("online14", BLUE, "-"), ("shuffled14", BLUE, "--"), ("online70", ORANGE, "-"),
              ("shuffled70", ORANGE, "--")]),
            ("gain_vs_distance_mlp.png", "Pythia-14M, online learning of fewer MLP layers (in order)",
             [("online14", BLUE, "-"), ("mlp14", AQUA, "-"), ("mlp_last2_14", INK, "-"),
              ("mlp_last1_14", "#9a9893", "-")])):
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        for key, color, ls in runs:
            ax.plot(x, [float(r[key]) for r in main_rows], color=color, ls=ls, lw=2, marker="o", ms=4,
                    label=GAINS[key][0])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("gain over own frozen model (nats/token)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("%s (top %d ids excluded)" % (title, TOP), fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=160)


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
