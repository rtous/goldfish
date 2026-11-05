"""Test 4: quick checks on test 3's result.

All on test 1's stream, chunks (L = 256) and distance bins, compared token by token with test 1's
frozen losses (gain = frozen - run):
    shuffled       full online learning (lr 1e-4) on the same chunks in a random order (seed 0)
                   across all books; losses put back in stream order. Locality effect per token =
                   gain(in order) - gain(shuffled).
    mlp            online learning updating only the MLP weights of every layer (lr 1e-4)
    attention      online learning updating only the attention weights of every layer (lr 1e-4)
    online_cache   test 1's online learner with test 3's unigram cache mixed into its predictions
                   (lam 0.05, half-life 1k). The cache depends only on the token ids, so mixing it
                   into test 1's per-token online probabilities is exact: no new run.
Grid extensions on the 2 check books: unigram cache lam 0.02 (half-life 1k) and output-only lr
3e-3. A value that beats test 3's pick is rerun on the stream and added to the table.

Writes test4_checks/: results.csv (distance table), checks.csv (grid extensions), losses.npz (per-token
losses of the new runs, in stream order, and the chunk order; token ids, books and positions are
in test1_distance/losses.npz), runtime.txt and gain_vs_distance.png. Usage:
    python test4_checks.py          # everything (about 5 min)
    python test4_checks.py report   # table and plot from the saved losses
"""
import csv
import os
import sys
import time

import numpy as np

from data import HERE
from online import L, LR_BOOKS, setup, stream_losses
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test1_distance import BINS, TOP, distance_bins
from test3_baselines import AQUA, unigram_losses

OUT = os.path.join(HERE, "test4_checks")
T1, T3 = os.path.join(HERE, "test1_distance"), os.path.join(HERE, "test3_baselines")
LR = 1e-4
CACHE = (0.05, 1000)  # test 3's pick
# (key, label, color, line style), in table and legend order
RUNS = [("online", "online, all, in order (test 1)", BLUE, "-"), ("shuffled", "online, all, shuffled chunks", BLUE, "--"),
        ("mlp", "online, MLP only", AQUA, "-"), ("attention", "online, attention only", ORANGE, "-"),
        ("online_cache", "online + unigram cache", INK, "-"), ("unigram", "unigram cache alone (test 3)", INK, "--")]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    model, init, info, d = setup()
    t1 = np.load(os.path.join(T1, "losses.npz"))
    ids = t1["token"]
    out = {}  # only the new runs; report() takes the rest from tests 1 and 3
    seconds = {}

    # Grid extensions on the check books, against test 3's best picks
    check = np.concatenate([d["ids"][b] for b in LR_BOOKS])
    frozen_check = stream_losses(model, init, check, 0)
    picks = read_csv(os.path.join(T3, "picks.csv"))
    best = {m: min(float(r["loss"]) for r in picks if r["method"] == m) for m in ("unigram", "output")}
    t = time.time()
    new = {"unigram": np.nanmean(unigram_losses(frozen_check, check, 0.02, 1000)),
           "output": np.nanmean(stream_losses(model, init, check, 3e-3, params="output"))}
    seconds["checks"] = time.time() - t
    rows = [{"method": "unigram", "setting": "lam=0.02 half_life=1000", "loss": "%.4f" % new["unigram"],
             "test3_best": "%.4f" % best["unigram"], "better": new["unigram"] < best["unigram"]},
            {"method": "output", "setting": "lr=0.003", "loss": "%.4f" % new["output"],
             "test3_best": "%.4f" % best["output"], "better": new["output"] < best["output"]}]
    write_csv("checks.csv", rows)
    print(rows, flush=True)
    if new["unigram"] < best["unigram"]:
        out["unigram_lam0.02"] = unigram_losses(t1["frozen"], ids, 0.02, 1000)
    if new["output"] < best["output"]:
        t = time.time()
        out["output_lr3e-3"] = stream_losses(model, init, ids, 3e-3, params="output")
        seconds["output_lr3e-3"] = time.time() - t

    n_chunks = (len(ids) + L - 1) // L
    order = np.random.default_rng(0).permutation(n_chunks)
    for key, kwargs in (("shuffled", {"order": order}), ("mlp", {"params": "mlp"}),
                        ("attention", {"params": "attention"})):
        t = time.time()
        out[key] = stream_losses(model, init, ids, LR, **kwargs)
        seconds[key] = time.time() - t
        print("%s: %.0f s" % (key, seconds[key]), flush=True)
    out["online_cache"] = unigram_losses(t1["online"], ids, *CACHE)
    out["order"] = order
    np.savez_compressed(os.path.join(OUT, "losses.npz"), **out)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("%s; %s\n" % (info["device"], ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    report()


def report():
    d = dict(np.load(os.path.join(OUT, "losses.npz")))
    t1 = np.load(os.path.join(T1, "losses.npz"))
    d.update(frozen=t1["frozen"], online=t1["online"], token=t1["token"],
             unigram=np.load(os.path.join(T3, "losses.npz"))["unigram"])
    runs = list(RUNS)
    if "unigram_lam0.02" in d:
        runs.append(("unigram_lam0.02", "unigram cache, lam 0.02", INK, ":"))
    if "output_lr3e-3" in d:
        runs.append(("output_lr3e-3", "online, output only, lr 3e-3", "#9a9893", ":"))
    scored = np.isfinite(d["frozen"])
    bins = distance_bins(d["token"])
    rows = []
    for k, name in [(None, "all tokens")] + list(enumerate(BINS)):
        m = scored if k is None else scored & (bins == k)
        f = d["frozen"][m].mean()
        r = {"bin": name, "tokens": int(m.sum()), "frozen": "%.4f" % f}
        r.update({"gain_" + key: "%.4f" % (f - d[key][m].mean()) for key, _, _, _ in runs})
        r["locality"] = "%.4f" % (d["shuffled"][m].mean() - d["online"][m].mean())
        rows.append(r)
        print(r)
    write_csv("results.csv", rows)

    style()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(BINS))
    for key, label, color, ls in runs:
        ax.plot(x, [float(r["gain_" + key]) for r in rows[1:]], color=color, ls=ls, lw=2, marker="o", ms=4,
                label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [b.replace(", ", "\n") for b in BINS])
    ax.set_ylabel("gain, frozen - run (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Gain by distance to the previous occurrence (top %d ids excluded)" % TOP, fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gain_vs_distance.png"), dpi=160)


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
