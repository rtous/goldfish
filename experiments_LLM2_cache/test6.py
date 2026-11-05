"""Test 6: tuning gaps, where the cache lives, and a size curve.

All on test 1's stream, chunks (L = 256), distance bins plus test 5's same-book / earlier-book
split, per-token comparisons; gain = the model's own frozen loss - run; locality = gain in order -
gain with the chunks shuffled (test 4's order).

Learning-rate picks use the 2 check books (lowest mean loss). If the best value is at the edge of
its grid, the grid is extended once by one step of LADDER in that direction.
    Part A  Pythia-70M, full online learning, lr in {1e-5, 3e-5}. If a lower rate than test 5's
            3e-5 wins, rerun in order and shuffled at it; otherwise keep test 5's runs.
    Part B  Pythia-14M, MLP of the last layer and of the last 2 layers, lr in {1e-4, 3e-4, 1e-3},
            in order.
    Part C  Pythia-14M, MLP of one layer i = 1..6 (1-based), lr in {1e-4, 3e-4, 1e-3} per layer,
            in order. Layer 6 is Part B's last layer (one run).
    Part D  Pythia-31M, lr in {3e-5, 1e-4, 3e-4}; frozen, online in order and shuffled.

Writes test6/: results.csv (one row per bin and book split, a column per run), picks.csv,
losses.npz (per-token losses of the new runs in stream order; token ids, books and positions are in
test1_distance/losses.npz), runtime.txt and two plots. Usage:
    python test6.py          # everything (about 30 min)
    python test6.py report   # tables and plots from the saved losses
"""
import csv
import json
import os
import sys
import time

import numpy as np

from data import HERE
from online import LR_BOOKS, setup, stream_losses, trained_params
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test1_distance import BINS, distance_bins
from test3_baselines import AQUA

OUT = os.path.join(HERE, "test6")
T1, T4, T5 = (os.path.join(HERE, t) for t in ("test1_distance", "test4_checks", "test5_capacity"))
LADDER = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
MID, BIG = "EleutherAI/pythia-31m", "EleutherAI/pythia-70m"
GRID_14 = [1e-4, 3e-4, 1e-3]
MIN_TOKENS = 500
SIZES = {"14m": "Pythia-14M", "31m": "Pythia-31M", "70m": "Pythia-70M"}


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pick(model, init, check, grid, run, picks, params="all"):
    """Best lr on the check books, with the one-step edge extension. Appends rows to picks."""
    res = {}

    def evaluate(lr):
        res[lr] = float(np.nanmean(stream_losses(model, init, check, lr, params=params)))
        picks.append({"run": run, "lr": lr, "loss": "%.4f" % res[lr], "extension": lr not in grid})
        print(picks[-1], flush=True)

    for lr in grid:
        evaluate(lr)
    best = min(res, key=res.get)
    if best in (min(grid), max(grid)):
        i = LADDER.index(best) + (-1 if best == min(grid) else 1)
        evaluate(LADDER[i])
        best = min(res, key=res.get)
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    t1 = np.load(os.path.join(T1, "losses.npz"))
    ids = t1["token"]
    order = np.load(os.path.join(T4, "losses.npz"))["order"]
    out, seconds, picks, chosen, n_params = {}, {}, [], {}, {}

    def timed(key, model, init, **kwargs):
        t = time.time()
        out[key] = stream_losses(model, init, ids, **kwargs)
        seconds[key] = time.time() - t
        print("%s: %.0f s" % (key, seconds[key]), flush=True)

    # Parts B and C: Pythia-14M, MLP of single layers and of the last 2 layers
    model, init, info, d = setup()
    check = np.concatenate([d["ids"][b] for b in LR_BOOKS])
    picks.append({"run": "14m frozen", "lr": 0, "loss": "%.4f" % np.nanmean(stream_losses(model, init, check, 0))})
    t = time.time()
    for key, params in [("mlp_l%d" % (i + 1), "mlp@%d" % i) for i in range(6)] + [("mlp_last2", "mlp@4,5")]:
        n_params[key] = sum(p.numel() for p in trained_params(model, params))
        chosen[key] = pick(model, init, check, GRID_14, key, picks, params=params)
    seconds["picks14"] = time.time() - t
    for key in ["mlp_l%d" % (i + 1) for i in range(6)] + ["mlp_last2"]:
        params = "mlp@4,5" if key == "mlp_last2" else "mlp@%d" % (int(key[-1]) - 1)
        timed(key, model, init, lr=chosen[key], params=params)
    n_params["14m"] = sum(p.numel() for p in model.parameters())

    # Part D: Pythia-31M
    model, init, info, d = setup(MID)
    n_params["31m"] = sum(p.numel() for p in model.parameters())
    picks.append({"run": "31m frozen", "lr": 0, "loss": "%.4f" % np.nanmean(stream_losses(model, init, check, 0))})
    t = time.time()
    chosen["31m"] = pick(model, init, check, [3e-5, 1e-4, 3e-4], "31m", picks)
    seconds["picks31"] = time.time() - t
    timed("frozen31", model, init, lr=0)
    timed("online31", model, init, lr=chosen["31m"])
    timed("shuffled31", model, init, lr=chosen["31m"], order=order)

    # Part A: Pythia-70M
    model, init, info, d = setup(BIG)
    n_params["70m"] = sum(p.numel() for p in model.parameters())
    t = time.time()
    chosen["70m"] = pick(model, init, check, [1e-5, 3e-5], "70m", picks)
    seconds["picks70"] = time.time() - t
    if chosen["70m"] < 3e-5:
        timed("online70", model, init, lr=chosen["70m"])
        timed("shuffled70", model, init, lr=chosen["70m"], order=order)

    write_csv("picks.csv", picks)
    np.savez_compressed(os.path.join(OUT, "losses.npz"), **out)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("chosen lr %s\nparameters %s\n%s; %s\n" % (json.dumps(chosen), json.dumps(n_params), info["device"],
                                                         ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    report()


def report():
    d = dict(np.load(os.path.join(OUT, "losses.npz")))
    t1, t4, t5 = (np.load(os.path.join(t, "losses.npz")) for t in (T1, T4, T5))
    d.update(frozen=t1["frozen"], online=t1["online"], shuffled=t4["shuffled"], mlp=t4["mlp"],
             frozen70=t5["frozen70"])
    new70 = "online70" in d
    if not new70:
        d.update(online70=t5["online70"], shuffled70=t5["shuffled70"])
    # column -> (losses, frozen losses)
    runs = {"online14": ("online", "frozen"), "shuffled14": ("shuffled", "frozen"), "mlp_all": ("mlp", "frozen")}
    runs.update({k: (k, "frozen") for k in ["mlp_l%d" % i for i in range(1, 7)] + ["mlp_last2"]})
    runs.update({"online31": ("online31", "frozen31"), "shuffled31": ("shuffled31", "frozen31"),
                 "online70": ("online70", "frozen70"), "shuffled70": ("shuffled70", "frozen70")})

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
            r = {"bin": name, "book": split, "tokens": int(m.sum())}
            for size in ("", "31", "70"):
                r["frozen" + (size or "14")] = "%.4f" % d["frozen" + size][m].mean()
            for key, (run, frozen) in runs.items():
                r[key] = "%.4f" % (d[frozen][m].mean() - d[run][m].mean())
            for size in ("14", "31", "70"):
                r["locality" + size] = "%.4f" % (float(r["online" + size]) - float(r["shuffled" + size]))
            rows.append(r)
    write_csv("results.csv", rows)
    for r in rows:
        print({k: v for k, v in r.items() if not k.startswith("frozen")})

    def row(name, split):
        return next(r for r in rows if r["bin"] == name and r["book"] == split)

    near, mid, overall = row("earlier chunk, < 1k", "all"), row("earlier chunk, 1k-10k", "same book"), \
        row("all tokens", "all")
    style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(1, 7)
    for r, label, color in ((near, "earlier chunk, < 1k", BLUE), (mid, "earlier chunk, 1k-10k, same book", ORANGE),
                            (overall, "all tokens", INK)):
        ax.plot(x, [float(r["mlp_l%d" % i]) for i in x], color=color, lw=2, marker="o", ms=4, label=label)
    with open(os.path.join(OUT, "runtime.txt")) as f:
        lrs = json.loads(f.read().splitlines()[0].split(" ", 2)[2])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["%d\nlr %g" % (i, lrs["mlp_l%d" % i]) for i in x])
    ax.set_xlabel("layer whose MLP is updated online (1 = first)")
    ax.set_ylabel("gain over frozen (nats/token)")
    ax.set_ylim(top=0.52)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
    ax.set_title("Pythia-14M, one layer's MLP learning online (lr picked per layer)", fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gain_vs_layer.png"), dpi=160)

    with open(os.path.join(OUT, "runtime.txt")) as f:
        n_params = json.loads(f.read().splitlines()[1].split(" ", 1)[1])
    sizes = [n_params[s] / 1e6 for s in ("14m", "31m", "70m")]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for r, key, label, color in ((near, "locality", "locality, earlier chunk < 1k", BLUE),
                                 (mid, "locality", "locality, 1k-10k, same book", ORANGE),
                                 (overall, "online", "overall gain, in order", INK)):
        y = [float(r[key + s]) for s in ("14", "31", "70")]
        ax.plot(sizes, y, color=color, lw=2, marker="o", ms=4, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sizes, ["14M", "31M", "70M"])
    ax.minorticks_off()
    ax.set_xlabel("model size (parameters, log scale)")
    ax.set_ylabel("nats/token")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pythia, full online learning (lr picked per size)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "locality_vs_size.png"), dpi=160)


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
