"""Test 3: is the "cache in the weights" more than a word-frequency cache?

Three runs on test 1's stream (same books, chunks of L = 256, same distance bins), compared token
by token with test 1's frozen and online (all parameters) losses:
    unigram        frozen model mixed with a cache of recent token ids, no training:
                   p = (1 - lam) p_model + lam p_cache, p_cache = normalized exponentially decaying
                   counts (half-life h tokens) of the ids seen so far. The counts are updated once
                   per chunk, after scoring it (the online learner also learns only after scoring).
                   The first chunk has an empty cache and uses p_model alone.
    output         online SGD as in test 1, updating only the output embedding (lm_head.weight)
    no_output      online SGD updating every parameter except the output embedding
Sanity picks on the 2 check books of online.py (lowest mean loss): lam in {0.05, 0.1, 0.2} x
h in {1k, 10k}; lr in {1e-4, 3e-4, 1e-3} for output and no_output.

Test 2 (eviction, same A/B books and interludes as test2_eviction.py) is repeated for the three
runs. For the unigram cache only the chunks touching A2 need frozen losses; they are recomputed for
each stream (at I = 0 the chunk that straddles into A2 holds the end of A1 or B1).

Writes test3_baselines/: picks.csv, distance.csv, position.csv, eviction.csv, losses.npz
(per-token losses of the test 1 stream), runtime.txt and plots/. Usage:
    python test3_baselines.py          # everything (about 15 min)
    python test3_baselines.py report   # tables and plots from the saved results
"""
import csv
import json
import os
import sys
import time

import numpy as np

from data import HERE, MAX_TOKENS
from online import L, LR_BOOKS, LRS, chosen_lr, setup, stream_losses
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test1_distance import BINS, N_BOOKS, TOP, distance_bins
from test2_eviction import A_BOOKS, B_FIRST, HALF, INTERLUDE_FIRST, INTERLUDES

OUT = os.path.join(HERE, "test3_baselines")
T1, T2 = os.path.join(HERE, "test1_distance"), os.path.join(HERE, "test2_eviction")
LAMS, HALF_LIVES = (0.05, 0.1, 0.2), (1000, 10000)
VOCAB = 50304
AQUA = "#1baf7a"
# (key, label, color): the order of the table columns and plot legends
METHODS = [("online", "online, all parameters (test 1)", BLUE), ("unigram", "unigram cache", INK),
           ("output", "online, output layer only", ORANGE), ("no_output", "online, all but output layer", AQUA)]


def unigram_losses(frozen, ids, lam, half_life):
    """Per-token loss of the frozen model mixed with the unigram cache (NaN where frozen is NaN)."""
    decay = 0.5 ** (1 / half_life)
    counts = np.zeros(VOCAB)
    out = frozen.copy()
    for s in range(0, len(ids), L):
        e = min(s + L, len(ids))
        total = counts.sum()
        if total > 0:
            p_cache = counts[ids[s:e]] / total
            out[s:e] = -np.log((1 - lam) * np.exp(-frozen[s:e].astype(np.float64)) + lam * p_cache)
        counts *= decay ** (e - s)
        np.add.at(counts, ids[s:e], decay ** np.arange(e - s - 1, -1, -1))
    return out.astype(np.float32)


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def picks(model, init, d):
    ids = np.concatenate([d["ids"][b] for b in LR_BOOKS])
    frozen = stream_losses(model, init, ids, 0)
    rows = [{"method": "frozen", "setting": "", "loss": "%.4f" % np.nanmean(frozen)}]
    for lam in LAMS:
        for h in HALF_LIVES:
            rows.append({"method": "unigram", "setting": "lam=%g half_life=%d" % (lam, h),
                         "loss": "%.4f" % np.nanmean(unigram_losses(frozen, ids, lam, h))})
    for params in ("output", "no_output"):
        for lr in LRS:
            rows.append({"method": params, "setting": "lr=%g" % lr,
                         "loss": "%.4f" % np.nanmean(stream_losses(model, init, ids, lr, params=params))})
    for r in rows:
        print(r, flush=True)
    write_csv("picks.csv", rows)
    best = {}
    for m in ("unigram", "output", "no_output"):
        r = min((r for r in rows if r["method"] == m), key=lambda r: float(r["loss"]))
        best[m] = dict(kv.split("=") for kv in r["setting"].split())
    return {"unigram": (float(best["unigram"]["lam"]), int(best["unigram"]["half_life"])),
            "output": float(best["output"]["lr"]), "no_output": float(best["no_output"]["lr"])}


def run_test1(model, init, best):
    t1 = np.load(os.path.join(T1, "losses.npz"))
    ids, frozen = t1["token"], t1["frozen"]
    out = {"frozen": frozen, "online": t1["online"], "unigram": unigram_losses(frozen, ids, *best["unigram"]),
           "book": t1["book"], "token": ids, "position": t1["position"]}
    seconds = {}
    for params in ("output", "no_output"):
        t = time.time()
        out[params] = stream_losses(model, init, ids, best[params], params=params)
        seconds[params] = time.time() - t
        print("test 1 stream, %s: %.0f s" % (params, seconds[params]), flush=True)
    np.savez_compressed(os.path.join(OUT, "losses.npz"), **out)
    return seconds


def run_test2(model, init, d, best):
    ids = d["ids"]
    pool = ids[INTERLUDE_FIRST:B_FIRST].reshape(-1)
    t2 = np.load(os.path.join(T2, "losses.npz"))
    rows, seconds = [], {"output": 0.0, "no_output": 0.0}
    for i, a in enumerate(A_BOOKS):
        a1, a2, b1 = ids[a, :HALF], ids[a, HALF:2 * HALF], ids[B_FIRST + i, :HALF]
        for I in INTERLUDES:
            streams = {"A1": np.r_[a1, pool[:I], a2], "B1": np.r_[b1, pool[:I], a2]}
            res = {}
            for run, stream in streams.items():
                start = (len(stream) - HALF) // L * L  # first chunk touching A2 (chunk boundaries kept)
                frozen = np.full(len(stream), np.nan, dtype=np.float32)
                frozen[start:] = stream_losses(model, init, stream[start:], 0)
                if run == "A1":  # same chunks as test 2's frozen run
                    assert np.allclose(frozen[-HALF:], t2["a%d_I%d_frozen" % (a, I)][-HALF:], atol=1e-4,
                                       equal_nan=True)
                    res["frozen"] = np.nanmean(frozen[-HALF:])
                res["unigram", run] = np.nanmean(unigram_losses(frozen, stream, *best["unigram"])[-HALF:])
                for params in ("output", "no_output"):
                    t = time.time()
                    res[params, run] = np.nanmean(stream_losses(model, init, stream, best[params],
                                                                params=params)[-HALF:])
                    seconds[params] += time.time() - t
            for m in ("unigram", "output", "no_output"):
                rows.append({"method": m, "A": a, "I": I, "frozen_A2": "%.4f" % res["frozen"],
                             "A2_after_A1": "%.4f" % res[m, "A1"], "A2_after_B1": "%.4f" % res[m, "B1"],
                             "memory": "%.4f" % (res[m, "B1"] - res[m, "A1"])})
                print(rows[-1], flush=True)
    for r in read_csv(os.path.join(T2, "results.csv")):  # test 2's full-model numbers, for comparison
        if r["A"] != "mean":
            rows.append({"method": "online", **{k: r[k] for k in ("A", "I", "frozen_A2", "A2_after_A1",
                                                                     "A2_after_B1", "memory")}})
    write_csv("eviction.csv", rows)
    return seconds


def report():
    d = np.load(os.path.join(OUT, "losses.npz"))
    scored = np.isfinite(d["frozen"])
    bins = distance_bins(d["token"])
    keys = [m for m, _, _ in METHODS]
    rows = []
    for k, name in [(None, "all tokens")] + list(enumerate(BINS)):
        m = scored if k is None else scored & (bins == k)
        f = d["frozen"][m].mean()
        r = {"bin": name, "tokens": int(m.sum()), "frozen": "%.4f" % f}
        r.update({key: "%.4f" % d[key][m].mean() for key in keys})
        r.update({"gain_" + key: "%.4f" % (f - d[key][m].mean()) for key in keys})
        rows.append(r)
    write_csv("distance.csv", rows)

    per_book = {}
    for key in keys:
        gain = np.where(scored, d["frozen"] - d[key], np.nan).reshape(N_BOOKS, -1, 1000)
        per_book[key] = np.nanmean(gain, axis=2)[1:]  # books 1-24, as in test 1
    pos = [{"position_in_book": "%dk-%dk" % (j, j + 1),
            **{key: "%.4f" % per_book[key][:, j].mean() for key in keys}} for j in range(MAX_TOKENS // 1000)]
    write_csv("position.csv", pos)

    ev = read_csv(os.path.join(OUT, "eviction.csv"))
    style()
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(BINS))
    for key, label, color in METHODS:
        ax.plot(x, [float(r["gain_" + key]) for r in rows[1:]], color=color, lw=2, marker="o", ms=4, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [b.replace(", ", "\n") for b in BINS])
    ax.set_ylabel("gain, frozen - method (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Gain by distance to the previous occurrence (top %d ids excluded)" % TOP, fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_distance.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(MAX_TOKENS // 1000) + 0.5
    for key, label, color in METHODS:
        ax.plot(x, per_book[key].mean(0), color=color, lw=2, marker="o", ms=3, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(0, MAX_TOKENS // 1000)
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("gain, frozen - method (nats/token)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Gain by position in the book, mean over books 1-24", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_position.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(INTERLUDES))
    for key, label, color in METHODS:
        mem = [np.mean([float(r["memory"]) for r in ev if r["method"] == key and int(r["I"]) == I])
               for I in INTERLUDES]
        ax.plot(x, mem, color=color, lw=2, marker="o", ms=4, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["%dk" % (I // 1000) for I in INTERLUDES])
    ax.set_xlabel("interlude between A1 and A2 (tokens of other books; x not to scale)")
    ax.set_ylabel("memory of A (nats/token on A2)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Memory of A, mean of 3 books", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "memory_vs_interlude.png"), dpi=160)

    for r in rows:
        print(r)
    for key in keys:
        print(key, [round(np.mean([float(r["memory"]) for r in ev if r["method"] == key and int(r["I"]) == I]), 4)
                    for I in INTERLUDES])


def main():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    model, init, info, d = setup()
    t0 = time.time()
    best = picks(model, init, d)
    t_picks = time.time() - t0
    print("picks:", best, "full model lr", chosen_lr(), "%.0f s" % t_picks, flush=True)
    s1 = run_test1(model, init, best)
    t1 = time.time()
    s2 = run_test2(model, init, d, best)
    t2 = time.time() - t1
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("picks %s; %s\npicks %.0f s; test 1 stream: output %.0f s, no_output %.0f s; test 2: %.0f s in total "
                "(output %.0f s, no_output %.0f s)\n" % (json.dumps(best), info["device"], t_picks, s1["output"],
                                                       s1["no_output"], t2, s2["output"], s2["no_output"]))
    report()


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
