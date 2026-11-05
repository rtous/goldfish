"""Test 1: online gain against distance to the previous occurrence of the same token.

Stream: the 25 books of data.py (first 20k tokens each) concatenated, run frozen and online
(online.py, lr from lr_check.csv). For each scored token, the distance to the previous occurrence
of the same token id anywhere earlier in the stream, binned (the 1000 most frequent token ids in
the stream are left out of the bins):
    same chunk        the previous occurrence is in the token's own chunk (attention sees it)
    earlier chunk     distance < 1k, 1k-10k, 10k-100k, >= 100k (the first bin includes the few
                      tokens whose previous occurrence is in an earlier chunk at distance < 256)
    never seen        first occurrence in the stream
Also the gain by position in each book (bins of 1k tokens), mean over books 1-24 (the 24 books that
follow a switch; book 0 starts the stream).

Writes test1_distance/results.csv (bins + overall), position.csv, losses.npz (per-token losses)
and plots/. Usage: python test1_distance.py   (about 2 min)
"""
import csv
import os
import time

import numpy as np

from data import HERE, MAX_TOKENS
from online import L, chosen_lr, setup, stream_losses
from plots import plt, style, BLUE, INK, MUTED

OUT = os.path.join(HERE, "test1_distance")
N_BOOKS, TOP = 25, 1000
BINS = ["same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
        "earlier chunk, >= 100k", "never seen"]


def distance_bins(ids):
    """Bin index (into BINS) per token, -1 for the TOP most frequent ids."""
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    t = np.arange(len(ids))
    dist = t - prev
    b = np.select([prev < 0, prev >= t - t % L, dist < 1000, dist < 10000, dist < 100000], [5, 0, 1, 2, 3], 4)
    frequent = np.argsort(-np.bincount(ids))[:TOP]
    b[np.isin(ids, frequent)] = -1
    return b


def main():
    model, init, info, d = setup()
    lr = chosen_lr()
    ids = d["ids"][:N_BOOKS].reshape(-1)
    book = np.repeat(np.arange(N_BOOKS), MAX_TOKENS)
    start = time.time()
    frozen = stream_losses(model, init, ids, 0)
    t_frozen = time.time() - start
    online = stream_losses(model, init, ids, lr)
    t_online = time.time() - start - t_frozen
    print("frozen %.0f s, online %.0f s" % (t_frozen, t_online))
    np.savez_compressed(os.path.join(OUT, "losses.npz"), frozen=frozen, online=online, book=book, token=ids,
                        position=np.arange(len(ids)), lr=lr)

    scored = np.isfinite(frozen)
    bins = distance_bins(ids)
    rows = []
    for k, name in [(None, "all tokens")] + list(enumerate(BINS)):
        m = scored if k is None else scored & (bins == k)
        f, o = frozen[m].mean(), online[m].mean()
        rows.append({"bin": name, "tokens": int(m.sum()), "frozen": "%.4f" % f, "online": "%.4f" % o,
                     "gain": "%.4f" % (f - o)})
        print(rows[-1])
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    gain = np.where(scored, frozen - online, np.nan).reshape(N_BOOKS, -1, 1000)  # [book, 1k bin, token]
    per_book = np.nanmean(gain, axis=2)[1:]  # books 1-24
    mean, se = per_book.mean(0), per_book.std(0, ddof=1) / np.sqrt(len(per_book))
    with open(os.path.join(OUT, "position.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position_in_book", "gain", "se", "book0_gain"])
        for j in range(len(mean)):
            w.writerow(["%dk-%dk" % (j, j + 1), "%.4f" % mean[j], "%.4f" % se[j], "%.4f" % np.nanmean(gain[0, j])])
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("lr %g; frozen %.0f s, online %.0f s; %s\n" % (lr, t_frozen, t_online, info["device"]))

    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    g = [float(r["gain"]) for r in rows[1:]]
    ax.bar(range(len(BINS)), g, width=0.6, color=BLUE)
    for i, r in enumerate(rows[1:]):
        ax.annotate("%+.3f\n%s tok" % (g[i], format(r["tokens"], ",")), (i, max(g[i], 0)), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(BINS)), [b.replace(", ", "\n") for b in BINS])
    ax.set_ylabel("gain, frozen - online (nats/token)")
    ax.set_ylim(min(0, min(g)) * 1.3 - 0.01, max(g) * 1.35)
    ax.set_title("Online gain by distance to the previous occurrence (top %d ids excluded)" % TOP, fontsize=9,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_distance.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(mean)) + 0.5
    ax.fill_between(x, mean - se, mean + se, color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, mean, color=BLUE, lw=2, marker="o", ms=4)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("gain, frozen - online (nats/token)")
    ax.set_xlim(0, len(mean))
    ax.set_title("Online gain by position in the book, mean over books 1-24 (band: ±1 SE)", fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_position.png"), dpi=160)


if __name__ == "__main__":
    main()
