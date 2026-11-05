"""Test 7: a more aggressive cache, and when it pays off.

Chunks: each book is chunked on its own (L = 256, remainder dropped), so every stream holds exactly
the same chunks and frozen losses are identical across streams.
    25-book chunk set   the 25 stream books (20k tokens): 78 chunks each
    long chunk set      data.long_books(): 5 books of 100k tokens: 390 chunks each
Streams (the locality knob):
    long      the long books one after the other
    K1        the 25 books one after the other
    K5        groups of 5 books (stream order); inside a group, chunks round-robin
    K25       round-robin across all 25 books
    shuffled  all 25-book chunks in one random order (seed 0)
Settings (Pythia-14M; prequential: score a chunk, then update):
    full_1e-4    all parameters, lr 1e-4
    mlp4_3e-3    layer 4's MLP (weights and biases), lr 3e-3
    mlp4_1e-2    layer 4's MLP, lr 1e-2
    fast4_d0.99  fast part F on layer 4's MLP: predict with W + F, F <- 0.99 F - 1e-2 g (online.py)
    fast4_d0.9   the same with decay 0.9
A setting whose run gives NaN, or that is worse than frozen by more than BAD nats/token on every
stream, is rerun once at half its lr (reported as an extra setting).
Frozen Pythia-14M, 31M and 70M are scored once on each chunk set.

Token classes for table 2 (25-book chunk set): previous occurrence of the same token id in book
order (the K1 stream), test 1's bins (1000 most frequent ids excluded): "same book < 10k" = the
previous occurrence is in an earlier chunk of the same book, under 10k tokens back; "never seen" =
first occurrence in book order. The same tokens are compared in every stream.

Writes test7_aggressive/: results.csv (table 1, with runtimes), bins.csv (table 2), position.csv
(plot 2), losses.npz (per-token losses as float16, [book, chunk, position] layout, for the frozen
models and the long, K1 and shuffled runs), runtime.txt and two plots. Usage:
    python test7_aggressive.py          # everything (about 40 min)
    python test7_aggressive.py plot     # plots from the saved CSVs
"""
import csv
import os
import sys
import time

import numpy as np

from data import HERE, long_books
from models import DEFAULT
from online import L, setup, stream_losses
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test1_distance import TOP
from test3_baselines import AQUA

OUT = os.path.join(HERE, "test7_aggressive")
N_BOOKS = 25
SETTINGS = [("full_1e-4", dict(lr=1e-4, params="all")), ("mlp4_3e-3", dict(lr=3e-3, params="mlp@3")),
            ("mlp4_1e-2", dict(lr=1e-2, params="mlp@3")),
            ("fast4_d0.99", dict(lr=1e-2, params="mlp@3", decay=0.99)),
            ("fast4_d0.9", dict(lr=1e-2, params="mlp@3", decay=0.9))]
STREAMS = ["long", "K1", "K5", "K25", "shuffled"]
BAD = 0.05
MODELS = {"14m": DEFAULT, "31m": "EleutherAI/pythia-31m", "70m": "EleutherAI/pythia-70m"}
POS_BIN = 5000


def chunk(ids):
    """[books, tokens] -> [books, chunks, L], remainder dropped."""
    n = ids.shape[1] // L
    return ids[:, :n * L].reshape(ids.shape[0], n, L)


def orders(n_books, n_chunks):
    """Stream name -> list of (book, chunk) in reading order (25-book chunk set)."""
    k1 = [(b, c) for b in range(n_books) for c in range(n_chunks)]
    k5 = [(b, c) for g in range(0, n_books, 5) for c in range(n_chunks) for b in range(g, g + 5)]
    k25 = [(b, c) for c in range(n_chunks) for b in range(n_books)]
    perm = np.random.default_rng(0).permutation(len(k1))
    return {"K1": k1, "K5": k5, "K25": k25, "shuffled": [k1[i] for i in perm]}


def run(model, init, chunks, order, **kwargs):
    """Per-token losses [books, chunks, L] for one pass over the chunks in this order."""
    stream = np.concatenate([chunks[b, c] for b, c in order])
    loss = stream_losses(model, init, stream, **kwargs).reshape(-1, L)
    out = np.full(chunks.shape, np.nan, dtype=np.float32)
    for i, (b, c) in enumerate(order):
        out[b, c] = loss[i]
    return out


def token_classes(chunks):
    """Boolean masks [books, chunks, L]: same book < 10k (earlier chunk), never seen; book order."""
    ids = chunks.reshape(-1)
    book = np.repeat(np.arange(chunks.shape[0]), chunks.shape[1] * L)
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    t = np.arange(len(ids))
    frequent = np.isin(ids, np.argsort(-np.bincount(ids))[:TOP])
    earlier_chunk = (prev >= 0) & (prev < t - t % L)
    same_book = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    near = earlier_chunk & same_book & (t - prev < 10000) & ~frequent
    never = (prev < 0) & ~frequent
    return near.reshape(chunks.shape), never.reshape(chunks.shape)


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    model, init, info, d = setup()
    sets = {"books25": chunk(d["ids"][:N_BOOKS]), "long": chunk(long_books(DEFAULT)["ids"])}
    print({k: v.shape for k, v in sets.items()}, flush=True)
    seconds, frozen, saved = {}, {}, {}

    # Frozen models, once per chunk set (order does not matter)
    for size, name in MODELS.items():
        if size != "14m":
            model, init, info, d = setup(name)
        for set_name, chunks in sets.items():
            order = [(b, c) for b in range(chunks.shape[0]) for c in range(chunks.shape[1])]
            t = time.time()
            frozen[size, set_name] = run(model, init, chunks, order, lr=0)
            seconds["frozen%s_%s" % (size, set_name)] = time.time() - t
            print("frozen %s %s: %.4f, %.0f s" % (size, set_name, np.nanmean(frozen[size, set_name]),
                                                  seconds["frozen%s_%s" % (size, set_name)]), flush=True)
            saved["frozen%s_%s" % (size, set_name)] = frozen[size, set_name]

    model, init, info, d = setup()
    streams = orders(N_BOOKS, sets["books25"].shape[1])
    streams["long"] = [(b, c) for b in range(sets["long"].shape[0]) for c in range(sets["long"].shape[1])]
    near, never = token_classes(sets["books25"])
    rows, bins, losses = [], [], {}
    settings = list(SETTINGS)
    i = 0
    while i < len(settings):
        name, kwargs = settings[i]
        gains = []
        for stream in STREAMS:
            set_name = "long" if stream == "long" else "books25"
            t = time.time()
            loss = run(model, init, sets[set_name], streams[stream], **kwargs)
            sec = time.time() - t
            losses[name, stream] = loss
            f14, f31, f70 = (float(np.nanmean(frozen[s, set_name])) for s in ("14m", "31m", "70m"))
            mean = float(np.nanmean(loss))
            gains.append(f14 - mean)
            rows.append({"setting": name, "stream": stream, "lr": kwargs["lr"], "decay": kwargs.get("decay", ""),
                         "params": kwargs["params"], "tokens": int(np.isfinite(loss).sum()),
                         "loss": "%.4f" % mean, "gain": "%.4f" % (f14 - mean),
                         "gap31_closed": "%.4f" % ((f14 - mean) / (f14 - f31)),
                         "gap70_closed": "%.4f" % ((f14 - mean) / (f14 - f70)),
                         "nan": bool(np.isnan(loss[:, :, 1:]).any()), "seconds": "%.0f" % sec})
            print(rows[-1], flush=True)
            if stream in ("K1", "shuffled"):
                fz = frozen["14m", "books25"]
                bins.append({"setting": name, "stream": stream,
                             "tokens_same_book_lt10k": int(near.sum()),
                             "gain_same_book_lt10k": "%.4f" % (np.nanmean(fz[near]) - np.nanmean(loss[near])),
                             "tokens_never_seen": int(never.sum()),
                             "gain_never_seen": "%.4f" % (np.nanmean(fz[never]) - np.nanmean(loss[never])),
                             "gain_all": "%.4f" % (f14 - mean)})
            if stream in ("long", "K1", "shuffled"):
                saved["%s_%s" % (name, stream)] = loss
        bad = any(r["nan"] for r in rows if r["setting"] == name) or all(g < -BAD for g in gains)
        if bad and not name.endswith("_halved"):
            print("%s is NaN or worse than frozen by > %g everywhere: rerun at half lr" % (name, BAD), flush=True)
            settings.insert(i + 1, (name + "_halved", dict(kwargs, lr=kwargs["lr"] / 2)))
        i += 1

    for size in ("31m", "70m"):  # frozen reference rows
        for stream in STREAMS:
            set_name = "long" if stream == "long" else "books25"
            f14, fs = float(np.nanmean(frozen["14m", set_name])), float(np.nanmean(frozen[size, set_name]))
            rows.append({"setting": "frozen" + size, "stream": stream, "tokens": int(np.isfinite(frozen[size, set_name]).sum()),
                         "loss": "%.4f" % fs, "gain": "%.4f" % (f14 - fs),
                         "seconds": "%.0f" % seconds["frozen%s_%s" % (size, set_name)]})
    write_csv("results.csv", rows)
    write_csv("bins.csv", bins)

    # Plot 2 data: loss by position in the long books (5k bins, mean over books of per-book means)
    n_bins = sets["long"].shape[1] * L // POS_BIN + (sets["long"].shape[1] * L % POS_BIN > 0)
    curves = {"frozen14m": frozen["14m", "long"], "frozen31m": frozen["31m", "long"],
              "frozen70m": frozen["70m", "long"]}
    curves.update({name: losses[name, "long"] for name, _ in settings})
    pos = []
    for j in range(n_bins):
        r = {"position": "%dk-%dk" % (j * POS_BIN // 1000, (j + 1) * POS_BIN // 1000)}
        for key, loss in curves.items():
            flat = loss.reshape(loss.shape[0], -1)[:, j * POS_BIN:(j + 1) * POS_BIN]
            r[key] = "%.4f" % np.mean([np.nanmean(row) for row in flat])
        pos.append(r)
    write_csv("position.csv", pos)
    np.savez_compressed(os.path.join(OUT, "losses.npz"), **{k: v.astype(np.float16) for k, v in saved.items()},
                        tokens_books25=sets["books25"], tokens_long=sets["long"])
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("%s; settings %s\n%s\n" % (info["device"], [n for n, _ in settings],
                                         ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    plot()


def plot():
    rows = list(csv.DictReader(open(os.path.join(OUT, "results.csv"))))
    pos = list(csv.DictReader(open(os.path.join(OUT, "position.csv"))))
    names = list(dict.fromkeys(r["setting"] for r in rows if not r["setting"].startswith("frozen")))
    # the decay settings share mlp4_1e-2's color (same layer, same lr) and differ by line style
    styles = {"full_1e-4": (BLUE, "-"), "mlp4_3e-3": (AQUA, "-"), "mlp4_1e-2": (ORANGE, "-"),
              "fast4_d0.99": (ORANGE, "--"), "fast4_d0.9": (ORANGE, ":")}

    def style_of(name):
        color, ls = styles.get(name.replace("_halved", ""), (MUTED, "-"))
        return color, ("-." if "halved" in name else ls)
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(STREAMS))
    for name in names:
        g = [float(next(r["gain"] for r in rows if r["setting"] == name and r["stream"] == s)) for s in STREAMS]
        color, ls = style_of(name)
        ax.plot(x, g, color=color, ls=ls, lw=2, marker="o", ms=4, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["long\n(5 x 100k)", "K = 1", "K = 5", "K = 25", "shuffled"])
    ax.set_xlabel("stream, from strongest to weakest locality")
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pythia-14M online settings across streams", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_stream.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xs = np.arange(len(pos)) * 5 + 2.5
    for key, label, ls in (("frozen14m", "frozen 14M", "-"), ("frozen31m", "frozen 31M", "--"),
                           ("frozen70m", "frozen 70M", ":")):
        ax.plot(xs, [float(r[key]) for r in pos], color=INK, lw=1.4, ls=ls, label=label)
    for name in names:
        color, ls = style_of(name)
        ax.plot(xs, [float(r[name]) for r in pos], color=color, ls=ls, lw=1.6, marker="o", ms=2.5, label=name)
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("loss (nats/token)")
    ax.set_xlim(0, len(pos) * 5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Long stream: loss by position in the book, mean of 5 books", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "loss_vs_position_long.png"), dpi=160)


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else main()
