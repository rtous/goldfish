"""Test 8: dual weights (slow SGD on all parameters + a fast decaying part on some MLP weights).

Update rule (dual.py): predict with S + F; after each chunk S <- S - 1e-4 g, F <- lam F - alpha g on
the fast tensors. Configs: fast part on (a) layer 4's MLP weights or (b) all six MLPs' weights
(weight matrices only), alpha in {1e-3, 3e-3}, lam in {0.9, 0.99}: 8 configs.
1. All 8 on val-pick K = 1 and val-pick shuffled (PG-19 validation books, dual.val_sets); the
   config with the lowest mean loss on val-pick K = 1 is picked. Frozen 14M and full_1e-4 on
   val-pick for reference.
2. The pick (and the best config on val-pick shuffled, if different) on test 7's test streams
   K1, shuffled and long, with frozen 14M and full_1e-4 rerun through dual.py (no fast part; this
   also checks that it reproduces test 7's full_1e-4 exactly).
Reported per test stream: gain over frozen 14M and over full_1e-4, locality (K1 gain - shuffled
gain), test 7's token classes (same book < 10k, never seen), and the long-stream loss by position
(5k bins) with frozen 31M from test 7 as reference.

Writes test8_dual/: results.csv, bins.csv, position.csv, losses.npz (float16 [book, chunk,
position], test-stream runs), runtime.txt, plots/. Usage:
    python test8_dual.py          # everything (about 25 min)
    python test8_dual.py plot     # plots from the saved CSVs
"""
import csv
import json
import os
import sys
import time

import numpy as np

from data import HERE, books, long_books
from dual import dual_losses, mlp_weights, val_sets
from models import DEFAULT
from online import setup
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test3_baselines import AQUA
from test7_aggressive import N_BOOKS, POS_BIN, chunk, orders, run, token_classes

OUT = os.path.join(HERE, "test8_dual")
T7 = os.path.join(HERE, "test7_aggressive")
PARTS = {"mlp4": mlp_weights([3]), "mlpall": mlp_weights(range(6))}
CONFIGS = [("%s_a%g_l%g" % (part, alpha, lam), part, alpha, lam)
           for part in PARTS for alpha in (1e-3, 3e-3) for lam in (0.9, 0.99)]
TEST_STREAMS = ["K1", "shuffled", "long"]


def fast_of(part, alpha, lam):
    return {n: (alpha, lam) for n in PARTS[part]}


def test_sets():
    """Test 7's chunk sets and the K1 / shuffled / long orders."""
    sets = {"books25": chunk(books(DEFAULT)["ids"][:N_BOOKS]), "long": chunk(long_books(DEFAULT)["ids"])}
    o = orders(N_BOOKS, sets["books25"].shape[1])
    streams = {"K1": ("books25", o["K1"]), "shuffled": ("books25", o["shuffled"]),
               "long": ("long", [(b, c) for b in range(sets["long"].shape[0]) for c in range(sets["long"].shape[1])])}
    return sets, streams


def position_curve(loss):
    """Mean over books of per-book mean loss in POS_BIN-token bins of book position."""
    flat = loss.reshape(loss.shape[0], -1)
    n = (flat.shape[1] + POS_BIN - 1) // POS_BIN
    return [float(np.mean([np.nanmean(row[j * POS_BIN:(j + 1) * POS_BIN]) for row in flat])) for j in range(n)]


def write_csv(out, name, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def evaluate_test(model, init, sets, streams, runs, out_dir, extra_rows=()):
    """Frozen 14M and full_1e-4 rerun + the given runs {name: fast dict} on the test streams. Returns rows,
    bins rows, position rows, per-token losses, seconds."""
    seconds, losses = {}, {}
    frozen = {}
    for set_name, chunks in sets.items():
        order = [(b, c) for b in range(chunks.shape[0]) for c in range(chunks.shape[1])]
        t = time.time()
        frozen[set_name] = run(model, init, chunks, order, lr=0)
        seconds["frozen14m_" + set_name] = time.time() - t
    t7 = {(r["setting"], r["stream"]): r for r in csv.DictReader(open(os.path.join(T7, "results.csv")))}
    rows = list(extra_rows)
    for name, fast in [("full_1e-4", None)] + list(runs.items()):
        for stream in TEST_STREAMS:
            set_name, order = streams[stream]
            t = time.time()
            loss = dual_losses(model, init, sets[set_name], order, fast)
            seconds["%s_%s" % (name, stream)] = time.time() - t
            losses[name, stream] = loss
            print(name, stream, "%.4f %.0f s" % (np.nanmean(loss), seconds["%s_%s" % (name, stream)]), flush=True)
    full = {s: float(np.nanmean(losses["full_1e-4", s])) for s in TEST_STREAMS}
    check = {s: "%.4f" % full[s] == t7["full_1e-4", s]["loss"] for s in TEST_STREAMS}
    print("full_1e-4 through dual.py reproduces test 7:", check, flush=True)
    near, never = token_classes(sets["books25"])
    bins = []
    gains = {}
    for name in ["full_1e-4"] + list(runs):
        for stream in TEST_STREAMS:
            set_name = streams[stream][0]
            f14 = float(np.nanmean(frozen[set_name]))
            mean = float(np.nanmean(losses[name, stream]))
            gains[name, stream] = f14 - mean
            rows.append({"phase": "test", "config": name, "stream": stream, "loss": "%.4f" % mean,
                         "gain_frozen": "%.4f" % (f14 - mean), "gain_full": "%.4f" % (full[stream] - mean),
                         "seconds": "%.0f" % seconds["%s_%s" % (name, stream)]})
            if stream in ("K1", "shuffled"):
                fz, ls, lf = frozen["books25"], losses[name, stream], losses["full_1e-4", stream]
                bins.append({"config": name, "stream": stream,
                             "gain_same_book_lt10k": "%.4f" % (np.nanmean(fz[near]) - np.nanmean(ls[near])),
                             "gain_never_seen": "%.4f" % (np.nanmean(fz[never]) - np.nanmean(ls[never])),
                             "vs_full_same_book_lt10k": "%.4f" % (np.nanmean(lf[near]) - np.nanmean(ls[near])),
                             "vs_full_never_seen": "%.4f" % (np.nanmean(lf[never]) - np.nanmean(ls[never])),
                             "tokens_same_book_lt10k": int(near.sum()), "tokens_never_seen": int(never.sum())})
        rows.append({"phase": "test", "config": name, "stream": "locality (K1 - shuffled)",
                     "gain_frozen": "%.4f" % (gains[name, "K1"] - gains[name, "shuffled"])})
    for set_name in sets:
        rows.append({"phase": "test", "config": "frozen14m", "stream": set_name,
                     "loss": "%.4f" % np.nanmean(frozen[set_name]), "seconds": "%.0f" % seconds["frozen14m_" + set_name]})
    t7pos = list(csv.DictReader(open(os.path.join(T7, "position.csv"))))
    curves = {"frozen14m": position_curve(frozen["long"])}
    curves.update({name: position_curve(losses[name, "long"]) for name in ["full_1e-4"] + list(runs)})
    pos = []
    for j, r7 in enumerate(t7pos[:len(curves["frozen14m"])]):  # same 5k bins as test 7
        r = {"position": r7["position"], "frozen31m": r7["frozen31m"], "frozen70m": r7["frozen70m"]}
        r.update({k: "%.4f" % v[j] for k, v in curves.items()})
        pos.append(r)
    saved = {"frozen14m_%s" % k: v for k, v in frozen.items()}
    saved.update({"%s_%s" % k: v for k, v in losses.items()})
    np.savez_compressed(os.path.join(out_dir, "losses.npz"), **{k: v.astype(np.float16) for k, v in saved.items()})
    return rows, bins, pos, seconds, check


def main():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    model, init, info, _ = setup()
    vsets, vstreams = val_sets()
    seconds, rows = {}, []
    chunks = vsets["val_pick"]
    order = [(b, c) for b in range(chunks.shape[0]) for c in range(chunks.shape[1])]
    t = time.time()
    f_val = float(np.nanmean(run(model, init, chunks, order, lr=0)))
    seconds["frozen14m_val_pick"] = time.time() - t
    val = {}
    for name, fast in [("full_1e-4", None)] + [(n, fast_of(p, a, l)) for n, p, a, l in CONFIGS]:
        for stream in ("val_pick_K1", "val_pick_shuffled"):
            set_name, order = vstreams[stream]
            t = time.time()
            val[name, stream] = float(np.nanmean(dual_losses(model, init, vsets[set_name], order, fast)))
            seconds["%s_%s" % (name, stream)] = time.time() - t
            print(name, stream, "%.4f" % val[name, stream], "%.0f s" % seconds["%s_%s" % (name, stream)], flush=True)
            rows.append({"phase": "val", "config": name, "stream": stream, "loss": "%.4f" % val[name, stream],
                         "gain_frozen": "%.4f" % (f_val - val[name, stream]),
                         "gain_full": "%.4f" % (val["full_1e-4", stream] - val[name, stream]),
                         "seconds": "%.0f" % seconds["%s_%s" % (name, stream)]})
    rows.append({"phase": "val", "config": "frozen14m", "stream": "val_pick", "loss": "%.4f" % f_val,
                 "seconds": "%.0f" % seconds["frozen14m_val_pick"]})
    names = [n for n, *_ in CONFIGS]
    pick_k1 = min(names, key=lambda n: val[n, "val_pick_K1"])
    pick_sh = min(names, key=lambda n: val[n, "val_pick_shuffled"])
    picks = [pick_k1] + ([pick_sh] if pick_sh != pick_k1 else [])
    print("picked:", pick_k1, "best on val shuffled:", pick_sh, flush=True)
    cfg = {n: (p, a, l) for n, p, a, l in CONFIGS}
    sets, streams = test_sets()
    trows, bins, pos, tsec, check = evaluate_test(model, init, sets, streams,
                                                  {n: fast_of(*cfg[n]) for n in picks}, OUT)
    seconds.update(tsec)
    write_csv(OUT, "results.csv", rows + trows)
    write_csv(OUT, "bins.csv", bins)
    write_csv(OUT, "position.csv", pos)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write(json.dumps({"pick_val_K1": pick_k1, "best_val_shuffled": pick_sh,
                            "full_1e-4_reproduces_test7": check}) + "\n")
        f.write("%s\n%s\n" % (info["device"], ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    plot()


def plot():
    rows = list(csv.DictReader(open(os.path.join(OUT, "results.csv"))))
    pos = list(csv.DictReader(open(os.path.join(OUT, "position.csv"))))
    info = json.loads(open(os.path.join(OUT, "runtime.txt")).readline())
    style()
    names = [n for n, *_ in CONFIGS]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(names))[::-1]
    for k, (stream, color, label) in enumerate((("val_pick_K1", BLUE, "val-pick K = 1"),
                                                ("val_pick_shuffled", ORANGE, "val-pick shuffled"))):
        g = [float(next(r["gain_full"] for r in rows if r["phase"] == "val" and r["config"] == n and r["stream"] == stream))
             for n in names]
        ax.barh(y - (k - 0.5) * 0.38, g, height=0.36, color=color, label=label)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [n.replace("_a", ", alpha ").replace("_l", ", lambda ") for n in names], fontsize=7)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("gain over full_1e-4 on val-pick (nats/token)")
    ax.legend(frameon=False, fontsize=7, loc="center left")
    ax.set_title("Dual configs on val-pick (picked: %s)" % info["pick_val_K1"], fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "val_pick.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xs = np.arange(len(pos)) * 5 + 2.5
    for key, label, ls in (("frozen14m", "frozen 14M", "-"), ("frozen31m", "frozen 31M (test 7)", "--")):
        ax.plot(xs, [float(r[key]) for r in pos], color=INK, lw=1.4, ls=ls, label=label)
    runs = [k for k in pos[0] if k not in ("position", "frozen14m", "frozen31m", "frozen70m")]
    for key, color in zip(runs, (BLUE, ORANGE, AQUA)):
        ax.plot(xs, [float(r[key]) for r in pos], color=color, lw=1.6, marker="o", ms=2.5, label=key)
    ax.set_xlabel("position in the book (thousands of tokens)")
    ax.set_ylabel("loss (nats/token)")
    ax.set_xlim(0, len(pos) * 5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Long stream: loss by position in the book, mean of 5 books", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "loss_vs_position_long.png"), dpi=160)


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else main()
