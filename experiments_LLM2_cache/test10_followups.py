"""Test 10: follow-ups. Error bars (part A), a wider dual grid (part B), longer contexts (part C).

Part A (no runs): standard errors across books for tests 1, 4, 6, 7 and 8, from their saved
per-token losses. For each run, the gain over frozen per book (mean over that book's scored
tokens), then mean and SE = sd / sqrt(books) over books; for comparisons of two runs on the same
chunks, the paired per-book differences. The same for two bins: previous occurrence in the same
book under 10k tokens back (earlier chunk), and never seen (test 1 / test 7 definitions, 1000 most
frequent ids excluded), keeping only books with at least MIN_BIN tokens in the bin. A comparison is
flagged when |mean| < 2 SE. Tokens of a shuffled stream are assigned to their book as usual.
Test 7's K5 and K25 runs saved no per-token losses, so they have no SE.

Part B: dual configs (test 8's rule, dual.py) on val-pick K = 1 and shuffled: layer 4 MLP alpha 3e-4
with lambda 0.99 and 0.9; layer 2 MLP alpha 1e-3 and 3e-4 with lambda 0.99; test 8's pick again for
reference. The best on val-pick K = 1 among all dual configs of tests 8 and 10 is picked; if it is
at an edge of the values tried for its layer (alpha on the 1-3-10 ladder; lambda on 0.5, 0.9, 0.99,
0.999), the grid is extended once by one step outward in each such direction and the pick redone.
The pick runs on the test streams K1, shuffled and long.

Part C: "very long" chunk set: the first 4 test books (dataset order, check books excluded) with at
least 200k tokens, cut to 200k, chunked per book (781 chunks each), read one after the other. Runs:
frozen 14M, 31M and 70M; 14M full_1e-4; 14M with part B's pick; 31M full online learning at lr
3e-4 (test 6's pick for 31M).

Part A also writes errorbars_distance.csv: token-level means per distance bin (test 1's bins and test 5's
same-book / earlier-book split) for the test-1 stream runs of tests 1, 3, 4, 5 and 6, with SE over books
(books with at least MIN_BIN tokens in the bin), and adds test 9's Meta-SGD lite comparisons to
errorbars.csv.

Writes test10_followups/: errorbars.csv, errorbars_distance.csv (A), partB_val.csv, partB_test.csv (B), partC.csv,
partC_position.csv (C), losses_B.npz, losses_C.npz (float16 [book, chunk, position]), runtime_*.txt
and plots/. Usage:
    python test10_followups.py A | B | C | C_pairs | plot   (C_pairs: paired differences from losses_C.npz)
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
from online import L, setup
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test1_distance import TOP
from test3_baselines import AQUA
from test7_aggressive import N_BOOKS, chunk, run, token_classes
from test8_dual import CONFIGS as CONFIGS8, TEST_STREAMS, test_sets, write_csv

OUT = os.path.join(HERE, "test10_followups")
T = {k: os.path.join(HERE, v) for k, v in (("1", "test1_distance"), ("3", "test3_baselines"), ("4", "test4_checks"),
                                          ("5", "test5_capacity"), ("6", "test6"), ("7", "test7_aggressive"),
                                          ("8", "test8_dual"))}
MIN_BIN = 50
BINS = ("all", "same book < 10k", "never seen")
ALPHAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
LAMS = [0.5, 0.9, 0.99, 0.999]
NEW_B = [("mlp4", 3, 3e-4, 0.99), ("mlp4", 3, 3e-4, 0.9), ("mlp2", 1, 1e-3, 0.99), ("mlp2", 1, 3e-4, 0.99)]
VERY_LONG = (4, 200000)
POS_C = 10000
LR31 = 3e-4


def name_of(part, alpha, lam):
    return "%s_a%g_l%g" % (part, alpha, lam)


# ---------------------------------------------------------------- part A


def per_book(diff, book, mask, n_books):
    """Per-book means of diff over masked tokens; books with fewer than MIN_BIN masked tokens dropped (bins)."""
    out = []
    for b in range(n_books):
        m = mask & (book == b)
        if m.sum() >= MIN_BIN:
            out.append(float(diff[m].mean()))
    return np.array(out)


def summarize(test, stream, comparison, diff, book, masks, n_books):
    rows = []
    for name in BINS:
        x = per_book(diff, book, masks[name], n_books)
        mean, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
        rows.append({"test": test, "stream": stream, "comparison": comparison, "bin": name, "books": len(x),
                     "mean": "%.4f" % mean, "se": "%.4f" % se, "under_2se": abs(mean) < 2 * se})
    return rows


def classes_stream(ids, book):
    """Bins for test 1's stream (stream order): same book < 10k (earlier chunk), never seen."""
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    t = np.arange(len(ids))
    frequent = np.isin(ids, np.argsort(-np.bincount(ids))[:TOP])
    same_book = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    near = (prev >= 0) & (prev < t - t % L) & same_book & (t - prev < 10000) & ~frequent
    return near, (prev < 0) & ~frequent


def distance_masks(ids, book):
    """Test 1's distance bins (1000 most frequent ids excluded) and test 5's same-book / earlier-book split of
    the earlier-chunk bins, for the test-1 stream; plus "all tokens"."""
    from test1_distance import BINS as DBINS, distance_bins
    bins = distance_bins(ids)
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    same_book = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    masks = {"all tokens": np.ones(len(ids), bool)}
    for k, name in enumerate(DBINS):
        masks[name] = bins == k
        if name.startswith("earlier chunk"):
            masks[name + ", same book"] = (bins == k) & same_book
            masks[name + ", earlier book"] = (bins == k) & ~same_book
    return masks


def distance_rows(test, comparison, diff, book, masks, scored, n_books=25):
    """Token-level mean of diff per bin, with SE over books (per-book means, books with >= MIN_BIN tokens)."""
    rows = []
    for name, m in masks.items():
        m = m & scored
        if m.sum() < MIN_BIN:
            continue
        per = per_book(diff, book, m, n_books)
        se = float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan")
        mean = float(diff[m].mean())
        rows.append({"test": test, "comparison": comparison, "bin": name, "tokens": int(m.sum()), "books": len(per),
                     "token_mean": "%.4f" % mean, "se": "%.4f" % se, "under_2se": bool(abs(mean) < 2 * se)})
    return rows


def part_a():
    rows = []
    t1 = np.load(os.path.join(T["1"], "losses.npz"))
    ids, book, frozen = t1["token"], t1["book"], t1["frozen"]
    scored = np.isfinite(frozen)
    near, never = classes_stream(ids, book)
    masks = {"all": scored, "same book < 10k": scored & near, "never seen": scored & never}
    t4, t6 = np.load(os.path.join(T["4"], "losses.npz")), np.load(os.path.join(T["6"], "losses.npz"))
    t3, t5 = np.load(os.path.join(T["3"], "losses.npz")), np.load(os.path.join(T["5"], "losses.npz"))
    run1 = {"online": t1["online"], "shuffled": t4["shuffled"], "mlp": t4["mlp"], "attention": t4["attention"],
            "online_cache": t4["online_cache"], "unigram": t3["unigram"], "output_lr3e-3": t4["output_lr3e-3"]}
    n = 25

    def add(test, stream, comparison, diff):
        rows.extend(summarize(test, stream, comparison, diff, book, masks, n))

    add("1", "stream", "online - frozen", frozen - run1["online"])
    for k in ("shuffled", "mlp", "attention", "online_cache", "unigram", "output_lr3e-3"):
        add("4", "stream", "%s - frozen" % k, frozen - run1[k])
    add("4", "stream", "locality (online - shuffled)", run1["shuffled"] - run1["online"])
    add("4", "stream", "online_cache - online", run1["online"] - run1["online_cache"])
    add("4", "stream", "online_cache - unigram", run1["unigram"] - run1["online_cache"])
    add("4", "stream", "mlp - attention", run1["attention"] - run1["mlp"])
    for size, fz, on, sh in (("31M", t6["frozen31"], t6["online31"], t6["shuffled31"]),
                             ("70M", t5["frozen70"], t6["online70"], t6["shuffled70"])):
        add("6", "stream", "%s online - frozen" % size, fz - on)
        add("6", "stream", "%s locality" % size, sh - on)
    add("6", "stream", "31M locality - 14M locality",
        (t6["shuffled31"] - t6["online31"]) - (run1["shuffled"] - run1["online"]))
    for k in ["mlp_l%d" % i for i in range(1, 7)] + ["mlp_last2"]:
        add("6", "stream", "%s - frozen" % k, frozen - t6[k])
        add("6", "stream", "%s - online (all params)" % k, run1["online"] - t6[k])

    # Tests 7 and 8: [book, chunk, position] arrays
    d7, d8 = np.load(os.path.join(T["7"], "losses.npz")), np.load(os.path.join(T["8"], "losses.npz"))
    cls = {s: token_classes(d7["tokens_" + s].astype(np.int64)) for s in ("books25", "long")}

    def add3(test, stream, comparison, set_name, diff):
        bk = np.broadcast_to(np.arange(diff.shape[0])[:, None, None], diff.shape).reshape(-1)
        near3, never3 = cls[set_name]
        sc = np.isfinite(diff).reshape(-1)
        m = {"all": sc, "same book < 10k": sc & near3.reshape(-1), "never seen": sc & never3.reshape(-1)}
        rows.extend(summarize(test, stream, comparison, diff.reshape(-1).astype(np.float64), bk, m, diff.shape[0]))

    settings7 = sorted({k.rsplit("_", 1)[0] for k in d7.files if not k.startswith(("frozen", "tokens"))})
    for stream in ("long", "K1", "shuffled"):
        set_name = "long" if stream == "long" else "books25"
        f14 = d7["frozen14m_%s" % set_name].astype(np.float64)
        full = d7["full_1e-4_%s" % stream].astype(np.float64)
        for s in settings7:
            r = d7["%s_%s" % (s, stream)].astype(np.float64)
            add3("7", stream, "%s - frozen" % s, set_name, f14 - r)
            if s != "full_1e-4":
                add3("7", stream, "%s - full_1e-4" % s, set_name, full - r)
        for size in ("31m", "70m"):
            add3("7", stream, "frozen %s - frozen 14M" % size, set_name, f14 - d7["frozen%s_%s" % (size, set_name)])
    duals = sorted({k.rsplit("_", 1)[0] for k in d8.files if k.startswith("mlp")})
    for stream in TEST_STREAMS:
        set_name = "long" if stream == "long" else "books25"
        f14 = d8["frozen14m_%s" % set_name].astype(np.float64)
        full = d8["full_1e-4_%s" % stream].astype(np.float64)
        add3("8", stream, "full_1e-4 - frozen", set_name, f14 - full)
        for dname in duals:
            r = d8["%s_%s" % (dname, stream)].astype(np.float64)
            add3("8", stream, "%s - frozen" % dname, set_name, f14 - r)
            add3("8", stream, "%s - full_1e-4" % dname, set_name, full - r)
    for dname in ["full_1e-4"] + duals:  # locality per book: same chunks in K1 and shuffled
        add3("8", "K1 vs shuffled", "%s locality" % dname, "books25",
             d8["%s_shuffled" % dname].astype(np.float64) - d8["%s_K1" % dname].astype(np.float64))
    for dname in duals:
        add3("8", "K1 vs shuffled", "%s locality - full_1e-4 locality" % dname, "books25",
             (d8["%s_shuffled" % dname].astype(np.float64) - d8["%s_K1" % dname].astype(np.float64))
             - (d8["full_1e-4_shuffled"].astype(np.float64) - d8["full_1e-4_K1"].astype(np.float64)))
    # Test 9 (Meta-SGD lite) against frozen, full_1e-4 and test 8's pick, same bins as test 8
    d9 = np.load(os.path.join(HERE, "test9_metasgd_lite", "losses.npz"))
    pick8 = json.loads(open(os.path.join(T["8"], "runtime.txt")).readline())["pick_val_K1"]
    for stream in TEST_STREAMS:
        set_name = "long" if stream == "long" else "books25"
        m9 = d9["metasgd_lite_%s" % stream].astype(np.float64)
        add3("9", stream, "metasgd_lite - frozen", set_name, d8["frozen14m_%s" % set_name].astype(np.float64) - m9)
        add3("9", stream, "metasgd_lite - full_1e-4", set_name, d8["full_1e-4_%s" % stream].astype(np.float64) - m9)
        add3("9", stream, "metasgd_lite - test 8 pick", set_name, d8["%s_%s" % (pick8, stream)].astype(np.float64) - m9)
    add3("9", "K1 vs shuffled", "metasgd_lite locality", "books25",
         d9["metasgd_lite_shuffled"].astype(np.float64) - d9["metasgd_lite_K1"].astype(np.float64))
    os.makedirs(OUT, exist_ok=True)
    write_csv(OUT, "errorbars.csv", rows)

    # Distance bins for the test-1 stream runs: token-level means with SE over books
    dmasks = distance_masks(ids, book)
    drows = []
    runs = {"online": t1["online"], "shuffled": t4["shuffled"], "mlp": t4["mlp"], "attention": t4["attention"],
            "online_cache": t4["online_cache"], "unigram": t3["unigram"], "output_lr1e-3": t3["output"],
            "no_output": t3["no_output"], "output_lr3e-3": t4["output_lr3e-3"]}
    runs.update({k: t6[k] for k in ["mlp_l%d" % i for i in range(1, 7)] + ["mlp_last2"]})
    for k, r in runs.items():
        drows += distance_rows("1-6", "%s - frozen" % k, frozen - r, book, dmasks, scored)
    drows += distance_rows("1-6", "locality 14M", run1["shuffled"] - run1["online"], book, dmasks, scored)
    for size, fz, on, sh in (("31M", t6["frozen31"], t6["online31"], t6["shuffled31"]),
                             ("70M", t5["frozen70"], t6["online70"], t6["shuffled70"])):
        drows += distance_rows("1-6", "%s online - frozen %s" % (size, size), fz - on, book, dmasks, scored)
        drows += distance_rows("1-6", "locality %s" % size, sh - on, book, dmasks, scored)
    drows += distance_rows("1-6", "online_cache - online", run1["online"] - run1["online_cache"], book, dmasks, scored)
    write_csv(OUT, "errorbars_distance.csv", drows)
    print("distance rows %d" % len(drows))
    print("%d rows; flagged under 2 SE: %d" % (len(rows), sum(r["under_2se"] for r in rows)))


# ---------------------------------------------------------------- part B


def part_b():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    model, init, info, _ = setup()
    vsets, vstreams = val_sets()
    val8 = {(r["config"], r["stream"]): float(r["loss"]) for r in csv.DictReader(open(os.path.join(T["8"], "results.csv")))
            if r["phase"] == "val"}
    # name -> (part, 0-based layer or None for all six MLPs, alpha, lam); test 8's configs first
    configs = {n: (p, 3 if p == "mlp4" else None, a, l) for n, p, a, l in CONFIGS8}
    names8 = set(configs)
    pick8 = json.loads(open(os.path.join(T["8"], "runtime.txt")).readline())["pick_val_K1"]
    seconds, val = {}, {}

    def fast(part, layer, alpha, lam):
        names = mlp_weights([layer]) if layer is not None else mlp_weights(range(6))
        return {n: (alpha, lam) for n in names}

    def run_val(name, part, layer, alpha, lam):
        configs[name] = (part, layer, alpha, lam)
        for stream in ("val_pick_K1", "val_pick_shuffled"):
            set_name, order = vstreams[stream]
            t = time.time()
            val[name, stream] = float(np.nanmean(dual_losses(model, init, vsets[set_name], order, fast(part, layer, alpha, lam))))
            seconds["%s_%s" % (name, stream)] = time.time() - t
            print(name, stream, "%.4f" % val[name, stream], flush=True)

    run_val(pick8 + " (rerun)", *configs[pick8])
    for part, layer, alpha, lam in NEW_B:
        run_val(name_of(part, alpha, lam), part, layer, alpha, lam)
    for n in configs:
        if n.endswith("(rerun)") or (n, "val_pick_K1") in val:
            continue
        val[n, "val_pick_K1"], val[n, "val_pick_shuffled"] = val8[n, "val_pick_K1"], val8[n, "val_pick_shuffled"]
    candidates = [n for n in configs if not n.endswith("(rerun)")]
    pick = min(candidates, key=lambda n: val[n, "val_pick_K1"])
    part, layer, alpha, lam = configs[pick]
    tried = [configs[n] for n in candidates if configs[n][0] == part]
    alphas, lams = sorted({c[2] for c in tried}), sorted({c[3] for c in tried})
    ext = []
    if alpha == alphas[0] and ALPHAS.index(alpha) > 0:
        ext.append((part, layer, ALPHAS[ALPHAS.index(alpha) - 1], lam))
    if alpha == alphas[-1] and ALPHAS.index(alpha) < len(ALPHAS) - 1:
        ext.append((part, layer, ALPHAS[ALPHAS.index(alpha) + 1], lam))
    if lam == lams[-1] and LAMS.index(lam) < len(LAMS) - 1:
        ext.append((part, layer, alpha, LAMS[LAMS.index(lam) + 1]))
    if lam == lams[0] and LAMS.index(lam) > 0:
        ext.append((part, layer, alpha, LAMS[LAMS.index(lam) - 1]))
    ext = [e for e in ext if name_of(e[0], e[2], e[3]) not in configs]
    print("first pick:", pick, "extensions:", [name_of(e[0], e[2], e[3]) for e in ext], flush=True)
    for e in ext:
        run_val(name_of(e[0], e[2], e[3]), *e)
    candidates = [n for n in configs if not n.endswith("(rerun)")]
    final = min(candidates, key=lambda n: val[n, "val_pick_K1"])
    vrows = [{"config": n, "source": "test 8" if n in names8 else "test 10",
              "part": configs[n][0], "layer": (configs[n][1] + 1) if configs[n][1] is not None else "all",
              "alpha": configs[n][2], "lam": configs[n][3],
              "val_pick_K1": "%.4f" % val[n, "val_pick_K1"], "val_pick_shuffled": "%.4f" % val[n, "val_pick_shuffled"],
              "gain_full_K1": "%.4f" % (val8["full_1e-4", "val_pick_K1"] - val[n, "val_pick_K1"]),
              "gain_full_shuffled": "%.4f" % (val8["full_1e-4", "val_pick_shuffled"] - val[n, "val_pick_shuffled"])}
             for n in configs]
    write_csv(OUT, "partB_val.csv", vrows)
    sets, streams = test_sets()
    d8 = np.load(os.path.join(T["8"], "losses.npz"))
    trows, saved = [], {}
    for stream in TEST_STREAMS:
        set_name, order = streams[stream]
        t = time.time()
        loss = dual_losses(model, init, sets[set_name], order, fast(*configs[final]))
        seconds["%s_%s" % (final, stream)] = time.time() - t
        saved["%s_%s" % (final, stream)] = loss
        f14 = d8["frozen14m_%s" % set_name].astype(np.float64)
        full, p8 = d8["full_1e-4_%s" % stream].astype(np.float64), d8["%s_%s" % (pick8, stream)].astype(np.float64)
        cls = token_classes(sets[set_name])
        for comp, base in (("gain over frozen 14M", f14), ("vs full_1e-4", full), ("vs test 8 pick", p8)):
            diff = base - loss.astype(np.float64)
            per = [np.nanmean(diff[b]) for b in range(diff.shape[0])]
            row = {"config": final, "stream": stream, "comparison": comp, "mean": "%.4f" % np.mean(per),
                   "se": "%.4f" % (np.std(per, ddof=1) / np.sqrt(len(per))), "books": len(per),
                   "token_mean": "%.4f" % np.nanmean(diff)}
            for bname, m in zip(("same book < 10k", "never seen"), cls):
                per_b = [np.nanmean(diff[b][m[b]]) for b in range(diff.shape[0]) if m[b].sum() >= MIN_BIN]
                row["%s mean" % bname] = "%.4f" % np.mean(per_b)
                row["%s se" % bname] = "%.4f" % (np.std(per_b, ddof=1) / np.sqrt(len(per_b)))
                row["%s books" % bname] = len(per_b)
            trows.append(row)
        print(final, stream, "%.4f" % np.nanmean(loss), flush=True)
    write_csv(OUT, "partB_test.csv", trows)
    np.savez_compressed(os.path.join(OUT, "losses_B.npz"), **{k: v.astype(np.float16) for k, v in saved.items()})
    fpart, flayer, falpha, flam = configs[final]
    with open(os.path.join(OUT, "runtime_B.txt"), "w") as f:
        f.write(json.dumps({"first_pick": pick, "extensions": [name_of(e[0], e[2], e[3]) for e in ext],
                            "pick": final, "pick_config": [fpart, flayer + 1 if flayer is not None else "all",
                                                           falpha, flam]}) + "\n")
        f.write("%s\n" % ", ".join("%s %.0f s" % kv for kv in seconds.items()))


# ---------------------------------------------------------------- part C


def part_c():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    pick = json.loads(open(os.path.join(OUT, "runtime_B.txt")).readline())
    part, layer, alpha, lam = pick["pick_config"]
    fast = {n: (alpha, lam) for n in (mlp_weights([layer - 1]) if layer != "all" else mlp_weights(range(6)))}
    chunks = chunk(long_books(DEFAULT, n_books=VERY_LONG[0], n_tokens=VERY_LONG[1])["ids"])
    order = [(b, c) for b in range(chunks.shape[0]) for c in range(chunks.shape[1])]
    losses, seconds = {}, {}
    for size, name in (("14m", DEFAULT), ("31m", "EleutherAI/pythia-31m"), ("70m", "EleutherAI/pythia-70m")):
        model, init, info, _ = setup(name)
        t = time.time()
        losses["frozen" + size] = run(model, init, chunks, order, lr=0)
        seconds["frozen" + size] = time.time() - t
        print("frozen", size, "%.4f" % np.nanmean(losses["frozen" + size]), "%.0f s" % seconds["frozen" + size], flush=True)
        if size == "14m":
            for key, f in (("14m full_1e-4", None), ("14m dual " + pick["pick"], fast)):
                t = time.time()
                losses[key] = dual_losses(model, init, chunks, order, f)
                seconds[key] = time.time() - t
                print(key, "%.4f" % np.nanmean(losses[key]), "%.0f s" % seconds[key], flush=True)
        if size == "31m":
            t = time.time()
            losses["31m online lr3e-4"] = dual_losses(model, init, chunks, order, None, slow_lr=LR31)
            seconds["31m online lr3e-4"] = time.time() - t
            print("31m online", "%.4f" % np.nanmean(losses["31m online lr3e-4"]), flush=True)
    f14, f31 = losses["frozen14m"].astype(np.float64), losses["frozen31m"].astype(np.float64)
    rows = []
    for key, loss in losses.items():
        diff = f14 - loss.astype(np.float64)
        per = [np.nanmean(diff[b]) for b in range(diff.shape[0])]
        gap = float(np.nanmean(f14) - np.nanmean(loss)) / float(np.nanmean(f14) - np.nanmean(f31))
        rows.append({"run": key, "loss": "%.4f" % np.nanmean(loss), "gain_over_frozen14m": "%.4f" % np.mean(per),
                     "se": "%.4f" % (np.std(per, ddof=1) / np.sqrt(len(per))), "gap31_closed": "%.4f" % gap,
                     "vs_frozen31m": "%.4f" % (np.nanmean(f31) - np.nanmean(loss)), "seconds": "%.0f" % seconds[key]})
    write_csv(OUT, "partC.csv", rows)
    pos = []
    flat = {k: v.reshape(v.shape[0], -1) for k, v in losses.items()}
    n_bins = (flat["frozen14m"].shape[1] + POS_C - 1) // POS_C
    for j in range(n_bins):
        r = {"position": "%dk-%dk" % (j * POS_C // 1000, (j + 1) * POS_C // 1000)}
        for k, v in flat.items():
            per = [np.nanmean(row[j * POS_C:(j + 1) * POS_C]) for row in v]
            r[k] = "%.4f" % np.mean(per)
            r[k + " se"] = "%.4f" % (np.std(per, ddof=1) / np.sqrt(len(per)))
        pos.append(r)
    write_csv(OUT, "partC_position.csv", pos)
    np.savez_compressed(os.path.join(OUT, "losses_C.npz"), **{k.replace(" ", "_"): v.astype(np.float16)
                                                              for k, v in losses.items()}, tokens=chunks)
    with open(os.path.join(OUT, "runtime_C.txt"), "w") as f:
        f.write(json.dumps({"books": long_books(DEFAULT, n_books=VERY_LONG[0], n_tokens=VERY_LONG[1])["book_ids"],
                            "dual": pick["pick"]}) + "\n")
        f.write("%s\n" % ", ".join("%s %.0f s" % kv for kv in seconds.items()))


def part_c_pairs():
    """Paired per-book differences between part C runs (reanalysis of losses_C.npz): partC_pairs.csv."""
    d = np.load(os.path.join(OUT, "losses_C.npz"))
    dual = next(k for k in d.files if k.startswith("14m_dual"))
    pairs = [("14M dual - 14M full_1e-4", "14m_full_1e-4", dual),
             ("31M online - 14M full_1e-4", "14m_full_1e-4", "31m_online_lr3e-4"),
             ("31M online - frozen 31M", "frozen31m", "31m_online_lr3e-4"),
             ("14M full_1e-4 - frozen 14M", "frozen14m", "14m_full_1e-4"),
             ("31M online - frozen 70M", "frozen70m", "31m_online_lr3e-4"),
             ("frozen 31M - 14M full_1e-4", "14m_full_1e-4", "frozen31m")]
    rows = []
    for name, base, run_ in pairs:
        diff = d[base].astype(np.float64) - d[run_].astype(np.float64)  # positive: run_ has lower loss
        per = [np.nanmean(diff[b]) for b in range(diff.shape[0])]
        rows.append({"comparison": name, "token_mean": "%.4f" % np.nanmean(diff), "se": "%.4f" % (np.std(per, ddof=1) / np.sqrt(len(per))),
                     "books": len(per)})
    write_csv(OUT, "partC_pairs.csv", rows)
    for r in rows:
        print(r)


# ---------------------------------------------------------------- plots


def plot():
    style()
    if os.path.exists(os.path.join(OUT, "partC_position.csv")):
        pos = list(csv.DictReader(open(os.path.join(OUT, "partC_position.csv"))))
        keys = [k for k in pos[0] if k != "position" and not k.endswith(" se")]
        fig, ax = plt.subplots(figsize=(7.8, 3.6))
        xs = np.arange(len(pos)) * POS_C / 1000 + POS_C / 2000
        styles = {"frozen14m": (INK, "-", "frozen 14M"), "frozen31m": (INK, "--", "frozen 31M"),
                  "frozen70m": (INK, ":", "frozen 70M"), "14m full_1e-4": (BLUE, "-", "14M online (full_1e-4)"),
                  "31m online lr3e-4": (AQUA, "-", "31M online (lr 3e-4)")}
        for k in keys:
            color, ls, label = styles.get(k, (ORANGE, "-", "14M dual (%s)" % k.split(" ", 2)[-1]))
            y = np.array([float(r[k]) for r in pos])
            se = np.array([float(r[k + " se"]) for r in pos])
            if not k.startswith("frozen"):
                ax.fill_between(xs, y - se, y + se, color=color, alpha=0.15, lw=0)
            ax.plot(xs, y, color=color, ls=ls, lw=1.6, marker=None if k.startswith("frozen") else "o", ms=2.5,
                    label=label)
        ax.set_xlabel("position in the book (thousands of tokens)")
        ax.set_ylabel("loss (nats/token)")
        ax.set_xlim(0, len(pos) * POS_C / 1000)
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_title("Very long stream (4 books x 200k): loss by position, mean of 4 books (band: +- 1 SE)",
                     fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "loss_vs_position_verylong.png"), dpi=160)
    if os.path.exists(os.path.join(OUT, "partB_test.csv")):
        rows = list(csv.DictReader(open(os.path.join(OUT, "partB_test.csv"))))
        eb = list(csv.DictReader(open(os.path.join(OUT, "errorbars.csv"))))
        pick8 = json.loads(open(os.path.join(T["8"], "runtime.txt")).readline())["pick_val_K1"]
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        x = np.arange(len(TEST_STREAMS))
        r8 = [next(r for r in eb if r["test"] == "8" and r["stream"] == s and r["comparison"] == "%s - full_1e-4" % pick8
                   and r["bin"] == "all") for s in TEST_STREAMS]
        rb = [next(r for r in rows if r["stream"] == s and r["comparison"] == "vs full_1e-4") for s in TEST_STREAMS]
        for k, (rr, color, label) in enumerate(((r8, BLUE, "test 8 pick (%s)" % pick8),
                                                (rb, ORANGE, "test 10 pick (%s)" % rows[0]["config"]))):
            ax.bar(x + (k - 0.5) * 0.36, [float(r["mean"]) for r in rr], width=0.34, color=color, label=label,
                   yerr=[2 * float(r["se"]) for r in rr], error_kw={"elinewidth": 1, "ecolor": MUTED, "capsize": 3})
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, ["K = 1", "shuffled", "long"])
        ax.set_ylabel("gain over full_1e-4 (nats/token)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Dual picks on the test streams (bars: +- 2 SE over books)", fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "dual_picks.png"), dpi=160)


if __name__ == "__main__":
    {"A": part_a, "B": part_b, "C": part_c, "C_pairs": part_c_pairs, "plot": plot}[sys.argv[1]]()
