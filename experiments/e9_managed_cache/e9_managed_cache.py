"""E9: a managed cache (fast store + eviction + fallback) on Pythia-14M.

The whole model stays frozen; one MLP layer learns online as a store W = W_frozen + D (store.py), with
eviction by decay (D <- (1 - lam) D after each step) and by flush (D = 0 when the store has been worse
than frozen over the last chunks), and a fallback: each token is predicted by a fixed-share mixture of
frozen + store and frozen (mixer.py), and the mixture is what is scored. The store learns on its own
loss, never the mixture's, so the mixer does not change the store: a store run gives both the store-only
configuration and the same store with the fallback.

Layer: the MLP of layer 2 (mlp@1), E2's single layer with the most gain beyond the window (+0.131 at
1k-10k same book, the only one positive at 10k-100k, the smallest loss on never-seen tokens); store lr
1e-3 = its E0 pick.

Configurations (14M; plain SGD on the store):
    1 frozen                   registry run (main streams); computed here on the mixed stream
    2 all                      online, all parameters, lr 1e-4: registry runs; computed here on mixed
    3 store                    lam 0, no flush, no mixer (must match the registry mlp@1 run)
    4 decay                    store + decay
    5 decay_flush              store + decay + flush
    6 store_fallback           store (3) + mixer
    7 full                     decay + flush (5) + mixer
Streams: main_K1, main_shuffled (registry), mixed (streams.py: the main books with a block of 4 WikiText
chunks after each). Snapshots (every 40 chunks, held-out WikiText + 6 held-out books, as E5) on main_K1
and mixed; not on shuffled.
Hyperparameters (pick): on valmixed only (val-pick books + WikiText train blocks), by mean loss over all
its scored tokens: decay (lr, lam), then flush (threshold, window) at the decay pick, then the mixer's
alpha for 6 (on 3's losses) and 7 (on 5's). Grids in grid_*.csv, picks in picks.json.

Usage (from experiments/):
    ../venv/bin/python e9_managed_cache/e9_managed_cache.py check
    ../venv/bin/python e9_managed_cache/e9_managed_cache.py pick
    ../venv/bin/python e9_managed_cache/e9_managed_cache.py run <job> [...]     # jobs: see JOBS; "all"
    ../venv/bin/python e9_managed_cache/e9_managed_cache.py report
"""
import csv
import json
import os
import sys
import time

OUT = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(OUT)
sys.path[:0] = [HERE, os.path.join(HERE, "lib"), OUT]

import numpy as np  # noqa: E402

import runs  # noqa: E402
import streams  # noqa: E402
from data import L, heldout  # noqa: E402
from mixer import check as mixer_check, fixed_share  # noqa: E402

RUNS = os.path.join(OUT, "runs")
LAYER = 1
SUBSET = "mlp@%d" % LAYER
GRID_LR = [1e-3, 3e-3]
GRID_LAM = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
GRID_THRESHOLD = [0.02, 0.05, 0.1, 0.2]
GRID_WINDOW = [1, 2, 4]
GRID_ALPHA = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
SNAP_STREAMS = ("main_K1", "mixed")
JOBS = ["%s:%s" % (st, j) for st in ("main_K1", "main_shuffled", "mixed") for j in ("store", "decay", "decay_flush")] + \
       ["mixed:all", "mixed:frozen"]


def store_lr():
    return runs.picks()[("14m", "sgd", SUBSET)]


def path_of(key):
    return os.path.join(RUNS, key + ".npz")


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def save(key, loss, order, meta, **extra):
    """runs.run's npz layout: loss [chunks, L] float16 in reading order, order [chunks, 2], meta json."""
    os.makedirs(RUNS, exist_ok=True)
    np.savez_compressed(path_of(key), loss=np.asarray(loss).reshape(-1, L).astype(np.float16),
                        order=np.array(order, dtype=np.int16), meta=json.dumps(meta), **extra)
    print("%s: mean loss %.4f, %s s" % (key, np.nanmean(np.asarray(loss, dtype=np.float64)), meta.get("seconds")), flush=True)


def load(key):
    """{loss [units, chunks, L] float64 (NaN where absent), weight (if any), snaps..., meta}."""
    z = np.load(path_of(key))
    order = z["order"].astype(np.int64)
    shape = (order[:, 0].max() + 1, order[:, 1].max() + 1, L)
    out = {"meta": json.loads(str(z["meta"])), "order": order}
    for name in ("loss", "weight"):
        if name in z.files:
            a = np.full(shape, np.nan)
            a[order[:, 0], order[:, 1]] = z[name].astype(np.float64)
            out[name] = a
    for name in ("snaps", "positions", "flushes"):
        if name in z.files:
            out[name] = z[name]
    if "n_wiki" in z.files:
        out["n_wiki"] = int(z["n_wiki"])
    return out


# ---------------------------------------------------------------- check


def check():
    from learner import score, setup
    from store import run_store
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    for name, ok, detail in mixer_check():
        say("mixer: %s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))
    model, init, info = setup("14m")
    ids, order, _ = streams.get("main_K1")
    lr = store_lr()
    t0 = time.time()
    r = run_store(model, init, ids, lr, LAYER)
    sec = time.time() - t0
    ref_key = runs.find("14m", "sgd", "main_K1", SUBSET)
    ref = runs.load(ref_key)["loss"]
    o = np.array(order)
    new = np.full(ref.shape, np.nan)
    new[o[:, 0], o[:, 1]] = r["loss"]
    c = runs.compare(new, ref)
    say("store (layer %d MLP, lr %g, lam 0, no flush, no mixer) vs %s: %s, %.0f s" % (LAYER + 1, lr, ref_key, c, sec))
    frozen = score(model, init, ids.reshape(-1)).reshape(len(ids), L)
    mix, _ = fixed_share(r["loss"], frozen, 0.01, force=0.0)
    fz = runs.load(runs.frozen_key("14m", "main_K1"))["loss"]
    new[o[:, 0], o[:, 1]] = mix
    c2 = runs.compare(new, fz)
    say("mixer forced to frozen (weight on store 0) vs %s: %s" % (runs.frozen_key("14m", "main_K1"), c2))
    with open(os.path.join(OUT, "check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- picks (validation only)


def pick():
    from learner import score, setup
    from store import run_store
    model, init, info = setup("14m")
    ids, order, n_books = streams.get("valmixed")
    frozen = score(model, init, ids.reshape(-1)).reshape(len(ids), L)
    wiki = np.array([u >= n_books for u, _ in order])

    def row(loss, **kw):
        return {**kw, "val_loss": "%.4f" % np.nanmean(loss), "val_loss_books": "%.4f" % np.nanmean(loss[~wiki]),
                "val_loss_wiki": "%.4f" % np.nanmean(loss[wiki]), "frozen_wiki": "%.4f" % np.nanmean(frozen[wiki])}

    t0 = time.time()
    lr0 = store_lr()
    store = run_store(model, init, ids, lr0, LAYER)["loss"]
    rows = [row(frozen, config="frozen"), row(store, config="store", lr=lr0, lam=0)]
    best = None
    for lr in GRID_LR:
        for lam in GRID_LAM:
            loss = run_store(model, init, ids, lr, LAYER, lam=lam)["loss"]
            rows.append(row(loss, config="decay", lr=lr, lam=lam))
            print(rows[-1], flush=True)
            if best is None or np.nanmean(loss) < best[0]:
                best = (np.nanmean(loss), lr, lam)
    write_csv("grid_decay.csv", rows)
    _, lr, lam = best
    frows, fbest = [], None
    for thr in GRID_THRESHOLD:
        for win in GRID_WINDOW:
            r = run_store(model, init, ids, lr, LAYER, lam=lam, flush=(thr, win), frozen=frozen)
            frows.append(row(r["loss"], config="decay_flush", lr=lr, lam=lam, threshold=thr, window=win,
                             flushes=len(r["flushes"])))
            print(frows[-1], flush=True)
            if fbest is None or np.nanmean(r["loss"]) < fbest[0]:
                fbest = (np.nanmean(r["loss"]), thr, win, r["loss"])
    write_csv("grid_flush.csv", frows)
    _, thr, win, flush_loss = fbest
    arows, apick = [], {}
    for config, base in (("store_fallback", store), ("full", flush_loss)):
        scores = []
        for a in GRID_ALPHA:
            mix, w = fixed_share(base, frozen, a)
            arows.append(row(mix, config=config, alpha=a, mean_weight_books="%.3f" % np.nanmean(w[~wiki]),
                             mean_weight_wiki="%.3f" % np.nanmean(w[wiki])))
            scores.append(np.nanmean(mix))
        apick[config] = GRID_ALPHA[int(np.argmin(scores))]
    write_csv("grid_alpha.csv", arows)
    picks = {"layer": LAYER, "store_lr": lr0, "decay": {"lr": lr, "lam": lam},
             "flush": {"lr": lr, "lam": lam, "threshold": thr, "window": win},
             "alpha": apick, "seconds": round(time.time() - t0)}
    with open(os.path.join(OUT, "picks.json"), "w") as f:
        json.dump(picks, f, indent=1)
    print(picks)


# ---------------------------------------------------------------- runs (test streams)


def run_job(job):
    from learner import score, setup
    from store import run_all_params, run_store
    st, kind = job.split(":")
    pk = json.load(open(os.path.join(OUT, "picks.json")))
    model, init, info = setup("14m")
    ids, order, n_books = streams.get(st)
    snaps = st in SNAP_STREAMS
    held, n_wiki = heldout()
    t0 = time.time()
    frozen = score(model, init, ids.reshape(-1)).reshape(len(ids), L)
    held_frozen = score(model, init, held) if snaps else None
    base_meta = {"stream": st, "size": "14m", "device": info["device"], "commit": info["commit"]}
    snap_extra = lambda r, key="snaps": ({"snaps": np.array(r[key], dtype=np.float32), "positions": np.array(r["positions"]),
                                          "n_wiki": n_wiki} if snaps else {})
    if kind == "frozen":
        save("%s__frozen" % st, frozen, order, {**base_meta, "config": "frozen", "seconds": round(time.time() - t0)})
        return
    if kind == "all":
        lr = runs.picks()[("14m", "sgd", "all")]
        r = run_all_params(model, init, ids, lr, held, held_frozen)
        save("%s__all" % st, r["loss"], order, {**base_meta, "config": "all", "rule": "sgd", "lr": lr, "params": "all",
                                                 "seconds": round(time.time() - t0)}, **snap_extra(r))
        return
    hp = {"store": {"lr": pk["store_lr"], "lam": 0.0, "flush": None, "mix": "store_fallback"},
          "decay": {"lr": pk["decay"]["lr"], "lam": pk["decay"]["lam"], "flush": None, "mix": None},
          "decay_flush": {"lr": pk["flush"]["lr"], "lam": pk["flush"]["lam"],
                          "flush": (pk["flush"]["threshold"], pk["flush"]["window"]), "mix": "full"}}[kind]
    alpha = pk["alpha"][hp["mix"]] if hp["mix"] else None
    r = run_store(model, init, ids, hp["lr"], LAYER, lam=hp["lam"], flush=hp["flush"], frozen=frozen,
                  held=held if snaps else None, held_frozen=held_frozen, alphas=(alpha,) if alpha else ())
    sec = round(time.time() - t0)
    meta = {**base_meta, "config": kind, "layer": LAYER, "params": SUBSET, "lr": hp["lr"], "lam": hp["lam"],
            "flush": hp["flush"], "seconds": sec}
    save("%s__%s" % (st, kind), r["loss"], order, meta, flushes=r["flushes"], **snap_extra(r))
    if alpha:
        mix, w = fixed_share(r["loss"], frozen, alpha)
        mmeta = {**meta, "config": hp["mix"], "alpha": alpha, "seconds": sec, "shared_run": "%s__%s" % (st, kind)}
        extra = {}
        if snaps:
            extra = {"snaps": np.array(r["snaps_mix"][alpha], dtype=np.float32), "positions": np.array(r["positions"]),
                     "n_wiki": n_wiki}
        save("%s__%s" % (st, hp["mix"]), mix, order, mmeta, weight=w.astype(np.float16), flushes=r["flushes"], **extra)


def run_cmd(args):
    todo = JOBS if args == ["all"] else args
    for job in todo:
        st, kind = job.split(":")
        if os.path.exists(path_of("%s__%s" % (st, kind))):
            continue
        run_job(job)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "pick":
        pick()
    elif cmd == "run":
        run_cmd(sys.argv[2:])
    elif cmd == "report":
        from report import report
        report()
