"""Registry and runner for every stream run.

A run is (size, rule, lr, parameter subset, stream); rule "frozen" is the pretrained model with no
updates. Learning rates come from e0_lr_picks/picks.csv. Each run saves runs/<key>.npz:
    loss       per-token stream losses [chunks, L], float16, in reading order (NaN at chunk starts)
    order      [chunks, 2]: (book, chunk) of each row
    snaps      (snapshot runs) per-chunk mean losses on the held-out text (detokenized WikiText, then
               the 6 held-out books, 15 chunks each), one row per snapshot: snapshot 0 = pretrained,
               then every EVERY = 40 chunks and at the end
    positions  (snapshot runs) chunks read at each snapshot
    meta       json: spec, seconds, device
Wall time per run is appended to runs/runtime.txt. Analyses only read these files (load()).

Usage (from experiments/):
    ../venv/bin/python runs.py list                 # registry, with the missing runs marked
    ../venv/bin/python runs.py run <key> [...]      # run some (keys from list; "missing" = all missing)
    ../venv/bin/python runs.py worker               # run missing runs, longest first; start up to 3 in parallel
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import numpy as np  # noqa: E402

from data import HERE, L, heldout, stream  # noqa: E402

RUNS = os.path.join(HERE, "runs")
PICKS = os.path.join(HERE, "e0_lr_picks", "picks.csv")
EVERY = 40
SIZES = ("14m", "31m", "70m")
RULES = ("sgd", "adam", "nsgd")
E2_SUBSETS = ["output", "no_output", "mlp", "attention"] + ["mlp@%d" % i for i in range(6)]


def picks():
    """(size, rule, params) -> picked lr."""
    if not os.path.exists(PICKS):
        return {}
    return {(r["size"], r["rule"], r["params"]): float(r["lr"]) for r in csv.DictReader(open(PICKS))
            if r["picked"] == "True"}


def key_of(size, rule, lr, params, stream_name):
    if rule == "frozen":
        return "%s__%s_frozen" % (stream_name, size)
    return "%s__%s_%s_%g_%s" % (stream_name, size, rule, lr, params)


def spec(size, rule, stream_name, params="all", snapshots=False, pk=None):
    """Registry entry; lr from the picks (None if not picked yet)."""
    lr = 0.0 if rule == "frozen" else (pk or {}).get((size, rule, params))
    return {"size": size, "rule": rule, "lr": lr, "params": params, "stream": stream_name, "snapshots": snapshots,
            "key": key_of(size, rule, lr, params, stream_name) if lr is not None else None}


def registry():
    """The full list of stream runs."""
    pk = picks()
    out = []
    for size in SIZES:
        out.append(spec(size, "frozen", "main_K1"))
    for st in ("main_K1", "main_shuffled"):
        for size in SIZES:
            for rule in RULES:
                out.append(spec(size, rule, st, snapshots=True, pk=pk))
    for params in E2_SUBSETS:
        out.append(spec("14m", "sgd", "main_K1", params, pk=pk))
    for st in ("main_K5", "main_K25"):
        out.append(spec("14m", "sgd", st, pk=pk))
    out += [spec("14m", "frozen", "long"), spec("14m", "sgd", "long", pk=pk)]
    out += [spec(size, "frozen", "verylong") for size in SIZES]
    out += [spec("14m", "sgd", "verylong", snapshots=True, pk=pk), spec("31m", "sgd", "verylong", pk=pk)]
    return out


def find(size, rule, stream_name, params="all"):
    """Key of a registry run (frozen runs: rule "frozen")."""
    for s in registry():
        if (s["size"], s["rule"], s["stream"], s["params"]) == (size, rule, stream_name, params):
            if s["key"] is None:
                raise SystemExit("no pick yet for %s" % s)
            return s["key"]
    raise KeyError((size, rule, stream_name, params))


def frozen_key(size, stream_name):
    """The frozen run whose losses are the baseline of a stream (main streams share main K1's chunks)."""
    return key_of(size, "frozen", 0, "all", "main_K1" if stream_name.startswith("main_") else stream_name)


def path_of(key):
    return os.path.join(RUNS, key + ".npz")


def chunk_means(loss):
    return np.nanmean(loss.reshape(-1, L), axis=1)


def run(s):
    from learner import Learner, score, setup
    model, init, info = setup(s["size"])
    chunks, order = stream(s["stream"])
    ids = np.concatenate([chunks[b, k] for b, k in order])
    t0 = time.time()
    extra = {}
    if s["rule"] == "frozen":
        loss = score(model, init, ids)
    elif not s["snapshots"]:
        loss = Learner(model, init, s["rule"], s["lr"], s["params"]).read(ids)
    else:
        held, n_wiki = heldout()
        lrn = Learner(model, init, s["rule"], s["lr"], s["params"])
        snaps, pos, losses = [chunk_means(score(model, init, held))], [0], []
        for a in range(0, len(order), EVERY):
            e = min(a + EVERY, len(order))
            losses.append(lrn.read(ids[a * L:e * L]))
            snaps.append(chunk_means(lrn.score(held)))
            pos.append(e)
        loss = np.concatenate(losses)
        extra = {"snaps": np.array(snaps, dtype=np.float32), "positions": np.array(pos), "n_wiki": n_wiki}
    sec = time.time() - t0
    meta = {k: v for k, v in s.items() if k != "key"}
    meta.update(seconds=round(sec), device=info["device"], commit=info["commit"])
    os.makedirs(RUNS, exist_ok=True)
    np.savez_compressed(path_of(s["key"]), loss=loss.reshape(-1, L).astype(np.float16),
                        order=np.array(order, dtype=np.int16), meta=json.dumps(meta), **extra)
    with open(os.path.join(RUNS, "runtime.txt"), "a") as f:
        f.write("%s %d s\n" % (s["key"], sec))
    print("%s: mean loss %.4f, %.0f s" % (s["key"], np.nanmean(loss), sec), flush=True)


def load(key):
    """{loss [books, chunks, L] float64 in book layout, snaps, positions, n_wiki, meta}."""
    z = np.load(path_of(key))
    order = z["order"].astype(np.int64)
    n_books, n_chunks = order[:, 0].max() + 1, order[:, 1].max() + 1
    loss = np.full((n_books, n_chunks, L), np.nan)
    loss[order[:, 0], order[:, 1]] = z["loss"].astype(np.float64)
    out = {"loss": loss, "meta": json.loads(str(z["meta"]))}
    if "snaps" in z.files:
        out.update(snaps=z["snaps"].astype(np.float64), positions=z["positions"], n_wiki=int(z["n_wiki"]))
    return out


def seconds(key):
    return load_meta(key)["seconds"]


def load_meta(key):
    return json.loads(str(np.load(path_of(key))["meta"]))


def compare(new, old):
    """Float16 comparison of two loss arrays of the same layout: share identical, max abs diff."""
    a, b = np.asarray(new, dtype=np.float16), np.asarray(old, dtype=np.float16)
    m = np.isfinite(b.astype(np.float32))
    return {"identical_share": float((a[m] == b[m]).mean()),
            "max_abs_diff": float(np.abs(a[m].astype(np.float32) - b[m].astype(np.float32)).max())}


def main():
    cmd = sys.argv[1]
    reg = registry()
    if cmd == "list":
        for s in reg:
            state = "needs pick" if s["key"] is None else ("done" if os.path.exists(path_of(s["key"])) else "missing")
            print("%-10s %s" % (state, s["key"] or "%(stream)s %(size)s %(rule)s %(params)s" % s))
    elif cmd == "run":
        keys = sys.argv[2:]
        if keys == ["missing"]:
            keys = [s["key"] for s in reg if s["key"] and not os.path.exists(path_of(s["key"]))]
        by_key = {s["key"]: s for s in reg if s["key"]}
        for k in keys:
            run(by_key[k])
    elif cmd == "worker":  # take missing runs one by one (longest first); start several workers in parallel
        cost = {"14m": 1, "31m": 2, "70m": 4}
        todo = sorted([s for s in reg if s["key"]], key=lambda s: -(cost[s["size"]] * (3 if s["snapshots"] else 1) *
                                                                     (1.6 if s["stream"] == "verylong" else 1) *
                                                                     (0.2 if s["rule"] == "frozen" else 1)))
        for s in todo:
            lock = os.path.join(RUNS, ".lock_" + s["key"])
            if os.path.exists(path_of(s["key"])):
                continue
            try:
                os.makedirs(lock)
            except FileExistsError:
                continue
            run(s)


if __name__ == "__main__":
    main()
