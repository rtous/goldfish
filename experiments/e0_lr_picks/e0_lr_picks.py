"""E0: learning-rate picks for every run of the registry.

Pick stream: val-pick (the first 10 PG-19 validation books with at least 20k tokens, 78 chunks each,
K = 1: 780 chunks); never test books. For each (size, rule, parameter subset): the rate with the
lowest mean online loss on a 1-3-10 ladder, starting from a 3-5 value grid and extended one step at a
time until the best value is interior (both of its ladder neighbours tried and worse). A NaN run counts
as worse than any finite one.

Groups (one process each, run in parallel):
    14m          14M sgd / adam / nsgd, all parameters
    14m_subsets  14M sgd on each E2 subset: output only, all but output, MLP only, attention only,
                 MLP of layer 1 ... 6
    31m, 70m     sgd / adam / nsgd, all parameters

Writes e0_lr_picks/rows_<group>.csv (every rate tried) and, with report, picks.csv (all rows, picked
column) read by runs.py. Usage (from experiments/):
    ../venv/bin/python e0_lr_picks/e0_lr_picks.py run <group>
    ../venv/bin/python e0_lr_picks/e0_lr_picks.py report
"""
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))

import numpy as np  # noqa: E402

from data import stream  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
LADDER = [float("%de%d" % (m, e)) for e in range(-10, 0) for m in (1, 3)]  # exact decimal literals
GRIDS = {("14m", "sgd"): [3e-5, 1e-4, 3e-4], ("31m", "sgd"): [1e-4, 3e-4, 1e-3], ("70m", "sgd"): [3e-6, 1e-5, 3e-5],
         "adam": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4], "nsgd": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3], "subset": [1e-4, 3e-4, 1e-3]}
SUBSETS = ["output", "no_output", "mlp", "attention"] + ["mlp@%d" % i for i in range(6)]
GROUPS = {"14m": [("14m", r, "all") for r in ("sgd", "adam", "nsgd")],
          "14m_subsets": [("14m", "sgd", p) for p in SUBSETS],
          "31m": [("31m", r, "all") for r in ("sgd", "adam", "nsgd")],
          "70m": [("70m", r, "all") for r in ("sgd", "adam", "nsgd")]}


def snap(lr):
    """The ladder value equal to lr (floating-point safe)."""
    return min(LADDER, key=lambda v: abs(math.log(v / lr)))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def run(group):
    from learner import Learner, score, setup
    chunks, order = stream("val_pick")
    ids = np.concatenate([chunks[b, k] for b, k in order])
    rows, model, size_loaded = [], None, None
    path = os.path.join(OUT, "rows_%s.csv" % group)
    for size, rule, params in GROUPS[group]:
        if size != size_loaded:
            model, init, info = setup(size)
            size_loaded = size
            rows.append({"size": size, "rule": "frozen", "params": "all", "lr": 0,
                         "val_pick_loss": "%.4f" % np.nanmean(score(model, init, ids))})
            print(rows[-1], flush=True)
        grid = GRIDS["subset"] if params != "all" else GRIDS.get((size, rule), GRIDS.get(rule))
        res = {}

        def evaluate(lr):
            t = time.time()
            loss = float(np.nanmean(Learner(model, init, rule, lr, params).read(ids)))
            res[lr] = loss if np.isfinite(loss) else float("inf")
            rows.append({"size": size, "rule": rule, "params": params, "lr": lr, "val_pick_loss": "%.4f" % loss,
                         "extension": lr not in grid, "seconds": round(time.time() - t)})
            print(rows[-1], flush=True)
            write_csv(path, rows)

        for lr in grid:
            evaluate(snap(lr))
        while True:  # extend until the best value is interior
            best = min(res, key=res.get)
            i = LADDER.index(best)
            todo = [LADDER[j] for j in (i - 1, i + 1) if 0 <= j < len(LADDER) and LADDER[j] not in res]
            if not todo:
                break
            for lr in todo:
                evaluate(lr)
        best = min(res, key=res.get)
        for r in rows:
            if r["rule"] == rule and r["params"] == params and r["size"] == size:
                r["picked"] = r["lr"] == best
        write_csv(path, rows)


def report():
    rows = []
    for g in GROUPS:
        p = os.path.join(OUT, "rows_%s.csv" % g)
        if not os.path.exists(p):
            print("missing %s (group not run yet)" % p)
            continue
        rows += list(csv.DictReader(open(p)))
    for r in rows:
        r["lr"] = "%g" % float(r["lr"])
        r.setdefault("picked", "")
    write_csv(os.path.join(OUT, "picks.csv"), rows)
    print("%-4s %-5s %-10s %-8s %s" % ("size", "rule", "params", "pick", "rates tried (val-pick loss)"))
    for g in GROUPS:
        for size, rule, params in GROUPS[g]:
            rs = sorted([r for r in rows if (r["size"], r["rule"], r["params"]) == (size, rule, params)],
                        key=lambda r: float(r["lr"]))
            pick = next((r["lr"] for r in rs if r["picked"] == "True"), "running")
            print("%-4s %-5s %-10s %-8s %s" % (size, rule, params, pick,
                                              ", ".join("%s %s" % (r["lr"], r["val_pick_loss"]) for r in rs)))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "report": report}[sys.argv[1]]()
