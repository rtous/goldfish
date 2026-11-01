"""Tune the online hyperparameters on the tuning stream (sequence_<stream>_tune.csv), never on the
evaluation stream that results.csv is scored on.

For one stream and one variant, every setting of a grid is run online at every model size, from
the offline weights of the experiment's <stream>_<h>_frozen configurations (train them with
mnist_mlp.py first). "stream" is the prefix of those configuration names: "rotated" in
experiment 1, "rotated_L1000" in experiment 2 (the tuning sequence is the configuration's own). The online gain is online - frozen accuracy on the tuning stream. One setting
per stream and variant is chosen: the best mean gain over the sizes.

Variants: sgd (grid over --lrs) and the fast-weight variants fastbias, fastout, fasthidden,
fastall (grid over --fast-lrs x --decays, with the slow SGD learning rate fixed at --slow-lr).

Every run is a row of <experiment>/tuning.csv (rows with the same stream, variant, setting and
size are replaced). --report prints the tables (gain per size and mean, best setting marked)
from tuning.csv without running anything.

Usage:
    python3 tune.py 1 rotated sgd
    python3 tune.py 1 rotated fasthidden --slow-lr 0.003
    python3 tune.py 1 rotated fasthidden --slow-lr 0.003 --fast-lrs 0.0003 --decays 0.9
    python3 tune.py 1 --report
    python3 tune.py 2 rotated_L1000 sgd --sizes 8 4 2 --lrs 0.0001 0.0003 0.001 0.003 0.01
"""
import argparse
import contextlib
import csv
import fcntl
import io
import os
import time

import numpy as np
import torch

from mnist_mlp import load_configurations, set_experiment
from online_mlp import Stream, run

SIZES = (128, 32, 8, 4, 2)
FAST_LAYERS = {
    "fastbias": ["2.bias"],
    "fastout": ["2.weight", "2.bias"],
    "fasthidden": ["0.weight", "0.bias"],
    "fastall": ["0.weight", "0.bias", "2.weight", "2.bias"],
}
VARIANTS = ("sgd",) + tuple(FAST_LAYERS)
FIELDS = ["stream", "variant", "slow_lr", "fast_lr", "decay", "hidden_size", "frozen", "online", "gain"]
KEY = ["stream", "variant", "slow_lr", "fast_lr", "decay", "hidden_size"]


def save_rows(path, new_rows):
    """Merge rows into tuning.csv under a file lock (several tunings may run in parallel)."""
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                rows = list(csv.DictReader(f))
        keys = {tuple(str(r[k]) for k in KEY) for r in new_rows}
        rows = [r for r in rows if tuple(r[k] for k in KEY) not in keys] + new_rows
        rows.sort(key=lambda r: (r["stream"], VARIANTS.index(r["variant"]), float(r["slow_lr"]),
                                 float(r["fast_lr"] or 0), float(r["decay"] or 0), -int(r["hidden_size"])))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    os.remove(path + ".lock")


def report(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault((r["stream"], r["variant"]), {}).setdefault(
            (r["slow_lr"], r["fast_lr"], r["decay"]), {})[int(r["hidden_size"])] = r["gain"]
    for (stream, variant), settings in groups.items():
        sizes = sorted({h for g in settings.values() for h in g}, reverse=True)

        def mean(g):
            vals = [g.get(h) for h in sizes]
            if any(v in (None, "diverged") for v in vals):
                return float("-inf")
            return float(np.mean([float(v) for v in vals]))
        best = max(settings, key=lambda s: mean(settings[s]))
        print("\n%s / %s: online gain on the tuning stream (pp)" % (stream, variant))
        print("| slow lr | fast lr | decay | " + " | ".join("h = %d" % h for h in sizes) + " | mean |")
        print("|---" * (4 + len(sizes)) + "|")
        for s in sorted(settings, key=lambda s: (float(s[0]), float(s[1] or 0), float(s[2] or 0))):
            g = settings[s]
            cells = [("%+.2f" % float(g[h])) if g.get(h) not in (None, "diverged") else g.get(h, "-")
                     for h in sizes]
            m = mean(g)
            m_text = "-" if m == float("-inf") else "%+.2f" % m
            if s == best:
                m_text = "**%s**" % m_text
            print("| %s | %s | %s | %s | %s |" % (s[0], s[1] or "-", s[2] or "-", " | ".join(cells), m_text))
        print("best: slow lr %s, fast lr %s, decay %s, mean gain %+.2f" % (best[0], best[1] or "-", best[2] or "-",
                                                                           mean(settings[best])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("stream", nargs="?", help="configuration prefix: rotated, permuted, rotated_L1000, ...")
    parser.add_argument("variant", nargs="?", choices=VARIANTS)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.0003, 0.001, 0.003, 0.01])
    parser.add_argument("--slow-lr", type=float, help="slow SGD lr of the fast variants (the tuned sgd lr)")
    parser.add_argument("--fast-lrs", nargs="+", type=float, default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--decays", nargs="+", type=float, default=[0.9, 0.97, 0.99])
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--report", action="store_true", help="only print the tables of tuning.csv")
    args = parser.parse_args()
    exp_dir = set_experiment(args.experiment)
    path = os.path.join(exp_dir, "tuning.csv")
    if args.report:
        return report(path)
    if not args.stream or not args.variant:
        parser.error("stream and variant are required unless --report")
    if args.variant != "sgd" and args.slow_lr is None:
        parser.error("--slow-lr is required for the fast variants")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    if args.variant == "sgd":
        settings = [({"optimizer": "sgd", "learning_rate": lr}, (lr, "", "")) for lr in args.lrs]
    else:
        settings = [({"optimizer": "sgd", "learning_rate": args.slow_lr,
                      "fast": {"learning_rate": f, "decay": d, "layers": FAST_LAYERS[args.variant]}},
                     (args.slow_lr, f, d)) for f in args.fast_lrs for d in args.decays]
    bases = load_configurations(["%s_%d_frozen" % (args.stream, h) for h in args.sizes])
    stream = Stream(bases[0], "tune", device)
    for online_params, (slow, fast_lr, decay) in settings:
        rows, start = [], time.time()
        for base in bases:
            config = dict(base, online_params=online_params)  # same name: loads that config's weights
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(config, device, stream)
            h = base["model_params"]["hidden_size"]
            row = {"stream": args.stream, "variant": args.variant, "slow_lr": slow, "fast_lr": fast_lr,
                   "decay": decay, "hidden_size": h}
            if result is None:
                row.update({"frozen": "", "online": "diverged", "gain": "diverged"})
            else:
                row.update({"frozen": "%.2f" % result["frozen"], "online": "%.2f" % result["accuracy"],
                            "gain": "%.2f" % (result["accuracy"] - result["frozen"])})
            rows.append(row)
        save_rows(path, rows)
        print("%s %s slow %g fast %s decay %s: gains %s  (%.0f s)" % (
            args.stream, args.variant, slow, fast_lr or "-", decay or "-",
            " ".join(r["gain"] for r in rows), time.time() - start), flush=True)


if __name__ == "__main__":
    main()
