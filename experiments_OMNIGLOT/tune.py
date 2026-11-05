"""Tune the label prior (control 2a) and the cache (control 3) on the val_alphabets stream (val
drawers 11-12, never the test drawers), with each architecture's frozen mixed model. Writes
<experiment>/tuned.json: {arch: {"prior": {decay, beta, eps, val_acc}, "cache": {M: {temperature,
mix, val_acc}}, "frozen_val_acc"}}.

Grids:
    prior   decay {0.9, 0.95, 0.98, 0.99, 0.995, 0.999} x beta {0.25, 0.5, 1, 2, 4} x eps {0.01, 0.1, 1}
    cache   temperature {0.01, 0.03, 0.1, 0.3} x mix {0.1, 0.25, 0.5, 0.75, 0.9}, for M in {100, 500}

Usage:
    python tune.py 1        # every architecture (python tune.py 1 A: only A)
"""
import itertools
import json
import os
import sys

import torch

from data import experiment_dir, load_config, load_mixed, load_split, load_stream
from models.mlp import forward
from online import run_cache, run_prior

DECAYS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
BETAS = (0.25, 0.5, 1, 2, 4)
EPSS = (0.01, 0.1, 1)
TEMPERATURES = (0.01, 0.03, 0.1, 0.3)
MIXES = (0.1, 0.25, 0.5, 0.75, 0.9)
MS = (100, 500)


def main():
    exp_dir = experiment_dir(sys.argv[1])
    config = load_config(exp_dir)
    archs = sys.argv[2:] or config["archs"]  # optional: only these architectures
    x, y = load_split("val")
    idx = torch.from_numpy(load_stream("val_alphabets")[0])
    x, y = x[idx], y[idx].numpy()
    path = os.path.join(exp_dir, "tuned.json")
    tuned = json.load(open(path)) if os.path.exists(path) else {}
    for arch in archs:
        with torch.no_grad():
            logits, feats = forward(load_mixed(arch), x, hidden=True)
        logits, feats = logits.numpy(), feats.numpy()
        frozen = (logits.argmax(1) == y).mean() * 100
        best = max((run_prior(logits, y, d, b, e).mean() * 100, d, b, e)
                   for d, b, e in itertools.product(DECAYS, BETAS, EPSS))
        out = {"frozen_val_acc": round(frozen, 2),
               "prior": {"val_acc": round(best[0], 2), "decay": best[1], "beta": best[2], "eps": best[3]},
               "cache": {}}
        print("%s val: frozen %.2f%%, prior %s" % (arch, frozen, out["prior"]), flush=True)
        for M in MS:
            c = max((run_cache(logits, feats, y, M, t, m).mean() * 100, t, m)
                    for t, m in itertools.product(TEMPERATURES, MIXES))
            out["cache"][str(M)] = {"val_acc": round(c[0], 2), "temperature": c[1], "mix": c[2]}
            print("%s val: cache M=%d %s" % (arch, M, out["cache"][str(M)]), flush=True)
        tuned[arch] = out
    with open(path, "w") as f:
        json.dump(tuned, f, indent=2)


if __name__ == "__main__":
    main()
