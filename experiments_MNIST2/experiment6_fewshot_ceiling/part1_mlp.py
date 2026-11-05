"""Experiment 6, part 1: few-shot oracle on the current MLP (784 -> h -> 10, h = 8 and 4).

Starts from weights/mixed_h<h>.pt and fine-tunes either every parameter ("all") or the first
layer only ("first") on N images of one context (see common.fewshot). Writes the best grid cell
per N to results.csv and the whole grid to grids/part1_h<h>_<variant>.csv.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment6_fewshot_ceiling/part1_mlp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, Net, accuracy, best_per_n, fewshot, n_params, save_grid, update_results  # noqa: E402
from contexts import under_every_context  # noqa: E402
from data import load_mixed, load_split  # noqa: E402

VARIANTS = {"all": ["layers.0.weight", "layers.0.bias", "head.weight", "head.bias"],
            "first": ["layers.0.weight", "layers.0.bias"]}


def mixed_mlp(h):
    model = Net("mlp", (h,))
    sd = load_mixed(h)
    model.load_state_dict({"layers.0.weight": sd["0.weight"], "layers.0.bias": sd["0.bias"],
                           "head.weight": sd["2.weight"], "head.bias": sd["2.bias"]})
    return model


def main():
    os.makedirs(os.path.join(HERE, "grids"), exist_ok=True)
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    cache = under_every_context(x_test)
    for h in (8, 4):
        model = mixed_mlp(h)
        mixed = accuracy(model, cache, y_test)
        rows = [dict(part=1, budget=h, arch="mlp1", r=0, params=n_params(model), measure="mixed", acc="%.2f" % mixed)]
        for variant, names in VARIANTS.items():
            grid = fewshot(model, names, x_train, y_train, cache, y_test, log="h=%d %s" % (h, variant))
            save_grid("part1_h%d_%s" % (h, variant), grid)
            for N, (a, lr, s) in best_per_n(grid).items():
                rows.append(dict(part=1, budget=h, arch="mlp1", r=0, params=n_params(model),
                                 measure="oracle_" + variant, N=N, acc="%.2f" % a, gain="%.2f" % (a - mixed),
                                 lr=lr, steps=s))
        update_results(rows)


if __name__ == "__main__":
    main()
