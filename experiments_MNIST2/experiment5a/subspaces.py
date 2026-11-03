"""Experiment 5a: which part of the network holds the context (rotation) knowledge?

For h in {8, 4}, each of the 8 contexts and each subset of parameters: start from the mixed model,
fine-tune only that subset on the training set with every image under that one context (Adam,
lr 1e-3, batch 128, EPOCHS epochs, seed 0), and score on the test set under the same context.
Writes results.csv (mean over contexts and per context).

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment5a/subspaces.py
"""
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contexts import N_CONTEXTS, transform  # noqa: E402
from data import load_mixed, load_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (8, 4)
EPOCHS, BATCH, LR, SEED = 5, 128, 1e-3, 0
MIXED = {8: 74.05, 4: 64.78}        # mixed model, mean over contexts (%)
PER_CONTEXT = {8: 93.26, 4: 87.31}  # per-context models trained from scratch (%)
SUBSETS = {  # subset -> trainable parameters (layer names of models/mlp.py)
    "none": [],
    "out_bias": ["2.bias"],
    "out_layer": ["2.weight", "2.bias"],
    "hidden_bias": ["0.bias"],
    "first_rank1": ["u", "v"],  # W1 = S1 + u v^T
    "first_layer": ["0.weight", "0.bias"],
    "all": ["0.weight", "0.bias", "2.weight", "2.bias"],
}


def forward(p, x):
    w1 = p["0.weight"] + p["u"][:, None] * p["v"][None, :] if "u" in p else p["0.weight"]
    return F.linear(F.relu(F.linear(x, w1, p["0.bias"])), p["2.weight"], p["2.bias"])


def run(state, subset, x_train, y_train, x_test, y_test):
    """Fine-tune the subset on one context; return (test accuracy %, number of trainable params)."""
    torch.manual_seed(SEED)
    p = {k: v.clone() for k, v in state.items()}
    if subset == "first_rank1":
        h = p["0.weight"].shape[0]
        p["u"] = torch.zeros(h)
        p["v"] = torch.randn(28 * 28) * 0.01
    names = SUBSETS[subset]
    for k in names:
        p[k].requires_grad_(True)
    if names:
        opt = torch.optim.Adam([p[k] for k in names], lr=LR)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(x_train))
            for i in range(0, len(x_train), BATCH):
                idx = perm[i:i + BATCH]
                loss = F.cross_entropy(forward(p, x_train[idx]), y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
    with torch.no_grad():
        acc = (forward(p, x_test).argmax(1) == y_test).float().mean().item() * 100
    return acc, sum(p[k].numel() for k in names)


def main():
    x_train_raw, y_train = load_split("train")
    x_test_raw, y_test = load_split("test")
    acc = {(h, s): [None] * N_CONTEXTS for h in SIZES for s in SUBSETS}
    n_params = {}
    for c in range(N_CONTEXTS):
        x_train = transform(x_train_raw, torch.full((len(x_train_raw),), c, dtype=torch.long))
        x_test = transform(x_test_raw, torch.full((len(x_test_raw),), c, dtype=torch.long))
        for h in SIZES:
            state = load_mixed(h)
            for s in SUBSETS:
                acc[h, s][c], n_params[h, s] = run(state, s, x_train, y_train, x_test, y_test)
                print("ctx %d  h=%d  %-12s %6.2f%%" % (c, h, s, acc[h, s][c]), flush=True)

    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h", "subset", "n_params", "acc", "recovered"] + ["acc_ctx%d" % c for c in range(N_CONTEXTS)])
        for h in SIZES:
            for s in SUBSETS:
                mean = sum(acc[h, s]) / N_CONTEXTS
                rec = round((mean - MIXED[h]) / (PER_CONTEXT[h] - MIXED[h]), 3) + 0.0  # no "-0.000"
                w.writerow([h, s, n_params[h, s], "%.2f" % mean, "%.3f" % rec] + ["%.2f" % a for a in acc[h, s]])
                print("h=%d  %-12s n=%5d  acc %.2f%%  recovered %.3f" % (h, s, n_params[h, s], mean, rec))


if __name__ == "__main__":
    main()
