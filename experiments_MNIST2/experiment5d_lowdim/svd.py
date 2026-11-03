"""Experiment 5d, part 1: how low-dimensional are experiment 5c's context parts D_k?

For each h: flatten each D_k (0.weight and 0.bias) into a row, stack the 8 rows and take the SVD
(no centering). Report the fraction of the total squared norm explained by the top r directions.
Then, for r in {1, 2, 4, 8}, with B = the top r right singular vectors (fixed): for each context
k, start from c_k = the projection of D_k onto B and fine-tune only c_k (r numbers) on the
training set under context k, with first layer S1 + sum_i c_ki B_i (S1, B and the head fixed;
Adam lr 1e-2, batch 128, 2 epochs, seed 0). Score on the test set under context k, then average.
r = 0 is S1 alone. Writes the explained and ceiling columns of results.csv.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment5d_lowdim/svd.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RANKS, SIZES, update_results  # noqa: E402
from contexts import N_CONTEXTS, transform  # noqa: E402
from data import HERE as ROOT, load_init, load_split  # noqa: E402

EPOCHS, BATCH, LR, SEED = 2, 128, 1e-2, 0


def forward(p, x, delta):
    """delta: [785 h], the flattened context part (0.weight rows, then 0.bias)."""
    h = p["0.bias"].numel()
    w1 = p["0.weight"] + delta[:784 * h].view(h, 784)
    b1 = p["0.bias"] + delta[784 * h:]
    return F.linear(F.relu(F.linear(x, w1, b1)), p["2.weight"], p["2.bias"])


def accuracy(p, x, y, delta):
    with torch.no_grad():
        return (forward(p, x, delta).argmax(1) == y).float().mean().item() * 100


def fine_tune(p, B, c, x, y):
    torch.manual_seed(SEED)
    c = c.clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=LR)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(forward(p, x[idx], c @ B), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return c.detach()


def main():
    x_train_raw, y_train = load_split("train")
    x_test_raw, y_test = load_split("test")
    S = {h: load_init("experiment5c/weights/general_h<h>.pt", h) for h in SIZES}
    D, B, explained = {}, {}, {}
    for h in SIZES:
        d = torch.load(os.path.join(ROOT, "experiment5c", "weights", "context_h%d.pt" % h))
        D[h] = torch.cat([d["0.weight"].flatten(1), d["0.bias"]], 1)  # [8, 785 h]
        _, sv, Vt = torch.linalg.svd(D[h], full_matrices=False)
        share = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        explained[h] = {r: (share[r - 1].item() if r else 0.0) for r in RANKS}
        B[h] = Vt
        print("h=%d singular values %s; explained by top 1/2/4/8: %s" % (
            h, " ".join("%.2f" % s for s in sv), " ".join("%.3f" % explained[h][r] for r in RANKS[1:])), flush=True)

    acc = {(h, r): [] for h in SIZES for r in RANKS}
    for k in range(N_CONTEXTS):
        x_train = transform(x_train_raw, torch.full((len(x_train_raw),), k, dtype=torch.long))
        x_test = transform(x_test_raw, torch.full((len(x_test_raw),), k, dtype=torch.long))
        for h in SIZES:
            for r in RANKS:
                if r == 0:
                    a = accuracy(S[h], x_test, y_test, torch.zeros(785 * h))
                    start = a
                else:
                    Br = B[h][:r]
                    c0 = D[h][k] @ Br.T
                    start = accuracy(S[h], x_test, y_test, c0 @ Br)
                    a = accuracy(S[h], x_test, y_test, fine_tune(S[h], Br, c0, x_train, y_train) @ Br)
                acc[h, r].append(a)
                print("ctx %d h=%d r=%d  projection %.2f%%  fine-tuned %.2f%%" % (k, h, r, start, a), flush=True)

    values = {}
    for h in SIZES:
        for r in RANKS:
            mean = sum(acc[h, r]) / N_CONTEXTS
            values[h, r] = {"explained": "%.3f" % explained[h][r], "ceiling": "%.2f" % mean}
            print("h=%d r=%d  explained %.3f  ceiling %.2f%%" % (h, r, explained[h][r], mean))
    update_results(values)


if __name__ == "__main__":
    main()
