"""Train the shared mixed model of each hidden size (784 -> h -> 10) and save weights/mixed_h<h>.pt.
Mixed: each time a training image is sampled it is shown under a context drawn uniformly at
random. Adam, lr 0.001, batch 128, 50 epochs, seed 0, on the CPU. This is the frozen baseline
that every online run is compared with, and the starting point of sgd and dual.

--check only scores the saved weights on the whole test set, averaged over the 8 contexts.

Usage:
    python offline.py            # train h = 8 and 4 (a few minutes each)
    python offline.py --check    # 74.05 (h = 8), 64.78 (h = 4)
"""
import sys

import torch
import torch.nn.functional as F

from contexts import N_CONTEXTS, transform, under_every_context
from data import load_mixed, load_split, mixed_path
from models.mlp import build_model

SIZES = (8, 4)
EPOCHS, BATCH, LR, SEED = 50, 128, 1e-3, 0


def train(h, x_train, y_train):
    torch.manual_seed(SEED)
    model = build_model(h)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(x_train), BATCH):
            idx = perm[i:i + BATCH]
            x = transform(x_train[idx], torch.randint(N_CONTEXTS, (len(idx),)))
            loss = F.cross_entropy(model(x), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print("h=%d epoch %2d/%d  loss %.4f" % (h, epoch, EPOCHS, total / len(x_train)), flush=True)
    return model


def test_accuracy(model, cache, y_test):
    with torch.no_grad():
        return sum((model(xk).argmax(1) == y_test).float().mean().item() for xk in cache) / len(cache) * 100


def main():
    x_test, y_test = load_split("test")
    cache = under_every_context(x_test)
    if "--check" not in sys.argv:
        x_train, y_train = load_split("train")
        for h in SIZES:
            model = train(h, x_train, y_train)
            torch.save({"hidden_size": h, "state_dict": model.state_dict()}, mixed_path(h))
    for h in SIZES:
        model = build_model(h)
        model.load_state_dict(load_mixed(h))
        print("h=%d: test set, mean over the %d contexts: %.2f%%" % (h, N_CONTEXTS, test_accuracy(model, cache, y_test)))


if __name__ == "__main__":
    main()
