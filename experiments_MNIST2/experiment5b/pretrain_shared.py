"""Experiment 5b, step 1: decoupled pretraining. One shared output layer (2.*) and one first layer
(0.*) per context: an image under context k goes through first layer k, ReLU, then the shared
head. Every first layer and the head start from weights/mixed_h<h>.pt. Training is the same as
offline.py (a random context per sampled image, Adam lr 1e-3, batch 128, seed 0), for 10 epochs.

Saves, for each h:
    experiment5b/weights/shared_h<h>.pt       a 784 -> h -> 10 state dict: 0.* = the mean of the
                                              8 first layers, 2.* = the shared head
    experiment5b/weights/per_context_h<h>.pt  the 8 first layers (for reference only)
and prints (a) the test accuracy with each context's own first layer (the ceiling) and (b) the
accuracy of shared_h<h>.pt frozen, both averaged over the 8 contexts.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment5b/pretrain_shared.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contexts import N_CONTEXTS, transform, under_every_context  # noqa: E402
from data import load_mixed, load_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (8, 4)
EPOCHS, BATCH, LR, SEED = 10, 128, 1e-3, 0


def forward(p, x, ctx):
    """p["W1"] [8, h, 784] and p["b1"] [8, h]: first layer ctx[i] for image i."""
    pre = torch.einsum("bhi,bi->bh", p["W1"][ctx], x) + p["b1"][ctx]
    return F.linear(F.relu(pre), p["W2"], p["b2"])


def train(h, x_train, y_train):
    torch.manual_seed(SEED)
    mixed = load_mixed(h)
    p = {"W1": mixed["0.weight"].repeat(N_CONTEXTS, 1, 1), "b1": mixed["0.bias"].repeat(N_CONTEXTS, 1),
         "W2": mixed["2.weight"].clone(), "b2": mixed["2.bias"].clone()}
    for v in p.values():
        v.requires_grad_(True)
    optimizer = torch.optim.Adam(p.values(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(x_train), BATCH):
            idx = perm[i:i + BATCH]
            ctx = torch.randint(N_CONTEXTS, (len(idx),))
            loss = F.cross_entropy(forward(p, transform(x_train[idx], ctx), ctx), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print("h=%d epoch %2d/%d  loss %.4f" % (h, epoch, EPOCHS, total / len(x_train)), flush=True)
    return {k: v.detach() for k, v in p.items()}


def main():
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    cache = under_every_context(x_test)
    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    for h in SIZES:
        p = train(h, x_train, y_train)
        shared = {"0.weight": p["W1"].mean(0), "0.bias": p["b1"].mean(0), "2.weight": p["W2"], "2.bias": p["b2"]}
        torch.save({"hidden_size": h, "state_dict": shared}, os.path.join(HERE, "weights", "shared_h%d.pt" % h))
        torch.save({"hidden_size": h, "0.weight": p["W1"], "0.bias": p["b1"]},
                   os.path.join(HERE, "weights", "per_context_h%d.pt" % h))
        with torch.no_grad():
            own, mean = [], []
            for k in range(N_CONTEXTS):
                ctx = torch.full((len(y_test),), k, dtype=torch.long)
                own.append((forward(p, cache[k], ctx).argmax(1) == y_test).float().mean().item() * 100)
                logits = F.linear(F.relu(F.linear(cache[k], shared["0.weight"], shared["0.bias"])),
                                  shared["2.weight"], shared["2.bias"])
                mean.append((logits.argmax(1) == y_test).float().mean().item() * 100)
        print("h=%d (a) own first layer: %.2f%%  [%s]" % (h, sum(own) / N_CONTEXTS, " ".join("%.2f" % a for a in own)))
        print("h=%d (b) shared_h%d.pt frozen: %.2f%%  [%s]" % (h, h, sum(mean) / N_CONTEXTS, " ".join("%.2f" % a for a in mean)),
              flush=True)


if __name__ == "__main__":
    main()
