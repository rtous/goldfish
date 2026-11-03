"""Experiment 5c, step 1: pretraining with a general first layer. Context k's first layer is
S1 + D_k, and S1 alone is trained too: every batch (a random context per sampled image) gives
    loss_ctx  cross-entropy with first layer S1 + D_k (k = each image's context)
    loss_gen  cross-entropy with first layer S1 alone
and we minimize loss_ctx + loss_gen. Both share the head (2.*). S1 and the head start from
weights/mixed_h<h>.pt, every D_k at 0. Adam lr 1e-3, batch 128, 10 epochs, seed 0 (as in 5b).

Saves, for each h:
    experiment5c/weights/general_h<h>.pt   a 784 -> h -> 10 state dict: 0.* = S1, 2.* = the head
    experiment5c/weights/context_h<h>.pt   the D_k (for reference only)
and prints (a) the test accuracy with S1 + D_k for each context's own k (the ceiling), (b) the
accuracy of general_h<h>.pt frozen, both averaged over the 8 contexts, and (c) the mean norm of
D_k over the norm of S1 (first-layer weights).

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment5c/pretrain_general.py
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


def forward(p, x, ctx=None):
    """First layer S1 + D_ctx[i] for image i, or S1 alone if ctx is None; then the shared head."""
    if ctx is None:
        pre = F.linear(x, p["W1"], p["b1"])
    else:
        pre = torch.einsum("bhi,bi->bh", p["W1"] + p["DW"][ctx], x) + p["b1"] + p["Db"][ctx]
    return F.linear(F.relu(pre), p["W2"], p["b2"])


def train(h, x_train, y_train):
    torch.manual_seed(SEED)
    mixed = load_mixed(h)
    p = {"W1": mixed["0.weight"].clone(), "b1": mixed["0.bias"].clone(),
         "DW": torch.zeros(N_CONTEXTS, h, 784), "Db": torch.zeros(N_CONTEXTS, h),
         "W2": mixed["2.weight"].clone(), "b2": mixed["2.bias"].clone()}
    for v in p.values():
        v.requires_grad_(True)
    optimizer = torch.optim.Adam(p.values(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(x_train))
        total_ctx = total_gen = 0.0
        for i in range(0, len(x_train), BATCH):
            idx = perm[i:i + BATCH]
            ctx = torch.randint(N_CONTEXTS, (len(idx),))
            x = transform(x_train[idx], ctx)
            loss_ctx = F.cross_entropy(forward(p, x, ctx), y_train[idx])
            loss_gen = F.cross_entropy(forward(p, x), y_train[idx])
            optimizer.zero_grad()
            (loss_ctx + loss_gen).backward()
            optimizer.step()
            total_ctx += loss_ctx.item() * len(idx)
            total_gen += loss_gen.item() * len(idx)
        print("h=%d epoch %2d/%d  loss_ctx %.4f  loss_gen %.4f"
              % (h, epoch, EPOCHS, total_ctx / len(x_train), total_gen / len(x_train)), flush=True)
    return {k: v.detach() for k, v in p.items()}


def main():
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    cache = under_every_context(x_test)
    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    for h in SIZES:
        p = train(h, x_train, y_train)
        general = {"0.weight": p["W1"], "0.bias": p["b1"], "2.weight": p["W2"], "2.bias": p["b2"]}
        torch.save({"hidden_size": h, "state_dict": general}, os.path.join(HERE, "weights", "general_h%d.pt" % h))
        torch.save({"hidden_size": h, "0.weight": p["DW"], "0.bias": p["Db"]},
                   os.path.join(HERE, "weights", "context_h%d.pt" % h))
        with torch.no_grad():
            own, gen = [], []
            for k in range(N_CONTEXTS):
                ctx = torch.full((len(y_test),), k, dtype=torch.long)
                own.append((forward(p, cache[k], ctx).argmax(1) == y_test).float().mean().item() * 100)
                gen.append((forward(p, cache[k]).argmax(1) == y_test).float().mean().item() * 100)
            ratio = (p["DW"].flatten(1).norm(dim=1).mean() / p["W1"].norm()).item()
        print("h=%d (a) S1 + own D_k: %.2f%%  [%s]" % (h, sum(own) / N_CONTEXTS, " ".join("%.2f" % a for a in own)))
        print("h=%d (b) general_h%d.pt frozen: %.2f%%  [%s]" % (h, h, sum(gen) / N_CONTEXTS, " ".join("%.2f" % a for a in gen)))
        print("h=%d (c) mean |D_k| / |S1| (0.weight): %.3f" % (h, ratio), flush=True)


if __name__ == "__main__":
    main()
