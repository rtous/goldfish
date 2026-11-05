"""Shared by part1_mlp.py and part2_*.py: architectures with FiLM conditioning, offline training,
the batched few-shot oracle and results.csv merging.

Architectures (Net): an MLP with any hidden widths, or a tiny CNN. Each hidden layer can get
FiLM: a context vector c (r numbers) is mapped by a bias-free linear layer to (gamma - 1, beta)
for each unit (or channel), applied before the ReLU. c = 0 gives gamma = 1, beta = 0 whatever the
FiLM weights are. r = 0 means no FiLM (a plain network).

Few-shot oracle (fewshot): for each context k and seed s, a nested sample of N training images
under context k; start from given weights and fine-tune some of them (or only c) full-batch with
Adam for up to 300 steps, scoring on the whole test set under context k at 10, 30, 100 and 300
steps. The 24 (context, seed) models of one learning rate are trained at once (torch.func.vmap):
the loss is the sum of their mean losses, and Adam is elementwise, so this is the same as 24
separate runs.
"""
import csv
import fcntl
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contexts import N_CONTEXTS, transform  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")
NS = (10, 20, 50, 100, 200, 1000)
LRS = (1e-4, 3e-4, 1e-3, 3e-3)
STEPS = (10, 30, 100, 300)
SEEDS = 3
EPOCHS, BATCH, LR, SEED = 50, 128, 1e-3, 0  # offline.py
JOINT_EPOCHS = 10                            # experiment 5c


class Net(nn.Module):
    """kind "mlp": hidden widths `hidden`; kind "cnn": conv 3x3 channels `hidden` (each followed
    by ReLU and 2x2 max pooling), then a linear head. FiLM (r > 0) on every hidden layer."""

    def __init__(self, kind, hidden, r=0):
        super().__init__()
        self.kind, self.hidden, self.r = kind, tuple(hidden), r
        if kind == "mlp":
            sizes = (784,) + self.hidden
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
            self.head = nn.Linear(sizes[-1], 10)
        else:
            chans = (1,) + self.hidden
            self.layers = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans[:-1], chans[1:]))
            side = 28 // 2 ** len(self.hidden)
            self.head = nn.Linear(chans[-1] * side * side, 10)
        if r:
            self.film = nn.ModuleList(nn.Linear(r, 2 * u, bias=False) for u in self.hidden)
            for f in self.film:
                nn.init.normal_(f.weight, std=r ** -0.5)

    def forward(self, x, c=None):
        """x [n, 784] (normalized); c None, [r] or [n, r]."""
        h = x if self.kind == "mlp" else x.view(-1, 1, 28, 28)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.r and c is not None:
                gb = self.film[i](c)
                u = self.hidden[i]
                gamma, beta = 1 + gb[..., :u], gb[..., u:]
                if self.kind == "cnn":
                    gamma, beta = gamma.reshape(-1, u, 1, 1), beta.reshape(-1, u, 1, 1)
                h = h * gamma + beta
            h = F.relu(h)
            if self.kind == "cnn":
                h = F.max_pool2d(h, 2)
        return self.head(h.flatten(1))


def n_params(model, film=True):
    return sum(p.numel() for n, p in model.named_parameters() if film or not n.startswith("film"))


def accuracy(model, cache, y_test, c=None):
    """Mean test accuracy (%) over the contexts; c None or [N_CONTEXTS, r] (one vector per context)."""
    with torch.no_grad():
        return float(np.mean([(model(cache[k], None if c is None else c[k]).argmax(1) == y_test).float().mean().item() * 100
                              for k in range(N_CONTEXTS)]))


def train_offline(model, x_train, y_train, context=None, epochs=EPOCHS, log="", x_rot=None):
    """offline.py's training: Adam lr 1e-3, batch 128, seed 0. context None = mixed (a random
    context per sampled image), else every image under that context. c is not used. x_rot: the
    training set already rotated under every context ([N_CONTEXTS, 60000, 784], on the model's
    device), to skip rotating each batch (the CNN, on the GPU)."""
    torch.manual_seed(SEED)
    dev = next(model.parameters()).device
    if x_rot is not None:
        x_train = x_rot
        y_train = y_train.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if context is not None and x_rot is None:  # the same images every epoch: rotate them once
        x_train = transform(x_train, torch.full((len(x_train),), context)).to(dev)
    n = len(y_train)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if x_rot is not None:
                idx = idx.to(dev)
                ctx = torch.randint(N_CONTEXTS, (len(idx),)).to(dev) if context is None else context
                loss = F.cross_entropy(model(x_rot[ctx, idx]), y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
                continue
            x = transform(x_train[idx], torch.randint(N_CONTEXTS, (len(idx),))).to(dev) if context is None \
                else x_train[idx.to(dev)]
            loss = F.cross_entropy(model(x), y_train[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == epochs:
            print("%s epoch %d/%d loss %.4f" % (log, epoch, epochs, total / n), flush=True)
    return model


def train_per_context_batched(models, x_rot, y_train, epochs=EPOCHS, log=""):
    """The 8 per-context models of train_offline (model k on context k) trained at once with
    vmap: the same settings, every model sees the same image indices each step."""
    torch.manual_seed(SEED)
    dev = x_rot.device
    y_train = y_train.to(dev)
    params, _ = stack_module_state(models)
    params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
    opt = torch.optim.Adam(params.values(), lr=LR)
    f = vmap(lambda p, x: functional_call(models[0], p, (x,)))
    n = len(y_train)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n).to(dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(f(params, x_rot[:, idx]).reshape(-1, 10), y_train[idx].repeat(N_CONTEXTS)) * N_CONTEXTS
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == epochs:
            print("%s epoch %d/%d loss %.4f" % (log, epoch, epochs, loss.item() / N_CONTEXTS), flush=True)
    return [{k: v[j].detach() for k, v in params.items()} for j in range(N_CONTEXTS)]


def train_joint(model, x_train, y_train, log="", x_rot=None):
    """Experiment 5c's joint pretraining with FiLM: shared weights, FiLM and one c_k per context,
    loss_ctx (c = c_k) + loss_gen (c = 0), 10 epochs, Adam lr 1e-3, batch 128, seed 0, c_k = 0 at
    start. Returns the c_k ([N_CONTEXTS, r])."""
    torch.manual_seed(SEED)
    dev = next(model.parameters()).device
    ck = torch.zeros(N_CONTEXTS, model.r, device=dev, requires_grad=True)
    opt = torch.optim.Adam(list(model.parameters()) + [ck], lr=LR)
    zero = torch.zeros(model.r, device=dev)
    for epoch in range(1, JOINT_EPOCHS + 1):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), BATCH):
            idx = perm[i:i + BATCH]
            ctx = torch.randint(N_CONTEXTS, (len(idx),))
            if x_rot is None:
                x, y = transform(x_train[idx], ctx).to(dev), y_train[idx].to(dev)
            else:
                x, y = x_rot[ctx.to(dev), idx.to(dev)], y_train[idx].to(dev)
            loss = F.cross_entropy(model(x, ck[ctx.to(dev)]), y) + F.cross_entropy(model(x, zero), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("%s joint epoch %d/%d loss %.4f" % (log, epoch, JOINT_EPOCHS, loss.item()), flush=True)
    return ck.detach()


def sample_indices(n_train):
    """Nested training samples: [SEEDS, N_CONTEXTS, max(NS)] indices, one permutation per (s, k)."""
    return np.stack([np.stack([np.random.default_rng(1000 * s + k).permutation(n_train)[:max(NS)]
                               for k in range(N_CONTEXTS)]) for s in range(SEEDS)])


def fewshot(model, trainable, x_train, y_train, cache, y_test, lrs=LRS, ns=NS, one_seed=(), log=""):
    """Grid of the few-shot oracle. trainable: list of parameter names, or "c" (only the context
    vector, starting at 0); for N in one_seed, only seed 0 is run (to save time). cache and y_test
    on the model's device. Returns {(N, lr, steps): mean accuracy over contexts and seeds (%)}."""
    idx = torch.from_numpy(sample_indices(len(x_train))).reshape(SEEDS * N_CONTEXTS, -1)
    params = {n: p.detach() for n, p in model.named_parameters()}
    dev = next(model.parameters()).device
    r = model.r

    def f(p_batched, x, c):
        return functional_call(model, {**params, **p_batched}, (x, c if r else None))

    batched = vmap(f, in_dims=(0, 0, 0))
    out = {}
    for N in ns:
        B = N_CONTEXTS if N in one_seed else SEEDS * N_CONTEXTS
        ctx_of = torch.arange(B) % N_CONTEXTS  # model b = s * 8 + k
        X = torch.stack([transform(x_train[idx[b, :N]], torch.full((N,), int(ctx_of[b]))) for b in range(B)]).to(dev)
        Y = torch.stack([y_train[idx[b, :N]] for b in range(B)]).to(dev)
        for lr in lrs:
            pb = {} if trainable == "c" else {n: params[n].expand(B, *params[n].shape).clone().requires_grad_(True)
                                              for n in trainable}
            c = torch.zeros(B, max(r, 1), device=dev, requires_grad=trainable == "c")
            opt = torch.optim.Adam(list(pb.values()) + ([c] if trainable == "c" else []), lr=lr)
            for step in range(1, max(STEPS) + 1):
                logits = batched(pb, X, c)
                loss = F.cross_entropy(logits.reshape(-1, 10), Y.reshape(-1)) * B
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step in STEPS:
                    out[N, lr, step] = evaluate(f, pb, c, ctx_of, cache, y_test)
            print("%s N=%d lr=%g: %s" % (log, N, lr, " ".join("%.2f" % out[N, lr, s] for s in STEPS)), flush=True)
    return out


def evaluate(f, pb, c, ctx_of, cache, y_test, chunk=2000):
    """Mean accuracy (%) of the batched models, each on the test set under its own context."""
    correct = 0.0
    with torch.no_grad():
        for k in range(N_CONTEXTS):
            sel = (ctx_of == k).nonzero().squeeze(1).to(c.device)
            pk = {n: v[sel] for n, v in pb.items()}
            g = vmap(f, in_dims=(0, None, 0))
            for i in range(0, len(y_test), chunk):
                pred = g(pk, cache[k, i:i + chunk], c[sel]).argmax(-1)
                correct += (pred == y_test[i:i + chunk]).float().sum().item()
    return correct / (len(ctx_of) * len(y_test)) * 100


def best_per_n(grid, lrs=LRS):
    """{N: (acc, lr, steps)} of the best grid cell per N (chosen on the test set: optimistic)."""
    best = {}
    for (N, lr, s), a in grid.items():
        if lr in lrs and (N not in best or a > best[N][0]):
            best[N] = (a, lr, s)
    return best


COLUMNS = ["part", "budget", "arch", "r", "params", "measure", "N", "acc", "gain", "lr", "steps"]


def update_results(rows):
    """Merge rows into results.csv, replacing rows with the same (part, budget, arch, r, measure, N)."""
    path = os.path.join(HERE, "results.csv")
    lock = open(path + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # several part2.py processes may write at once
    key = lambda row: tuple(str(row.get(k, "")) for k in ("part", "budget", "arch", "r", "measure", "N"))  # noqa: E731
    merged = {}
    if os.path.exists(path):
        with open(path) as f:
            merged = {key(row): row for row in csv.DictReader(f)}
    for row in rows:
        merged[key(row)] = {k: str(v) for k, v in row.items()}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(merged.values())
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def save_grid(name, grid):
    with open(os.path.join(HERE, "grids", name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "lr", "steps", "acc"])
        for (N, lr, s), a in sorted(grid.items()):
            w.writerow([N, lr, s, "%.2f" % a])
