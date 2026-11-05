"""Offline models, trained on the fit drawers (1-10) only:

mixed         per architecture (A, B): every class, Adam lr 1e-3, batch 128, seed 0, light
              augmentation (below). Saved to weights/mixed_<arch>.pt.
per-alphabet  per architecture and alphabet: the same architecture with its output restricted to
              that alphabet's classes, trained only on that alphabet's fit images, same settings.
              The "knows the context and has all the data" ceiling. Not saved.

Augmentation (offline training only, identical for every model; stream images are never
augmented): each sampled image gets a random affine map, rotation +-10 degrees, scale 0.9-1.1,
shift +-2 pixels, bilinear, on the [0, 1] image before normalizing.

Stopping by training loss, not the test drawers: train in rounds of max(25 epochs, 500 steps);
stop when a round's mean loss is above 95% of the previous round's, or after 12 rounds.

Reports (test drawers, averaged over images, i.e. weighted like the stream): the mixed model,
the mixed model with its logits masked to the true alphabet's classes, and the per-alphabet
ceiling. Writes offline_results.csv.

Usage:
    python offline.py            # mixed + per-alphabet, both architectures (about 15 minutes)
    python offline.py --sweep    # the width sweep used to choose models/mlp.py's WIDTHS
    python offline.py --check    # only score the saved mixed models (+ mask)
"""
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import ALPHABET, MEAN, N_ALPHABETS, STD, alphabet_classes, load_split, mixed_path
from models.mlp import WIDTHS, build_model, layer_params

LR, BATCH, SEED = 1e-3, 128, 0
ROUND_EPOCHS, ROUND_MIN_STEPS, MAX_ROUNDS, PLATEAU = 25, 500, 12, 0.95
HERE = os.path.dirname(os.path.abspath(__file__))


def augment(x):
    """Random affine distortion of normalized images [n, 784] (applied to the [0, 1] image)."""
    n = len(x)
    a = (torch.rand(n) * 2 - 1) * math.radians(10)
    s = 1 + (torch.rand(n) * 2 - 1) * 0.1
    t = (torch.rand(n, 2) * 2 - 1) * (2 / 14)  # 2 pixels in affine_grid's [-1, 1] coordinates
    theta = torch.stack([torch.stack([a.cos() / s, -a.sin() / s, t[:, 0]], 1),
                         torch.stack([a.sin() / s, a.cos() / s, t[:, 1]], 1)], 1)
    raw = (x * STD + MEAN).view(n, 1, 28, 28)
    grid = F.affine_grid(theta, [n, 1, 28, 28], align_corners=False)
    return (F.grid_sample(raw, grid, align_corners=False).view(n, 784) - MEAN) / STD


def train(model, x, y, log="", max_rounds=MAX_ROUNDS, fixed_epochs=None):
    """Adam on augmented batches until the training loss plateaus (see the module docstring)."""
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    per_epoch = math.ceil(len(x) / BATCH)
    round_steps = max(ROUND_EPOCHS * per_epoch, ROUND_MIN_STEPS) if fixed_epochs is None else fixed_epochs * per_epoch
    prev, steps, rounds = None, 0, 0
    while True:
        total, count = 0.0, 0
        while count < round_steps:
            perm = torch.randperm(len(x))
            for i in range(0, len(x), BATCH):
                idx = perm[i:i + BATCH]
                loss = F.cross_entropy(model(augment(x[idx])), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                count += 1
        steps += count
        rounds += 1
        mean = total / count
        if log:
            print("%s round %d (%d steps): mean loss %.4f" % (log, rounds, steps, mean), flush=True)
        if fixed_epochs is not None or (prev is not None and mean > PLATEAU * prev) or rounds >= max_rounds:
            return model, steps, mean
        prev = mean


def accuracy(model, x, y, mask_alphabet=False):
    """Correctness per image; with mask_alphabet, logits restricted to the true alphabet's classes."""
    with torch.no_grad():
        logits = model(x)
        if mask_alphabet:
            own = torch.from_numpy(ALPHABET)[y]
            logits = logits.masked_fill(torch.from_numpy(ALPHABET)[None, :] != own[:, None], float("-inf"))
        return (logits.argmax(1) == y).numpy()


def sweep():
    x, y = load_split("fit")
    xt, yt = load_split("test")
    for widths in [(32, 32), (64, 64), (128, 128), (256, 256)]:
        start = time.time()
        model, _, loss = train(build_model("A", widths), x, y, fixed_epochs=150)
        print("A %s: %d params, 150 epochs, loss %.3f, test %.2f%% (%.0f s)"
              % (widths, sum(p.numel() for p in model.parameters()), loss, accuracy(model, xt, yt).mean() * 100,
                 time.time() - start), flush=True)


def main():
    if "--sweep" in sys.argv:
        return sweep()
    x, y = load_split("fit")
    xt, yt = load_split("test")
    os.makedirs(os.path.dirname(mixed_path("A")), exist_ok=True)
    rows = []
    for arch in WIDTHS:
        counts, total = layer_params(arch)
        print("%s: widths %s, params per layer %s, total %d" % (arch, WIDTHS[arch], counts, total), flush=True)
        model = build_model(arch)
        if "--check" in sys.argv:
            saved = torch.load(mixed_path(arch))
            model.load_state_dict(saved["state_dict"])
            steps, loss = saved["steps"], saved["final_loss"]
        else:
            model, steps, loss = train(model, x, y, log="%s mixed" % arch)
            torch.save({"arch": arch, "widths": WIDTHS[arch], "state_dict": model.state_dict(), "steps": steps,
                        "final_loss": loss}, mixed_path(arch))
        mixed = accuracy(model, xt, yt).mean() * 100
        masked = accuracy(model, xt, yt, mask_alphabet=True).mean() * 100
        print("%s mixed: %.2f%%, masked to the alphabet: %.2f%%" % (arch, mixed, masked), flush=True)
        ceiling = None
        if "--check" not in sys.argv:
            correct, seen = 0, 0
            for a in range(N_ALPHABETS):
                classes = torch.from_numpy(alphabet_classes(a))
                remap = torch.full((len(ALPHABET),), -1, dtype=torch.long)
                remap[classes] = torch.arange(len(classes))
                sel, selt = remap[y] >= 0, remap[yt] >= 0
                m, _, _ = train(build_model(arch, n_out=len(classes)), x[sel], remap[y[sel]])
                correct += accuracy(m, xt[selt], remap[yt[selt]]).sum()
                seen += int(selt.sum())
                print("%s alphabet %d/%d: running ceiling %.2f%%" % (arch, a + 1, N_ALPHABETS, correct / seen * 100),
                      flush=True)
            ceiling = correct / len(yt) * 100
            print("%s per-alphabet ceiling: %.2f%%" % (arch, ceiling), flush=True)
        rows.append({"arch": arch, "widths": "x".join(map(str, WIDTHS[arch])), "params": total,
                     "kb": "%.1f" % (total * 4 / 1024), "steps": steps, "final_loss": "%.4f" % loss,
                     "mixed": "%.2f" % mixed, "mixed_masked": "%.2f" % masked,
                     "per_alphabet": "" if ceiling is None else "%.2f" % ceiling})
    if "--check" not in sys.argv:
        with open(os.path.join(HERE, "offline_results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
