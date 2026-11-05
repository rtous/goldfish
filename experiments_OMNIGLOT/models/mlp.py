"""MLPs for 28x28 Omniglot (784 inputs, 1623 classes), ReLU between layers.

    A: 784 -> w -> w -> 1623          (2 hidden layers)
    B: 784 -> w' -> w' -> w' -> 1623  (3 hidden layers)

Parameter names follow nn.Sequential: 0.*, 2.*, 4.* (and 6.* for B); the last pair is the
output layer.
"""
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 1623
WIDTHS = {"A": (64, 64), "B": (62, 62, 62)}  # chosen by offline.py --sweep (README)


def build_model(arch, widths=None, n_out=N_CLASSES):
    sizes = (784,) + tuple(widths or WIDTHS[arch]) + (n_out,)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def param_names(arch):
    n = len(WIDTHS[arch]) + 1
    return tuple("%d.%s" % (2 * i, p) for i in range(n) for p in ("weight", "bias"))


def forward(W, x, hidden=False):
    """Functional forward with a state dict W (names as build_model). With hidden=True, also
    returns the last hidden layer's activations."""
    n = len(W) // 2
    h = x
    for i in range(n):
        h = F.linear(h, W["%d.weight" % (2 * i)], W["%d.bias" % (2 * i)])
        if i < n - 1:
            h = F.relu(h)
            last = h
    return (h, last) if hidden else h


def layer_params(arch):
    """Parameter count of each layer, and the total."""
    sizes = (784,) + WIDTHS[arch] + (N_CLASSES,)
    counts = [(a + 1) * b for a, b in zip(sizes[:-1], sizes[1:])]
    return counts, sum(counts)
