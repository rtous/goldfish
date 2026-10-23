"""One-hidden-layer MLP for 28x28 MNIST images (flattened to 784 inputs).

Reconstructed: the parameter counts match the original results exactly
(h=128 -> 101770, h=2 -> 1600), i.e. 784 -> hidden_size -> 10.
"""
import torch.nn as nn


def build_model(hidden_size=128):
    return nn.Sequential(
        nn.Linear(28 * 28, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )
