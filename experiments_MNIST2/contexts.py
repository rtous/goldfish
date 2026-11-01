"""Rotated contexts: context k rotates the raw 28x28 image (pixels in [0, 1]) by k * 180 / 8
degrees counterclockwise (0, 22.5, ..., 157.5), bilinear, black outside, then normalizes it."""
import numpy as np
import torch
import torch.nn.functional as F

N_CONTEXTS = 8
MEAN, STD = 0.1307, 0.3081  # standard MNIST normalization
CHUNK = 5000  # images per rotation call (bounds the memory of the sampling grid)

_rad = torch.tensor([k * 180.0 / N_CONTEXTS for k in range(N_CONTEXTS)], dtype=torch.float64) * np.pi / 180
_zero = torch.zeros_like(_rad)
# affine_grid maps output coordinates to input coordinates; this turns the digit counterclockwise.
THETA = torch.stack([torch.stack([_rad.cos(), -_rad.sin(), _zero], 1),
                     torch.stack([_rad.sin(), _rad.cos(), _zero], 1)], 1).float()  # [8, 2, 3]


def transform(raw, ctx):
    """Images raw ([N, 784] in [0, 1]) under contexts ctx ([N] long), normalized."""
    out = torch.empty_like(raw)
    for i in range(0, len(raw), CHUNK):
        img = raw[i:i + CHUNK].view(-1, 1, 28, 28)
        grid = F.affine_grid(THETA[ctx[i:i + CHUNK]], list(img.shape), align_corners=False)
        out[i:i + CHUNK] = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros",
                                         align_corners=False).view(-1, 28 * 28)
    return (out - MEAN) / STD


def under_every_context(raw):
    """Every image under every context: [N_CONTEXTS, N, 784]."""
    return torch.stack([transform(raw, torch.full((len(raw),), k, dtype=torch.long)) for k in range(N_CONTEXTS)])
