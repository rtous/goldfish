"""Context families: fixed, deterministic transformations of MNIST images.

A context family has n_contexts contexts (default 8). Each context is one fixed transformation,
deterministic given the family, the context index k and the seed:

    rotated   context k rotates the image by k * 180 / n_contexts degrees counterclockwise
              (0, 22.5, ..., 157.5 for 8), bilinear, with black (0) outside the image. The raw
              28x28 image (pixels in [0, 1]) is rotated before normalization. The seed is unused.
    permuted  context k reorders the 784 pixels with a fixed random permutation drawn from
              numpy's default_rng(seed + k). Context 0 is a random permutation too, not the
              identity, so no context is privileged.

Family(name).transform(raw, ctx) returns the transformed, normalized images for a batch of raw
images (float, [N, 784], pixels in [0, 1]) and their context indices ([N] long). test_cache()
precomputes every test image under every context ([n_contexts, 10000, 784]).

Run as a script, it saves a grid of a few test images under every context of each family to
the experiment's plots/ folder (contexts_<family>.png), to check the transformations by eye.

Usage:
    python3 contexts.py experiment1          # experiment1/plots/contexts_{rotated,permuted}.png
    python3 contexts.py 1 --n-contexts 16
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

FAMILIES = ("rotated", "permuted")
DEFAULT_CONTEXTS = 8
MEAN, STD = 0.1307, 0.3081  # standard MNIST normalization
CHUNK = 5000  # images per rotation call (bounds the memory of the sampling grid)


def normalize(x):
    return (x - MEAN) / STD


class Family:
    """One context family: name, number of contexts and seed fix every transformation."""

    def __init__(self, name, n_contexts=DEFAULT_CONTEXTS, seed=0):
        if name not in FAMILIES:
            raise SystemExit("Unknown context family '%s'. Available: %s" % (name, ", ".join(FAMILIES)))
        self.name, self.n_contexts, self.seed = name, n_contexts, seed
        if name == "rotated":
            self.angles = [k * 180.0 / n_contexts for k in range(n_contexts)]
            rad = torch.tensor(self.angles, dtype=torch.float64) * np.pi / 180
            cos, sin = rad.cos(), rad.sin()
            zero = torch.zeros_like(rad)
            # affine_grid maps output coordinates (x right, y down) to input coordinates; this
            # matrix shows the digit turned counterclockwise by the angle on screen.
            self.theta = torch.stack([torch.stack([cos, -sin, zero], 1),
                                      torch.stack([sin, cos, zero], 1)], 1).float()  # [n, 2, 3]
        else:
            self.perms = torch.from_numpy(np.stack(
                [np.random.default_rng(seed + k).permutation(28 * 28) for k in range(n_contexts)]))

    def describe(self, k):
        if self.name == "rotated":
            return "%g deg" % self.angles[k]
        return "perm %d" % k

    def transform(self, raw, ctx):
        """Images raw ([N, 784] float in [0, 1]) under contexts ctx ([N] long), normalized."""
        if self.name == "permuted":
            return normalize(raw.gather(1, self.perms.to(raw.device)[ctx]))
        out = torch.empty_like(raw)
        theta = self.theta.to(raw.device)
        for i in range(0, len(raw), CHUNK):
            img = raw[i:i + CHUNK].view(-1, 1, 28, 28)
            grid = F.affine_grid(theta[ctx[i:i + CHUNK]], list(img.shape), align_corners=False)
            rotated = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            out[i:i + CHUNK] = rotated.view(-1, 28 * 28)
        return normalize(out)

    def under(self, raw, k):
        """All images raw under the single context k."""
        return self.transform(raw, torch.full((len(raw),), k, dtype=torch.long, device=raw.device))

    def test_cache(self, raw_test):
        """Every test image under every context: [n_contexts, N, 784]."""
        return torch.stack([self.under(raw_test, k) for k in range(self.n_contexts)])


def raw_tensor(images, device="cpu"):
    """uint8 images [N, 784] -> float tensor in [0, 1]."""
    return torch.from_numpy(images.astype(np.float32) / 255.0).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", help="experiment folder or number (plots go to its plots/)")
    parser.add_argument("--n-contexts", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images", type=int, default=6, help="test images shown per context")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mnist_mlp import FILES, experiment_dir, load_idx

    plots_dir = os.path.join(experiment_dir(args.experiment), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    raw = raw_tensor(load_idx(FILES["test_images"])[:args.images])
    labels = load_idx(FILES["test_labels"])[:args.images]
    for name in FAMILIES:
        family = Family(name, args.n_contexts, args.seed)
        fig, axes = plt.subplots(args.images, family.n_contexts,
                                 figsize=(1.1 * family.n_contexts, 1.2 * args.images), squeeze=False)
        for k in range(family.n_contexts):
            x = family.under(raw, k).view(-1, 28, 28) * STD + MEAN
            for i in range(args.images):
                ax = axes[i, k]
                ax.imshow(x[i].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(family.describe(k), fontsize=8)
                if k == 0:
                    ax.set_ylabel(str(labels[i]), rotation=0, labelpad=10)
        fig.suptitle("%s: %d test images (rows, label at left) under each context (columns)"
                     % (name, args.images), fontsize=9)
        fig.tight_layout()
        path = os.path.join(plots_dir, "contexts_%s.png" % name)
        fig.savefig(path, dpi=120)
        print("Saved", path)


if __name__ == "__main__":
    main()
