"""Rebuild sequences/ (seed 0, 8 contexts): two streams of MNIST test images, each with its
shuffled control (_random: the same rows in random order, burst = -1, each image keeps its context).

short   10000 positions, every test image once in random order; bursts of 20-200 images (uniform,
        the last one cut at the end), each with a random context different from the previous one.
L1000   80000 positions, every test image once under every context; each context has its own
        random image order; bursts of exactly 1000 images, contexts in random order, never the same
        twice in a row. Extra column first_seen: 1 at an image's first appearance in the file.

Usage:
    python make_sequence.py
"""
import os

import numpy as np

from contexts import N_CONTEXTS
from data import load_idx, stream_path

FIELDS = ["position", "image_index", "label", "burst", "context"]


def write(path, rows, fields=FIELDS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(",".join(fields) + "\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in rows)
    print("Wrote %d rows to %s" % (len(rows), path))


def shuffle(rng, rows):
    return [[pos, idx, d, -1, c] for pos, (_, idx, d, _, c) in enumerate(rows[i] for i in rng.permutation(len(rows)))]


def short(labels, rng):
    order = rng.permutation(len(labels))
    rows, burst, context = [], 0, None
    while len(rows) < len(order):
        size = int(rng.integers(20, 201))
        context = int(rng.integers(N_CONTEXTS)) if context is None else \
            (context + 1 + int(rng.integers(N_CONTEXTS - 1))) % N_CONTEXTS  # any but the previous
        for _ in range(min(size, len(order) - len(rows))):
            idx = int(order[len(rows)])
            rows.append([len(rows), idx, int(labels[idx]), burst, context])
        burst += 1
    return rows, shuffle(rng, rows)


def context_order(rng, per_context):
    """Each context per_context times, never twice in a row; a draw proportional to the bursts
    left, restarted at a dead end."""
    while True:
        left, order = [per_context] * N_CONTEXTS, []
        for _ in range(N_CONTEXTS * per_context):
            allowed = [c for c in range(N_CONTEXTS) if left[c] > 0 and (not order or c != order[-1])]
            if not allowed:
                break
            weights = np.array([left[c] for c in allowed], dtype=float)
            c = allowed[int(rng.choice(len(allowed), p=weights / weights.sum()))]
            left[c] -= 1
            order.append(c)
        else:
            return order


def with_first_seen(rows):
    seen, out = set(), []
    for r in rows:
        out.append(r + [0 if r[1] in seen else 1])
        seen.add(r[1])
    return out


def fixed_bursts(labels, rng, burst):
    orders = [rng.permutation(len(labels)) for _ in range(N_CONTEXTS)]
    used, rows = [0] * N_CONTEXTS, []
    for b, c in enumerate(context_order(rng, len(labels) // burst)):
        for idx in orders[c][used[c]:used[c] + burst]:
            rows.append([len(rows), int(idx), int(labels[idx]), b, c])
        used[c] += burst
    return with_first_seen(rows), with_first_seen(shuffle(rng, rows))


def main():
    labels = load_idx("t10k-labels")
    rows, random_rows = short(labels, np.random.default_rng(0))
    write(stream_path("short"), rows)
    write(stream_path("short", True), random_rows)
    rows, random_rows = fixed_bursts(labels, np.random.default_rng(0), 1000)
    write(stream_path("L1000"), rows, FIELDS + ["first_seen"])
    write(stream_path("L1000", True), random_rows, FIELDS + ["first_seen"])


if __name__ == "__main__":
    main()
