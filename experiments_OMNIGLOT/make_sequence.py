"""Rebuild sequences/ (seed 0).

alphabets          every test image (drawers 13-20) exactly once: 12984 positions. One burst per
                   alphabet (8 x its number of characters: 112-440 images), alphabets in random
                   order, images shuffled within each burst.
alphabets_random   its shuffled control: the same rows in random order (burst = -1,
                   occurrence = -1).
val_alphabets      the same construction on the val drawers (11-12; 2 x characters = 28-110
                   images per burst), for tuning the label prior and the cache. Never the test drawers.

Columns: position, image_index (row of data.load_split(split)), label, burst, alphabet,
occurrence (how many times that class has already appeared in the current burst: 0 = first).

Usage:
    python make_sequence.py
"""
import os

import numpy as np

from data import ALPHABET, N_ALPHABETS, SPLITS, alphabet_classes, stream_path

FIELDS = ["position", "image_index", "label", "burst", "alphabet", "occurrence"]


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(",".join(FIELDS) + "\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in rows)
    print("Wrote %d rows to %s" % (len(rows), path))


def bursts(split, rng):
    per = len(SPLITS[split])
    rows = []
    for b, a in enumerate(rng.permutation(N_ALPHABETS)):
        idx = np.concatenate([c * per + np.arange(per) for c in alphabet_classes(a)])
        seen = {}
        for i in rng.permutation(idx):
            label = int(i) // per
            rows.append([len(rows), int(i), label, b, int(a), seen.get(label, 0)])
            seen[label] = seen.get(label, 0) + 1
    assert all(ALPHABET[r[2]] == r[4] for r in rows)
    return rows


def main():
    rows = bursts("test", np.random.default_rng(0))
    write(stream_path("alphabets"), rows)
    rng = np.random.default_rng(0)
    write(stream_path("alphabets", True),
          [[p, r[1], r[2], -1, r[4], -1] for p, r in enumerate(rows[i] for i in rng.permutation(len(rows)))])
    write(stream_path("val_alphabets"), bursts("val", np.random.default_rng(1)))


if __name__ == "__main__":
    main()
