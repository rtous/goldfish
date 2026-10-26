"""Build the burst sequence (sequence.csv) and its shuffled control (sequence_random.csv).

Reconstructed from the original sequence.csv format and statistics:
  - 10000 positions, made of bursts; each burst uses a random subset of 3 distinct digits
  - burst lengths vary (the original had 192 bursts, i.e. ~52 images per burst on average)
  - images are drawn at random, with replacement, from the MNIST test images of those digits

sequence_random.csv is a random permutation of the rows of sequence.csv: the very same
images and labels, with the temporal structure destroyed. Its burst column is -1.

Usage:
    python3 make_sequence.py                       # defaults below
    python3 make_sequence.py --seed 1 --min-burst 10 --max-burst 95
"""
import argparse
import csv
import os

import numpy as np

from mnist_mlp import FILES, HERE, load_idx

FIELDS = ["position", "image_index", "label", "burst"]


def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print("Wrote %d rows to %s" % (len(rows), path))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--digits-per-burst", type=int, default=3)
    parser.add_argument("--min-burst", type=int, default=10)
    parser.add_argument("--max-burst", type=int, default=95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = load_idx(FILES["test_labels"])
    by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}

    rows, burst = [], 0
    while len(rows) < args.length:
        digits = rng.choice(10, size=args.digits_per_burst, replace=False)
        size = int(rng.integers(args.min_burst, args.max_burst + 1))
        for _ in range(min(size, args.length - len(rows))):
            d = int(rng.choice(digits))
            idx = int(rng.choice(by_digit[d]))
            rows.append([len(rows), idx, d, burst])
        burst += 1
    print("%d bursts, mean length %.1f" % (burst, len(rows) / burst))
    write(os.path.join(HERE, "sequence.csv"), rows)

    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    write(os.path.join(HERE, "sequence_random.csv"),
          [[pos, idx, d, -1] for pos, (_, idx, d, _) in enumerate(shuffled)])


if __name__ == "__main__":
    main()
