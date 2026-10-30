"""Build the burst sequence (sequence.csv) and its shuffled control (sequence_random.csv).

sequence.csv starts from the v1 sequence (sequence_v1.csv), so the bursts are identical, and
replaces a fraction of its positions (--intrusion-rate, default 0.1) with intruders:
  - the positions are picked at random, without replacement
  - each intruder is a digit that does not appear in that position's burst (in v1), with a random
    MNIST test image of that digit
  - the "intruder" column is 1 for intruders, 0 otherwise; the "burst" column is kept as is

The v1 sequence (sequence_v1.csv, the first version of the sequence, without intruders), --v1 to
rebuild it. Reconstructed
from the original sequence.csv format and statistics:
  - 10000 positions, made of bursts; each burst uses a random subset of 3 distinct digits
  - burst lengths vary (the original had 192 bursts, i.e. ~52 images per burst on average)
  - images are drawn at random, with replacement, from the MNIST test images of those digits

sequence_random.csv (sequence_random_v1.csv) is a random permutation of the rows of sequence.csv
(sequence_v1.csv): the very same images and labels, with the temporal structure destroyed. Its
burst column is -1.

Usage:
    python3 make_sequence.py                          # sequence.csv from sequence_v1.csv, 10% intruders
    python3 make_sequence.py --intrusion-rate 0.2 --seed 1
    python3 make_sequence.py --v1                     # rebuild sequence_v1.csv (defaults below)
    python3 make_sequence.py --v1 --seed 1 --min-burst 10 --max-burst 95
"""
import argparse
import csv
import os

import numpy as np

from mnist_mlp import FILES, HERE, load_idx

FIELDS_V1 = ["position", "image_index", "label", "burst"]
FIELDS = FIELDS_V1 + ["intruder"]
V1 = os.path.join(HERE, "sequence_v1.csv")
RANDOM_V1 = os.path.join(HERE, "sequence_random_v1.csv")


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    print("Wrote %d rows to %s" % (len(rows), path))


def make_v1(args, by_digit):
    rng = np.random.default_rng(args.seed)
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
    write(V1, FIELDS_V1, rows)

    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    write(RANDOM_V1, FIELDS_V1, [[pos, idx, d, -1] for pos, (_, idx, d, _) in enumerate(shuffled)])


def make_intruders(args, by_digit):
    if not os.path.exists(V1):
        raise SystemExit("No %s: run make_sequence.py --v1 first." % os.path.basename(V1))
    with open(V1) as f:
        rows = [[int(r[k]) for k in FIELDS_V1] + [0] for r in csv.DictReader(f)]
    rng = np.random.default_rng(args.seed)
    burst_digits = {}
    for _, _, d, b, _ in rows:
        burst_digits.setdefault(b, set()).add(d)

    n = int(round(args.intrusion_rate * len(rows)))
    for pos in rng.choice(len(rows), size=n, replace=False):
        b = rows[pos][3]
        d = int(rng.choice(sorted(set(range(10)) - burst_digits[b])))
        rows[pos] = [int(pos), int(rng.choice(by_digit[d])), d, b, 1]

    intruders = [r for r in rows if r[4]]
    if any(r[2] in burst_digits[r[3]] for r in intruders):
        raise SystemExit("Bug: an intruder's digit is in its burst.")
    print("%d intruders out of %d positions (%.1f%%); none has a digit of its burst"
          % (len(intruders), len(rows), 100 * len(intruders) / len(rows)))
    write(os.path.join(HERE, "sequence.csv"), FIELDS, rows)

    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    write(os.path.join(HERE, "sequence_random.csv"), FIELDS,
          [[pos, idx, d, -1, intr] for pos, (_, idx, d, _, intr) in enumerate(shuffled)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intrusion-rate", type=float, default=0.1,
                        help="fraction of the v1 positions replaced by intruders (default 0.1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v1", action="store_true", help="rebuild the v1 sequence instead")
    parser.add_argument("--length", type=int, default=10000, help="--v1 only")
    parser.add_argument("--digits-per-burst", type=int, default=3, help="--v1 only")
    parser.add_argument("--min-burst", type=int, default=10, help="--v1 only")
    parser.add_argument("--max-burst", type=int, default=95, help="--v1 only")
    args = parser.parse_args()
    if not 0 <= args.intrusion_rate < 1:
        raise SystemExit("--intrusion-rate must be in [0, 1).")

    labels = load_idx(FILES["test_labels"])
    by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}
    if args.v1:
        make_v1(args, by_digit)
    else:
        make_intruders(args, by_digit)


if __name__ == "__main__":
    main()
