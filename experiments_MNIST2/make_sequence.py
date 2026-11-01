"""Build the context-burst streams and their shuffled controls, for each context family.

For each family (rotated, permuted) and each of two seeds, the evaluation seed (--seed) and the
tuning seed (--tune-seed):
  - 10000 positions; every MNIST test image is used exactly once, in a random order, so there
    are no repeated images and the labels are uniform everywhere (no label locality)
  - the positions are split into bursts with lengths drawn uniformly in [min-burst, max-burst]
    (the last burst is cut at the end of the sequence, so it may be shorter)
  - each burst gets a context drawn at random, never the same as the previous burst's; contexts
    recur (with 8 contexts and ~90 bursts, each one returns ~11 times)
The bursts and contexts depend only on the seed and the number of contexts, not on the family,
so with the same options both families get the same images, bursts and context indices: their
streams differ only in what the contexts do (contexts.py).

Files, next to this script:
    sequence_<family>.csv              evaluation stream (--seed)
    sequence_<family>_random.csv       the same rows randomly permuted (burst = -1): the control
                                       without locality (each image keeps its context)
    sequence_<family>_tune.csv         tuning stream (--tune-seed), only to tune online settings
    sequence_<family>_tune_random.csv  its shuffled control
Columns: position, image_index, label, burst, context.

The defaults rebuild the committed files exactly.

With --burst-lengths, it builds instead long streams where every test image appears once under
every context (experiment 2), one set per burst length L, in --out-dir:
  - length 10000 x n_contexts (80000 for 8 contexts); each context has its own random order of
    the 10000 test images and uses them in that order whenever it is active, so within a context
    no image repeats and every image appears exactly once per context
  - every burst has exactly L images (L must divide 10000), so each context gets 10000 / L bursts;
    the contexts of the bursts are in random order, never the same twice in a row
  - an extra column first_seen: 1 if this is the image's first appearance in the stream, under
    any context (recomputed for the shuffled control, in its own order)
The streams only hold context indices, so one file serves both families (the configurations'
"stream" field picks the family). The per-context image orders depend only on the seed, so they
are the same for every L. Files, for each L:
    sequence_L<L>.csv, sequence_L<L>_random.csv              (--seed)
    sequence_L<L>_tune.csv, sequence_L<L>_tune_random.csv    (--tune-seed)

Usage:
    python3 make_sequence.py
    python3 make_sequence.py --n-contexts 16 --families rotated
    python3 make_sequence.py --min-burst 10 --max-burst 100 --seed 3 --tune-seed 4
    python3 make_sequence.py --burst-lengths 100 500 1000 2500 5000 --out-dir experiment2/sequences
"""
import argparse
import os

import numpy as np

from contexts import DEFAULT_CONTEXTS, FAMILIES
from mnist_mlp import FILES, HERE, load_idx, sequence_path

FIELDS = ["position", "image_index", "label", "burst", "context"]
LONG_FIELDS = FIELDS + ["first_seen"]


def write(path, rows, fields=FIELDS):
    with open(path, "w", newline="") as f:
        f.write(",".join(fields) + "\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in rows)
    print("Wrote %d rows to %s" % (len(rows), os.path.basename(path)))


def build(labels, n_contexts, min_burst, max_burst, seed):
    """Rows of the burst stream and of its shuffled control."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    rows, burst, context = [], 0, None
    while len(rows) < len(order):
        size = int(rng.integers(min_burst, max_burst + 1))
        if context is None:
            context = int(rng.integers(n_contexts))
        else:  # any context but the previous one
            context = (context + 1 + int(rng.integers(n_contexts - 1))) % n_contexts
        for _ in range(min(size, len(order) - len(rows))):
            idx = int(order[len(rows)])
            rows.append([len(rows), idx, int(labels[idx]), burst, context])
        burst += 1
    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    random_rows = [[pos, idx, d, -1, c] for pos, (_, idx, d, _, c) in enumerate(shuffled)]
    return rows, random_rows


def context_order(rng, n_contexts, per_context):
    """Random order of n_contexts * per_context bursts, each context per_context times, never the
    same context twice in a row. Each burst picks among the allowed contexts with probability
    proportional to how many bursts they have left; a dead end (only the previous context left)
    restarts the draw."""
    while True:
        left, order = [per_context] * n_contexts, []
        for _ in range(n_contexts * per_context):
            allowed = [c for c in range(n_contexts) if left[c] > 0 and (not order or c != order[-1])]
            if not allowed:
                break
            weights = np.array([left[c] for c in allowed], dtype=float)
            c = allowed[int(rng.choice(len(allowed), p=weights / weights.sum()))]
            left[c] -= 1
            order.append(c)
        else:
            return order


def with_first_seen(rows):
    """Append first_seen (1 at an image's first appearance in this order) to each row."""
    seen = set()
    out = []
    for r in rows:
        out.append(r + [0 if r[1] in seen else 1])
        seen.add(r[1])
    return out


def build_long(labels, n_contexts, burst, seed):
    """Rows of a stream where every image appears once under every context, in bursts of exactly
    burst images, and of its shuffled control."""
    if len(labels) % burst:
        raise SystemExit("The burst length %d does not divide %d." % (burst, len(labels)))
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(labels)) for _ in range(n_contexts)]
    used = [0] * n_contexts
    rows = []
    for b, c in enumerate(context_order(rng, n_contexts, len(labels) // burst)):
        for idx in orders[c][used[c]:used[c] + burst]:
            rows.append([len(rows), int(idx), int(labels[idx]), b, c])
        used[c] += burst
    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    random_rows = [[pos, idx, d, -1, c] for pos, (_, idx, d, _, c) in enumerate(shuffled)]
    return with_first_seen(rows), with_first_seen(random_rows)


def report(rows, n_contexts):
    bursts = rows[-1][3] + 1
    starts = [r for i, r in enumerate(rows) if i == 0 or r[3] != rows[i - 1][3]]
    counts = np.bincount([r[4] for r in starts], minlength=n_contexts)
    print("%d bursts, mean length %.1f; bursts per context: %s"
          % (bursts, len(rows) / bursts, " ".join(str(c) for c in counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--n-contexts", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--min-burst", type=int, default=20)
    parser.add_argument("--max-burst", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tune-seed", type=int, default=1)
    parser.add_argument("--burst-lengths", nargs="+", type=int,
                        help="build the long streams (every image under every context) instead, "
                             "one per fixed burst length")
    parser.add_argument("--out-dir", help="folder of the long streams (relative to this script)")
    args = parser.parse_args()
    if args.n_contexts < 2:
        raise SystemExit("--n-contexts must be at least 2 (consecutive bursts differ).")

    labels = load_idx(FILES["test_labels"])
    if args.burst_lengths:
        if not args.out_dir:
            raise SystemExit("--burst-lengths needs --out-dir (e.g. experiment2/sequences).")
        folder = os.path.join(HERE, args.out_dir)
        os.makedirs(folder, exist_ok=True)
        for length in args.burst_lengths:
            for kind, seed in (("bursts", args.seed), ("tune", args.tune_seed)):
                rows, random_rows = build_long(labels, args.n_contexts, length, seed)
                print("\nL = %d, %s stream (seed %d): %d images;" % (length, kind, seed, len(rows)), end=" ")
                report(rows, args.n_contexts)
                name = "L%d" % length
                write(sequence_path(None, kind, name, folder), rows, LONG_FIELDS)
                write(sequence_path(None, "random" if kind == "bursts" else "tune_random", name, folder),
                      random_rows, LONG_FIELDS)
        return
    for family in args.families:
        for kind, seed in (("bursts", args.seed), ("tune", args.tune_seed)):
            print("\n%s, %s stream (seed %d):" % (family, kind, seed))
            rows, random_rows = build(labels, args.n_contexts, args.min_burst, args.max_burst, seed)
            report(rows, args.n_contexts)
            write(sequence_path(family, kind), rows)
            write(sequence_path(family, "random" if kind == "bursts" else "tune_random"), random_rows)


if __name__ == "__main__":
    main()
