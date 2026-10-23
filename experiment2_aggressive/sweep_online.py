"""Sweep the online optimizer and learning rate: for every configuration (or those named),
each optimizer and each learning rate, run online_mlp.py's run() on both sequences.

Each run appends one row to online_sweep.csv; read the results with show_sweep.py.
Runs that diverge (NaN/inf loss) are stopped early and recorded as diverged.
The data is loaded once and the runs happen in this process.

Usage:
    python3 sweep_online.py                                  # all configurations, default lrs
    python3 sweep_online.py mlp_2 mlp_4                      # only the named configurations
    python3 sweep_online.py mlp_2 --adam-lrs 0.001,0.01 --sgd-lrs 0.1
    python3 sweep_online.py --sgd-lrs ""                     # Adam only
"""
import argparse
import time

import torch

from mnist_mlp import describe, get_device, load_configurations, load_sequence
from online_mlp import available_sequences, load_test_data, run

ADAM_LRS = [0.001, 0.003, 0.01, 0.03]
SGD_LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def lr_list(text):
    return [float(v) for v in text.replace(" ", ",").split(",") if v]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configurations", nargs="*", help="configuration names (default: all)")
    parser.add_argument("--adam-lrs", type=lr_list, default=ADAM_LRS,
                        help="comma-separated Adam learning rates (default: %s)" % ADAM_LRS)
    parser.add_argument("--sgd-lrs", type=lr_list, default=SGD_LRS,
                        help="comma-separated SGD learning rates (default: %s)" % SGD_LRS)
    args = parser.parse_args()

    configs = load_configurations(args.configurations)
    device = get_device()
    x_all, y_all = load_test_data(device)
    sequences = {}
    for name in available_sequences():
        indices, bursts = load_sequence(name)
        sequences[name] = (indices, bursts, torch.from_numpy(indices).to(device))

    grid = [("adam", lr) for lr in args.adam_lrs] + [("sgd", lr) for lr in args.sgd_lrs]
    total = len(configs) * len(grid) * len(sequences)
    print("Sweep: %d configurations x %d optimizer/lr x %d sequences = %d runs"
          % (len(configs), len(grid), len(sequences), total))

    n, diverged, start = 0, [], time.time()
    for config in configs:
        for optimizer_name, lr in grid:
            for name, (indices, bursts, seq) in sequences.items():
                n += 1
                print("\n=== [%d/%d] %s (%s), online %s lr=%g ===" % (
                    n, total, describe(config), name, optimizer_name, lr))
                row = run(config, device, x_all, y_all, indices, bursts, seq, name, optimizer_name, lr)
                if row["status"] == "diverged":
                    diverged.append("%s %s %s lr=%g" % (config["name"], name, optimizer_name, lr))

    print("\nSweep done: %d runs in %.1f min." % (total, (time.time() - start) / 60))
    if diverged:
        print("Diverged (%d): %s" % (len(diverged), "; ".join(diverged)))
    print("Show the results with: python3 show_sweep.py")


if __name__ == "__main__":
    main()
