"""Online learning on each input sequence, one image at a time, for every configuration in
configurations.json: "bursts" (sequence.csv) and "random" (sequence_random.csv, the same images
shuffled). By default every sequence file that exists is run. The images come from the MNIST test set.

The network starts from the weights mnist_mlp.py saved in weights/<configuration>.pt
(run that first), not from scratch. Those weights are never modified.

For every image in the sequence the network first predicts its digit (this counts
towards the accuracy), then takes one training step on that single image.
The online optimizer is plain SGD (momentum 0), independent of the offline training (Adam).
The online learning rate is set on the command line; by default the configuration's learning_rate.

Before any update the frozen model is also scored on the same sequence, so each run gives
online - frozen accuracy. Every run appends one row to online_sweep.csv (see show_sweep.py).
Runs with the default online settings also update results.csv and
online_results_<configuration>_<sequence>.csv as before; other settings save their per-image
results under online_runs/ instead, so the default results are not overwritten.

A run whose loss becomes NaN or infinite is stopped early and recorded as diverged.

Usage:
    python3 online_mlp.py                              # all configurations, all sequences
    python3 online_mlp.py mlp_128 other                # only the named configurations
    python3 online_mlp.py --sequence random            # only the shuffled control sequence
    python3 online_mlp.py mlp_2 --online-lr 0.01
"""
import argparse
import csv
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mnist_mlp import (FILES, HERE, SEQUENCES, build_model, describe, get_device, load_configurations,
                       load_idx, load_sequence, load_weights, print_summary, save_result,
                       size_report, to_tensors)

REPORT_EVERY = 1000
NAN_CHECK_EVERY = 100  # how often (in images) to check the loss for NaN/inf
OPTIMIZERS = ("sgd",)
SWEEP_FILE = os.path.join(HERE, "online_sweep.csv")
SWEEP_COLUMNS = ["configuration", "sequence", "optimizer", "online_lr", "frozen_accuracy",
                 "online_accuracy", "first_in_burst_accuracy", "rest_of_burst_accuracy",
                 "status", "diverged_at", "updated"]
RUNS_DIR = os.path.join(HERE, "online_runs")


def default_lr(config):
    return config["training_params"]["learning_rate"]


def is_default(config, optimizer_name, lr):
    return optimizer_name == "sgd" and lr == default_lr(config)


def results_file(config, sequence, optimizer_name, lr):
    if is_default(config, optimizer_name, lr):
        return os.path.join(HERE, "online_results_%s_%s.csv" % (config["name"], sequence))
    os.makedirs(RUNS_DIR, exist_ok=True)
    return os.path.join(RUNS_DIR, "online_results_%s_%s_%s_lr%g.csv"
                        % (config["name"], sequence, optimizer_name, lr))


def save_results(path, indices, labels, preds, bursts):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "image_index", "label", "prediction", "correct", "burst"])
        for pos in range(len(indices)):
            writer.writerow([pos, indices[pos], labels[pos], preds[pos],
                             int(preds[pos] == labels[pos]), bursts[pos]])
    print("Saved per-image results to", path)


def append_sweep_row(values):
    new = not os.path.exists(SWEEP_FILE)
    with open(SWEEP_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS, restval="")
        if new:
            writer.writeheader()
        writer.writerow(values)
    print("Appended run to", SWEEP_FILE)


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0)
    raise SystemExit("Unknown optimizer '%s'. Available: %s" % (name, ", ".join(OPTIMIZERS)))


def pct(mask_correct):
    return "%.2f" % (mask_correct.mean() * 100)


def run(config, device, x_all, y_all, indices, bursts, seq, sequence, optimizer_name="sgd", lr=None):
    """One online pass over a sequence. Returns a dict with the online_sweep.csv values,
    plus the size report under "report" (None if the run diverged)."""
    if lr is None:
        lr = default_lr(config)
    torch.manual_seed(0)
    model = build_model(config).to(device)
    load_weights(model, config, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
    print("Online optimizer: %s, lr=%g" % (optimizer_name, lr))

    labels = y_all[seq].cpu().numpy()
    has_bursts = not (bursts == -1).all()  # the random sequence has burst = -1
    first_in_burst = np.r_[True, bursts[1:] != bursts[:-1]]

    # Frozen model on the same sequence, before any update (one batched forward pass).
    model.eval()
    with torch.no_grad():
        frozen_correct = (model(x_all[seq]).argmax(dim=1) == y_all[seq]).cpu().numpy()
    print("Frozen accuracy (before updates):  %6.2f%%" % (frozen_correct.mean() * 100))

    # Predictions and losses stay on the GPU during the loop to avoid a CPU sync on every image.
    predictions = torch.empty(len(indices), dtype=torch.long, device=device)
    losses = torch.empty(len(indices), device=device)
    diverged_at = None

    start = time.time()
    for t in range(len(indices)):
        x = x_all[seq[t]].unsqueeze(0)
        y = y_all[seq[t]].unsqueeze(0)

        # 1) Test: predict before the model has seen this image's label.
        model.eval()
        with torch.no_grad():
            predictions[t] = model(x).argmax(dim=1)[0]

        # 2) Train: one step on this single image.
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses[t] = loss.detach()

        if (t + 1) % NAN_CHECK_EVERY == 0 or t + 1 == len(indices):
            window = losses[t + 1 - ((t % NAN_CHECK_EVERY) + 1):t + 1]
            bad = ~torch.isfinite(window)
            if bad.any().item():
                diverged_at = t + 1 - len(window) + int(bad.nonzero()[0, 0].item())
                print("Loss became NaN/inf at image %d: stopping this run (diverged)." % diverged_at)
                break

        if (t + 1) % REPORT_EVERY == 0:
            correct = (predictions[:t + 1] == y_all[seq[:t + 1]]).float()
            print("Image %5d  accuracy so far: %6.2f%%   last %d: %6.2f%%"
                  % (t + 1, correct.mean().item() * 100, REPORT_EVERY,
                     correct[-REPORT_EVERY:].mean().item() * 100))
    torch.cuda.synchronize()
    run_time = time.time() - start
    print("Time: %.1f s" % run_time)

    row = {
        "configuration": config["name"],
        "sequence": sequence,
        "optimizer": optimizer_name,
        "online_lr": "%g" % lr,
        "frozen_accuracy": pct(frozen_correct),
        "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if diverged_at is not None:
        row.update({"status": "diverged", "diverged_at": diverged_at})
        append_sweep_row(row)
        row["report"] = None
        return row

    preds = predictions.cpu().numpy()
    correct = preds == labels
    save_results(results_file(config, sequence, optimizer_name, lr), indices, labels, preds, bursts)

    print("\nOnline accuracy (all images):       %6.2f%%" % (correct.mean() * 100))
    print("  online - frozen:                  %+6.2f" % ((correct.mean() - frozen_correct.mean()) * 100))
    if has_bursts:
        print("  first image of each burst:        %6.2f%%  (%d images)"
              % (correct[first_in_burst].mean() * 100, first_in_burst.sum()))
        print("  rest of the burst:                %6.2f%%" % (correct[~first_in_burst].mean() * 100))

    # Baseline: always guess the previous image's label.
    baseline = (labels[1:] == labels[:-1]).mean()
    print("Baseline 'repeat previous label':  %6.2f%%" % (baseline * 100))

    row.update({
        "online_accuracy": pct(correct),
        "first_in_burst_accuracy": pct(correct[first_in_burst]) if has_bursts else "",
        "rest_of_burst_accuracy": pct(correct[~first_in_burst]) if has_bursts else "",
        "status": "ok",
    })
    append_sweep_row(row)

    report = size_report(model, correct.mean() * 100)
    if is_default(config, optimizer_name, lr):
        save_result(config, {
            "params": report["params"],
            "size_bytes": report["size"],
            "online_accuracy": "%.2f" % report["accuracy"],
            "online_first_in_burst_accuracy":
                "%.2f" % (correct[first_in_burst].mean() * 100) if has_bursts else "",
            "online_efficiency": "%.4f" % report["efficiency"],
            "online_time_s": "%.1f" % run_time,
        }, sequence=sequence)
    else:
        print("Non-default online settings: results.csv not updated.")
    row["report"] = report
    return row


def load_test_data(device):
    return to_tensors(load_idx(FILES["test_images"]), load_idx(FILES["test_labels"]), device)


def available_sequences(sequence=None):
    if sequence:
        if sequence not in SEQUENCES:
            raise SystemExit("Unknown sequence '%s'. Available: %s" % (sequence, ", ".join(SEQUENCES)))
        return [sequence]
    sequences = [n for n, path in SEQUENCES.items() if os.path.exists(path)]
    if not sequences:
        raise SystemExit("No sequence files: run make_sequence.py first.")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configurations", nargs="*", help="configuration names (default: all)")
    parser.add_argument("--sequence", choices=list(SEQUENCES), help="run only this sequence")
    parser.add_argument("--online-lr", type=float, default=None,
                        help="online learning rate (default: the configuration's learning_rate)")
    parser.add_argument("--online-optimizer", choices=OPTIMIZERS, default="sgd",
                        help="online optimizer (only sgd: plain SGD, momentum 0)")
    args = parser.parse_args()

    configs = load_configurations(args.configurations)
    device = get_device()
    sequences = available_sequences(args.sequence)
    x_all, y_all = load_test_data(device)

    for sequence in sequences:
        indices, bursts = load_sequence(sequence)
        seq = torch.from_numpy(indices).to(device)
        print("\n##### Sequence '%s', length: %d #####" % (sequence, len(indices)))
        results = []
        for config in configs:
            print("\n=== %s (%s) ===" % (describe(config), sequence))
            row = run(config, device, x_all, y_all, indices, bursts, seq, sequence,
                      args.online_optimizer, args.online_lr)
            if row["report"] is not None:
                results.append((config["name"], row["report"]))
        print("\nSequence '%s':" % sequence)
        print_summary(results)


if __name__ == "__main__":
    main()
