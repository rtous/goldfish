"""Online learning on each input sequence, one image at a time, for every configuration in
configurations.json: "bursts" (sequence.csv) and "random" (sequence_random.csv, the same images
shuffled). By default every sequence file that exists is run. The images come from the MNIST test set.

The network starts from the weights mnist_mlp.py saved in weights/<configuration>.pt
(run that first), not from scratch.

For every image in the sequence the network first predicts its digit (this counts
towards the accuracy), then takes one training step on that single image.
Only the learning_rate training parameter is used here (no epochs or batches).

Usage:
    python3 online_mlp.py                              # all configurations, all sequences
    python3 online_mlp.py mlp_128 other                # only the named configurations
    python3 online_mlp.py --sequence random            # only the shuffled control sequence
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from mnist_mlp import (FILES, HERE, SEQUENCES, build_model, describe, get_device, load_configurations,
                       load_idx, load_sequence, load_weights, parse_args, print_summary, save_result,
                       size_report, to_tensors)

REPORT_EVERY = 1000


def results_file(name, sequence):
    return os.path.join(HERE, "online_results_%s_%s.csv" % (name, sequence))


def save_results(path, indices, labels, preds, bursts):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "image_index", "label", "prediction", "correct", "burst"])
        for pos in range(len(indices)):
            writer.writerow([pos, indices[pos], labels[pos], preds[pos],
                             int(preds[pos] == labels[pos]), bursts[pos]])
    print("Saved per-image results to", path)


def run(config, device, x_all, y_all, indices, bursts, seq, sequence):
    torch.manual_seed(0)
    model = build_model(config).to(device)
    load_weights(model, config, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training_params"]["learning_rate"])

    # Predictions stay on the GPU during the loop to avoid a CPU sync on every image.
    predictions = torch.empty(len(indices), dtype=torch.long, device=device)

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
        loss_fn(model(x), y).backward()
        optimizer.step()

        if (t + 1) % REPORT_EVERY == 0:
            correct = (predictions[:t + 1] == y_all[seq[:t + 1]]).float()
            print("Image %5d  accuracy so far: %6.2f%%   last %d: %6.2f%%"
                  % (t + 1, correct.mean().item() * 100, REPORT_EVERY,
                     correct[-REPORT_EVERY:].mean().item() * 100))
    torch.cuda.synchronize()
    run_time = time.time() - start
    print("Time: %.1f s" % run_time)

    labels = y_all[seq].cpu().numpy()
    preds = predictions.cpu().numpy()
    correct = preds == labels
    save_results(results_file(config["name"], sequence), indices, labels, preds, bursts)
    has_bursts = not (bursts == -1).all()  # the random sequence has burst = -1
    first_in_burst = np.r_[True, bursts[1:] != bursts[:-1]]

    print("\nOnline accuracy (all images):       %6.2f%%" % (correct.mean() * 100))
    if has_bursts:
        print("  first image of each burst:        %6.2f%%  (%d images)"
              % (correct[first_in_burst].mean() * 100, first_in_burst.sum()))
        print("  rest of the burst:                %6.2f%%" % (correct[~first_in_burst].mean() * 100))

    # Baseline: always guess the previous image's label.
    baseline = (labels[1:] == labels[:-1]).mean()
    print("Baseline 'repeat previous label':  %6.2f%%" % (baseline * 100))

    report = size_report(model, correct.mean() * 100)
    save_result(config, {
        "params": report["params"],
        "size_bytes": report["size"],
        "online_accuracy": "%.2f" % report["accuracy"],
        "online_first_in_burst_accuracy":
            "%.2f" % (correct[first_in_burst].mean() * 100) if has_bursts else "",
        "online_efficiency": "%.4f" % report["efficiency"],
        "online_time_s": "%.1f" % run_time,
    }, sequence=sequence)
    return report


def main():
    names, sequence = parse_args(sys.argv[1:])
    configs = load_configurations(names)
    device = get_device()

    if sequence:
        sequences = [sequence]
    else:
        sequences = [n for n, path in SEQUENCES.items() if os.path.exists(path)]
        if not sequences:
            raise SystemExit("No sequence files: run make_sequence.py first.")
    x_all, y_all = to_tensors(load_idx(FILES["test_images"]), load_idx(FILES["test_labels"]), device)

    for sequence in sequences:
        indices, bursts = load_sequence(sequence)
        seq = torch.from_numpy(indices).to(device)
        print("\n##### Sequence '%s', length: %d #####" % (sequence, len(indices)))
        results = []
        for config in configs:
            print("\n=== %s (%s) ===" % (describe(config), sequence))
            results.append((config["name"],
                            run(config, device, x_all, y_all, indices, bursts, seq, sequence)))
        print("\nSequence '%s':" % sequence)
        print_summary(results)


if __name__ == "__main__":
    main()
