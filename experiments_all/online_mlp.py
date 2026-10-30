"""Online learning on each input sequence, one image at a time, for every configuration in
configurations.json: "bursts" (sequence.csv) and "random" (sequence_random.csv, the same images
shuffled). By default every sequence file that exists is run. The images come from the MNIST test set.

The network starts from the weights mnist_mlp.py saved in weights/<configuration>.pt
(run that first), not from scratch. Those weights are never modified.

For every image in the sequence the network first predicts its digit (this counts
towards the accuracy), then takes one training step on that single image.

The online settings come from the configuration's online_params:
    learning_rate   required; 0 freezes the weights (no optimizer step)
    optimizer       "sgd" (default): plain SGD, momentum 0, no weight decay
                    "adam": Adam with PyTorch defaults (betas 0.9/0.999, no weight decay)
    bn_mode         BatchNorm during the training step (default "frozen"):
                      frozen  BN layers in eval mode: running statistics fixed (still the offline
                              ones), BN weight and bias are trained like any other parameter
                      update  BN layers in train mode: normalized with the statistics of the
                              single image, and their running statistics are updated
                    It has no effect on models without BatchNorm (the MLPs).
    label_smoothing label smoothing of the cross-entropy loss (default 0)
    fast            two time scales (Hinton & Plaut 1987), e.g.
                      {"learning_rate": 1.0, "decay": 0.97, "layers": ["2.bias"]}
                    Each listed parameter (names from model.named_parameters()) gets a fast part
                    F, starting at 0, and the network uses slow + fast for both prediction and the
                    training step. With g the gradient on the current image (the same for the slow
                    part as for the sum), after each image:
                      slow = optimizer step with learning_rate (skipped if it is 0)
                      F    = decay * F - fast learning_rate * g
                    All three keys are required.
    label_prior     rate r of a running label prior p (10 values, starting at 0.1). At prediction
                    only (not in the loss) log(p) is added to the logits; after predicting,
                    p = (1 - r) * p + r * onehot(true label), clamped to >= 1e-3. It learns no
                    weights: with learning_rate 0 it is a baseline that only tracks the labels.
    label_prior_mix m (default 0), with label_prior: predict with log((1 - m) * p + m / 10), so that
                    digits absent from the recent past are not ruled out
Prediction always uses model.eval() (dropout off, BN running statistics); the training step has
dropout on. There is no augmentation.

The online accuracy goes to the configuration's row in results.csv (<sequence>_accuracy). If the
sequence has intruders, the accuracy on them and on the other (regular) images goes to
<sequence>_intruder_accuracy and <sequence>_regular_accuracy. On the random sequence these are
the same images out of their burst context.
A run whose loss becomes NaN or infinite is stopped early and recorded as "diverged".

The first argument is the experiment (experiment1, experiment2, ...; or just its number).

Usage:
    python3 online_mlp.py experiment2                    # all configurations, all sequences
    python3 online_mlp.py experiment2 mlp_2_sgd 'mlp_8*' # only the named configurations (patterns work)
    python3 online_mlp.py 2 'mlp_2_*' --sequence random  # only the shuffled control sequence
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.func import functional_call
except ImportError:  # PyTorch < 2.0
    from torch.nn.utils.stateless import functional_call

from mnist_mlp import (FILES, SEQUENCES, build_model, config_seed, describe, get_device,
                       list_experiments, set_experiment,
                       load_configurations, load_idx, load_sequence, load_weights, params_json,
                       predict, read_results, save_result, split_accuracy, to_tensors)

REPORT_EVERY = 1000
NAN_CHECK_EVERY = 100  # how often (in images) to check the loss for NaN/inf
OPTIMIZERS = ("sgd", "adam")
BN_MODES = ("frozen", "update")
FAST_KEYS = ("learning_rate", "decay", "layers")
PRIOR_MIN = 1e-3  # floor of each label prior value


def online_params(config):
    """The configuration's online settings, with optimizer and bn_mode filled in."""
    p = config.get("online_params")
    if not p or "learning_rate" not in p:
        raise SystemExit("%s has no online_params.learning_rate in configurations.json." % config["name"])
    p = dict({"optimizer": "sgd", "bn_mode": "frozen", "label_smoothing": 0.0}, **p)
    if p["optimizer"] not in OPTIMIZERS:
        raise SystemExit("Unknown online optimizer '%s'. Available: %s" % (p["optimizer"], ", ".join(OPTIMIZERS)))
    if p["bn_mode"] not in BN_MODES:
        raise SystemExit("Unknown bn_mode '%s'. Available: %s" % (p["bn_mode"], ", ".join(BN_MODES)))
    if "fast" in p:
        fast = p["fast"]
        missing = [k for k in FAST_KEYS if not isinstance(fast, dict) or k not in fast]
        if missing:
            raise SystemExit("%s: online_params.fast needs %s." % (config["name"], ", ".join(missing)))
        names = [n for n, _ in build_model(config).named_parameters()]
        unknown = [n for n in fast["layers"] if n not in names]
        if unknown or not fast["layers"]:
            raise SystemExit("%s: unknown fast layer(s) %s. Available: %s"
                             % (config["name"], ", ".join(unknown) or "(none given)", ", ".join(names)))
    if "label_prior" in p and not 0 < p["label_prior"] <= 1:
        raise SystemExit("%s: online_params.label_prior must be in (0, 1]." % config["name"])
    if "label_prior_mix" in p and ("label_prior" not in p or not 0 <= p["label_prior_mix"] <= 1):
        raise SystemExit("%s: online_params.label_prior_mix must be in [0, 1], with label_prior."
                         % config["name"])
    return p


def make_optimizer(p, params):
    if p["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=p["learning_rate"])
    return torch.optim.SGD(params, lr=p["learning_rate"], momentum=0)


def set_train_mode(model, bn_mode):
    """Training step mode: dropout on; BatchNorm in train mode only if bn_mode is "update"."""
    model.train()
    if bn_mode == "frozen":
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def run(config, device, x_all, y_all, bursts, intruders, seq):
    """One online pass over a sequence. Returns {"accuracy": %, "intruder": "..", "regular": ".."}
    (see split_accuracy), or None if it diverged."""
    p = online_params(config)
    torch.manual_seed(config_seed(config))
    model = build_model(config).to(device)
    load_weights(model, config, device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=p["label_smoothing"])
    optimizer = make_optimizer(p, model.parameters())
    print("Online: %s lr=%g, bn_mode=%s, label_smoothing=%g"
          % (p["optimizer"], p["learning_rate"], p["bn_mode"], p["label_smoothing"]))

    # Fast weights: the network runs with slow + fast for the listed parameters.
    slow = dict(model.named_parameters())
    fast = {}
    if "fast" in p:
        fast = {n: torch.zeros_like(slow[n]) for n in p["fast"]["layers"]}
        print("Fast weights: lr=%g, decay=%g, layers=%s"
              % (p["fast"]["learning_rate"], p["fast"]["decay"], ", ".join(fast)))

    def forward(x):
        if not fast:
            return model(x)
        return functional_call(model, {n: slow[n] + f for n, f in fast.items()}, (x,))

    prior = None
    if "label_prior" in p:
        prior = torch.full((10,), 0.1, device=device)
        mix = p.get("label_prior_mix", 0.0)
        print("Label prior: rate=%g, mix=%g" % (p["label_prior"], mix))

    labels = y_all[seq].cpu().numpy()
    has_bursts = not (bursts == -1).all()  # the random sequence has burst = -1
    first_in_burst = np.r_[True, bursts[1:] != bursts[:-1]]

    # Frozen model on the same sequence, before any update (one batched forward pass).
    model.eval()
    with torch.no_grad():
        frozen_correct = (predict(model, x_all[seq]) == y_all[seq]).cpu().numpy()
    print("Frozen accuracy (before updates):  %6.2f%%" % (frozen_correct.mean() * 100))
    frozen_split = split_accuracy(frozen_correct, intruders)
    if frozen_split["intruder"]:
        print("  intruders: %s%%  (%d images)   regular: %s%%"
              % (frozen_split["intruder"], intruders.sum(), frozen_split["regular"]))

    # Predictions and losses stay on the GPU during the loop to avoid a CPU sync on every image.
    predictions = torch.empty(len(seq), dtype=torch.long, device=device)
    losses = torch.empty(len(seq), device=device)

    start = time.time()
    for t in range(len(seq)):
        x = x_all[seq[t]].unsqueeze(0)
        y = y_all[seq[t]].unsqueeze(0)

        # 1) Test: predict before the model has seen this image's label.
        model.eval()
        with torch.no_grad():
            logits = forward(x)
            if prior is not None:
                logits = logits + ((1 - mix) * prior + mix / 10).log()
            predictions[t] = logits.argmax(dim=1)[0]
            if prior is not None:
                prior.mul_(1 - p["label_prior"])
                prior[y[0]] += p["label_prior"]
                prior.clamp_(min=PRIOR_MIN)

        # 2) Train: one step on this single image.
        set_train_mode(model, p["bn_mode"])
        optimizer.zero_grad()
        loss = loss_fn(forward(x), y)
        loss.backward()
        if fast:
            with torch.no_grad():
                for n, f in fast.items():
                    f.mul_(p["fast"]["decay"]).sub_(slow[n].grad, alpha=p["fast"]["learning_rate"])
        if p["learning_rate"] != 0:
            optimizer.step()
        losses[t] = loss.detach()

        if (t + 1) % NAN_CHECK_EVERY == 0 or t + 1 == len(seq):
            window = losses[t + 1 - ((t % NAN_CHECK_EVERY) + 1):t + 1]
            bad = ~torch.isfinite(window)
            if bad.any().item():
                at = t + 1 - len(window) + int(bad.nonzero()[0, 0].item())
                print("Loss became NaN/inf at image %d: stopping this run (diverged)." % at)
                return None

        if (t + 1) % REPORT_EVERY == 0:
            correct = (predictions[:t + 1] == y_all[seq[:t + 1]]).float()
            print("Image %5d  accuracy so far: %6.2f%%   last %d: %6.2f%%"
                  % (t + 1, correct.mean().item() * 100, REPORT_EVERY,
                     correct[-REPORT_EVERY:].mean().item() * 100))
    torch.cuda.synchronize()
    print("Time: %.1f s" % (time.time() - start))

    correct = predictions.cpu().numpy() == labels
    accuracy = correct.mean() * 100
    print("\nOnline accuracy (all images):       %6.2f%%" % accuracy)
    print("  online - frozen:                  %+6.2f" % (accuracy - frozen_correct.mean() * 100))
    if has_bursts:
        print("  first image of each burst:        %6.2f%%  (%d images)"
              % (correct[first_in_burst].mean() * 100, first_in_burst.sum()))
        print("  rest of the burst:                %6.2f%%" % (correct[~first_in_burst].mean() * 100))
    split = split_accuracy(correct, intruders)
    if split["intruder"]:
        print("  intruders:                        %6s%%  (%d images; frozen %s%%)"
              % (split["intruder"], intruders.sum(), frozen_split["intruder"]))
        print("  regular images:                   %6s%%  (frozen %s%%)"
              % (split["regular"], frozen_split["regular"]))
    # Baseline: always guess the previous image's label.
    print("Baseline 'repeat previous label':  %6.2f%%" % ((labels[1:] == labels[:-1]).mean() * 100))
    return dict(split, accuracy=accuracy)


ONLINE_COLUMNS = ("accuracy", "intruder_accuracy", "regular_accuracy")  # <sequence>_<column>


def save_online(config, sequence, result):
    """Store the online accuracies; if online_params changed since the last run, clear the other
    sequence's accuracies too, since they were measured with the old settings."""
    current = params_json(config, "online_params")
    values = {"online_params": current}
    if result is None:
        values.update({"%s_%s" % (sequence, c): "diverged" for c in ONLINE_COLUMNS})
    else:
        values.update({sequence + "_accuracy": "%.2f" % result["accuracy"],
                       sequence + "_intruder_accuracy": result["intruder"],
                       sequence + "_regular_accuracy": result["regular"]})
    if read_results().get(config["name"], {}).get("online_params") != current:
        for s in SEQUENCES:
            for c in ONLINE_COLUMNS:
                values.setdefault("%s_%s" % (s, c), "")
    save_result(config, values)


def available_sequences(sequence=None):
    if sequence:
        return [sequence]
    sequences = [n for n, path in SEQUENCES.items() if os.path.exists(path)]
    if not sequences:
        raise SystemExit("No sequence files: run make_sequence.py first.")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", help="experiment: %s (or its number)" % ", ".join(list_experiments()))
    parser.add_argument("configurations", nargs="*", help="configuration names (default: all)")
    parser.add_argument("--sequence", choices=list(SEQUENCES), help="run only this sequence")
    args = parser.parse_args()

    set_experiment(args.experiment)
    configs = load_configurations(args.configurations)
    for config in configs:
        online_params(config)  # fail early on a bad configuration
    device = get_device()
    x_all, y_all = to_tensors(load_idx(FILES["test_images"]), load_idx(FILES["test_labels"]), device)

    summary = []
    for sequence in available_sequences(args.sequence):
        indices, bursts, intruders = load_sequence(sequence)
        seq = torch.from_numpy(indices).to(device)
        print("\n##### Sequence '%s', length: %d #####" % (sequence, len(indices)))
        for config in configs:
            print("\n=== %s (%s) ===" % (describe(config), sequence))
            result = run(config, device, x_all, y_all, bursts, intruders, seq)
            save_online(config, sequence, result)
            summary.append((config["name"], sequence, result))

    print("\n=== Summary ===")
    for name, sequence, result in summary:
        if result is None:
            print("%-20s %-8s diverged" % (name, sequence))
        else:
            print("%-20s %-8s %.2f%%" % (name, sequence, result["accuracy"])
                  + ("   intruders %s%%  regular %s%%" % (result["intruder"], result["regular"])
                     if result["intruder"] else ""))


if __name__ == "__main__":
    main()
