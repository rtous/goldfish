"""Online learning on each input sequence, one image at a time, for every configuration in
configurations.json: "bursts" (sequence_<stream>.csv) and "random" (sequence_<stream>_random.csv,
the same rows shuffled). The configuration's "stream" picks the context family. Each position
of a sequence is a MNIST test image under the context of that position (contexts.py).

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
Prediction always uses model.eval() (dropout off, BN running statistics); the training step has
dropout on. There is no augmentation.

The online accuracy goes to the configuration's row in results.csv (<sequence>_accuracy).
A run whose loss becomes NaN or infinite is stopped early and recorded as "diverged".

On the bursts sequence these metrics are also saved (accuracy in %, online):
    bursts_pos1_10, bursts_pos11_50, bursts_pos51plus
                    accuracy by position within the burst: images 1-10, 11-50 and 51 on
                    (how fast the model adapts to a new context)
    bursts_seen_first20, bursts_new_first20
                    accuracy on the first 20 images of the bursts whose context has occurred in
                    an earlier burst, and of the bursts whose context is new (the first burst of
                    each context; few images: 20 per context)
    bursts_retention, bursts_retention_slow
                    retention: the weights as they are at the end of the stream, frozen, scored on
                    all 10000 test images under every context, averaged over the contexts (compare
                    with test_set_accuracy, the mixed model before the stream). _slow: the slow
                    part only, fast parts removed; the same as bursts_retention without fast weights

Configurations with a "sequence" field use the experiment's own sequences (sequences/, built by
make_sequence.py --burst-lengths; they have a first_seen column). For them these bursts metrics
are also saved:
    bursts_bin1_100, bursts_bin101_500, bursts_bin501_1000, bursts_bin1001_2500, bursts_bin2501plus
                    online accuracy by position within the burst (blank when no burst is that long)
    bursts_first_seen, bursts_repeated, and their _frozen counterparts
                    online and frozen accuracy on the images at their first appearance in the
                    stream, and on the images already seen under another context
    bursts_first100_firstburst, bursts_first100_returning
                    online accuracy on the first 100 images of each context's first burst, and on
                    the first 100 images of the bursts whose context has occurred before

The first argument is the experiment (experiment1, experiment2, ...; or just its number).

Usage:
    python3 online_mlp.py experiment1                    # all configurations, both sequences
    python3 online_mlp.py experiment1 rotated_2_sgd 'permuted_8*'  # only these (patterns work)
    python3 online_mlp.py 1 'rotated_2_*' --sequence random  # only the shuffled control sequence
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

from contexts import DEFAULT_CONTEXTS
from mnist_mlp import (BURSTS_METRICS, EXTRA_COLUMNS, SEQUENCES, build_model, config_family, config_seed,
                       context_accuracies, describe, get_device, list_experiments, load_configurations,
                       load_sequence, load_split, load_weights, params_json, predict, read_results,
                       save_result, set_experiment, synchronize)

REPORT_EVERY = 1000
NAN_CHECK_EVERY = 100  # how often (in images) to check the loss for NaN/inf
OPTIMIZERS = ("sgd", "adam")
BN_MODES = ("frozen", "update")
FAST_KEYS = ("learning_rate", "decay", "layers")
PRIOR_MIN = 1e-3  # floor of each label prior value
POSITION_SEGMENTS = (("pos1_10", 1, 10), ("pos11_50", 11, 50), ("pos51plus", 51, None))
POSITION_BINS = (("bin1_100", 1, 100), ("bin101_500", 101, 500), ("bin501_1000", 501, 1000),
                 ("bin1001_2500", 1001, 2500), ("bin2501plus", 2501, None))
RETURN_IMAGES = 100  # images at the start of a burst for first-burst vs returning-context accuracy
FIRST_IMAGES = 20  # images at the start of a burst for the seen / new context metrics


class Stream:
    """A sequence of a context family: its inputs (each test image under its position's context),
    labels, bursts and contexts, plus every test image under every context (for retention)."""

    _caches = {}  # test images under every context, per (family, n_contexts)

    def __init__(self, config, kind, device):
        family = config_family(config)
        key = (family.name, family.n_contexts)
        if key not in Stream._caches:
            x_test, y_test = load_split("test", device)
            Stream._caches[key] = (family.test_cache(x_test), y_test)
        self.cache, self.y_test = Stream._caches[key]
        indices, self.bursts, self.contexts, self.first_seen = load_sequence(
            config["stream"], kind, config.get("sequence"), first_seen=True)
        if self.contexts.max() >= family.n_contexts:
            raise SystemExit("The %s sequence of %s has more contexts than the configuration (%d)."
                             % (kind, config["stream"], family.n_contexts))
        idx = torch.from_numpy(indices).to(device)
        self.x = self.cache[torch.from_numpy(self.contexts).to(device), idx]
        self.y = self.y_test[idx]


def burst_positions(bursts, contexts):
    """1-based position of each image within its burst, and whether its burst's context has
    occurred in an earlier burst."""
    starts = np.r_[True, bursts[1:] != bursts[:-1]]
    start_at = np.maximum.accumulate(np.where(starts, np.arange(len(bursts)), 0))
    seen, returning = set(), np.zeros(len(bursts), dtype=bool)
    for b in np.flatnonzero(starts):
        returning[b] = contexts[b] in seen
        seen.add(contexts[b])
    return np.arange(len(bursts)) - start_at + 1, returning[start_at]


def long_metrics(correct, frozen_correct, bursts, contexts, first_seen):
    """Metrics of the long streams (see the docstring); None for an empty position bin."""
    position, returning = burst_positions(bursts, contexts)
    out = {}
    for name, lo, hi in POSITION_BINS:
        mask = (position >= lo) & (position <= (hi or len(bursts)))
        out[name] = correct[mask].mean() * 100 if mask.any() else None
    for name, mask in (("first_seen", first_seen), ("repeated", ~first_seen)):
        out[name] = correct[mask].mean() * 100
        out[name + "_frozen"] = frozen_correct[mask].mean() * 100
    start = position <= RETURN_IMAGES
    out["first100_firstburst"] = correct[start & ~returning].mean() * 100
    out["first100_returning"] = correct[start & returning].mean() * 100
    return out


def burst_metrics(correct, bursts, contexts):
    """Accuracy (%) by position within the burst, and on the first FIRST_IMAGES images of bursts
    whose context is seen before vs new."""
    starts = np.r_[True, bursts[1:] != bursts[:-1]]
    start_at = np.maximum.accumulate(np.where(starts, np.arange(len(bursts)), 0))
    position = np.arange(len(bursts)) - start_at + 1  # 1-based position within the burst
    seen, new_burst = set(), np.zeros(len(bursts), dtype=bool)
    for b in np.flatnonzero(starts):
        new_burst[b] = contexts[b] not in seen
        seen.add(contexts[b])
    new_burst = new_burst[start_at]  # spread to every position of the burst
    out = {}
    for name, lo, hi in POSITION_SEGMENTS:
        mask = (position >= lo) & (position <= (hi or len(bursts)))
        out[name] = correct[mask].mean() * 100
    first = position <= FIRST_IMAGES
    out["seen_first20"] = correct[first & ~new_burst].mean() * 100
    out["new_first20"] = correct[first & new_burst].mean() * 100
    return out


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


def run(config, device, stream):
    """One online pass over a sequence (a Stream). Returns a dict of metrics ("accuracy" in %, and
    the burst metrics and retention), or None if it diverged."""
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
        print("Label prior: rate=%g" % p["label_prior"])

    x_seq, y_seq, bursts = stream.x, stream.y, stream.bursts
    labels = y_seq.cpu().numpy()
    has_bursts = not (bursts == -1).all()  # the random sequence has burst = -1
    first_in_burst = np.r_[True, bursts[1:] != bursts[:-1]]

    # Frozen model on the same sequence, before any update (one batched forward pass).
    model.eval()
    with torch.no_grad():
        frozen_correct = (predict(model, x_seq) == y_seq).cpu().numpy()
    print("Frozen accuracy (before updates):  %6.2f%%" % (frozen_correct.mean() * 100))

    # Predictions and losses stay on the device during the loop to avoid a CPU sync on every image.
    n = len(y_seq)
    predictions = torch.empty(n, dtype=torch.long, device=device)
    losses = torch.empty(n, device=device)

    start = time.time()
    for t in range(n):
        x = x_seq[t].unsqueeze(0)
        y = y_seq[t].unsqueeze(0)

        # 1) Test: predict before the model has seen this image's label.
        model.eval()
        with torch.no_grad():
            logits = forward(x)
            if prior is not None:
                logits = logits + prior.log()
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

        if (t + 1) % NAN_CHECK_EVERY == 0 or t + 1 == n:
            window = losses[t + 1 - ((t % NAN_CHECK_EVERY) + 1):t + 1]
            bad = ~torch.isfinite(window)
            if bad.any().item():
                at = t + 1 - len(window) + int(bad.nonzero()[0, 0].item())
                print("Loss became NaN/inf at image %d: stopping this run (diverged)." % at)
                return None

        if (t + 1) % REPORT_EVERY == 0:
            correct = (predictions[:t + 1] == y_seq[:t + 1]).float()
            print("Image %5d  accuracy so far: %6.2f%%   last %d: %6.2f%%"
                  % (t + 1, correct.mean().item() * 100, REPORT_EVERY,
                     correct[-REPORT_EVERY:].mean().item() * 100))
    synchronize(device)
    print("Time: %.1f s" % (time.time() - start))

    correct = predictions.cpu().numpy() == labels
    accuracy = correct.mean() * 100
    result = {"accuracy": accuracy, "frozen": frozen_correct.mean() * 100}
    print("\nOnline accuracy (all images):       %6.2f%%" % accuracy)
    print("  online - frozen:                  %+6.2f" % (accuracy - frozen_correct.mean() * 100))
    if has_bursts:
        print("  first image of each burst:        %6.2f%%  (%d images)"
              % (correct[first_in_burst].mean() * 100, first_in_burst.sum()))
        print("  rest of the burst:                %6.2f%%" % (correct[~first_in_burst].mean() * 100))
        online, frozen = burst_metrics(correct, bursts, stream.contexts), burst_metrics(frozen_correct, bursts, stream.contexts)
        for name in online:
            print("  %-32s %6.2f%%  (frozen %6.2f%%)" % (name + ":", online[name], frozen[name]))
        result.update(online)
        if stream.first_seen is not None:
            long = long_metrics(correct, frozen_correct, bursts, stream.contexts, stream.first_seen)
            for name, value in long.items():
                print("  %-32s %s" % (name + ":", "-" if value is None else "%6.2f%%" % value))
            result.update(long)
    # Baseline: always guess the previous image's label.
    print("Baseline 'repeat previous label':  %6.2f%%" % ((labels[1:] == labels[:-1]).mean() * 100))

    # Retention: the final weights, frozen, on the whole test set under every context.
    retention_slow = float(np.mean(context_accuracies(model, stream.cache, stream.y_test)))
    if fast:
        final = build_model(config).to(device)
        final.load_state_dict({k: v + fast[k] if k in fast else v
                               for k, v in model.state_dict().items()})
        retention = float(np.mean(context_accuracies(final, stream.cache, stream.y_test)))
    else:
        retention = retention_slow
    print("Retention (test set, all contexts): %6.2f%%  slow part only: %6.2f%%" % (retention, retention_slow))
    result.update({"retention": retention, "retention_slow": retention_slow,
                   "correct": correct, "frozen_correct": frozen_correct})
    return result


def save_online(config, sequence, result):
    """Store the online accuracy; if online_params changed since the last run, clear the other
    sequence's accuracy too, since it was measured with the old settings."""
    current = params_json(config, "online_params")
    values = {"online_params": current,
              sequence + "_accuracy": "diverged" if result is None else "%.2f" % result["accuracy"]}
    long = [c for c in EXTRA_COLUMNS if c.startswith("bursts_")] if "sequence" in config else []
    if sequence == "bursts":
        for column in BURSTS_METRICS + long:
            value = None if result is None else result[column[len("bursts_"):]]
            values[column] = "" if value is None else "%.2f" % value
    if read_results().get(config["name"], {}).get("online_params") != current:
        for column in [s + "_accuracy" for s in SEQUENCES] + BURSTS_METRICS + long:
            values.setdefault(column, "")
    save_result(config, values)


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

    summary, streams = [], {}
    for sequence in [args.sequence] if args.sequence else list(SEQUENCES):
        print("\n##### Sequence '%s' #####" % sequence)
        for config in configs:
            key = (config["stream"], config.get("n_contexts", DEFAULT_CONTEXTS), config.get("sequence"), sequence)
            if key not in streams:
                streams[key] = Stream(config, sequence, device)
            print("\n=== %s (%s) ===" % (describe(config), sequence))
            result = run(config, device, streams[key])
            save_online(config, sequence, result)
            summary.append((config["name"], sequence, result))

    print("\n=== Summary ===")
    for name, sequence, result in summary:
        if result is None:
            print("%-24s %-8s diverged" % (name, sequence))
        else:
            print("%-24s %-8s %.2f%%" % (name, sequence, result["accuracy"]))


if __name__ == "__main__":
    main()
