"""Online learning for every configuration x size x stream of an experiment, on the bursts stream
and its shuffled control. For each image the network first predicts its label (this counts
towards the accuracy), then takes one update on that image alone.

One update rule covers every method. The network predicts with S + F, F starts at 0, and with g
the gradient of the image's cross-entropy loss at S + F:
    S <- S - lr * g                    (skipped if lr = 0)
    F <- decay * F - fast_lr * g       (only on the fast parameters)
    sgd      {"lr"}: no fast part
    dual     {"lr", "fast_lr", "decay", "fast_layers": "hidden" (0.weight, 0.bias) or "all"}
    metasgd  lr = 0, fast part on every parameter; S, fast_lr (alpha) and decay (lambda) are
             per-parameter tensors from <experiment>/weights/<config>_h<h>.pt (meta_train.py)
sgd and dual start from the shared mixed model (weights/mixed_h<h>.pt, offline.py). A NaN or inf
loss stops the run (diverged).

Writes one row per config x h x stream to <experiment>/results.csv (rows not run are kept) and
the per-position correctness on the bursts stream to curves/<config>_h<h>_<stream>.npz.

Usage:
    python online.py 1                  # every configuration of experiment1
    python online.py 2 dual_f1e-3_d0.99 # only this one
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from contexts import under_every_context
from data import experiment_dir, load_config, load_mixed, load_split, load_stream, position_in_burst

NAMES = ("0.weight", "0.bias", "2.weight", "2.bias")
FAST_LAYERS = {"hidden": ("0.weight", "0.bias"), "all": NAMES}
BINS = (("pos1_10", 1, 10), ("pos11_50", 11, 50), ("pos51_200", 51, 200), ("pos201_1000", 201, 1000))
COLUMNS = (["config", "method", "h", "stream", "frozen", "bursts", "shuffled", "gain", "locality"]
           + [b[0] for b in BINS] + ["diverged"])


def forward(W, x):
    return F.linear(F.relu(F.linear(x, W["0.weight"], W["0.bias"])), W["2.weight"], W["2.bias"])


def setup(cfg, h, exp_dir):
    """Starting weights S, slow learning rate lr, and the fast part {name: (fast_lr, decay)}."""
    if cfg["method"] == "metasgd":
        path = os.path.join(exp_dir, "weights", "%s_h%d.pt" % (cfg["name"], h))
        if not os.path.exists(path):
            raise SystemExit("No %s: run meta_train.py %s first." % (path, os.path.basename(exp_dir)))
        ck = torch.load(path)
        return ck["S"], 0.0, {n: (ck["alpha"][n], ck["lambda"][n]) for n in NAMES}
    S = load_mixed(h)
    if cfg["method"] == "sgd":
        return S, cfg["lr"], {}
    if cfg["method"] == "dual":
        return S, cfg["lr"], {n: (cfg["fast_lr"], cfg["decay"]) for n in FAST_LAYERS[cfg["fast_layers"]]}
    raise SystemExit("%s: unknown method '%s' (sgd, dual or metasgd)." % (cfg["name"], cfg["method"]))


def run(S, lr, fast, x, y):
    """One online pass. Returns the correctness of each prediction, or None if it diverged."""
    S = {n: S[n].clone() for n in NAMES}
    Fw = {n: torch.zeros_like(S[n]) for n in fast}
    predictions = torch.empty(len(y), dtype=torch.long)
    for t in range(len(y)):
        W = {n: (S[n] + Fw[n] if n in Fw else S[n]).detach().requires_grad_() for n in NAMES}
        logits = forward(W, x[t:t + 1])
        predictions[t] = logits.argmax(1)[0]
        loss = F.cross_entropy(logits, y[t:t + 1])
        if not torch.isfinite(loss):
            return None
        grads = torch.autograd.grad(loss, [W[n] for n in NAMES])
        with torch.no_grad():
            for n, g in zip(NAMES, grads):
                if lr:
                    S[n].add_(g, alpha=-lr)
                if n in Fw:
                    rate, decay = fast[n]
                    if torch.is_tensor(rate):
                        Fw[n].mul_(decay).sub_(rate * g)
                    else:
                        Fw[n].mul_(decay).sub_(g, alpha=rate)
    return (predictions == y).numpy()


def pct(correct):
    return "%.2f" % (correct.mean() * 100)


def result_row(cfg, h, stream, frozen, bursts, shuffled, position):
    row = {"config": cfg["name"], "method": cfg["method"], "h": h, "stream": stream, "frozen": pct(frozen),
           "diverged": "true" if bursts is None or shuffled is None else "false"}
    if bursts is not None:
        row.update(bursts=pct(bursts), gain="%.2f" % ((bursts.mean() - frozen.mean()) * 100))
        for name, lo, hi in BINS:
            mask = (position >= lo) & (position <= hi)
            row[name] = pct(bursts[mask]) if mask.any() else ""
    if shuffled is not None:
        row["shuffled"] = pct(shuffled)
    if bursts is not None and shuffled is not None:
        row["locality"] = "%.2f" % ((bursts.mean() - shuffled.mean()) * 100)
    return row


def save_rows(exp_dir, config, new):
    """Write results.csv: the new rows replace those with the same config, h and stream."""
    path = os.path.join(exp_dir, "results.csv")
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            rows = {(r["config"], int(r["h"]), r["stream"]): r for r in csv.DictReader(f)}
    rows.update({(r["config"], r["h"], r["stream"]): r for r in new})
    names = [c["name"] for c in config["configurations"]]
    order = lambda k: (names.index(k[0]) if k[0] in names else len(names), k[0],
                       -k[1], config["streams"].index(k[2]) if k[2] in config["streams"] else 9)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows, key=order))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("configurations", nargs="*", help="names to run (default: all)")
    args = parser.parse_args()
    exp_dir = experiment_dir(args.experiment)
    config = load_config(exp_dir)
    configs = [c for c in config["configurations"] if not args.configurations or c["name"] in args.configurations]
    unknown = set(args.configurations) - {c["name"] for c in configs}
    if unknown:
        raise SystemExit("Unknown configuration(s): %s" % ", ".join(sorted(unknown)))
    for cfg in configs:  # fail early on a bad method or a missing checkpoint
        for h in config["sizes"]:
            setup(cfg, h, exp_dir)
    os.makedirs(os.path.join(exp_dir, "curves"), exist_ok=True)

    x_test, y_test = load_split("test")
    cache = under_every_context(x_test)
    rows = []
    for stream in config["streams"]:
        data = {}
        for shuffled in (False, True):
            idx, bursts, ctx = load_stream(stream, shuffled)
            data[shuffled] = (cache[torch.from_numpy(ctx), torch.from_numpy(idx)], y_test[torch.from_numpy(idx)], bursts)
        x, y, bursts = data[False]
        position = position_in_burst(bursts)
        for h in config["sizes"]:
            with torch.no_grad():
                frozen = (forward(load_mixed(h), x).argmax(1) == y).numpy()
            for cfg in configs:
                start = time.time()
                torch.manual_seed(config.get("seed", 0))
                out = {s: run(*setup(cfg, h, exp_dir), data[s][0], data[s][1]) for s in (False, True)}
                row = result_row(cfg, h, stream, frozen, out[False], out[True], position)
                rows.append(row)
                curve = os.path.join(exp_dir, "curves", "%s_h%d_%s.npz" % (cfg["name"], h, stream))
                if out[False] is not None:
                    np.savez_compressed(curve, online=out[False], frozen=frozen, position=position)
                elif os.path.exists(curve):
                    os.remove(curve)
                print("%-24s h=%d %-6s frozen %s  bursts %6s  shuffled %6s  gain %6s  locality %6s  (%.0f s)"
                      % (cfg["name"], h, stream, row["frozen"], row.get("bursts", "div"), row.get("shuffled", "div"),
                         row.get("gain", "-"), row.get("locality", "-"), time.time() - start), flush=True)
                save_rows(exp_dir, config, rows)


if __name__ == "__main__":
    main()
