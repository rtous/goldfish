"""Online learning and its controls for every configuration x architecture of an experiment, on the
"alphabets" stream and its shuffled control. For each image the model first predicts its label
(this counts towards the accuracy), then may learn from it.

Gradient methods (MNIST2's update rule). The network predicts with S + F, F starts at 0, and with
g the gradient of the image's cross-entropy loss at S + F:
    S <- S - lr * g                    (skipped if lr = 0)
    F <- decay * F - fast_lr * g       (only on the fast parameters)
    sgd      {"lr", optional "layers": "out_bias" (only the output bias; control 2c)}
    dual     {"lr", "fast_lr", "decay"}: a fast part on every layer
    metasgd  lr = 0, a fast part on every parameter; S, fast_lr (alpha) and decay (lambda) are
             per-parameter tensors from <experiment>/weights/<config>_<arch>.pt (meta_train.py)
sgd and dual start from the mixed model (weights/mixed_<arch>.pt, offline.py).

Controls (no gradient; the mixed model frozen):
    frozen   no updates (control 1)
    prior    {"tuned": true}: logits + beta * log(count + eps), count an exponentially decaying
             count of the labels seen so far (count <- decay * count, then +1 for the label just
             seen); decay, beta, eps from <experiment>/tuned.json (tune.py, on val_alphabets)
             (control 2a). Optional "decay", "beta", "eps" keys override them.
    mask     logits restricted to the current image's alphabet: the model is told the alphabet
             (control 2b, an oracle)
    cache    {"M"}: the last M (image, label) pairs; p = (1 - mix) p_model + mix p_knn, with
             p_knn(y) proportional to sum over cached items of label y of exp(cos / temperature),
             cos the cosine similarity of the last hidden layer's activations; temperature and
             mix from tuned.json (control 3)

A NaN or inf loss stops a run (diverged). Writes one row per config x arch to
<experiment>/results.csv (see README) and curves/<config>_<arch>.npz (per-position correctness
of the bursts run and of the frozen model, position in the burst, occurrence).

Usage:
    python online.py 1                  # every configuration of experiment1
    python online.py 1 sgd_1e-3 prior   # only these
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import ALPHABET, experiment_dir, load_config, load_mixed, load_split, load_stream, position_in_burst
from models.mlp import N_CLASSES, forward, layer_params, param_names

STREAM = "alphabets"
POS_BINS = (("pos1_10", 1, 10), ("pos11_50", 11, 50), ("pos51_200", 51, 200), ("pos201_", 201, 10 ** 9))
OCC_BINS = (("occ0", 0, 0), ("occ1", 1, 1), ("occ2_3", 2, 3), ("occ4_7", 4, 7))
COLUMNS = (["config", "method", "arch", "stream", "frozen", "bursts", "shuffled", "gain", "gain_vs_prior",
            "gain_vs_mask", "locality"] + [b[0] for b in POS_BINS] + [b[0] for b in OCC_BINS]
           + ["model_kb", "extra_kb", "diverged"])
PRIOR_REF, MASK_REF = "prior", "mask"  # the configs that gain_vs_prior and gain_vs_mask refer to


def kb(n_floats):
    return n_floats * 4 / 1024


def load_tuned(exp_dir, arch):
    path = os.path.join(exp_dir, "tuned.json")
    if not os.path.exists(path):
        raise SystemExit("No %s: run tune.py first." % path)
    with open(path) as f:
        return json.load(f)[arch]


def setup(cfg, arch, exp_dir):
    """Starting weights S, slow lr, the fast part {name: (fast_lr, decay)}, the names S updates
    (None = all), and the extra online state in floats."""
    names = param_names(arch)
    n_params = layer_params(arch)[1]
    if cfg["method"] == "metasgd":
        path = os.path.join(exp_dir, "weights", "%s_%s.pt" % (cfg["name"], arch))
        if not os.path.exists(path):
            raise SystemExit("No %s: run meta_train.py first." % path)
        ck = torch.load(path)
        return ck["S"], 0.0, {n: (ck["alpha"][n], ck["lambda"][n]) for n in names}, None, 3 * n_params
    S = load_mixed(arch)
    if cfg["method"] == "sgd":
        out_bias = names[-1]
        return S, cfg["lr"], {}, ([out_bias] if cfg.get("layers") == "out_bias" else None), 0
    if cfg["method"] == "dual":
        return S, cfg["lr"], {n: (cfg["fast_lr"], cfg["decay"]) for n in names}, None, n_params
    raise SystemExit("%s: unknown method '%s'." % (cfg["name"], cfg["method"]))


def run_gradient(S, lr, fast, slow_names, x, y):
    """One online pass of the S + F rule; correctness per image, or None if it diverged."""
    names = list(S)
    S = {n: S[n].clone() for n in names}
    Fw = {n: torch.zeros_like(S[n]) for n in fast}
    learn = [n for n in names if (lr and (slow_names is None or n in slow_names)) or n in Fw]
    correct = np.zeros(len(y), dtype=bool)
    for t in range(len(y)):
        W = {n: (S[n] + Fw[n] if n in Fw else S[n]) for n in names}
        W = {n: W[n].detach().requires_grad_(n in learn) for n in names}
        logits = forward(W, x[t:t + 1])
        correct[t] = logits.argmax(1).item() == y[t].item()
        loss = F.cross_entropy(logits, y[t:t + 1])
        if not torch.isfinite(loss):
            return None
        grads = torch.autograd.grad(loss, [W[n] for n in learn])
        with torch.no_grad():
            for n, g in zip(learn, grads):
                if lr and (slow_names is None or n in slow_names):
                    S[n].add_(g, alpha=-lr)
                if n in Fw:
                    rate, decay = fast[n]
                    if torch.is_tensor(rate):
                        Fw[n].mul_(decay).sub_(rate * g)
                    else:
                        Fw[n].mul_(decay).sub_(g, alpha=rate)
    return correct


def run_prior(logits, y, decay, beta, eps):
    """Frozen logits plus beta * log(decaying label count + eps)."""
    count = np.zeros(logits.shape[1])
    pred = np.zeros(len(y), dtype=np.int64)
    for t in range(len(y)):
        pred[t] = np.argmax(logits[t] + beta * np.log(count + eps))
        count *= decay
        count[y[t]] += 1
    return pred == y


def run_cache(logits, feats, y, M, temperature, mix):
    """Mix of the frozen model's probabilities and a kNN distribution over the last M items."""
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    pred = np.zeros(len(y), dtype=np.int64)
    for t in range(len(y)):
        lo = max(0, t - M)
        p = probs[t]
        if t > lo:
            w = np.exp((f[lo:t] @ f[t] - 1) / temperature)
            knn = np.bincount(y[lo:t], weights=w, minlength=logits.shape[1])
            p = (1 - mix) * p + mix * knn / knn.sum()
        pred[t] = np.argmax(p)
    return pred == y


def run_control(cfg, tuned, logits, feats, y):
    m = cfg["method"]
    if m == "frozen":
        return logits.argmax(1) == y, 0
    if m == "mask":
        masked = np.where(ALPHABET[None, :] == ALPHABET[y][:, None], logits, -np.inf)
        return masked.argmax(1) == y, 0
    if m == "prior":
        p = {k: cfg.get(k, tuned["prior"][k]) for k in ("decay", "beta", "eps")}
        return run_prior(logits, y, p["decay"], p["beta"], p["eps"]), N_CLASSES
    if m == "cache":
        c = tuned["cache"][str(cfg["M"])]
        return run_cache(logits, feats, y, cfg["M"], c["temperature"], c["mix"]), cfg["M"] * (feats.shape[1] + 1)
    return None


def pct(correct):
    return "%.2f" % (correct.mean() * 100)


def result_row(cfg, arch, frozen, bursts, shuffled, position, occurrence, extra):
    row = {"config": cfg["name"], "method": cfg["method"], "arch": arch, "stream": STREAM, "frozen": pct(frozen),
           "model_kb": "%.1f" % kb(layer_params(arch)[1]), "extra_kb": "%.1f" % kb(extra),
           "diverged": "true" if bursts is None or shuffled is None else "false"}
    if bursts is not None:
        row.update(bursts=pct(bursts), gain="%.2f" % ((bursts.mean() - frozen.mean()) * 100))
        for bins, values in ((POS_BINS, position), (OCC_BINS, occurrence)):
            for name, lo, hi in bins:
                mask = (values >= lo) & (values <= hi)
                row[name] = pct(bursts[mask]) if mask.any() else ""
    if shuffled is not None:
        row["shuffled"] = pct(shuffled)
    if bursts is not None and shuffled is not None:
        row["locality"] = "%.2f" % ((bursts.mean() - shuffled.mean()) * 100)
    return row


def save_rows(exp_dir, config, new):
    """Write results.csv (new rows replace those with the same config and arch) and recompute
    gain_vs_prior and gain_vs_mask from the "prior" and "mask" rows."""
    path = os.path.join(exp_dir, "results.csv")
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            rows = {(r["config"], r["arch"]): r for r in csv.DictReader(f)}
    rows.update({(r["config"], r["arch"]): r for r in new})
    for (name, arch), r in rows.items():
        for col, ref in (("gain_vs_prior", PRIOR_REF), ("gain_vs_mask", MASK_REF)):
            other = rows.get((ref, arch))
            r[col] = "%.2f" % (float(r["bursts"]) - float(other["bursts"])) if other and r.get("bursts") else ""
    names = [c["name"] for c in config["configurations"]]
    order = lambda k: (config["archs"].index(k[1]) if k[1] in config["archs"] else 9,
                       names.index(k[0]) if k[0] in names else len(names), k[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows, key=order))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("configurations", nargs="*", help="names to run (default: all)")
    parser.add_argument("--archs", nargs="*", help="architectures to run (default: all)")
    args = parser.parse_args()
    exp_dir = experiment_dir(args.experiment)
    config = load_config(exp_dir)
    configs = [c for c in config["configurations"] if not args.configurations or c["name"] in args.configurations]
    unknown = set(args.configurations) - {c["name"] for c in configs}
    if unknown:
        raise SystemExit("Unknown configuration(s): %s" % ", ".join(sorted(unknown)))
    os.makedirs(os.path.join(exp_dir, "curves"), exist_ok=True)
    torch.manual_seed(config.get("seed", 0))

    x_test, y_test = load_split("test")
    data = {}
    for shuffled in (False, True):
        idx, bursts, _, occurrence = load_stream(STREAM, shuffled)
        data[shuffled] = (x_test[torch.from_numpy(idx)], y_test[torch.from_numpy(idx)])
        if not shuffled:
            position, occ = position_in_burst(bursts), occurrence
    rows = []
    for arch in args.archs or config["archs"]:
        S0 = load_mixed(arch)
        frozen_out = {}
        for s in (False, True):
            with torch.no_grad():
                logits, feats = forward(S0, data[s][0], hidden=True)
            frozen_out[s] = (logits.numpy(), feats.numpy(), data[s][1].numpy())
        frozen = frozen_out[False][0].argmax(1) == frozen_out[False][2]
        for cfg in configs:
            start = time.time()
            if cfg["method"] in ("frozen", "mask", "prior", "cache"):
                tuned = load_tuned(exp_dir, arch) if cfg["method"] in ("prior", "cache") else None
                out = {s: run_control(cfg, tuned, *frozen_out[s]) for s in (False, True)}
                extra = out[False][1]
                out = {s: out[s][0] for s in out}
            else:
                S, lr, fast, slow_names, extra = setup(cfg, arch, exp_dir)
                out = {s: run_gradient(S, lr, fast, slow_names, *data[s]) for s in (False, True)}
            row = result_row(cfg, arch, frozen, out[False], out[True], position, occ, extra)
            rows.append(row)
            if out[False] is not None:
                np.savez_compressed(os.path.join(exp_dir, "curves", "%s_%s.npz" % (cfg["name"], arch)),
                                    online=out[False], frozen=frozen, position=position, occurrence=occ)
            print("%-18s %s frozen %s  bursts %6s  shuffled %6s  gain %6s  locality %6s  occ0 %6s  occ4_7 %6s  (%.0f s)"
                  % (cfg["name"], arch, row["frozen"], row.get("bursts", "div"), row.get("shuffled", "div"),
                     row.get("gain", "-"), row.get("locality", "-"), row.get("occ0", "-"), row.get("occ4_7", "-"),
                     time.time() - start), flush=True)
            save_rows(exp_dir, config, rows)


if __name__ == "__main__":
    main()
