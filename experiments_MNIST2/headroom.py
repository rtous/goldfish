"""Offline headroom check: is there anything for online adaptation to gain?

For each context family and model size h, two offline trainings (Adam, lr 0.001, batch 128):
    mixed        one model on the training set, each image under a context drawn uniformly at
                 random every time it is sampled (the frozen baseline; the same training as
                 mnist_mlp.py, so with the same seed and epochs it gives the same weights)
    per-context  one model per context, trained only on that context: the ceiling, what perfect
                 adaptation to the current context could reach at this size
Both are scored on all 10000 test images under each context (the per-context models on their own
context only), averaged over the contexts. gap = per-context - mixed, in percentage points.

Writes, in the experiment folder:
    headroom.csv                 one row per family, number of contexts and h (rows of other runs
                                 are kept)
    headroom_curves.csv          test accuracy after every epoch, to check convergence
    weights/percontext/<family>_<h>_c<k>of<n>.pt   the per-context models (mnist_mlp.py scores
                                 them on the stream as the oracle_accuracy column)
    plots/headroom.{png,pdf}     mixed vs per-context against model size, one panel per family
    plots/headroom_curves.png    the epoch curves

Usage:
    python3 headroom.py experiment1                                  # both families, all sizes
    python3 headroom.py 1 --families rotated --sizes 8 4 2 --n-contexts 16 --epochs 30
    python3 headroom.py 1 --plot-only                                # replot from the csv files
"""
import argparse
import csv
import fcntl
import os
import time

import numpy as np
import torch

import mnist_mlp
from contexts import DEFAULT_CONTEXTS, FAMILIES, Family
from mnist_mlp import (build_model, context_accuracies, get_device, load_split, model_size_bytes,
                       percontext_file, predict, save_weights, set_experiment, train_model)

SIZES = (128, 32, 8, 4, 2)
FIELDS = ["family", "n_contexts", "hidden_size", "epochs", "size_bytes", "mixed_accuracy",
          "percontext_accuracy", "gap_pp", "mixed_by_context", "percontext_by_context"]
CURVE_FIELDS = ["family", "n_contexts", "hidden_size", "model", "epoch", "loss", "test_accuracy"]


def config_for(family, h, epochs, seed):
    return {"name": "%s_%d_headroom" % (family.name, h), "model": "mlp", "seed": seed,
            "stream": family.name, "n_contexts": family.n_contexts,
            "model_params": {"hidden_size": h},
            "training_params": {"learning_rate": 0.001, "epochs": epochs, "batch_size": 128}}


def locked_update(path, fields, key_fields, new_rows):
    """Merge new_rows into the csv at path (rows with the same key are replaced), under a file
    lock so that runs in parallel processes do not overwrite each other."""
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                rows = list(csv.DictReader(f))
        keys = {tuple(str(r[k]) for k in key_fields) for r in new_rows}
        rows = [r for r in rows if tuple(r[k] for k in key_fields) not in keys] + new_rows
        rows.sort(key=lambda r: (FAMILIES.index(r["family"]), int(r["n_contexts"]),
                                 -int(r["hidden_size"]), r.get("model", ""), int(r.get("epoch", 0))))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.remove(path + ".lock")


def run_size(family, h, epochs, seed, device, x_train, y_train, cache, y_test, exp_dir):
    config = config_for(family, h, epochs, seed)
    curves = []

    def recorder(name, evaluate):
        def after_epoch(epoch, model, loss):
            acc = evaluate(model)
            curves.append({"family": family.name, "n_contexts": family.n_contexts, "hidden_size": h,
                           "model": name, "epoch": epoch, "loss": "%.4f" % loss,
                           "test_accuracy": "%.2f" % acc})
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print("  %s h=%d %s epoch %2d  loss %.4f  test %.2f%%"
                      % (family.name, h, name, epoch, loss, acc))
        return after_epoch

    start = time.time()
    mixed = train_model(config, family, x_train, y_train, device,
                        after_epoch=recorder("mixed", lambda m: np.mean(context_accuracies(m, cache, y_test))))
    mixed_by_ctx = context_accuracies(mixed, cache, y_test)

    def own_context(k):
        def evaluate(model):
            model.eval()
            with torch.no_grad():
                return (predict(model, cache[k]) == y_test).float().mean().item() * 100
        return evaluate

    ceiling_by_ctx = []
    for k in range(family.n_contexts):
        model = train_model(config, family, x_train, y_train, device, context=k,
                            after_epoch=recorder("c%d" % k, own_context(k)))
        ceiling_by_ctx.append(own_context(k)(model))
        save_weights(model, config, percontext_file(family.name, h, k, family.n_contexts))
    mixed_acc, ceiling_acc = float(np.mean(mixed_by_ctx)), float(np.mean(ceiling_by_ctx))
    print("%s h=%d: mixed %.2f%%, per-context %.2f%%, gap %+.2f pp  (%.0f s)"
          % (family.name, h, mixed_acc, ceiling_acc, ceiling_acc - mixed_acc, time.time() - start))
    row = {"family": family.name, "n_contexts": family.n_contexts, "hidden_size": h,
           "epochs": epochs, "size_bytes": model_size_bytes(mixed),
           "mixed_accuracy": "%.2f" % mixed_acc, "percontext_accuracy": "%.2f" % ceiling_acc,
           "gap_pp": "%.2f" % (ceiling_acc - mixed_acc),
           "mixed_by_context": " ".join("%.2f" % a for a in mixed_by_ctx),
           "percontext_by_context": " ".join("%.2f" % a for a in ceiling_by_ctx)}
    locked_update(os.path.join(exp_dir, "headroom.csv"), FIELDS,
                  ["family", "n_contexts", "hidden_size"], [row])
    locked_update(os.path.join(exp_dir, "headroom_curves.csv"), CURVE_FIELDS,
                  ["family", "n_contexts", "hidden_size", "model", "epoch"], curves)


def plot(exp_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_results import OFFLINE_COLOR, SERIES_COLORS, TEXT, line, style

    with open(os.path.join(exp_dir, "headroom.csv")) as f:
        rows = list(csv.DictReader(f))
    groups = sorted({(r["family"], int(r["n_contexts"])) for r in rows},
                    key=lambda g: (FAMILIES.index(g[0]), g[1]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4), squeeze=False, sharey=True)
    for ax, (fam, n) in zip(axes.flat, groups):
        rs = sorted([r for r in rows if r["family"] == fam and int(r["n_contexts"]) == n],
                    key=lambda r: int(r["size_bytes"]))
        kb = [int(r["size_bytes"]) / 1024 for r in rs]
        line(ax, kb, [float(r["mixed_accuracy"]) for r in rs], OFFLINE_COLOR, "mixed (frozen baseline)", dashed=True)
        line(ax, kb, [float(r["percontext_accuracy"]) for r in rs], SERIES_COLORS[0], "per-context (ceiling)")
        for x, r in zip(kb, rs):
            mid = (float(r["mixed_accuracy"]) + float(r["percontext_accuracy"])) / 2
            ax.annotate("h=%s: +%.1f" % (r["hidden_size"], float(r["gap_pp"])), (x, mid),
                        textcoords="offset points", xytext=(5, 0), va="center", fontsize=7, color=TEXT)
        ax.set_title("%s, %d contexts" % (fam, n), fontsize=10, color=TEXT)
        style(ax, "test accuracy, mean over contexts (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(exp_dir, "plots", "headroom.%s" % ext), dpi=150)

    with open(os.path.join(exp_dir, "headroom_curves.csv")) as f:
        curves = list(csv.DictReader(f))
    sizes = sorted({int(r["hidden_size"]) for r in curves}, reverse=True)
    fig, axes = plt.subplots(len(groups), len(sizes), figsize=(2.8 * len(sizes), 2.4 * len(groups)),
                             squeeze=False)
    for i, (fam, n) in enumerate(groups):
        for j, h in enumerate(sizes):
            ax = axes[i, j]
            rs = [r for r in curves if r["family"] == fam and int(r["n_contexts"]) == n and int(r["hidden_size"]) == h]
            for name in sorted({r["model"] for r in rs}, key=lambda m: (m != "mixed", m)):
                pts = [(int(r["epoch"]), float(r["test_accuracy"])) for r in rs if r["model"] == name]
                ax.plot(*zip(*pts), color=OFFLINE_COLOR if name == "mixed" else SERIES_COLORS[0],
                        linewidth=2 if name == "mixed" else 0.8, alpha=1 if name == "mixed" else 0.6)
            ax.set_title("%s/%d h=%d" % (fam, n, h), fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("test accuracy per epoch: mixed (grey) and each per-context model (blue)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "plots", "headroom_curves.png"), dpi=120)
    print("Saved", os.path.join(exp_dir, "plots", "headroom.{png,pdf}"), "and headroom_curves.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-contexts", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: default)")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    exp_dir = set_experiment(args.experiment)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    if args.plot_only:
        return plot(exp_dir)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device()
    x_train, y_train = load_split("train", device)
    x_test, y_test = load_split("test", device)
    for name in args.families:
        family = Family(name, args.n_contexts, args.seed)
        cache = family.test_cache(x_test)
        for h in args.sizes:
            run_size(family, h, args.epochs, args.seed, device, x_train, y_train, cache, y_test, exp_dir)
    plot(exp_dir)


if __name__ == "__main__":
    main()
