"""Test 9: Meta-SGD lite (one learned alpha and lam per weight tensor).

Fast part on 25 tensors (dual.all_weight_matrices): per layer attention qkv, attention out, MLP in,
MLP out, plus the output embedding; no fast part on the input embedding, biases or layer norms.
alpha = exp(a), lam = sigmoid(b), one pair per tensor (50 numbers). Initialized from test 8's
picked config (its fast tensors get its alpha and lam; the other tensors alpha 1e-4, lam 0.9). The
slow part is plain SGD on all parameters, lr 1e-4 (dual.py).

Meta-training on val-meta-train, K = 1: the stream is read continuously in episodes of T = 16
chunks, carrying S and F across episodes (a new pass over the stream starts from the pretrained
weights with F = 0). Meta-loss: mean prequential loss over the episode's chunks, backpropagated
to (a, b) through the episode's 16 updates with exact second-order terms (dual.meta_episode; the
state at the episode start is a constant). Adam on (a, b), lr 1e-2. Every EVAL_EVERY meta-steps
(and before the first), a full prequential pass on val-meta-heldout with the current values; the
best is kept. Stop after MAX_MINUTES or when the held-out loss has not improved for PATIENCE
meta-steps. Then the best rule runs on test 7's test streams (K1, shuffled, long), evaluated like
test 8.

Writes test9_metasgd_lite/: results.csv, bins.csv, position.csv, learned.csv (alpha, lam per
tensor), meta_curve.csv, losses.npz, runtime.txt, plots/. Usage:
    python test9_metasgd_lite.py          # everything (about 45 min)
    python test9_metasgd_lite.py plot     # plots from the saved CSVs
"""
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch

from data import HERE
from dual import (MetaRule, all_weight_matrices, dual_losses, fresh_state, half_life_chunks, meta_episode,
                  val_sets)
from online import setup
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from test3_baselines import AQUA
from test8_dual import CONFIGS, PARTS, evaluate_test, test_sets, write_csv

OUT = os.path.join(HERE, "test9_metasgd_lite")
T8 = os.path.join(HERE, "test8_dual")
T = 16
META_LR = 1e-2
EVAL_EVERY = 50
PATIENCE = 200
MAX_MINUTES = 30
OTHER = (1e-4, 0.9)
TYPES = {"attention.query_key_value": "attn qkv", "attention.dense": "attn out", "mlp.dense_h_to_4h": "MLP in",
         "mlp.dense_4h_to_h": "MLP out", "lm_head": "output emb"}


def describe(name):
    m = re.search(r"layers\.(\d+)\.", name)
    layer = str(int(m.group(1)) + 1) if m else "-"
    kind = next(v for k, v in TYPES.items() if k in name)
    return layer, kind


def init_values():
    pick = json.loads(open(os.path.join(T8, "runtime.txt")).readline())["pick_val_K1"]
    _, part, alpha, lam = next(c for c in CONFIGS if c[0] == pick)
    return pick, {n: ((alpha, lam) if n in PARTS[part] else OTHER) for n in all_weight_matrices()}


def heldout_loss(model, init, vsets, vstreams, rule):
    set_name, order = vstreams["val_heldout_K1"]
    fast = {n: (float(a), float(l)) for n, (a, l) in rule.detached().items()}
    model.set_attn_implementation("sdpa")
    loss = float(np.nanmean(dual_losses(model, init, vsets[set_name], order, fast)))
    model.set_attn_implementation("eager")
    return loss


def main():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    torch.manual_seed(0)
    model, init, info, _ = setup()
    pick, start = init_values()
    rule = MetaRule(start)
    opt = torch.optim.Adam(rule.parameters(), lr=META_LR)
    vsets, vstreams = val_sets()
    set_name, order = vstreams["val_train_K1"]
    train = [torch.from_numpy(vsets[set_name][b, c]) for b, c in order]
    model.set_attn_implementation("eager")
    for p in model.parameters():
        p.requires_grad_(False)
    t0 = time.time()
    curve, train_losses = [], []
    best = (heldout_loss(model, init, vsets, vstreams, rule), 0, {k: v.detach().clone() for k, v in rule.state_dict().items()})
    curve.append({"step": 0, "minutes": "%.1f" % ((time.time() - t0) / 60), "train_meta_loss": "",
                  "heldout_loss": "%.4f" % best[0]})
    print(curve[-1], flush=True)
    state, pos, step, passes = fresh_state(model, init, rule), 0, 0, 0
    while True:
        if pos + T > len(train):  # end of the training stream: a new pass from the pretrained weights
            state, pos, passes = fresh_state(model, init, rule), 0, passes + 1
        opt.zero_grad()
        loss, state = meta_episode(model, rule, state, train[pos:pos + T], exact=True)
        loss.backward()
        opt.step()
        pos += T
        step += 1
        train_losses.append(loss.item())
        if step % EVAL_EVERY == 0:
            h = heldout_loss(model, init, vsets, vstreams, rule)
            if h < best[0]:
                best = (h, step, {k: v.detach().clone() for k, v in rule.state_dict().items()})
            curve.append({"step": step, "minutes": "%.1f" % ((time.time() - t0) / 60),
                          "train_meta_loss": "%.4f" % np.mean(train_losses[-EVAL_EVERY:]), "heldout_loss": "%.4f" % h})
            print(curve[-1], "best step", best[1], flush=True)
            if step - best[1] >= PATIENCE:
                stop = "no held-out improvement for %d meta-steps" % PATIENCE
                break
        if time.time() - t0 > MAX_MINUTES * 60:
            stop = "time limit (%d min)" % MAX_MINUTES
            break
    meta_minutes = (time.time() - t0) / 60
    rule.load_state_dict(best[2])
    model.set_attn_implementation("sdpa")
    for p in model.parameters():
        p.requires_grad_(True)
    learned = []
    for n, (a, l) in rule.detached().items():
        layer, kind = describe(n)
        a0, l0 = start[n]
        learned.append({"tensor": n, "layer": layer, "type": kind, "alpha": "%.3e" % float(a), "lam": "%.4f" % float(l),
                        "half_life_chunks": "%.1f" % half_life_chunks(float(l)), "alpha_init": "%g" % a0,
                        "lam_init": "%g" % l0})
    write_csv(OUT, "learned.csv", learned)
    write_csv(OUT, "meta_curve.csv", curve)
    fast = {n: (float(a), float(l)) for n, (a, l) in rule.detached().items()}
    sets, streams = test_sets()
    rows, bins, pos_rows, seconds, check = evaluate_test(model, init, sets, streams, {"metasgd_lite": fast}, OUT)
    write_csv(OUT, "results.csv", rows)
    write_csv(OUT, "bins.csv", bins)
    write_csv(OUT, "position.csv", pos_rows)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write(json.dumps({"init_from_test8_pick": pick, "meta_steps": step, "best_step": best[1],
                            "best_heldout_loss": round(best[0], 4), "passes_over_train": passes, "stop": stop,
                            "meta_minutes": round(meta_minutes, 1), "exact_meta_gradients": True,
                            "full_1e-4_reproduces_test7": check}) + "\n")
        f.write("%s\n%s\n" % (info["device"], ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    plot()


def plot():
    learned = list(csv.DictReader(open(os.path.join(OUT, "learned.csv"))))
    curve = list(csv.DictReader(open(os.path.join(OUT, "meta_curve.csv"))))
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    kinds = list(TYPES.values())
    colors = dict(zip(kinds, (BLUE, AQUA, ORANGE, INK, MUTED)))
    markers = dict(zip(kinds, ("o", "s", "D", "^", "*")))
    xpos = {str(i): i for i in range(1, 7)}
    xpos["-"] = 7
    for ax, key, label in ((axes[0], "alpha", "fast learning rate alpha (log)"),
                           (axes[1], "half_life_chunks", "fast-part half-life (chunks, log)")):
        for kind in kinds:
            pts = [(xpos[r["layer"]], float(r[key])) for r in learned if r["type"] == kind]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[kind], marker=markers[kind], ms=6, lw=1,
                    label=kind)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 8), ["1", "2", "3", "4", "5", "6", "out"])
        ax.set_xlabel("layer")
        ax.set_title(label, fontsize=8, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "learned_values.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    steps = [int(r["step"]) for r in curve]
    ax.plot(steps, [float(r["heldout_loss"]) for r in curve], color=BLUE, lw=2, marker="o", ms=3,
            label="val-meta-heldout (full pass)")
    tr = [(int(r["step"]), float(r["train_meta_loss"])) for r in curve if r["train_meta_loss"]]
    ax.plot([s for s, _ in tr], [v for _, v in tr], color=ORANGE, lw=1.4, marker="o", ms=3,
            label="val-meta-train (mean of last %d episodes)" % EVAL_EVERY)
    ax.set_xlabel("meta-step")
    ax.set_ylabel("loss (nats/token)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax.set_title("Meta-training", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "meta_curve.png"), dpi=160)


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else main()
