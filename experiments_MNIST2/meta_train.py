"""Meta-SGD with a slow/fast split (experiment3_metaSGD): meta-learn the slow weights S and a
per-parameter fast learning rate alpha and decay lambda, so that online adaptation to a rotation
context is fast.

Online protocol (the same as online_mlp.py with fast weights on every parameter and no slow
step): the network predicts with S + F; F starts at 0 and after each image
    F <- lambda * F - alpha * g,   g = gradient of that image's loss w.r.t. the weights (at S + F)
with alpha = exp(log_alpha) > 0 and lambda = sigmoid(logit_lambda) in (0, 1), one value per
parameter. S is frozen online.

Meta-training (MNIST training images only; 10000 of them are held out for early stopping):
    episode     3 consecutive bursts with different rotation contexts (contexts.py, 8 contexts),
                each of B images, B uniform in [20, 60], images drawn at random
    inner loop  the online protocol, from F = 0
    meta-loss   mean prediction loss over the episode (each image's loss before its update),
                backpropagated through the whole unroll. The per-image gradient of the MLP is
                written in closed form with differentiable operations, so the meta-gradient is
                exact (second order).
    optimizer   Adam, lr 1e-3, meta-batch of 32 episodes, meta-gradient norm clipped to 1
    init        S from experiment1's mixed rotated model of the same h, alpha = 1e-3,
                lambda = 0.99
    stopping    every --eval-every steps, online accuracy on 256 fixed held-out episodes; stop
                after --patience evaluations without improvement (or --max-minutes); keep the best

Evaluation, on experiment1's rotated stream (sequence_rotated.csv) and its shuffled control:
the frozen mixed model, the meta S frozen (alpha = 0), and meta-SGD online, with accuracy by
position within the burst (1-5, 6-20, 21-60, 61+) and the meta-learned alpha and lambda.

Writes experiment3_metaSGD/results.csv (one row per h), weights/meta_h<h>.pt,
meta_curve_h<h>.csv and plots/meta_training.{png,pdf}.

Usage:
    python3 meta_train.py experiment3_metaSGD                # h = 4 and 8
    python3 meta_train.py 3 --sizes 4 --max-minutes 30
    python3 meta_train.py 3 --eval-only                      # reevaluate the saved weights
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from contexts import Family
from mnist_mlp import HERE, load_sequence, load_split

NAMES = ("0.weight", "0.bias", "2.weight", "2.bias")
HOLDOUT = 10000
EPISODE_BURSTS = 3
BURST_RANGE = (20, 60)
POSITION_BINS = (("pos1_5", 1, 5), ("pos6_20", 6, 20), ("pos21_60", 21, 60), ("pos61plus", 61, None))
FIELDS = (["hidden_size", "frozen_mixed", "frozen_mixed_random", "meta_s_frozen", "meta_s_test_set",
           "mixed_test_set", "meta_online_bursts", "meta_online_random", "gain", "locality_effect",
           "best_handtuned_variant", "best_handtuned_bursts", "oracle", "headroom_percontext"]
          + ["%s_%s" % (m, b[0]) for m in ("frozen", "metaS", "meta") for b in POSITION_BINS]
          + ["median_alpha_" + n for n in NAMES] + ["median_lambda_" + n for n in NAMES]
          + ["meta_steps", "meta_minutes", "best_val_online", "val_frozen_start"])


def forward(W, x):
    """Logits and hidden activations of the MLP with weights W (batched over episodes: W[n] has a
    leading episode dimension, x is [E, 784])."""
    pre = torch.einsum("ehi,ei->eh", W["0.weight"], x) + W["0.bias"]
    hid = pre.clamp(min=0)
    logits = torch.einsum("eoh,eh->eo", W["2.weight"], hid) + W["2.bias"]
    return logits, pre, hid


def grads(W, x, y, logits, pre, hid):
    """Gradient of each episode's cross-entropy loss on its image, in closed form; differentiable,
    so backpropagating through it gives the exact second-order meta-gradient."""
    d_logits = logits.softmax(dim=1) - F.one_hot(y, 10).to(logits.dtype)  # [E, 10]
    d_pre = torch.einsum("eoh,eo->eh", W["2.weight"], d_logits) * (pre > 0).to(pre.dtype)
    return {"2.weight": d_logits.unsqueeze(2) * hid.unsqueeze(1), "2.bias": d_logits,
            "0.weight": d_pre.unsqueeze(2) * x.unsqueeze(1), "0.bias": d_pre}


class MetaSGD(torch.nn.Module):
    def __init__(self, state_dict, alpha=1e-3, lam=0.99):
        super().__init__()
        self.S = torch.nn.ParameterDict({n.replace(".", "_"): torch.nn.Parameter(state_dict[n].clone())
                                         for n in NAMES})
        self.log_alpha = torch.nn.ParameterDict({n.replace(".", "_"): torch.nn.Parameter(
            torch.full_like(state_dict[n], float(np.log(alpha)))) for n in NAMES})
        self.logit_lambda = torch.nn.ParameterDict({n.replace(".", "_"): torch.nn.Parameter(
            torch.full_like(state_dict[n], float(np.log(lam / (1 - lam))))) for n in NAMES})

    def params(self):
        """S, alpha and lambda by parameter name."""
        S = {n: self.S[n.replace(".", "_")] for n in NAMES}
        alpha = {n: self.log_alpha[n.replace(".", "_")].exp() for n in NAMES}
        lam = {n: torch.sigmoid(self.logit_lambda[n.replace(".", "_")]) for n in NAMES}
        return S, alpha, lam

    def unroll(self, x, y, mask=None, alpha_scale=1.0):
        """The online protocol over E parallel streams: x [E, T, 784], y [E, T]. Returns the
        per-image losses and correctness ([E, T], before each image's update). mask [E, T]
        (optional) marks the real positions; updates at padded positions use no gradient."""
        S, alpha, lam = self.params()
        E, T = y.shape
        fast = {n: torch.zeros((E,) + S[n].shape, dtype=S[n].dtype) for n in NAMES}
        losses, correct = [], []
        for t in range(T):
            W = {n: S[n] + fast[n] for n in NAMES}
            logits, pre, hid = forward(W, x[:, t])
            losses.append(F.cross_entropy(logits, y[:, t], reduction="none"))
            correct.append(logits.argmax(1) == y[:, t])
            g = grads(W, x[:, t], y[:, t], logits, pre, hid)
            keep = None if mask is None else mask[:, t].to(x.dtype)
            for n in NAMES:
                step = alpha[n] * alpha_scale * g[n]
                if keep is not None:
                    step = step * keep.view((E,) + (1,) * (step.dim() - 1))
                fast[n] = lam[n] * fast[n] - step
        return torch.stack(losses, 1), torch.stack(correct, 1)


def episodes(rng, family, x_raw, y, pool, count):
    """count episodes: 3 bursts of B in [20, 60] images (drawn from pool), consecutive contexts
    different. Returns x [E, T, 784], y [E, T], mask [E, T], padded to the longest episode."""
    lengths, items = [], []
    for _ in range(count):
        ctx = [int(rng.integers(family.n_contexts))]
        while len(ctx) < EPISODE_BURSTS:
            ctx.append((ctx[-1] + 1 + int(rng.integers(family.n_contexts - 1))) % family.n_contexts)
        sizes = rng.integers(BURST_RANGE[0], BURST_RANGE[1] + 1, size=EPISODE_BURSTS)
        idx = rng.choice(pool, size=int(sizes.sum()), replace=False)
        items.append((idx, np.repeat(ctx, sizes)))
        lengths.append(int(sizes.sum()))
    T = max(lengths)
    xs = torch.zeros(count, T, 784)
    ys = torch.zeros(count, T, dtype=torch.long)
    mask = torch.zeros(count, T, dtype=torch.bool)
    for e, (idx, ctx) in enumerate(items):
        n = len(idx)
        xs[e, :n] = family.transform(x_raw[torch.from_numpy(idx)], torch.from_numpy(ctx))
        ys[e, :n] = y[torch.from_numpy(idx)]
        mask[e, :n] = True
    return xs, ys, mask


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def train(h, exp_dir, args, x_train, y_train, family, log):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    order = rng.permutation(len(x_train))
    holdout, pool = np.sort(order[:HOLDOUT]), np.sort(order[HOLDOUT:])
    assert len(set(holdout) & set(pool)) == 0 and len(x_train) == 60000  # training images only
    init = torch.load(os.path.join(HERE, "experiment1", "weights", "rotated_%d_frozen.pt" % h))["state_dict"]
    model = MetaSGD(init)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    val = episodes(np.random.default_rng(args.seed + 1), family, x_train, y_train, holdout, args.val_episodes)

    def validate():
        with torch.no_grad():
            _, c = model.unroll(val[0], val[1], val[2])
            _, c0 = model.unroll(val[0], val[1], val[2], alpha_scale=0.0)
        return masked_mean(c.float(), val[2]).item() * 100, masked_mean(c0.float(), val[2]).item() * 100

    curve = []
    best, best_state, bad, step, start = -1.0, None, 0, 0, time.time()
    online, frozen = validate()
    val_frozen_start = frozen
    curve.append((0, None, online, frozen, 0.0))
    log("h=%d step 0: val online %.2f%%, S frozen %.2f%%" % (h, online, frozen))
    while True:
        x, y, mask = episodes(rng, family, x_train, y_train, pool, args.meta_batch)
        losses, _ = model.unroll(x, y, mask)
        loss = masked_mean(losses, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.eval_every == 0:
            online, frozen = validate()
            minutes = (time.time() - start) / 60
            curve.append((step, loss.item(), online, frozen, minutes))
            log("h=%d step %d (%.1f min): meta-loss %.4f, val online %.2f%%, S frozen %.2f%%"
                % (h, step, minutes, loss.item(), online, frozen))
            if online > best:
                best, bad = online, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= args.patience or minutes >= args.max_minutes:
                break
    model.load_state_dict(best_state)
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    torch.save({"hidden_size": h, "state_dict": best_state, "steps": step,
                "minutes": (time.time() - start) / 60, "best_val_online": best,
                "val_frozen_start": val_frozen_start},
               os.path.join(exp_dir, "weights", "meta_h%d.pt" % h))
    with open(os.path.join(exp_dir, "meta_curve_h%d.csv" % h), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "meta_loss", "val_online", "val_s_frozen", "minutes"])
        for row in curve:
            w.writerow(["" if v is None else ("%.4f" % v if isinstance(v, float) else v) for v in row])
    return model


def position_in_burst(bursts):
    starts = np.r_[True, bursts[1:] != bursts[:-1]]
    start_at = np.maximum.accumulate(np.where(starts, np.arange(len(bursts)), 0))
    return np.arange(len(bursts)) - start_at + 1


def bins(correct, position):
    out = {}
    for name, lo, hi in POSITION_BINS:
        m = (position >= lo) & (position <= (hi or len(position)))
        out[name] = correct[m].mean() * 100
    return out


def evaluate(model, h, x_test, y_test, family):
    """Online on the experiment1 rotated stream and its control (one stream: E = 1)."""
    cache = family.test_cache(x_test)
    out = {}
    for kind in ("bursts", "random"):
        idx, bursts, ctx = load_sequence("rotated", kind)
        x = cache[torch.from_numpy(ctx), torch.from_numpy(idx)].unsqueeze(0)
        y = y_test[torch.from_numpy(idx)].unsqueeze(0)
        with torch.no_grad():
            _, online = model.unroll(x, y)
            _, frozen = model.unroll(x, y, alpha_scale=0.0)
        out[kind] = (online[0].numpy(), frozen[0].numpy(), bursts)
    S, _, _ = model.params()
    with torch.no_grad():
        logits = cache.reshape(-1, 784) @ S["0.weight"].T + S["0.bias"]
        logits = logits.clamp(min=0) @ S["2.weight"].T + S["2.bias"]
        test_set = (logits.argmax(1).view(family.n_contexts, -1) == y_test).float().mean().item() * 100
        idx, _, ctx = load_sequence("rotated", "bursts")
        xs = cache[torch.from_numpy(ctx), torch.from_numpy(idx)]
        plain = ((xs @ S["0.weight"].T + S["0.bias"]).clamp(min=0) @ S["2.weight"].T + S["2.bias"]).argmax(1)
        plain = (plain == y_test[torch.from_numpy(idx)]).numpy()
    assert np.array_equal(plain, out["bursts"][1]), "alpha = 0 does not reproduce the frozen S"
    return out, test_set


def exp1_rows():
    with open(os.path.join(HERE, "experiment1", "results.csv")) as f:
        return {r["configuration"]: r for r in csv.DictReader(f)}


def exp1_headroom():
    with open(os.path.join(HERE, "experiment1", "headroom.csv")) as f:
        return {int(r["hidden_size"]): r for r in csv.DictReader(f) if r["family"] == "rotated"}


def mixed_bins(h, x_test, y_test, family):
    """Frozen mixed model (experiment1) correctness by position on the bursts stream."""
    sd = torch.load(os.path.join(HERE, "experiment1", "weights", "rotated_%d_frozen.pt" % h))["state_dict"]
    cache = family.test_cache(x_test)
    idx, bursts, ctx = load_sequence("rotated", "bursts")
    xs = cache[torch.from_numpy(ctx), torch.from_numpy(idx)]
    with torch.no_grad():
        logits = (xs @ sd["0.weight"].T + sd["0.bias"]).clamp(min=0) @ sd["2.weight"].T + sd["2.bias"]
    correct = (logits.argmax(1) == y_test[torch.from_numpy(idx)]).numpy()
    return correct, bins(correct, position_in_burst(bursts))


def report_row(h, model, saved, out, test_set, x_test, y_test, family):
    r1, head = exp1_rows(), exp1_headroom()
    frozen_row = r1["rotated_%d_frozen" % h]
    handtuned = {v: float(r1["rotated_%d_%s" % (h, v)]["bursts_accuracy"])
                 for v in ("sgd", "fastbias", "fastout", "fasthidden", "fastall")}
    best_v = max(handtuned, key=handtuned.get)
    online_b, frozen_b, bursts = out["bursts"]
    online_r, _, _ = out["random"]
    position = position_in_burst(bursts)
    mixed_correct, mixed_by_pos = mixed_bins(h, x_test, y_test, family)
    assert abs(mixed_correct.mean() * 100 - float(frozen_row["offline_accuracy"])) < 0.005
    _, alpha, lam = model.params()
    row = {"hidden_size": h,
           "frozen_mixed": frozen_row["offline_accuracy"],
           "frozen_mixed_random": frozen_row["random_accuracy"],
           "meta_s_frozen": "%.2f" % (frozen_b.mean() * 100),
           "meta_s_test_set": "%.2f" % test_set,
           "mixed_test_set": frozen_row["test_set_accuracy"],
           "meta_online_bursts": "%.2f" % (online_b.mean() * 100),
           "meta_online_random": "%.2f" % (online_r.mean() * 100),
           "gain": "%.2f" % (online_b.mean() * 100 - float(frozen_row["offline_accuracy"])),
           "locality_effect": "%.2f" % ((online_b.mean() - online_r.mean()) * 100),
           "best_handtuned_variant": best_v,
           "best_handtuned_bursts": "%.2f" % handtuned[best_v],
           "oracle": frozen_row["oracle_accuracy"],
           "headroom_percontext": head[h]["percontext_accuracy"],
           "meta_steps": saved["steps"], "meta_minutes": "%.1f" % saved["minutes"],
           "best_val_online": "%.2f" % saved["best_val_online"],
           "val_frozen_start": "%.2f" % saved["val_frozen_start"]}
    for prefix, values in (("frozen", mixed_by_pos), ("metaS", bins(frozen_b, position)),
                           ("meta", bins(online_b, position))):
        for name, v in values.items():
            row["%s_%s" % (prefix, name)] = "%.2f" % v
    for n in NAMES:
        row["median_alpha_" + n] = "%.3g" % alpha[n].median().item()
        row["median_lambda_" + n] = "%.4g" % lam[n].median().item()
    return row


def plot(exp_dir, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_results import OFFLINE_COLOR, SERIES_COLORS, line
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 3.6), squeeze=False)
    for ax, h in zip(axes.flat, sizes):
        with open(os.path.join(exp_dir, "meta_curve_h%d.csv" % h)) as f:
            rows = list(csv.DictReader(f))
        steps = [int(r["step"]) for r in rows]
        line(ax, steps, [float(r["val_online"]) for r in rows], SERIES_COLORS[0], "meta-SGD online (held-out episodes)")
        line(ax, steps, [float(r["val_s_frozen"]) for r in rows], OFFLINE_COLOR, "S frozen", dashed=True)
        ax.set_title("h = %d" % h, fontsize=10)
        ax.set_xlabel("meta-training step")
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, color="#e4e3df")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(exp_dir, "plots", "meta_training.%s" % ext), dpi=150)


def experiment_folder(name):
    """The experiment folder, given as a path, a folder name or a number ("3" ->
    experiment3_metaSGD); created if it does not exist (it has no configurations.json)."""
    for candidate in (name, os.path.join(HERE, name)):
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    matches = [d for d in os.listdir(HERE) if d.startswith("experiment" + name + "_")]
    if len(matches) == 1:
        return os.path.join(HERE, matches[0])
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--meta-batch", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--val-episodes", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6, help="evaluations without improvement")
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    exp_dir = experiment_folder(args.experiment)
    log_path = os.path.join(exp_dir, "meta_train.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    family = Family("rotated")
    x_train, y_train = load_split("train", "cpu")
    x_test, y_test = load_split("test", "cpu")
    rows = []
    for h in args.sizes:
        if not args.eval_only:
            train(h, exp_dir, args, x_train, y_train, family, log)
        saved = torch.load(os.path.join(exp_dir, "weights", "meta_h%d.pt" % h))
        init = torch.load(os.path.join(HERE, "experiment1", "weights", "rotated_%d_frozen.pt" % h))["state_dict"]
        model = MetaSGD(init)
        model.load_state_dict(saved["state_dict"])
        out, test_set = evaluate(model, h, x_test, y_test, family)
        row = report_row(h, model, saved, out, test_set, x_test, y_test, family)
        log(json.dumps(row))
        rows.append(row)
    path = os.path.join(exp_dir, "results.csv")
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if int(r["hidden_size"]) not in args.sizes]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(old + rows, key=lambda r: int(r["hidden_size"])))
    sizes = sorted({int(r["hidden_size"]) for r in old + rows})
    plot(exp_dir, [h for h in sizes if os.path.exists(os.path.join(exp_dir, "meta_curve_h%d.csv" % h))])
    log("Saved %s" % path)


if __name__ == "__main__":
    main()
