"""Meta-train every "metasgd" configuration of an experiment, for each architecture: the slow
weights S and a per-parameter fast learning rate alpha and decay lambda, for online.py's rule with
lr = 0 and a fast part on every parameter (as in MNIST2 experiment 3):
    predict with S + F;  F <- lambda * F - alpha * g   (F starts at 0)
alpha = exp(log_alpha) > 0, lambda = sigmoid(logit_lambda) in (0, 1). S starts at the mixed model.

Episodes come from the val drawers (11-12) only, never the test drawers. The mixed model never
trained on them, so the fast weights get a real learning signal. 20% of the classes (random,
seed) are held out for the validation episodes that pick the best checkpoint.
    episode     `bursts` consecutive bursts, each of one alphabet (consecutive alphabets
                different): up to `max_chars` random characters of that alphabet, both val
                drawings of each, shuffled (so each character is seen twice: occurrence 0 and 1)
    meta-loss   mean prediction loss over the episode (each image before its update),
                backpropagated through the whole unroll; the per-image gradient of the MLP is
                written in closed form, so the meta-gradient is exact (second order)
    settings    the "meta" block: lr (Adam), meta_batch, clip, init_alpha, init_lambda, bursts,
                max_chars, eval_every, patience, val_episodes, max_minutes
    stopping    keep the best online accuracy on the held-out episodes; stop after `patience`
                evaluations without improvement or at max_minutes (recorded as hit_cap)

Writes <experiment>/weights/<config>_<arch>.pt and <experiment>/meta_curve_<config>_<arch>.csv.

Usage:
    python meta_train.py 1
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import ALPHABET, N_ALPHABETS, experiment_dir, load_config, load_mixed, load_split
from models.mlp import N_CLASSES, param_names

HOLDOUT_FRACTION = 0.2


def forward(W, x, L):
    """Logits and the intermediate values, batched over episodes (W[n] has a leading E dim)."""
    acts, pres = [x], []
    a = x
    for i in range(L):
        pre = torch.einsum("eoi,ei->eo", W["%d.weight" % (2 * i)], a) + W["%d.bias" % (2 * i)]
        if i < L - 1:
            pres.append(pre)
            a = pre.clamp(min=0)
            acts.append(a)
    return pre, acts, pres


def grads(W, y, logits, acts, pres, L):
    """Closed-form gradient of each episode's cross-entropy on its image (differentiable)."""
    d = logits.softmax(dim=1) - F.one_hot(y, N_CLASSES).to(logits.dtype)
    g = {}
    for i in reversed(range(L)):
        g["%d.weight" % (2 * i)] = d.unsqueeze(2) * acts[i].unsqueeze(1)
        g["%d.bias" % (2 * i)] = d
        if i > 0:
            d = torch.einsum("eoi,eo->ei", W["%d.weight" % (2 * i)], d) * (pres[i - 1] > 0).to(d.dtype)
    return g


class MetaSGD(torch.nn.Module):
    def __init__(self, S, alpha, lam, learn_S, names):
        super().__init__()
        self.names, self.L = names, len(names) // 2
        key = lambda n: n.replace(".", "_")
        self.S = torch.nn.ParameterDict({key(n): torch.nn.Parameter(S[n].clone(), requires_grad=learn_S) for n in names})
        self.log_alpha = torch.nn.ParameterDict({key(n): torch.nn.Parameter(torch.full_like(S[n], math.log(alpha)))
                                                 for n in names})
        self.logit_lambda = torch.nn.ParameterDict({key(n): torch.nn.Parameter(torch.full_like(S[n], math.log(lam / (1 - lam))))
                                                    for n in names})

    def params(self):
        key = lambda n: n.replace(".", "_")
        return ({n: self.S[key(n)] for n in self.names}, {n: self.log_alpha[key(n)].exp() for n in self.names},
                {n: torch.sigmoid(self.logit_lambda[key(n)]) for n in self.names})

    def unroll(self, x, y, mask, alpha_scale=1.0):
        """The online rule over E episodes (x [E, T, 784], y and mask [E, T]; padded positions make
        no update). Returns per-image losses and correctness, before each image's update."""
        S, alpha, lam = self.params()
        E, T = y.shape
        fast = {n: torch.zeros((E,) + S[n].shape) for n in self.names}
        losses, correct = [], []
        for t in range(T):
            W = {n: S[n] + fast[n] for n in self.names}
            logits, acts, pres = forward(W, x[:, t], self.L)
            losses.append(F.cross_entropy(logits, y[:, t], reduction="none"))
            correct.append(logits.argmax(1) == y[:, t])
            g = grads(W, y[:, t], logits, acts, pres, self.L)
            keep = mask[:, t].float()
            for n in self.names:
                fast[n] = lam[n] * fast[n] - alpha[n] * alpha_scale * g[n] * keep.view((E,) + (1,) * S[n].dim())
        return torch.stack(losses, 1), torch.stack(correct, 1)


def episodes(rng, x, classes, count, n_bursts, max_chars):
    """count episodes of n_bursts alphabet bursts from the given classes, padded (mask)."""
    by_alpha = {a: classes[ALPHABET[classes] == a] for a in range(N_ALPHABETS)}
    alphas = [a for a in by_alpha if len(by_alpha[a])]
    items = []
    for _ in range(count):
        seq, prev = [], None
        for _ in range(n_bursts):
            a = alphas[int(rng.integers(len(alphas)))]
            while a == prev:
                a = alphas[int(rng.integers(len(alphas)))]
            prev = a
            chars = rng.permutation(by_alpha[a])[:max_chars]
            seq.append(rng.permutation(np.concatenate([2 * chars, 2 * chars + 1])))  # both val drawings
        items.append(np.concatenate(seq))
    T = max(len(i) for i in items)
    xs, ys = torch.zeros(count, T, 784), torch.zeros(count, T, dtype=torch.long)
    mask = torch.zeros(count, T, dtype=torch.bool)
    for e, idx in enumerate(items):
        n = len(idx)
        xs[e, :n] = x[torch.from_numpy(idx)]
        ys[e, :n], mask[e, :n] = torch.from_numpy(idx // 2), True
    return xs, ys, mask


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def train(cfg, arch, exp_dir, seed, x):
    m = cfg["meta"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(N_CLASSES)
    n_hold = int(HOLDOUT_FRACTION * N_CLASSES)
    holdout, pool = np.sort(order[:n_hold]), np.sort(order[n_hold:])
    model = MetaSGD(load_mixed(arch), m["init_alpha"], m["init_lambda"], m["learn_S"], param_names(arch))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=m["lr"])
    val = episodes(np.random.default_rng(seed + 1), x, holdout, m["val_episodes"], m["bursts"], m["max_chars"])

    def validate():
        with torch.no_grad():
            return [masked_mean(model.unroll(*val, alpha_scale=a)[1].float(), val[2]).item() * 100 for a in (1.0, 0.0)]

    name = "%s_%s" % (cfg["name"], arch)
    curve, best, bad, step, start, hit_cap = [], -1.0, 0, 0, time.time(), False
    online, frozen = validate()
    best = online  # the starting point (step 0) is a candidate too
    curve.append([0, "", "%.2f" % online, "%.2f" % frozen, "0.0"])
    print("%s step 0: val online %.2f%%, S frozen %.2f%%" % (name, online, frozen), flush=True)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    while True:
        xb, yb, mask = episodes(rng, x, pool, m["meta_batch"], m["bursts"], m["max_chars"])
        loss = masked_mean(model.unroll(xb, yb, mask)[0], mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), m["clip"])
        opt.step()
        step += 1
        if step % m["eval_every"] == 0:
            online, frozen = validate()
            minutes = (time.time() - start) / 60
            curve.append([step, "%.4f" % loss.item(), "%.2f" % online, "%.2f" % frozen, "%.1f" % minutes])
            print("%s step %d (%.1f min): meta-loss %.4f, val online %.2f%%, S frozen %.2f%%"
                  % (name, step, minutes, loss.item(), online, frozen), flush=True)
            if online > best:
                best, bad = online, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= m["patience"]:
                break
            if minutes >= m["max_minutes"]:
                hit_cap = True
                break
    model.load_state_dict(best_state)
    S, alpha, lam = model.params()
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    detach = lambda d: {n: v.detach().clone() for n, v in d.items()}
    torch.save({"S": detach(S), "alpha": detach(alpha), "lambda": detach(lam), "steps": step,
                "minutes": (time.time() - start) / 60, "best_val_online": best, "hit_cap": hit_cap},
               os.path.join(exp_dir, "weights", name + ".pt"))
    print("%s done: %d steps, best val online %.2f%%, hit the time cap: %s" % (name, step, best, hit_cap), flush=True)
    with open(os.path.join(exp_dir, "meta_curve_%s.csv" % name), "w", newline="") as f:
        csv.writer(f).writerows([["step", "meta_loss", "val_online", "val_s_frozen", "minutes"]] + curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("configurations", nargs="*")
    parser.add_argument("--archs", nargs="*")
    args = parser.parse_args()
    exp_dir = experiment_dir(args.experiment)
    config = load_config(exp_dir)
    configs = [c for c in config["configurations"] if c["method"] == "metasgd"
               and (not args.configurations or c["name"] in args.configurations)]
    if not configs:
        raise SystemExit("No metasgd configuration to train.")
    x, _ = load_split("val")
    for cfg in configs:
        for arch in args.archs or config["archs"]:
            train(cfg, arch, exp_dir, config.get("seed", 0), x)


if __name__ == "__main__":
    main()
