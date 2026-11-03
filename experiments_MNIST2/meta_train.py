"""Meta-train every "metasgd" configuration of an experiment, for each size: the slow weights S
and a per-parameter fast learning rate alpha and decay lambda, for online.py's update rule with
lr = 0 and a fast part on every parameter:
    predict with S + F;  F <- lambda * F - alpha * g   (F starts at 0)
alpha = exp(log_alpha) > 0, lambda = sigmoid(logit_lambda) in (0, 1). With "learn_S": false, S
stays at the mixed model and only alpha and lambda are learned.

With "gate": true (surprise-gated forgetting), a scalar gate z per image wipes F when the loss
jumps. With m the running mean of past losses (m = loss_1 at the first image, so the first
surprise is 0):
    z <- sigmoid(b * (loss_t - m - c));  F <- lambda * (1 - z) * F - alpha * (1 + k * z) * g
    m <- beta * m + (1 - beta) * loss_t
b = softplus(raw_b), k = softplus(raw_k), beta = sigmoid(raw_beta) and c are meta-learned too,
from init_gate_b, init_gate_c, init_gate_k and init_beta.

Meta-training uses MNIST training images only (10000 held out for early stopping):
    episode     3 consecutive bursts of 20-60 random images each, consecutive contexts different
    meta-loss   mean prediction loss over the episode (each image before its update),
                backpropagated through the whole unroll; the MLP's per-image gradient is written
                in closed form, so the meta-gradient is exact (second order)
    settings    the configuration's "meta" block: lr (Adam), meta_batch, clip (gradient norm),
                init_alpha, init_lambda, learn_S, eval_every (steps), patience (evaluations
                without improvement), val_episodes, max_minutes; S starts at the mixed model, or
                at "init" (a weights path relative to experiments_MNIST2, <h> = the size);
                "fast_layers": "hidden" restricts alpha, lambda and the learned S to 0.weight and
                0.bias (the head stays fixed at its initial value, with no fast part)
    stopping    keep the best online accuracy on the held-out episodes

Writes <experiment>/weights/<config>_h<h>.pt and <experiment>/meta_curve_<config>_h<h>.csv.

Usage:
    python meta_train.py 3                  # every metasgd configuration, every size
    python meta_train.py 3 metasgd_fixedS   # only this one
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from contexts import N_CONTEXTS, transform
from data import experiment_dir, load_config, load_init, load_split

NAMES = ("0.weight", "0.bias", "2.weight", "2.bias")
FAST_LAYERS = {"hidden": ("0.weight", "0.bias"), "all": NAMES}
HOLDOUT = 10000


def forward(W, x):
    """Logits and hidden layer, batched over episodes: W[n] has a leading episode dimension."""
    pre = torch.einsum("ehi,ei->eh", W["0.weight"], x) + W["0.bias"]
    hid = pre.clamp(min=0)
    return torch.einsum("eoh,eh->eo", W["2.weight"], hid) + W["2.bias"], pre, hid


def grads(W, x, y, logits, pre, hid):
    """Closed-form gradient of each episode's cross-entropy on its image (differentiable)."""
    d_logits = logits.softmax(dim=1) - F.one_hot(y, 10).to(logits.dtype)
    d_pre = torch.einsum("eoh,eo->eh", W["2.weight"], d_logits) * (pre > 0).to(pre.dtype)
    return {"2.weight": d_logits.unsqueeze(2) * hid.unsqueeze(1), "2.bias": d_logits,
            "0.weight": d_pre.unsqueeze(2) * x.unsqueeze(1), "0.bias": d_pre}


class MetaSGD(torch.nn.Module):
    def __init__(self, S, alpha, lam, learn_S, gate=None, fast_names=NAMES):
        super().__init__()
        self.fast_names = fast_names
        self.gate = gate is not None
        if self.gate:  # b, c, k, beta of the surprise gate (scalars)
            inv_softplus = lambda v: math.log(math.expm1(v))
            self.raw_b = torch.nn.Parameter(torch.tensor(inv_softplus(gate["b"])))
            self.c = torch.nn.Parameter(torch.tensor(float(gate["c"])))
            self.raw_k = torch.nn.Parameter(torch.tensor(inv_softplus(gate["k"])))
            self.raw_beta = torch.nn.Parameter(torch.tensor(math.log(gate["beta"] / (1 - gate["beta"]))))
        key = lambda n: n.replace(".", "_")
        self.S = torch.nn.ParameterDict({key(n): torch.nn.Parameter(S[n].clone(), requires_grad=learn_S and n in fast_names)
                                         for n in NAMES})
        self.log_alpha = torch.nn.ParameterDict({key(n): torch.nn.Parameter(torch.full_like(S[n], math.log(alpha)))
                                                 for n in fast_names})
        self.logit_lambda = torch.nn.ParameterDict({key(n): torch.nn.Parameter(torch.full_like(S[n], math.log(lam / (1 - lam))))
                                                    for n in fast_names})

    def params(self):
        """S by parameter name; alpha and lambda by fast parameter name."""
        key = lambda n: n.replace(".", "_")
        return ({n: self.S[key(n)] for n in NAMES}, {n: self.log_alpha[key(n)].exp() for n in self.fast_names},
                {n: torch.sigmoid(self.logit_lambda[key(n)]) for n in self.fast_names})

    def gate_params(self):
        """b, c, k and beta of the surprise gate."""
        return F.softplus(self.raw_b), self.c, F.softplus(self.raw_k), torch.sigmoid(self.raw_beta)

    def unroll(self, x, y, mask, alpha_scale=1.0):
        """The online rule over E episodes (x [E, T, 784], y and mask [E, T]; padded positions make
        no update). Returns per-image losses and correctness, before each image's update."""
        S, alpha, lam = self.params()
        E, T = y.shape
        fast = {n: torch.zeros((E,) + S[n].shape) for n in self.fast_names}
        losses, correct = [], []
        if self.gate:
            b, c, k, beta = self.gate_params()
            m = None
        for t in range(T):
            W = {n: S[n] + fast[n] if n in fast else S[n].expand((E,) + S[n].shape) for n in NAMES}
            logits, pre, hid = forward(W, x[:, t])
            losses.append(F.cross_entropy(logits, y[:, t], reduction="none"))
            correct.append(logits.argmax(1) == y[:, t])
            g = grads(W, x[:, t], y[:, t], logits, pre, hid)
            keep = mask[:, t].float()
            if self.gate:
                loss_t = losses[-1]
                m = loss_t if m is None else m
                z = torch.sigmoid(b * (loss_t - m - c))  # [E]
                for n in self.fast_names:
                    shape = (E,) + (1,) * S[n].dim()
                    step = alpha[n] * alpha_scale * (1 + k * z.view(shape)) * g[n] * keep.view(shape)
                    fast[n] = lam[n] * (1 - z.view(shape)) * fast[n] - step
                m = beta * m + (1 - beta) * loss_t
                continue
            for n in self.fast_names:
                step = alpha[n] * alpha_scale * g[n] * keep.view((E,) + (1,) * S[n].dim())
                fast[n] = lam[n] * fast[n] - step
        return torch.stack(losses, 1), torch.stack(correct, 1)


def episodes(rng, x_raw, y, pool, count):
    """count episodes of 3 bursts of 20-60 images from pool, padded to the longest (mask)."""
    items = []
    for _ in range(count):
        ctx = [int(rng.integers(N_CONTEXTS))]
        while len(ctx) < 3:
            ctx.append((ctx[-1] + 1 + int(rng.integers(N_CONTEXTS - 1))) % N_CONTEXTS)
        sizes = rng.integers(20, 61, size=3)
        items.append((rng.choice(pool, size=int(sizes.sum()), replace=False), np.repeat(ctx, sizes)))
    T = max(len(idx) for idx, _ in items)
    xs, ys = torch.zeros(count, T, 784), torch.zeros(count, T, dtype=torch.long)
    mask = torch.zeros(count, T, dtype=torch.bool)
    for e, (idx, ctx) in enumerate(items):
        n = len(idx)
        xs[e, :n] = transform(x_raw[torch.from_numpy(idx)], torch.from_numpy(ctx))
        ys[e, :n], mask[e, :n] = y[torch.from_numpy(idx)], True
    return xs, ys, mask


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def train(cfg, h, exp_dir, seed, x_train, y_train):
    m = cfg["meta"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(x_train))
    holdout, pool = np.sort(order[:HOLDOUT]), np.sort(order[HOLDOUT:])
    gate = {"b": m["init_gate_b"], "c": m["init_gate_c"], "k": m["init_gate_k"], "beta": m["init_beta"]} \
        if m.get("gate") else None
    model = MetaSGD(load_init(m.get("init"), h), m["init_alpha"], m["init_lambda"], m["learn_S"], gate,
                    FAST_LAYERS[m.get("fast_layers", "all")])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=m["lr"])
    val = episodes(np.random.default_rng(seed + 1), x_train, y_train, holdout, m["val_episodes"])

    def validate():
        with torch.no_grad():
            return [masked_mean(model.unroll(*val, alpha_scale=a)[1].float(), val[2]).item() * 100 for a in (1.0, 0.0)]

    name = "%s_h%d" % (cfg["name"], h)
    curve, best, bad, step, start = [], -1.0, 0, 0, time.time()
    online, frozen = validate()
    curve.append([0, "", "%.2f" % online, "%.2f" % frozen, "0.0"])
    print("%s step 0: val online %.2f%%, S frozen %.2f%%" % (name, online, frozen), flush=True)
    while True:
        x, y, mask = episodes(rng, x_train, y_train, pool, m["meta_batch"])
        loss = masked_mean(model.unroll(x, y, mask)[0], mask)
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
            if bad >= m["patience"] or minutes >= m["max_minutes"]:
                break
    model.load_state_dict(best_state)
    S, alpha, lam = model.params()
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    detach = lambda d: {n: v.detach().clone() for n, v in d.items()}
    checkpoint = {"S": detach(S), "alpha": detach(alpha), "lambda": detach(lam), "steps": step,
                  "minutes": (time.time() - start) / 60, "best_val_online": best}
    if model.gate:
        checkpoint["gate"] = dict(zip(("b", "c", "k", "beta"), (v.item() for v in model.gate_params())))
        print("%s gate: %s" % (name, checkpoint["gate"]), flush=True)
    torch.save(checkpoint, os.path.join(exp_dir, "weights", name + ".pt"))
    with open(os.path.join(exp_dir, "meta_curve_%s.csv" % name), "w", newline="") as f:
        csv.writer(f).writerows([["step", "meta_loss", "val_online", "val_s_frozen", "minutes"]] + curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("configurations", nargs="*",
                        help="names to train (default: every metasgd one without checkpoint_from)")
    args = parser.parse_args()
    exp_dir = experiment_dir(args.experiment)
    config = load_config(exp_dir)
    configs = [c for c in config["configurations"] if c["method"] == "metasgd" and "checkpoint_from" not in c
               and (not args.configurations or c["name"] in args.configurations)]
    if not configs:
        raise SystemExit("No metasgd configuration to train.")
    x_train, y_train = load_split("train")
    for cfg in configs:
        for h in config["sizes"]:
            train(cfg, h, exp_dir, config.get("seed", 0), x_train, y_train)


if __name__ == "__main__":
    main()
