"""Slow + fast weights and Meta-SGD for tests 8-10.

The model predicts with S + F. F starts at 0. After scoring a chunk, with g the gradient of the
chunk loss (mean over its scored tokens) at S + F:
    S <- S - SLOW_LR * g          slow part: plain SGD on all parameters (test 7's full_1e-4)
    F <- lam * F - alpha * g      only on the tensors that have a fast part
With no fast part this is full_1e-4. Non-fast tensors use the same in-place op as torch.optim.SGD
(p.add_(g, alpha=-lr)), so the two match bit for bit (checked in test 8).

Streams: chunk sets are [books, chunks, L] arrays (test7_aggressive.chunk) and a stream is a list
of (book, chunk) in reading order. Validation streams come from data.val_books():
    val-pick           the first N_PICK books (K = 1, and the same chunks shuffled, seed 0)
    val-meta-train     the next 80% of the remaining books (K = 1)
    val-meta-heldout   the last 20% (K = 1)

Meta-SGD (tests 9-10): alpha = exp(a), lam = sigmoid(b), per tensor or per parameter. A
meta-step reads T chunks of the training stream, carrying S and F across episodes; the meta-loss
(mean prequential loss over the T chunks) is backpropagated to (a, b) through the T updates, with
the state at the episode start treated as a constant (truncated). exact=True keeps the
second-order terms (gradients of the updates through g); exact=False is the first-order
approximation (g treated as a constant). Exact meta-gradients need a twice-differentiable attention:
the CPU flash-attention kernel (sdpa) has no double backward, so meta-training switches the model to
eager attention (per-token losses differ from sdpa by float rounding, at most 1.4e-3 nats in a
check); all evaluation passes use the default kernel, like tests 1-7.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F_
from torch.func import functional_call

from data import val_books
from models import DEFAULT
from online import L
from test7_aggressive import chunk, orders

SLOW_LR = 1e-4
N_PICK = 10
TRAIN_SHARE = 0.8


def val_sets():
    """Chunk sets and K = 1 / shuffled orders for val-pick, val-meta-train and val-meta-heldout."""
    ids = val_books(DEFAULT)["ids"]
    n_rest = len(ids) - N_PICK
    n_train = round(TRAIN_SHARE * n_rest)
    sets = {"val_pick": chunk(ids[:N_PICK]), "val_train": chunk(ids[N_PICK:N_PICK + n_train]),
            "val_heldout": chunk(ids[N_PICK + n_train:])}
    pick = orders(N_PICK, sets["val_pick"].shape[1])
    streams = {"val_pick_K1": ("val_pick", pick["K1"]), "val_pick_shuffled": ("val_pick", pick["shuffled"])}
    for name in ("val_train", "val_heldout"):
        c = sets[name]
        streams[name + "_K1"] = (name, [(b, k) for b in range(c.shape[0]) for k in range(c.shape[1])])
    return sets, streams


def mlp_weights(layers):
    """Names of the MLP weight matrices (no biases) of the given 0-based layers."""
    return ["gpt_neox.layers.%d.mlp.%s.weight" % (i, w) for i in layers for w in ("dense_h_to_4h", "dense_4h_to_h")]


def all_weight_matrices(n_layers=6):
    """The 25 tensors of test 9: per layer attention qkv, attention out, MLP in, MLP out; plus the output
    embedding."""
    names = []
    for i in range(n_layers):
        names += ["gpt_neox.layers.%d.attention.query_key_value.weight" % i,
                  "gpt_neox.layers.%d.attention.dense.weight" % i]
        names += mlp_weights([i])
    return names + ["lm_head.weight"]


def dual_losses(model, init, chunks, order, fast=None, slow_lr=SLOW_LR):
    """Per-token losses [books, chunks, L] (NaN at each chunk's first token) for one prequential pass.
    fast: {parameter name: (alpha, lam)}, alpha and lam floats or tensors shaped like the parameter."""
    fast = fast or {}
    model.load_state_dict(init)
    params = dict(model.named_parameters())
    unknown = set(fast) - set(params)
    assert not unknown, unknown
    slow = {n: params[n].detach().clone() for n in fast}
    fastw = {n: torch.zeros_like(params[n]) for n in fast}
    for p in model.parameters():
        p.requires_grad_(True)
    out = np.full(chunks.shape, np.nan, dtype=np.float32)
    for b, c in order:
        x = torch.from_numpy(chunks[b, c]).unsqueeze(0)
        nll = F_.cross_entropy(model(input_ids=x).logits[0, :-1], x[0, 1:], reduction="none")
        out[b, c, 1:] = nll.detach().numpy()
        model.zero_grad(set_to_none=True)
        nll.mean().backward()
        with torch.no_grad():
            for n, p in params.items():
                if n in fast:
                    alpha, lam = fast[n]
                    slow[n].add_(p.grad, alpha=-slow_lr)
                    fastw[n].mul_(lam).sub_(alpha * p.grad)
                    p.copy_(slow[n] + fastw[n])
                else:
                    p.add_(p.grad, alpha=-slow_lr)
    return out


class MetaRule(torch.nn.Module):
    """alpha = exp(a), lam = sigmoid(b) for each fast tensor; per tensor (scalars) or per parameter."""

    def __init__(self, init_values, shapes=None):
        """init_values: {name: (alpha, lam)}; shapes: {name: shape} for per-parameter values."""
        super().__init__()
        self.names = list(init_values)
        self.a, self.b = torch.nn.ParameterList(), torch.nn.ParameterList()
        for n in self.names:
            alpha, lam = init_values[n]
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            lam = torch.as_tensor(lam, dtype=torch.float32)
            if shapes and n in shapes:
                alpha, lam = alpha.expand(shapes[n]).clone(), lam.expand(shapes[n]).clone()
            self.a.append(torch.nn.Parameter(alpha.log()))
            self.b.append(torch.nn.Parameter(torch.log(lam / (1 - lam))))

    def values(self):
        return {n: (a.exp(), torch.sigmoid(b)) for n, a, b in zip(self.names, self.a, self.b)}

    def detached(self):
        return {n: (al.detach(), lm.detach()) for n, (al, lm) in self.values().items()}


def meta_episode(model, rule, state, xs, exact=True, slow_lr=SLOW_LR):
    """One truncated episode over the chunks xs (list of LongTensor [L]). state = (S, F) dicts of detached
    tensors (S: every parameter; F: the rule's tensors). Returns the meta-loss (differentiable in the rule)
    and the new detached state."""
    S0, F0 = state
    S = {n: t.detach().requires_grad_(True) for n, t in S0.items()}
    Fw = {n: t.detach() for n, t in F0.items()}
    vals = rule.values()
    names = list(S)
    losses = []
    for x in xs:
        params = {n: (S[n] + Fw[n]) if n in Fw else S[n] for n in names}
        logits = functional_call(model, params, (x.unsqueeze(0),)).logits[0, :-1]
        loss = F_.cross_entropy(logits, x[1:])
        losses.append(loss)
        grads = torch.autograd.grad(loss, [params[n] for n in names], create_graph=exact, retain_graph=True)
        g = dict(zip(names, grads))
        S = {n: S[n] - slow_lr * g[n] for n in names}
        Fw = {n: vals[n][1] * Fw[n] - vals[n][0] * g[n] for n in Fw}
    meta_loss = torch.stack(losses).mean()
    new_state = ({n: t.detach() for n, t in S.items()}, {n: t.detach() for n, t in Fw.items()})
    return meta_loss, new_state


def fresh_state(model, init, rule):
    S = {n: init[n].detach().clone() for n, _ in model.named_parameters()}
    Fw = {n: torch.zeros_like(init[n]) for n in rule.names}
    return S, Fw


def nan_mean(x):
    return float(np.nanmean(x))


def half_life_chunks(lam):
    return math.log(0.5) / math.log(lam) if 0 < lam < 1 else float("inf")
