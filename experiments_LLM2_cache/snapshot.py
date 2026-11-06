"""Online learners that can be paused, and frozen scoring with a snapshot of their weights (tests 11-13).

A Learner reads a stream in consecutive segments (lengths multiples of L = 256) and keeps its state
between them; snapshot scoring then evaluates held-out text with the current weights, no updates.
    rule "sgd"   plain SGD (online.stream_losses with init = the current state; stateless, so a
                 segmented run equals one call over the whole stream). Any params subset.
    rule "dual"  dual.py's slow + fast rule: S <- S - slow_lr g on all parameters, F <- lam F - alpha g
                 on the fast tensors, predict with S + F. Same in-place ops as dual.dual_losses.
    rule "decay" online.stream_losses with decay set (test 7's fast4): the trained parameters are the
                 pretrained weights W plus a fast part F, F <- decay F - lr g. Same ops as stream_losses.
    rule "adam"  torch.optim.Adam (betas 0.9, 0.999, eps 1e-8, no weight decay) on all parameters; a fresh
                 optimizer per Learner (per stream), its state carried across segments.
    rule "nsgd"  normalized SGD per tensor: W <- W - lr * n(W) * g / (||g|| + 1e-12), n(W) = ||W|| for
                 matrices and ||W|| + 1e-3 for 1-D tensors (biases, layer norms); every tensor moves by
                 about the same relative amount lr per step. All parameters.
Every rule takes one update per chunk after scoring it (prequential), like online.py.

check() verifies, on one book, that segmented runs match single runs (sgd: stream_losses; dual:
dual_losses; decay: stream_losses with decay). Usage: python snapshot.py   (check, about 1 min)
"""
import numpy as np
import torch
import torch.nn.functional as F_

from online import L, stream_losses


def copy_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def score(model, state, ids):
    """Frozen per-token losses of ids with the weights state (batched, as online.stream_losses lr = 0)."""
    return stream_losses(model, state, ids, 0)


class Learner:
    def __init__(self, model, init, rule="sgd", lr=1e-4, params="all", fast=None, slow_lr=1e-4, decay=None):
        """fast (rule "dual"): {parameter name: (alpha, lam)}; decay (rule "decay"): the fast part's decay."""
        self.model, self.rule, self.lr, self.params = model, rule, lr, params
        self.state = {k: v.clone() for k, v in init.items()}
        if rule == "decay":
            self.decay, self.init = decay, {k: v.clone() for k, v in init.items()}
        elif rule == "adam":
            model.load_state_dict(self.state)
            self.opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        elif rule == "nsgd":
            pass
        elif rule == "dual":
            self.slow_lr, self.fast = slow_lr, fast
            self.slow = {n: init[n].clone() for n in fast}
            self.fastw = {n: torch.zeros_like(init[n]) for n in fast}
        elif rule != "sgd":
            raise ValueError(rule)

    def read(self, ids):
        """Prequential losses for the next segment of the stream; the state moves on."""
        assert len(ids) % L == 0
        if self.rule == "sgd":
            out = stream_losses(self.model, self.state, ids, self.lr, params=self.params)
        elif self.rule == "decay":
            out = self._read_decay(ids)
        elif self.rule in ("adam", "nsgd"):
            out = self._read_normalized(ids)
        else:
            out = self._read_dual(ids)
        self.state = copy_state(self.model)
        return out

    def _read_normalized(self, ids):
        """Adam or normalized SGD, all parameters (load_state_dict copies in place, so Adam's state stays
        attached to the same tensors across segments)."""
        model = self.model
        model.load_state_dict(self.state)
        params = list(model.parameters())
        for p in params:
            p.requires_grad_(True)
        x = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        out = np.full(len(x), np.nan, dtype=np.float32)
        for s in range(0, len(x), L):
            xb = x[s:s + L].unsqueeze(0)
            nll = F_.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
            out[s + 1:s + L] = nll.detach().numpy()
            for p in params:
                p.grad = None
            nll.mean().backward()
            if self.rule == "adam":
                self.opt.step()
            else:
                with torch.no_grad():
                    for p in params:
                        scale = p.norm() + (1e-3 if p.dim() == 1 else 0.0)
                        p.sub_(p.grad, alpha=float(self.lr * scale / (p.grad.norm() + 1e-12)))
        for p in params:
            p.grad = None
        return out

    def _read_decay(self, ids):
        from online import trained_params
        model = self.model
        model.load_state_dict(self.state)
        train = trained_params(model, self.params)
        ids_train = {id(p) for p in train}
        for p in model.parameters():
            p.requires_grad_(id(p) in ids_train)
        names = {id(p): k for k, p in model.named_parameters()}
        slow = [self.init[names[id(p)]] for p in train]
        x = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        out = np.full(len(x), np.nan, dtype=np.float32)
        for s in range(0, len(x), L):
            xb = x[s:s + L].unsqueeze(0)
            nll = F_.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
            out[s + 1:s + L] = nll.detach().numpy()
            for p in train:
                p.grad = None
            nll.mean().backward()
            with torch.no_grad():
                for p, w in zip(train, slow):
                    p.copy_(w + self.decay * (p - w) - self.lr * p.grad)
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None
        return out

    def _read_dual(self, ids):
        model = self.model
        model.load_state_dict(self.state)
        params = dict(model.named_parameters())
        for p in model.parameters():
            p.requires_grad_(True)
        x = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        out = np.full(len(x), np.nan, dtype=np.float32)
        for s in range(0, len(x), L):
            xb = x[s:s + L].unsqueeze(0)
            nll = F_.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
            out[s + 1:s + L] = nll.detach().numpy()
            model.zero_grad(set_to_none=True)
            nll.mean().backward()
            with torch.no_grad():
                for n, p in params.items():
                    if n in self.fast:
                        alpha, lam = self.fast[n]
                        self.slow[n].add_(p.grad, alpha=-self.slow_lr)
                        self.fastw[n].mul_(lam).sub_(alpha * p.grad)
                        p.copy_(self.slow[n] + self.fastw[n])
                    else:
                        p.add_(p.grad, alpha=-self.slow_lr)
        model.zero_grad(set_to_none=True)
        return out

    def score(self, ids):
        """Frozen losses of ids with the current weights; the learner's state is untouched."""
        return score(self.model, self.state, ids)


def check():
    from dual import dual_losses, mlp_weights
    from online import setup
    from test7_aggressive import chunk
    model, init, info, d = setup()
    ids = d["ids"][0][:78 * L]
    single = stream_losses(model, init, ids, 1e-4)
    lrn = Learner(model, init, "sgd", 1e-4)
    first = lrn.read(ids[:40 * L])
    _ = lrn.score(ids[:2 * L])
    seg = np.concatenate([first, lrn.read(ids[40 * L:41 * L]), lrn.read(ids[41 * L:])])
    print("sgd segmented vs single: max abs diff %.2e" % np.nanmax(np.abs(seg - single)))
    fast = {n: (1e-3, 0.99) for n in mlp_weights([1])}
    single = dual_losses(model, init, chunk(ids[None]), [(0, c) for c in range(78)], fast).reshape(-1)
    lrn = Learner(model, init, "dual", fast=fast)
    first = lrn.read(ids[:40 * L])
    _ = lrn.score(ids[:2 * L])  # scoring in between must not disturb the stream
    seg = np.concatenate([first, lrn.read(ids[40 * L:])])
    print("dual segmented vs single: max abs diff %.2e" % np.nanmax(np.abs(seg - single)))
    single = stream_losses(model, init, ids, 1e-2, params="mlp@3", decay=0.99)
    lrn = Learner(model, init, "decay", 1e-2, params="mlp@3", decay=0.99)
    first = lrn.read(ids[:40 * L])
    _ = lrn.score(ids[:2 * L])
    seg = np.concatenate([first, lrn.read(ids[40 * L:])])
    print("decay segmented vs single: max abs diff %.2e" % np.nanmax(np.abs(seg - single)))
    for rule, lr in (("adam", 1e-5), ("nsgd", 1e-4)):
        single = Learner(model, init, rule, lr).read(ids)
        lrn = Learner(model, init, rule, lr)
        first = lrn.read(ids[:40 * L])
        _ = lrn.score(ids[:2 * L])
        seg = np.concatenate([first, lrn.read(ids[40 * L:])])
        print("%s segmented vs single: max abs diff %.2e" % (rule, np.nanmax(np.abs(seg - single))))


if __name__ == "__main__":
    check()
