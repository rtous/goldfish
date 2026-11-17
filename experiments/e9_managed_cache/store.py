"""The fast store: one MLP layer learning online on top of the frozen model, with decay and flush.

The store is W = W_frozen + D on the parameters of one layer's MLP (trained_params "mlp@<layer>"), D = 0
at the start, every other parameter frozen. It is kept as the layer's own weights, updated in place, with
a frozen copy W_f, so D = W - W_f; the SGD step on D is the SGD step on W. With lam = 0 and no flush this
runs exactly the float operations of learner.stream_losses (same loss computation, same optimizer, same
requires_grad pattern), so it reproduces the registry run of that subset bit for bit.
Per chunk (prequential, as Learner): score the chunk with the current store, then
    1. one plain-SGD step on the mean loss of the chunk's scored tokens (the store's own loss);
    2. decay:  D <- (1 - lam) D, i.e. W <- W_f + (1 - lam) (W - W_f)              (lam = 0: skipped);
    3. flush:  d_k = mean store loss - mean frozen loss on chunk k; if the mean of d over the last `window`
       chunks since the last flush exceeds `threshold`, reset D = 0 (W <- W_f).
Snapshots every EVERY chunks and at the end (as runs.run): the current weights, frozen, score the held-out
text; the full system's snapshot mixes those losses with the pretrained model's by a fixed-share mixer
started fresh for each snapshot (the store is not updated while scoring).
"""
import numpy as np
import torch
import torch.nn.functional as F

from data import L
from learner import Learner, copy_state, score, trained_params
from mixer import fixed_share

EVERY = 40


def chunk_means(loss):
    return np.nanmean(loss.reshape(-1, L), axis=1)


def run_store(model, init, ids, lr, layer, lam=0.0, flush=None, frozen=None, held=None, held_frozen=None,
              alphas=()):
    """ids [n, L] in reading order. flush: (threshold, window) or None (needs frozen [n, L]). held: held-out ids
    (flat) for snapshots, held_frozen their pretrained losses. alphas: mixer shares for full-system snapshots.
    Returns {loss [n, L] float32, flushes, and with held: snaps, positions, snaps_mix {alpha: [...]}}."""
    model.load_state_dict(init)
    train = trained_params(model, "mlp@%d" % layer)
    ids_train = {id(p) for p in train}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids_train)
    opt = torch.optim.SGD(train, lr=lr, momentum=0, weight_decay=0)
    base = [p.detach().clone() for p in train]
    x = torch.from_numpy(np.asarray(ids, dtype=np.int64).reshape(-1))
    n = len(ids)
    out = np.full((n, L), np.nan, dtype=np.float32)
    flushes, hist = [], []
    res = {}
    if held is not None:
        res.update(snaps=[chunk_means(held_frozen)], positions=[0], snaps_mix={a: [chunk_means(held_frozen)] for a in alphas})
    for i in range(n):
        s = i * L
        xb = x[s:s + L].unsqueeze(0)
        nll = F.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
        out[i, 1:] = nll.detach().cpu().numpy()
        opt.zero_grad(set_to_none=True)
        nll.mean().backward()
        opt.step()
        if lam:
            with torch.no_grad():
                for p, b in zip(train, base):
                    p.mul_(1 - lam).add_(b, alpha=lam)
        if flush is not None:
            hist.append(float(np.nanmean(out[i]) - np.nanmean(frozen[i])))
            threshold, window = flush
            if len(hist) >= window and np.mean(hist[-window:]) > threshold:
                with torch.no_grad():
                    for p, b in zip(train, base):
                        p.copy_(b)
                flushes.append(i)
                hist = []
        if held is not None and ((i + 1) % EVERY == 0 or i == n - 1):
            h = score(model, copy_state(model), held)  # frozen copy of the current weights; model unchanged
            res["snaps"].append(chunk_means(h))
            res["positions"].append(i + 1)
            for a in alphas:
                res["snaps_mix"][a].append(chunk_means(fixed_share(h, held_frozen, a)[0]))
    for p in model.parameters():
        p.requires_grad_(True)
    res.update(loss=out, flushes=np.array(flushes, dtype=np.int64))
    return res


def run_all_params(model, init, ids, lr, held, held_frozen):
    """Configuration 2 (plain SGD on all parameters) with snapshots, as runs.run (Learner in segments)."""
    flat = np.asarray(ids).reshape(-1)
    lrn = Learner(model, init, "sgd", lr, "all")
    snaps, pos, losses = [chunk_means(held_frozen)], [0], []
    n = len(ids)
    for a in range(0, n, EVERY):
        e = min(a + EVERY, n)
        losses.append(lrn.read(flat[a * L:e * L]))
        snaps.append(chunk_means(lrn.score(held)))
        pos.append(e)
    return {"loss": np.concatenate(losses).reshape(n, L), "snaps": snaps, "positions": pos}
