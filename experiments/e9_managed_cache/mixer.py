"""Fixed-share mixture of two experts (Herbster & Warmuth 1998), token by token.

Experts: a = frozen + store, b = frozen. For each scored token t (NaN losses are skipped: chunk starts):
    p_mix(t) = w p_a(t) + (1 - w) p_b(t)                       (scored: -log p_mix)
    v = w p_a(t) / p_mix(t)                                     (Bayesian update by the observed token)
    w <- (1 - alpha) v + alpha (1 - v)                          (share: two experts, alpha moves to the other)
The weight used for token t depends only on earlier tokens (prequential). alpha = 0 is the exact Bayesian
mixture with prior (w0, 1 - w0).
"""
import math

import numpy as np


def fixed_share(loss_a, loss_b, alpha, w0=0.5, force=None):
    """(mixture loss, weight on expert a used for each token), float64, NaN where loss_a is NaN. Inputs are
    per-token losses in nats (any shape, flattened in order). force: a fixed weight on a (no updates)."""
    la = np.asarray(loss_a, dtype=np.float64).ravel().tolist()
    lb = np.asarray(loss_b, dtype=np.float64).ravel().tolist()
    out = [math.nan] * len(la)
    wt = [math.nan] * len(la)
    w = w0 if force is None else force
    exp, log = math.exp, math.log
    for t, a in enumerate(la):
        if a != a:
            continue
        pa, pb = exp(-a), exp(-lb[t])
        m = w * pa + (1 - w) * pb
        out[t], wt[t] = -log(m), w
        if force is None:
            v = w * pa / m
            w = (1 - alpha) * v + alpha * (1 - v)
    shape = np.shape(loss_a)
    return np.array(out).reshape(shape), np.array(wt).reshape(shape)


def check():
    """Synthetic checks; returns a list of (name, passed, detail)."""
    rng = np.random.default_rng(0)
    res = []
    # 1. alpha = 0 equals the exact Bayesian mixture: sum of mixture losses = -log(w0 prod p_a + (1-w0) prod p_b)
    la, lb = rng.uniform(0.1, 6, 2000), rng.uniform(0.1, 6, 2000)
    la[::256] = np.nan
    mix, _ = fixed_share(la, lb, 0.0, w0=0.3)
    ok = np.isfinite(la)
    sa, sb = -la[ok].sum(), -lb[ok].sum()
    top = max(sa, sb)
    exact = -(top + math.log(0.3 * math.exp(sa - top) + 0.7 * math.exp(sb - top)))
    d = abs(np.nansum(mix) - exact)
    res.append(("alpha=0 equals the Bayesian mixture", d < 1e-6 * abs(exact), "sum %.6f vs exact %.6f" % (np.nansum(mix), exact)))
    # 2. one expert always better: the weight converges to it
    la, lb = rng.uniform(1.0, 2.0, 3000), rng.uniform(1.0, 2.0, 3000) + 0.5
    _, w = fixed_share(la, lb, 0.01)
    res.append(("better expert gets the weight", w[-1] > 0.95 and w[200:].min() > 0.9,
                "final weight %.4f, min after 200 tokens %.4f" % (w[-1], w[200:].min())))
    # 3. switch: a better for 2000 tokens, then b better (by 0.5 nats/token); recovery time with alpha 0.01
    la = np.concatenate([rng.uniform(1, 2, 2000), rng.uniform(1, 2, 2000) + 0.5])
    lb = np.concatenate([rng.uniform(1, 2, 2000) + 0.5, rng.uniform(1, 2, 2000)])
    _, w = fixed_share(la, lb, 0.01)
    after = np.nonzero(w[2000:] < 0.5)[0]
    k = int(after[0]) + 1 if len(after) else None
    _, w0 = fixed_share(la, lb, 0.0)
    after0 = np.nonzero(w0[2000:] < 0.5)[0]
    k0 = int(after0[0]) + 1 if len(after0) else None
    res.append(("alpha>0 recovers after a switch", k is not None and k <= 20,
                "alpha 0.01: weight below 0.5 after %s tokens (alpha 0: %s)" % (k, k0 if k0 else "never within 2000")))
    # 4. forced weight 0 returns expert b exactly
    mix, _ = fixed_share(la, lb, 0.01, force=0.0)
    res.append(("forced weight 0 gives expert b", float(np.max(np.abs(mix - lb))) < 1e-12,
                "max abs diff %.1e" % float(np.max(np.abs(mix - lb)))))
    return res
