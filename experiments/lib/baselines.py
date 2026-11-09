"""Unigram cache baseline (no training).

p = (1 - lam) p_model + lam p_cache, with p_model the frozen model and p_cache the normalized,
exponentially decaying counts (half-life h tokens) of the token ids seen so far in the stream. The
counts are updated once per chunk, after scoring it (the online learner also learns only after
scoring). The first chunk has an empty cache and uses p_model alone. Defaults: lam 0.05, half-life
1k tokens (old test 3's pick).
"""
import numpy as np

from data import L

VOCAB = 50304
LAM, HALF_LIFE = 0.05, 1000


def unigram_losses(frozen, ids, lam=LAM, half_life=HALF_LIFE):
    """Per-token loss of the frozen model mixed with the unigram cache (NaN where frozen is NaN)."""
    decay = 0.5 ** (1 / half_life)
    counts = np.zeros(VOCAB)
    out = frozen.copy()
    for s in range(0, len(ids), L):
        e = min(s + L, len(ids))
        total = counts.sum()
        if total > 0:
            p_cache = counts[ids[s:e]] / total
            out[s:e] = -np.log((1 - lam) * np.exp(-frozen[s:e].astype(np.float64)) + lam * p_cache)
        counts *= decay ** (e - s)
        np.add.at(counts, ids[s:e], decay ** np.arange(e - s - 1, -1, -1))
    return out.astype(np.float32)
