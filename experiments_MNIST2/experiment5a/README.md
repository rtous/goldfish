# Experiment 5a: which part of the network holds the context knowledge?

**Question.** The mixed model scores 74.05% (h = 8) and 64.78% (h = 4) averaged over the 8
rotations. Per-context models trained from scratch score 93.26% and 87.31%. How much of that gap
can we recover if we start from the mixed model and, for one rotation at a time, let only one part
of the network change? This is an offline diagnostic, not an online method.

**Setup** (`subspaces.py`). For each h, context and subset: start from `weights/mixed_h<h>.pt`,
fine-tune only the subset on the whole training set under that one context (Adam, lr 1e-3, batch
128, 5 epochs, seed 0), score on the test set under the same context, then average over the 8
contexts. Layer names follow `models/mlp.py` (0 = first layer, 2 = output layer).

| subset | trainable |
|---|---|
| `none` | nothing (the mixed model) |
| `out_bias` | 2.bias |
| `out_layer` | 2.weight, 2.bias |
| `hidden_bias` | 0.bias |
| `first_rank1` | W1 = S1 + u v^T, u (h) and v (784) trainable; u = 0, v ~ N(0, 0.01^2) at start |
| `first_layer` | 0.weight, 0.bias |
| `all` | every parameter |

`recovered` = (acc - mixed) / (per_context - mixed). `results.csv` also has the accuracy per
context. The run takes about a minute on the CPU.

## Results

| subset | h = 8: params | acc (%) | recovered | h = 4: params | acc (%) | recovered |
|---|---|---|---|---|---|---|
| `none` | 0 | 74.05 | 0.00 | 0 | 64.78 | 0.00 |
| `out_bias` | 10 | 74.56 | 0.03 | 10 | 65.26 | 0.02 |
| `hidden_bias` | 8 | 74.51 | 0.02 | 4 | 64.95 | 0.01 |
| `out_layer` | 90 | 78.37 | 0.22 | 50 | 68.41 | 0.16 |
| `first_rank1` | 792 | 78.92 | 0.25 | 788 | 71.38 | 0.29 |
| `first_layer` | 6280 | 90.06 | 0.83 | 3140 | 81.23 | 0.73 |
| `all` | 6370 | 91.35 | 0.90 | 3190 | 86.04 | 0.94 |

Checks: `none` reproduces the mixed model exactly. `all` lands 1.9 pp (h = 8) and 1.3 pp (h = 4)
below the per-context models, so 5 epochs were kept for every subset.

## Findings

- **The context knowledge sits in the first layer's weights.** `first_layer` is the smallest
  subset that recovers most of the gap: 83% (h = 8) and 73% (h = 4). Nothing smaller comes close.
  With it, every context lands near 90% (h = 8) or 81% (h = 4). Subsets that keep the first
  layer fixed keep the mixed model's profile: contexts 0 and 7 stay the worst.
- **Biases alone recover almost nothing** (at most 3%). A per-context shift, of either the hidden
  units or the logits, does not undo a rotation.
- **A rank-1 change to the first layer recovers only 25-29%**, about a third of the full first
  layer, for 13% (h = 8) or 25% (h = 4) of its parameters. At h = 8 the output layer (90
  parameters) recovers nearly as much (22%), so rank 1 is not an efficient summary of a rotation.
- At h = 4, training the output layer together with the first layer adds a further 21% of the gap
  (`all` vs `first_layer`). At h = 8 it adds only 7%: with 4 features, the readout must also
  adapt.
