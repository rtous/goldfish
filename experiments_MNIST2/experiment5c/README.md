# Experiment 5c: a general first layer plus context parts

**Question.** In experiment 5b, the shared head made the first layer much easier to adapt (the best
L1000 result so far). But the online starting point, the mean of the 8 first layers, scored only
55.80% / 44.41%, so every burst started badly and short bursts lost. What if the starting first
layer is trained explicitly to be good on every rotation?

**Pretraining** (`pretrain_general.py`, adapted from 5b's `pretrain_shared.py`). Context k's
first layer is S1 + D_k, with one shared head. Every batch (a random context per sampled image)
gives two losses on the same images: loss_ctx with first layer S1 + D_k (k = each image's
context), and loss_gen with S1 alone. Training minimizes loss_ctx + loss_gen. S1 and the head start
from `weights/mixed_h<h>.pt`, every D_k from 0. Adam lr 1e-3, batch 128, 10 epochs, seed 0, as in
5b. `weights/general_h<h>.pt` is S1 plus the head, a normal 784 -> h -> 10 model.
`weights/context_h<h>.pt` keeps the D_k, for reference only. At inference the model is the same
size as the mixed model.

| test set, %, mean over the 8 contexts | h = 8 | h = 4 |
|---|---|---|
| (a) ceiling: S1 + each context's own D_k | 89.86 | 84.27 |
| (a) in 5b: each context's own first layer | 90.39 | 85.52 |
| (b) starting point: `general_h<h>.pt` frozen | **73.70** | **64.83** |
| (b) in 5b: the mean of the first layers | 55.80 | 44.41 |
| mixed model | 74.05 | 64.78 |
| (c) mean \|D_k\| / \|S1\| (0.weight, Frobenius) | 0.56 | 0.62 |

**Online configurations** (`configurations.json`): the same three configurations as 5b, with
`"init"` = the general weights. Only 0.weight and 0.bias adapt; the head stays fixed.
`frozen_general` (dual, lr 0, fast_lr 0), `dual_general` (dual, lr 0, fast_lr 1e-2, decay 0.999,
hidden), and `metasgd_general` (S, alpha and lambda meta-learned on the first layer only; meta
settings as in 5b). No code changes were needed beyond 5b's.

Meta-training: both runs reached the 15 min cap, 6350 steps (h = 8) and 7200 steps (h = 4). The
first h = 8 run lost 15 min of wall-clock time to a system sleep and stopped after 4150 steps, so
h = 8 was rerun alone with the same code and settings (`meta_train.train` on this configuration).
The results use the rerun.

## Results

Gain over frozen / locality effect (bursts - shuffled), in pp:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| frozen mixed (%) | 73.25 | 64.41 | 74.05 | 64.78 |
| `frozen_general` | -0.76 / 0.00 | -0.91 / 0.00 | -0.35 / 0.00 | +0.06 / 0.00 |
| `dual_general` | -3.45 / +2.95 | -2.93 / +4.66 | +4.35 / +10.73 | +6.46 / +13.38 |
| `metasgd_general` | +2.27 / +3.20 | +2.47 / +4.79 | +3.09 / +3.72 | +3.84 / +4.69 |
| `dual_shared` (5b) | -5.57 / +5.24 | -4.57 / +7.73 | +4.91 / +14.91 | +7.82 / +18.75 |
| `metasgd_shared` (5b) | +1.25 / +4.06 | +1.29 / +7.11 | +2.92 / +5.53 | +3.53 / +8.16 |
| `dual_slow0_f1e-2_d0.999` (experiment 2) | -3.66 / +2.04 | -3.77 / +3.06 | +2.87 / +8.81 | +3.02 / +9.57 |
| `metasgd` (experiment 3) | +2.70 / +3.56 | +2.36 / +4.05 | +3.51 / +3.99 | +3.77 / +4.41 |

Accuracy at positions 1-10 of a burst (switching cost), %:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| `dual_general` | 65.68 | 55.45 | 61.50 | 51.12 |
| `dual_shared` (5b) | 59.77 | 50.45 | 56.00 | 47.38 |
| `dual_slow0_f1e-2_d0.999` (experiment 2) | 67.39 | 55.34 | 64.88 | 54.25 |
| `metasgd_general` | 71.82 | 63.64 | 76.00 | 67.12 |
| `metasgd_shared` (5b) | 68.75 | 60.91 | 73.50 | 63.12 |
| `metasgd` (experiment 3) | 72.61 | 63.18 | 75.38 | 65.62 |

Plots: `plots/summary.png`, `plots/burst_curve_short.png`, `plots/burst_curve_L1000.png`.

## Findings

1. **The starting point is fixed, and the ceiling stays close.** S1 alone scores 73.70% /
   64.83%, essentially the mixed model (-0.35 / +0.05 pp), against 55.80% / 44.41% in 5b. The
   ceiling drops only slightly (89.86 / 84.27 against 90.39 / 85.52). The context parts are large,
   though: on average D_k has 56-62% of S1's norm.
2. **`dual_general` keeps most of 5b's long-burst gain and loses less on short bursts.** On L1000
   it gains +4.35 / +6.46 pp, against +4.91 / +7.82 in 5b and +2.87 / +3.02 for experiment 2's
   `dual_slow0`. On short bursts it loses -3.45 / -2.93 pp, against -5.57 / -4.57 in 5b, and now
   slightly less than `dual_slow0` (-3.66 / -3.77). It still loses on short bursts.
3. **`metasgd_general` does not beat experiment 3's `metasgd`.** It is roughly level: -0.43 and
   -0.42 pp at h = 8, +0.11 and +0.07 pp at h = 4 (short, L1000). That is well within what one seed
   and one time-capped run can resolve. It clearly beats `metasgd_shared` from 5b (+0.2 to +1.2 pp).
4. **Switching cost: back to the level of the mixed-model runs.** At images 1-10, `dual_general`
   is 4-6 pp above `dual_shared` but still 0-3 pp below `dual_slow0` (most on L1000, where F has
   drifted furthest from S1 by the switch). `metasgd_general` is at or above experiment 3's
   `metasgd` (-0.8 pp on short at h = 8, +0.5 to +1.5 pp elsewhere).

Caveats: one seed, and one time-capped meta-training run per h. The settings were compared on the
evaluation streams.
