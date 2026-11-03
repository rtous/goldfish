# Experiment 5b: decoupled pretraining (shared head, one first layer per context)

**Question.** Experiment 5a found that the rotation knowledge sits in the first layer and the
output layer can stay fixed. If we pretrain with that split (one shared head and one first layer
per rotation), start online from the mean of the first layers, and adapt only the first layer
online, do we get a better starting point for online adaptation than the mixed model?

**Pretraining** (`pretrain_shared.py`). There is one shared output layer (2.*) and 8 first layers
(0.*). An image under context k goes through first layer k, then ReLU, then the shared head.
Every first layer and the head start from `weights/mixed_h<h>.pt`. Training is the same as
`offline.py` (a random context per sampled image, Adam lr 1e-3, batch 128, seed 0), for 10 epochs.
`weights/shared_h<h>.pt` is a normal 784 -> h -> 10 model: the mean of the 8 first layers plus the
head. `weights/per_context_h<h>.pt` keeps the 8 first layers, for reference only.

| | h = 8 | h = 4 |
|---|---|---|
| (a) each context with its own first layer (the ceiling) | 90.39 | 85.52 |
| (b) `shared_h<h>.pt` frozen (the online starting point) | 55.80 | 44.41 |
| mixed model, for reference | 74.05 | 64.78 |

(test set, %, mean over the 8 contexts)

**(b) is far below the mixed model: -18.3 pp (h = 8) and -20.4 pp (h = 4).** The mean of the
first layers is not a good model for any one rotation. It is worst at the extremes (40-45% at
contexts 0 and 7, h = 8). Nothing was tuned to fix this.

**Fairness.** At inference the model is the same size as the mixed model (one first layer plus the
head). The other 7 first layers exist only during pretraining.

**Online configurations.** Every configuration starts from `shared_h<h>.pt` (the new `"init"` key)
and adapts only 0.weight and 0.bias. The head is never updated. "frozen" and "gain" are relative to
the frozen mixed model, as in every experiment.

| config | method | settings |
|---|---|---|
| `frozen_shared` | dual | lr 0, fast_lr 0: the shared weights, frozen |
| `dual_shared` | dual | lr 0, fast_lr 1e-2, decay 0.999, hidden (experiment 2's `dual_slow0_f1e-2_d0.999` from the new start) |
| `metasgd_shared` | metasgd | meta-learned S, alpha, lambda on the first layer only (`"fast_layers": "hidden"`), S starting from the shared weights; other meta settings as in experiment 3 |

Both meta-training runs reached the 15 min cap (6250 steps at h = 8, 7200 at h = 4). The
meta-learned first layer recovers quickly from the poor start. Frozen, it scores about 72% (h = 8)
and 61% (h = 4) on the held-out episodes, against about 73.6% and 64.7% for experiment 3's S.

Shared code changes (old behavior by default): an optional `"init"` in `online.py`, and `"init"`
and `"fast_layers"` in `meta_train.py`'s meta block. With these changes, experiment 3's `metasgd`
(h = 8, short) reproduces its row exactly.

## Results

Gain over frozen / locality effect (bursts - shuffled), in pp:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| frozen mixed (%) | 73.25 | 64.41 | 74.05 | 64.78 |
| `frozen_shared` | -18.86 / 0.00 | -21.59 / 0.00 | -18.25 / 0.00 | -20.36 / 0.00 |
| `dual_shared` | -5.57 / +5.24 | -4.57 / +7.73 | **+4.91** / +14.91 | **+7.82** / +18.75 |
| `metasgd_shared` | +1.25 / +4.06 | +1.29 / +7.11 | +2.92 / +5.53 | +3.53 / +8.16 |
| `dual_slow0_f1e-2_d0.999` (experiment 2) | -3.66 / +2.04 | -3.77 / +3.06 | +2.87 / +8.81 | +3.02 / +9.57 |
| `metasgd` (experiment 3) | +2.70 / +3.56 | +2.36 / +4.05 | +3.51 / +3.99 | +3.77 / +4.41 |

Accuracy at positions 1-10 of a burst (switching cost), %:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| `dual_shared` | 59.77 | 50.45 | 56.00 | 47.38 |
| `dual_slow0_f1e-2_d0.999` (experiment 2) | 67.39 | 55.34 | 64.88 | 54.25 |
| `metasgd_shared` | 68.75 | 60.91 | 73.50 | 63.12 |
| `metasgd` (experiment 3) | 72.61 | 63.18 | 75.38 | 65.62 |

Plots: `plots/summary.png`, `plots/burst_curve_short.png`, `plots/burst_curve_L1000.png`.

## Findings

1. **Is it a better starting point? Only for long bursts, and only for dual.** On L1000,
   `dual_shared` beats `dual_slow0_f1e-2_d0.999` by +2.0 pp (h = 8) and +4.8 pp (h = 4). It is
   also the best result on L1000 of any experiment so far (+4.9 and +7.8 pp; the previous best was
   +4.1 and +6.1 pp, experiment 2). At images 201-1000 it reaches 81.4% (h = 8) and 75.2% (h = 4),
   against 78.5% and 69.3% for the old start. On short bursts it is worse (-5.6 and -4.6 pp, against
   -3.7 and -3.8). `metasgd_shared` is below experiment 3's `metasgd` in every cell (-0.2 to -1.5
   pp). Meta-training spends its effort pulling S back from the poor mean, and still ends below
   experiment 3's S.
2. **How much of the frozen-to-ceiling gap is recovered?** On L1000, `dual_shared` recovers 67%
   (h = 8) and 69% (h = 4) of the gap between (b) and (a) over the whole stream, and 74-75% by
   images 201-1000. That still leaves 9-10 pp below the ceiling. On short bursts it recovers 37-40%.
   `metasgd_shared` recovers 54-61%, but most of that comes from its meta-learned S, not from online
   adaptation.
3. **Is the switching cost better? No, it is worse.** At images 1-10, `dual_shared` is 5-9 pp below
   `dual_slow0`, and `metasgd_shared` is 2-4 pp below `metasgd`. The model starts every burst from
   a poor point: after a switch, F still holds the last rotation, and without it the model is the
   55.8% / 44.4% mean.
4. The split creates a model that is easy to specialize: its per-context first layers reach the
   ceiling (a). But it does not create a good shared default. Online adaptation pays for the poor
   mean at every switch, so the split helps only when bursts are long enough to amortize that.

Caveats: one seed, and one time-capped meta-training run per h. The settings were compared on the
evaluation streams.
