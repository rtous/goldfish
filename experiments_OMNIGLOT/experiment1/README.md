# Experiment 1: online learning vs label-locality controls on alphabet bursts

**Question.** On the `alphabets` stream (one burst per alphabet), does any online method beat the
controls that only exploit which labels are likely (the label prior 2a and the output-bias sgd 2c)
and the cache (3)? How close does it get to the alphabet mask (2b) and to the per-alphabet ceiling?
A gain that only matches 2a or 2c is label locality, not the hypothesis. A gain above 2b would mean
the model is learning about the images, not just about which labels are likely.

**Answer: no.** No online method beats the label prior, on either architecture: the best one
falls 2.7 pp (A) and 8.5 pp (B) short of it (prior: +10.7 / +15.6 pp over frozen). Every
online method's gain comes from classes already seen in the burst, while first sightings get
*worse* than frozen. That is the signature of learning which labels are likely. Everything stays
17-26 pp below the alphabet mask, and the mask is itself above the per-alphabet ceiling. On this
data the hypothesis gets no support: the context information an online learner picks up is the
label distribution, and a 1623-number decaying count captures it better.

## Setup

Architectures A (784 -> 64 -> 64 -> 1623) and B (784 -> 62 -> 62 -> 62 -> 1623), about 160k
parameters each (625 KB), mixed models from `offline.py` (53.19 / 52.13%). One stream, one seed.
Every method predicts each image before it may learn from it. See `../README.md` for the data,
splits and models.

| config | what it is | tuned on |
|---|---|---|
| `frozen` | the mixed model, no updates (control 1) | |
| `prior` | logits + beta log(count + eps), count a decaying count of the labels seen (control 2a): decay 0.98, beta 0.5 (A) / 1 (B), eps 0.01 | val stream |
| `mask` | logits restricted to the true alphabet's classes; an oracle (control 2b) | |
| `bias_sgd_<lr>` | online sgd on the output bias only (control 2c), lr 0.1 to 3 | **test stream (optimistic)** |
| `cache_M<M>` | p = 0.75 p_model + 0.25 p_kNN over the last M items (cosine on the last hidden layer, temperature 0.03) (control 3) | val stream |
| `sgd_<lr>` | online sgd on every parameter, lr 1e-4 to 3e-3 | **test stream (optimistic)** |
| `dual_f<fast_lr>_d<decay>` | S + F, a fast part on every layer, slow lr 1e-5 | **test stream (optimistic)** |
| `metasgd` | S, per-parameter alpha and lambda meta-learned (as in MNIST2 experiment 3) | val drawers |
| `metasgd_fixedS` | S fixed at the mixed model; only alpha and lambda meta-learned | val drawers |

The sgd, dual and bias-sgd grids were first set too high (sgd up to 3e-2, fast rates up to 3e-2:
collapse to 17-37%), then moved down after a look at architecture A on the test stream, so all
three are chosen on the evaluation stream. That makes them optimistic, which only strengthens the
negative answer.

**Meta-SGD** (`meta_train.py`): episodes of 2 alphabet bursts, each up to 25 random characters of
one alphabet with both val drawings (so every character is seen twice), meta-batch 8, from the val
drawers of 80% of the classes. The other 20% give the validation episodes that pick the
checkpoint. With S learned, the meta-learner memorizes the 2 drawings per class through S:
held-out online accuracy fell from 60.6% to 29% (A) within 300 steps. Early stopping kept the
starting point (the mixed model with alpha = 1e-3, lambda = 0.99, i.e. not meta-learned at all),
which is why `metasgd_fixedS` was added. It stopped at the 15-minute cap on both architectures
(1125 steps for A, 1825 for B), with held-out online accuracy +0.8 pp (A) and +3.2 pp (B) over its
start. `metasgd` stopped by patience (500 steps), not at the cap.

## Results

`results.csv` (one row per config x arch): `frozen`, `bursts` and `shuffled` accuracy (%); `gain`
over frozen, `gain_vs_prior` over 2a, `gain_vs_mask` over 2b, `locality` = bursts - shuffled (pp);
accuracy by position in the burst (`pos1_10` ... `pos201_`) and by occurrence (`occ0` = first
sighting of the class in the burst, `occ1`, `occ2_3`, `occ4_7`); `model_kb` and `extra_kb` (online
state: F; F, alpha and lambda; the label counts; or the cached features and labels).

Best config of each method (all rows in `results.csv`; `show_results.py 1` prints them):

| arch | config | bursts | shuffled | gain | vs prior | vs mask | occ0 | occ1 | occ4_7 | KB |
|---|---|---|---|---|---|---|---|---|---|---|
| A | `frozen` | 53.19 | 53.19 | 0.00 | -10.73 | -25.65 | 53.05 | 54.78 | 52.91 | 625 |
| A | `mask` (2b, oracle) | 78.84 | 78.84 | +25.65 | +14.92 | 0.00 | 78.31 | 80.10 | 78.50 | 625 |
| A | `prior` (2a) | **63.92** | 52.45 | +10.74 | 0.00 | -14.92 | 49.29 | 66.24 | 65.93 | 631 |
| A | `metasgd_fixedS` | 61.23 | 48.97 | +8.04 | -2.69 | -17.61 | 46.58 | 60.01 | 64.25 | 2498 |
| A | `dual_f3e-4_d0.99` | 60.20 | 53.13 | +7.02 | -3.72 | -18.64 | 51.02 | 58.41 | 62.65 | 1249 |
| A | `bias_sgd_3` (2c) | 59.99 | 48.07 | +6.80 | -3.93 | -18.85 | 33.52 | 49.35 | 68.64 | 625 |
| A | `metasgd` | 59.03 | 45.95 | +5.84 | -4.89 | -19.81 | 43.68 | 58.84 | 61.58 | 2498 |
| A | `cache_M100` (3) | 56.66 | 52.94 | +3.47 | -7.26 | -22.18 | 52.56 | 57.79 | 57.07 | 650 |
| A | `sgd_3e-4` | 55.93 | 48.66 | +2.74 | -7.99 | -22.91 | 38.39 | 48.92 | 62.72 | 625 |
| B | `frozen` | 52.13 | 52.13 | 0.00 | -15.64 | -26.41 | 49.85 | 52.25 | 52.25 | 620 |
| B | `mask` (2b, oracle) | 78.54 | 78.54 | +26.40 | +10.77 | 0.00 | 77.76 | 78.93 | 78.56 | 620 |
| B | `prior` (2a) | **67.77** | 49.36 | +15.63 | 0.00 | -10.77 | 42.33 | 70.06 | 71.50 | 626 |
| B | `bias_sgd_3` (2c) | 59.32 | 46.93 | +7.19 | -8.45 | -19.22 | 31.92 | 47.69 | 67.87 | 620 |
| B | `dual_f3e-4_d0.99` | 58.44 | 48.71 | +6.31 | -9.33 | -20.10 | 49.35 | 56.25 | 60.74 | 1240 |
| B | `cache_M100` (3) | 55.74 | 52.01 | +3.60 | -12.03 | -22.80 | 49.48 | 55.70 | 56.56 | 645 |
| B | `sgd_1e-4` | 54.43 | 50.05 | +2.30 | -13.34 | -24.11 | 45.41 | 49.91 | 58.30 | 620 |
| B | `metasgd_fixedS` | 52.76 | 38.62 | +0.63 | -15.01 | -25.78 | 36.66 | 51.69 | 55.88 | 2480 |
| B | `metasgd` | 49.92 | 34.63 | -2.22 | -17.85 | -28.62 | 33.09 | 47.94 | 53.28 | 2480 |

Offline references: per-alphabet ceiling 73.87% (A) and 72.91% (B); mixed masked to the alphabet
78.84 / 78.54% (the same as `mask`).

Plots: `plots/burst_curve.png` (accuracy by position in the burst) and `plots/occurrence.png`
(accuracy by occurrence), best config of each method.

## Findings

1. **Nothing beats the label prior.** The best online method is `metasgd_fixedS` on A (+8.0 pp,
   2.7 pp below the prior); on B, `bias_sgd_3`, which only learns the output bias, is the best
   online method (+7.2 pp, 8.5 below the prior). A fixed 1623-number decaying label count with two
   parameters tuned on the val stream beats every learner, at no extra memory beyond 6 KB.
2. **The gains come from labels already seen, and first sightings get worse.** Every gradient
   method is below frozen at occurrence 0 (A: from 33.5% for `bias_sgd_3` to 51.0% for `dual`,
   vs 53.1% frozen), and gains from occurrence 1 on. So there is no generic tuning to the alphabet
   (that would also help first sightings); the methods shift probability toward recent labels,
   exactly what the prior does explicitly. The prior pays the same first-sighting cost (49.3 /
   42.3%).
3. **No sign of learning image details beyond labels.** If the model learned the current
   characters' details, its accuracy on repeat sightings should beat a control that knows only the
   labels. It doesn't: at occurrences 4-7, the output-bias-only control (68.6 / 67.9%) and the prior
   (65.9 / 71.5%) are at or above every full online learner (at most 64.3 / 60.7%).
4. **The cache adds +3.5 pp** (M = 100; M = 500 is slightly worse), also below the prior, but it is
   the one method that barely changes first sightings (52.6 vs 53.1% for A, 49.5 vs 49.9% for B).
5. **The context gain available here is label locality.** The alphabet mask alone gains +25.7 /
   +26.4 pp, and per-alphabet models trained only on their alphabet (73.9 / 72.9%) are 5 pp below
   it. With 10 drawings per character, knowing the label set is worth more than everything an
   alphabet-specific model learns from its own images.
6. **Locality has a cost.** Most online methods score below frozen on the shuffled control (e.g.
   `metasgd_fixedS` A: 48.97 vs 53.19%): they adapt to whatever passed recently, which hurts when
   there is no locality to exploit.
7. **Meta-SGD barely helps and costs 4x the memory.** Learning S on the val drawers overfits
   (2 drawings per class) and is rejected by early stopping; with S fixed, meta-learned rates give
   +8.0 (A) but only +0.6 pp (B) on the test stream.

**Implication for the hypothesis.** At this budget and data size, the "details of the current
context" that an online learner can exploit reduce to which labels are likely. To test the
hypothesis properly, a stream would need context information that is not in the label
distribution, e.g. bursts where the label set is the same in every context but the images
differ (a new drawer's style per burst), or a control that removes label locality by masking to
the alphabet everywhere and asking whether online learning adds anything on top of the mask.

## Caveats

- One seed, one stream, one run per setting.
- sgd, dual and output-bias sgd learning rates are chosen on the evaluation stream (optimistic);
  the prior, the cache and Meta-SGD are tuned on the val drawers only.
- The offline models use drawers 1-10, not all 12 training drawers, so that tuning and
  meta-training use images the models haven't memorized (`../README.md`).
- Meta-SGD's episodes (two drawings per character, bursts of up to 50 images) are shorter and
  less repetitive than the test bursts (8 drawings per character, 112-440 images); the val drawers
  don't allow more.
