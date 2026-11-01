# Experiment 3: Meta-SGD

**Question.** If the fast part's per-parameter learning rates and decays are meta-learned
instead of set by hand, does online adaptation beat the frozen mixed model, including on short
bursts where experiments 1 and 2 gain nothing?

The update is online.py's dual rule with lr = 0 and a fast part on every parameter: predict with
S + F, then F <- lambda * F - alpha * g. alpha and lambda are per-parameter tensors.
`meta_train.py` learns them on episodes of 3 bursts of 20-60 training images, backpropagating
through the whole unroll. The same checkpoint is evaluated on both streams.

| config | meta-learned | S |
|---|---|---|
| `metasgd` | S, alpha, lambda | starts at the mixed model, then meta-learned |
| `metasgd_fixedS` | alpha, lambda | the mixed model, fixed |

Meta settings (both): Adam 1e-3, meta-batch 32, gradient clip 1, init alpha 1e-3 and lambda
0.99, held-out accuracy every 50 steps, patience 30, at most 15 min per config and h. All four
runs reached the 15 min cap (4800-5850 steps).

## Results

Gain over frozen / locality effect (bursts - shuffled), in pp:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| frozen mixed (%) | 73.25 | 64.41 | 74.05 | 64.78 |
| `metasgd` | +2.70 / +3.56 | +2.36 / +4.05 | +3.51 / +3.99 | +3.77 / +4.41 |
| `metasgd_fixedS` | +1.19 / +1.20 | +1.07 / +1.31 | +1.89 / +1.99 | +1.87 / +2.05 |

Plots: `plots/summary.png`, `plots/burst_curve_short.png`, `plots/burst_curve_L1000.png`.

## Findings

- Meta-SGD is the only method that gains on short bursts: +2.7 pp (h = 8) and +2.4 pp (h = 4),
  with a positive locality effect. On L1000 it gains +3.5 and +3.8 pp, less than the best dual
  setting there (+4.1 and +6.1 pp, experiment 2), but without losing anything on short bursts.
- Learning S matters: with S fixed, the gain roughly halves (+1.1 to +1.9 pp). The meta-learned S
  alone, frozen, is only -0.3 to +0.5 pp from the mixed model. So learning S mostly makes the
  model easier to adapt, not more accurate by itself.
- Adaptation is fast and the switching cost is small: on short bursts accuracy is 0.6-1.2 pp
  below frozen at images 1-10, then above it from image 11 (h = 8: 72.6%, 75.9% at 11-50, 76.5%
  at 51+), against -6 to -8 pp for `dual_f1e-2_d0.999` (experiment 2).
- On L1000 the curve flattens after about 50 images (77.1% at 51-200, 77.8% at 201+, h = 8): the
  meta-learned decays (median lambda 0.74 to 0.97) forget faster than a 1000-image burst needs.
  Meta-training only saw bursts of 20-60.

Caveats: one seed and one meta-training run per config and h; meta-training is time-capped and
had not converged (the old h = 8 run reached a better held-out accuracy after 40 min); the
settings were compared on the evaluation streams.

Next step: meta-train on longer episodes (or with a decay floor) so the fast part keeps learning
through long bursts, and run it longer than 15 min.

Previous, longer version: `git show a974b9b:experiments_MNIST2/experiment3_metaSGD/README.md`.
