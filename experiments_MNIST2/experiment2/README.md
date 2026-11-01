# Experiment 2: slow + fast weights

**Question.** Does a fast part on the hidden layer, which decays back towards the slow weights,
beat the frozen mixed model where plain SGD (experiment 1) does not?

The network uses S + F. After each image: S <- S - lr * g (skipped if lr = 0), then
F <- decay * F - fast_lr * g on 0.weight and 0.bias. F starts at 0, and S at the mixed model.

| config | slow lr | fast_lr | decay | fast layers |
|---|---|---|---|---|
| `dual_f1e-3_d0.99` | 1e-4 | 1e-3 | 0.99 | hidden |
| `dual_f3e-3_d0.997` | 1e-4 | 3e-3 | 0.997 | hidden |
| `dual_f1e-2_d0.999` | 1e-4 | 1e-2 | 0.999 | hidden |
| `dual_slow0_f1e-2_d0.999` | 0 | 1e-2 | 0.999 | hidden |

`dual_slow0_f1e-2_d0.999` isolates the fast part. It is also the closest hand-set analogue of
Meta-SGD (experiment 3).

## Results

Gain over frozen / locality effect (bursts - shuffled), in pp:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| frozen mixed (%) | 73.25 | 64.41 | 74.05 | 64.78 |
| `dual_f1e-3_d0.99` | +0.28 / +0.29 | +0.09 / +0.26 | +1.10 / +0.93 | +1.43 / +1.33 |
| `dual_f3e-3_d0.997` | +0.07 / +1.12 | -0.36 / +0.99 | +2.92 / +3.64 | +3.99 / +4.90 |
| `dual_f1e-2_d0.999` | -3.47 / +2.09 | -3.07 / +2.47 | +4.11 / +8.47 | +6.11 / +10.22 |
| `dual_slow0_f1e-2_d0.999` | -3.66 / +2.04 | -3.77 / +3.06 | +2.87 / +8.81 | +3.02 / +9.57 |

Plots: `plots/summary.png`, `plots/burst_curve_short.png`, `plots/burst_curve_L1000.png`.

## Findings

- L1000: a clear gain, up to +4.1 pp (h = 8) and +6.1 pp (h = 4) with `dual_f1e-2_d0.999`,
  against +0.7 and +1.6 pp for the best SGD. The locality effect (+8.5 and +10.2 pp) shows that
  the gain comes from the bursts.
- Short bursts: no gain. The settings that adapt fast lose 3 to 4 pp: accuracy falls to 56-67%
  over the first 50 images of a burst and recovers only after about 100 images.
- The fast part alone (slow lr 0) gains +2.9 and +3.0 pp on L1000. The slow part adds +1.2 pp
  (h = 8) and +3.1 pp (h = 4), partly by learning the repeated test images (see experiment 1).
- The pattern is the same at both sizes. The larger the fast learning rate, the more the model
  gains late in a burst and the more it loses right after a switch.

Caveats: one seed; the settings were compared on the evaluation streams, so the best one is
slightly optimistic.

Next step: lower the switching cost, e.g. by resetting or shrinking F at a context switch, which
the model would have to detect itself.

Previous, longer version:
`git show a974b9b:experiments_MNIST2/experiment2/README.md`.
