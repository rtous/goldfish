# Experiment 1: plain online SGD

**Question.** Does plain online SGD (batch 1, no momentum), starting from the mixed model, beat
the frozen mixed model on either stream?

| config | method | lr |
|---|---|---|
| `sgd_1e-5` | sgd | 1e-5 |
| `sgd_1e-4` | sgd | 1e-4 |
| `sgd_1e-3` | sgd | 1e-3 |
| `sgd_3e-3` | sgd | 3e-3 |

## Results

Gain over frozen / locality effect (bursts - shuffled), in pp:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| frozen mixed (%) | 73.25 | 64.41 | 74.05 | 64.78 |
| `sgd_1e-5` | +0.25 / +0.21 | +0.09 / +0.05 | +0.53 / +0.10 | +0.37 / -0.01 |
| `sgd_1e-4` | +0.17 / +0.00 | -0.15 / -0.01 | +0.69 / +0.35 | +0.66 / +0.33 |
| `sgd_1e-3` | -2.88 / +0.61 | -2.40 / +0.39 | +0.20 / +2.81 | +1.58 / +3.24 |
| `sgd_3e-3` | -7.03 / +0.86 | -5.13 / +1.28 | -0.63 / +5.89 | +1.41 / +6.66 |

Plots: `plots/summary.png`, `plots/burst_curve_short.png`, `plots/burst_curve_L1000.png`.

## Findings

- Short bursts: no gain. Small learning rates stay within ±0.25 pp of the frozen model, and larger
  ones lose 2 to 7 pp. The bursts are too short to pay back the cost of each context switch.
- L1000: at most +0.7 pp (h = 8, lr 1e-4) and +1.6 pp (h = 4, lr 1e-3).
- Larger learning rates give a larger locality effect (up to +6.7 pp). They adapt within a burst,
  reaching 75.6% (h = 8) and 68.4% (h = 4) after image 200 at lr 3e-3. But they pay for it at the
  start of each burst (50-58% over images 1-10 on L1000), so the net gain stays small.
- With small learning rates, part of the L1000 gain is not locality: the shuffled control also
  beats the frozen model (e.g. +0.37 pp at h = 4, lr 1e-5). Each test image appears 8 times in
  L1000 (once per context), so the model can learn the images themselves.

Caveats: one seed; the settings were compared on the evaluation streams, so the best one is
slightly optimistic.

Next step: none for plain SGD on its own; it is the reference for experiments 2 and 3.

Previous, longer version:
`git show a974b9b:experiments_MNIST2/experiment1/README.md`.
