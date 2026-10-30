# Experiment 2: two time scales (slow + fast weights)

**Question.** The online loss only sees the current image, so plain online SGD has to trade
adapting to the current burst against keeping what it knows (experiment 1). Does it help to split
knowledge by time scale: slow weights for the task, fast weights that adapt to the current
burst and then fade?

## Idea

Each chosen parameter gets a slow part S and a fast part F, and the network uses S + F
(Hinton & Plaut 1987, "Using fast weights to deblur old memories"). After each image, with g the
gradient on that image (the gradient w.r.t. S + F equals the gradient w.r.t. S):

    S = S - lr_slow * g             (normal optimizer step; skipped if lr_slow is 0)
    F = decay * F - lr_fast * g     (F starts at 0)

F is a decaying sum of recent gradients: it tracks the current burst and forgets it within
about 1 / (1 − decay) images, while S keeps learning slowly.

Related work:

- Complementary learning systems (McClelland et al. 1995): a fast-learning hippocampus and a
  slow-learning neocortex, so that new experiences do not overwrite old knowledge.
- Fast weights (Ba et al. 2016, "Using fast weights to attend to the recent past"): a decaying
  outer-product memory of recent hidden states, inside a recurrent network.
- Elastic weight consolidation (Kirkpatrick et al. 2017) aims the other way: it protects old
  knowledge from new learning. Here the fast part is meant to forget.

## Setup

- Sequence: `sequence.csv`, 10000 MNIST test images in 193 bursts of 3 digits
  (`make_sequence.py`). `sequence_random.csv` holds the same rows shuffled.
  It is a control, not a scenario we expect in practice: it holds the same images, so it tells
  whether an online gain comes from temporal locality or merely from training on more data. Real
  input streams are assumed to have temporal locality, not to be random.
- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 8, 4, 2.
- Offline: the weights of experiment 1 (Adam, lr 0.001, 10 epochs, batch size 128, seed 0),
  copied, not retrained.
- Online: one configuration per size and variant, `mlp_<h>_<variant>` (24 in all):

| variant | online_params | fast part | extra state at inference |
|---|---|---|---|
| `sgd` | SGD lr 0.001 | none (experiment 1's plain online SGD) | none |
| `prior` | slow lr 0 (frozen), `label_prior` 0.1 | none: a running label prior p, `logits += log(p)` | 10 values |
| `fastbias` | SGD lr 0.001, `fast` lr 1.0, decay 0.97 | output bias (`2.bias`) | 10 values |
| `fastout` | SGD lr 0.001, `fast` lr 0.003, decay 0.97 | output layer (`2.weight`, `2.bias`) | 10 h + 10 values |
| `fasthidden` | SGD lr 0.001, `fast` lr 0.003, decay 0.97 | first layer (`0.weight`, `0.bias`) | 785 h values |
| `fastall` | SGD lr 0.001, `fast` lr 0.003, decay 0.99 | every weight and bias | doubles the model |

The four fast variants vary which connections carry a fast weight, from the output bias alone to
every connection. `sgd` is any of them with the fast part removed; its results match
experiment 1's lr 0.001 runs exactly. `prior` learns no weights: after predicting each image,
p = 0.9 p + 0.1 onehot(label), clamped to ≥ 1e-3. It is the baseline that only tracks which
labels are frequent right now. The floor of 1e-3 makes digits absent from the recent past
very unlikely, e.g. at the start of a new burst.

- Oracle ceiling: the frozen model with its prediction restricted to the digits of the current
  burst (3 digits; one burst contains only 2). It knows the burst's digits in advance, including
  at the first image of each burst. It was computed separately from the saved weights (it is not
  a configuration).

## Tuning of `fastout` and `fasthidden`

The fast learning rate and decay of the two new splits were tuned on the bursts sequence over
fast lr {0.003, 0.01, 0.03, 0.1, 0.3, 1.0} × decay {0.97, 0.99}, at all four sizes, from copies
of the same offline weights. One setting per variant is used at all sizes, chosen by the best
mean online gain on bursts over the four sizes (ties broken by h = 2). No setting diverged.
Online gain on bursts (online bursts − frozen, pp):

| variant | fast lr | decay | h = 128 | h = 8 | h = 4 | h = 2 | mean |
|---|---|---|---|---|---|---|---|
| `fastout` | 0.003 | 0.97 | +0.39 | +1.51 | +2.48 | +8.34 | **+3.18** |
| `fastout` | 0.003 | 0.99 | +0.19 | +1.49 | +2.52 | +8.13 | +3.08 |
| `fastout` | 0.01 | 0.97 | -0.12 | +0.48 | +1.19 | +7.13 | +2.17 |
| `fastout` | 0.01 | 0.99 | -0.49 | +0.20 | +1.10 | +6.91 | +1.93 |
| `fastout` | 0.03 | 0.97 | -1.95 | -1.88 | -1.98 | +2.97 | -0.71 |
| `fastout` | 0.03 | 0.99 | -2.64 | -2.10 | -2.37 | +2.47 | -1.16 |
| `fastout` | 0.1 | 0.97 | -4.69 | -5.86 | -11.98 | -2.14 | -6.17 |
| `fastout` | 0.1 | 0.99 | -5.61 | -6.75 | -13.19 | -2.61 | -7.04 |
| `fastout` | 0.3 | 0.97 | -8.19 | -12.42 | -28.04 | -11.60 | -15.06 |
| `fastout` | 0.3 | 0.99 | -9.56 | -16.95 | -33.33 | -17.84 | -19.42 |
| `fastout` | 1.0 | 0.97 | -17.94 | -43.28 | -43.31 | -26.59 | -32.78 |
| `fastout` | 1.0 | 0.99 | -20.28 | -42.80 | -46.64 | -30.71 | -35.11 |
| `fasthidden` | 0.003 | 0.97 | +0.27 | +1.68 | +2.50 | +6.49 | **+2.73** |
| `fasthidden` | 0.003 | 0.99 | +0.31 | +1.61 | +2.51 | +6.35 | +2.70 |
| `fasthidden` | 0.01 | 0.97 | +0.23 | +1.50 | +2.15 | +5.68 | +2.39 |
| `fasthidden` | 0.01 | 0.99 | +0.12 | +1.37 | +1.65 | +4.91 | +2.01 |
| `fasthidden` | 0.03 | 0.97 | -0.02 | -0.39 | -1.10 | +0.71 | -0.20 |
| `fasthidden` | 0.03 | 0.99 | -0.49 | -1.60 | -2.31 | -1.54 | -1.48 |
| `fasthidden` | 0.1 | 0.97 | -4.21 | -6.30 | -10.22 | -9.77 | -7.62 |
| `fasthidden` | 0.1 | 0.99 | -8.22 | -8.00 | -15.57 | -13.53 | -11.33 |
| `fasthidden` | 0.3 | 0.97 | -13.59 | -16.42 | -29.54 | -23.10 | -20.66 |
| `fasthidden` | 0.3 | 0.99 | -17.30 | -19.47 | -41.20 | -27.11 | -26.27 |
| `fasthidden` | 1.0 | 0.97 | -26.19 | -33.22 | -49.23 | -34.55 | -35.80 |
| `fasthidden` | 1.0 | 0.99 | -29.36 | -46.24 | -64.02 | -41.19 | -45.20 |

Both variants chose fast lr 0.003 and decay 0.97. The gain falls steadily as the fast learning
rate rises, so 0.003 is the edge of the grid and a smaller rate might do slightly better. The
tuning was done on the same sequence the results are scored on, as for `fastbias` and
`fastall`.

## Results

Differences in percentage points against the frozen model on the same images.

Online gain on bursts (online bursts − frozen):

| h | KB | frozen % | sgd | prior | fastbias | fastout | fasthidden | fastall | oracle |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 397.5 | 97.86 | +0.19 | +0.80 | +0.30 | +0.39 | +0.27 | +0.17 | +1.45 |
| 8 | 24.9 | 92.59 | +1.36 | +2.76 | +1.83 | +1.51 | +1.68 | +1.34 | +4.83 |
| 4 | 12.5 | 85.85 | +2.30 | +5.17 | +3.51 | +2.48 | +2.50 | +2.39 | +8.86 |
| 2 | 6.2 | 64.27 | +6.44 | +13.43 | +12.66 | +8.34 | +6.49 | +7.92 | +21.48 |

Locality effect (online bursts − online random):

| h | sgd | prior | fastbias | fastout | fasthidden | fastall |
|---|---|---|---|---|---|---|
| 128 | -0.05 | +1.22 | +0.09 | +0.17 | +0.04 | +0.34 |
| 8 | +1.49 | +4.78 | +2.46 | +2.82 | +2.09 | +3.63 |
| 4 | +3.69 | +9.83 | +6.31 | +6.11 | +4.53 | +7.77 |
| 2 | +6.99 | +24.00 | +19.98 | +13.05 | +7.68 | +14.38 |

Shuffled control (online random − frozen):

| h | sgd | prior | fastbias | fastout | fasthidden | fastall |
|---|---|---|---|---|---|---|
| 128 | +0.24 | -0.42 | +0.21 | +0.22 | +0.23 | -0.17 |
| 8 | -0.13 | -2.02 | -0.63 | -1.31 | -0.41 | -2.29 |
| 4 | -1.39 | -4.66 | -2.80 | -3.63 | -2.03 | -5.38 |
| 2 | -0.55 | -10.57 | -7.32 | -4.71 | -1.19 | -6.46 |

Plots: `plots/accuracy.*` and `plots/locality.*` (`plot_results.py 2`).

- `prior` gains most on bursts at every size (+13.43 pp at h = 2, 63% of the oracle's
  +21.48), without learning any weights.
- `fastbias` is second at every size and close to `prior` at h = 2 (+12.66 pp), nearly double
  `sgd` (+6.44). At h = 128 every gain is below 1 pp.
- `fastout` is not close to `fastbias`. At h = 2 it gains +8.34 pp, about a third of the way
  from `sgd` to `fastbias`, and at h = 8 and 4 it is level with `sgd`. Its tuned fast learning
  rate is 0.003, against 1.0 for the bias: larger rates make the fast output weights, whose
  gradients scale with the hidden activations, hurt at every size (see Tuning). A likely
  reading: the fast output weights add little to what the fast bias does, and they force a
  fast learning rate too small for the bias to track the burst.
- `fasthidden` is level with `sgd` at h = 2 (+6.49 vs +6.44 pp) and slightly above it at h = 8
  and 4 (+1.68 and +2.50 vs +1.36 and +2.30), where it is level with `fastout`. It is the fast
  variant that stays closest to the frozen model on the shuffled control (-1.19 pp at h = 2).
- `fastall` is level with `fastout` (+7.92 vs +8.34 pp at h = 2, within 0.25 pp elsewhere) and
  doubles the model.
- On the shuffled control every variant ends up at or below the frozen model for h ≤ 8 (at
  h = 2 `prior` -10.57 pp, `fastbias` -7.32, `fastall` -6.46, `fastout` -4.71, `fasthidden`
  -1.19, `sgd` -0.55), so their gains on bursts come from the temporal locality, not from
  training on more data. The variants that adapt most to the recent past depend on it most; the
  prior most of all, since it nearly rules out digits it has not seen recently.

The locality in this sequence is only in the labels: the images of a burst share nothing but
their digits. The label prior is therefore the natural winner, and a fast output bias, which can
learn the same thing with a large fast learning rate, comes close to it. Every split that gives
fast parts to weights has to use a fast learning rate over 300 times smaller, and gains far less:
`fastout`, `fastall` and `fasthidden` all end up between `sgd` and `fastbias`, closer to `sgd`.
Which layer the fast weights sit in matters less here than whether the bias can move freely.

Caveats: the fast learning rates and decays were tuned on this sequence, the same one they are
scored on, and the tuned rate of `fastout` and `fasthidden` is at the edge of its grid. There is
a single seed. The gaps at h = 128 (tenths of a point) and most gaps between `fastout`,
`fasthidden`, `fastall` and `sgd` at h ≥ 4 are within what another seed could change.

**Next step.** A sequence with locality in the inputs, not only in the labels, e.g. one shared
distortion (shift, rotation, thickness) per burst. A label prior cannot exploit that; fast
weights might. `fasthidden` is the natural variant to test there, since the first layer is
where a shared input distortion would have to be absorbed.

## Notes

The online runs in `results.csv` were made on a CPU (`get_device` falls back to it when there is
no CUDA GPU). Earlier runs on this sequence matched an independent NumPy replica to the second
decimal.
