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
- Online: one configuration per size and variant, `mlp_<h>_<variant>` (16 in all):

| variant | online_params | fast part | extra state at inference |
|---|---|---|---|
| `sgd` | SGD lr 0.001 | none (experiment 1's plain online SGD) | none |
| `prior` | slow lr 0 (frozen), `label_prior` 0.1 | none: a running label prior p, `logits += log(p)` | 10 values |
| `fastbias` | SGD lr 0.001, `fast` lr 1.0, decay 0.97 | output bias (`2.bias`) | 10 values |
| `fastall` | SGD lr 0.001, `fast` lr 0.003, decay 0.99 | every weight and bias | doubles the model |

`sgd` is `fastbias`/`fastall` with the fast part removed; its results match experiment 1's lr
0.001 runs exactly. `prior` learns no weights: after predicting each image,
p = 0.9 p + 0.1 onehot(label), clamped to ≥ 1e-3. It is the baseline that only tracks which
labels are frequent right now. The floor of 1e-3 makes digits absent from the recent past
very unlikely, e.g. at the start of a new burst.

- Oracle ceiling: the frozen model with its prediction restricted to the digits of the current
  burst (3 digits; one burst contains only 2). It knows the burst's digits in advance, including
  at the first image of each burst. It was computed separately from the saved weights (it is not
  a configuration).

## Results

Differences in percentage points against the frozen model on the same images.

Online gain on bursts (online bursts − frozen):

| h | KB | frozen % | sgd | prior | fastbias | fastall | oracle |
|---|---|---|---|---|---|---|---|
| 128 | 397.5 | 97.86 | +0.19 | +0.80 | +0.30 | +0.17 | +1.45 |
| 8 | 24.9 | 92.59 | +1.36 | +2.76 | +1.83 | +1.34 | +4.83 |
| 4 | 12.5 | 85.85 | +2.30 | +5.17 | +3.51 | +2.39 | +8.86 |
| 2 | 6.2 | 64.27 | +6.44 | +13.43 | +12.66 | +7.92 | +21.48 |

Locality effect (online bursts − online random):

| h | sgd | prior | fastbias | fastall |
|---|---|---|---|---|
| 128 | -0.05 | +1.22 | +0.09 | +0.34 |
| 8 | +1.49 | +4.78 | +2.46 | +3.63 |
| 4 | +3.69 | +9.83 | +6.31 | +7.77 |
| 2 | +6.99 | +24.00 | +19.98 | +14.38 |

Shuffled control (online random − frozen):

| h | sgd | prior | fastbias | fastall |
|---|---|---|---|---|
| 128 | +0.24 | -0.42 | +0.21 | -0.17 |
| 8 | -0.13 | -2.02 | -0.63 | -2.29 |
| 4 | -1.39 | -4.66 | -2.80 | -5.38 |
| 2 | -0.55 | -10.57 | -7.32 | -6.46 |

Plots: `plots/accuracy.*` and `plots/locality.*` (`plot_results.py 2`).

- `prior` gains most on bursts at every size (+13.43 pp at h = 2, 63% of the oracle's
  +21.48), without learning any weights.
- `fastbias` is second at every size and close to `prior` at h = 2 (+12.66 pp), nearly double
  `sgd` (+6.44). At h = 128 every gain is below 1 pp.
- `fastall` is barely better than `sgd` on bursts (+7.92 vs +6.44 pp at h = 2, level at the
  other sizes), and doubles the model.
- On the shuffled control every variant ends up at or below the frozen model (at h = 2 `prior`
  -10.57 pp, `fastbias` and `fastall` -6.5 to -7.3, `sgd` -0.55), so their gains on bursts
  come from the temporal locality, not from training on more data. The variants that adapt most
  to the recent past depend on it most; the prior most of all, since it nearly rules out digits
  it has not seen recently.

The locality in this sequence is only in the labels: the images of a burst share nothing but
their digits. The label prior is therefore the natural winner, and a fast output bias, which can
learn the same thing, comes close to it. A fast copy of the whole network has more to learn and
gains less.

Caveats: the fast learning rates and decays were tuned on this sequence, the same one they are
scored on. There is a single seed. The gaps at h = 128 (tenths of a point) are within what
another seed could change.

**Next step.** A sequence with locality in the inputs, not only in the labels, e.g. one shared
distortion (shift, rotation, thickness) per burst. A label prior cannot exploit that; fast
weights might.

## Notes

The online runs in `results.csv` were made on a CPU (`get_device` falls back to it when there is
no CUDA GPU). Earlier runs on this sequence matched an independent NumPy replica to the second
decimal.
