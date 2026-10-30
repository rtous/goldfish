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

- Sequence: `sequence.csv` with 10% intruders (`make_sequence.py`). It is the first version of
  the sequence (`sequence_v1.csv`) with 1000 of its 10000 positions, picked at random, replaced
  by a digit that does not appear in that position's burst (a random MNIST test image of it).
  The 193 bursts are otherwise unchanged. `sequence_random.csv` holds the same rows shuffled. On it, the
  "intruders" are the same images, now out of their burst context.
- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 8, 4, 2.
- Offline: the weights of experiment 1 (Adam, lr 0.001, 10 epochs, batch size 128, seed 0),
  copied, not retrained. The frozen accuracy was rescored on the new sequence
  (`mnist_mlp.py 2 --score-only`), since its images changed.
- Online: one configuration per size and variant, `mlp_<h>_<variant>` (16 in all):

| variant | online_params | fast part | extra state at inference |
|---|---|---|---|
| `sgd` | SGD lr 0.001 | none (experiment 1's plain online SGD) | none |
| `prior` | slow lr 0 (frozen), `label_prior` 0.1, `label_prior_mix` 0.2 | none: a running label prior p, `logits += log((1 − m) p + m / 10)` | 10 values |
| `fastbias` | SGD lr 0.001, `fast` lr 1.0, decay 0.97 | output bias (`2.bias`) | 10 values |
| `fastall` | SGD lr 0.001, `fast` lr 0.003, decay 0.99 | every weight and bias | doubles the model |

`sgd` is `fastbias`/`fastall` with the fast part removed; its results match experiment 1's lr
0.001 runs exactly. `prior` learns no weights: after
predicting each image, p = 0.9 p + 0.1 onehot(label), clamped to ≥ 1e-3. It is the baseline that
only tracks which labels are frequent right now. Without the mix (m = 0), the floor of 1e-3
nearly forbids digits absent from the recent past, which guarantees failure on intruders. m was
chosen at h = 2 as the best overall accuracy on the bursts sequence among 0, 0.05, 0.1 and
0.2:

| m | overall % | intruders % | regular % |
|---|---|---|---|
| 0 | 71.24 | 20.20 | 76.91 |
| 0.05 | 71.95 | 25.40 | 77.12 |
| 0.1 | 72.29 | 28.60 | 77.14 |
| 0.2 | 72.73 | 33.60 | 77.08 |

0.2 is the edge of that grid. A later check outside it gave 73.19% at m = 0.3 and 73.20% at
m = 0.5 (intruders 38.0 and 43.0%), so a larger m would help the prior a little more.

- Oracle ceiling, on the regular images only: the frozen model with its prediction restricted
  to the digits of the current burst (3 digits; one burst contains only 2). It knows the burst's
  digits in advance, including at the first image of each burst. On intruders it is always wrong
  by construction.

## Results

All on the bursts sequence unless stated; differences in percentage points against the frozen
model on the same images.

Overall (online bursts − frozen):

| h | KB | frozen % | sgd | prior | fastbias | fastall |
|---|---|---|---|---|---|---|
| 128 | 397.5 | 97.82 | +0.17 | +0.42 | +0.31 | -0.03 |
| 8 | 24.9 | 92.46 | +1.00 | +2.20 | +1.28 | +0.29 |
| 4 | 12.5 | 85.68 | +1.12 | +3.22 | +1.56 | -0.31 |
| 2 | 6.2 | 63.95 | +5.00 | +8.78 | +7.51 | +4.25 |

Regular (burst) images, 9000 (online − frozen on them):

| h | frozen % | sgd | prior | fastbias | fastall | oracle |
|---|---|---|---|---|---|---|
| 128 | 97.86 | +0.14 | +0.60 | +0.31 | +0.05 | +1.41 |
| 8 | 92.57 | +1.40 | +2.89 | +1.70 | +1.05 | +4.77 |
| 4 | 85.70 | +2.60 | +5.00 | +3.51 | +1.74 | +9.02 |
| 2 | 64.18 | +6.98 | +12.90 | +11.42 | +6.84 | +21.55 |

Intruders, 1000 (online − frozen on them):

| h | frozen % | sgd | prior | fastbias | fastall |
|---|---|---|---|---|---|
| 128 | 97.50 | +0.40 | -1.20 | +0.30 | -0.80 |
| 8 | 91.50 | -2.60 | -4.00 | -2.50 | -6.60 |
| 4 | 85.50 | -12.20 | -12.80 | -16.00 | -18.80 |
| 2 | 61.90 | -12.80 | -28.30 | -27.70 | -19.10 |

Shuffled stream (online random − frozen):

| h | sgd | prior | fastbias | fastall |
|---|---|---|---|---|
| 128 | +0.28 | -0.12 | +0.19 | -0.10 |
| 8 | -0.18 | -0.80 | -0.72 | -2.34 |
| 4 | -1.48 | -2.30 | -2.90 | -5.78 |
| 2 | +0.01 | -6.71 | -6.26 | -5.48 |

Plots: `plots/accuracy.*` and `plots/locality.*` (`plot_results.py 2`), and
`plots/intruders.*` (`experiment2/plot_intruders.py`): accuracy on intruders and on regular
images against model size, per variant, with the frozen model dashed.

**Which variant gains most on the burst images while losing least on the intruders?** None
gets both. Below h = 128 every variant loses on intruders, and the two that gain most on the
regular images (`prior`, `fastbias`) lose the most on them at h = 2. Variant by variant:

- `prior` gains most on the regular images at every size (+12.90 pp at h = 2, 60% of the
  oracle's +21.55) and has the best overall accuracy at every size. On intruders it loses about
  as much as `fastbias` at h = 2 (-28.3 pp) and more than `sgd` at every size. With the mix it is
  no longer guaranteed to fail on them, but at h = 2 it still gets only 33.6% of them right.
- `fastbias` is second on the regular images and second overall at every size. At h = 128 and
  8 it loses about the same on intruders as `sgd` (+0.3 vs +0.4, -2.5 vs -2.6 pp), so there it
  is a clear improvement over `sgd`.
  At h = 4 and 2 it loses much more (-16.0 and -27.7 vs -12.2 and -12.8 pp).
- `sgd` gains least among the three, but loses least on intruders at h = 4 and 2 and has no
  loss at all on the shuffled stream at h = 2. Its slow, single time scale does not commit to
  the burst as strongly.
- `fastall` is worse than `sgd` on both counts at every size: less gain on the regular images
  and more loss on intruders. Its overall accuracy is below the frozen model at h = 128 and 4.

Compared with the v1 sequence (below), intruders shrink every variant's overall gain at h = 2
(`fastbias` +7.51 instead of +12.66 pp, `prior` +8.78 instead of +13.43). The ranking on
bursts is unchanged (prior > fastbias > sgd, fastall last or near it), and `fastbias` is no longer
nearly double `sgd` (+7.51 vs +5.00).

The locality in this sequence is still only in the labels: intruders break the label pattern,
but the images of a burst share nothing else. The label prior is therefore still the natural
winner.

Caveats: the fast learning rates and decays were tuned on the v1 sequence, and m on this one,
both on the same sequences they are scored on. There is a single seed. The gaps at h = 128
(tenths of a point) and the 1000-image intruder accuracies (±1.5 pp at h = 2 from sampling
alone) are within what another seed or draw could change.

**Next step.** A sequence with locality in the inputs, not only in the labels, e.g. one shared
distortion (shift, rotation, thickness) per burst. A label prior cannot exploit that; fast
weights might.

## Previous sequence (v1, no intruders)

The first runs of this experiment used `sequence_v1.csv`, the first version of the sequence,
with the prior at m = 0 (no mix). There, `sgd` reproduced experiment 1's lr 0.001 runs
exactly. Accuracy in %, online on bursts / online on random:

| h | frozen | sgd | prior (m = 0) | fastbias | fastall | oracle |
|---|---|---|---|---|---|---|
| 128 | 97.86 | 98.05 / 98.10 | 98.66 / 97.44 | 98.16 / 98.07 | 98.03 / 97.69 | 99.31 |
| 8 | 92.59 | 93.95 / 92.46 | 95.35 / 90.57 | 94.42 / 91.96 | 93.93 / 90.30 | 97.42 |
| 4 | 85.85 | 88.15 / 84.46 | 91.02 / 81.19 | 89.36 / 83.05 | 88.24 / 80.47 | 94.71 |
| 2 | 64.27 | 70.71 / 63.72 | 77.70 / 53.70 | 76.93 / 56.95 | 72.19 / 57.81 | 85.75 |

On v1, the label prior was best on bursts at every size, `fastbias` nearly doubled `sgd`'s gain
at h = 2, and `fastall` gained less than `fastbias`.

## Notes

These runs were made on a CPU (the machine had no CUDA GPU) through a wrapper that only
replaces the device. The v1 runs matched an independent NumPy replica to the second decimal.
