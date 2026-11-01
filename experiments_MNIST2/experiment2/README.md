# Experiment 2: burst-length sweep (exploratory, simplified)

**Question.** Experiment 1's bursts (20-200 images) were far too short for online learning to
adapt to a rotation. Does online learning beat the frozen mixed model when each context lasts
longer? At which burst length does it start to pay?

**Answer.** On rotated MNIST, yes, from L ≈ 500-1000 with a fast hidden layer and from L ≈ 2500
with plain SGD (h = 8 and 4). At L = 5000 it recovers about a quarter to 40% of the per-context
ceiling. Each context switch costs heavily: accuracy falls to about 50% at the start of every burst. On
permuted MNIST nothing pays (SGD gains a flat +0.4 to +0.6 pp at every L, with no locality effect).

This is a quick, reduced version of the planned sweep: 3 variants, 3 sizes, no new tuning.

## Setup

- Streams (`sequences/`, `make_sequence.py --burst-lengths 100 500 1000 2500 5000 --out-dir
  experiment2/sequences`): 80000 positions, each MNIST test image once under each of the 8
  contexts (each context has its own order of the 10000 images). Fixed burst length L, so each context
  gets 10000 / L bursts (2 at L = 5000), random order, never the same context twice in a row.
  One set of files serves both families (they hold only context indices). `_random` is the
  shuffled control; `_tune` (another seed) was used for tuning.
- Offline weights: experiment1's mixed models, copied (not retrained). Since every stream holds
  every (image, context) pair once, the frozen accuracy equals the test-set accuracy averaged
  over contexts, at every L.
- Sizes h = 8, 4, 2. Configurations `<family>_L<L>_<h>_<variant>`, 90 in all:

| variant | rotated | permuted |
|---|---|---|
| `frozen` | lr 0 | lr 0 |
| `sgd` | SGD lr 3e-5 (L ≤ 500), 1e-4 (L = 1000), 3e-3 (L ≥ 2500) | SGD lr 1e-4 |
| `fasthidden` | sgd's lr + fast first layer: lr / decay 0.001 / 0.99 (L = 100), 0.003 / 0.997 (L = 500), 0.01 / 0.999 (L ≥ 1000) | same as rotated, untuned |

- Tuning (`tuning.csv`, on the `_tune` streams, mean gain over h = 8, 4, 2): the sgd lr over
  {0.00001 … 0.01} for every family and L; `fasthidden` over fast lr {0.0003 … 1.0} × decay
  {0.99, 0.997, 0.999, 0.9997} on rotated at L = 100, 500, 1000 only. The full tuning was stopped
  early to keep the experiment small: L = 2500 and 5000 reuse L = 1000's fast setting, and
  permuted reuses rotated's (so permuted `fasthidden` is untuned).
- The quick gate (rotated tuning streams, SGD, h = 8 / 4): gain −0.31 / +0.03 pp at L = 100,
  +5.42 / +7.54 pp at L = 5000. It passed.

## Results

Online gain on bursts (online − frozen, pp); ceiling = per-context oracle − frozen:

| family | h | variant | L = 100 | L = 500 | L = 1000 | L = 2500 | L = 5000 | ceiling |
|---|---|---|---|---|---|---|---|---|
| rotated | 8 | sgd | +0.42 | +0.60 | +0.69 | +2.67 | +5.04 | +19.21 |
| rotated | 8 | fasthidden | +0.59 | +1.89 | **+4.11** | +3.21 | +4.88 | +19.21 |
| rotated | 4 | sgd | +0.41 | +0.45 | +0.66 | +5.39 | **+8.72** | +22.53 |
| rotated | 4 | fasthidden | +0.55 | +2.48 | +6.11 | +6.05 | +7.09 | +22.53 |
| rotated | 2 | sgd | -0.11 | -0.17 | -0.09 | +0.16 | +2.24 | +20.10 |
| rotated | 2 | fasthidden | +0.02 | +0.58 | +0.54 | -2.52 | -1.24 | +20.10 |
| permuted | 8 | sgd | +0.42 | +0.43 | +0.42 | +0.42 | +0.42 | +3.34 |
| permuted | 8 | fasthidden | +0.36 | -0.30 | -3.69 | -3.15 | -2.91 | +3.34 |
| permuted | 4 | sgd | +0.34 | +0.35 | +0.40 | +0.42 | +0.45 | +5.21 |
| permuted | 4 | fasthidden | +0.24 | -0.83 | -4.98 | -4.04 | -3.60 | +5.21 |
| permuted | 2 | sgd | +0.56 | +0.61 | +0.58 | +0.64 | +0.63 | +7.41 |
| permuted | 2 | fasthidden | +0.37 | -0.45 | -3.19 | -2.25 | -1.88 | +7.41 |

Locality effect (online bursts − online shuffled, pp):

| family | h | variant | L = 100 | L = 500 | L = 1000 | L = 2500 | L = 5000 |
|---|---|---|---|---|---|---|---|
| rotated | 8 | sgd | -0.04 | +0.07 | +0.35 | +9.29 | +11.38 |
| rotated | 8 | fasthidden | +0.24 | +2.46 | +8.48 | +15.05 | +16.53 |
| rotated | 4 | sgd | -0.04 | +0.06 | +0.33 | +10.61 | +13.94 |
| rotated | 4 | fasthidden | +0.32 | +3.23 | +10.22 | +15.34 | +16.62 |
| rotated | 2 | sgd | -0.06 | -0.16 | +0.25 | +4.87 | +6.93 |
| rotated | 2 | fasthidden | +0.27 | +1.33 | +3.85 | +5.24 | +6.57 |
| permuted | 8 | sgd | -0.03 | +0.01 | -0.06 | -0.01 | -0.02 |
| permuted | 4 | sgd | -0.02 | -0.04 | +0.05 | +0.12 | +0.12 |
| permuted | 2 | sgd | -0.03 | -0.07 | -0.07 | +0.05 | +0.08 |

Plots: `plots/sweep_gain.*`, `plots/sweep_locality.*` (against L), `plots/burst_curves.*`
(accuracy against position in the burst at L = 5000).

## Findings

- **Rotated: online learning pays once contexts last long enough.** `fasthidden` beats the frozen
  model by +2 to +2.5 pp at L = 500 and +4 to +6 pp at L = 1000 (h = 8, 4). Plain SGD needs
  L ≥ 2500 (its tuned lr is tiny below that). At L = 5000 the best gain is +5.0 pp (h = 8) and
  +8.7 pp (h = 4): 26% and 39% of the ceiling. The locality effect is large (up to +16.6 pp), so
  the gain comes from the bursts, not from extra training.
- **The fast hidden layer helps at medium lengths.** At L = 500-1000 it gains 1.5 to 5.5 pp more
  than SGD. At L ≥ 2500 SGD, with its larger tuned lr, catches up or passes it (`fasthidden` there uses
  L = 1000's untuned setting).
- **h = 2 barely benefits** (SGD +2.2 pp at L = 5000, `fasthidden` negative at L ≥ 2500).
- **Switching costs a lot, and returning contexts are not re-adapted faster.** At L = 5000 the
  first 100 images of a burst score about 50% (h = 8) against 73-74% for the frozen model. The
  online model has drifted to the previous context. The burst curves cross the frozen line only
  after about 600-800 images. The first 100 images of a returning context score only 0.5 to 5 pp
  better than those of the context's first burst. The slow weights keep little that speeds up
  re-adaptation.
- **Not memorization.** The gains on images seen for the first time and on images already seen
  under another context are similar (e.g. rotated h = 4, L = 5000, SGD: +8.2 vs +8.8 pp).
- **Permuted: no locality to exploit with these methods.** SGD's +0.4 to +0.6 pp is the same at
  every L and on the shuffled stream. The untuned `fasthidden` (rotated's settings) loses 2 to
  5 pp at L ≥ 1000.

Caveats: single seed; reduced tuning (see Setup), so `fasthidden` at L ≥ 2500 and on permuted is
not tuned; the tuning streams contain the same test images as the evaluation streams (in
another order), since every stream uses every (image, context) pair.

**Next step.** Cut the switching cost: keep the fast part from overwriting what other contexts
need (e.g. a fast part that resets or is recalled per context, or a slow part that is not updated
online), so that the model gains within a burst without losing 20 pp at every switch.
