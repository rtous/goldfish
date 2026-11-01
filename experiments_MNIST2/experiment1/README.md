# Experiment 1: locality in the inputs (rotated and permuted MNIST streams)

**Question.** In `experiments_MNIST/` the stream's locality was only in the labels, so a label
prior that learns no weights beat every online method. Here every burst holds all 10 digits, and
what the burst shares is a context: a fixed transformation of its images. For a small model, does
online learning (plain SGD, or slow + fast weights) beat the same model trained offline on all
contexts and then frozen?

**Answer.** No. The offline headroom is large (a model trained on the current context alone is
19 to 23 pp more accurate than the mixed model on rotated MNIST at h ≤ 8). But no online variant
gains more than +0.4 pp over the frozen mixed model, at any size, on either stream. Any learning
rate large enough to change the model hurts. The tuning drives every variant to near-zero
learning rates, where it is the frozen model.

## Setup

- Contexts (`contexts.py`, see `plots/contexts_rotated.png` and `plots/contexts_permuted.png`):
  - `rotated`: 8 contexts, rotations by 0, 22.5, …, 157.5 degrees counterclockwise.
  - `permuted`: 8 contexts, each a fixed random permutation of the 784 pixels (seeds 0 to 7).
- Streams (`make_sequence.py`): `sequence_<family>.csv` holds each of the 10000 MNIST test images
  exactly once, in a random order, so labels are uniform everywhere. It has 88 bursts of 20 to
  200 images (mean 113.6), and each burst has a context different from the previous burst's. Each context starts
  4 to 14 bursts (13 13 10 14 13 4 9 12). Both families use the same images, bursts and
  context indices. `sequence_<family>_random.csv` holds the same rows shuffled (the control).
  The online settings were tuned on `sequence_<family>_tune.csv` (another seed, 82 bursts), never
  on the stream they are scored on.
- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 32, 8, 4, 2.
- Offline, mixed (the frozen baseline): Adam, lr 0.001, batch 128, 50 epochs, seed 0. Each time a
  training image is sampled it is shown under a context drawn uniformly at random.
- Online: one configuration per stream, size and variant, `<stream>_<h>_<variant>` (70 in all).
  All variants of a stream and size start from the same offline weights.

| variant | online_params, rotated | online_params, permuted |
|---|---|---|
| `frozen` | lr 0 | lr 0 |
| `sgd` | SGD lr 0.00001 | SGD lr 0.0001 |
| `prior` | lr 0, `label_prior` 0.1 | lr 0, `label_prior` 0.1 |
| `fastbias` | SGD 0.00001, fast 0.00003 / decay 0.8 on `2.bias` | SGD 0.0001, fast 0.00003 / 0.999 |
| `fastout` | SGD 0.00001, fast 0.00003 / 0.8 on `2.weight`, `2.bias` | SGD 0.0001, fast 0.00003 / 0.9 |
| `fasthidden` | SGD 0.00001, fast 0.001 / 0.99 on `0.weight`, `0.bias` | SGD 0.0001, fast 0.00003 / 0.97 |
| `fastall` | SGD 0.00001, fast 0.00003 / 0.9 on every parameter | SGD 0.0001, fast 0.00003 / 0.8 |

The fast weights are those of `experiments_MNIST/experiment2`: slow part S and fast part F, the
network uses S + F, and after each image S takes an SGD step and F = decay · F − fast lr · g.

- Oracle (`oracle_accuracy`): the per-context models of the headroom check, frozen, each position
  of the stream scored by the model of its context. It knows the context in advance and has
  been trained on it alone.

## Headroom (offline)

`headroom.py`: test accuracy on all 10000 test images under each context, averaged over the 8
contexts. "mixed" is the frozen baseline; "per-context" is one model per context, each scored on
its own context (the ceiling for perfect adaptation at this size). Gap = per-context − mixed.
50 epochs for both; `headroom.csv`, `plots/headroom.png`.

| h | KB | rotated mixed % | rotated per-context % | rotated gap | permuted mixed % | permuted per-context % | permuted gap |
|---|---|---|---|---|---|---|---|
| 128 | 397.5 | 95.86 | 97.94 | +2.08 | 96.94 | 97.84 | +0.89 |
| 32 | 99.4 | 91.45 | 96.71 | +5.26 | 95.13 | 96.74 | +1.61 |
| 8 | 24.9 | 74.05 | 93.26 | **+19.21** | 89.17 | 92.51 | **+3.35** |
| 4 | 12.5 | 64.78 | 87.31 | **+22.53** | 81.04 | 86.25 | **+5.21** |
| 2 | 6.2 | 48.81 | 68.91 | **+20.10** | 58.08 | 65.48 | **+7.40** |

Both families pass the gate (gap > 1 pp at every h ≤ 8), so 16 contexts were not tried.

- The run was first made with 20 epochs, as the brief suggested. The per-context models had
  mostly converged, but the mixed models were still improving (`plots/headroom_curves.png`). At
  20 epochs the gaps were rotated +2.48, +6.81, +18.92, +21.50, +20.92 and permuted +1.10, +2.47,
  +3.70, +5.74, +9.61 (h = 128 … 2). At 50 epochs the mixed curves are flat except rotated
  h = 32 and 128, which still creep up by about 0.5 pp over the last 20 epochs.
- Rotation is much harder to absorb than permutation: an MLP has no notion of pixel
  neighbourhoods, so a permutation costs a fully connected layer little, while rotations of the
  same digit overlap and conflict at small h. The mixed rotated model is worst at 0° and 157.5°,
  the ends of the range (e.g. h = 8: 68.7% and 67.2% vs 76.9% at 90°).
- One permuted per-context model at h = 2 (context 1) is stuck at 40.9% (the others reach
  about 69%), so the permuted ceiling at h = 2 is about 3.5 pp lower than it would otherwise be.
- `mnist_mlp.py` reproduces the mixed models of `headroom.py` exactly (same training code and
  seed: `test_set_accuracy` equals `mixed_accuracy`).

## Tuning (on the tuning streams)

`tune.py`, full tables in `tuning.csv` (`python3 tune.py 1 --report`). One setting per stream and
variant, chosen by the best mean online gain over the five sizes (online − frozen on
`sequence_<family>_tune.csv`, pp). The grids of the brief (sgd lr {0.0003, 0.001, 0.003, 0.01};
fast lr {0.001, …, 1.0} × decay {0.9, 0.97, 0.99}) put the best value at the low edge
everywhere, so they were extended: sgd lr down to 0.00001, fast lr down to 0.00003, decay
{0.8, 0.995, 0.999}. 1750 runs; 7 diverged (permuted `fastall`, fast lr ≥ 0.3).

Plain SGD:

| stream | lr | h = 128 | h = 32 | h = 8 | h = 4 | h = 2 | mean |
|---|---|---|---|---|---|---|---|
| rotated | 0.00001 | -0.01 | -0.03 | +0.25 | +0.00 | +0.05 | **+0.05** |
| rotated | 0.00003 | +0.05 | +0.03 | +0.28 | +0.01 | -0.44 | -0.01 |
| rotated | 0.0001 | +0.14 | -0.06 | +0.21 | -0.02 | -0.96 | -0.14 |
| rotated | 0.0003 | -0.03 | -0.39 | -0.53 | -0.07 | -1.92 | -0.59 |
| rotated | 0.001 | -0.46 | -2.04 | -2.60 | -1.13 | -3.12 | -1.87 |
| rotated | 0.003 | -2.51 | -5.68 | -6.38 | -3.95 | -5.71 | -4.85 |
| rotated | 0.01 | -9.32 | -14.06 | -19.14 | -15.11 | -12.88 | -14.10 |
| permuted | 0.00001 | -0.02 | +0.03 | +0.02 | +0.04 | +0.18 | +0.05 |
| permuted | 0.00003 | +0.00 | +0.03 | +0.19 | +0.13 | +0.10 | +0.09 |
| permuted | 0.0001 | +0.02 | +0.07 | +0.29 | +0.17 | +0.36 | **+0.18** |
| permuted | 0.0003 | +0.11 | +0.10 | +0.28 | -0.10 | +0.05 | +0.09 |
| permuted | 0.001 | +0.04 | -0.05 | -0.35 | -0.81 | -0.99 | -0.43 |
| permuted | 0.003 | -0.73 | -1.50 | -3.09 | -3.86 | -4.16 | -2.67 |
| permuted | 0.01 | -7.43 | -12.15 | -12.42 | -16.43 | -16.11 | -12.91 |

Rotated's best SGD rate is the low edge (0.00001). A smaller one was not tried, since the gain
is already that of the frozen model. Its fast variants use slow lr 0.00001, permuted's 0.0001.

Fast variants: mean gain over the sizes at each fast lr, with its best decay in brackets
(settings with a diverged size left out):

| fast lr | rotated fastbias | rotated fastout | rotated fasthidden | rotated fastall | permuted fastbias | permuted fastout | permuted fasthidden | permuted fastall |
|---|---|---|---|---|---|---|---|---|
| 0.00003 | **+0.05** (0.99; 0.8 chosen, equal) | **+0.04** (0.8) | +0.08 (0.999) | **+0.04** (0.9) | **+0.19** (0.999) | **+0.18** (0.9) | **+0.19** (0.97) | **+0.18** (0.8) |
| 0.0001 | +0.05 (0.8) | +0.01 (0.9) | +0.08 (0.999) | +0.03 (0.9) | +0.19 (0.99) | +0.17 (0.8) | +0.19 (0.8) | +0.18 (0.8) |
| 0.0003 | +0.05 (0.8) | -0.04 (0.8) | +0.13 (0.999) | -0.06 (0.8) | +0.19 (0.999) | +0.17 (0.8) | +0.17 (0.8) | +0.14 (0.8) |
| 0.001 | +0.05 (0.97) | -0.42 (0.8) | **+0.20** (0.99) | -0.43 (0.8) | +0.19 (0.995) | +0.11 (0.8) | +0.13 (0.8) | +0.12 (0.8) |
| 0.003 | +0.05 (0.9) | -3.74 (0.9) | +0.09 (0.99) | -3.97 (0.9) | +0.18 (0.9) | -0.26 (0.9) | -0.01 (0.9) | -0.62 (0.9) |
| 0.01 | +0.03 (0.9) | -15.72 (0.9) | -0.70 (0.9) | -15.25 (0.99) | +0.17 (0.97) | -2.42 (0.9) | -1.32 (0.9) | -4.72 (0.9) |
| 0.03 | +0.00 (0.9) | -30.01 (0.99) | -4.77 (0.9) | -32.48 (0.99) | +0.16 (0.99) | -10.73 (0.9) | -7.99 (0.9) | -21.35 (0.9) |
| 0.1 | -0.03 (0.9) | -34.41 (0.99) | -20.80 (0.9) | -55.63 (0.9) | +0.10 (0.9) | -29.29 (0.99) | -31.27 (0.9) | -63.57 (0.9) |
| 0.3 | -0.28 (0.9) | -36.20 (0.99) | -46.47 (0.9) | -62.11 (0.9) | +0.05 (0.9) | -35.38 (0.99) | -54.44 (0.9) | -71.53 (0.9) |
| 1.0 | -1.69 (0.9) | -39.33 (0.99) | -55.44 (0.97) | -63.16 (0.9) | -0.91 (0.9) | -42.90 (0.99) | -63.25 (0.9) | -73.67 (0.9) |

Bold: the chosen setting (within a few hundredths of a point, several ties; the first one found
was kept). Most variants chose the smallest fast lr, 0.00003, where the fast part does almost
nothing and the variant equals `sgd`. A smaller rate was not tried, since it can only move
closer to `sgd`. The one exception is rotated `fasthidden` (fast lr 0.001, decay 0.99, +0.20).
Choosing per size does not change the picture: the best of all 175 settings at each size gains
at most +0.58 pp (rotated h = 4, `fasthidden` 0.003 / 0.99) and +0.36 pp on permuted.

## Results (evaluation streams)

Differences in percentage points against the frozen mixed model on the same images.
`results.csv`, `plots/accuracy_<stream>.*`, `plots/locality_<stream>.*`.

Online gain on bursts (online bursts − frozen), with the oracle's gain for reference:

| stream | h | frozen % | sgd | prior | fastbias | fastout | fasthidden | fastall | oracle |
|---|---|---|---|---|---|---|---|---|---|
| rotated | 128 | 95.81 | +0.03 | -0.98 | +0.03 | +0.04 | +0.05 | +0.03 | +2.13 |
| rotated | 32 | 91.17 | +0.01 | -2.76 | +0.01 | +0.04 | +0.08 | +0.06 | +5.56 |
| rotated | 8 | 73.25 | +0.25 | -6.11 | +0.25 | +0.26 | +0.40 | +0.24 | +20.00 |
| rotated | 4 | 64.41 | +0.09 | -6.45 | +0.09 | +0.05 | +0.16 | +0.06 | +23.10 |
| rotated | 2 | 48.39 | -0.03 | -7.80 | -0.03 | -0.09 | -0.13 | -0.05 | +20.51 |
| permuted | 128 | 97.02 | +0.06 | -0.32 | +0.06 | +0.06 | +0.07 | +0.06 | +1.01 |
| permuted | 32 | 95.20 | +0.15 | -1.06 | +0.15 | +0.15 | +0.15 | +0.15 | +1.54 |
| permuted | 8 | 89.25 | +0.10 | -2.73 | +0.10 | +0.09 | +0.10 | +0.08 | +3.46 |
| permuted | 4 | 80.95 | +0.23 | -5.37 | +0.23 | +0.25 | +0.25 | +0.24 | +4.82 |
| permuted | 2 | 58.18 | -0.01 | -7.49 | -0.01 | -0.02 | -0.01 | -0.02 | +6.03 |

Locality effect (online bursts − online random):

| stream | h | sgd | prior | fastbias | fastout | fasthidden | fastall |
|---|---|---|---|---|---|---|---|
| rotated | 128 | -0.01 | -0.09 | -0.01 | -0.01 | +0.04 | -0.01 |
| rotated | 32 | -0.04 | -0.29 | -0.04 | -0.02 | -0.07 | -0.01 |
| rotated | 8 | +0.21 | -0.42 | +0.21 | +0.20 | +0.24 | +0.16 |
| rotated | 4 | +0.05 | +0.56 | +0.05 | +0.01 | +0.14 | +0.02 |
| rotated | 2 | -0.19 | -0.73 | -0.19 | -0.19 | -0.03 | -0.11 |
| permuted | 128 | +0.01 | +0.05 | +0.01 | +0.01 | +0.02 | +0.01 |
| permuted | 32 | -0.01 | +0.12 | -0.01 | -0.01 | -0.01 | -0.01 |
| permuted | 8 | -0.11 | +0.46 | -0.11 | -0.14 | -0.10 | -0.13 |
| permuted | 4 | +0.00 | +0.21 | +0.00 | +0.03 | +0.01 | +0.02 |
| permuted | 2 | +0.08 | +0.36 | +0.09 | +0.10 | +0.07 | +0.09 |

Shuffled control (online random − frozen):

| stream | h | sgd | prior | fastbias | fastout | fasthidden | fastall |
|---|---|---|---|---|---|---|---|
| rotated | 128 | +0.04 | -0.89 | +0.04 | +0.05 | +0.01 | +0.04 |
| rotated | 32 | +0.05 | -2.47 | +0.05 | +0.06 | +0.15 | +0.07 |
| rotated | 8 | +0.04 | -5.69 | +0.04 | +0.06 | +0.16 | +0.08 |
| rotated | 4 | +0.04 | -7.01 | +0.04 | +0.04 | +0.02 | +0.04 |
| rotated | 2 | +0.16 | -7.07 | +0.16 | +0.10 | -0.10 | +0.06 |
| permuted | 128 | +0.05 | -0.37 | +0.05 | +0.05 | +0.05 | +0.05 |
| permuted | 32 | +0.16 | -1.18 | +0.16 | +0.16 | +0.16 | +0.16 |
| permuted | 8 | +0.21 | -3.19 | +0.21 | +0.23 | +0.20 | +0.21 |
| permuted | 4 | +0.23 | -5.58 | +0.23 | +0.22 | +0.24 | +0.22 |
| permuted | 2 | -0.09 | -7.85 | -0.10 | -0.12 | -0.08 | -0.11 |

Adaptation speed, rotated (online − frozen on the same positions within each burst; `sgd` /
`fasthidden`). The other fast variants match `sgd` to within 0.1 to 0.3 pp, and permuted is
within ±0.65 pp everywhere:

| h | images 1-10 | images 11-50 | images 51+ | first 20, seen context | first 20, new context |
|---|---|---|---|---|---|
| 128 | +0.00 / +0.00 | +0.24 / +0.21 | -0.09 / -0.04 | +0.19 / +0.19 | +0.00 / -0.63 |
| 32 | +0.11 / +0.11 | -0.18 / -0.21 | +0.10 / +0.24 | +0.12 / +0.19 | +0.00 / -1.24 |
| 8 | +0.22 / +0.11 | +0.09 / +0.30 | +0.34 / +0.50 | +0.50 / +0.44 | +0.00 / +1.26 |
| 4 | -0.12 / -1.02 | +0.30 / -0.18 | +0.00 / +0.54 | -0.12 / -0.43 | +0.00 / +0.00 |
| 2 | +0.57 / -0.45 | -0.06 / -0.42 | -0.10 / +0.09 | -0.38 / -0.94 | +0.62 / -0.63 |

The positions hold 880, about 3500 and about 5600 images. The "new context" column covers only 8
bursts (160 images; one image is 0.63 pp).

Retention (the final weights, frozen, on all test images under every context, averaged, minus
the mixed model): within −0.35 to +0.29 pp on rotated and +0.13 to +0.60 pp on permuted, for
every variant and size, both slow + fast and slow only. The online runs neither forgot the
general knowledge nor stored the contexts in a way the frozen weights could use.

## Findings

- **The headroom is there, but online learning does not reach it.** On rotated MNIST a model
  that knew the current context could gain 20 to 23 pp at h ≤ 8, and 3.5 to 6 pp on permuted.
  The online variants gain at most +0.40 pp (rotated h = 8, `fasthidden`). That is under 2% of
  the oracle's gain. The locality effect is equally small (≤ +0.24 pp outside the prior) and
  of either sign.
- **Every learning rate that moves the model hurts.** On the tuning streams, plain SGD at
  lr ≥ 0.0003 (rotated) or ≥ 0.001 (permuted) is below the frozen model at almost every size.
  Fast parts on the output layer, the hidden layer or everywhere are below it from fast lr 0.003
  or 0.01 up, and lose 20 to 74 pp at fast lr ≥ 0.1. The tuning therefore drove the learning
  rates to 0.00001 to 0.0001, and the online models are, to within a few tenths of a point, the
  frozen one. Only the fast output bias tolerates large rates (it cannot do much harm or good
  with uniform labels).
- **`fasthidden` is the best variant on rotated, but barely.** It is the only fast variant that
  tuned to a fast lr above the minimum (0.001, decay 0.99) and it leads at h = 128 to 4
  (+0.05 to +0.40 pp). It still sits at the noise level, and at h = 2 it is below the frozen model.
  On permuted every variant equals `sgd` (+0.06 to +0.25 pp at h ≥ 4). That gain shows up equally
  on the shuffled control, so it is a little more training, not locality.
- **The prior control behaves as expected for a stream without label locality.** It loses 0.3 to
  8 pp on both the bursts and the shuffled stream (locality effect −0.7 to +0.6 pp), because it
  sharpens towards the last ~10 labels, which are noise when labels are uniform. This is
  outside the brief's ±1 pp band. But the cost shrinks with the prior's rate (at h = 2,
  rotated: −7.80 / −1.71 / −0.56 pp at rate 0.1 / 0.01 / 0.001 on bursts; −7.07 / −1.39 / −0.63
  on random), and the next label repeats the previous one 10.28% of the time on bursts vs 10.11%
  shuffled. So the stream has no label locality.

A likely reading: one SGD step per image, from weights that fit all contexts at once, is a poor
way to reach a per-context solution. The per-context model at h = 8 needs many epochs over 60000
images of its context. A burst offers on average 114 images, each seen once. A fast hidden layer
can only take a few small steps before the context changes, and larger steps destroy the
features the other digits need before they help the current context.

Caveats: a single seed for offline training, sequences and online runs. Most differences
between variants are a few tenths of a point, within what another seed could change. The
"new context" metric rests on 8 bursts. The mixed rotated models at h = 32 and 128 were still
improving slightly at 50 epochs, so their headroom is a slight overestimate. The fast variants'
slow learning rate was fixed at the tuned SGD rate, as in the brief, so a larger slow rate
combined with a fast part was not tried. The online runs were made on a CPU.

**Next step.** Give adaptation a mechanism that can absorb a context from few images, rather
than more gradient steps. Options:
- a small per-context input transform learned online (e.g. a fast affine/rotation or
  normalization layer before the MLP, a few parameters instead of 785 h);
- a context-inference step (keep a few slow "experts", or per-context fast weights that are
  stored and recalled when a context recurs, instead of decaying);
- several passes or a replay buffer over the current burst.

The seen/new-context metrics are in place to test whether recalled fast weights let a recurring
context be re-adapted quickly.

## Checks

- `frozen` (lr 0) reproduces the offline accuracy exactly on both sequences, at every size.
- Rerunning 17 configurations (`rotated_8_*`, `permuted_2_fasthidden`, `rotated_2_prior`) gave
  identical rows; rebuilding the sequences with the defaults gives byte-identical files.
- `plots/contexts_rotated.png` and `plots/contexts_permuted.png` show the transformed images.
