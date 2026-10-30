# Experiment 1: online learning rate

**Question.** Do small MLPs exploit temporal locality when they keep learning online, one image at
a time, with plain SGD? Which online learning rate does it best?

## Setup

- Sequence: `sequence.csv`, bursts of 3 digits with 10% intruders (a digit that does not appear
  in its burst; see `make_sequence.py`). `sequence_random.csv` holds the same rows shuffled.
- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 16, 8, 4, 2.
- Offline: Adam, lr 0.001, 10 epochs, batch size 128, seed 0.
- Online: plain SGD (momentum 0), lr = 0.0003, 0.001, 0.003, 0.01.
- One configuration per size and learning rate: `mlp_<h>_lr<lr>` (20 in all). Configurations
  of the same size share copies of the same offline weights.

## Results

All on the bursts sequence unless stated; differences in percentage points against the frozen
model on the same images.

Online gain on bursts (online bursts − frozen):

| h | KB | frozen % | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|---|---|
| 128 | 397.5 | 97.82 | +0.22 | +0.17 | -0.24 | -2.59 |
| 16 | 49.7 | 95.07 | +0.16 | +0.22 | -0.60 | -5.37 |
| 8 | 24.9 | 92.46 | +0.62 | +1.00 | +0.03 | -4.42 |
| 4 | 12.5 | 85.68 | +1.02 | +1.12 | -0.51 | -8.33 |
| 2 | 6.2 | 63.95 | +3.41 | +5.00 | +4.51 | -6.90 |

Locality effect (online bursts − online random):

| h | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|
| 128 | -0.05 | -0.11 | +0.03 | +0.68 |
| 16 | +0.00 | +0.15 | +0.30 | +1.97 |
| 8 | +0.36 | +1.18 | +1.99 | +3.50 |
| 4 | +1.41 | +2.60 | +3.64 | +8.05 |
| 2 | +1.91 | +4.99 | +9.05 | +9.71 |

Regular (burst) images, 9000 (online − frozen on them):

| h | frozen % | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|---|
| 128 | 97.86 | +0.21 | +0.14 | -0.19 | -2.34 |
| 16 | 95.06 | +0.16 | +0.37 | -0.16 | -4.27 |
| 8 | 92.57 | +0.71 | +1.40 | +0.83 | -3.03 |
| 4 | 85.70 | +1.80 | +2.60 | +1.58 | -5.27 |
| 2 | 64.18 | +4.41 | +6.98 | +7.13 | -3.98 |

Intruders, 1000 (online − frozen on them):

| h | frozen % | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|---|
| 128 | 97.50 | +0.30 | +0.40 | -0.70 | -4.90 |
| 16 | 95.20 | +0.10 | -1.20 | -4.60 | -15.30 |
| 8 | 91.50 | -0.20 | -2.60 | -7.20 | -17.00 |
| 4 | 85.50 | -6.00 | -12.20 | -19.30 | -35.90 |
| 2 | 61.90 | -5.60 | -12.80 | -19.10 | -33.20 |

- With online SGD, small models beat their frozen offline version on bursts. At h = 2 the gain is
  +5.00 pp at lr 0.001.
- lr 0.001 is the best learning rate overall at every size except h = 128, where lr 0.0003 is
  ahead by 0.05 pp. At h = 2, lr 0.003 gains as much on the regular images (+7.13 vs +6.98 pp)
  but loses more on intruders (-19.1 vs -12.8 pp), so it is behind overall.
- The higher the learning rate, the more the model loses on intruders: at h = 2, -5.6 pp at lr
  0.0003 up to -33.2 at lr 0.01. Adapting to the burst costs accuracy on what does not fit it.
- The locality effect grows as the model shrinks and as the learning rate rises, up to +9.71 pp
  (h = 2, lr 0.01).
- At lr 0.01 every model ends up below the frozen one, on the regular images as well: the
  locality effect is largest there only because accuracy on the shuffled stream collapses.

## Previous sequence (v1, no intruders)

The first version of these results used `sequence_v1.csv`, the same bursts without intruders.
They came from an earlier version of the scripts (`results.csv` + `online_sweep.csv`), converted
without rerunning; the current code reproduced them exactly on that sequence.

Online gain on bursts (online bursts − frozen), with the locality effect (bursts − random) in
brackets, in percentage points:

| h | frozen % | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|---|
| 128 | 97.86 | +0.33 (+0.16) | +0.19 (-0.05) | -0.22 (-0.15) | -2.35 (+1.13) |
| 16 | 95.04 | +0.25 (+0.05) | +0.37 (+0.19) | +0.25 (+1.38) | -4.43 (+2.36) |
| 8 | 92.59 | +0.75 (+0.48) | +1.36 (+1.49) | +0.72 (+2.35) | -2.87 (+4.31) |
| 4 | 85.85 | +1.38 (+1.49) | +2.30 (+3.69) | +1.59 (+5.55) | -4.37 (+12.28) |
| 2 | 64.27 | +4.25 (+2.98) | +6.44 (+6.99) | +6.45 (+10.93) | -4.50 (+11.69) |

On v1, lr 0.001 and 0.003 were tied at h = 2 (+6.4 pp) and the locality effect reached +12.28 pp.
The intruders reduce the gains and separate the two learning rates.
