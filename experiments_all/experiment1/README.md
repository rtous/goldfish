# Experiment 1: online learning rate

**Question.** Do small MLPs exploit temporal locality when they keep learning online, one image at
a time, with plain SGD? Which online learning rate does it best?

## Setup

- Sequence: `sequence.csv`, 10000 MNIST test images in 193 bursts of 3 digits (see
  `make_sequence.py`). `sequence_random.csv` holds the same rows shuffled.
  It is a control, not a scenario we expect in practice: it holds the same images, so it tells
  whether an online gain comes from temporal locality or merely from training on more data. Real
  input streams are assumed to have temporal locality, not to be random.
- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 16, 8, 4, 2.
- Offline: Adam, lr 0.001, 10 epochs, batch size 128, seed 0.
- Online: plain SGD (momentum 0), lr = 0.0003, 0.001, 0.003, 0.01.
- One configuration per size and learning rate: `mlp_<h>_lr<lr>` (20 in all). Configurations
  of the same size share copies of the same offline weights.

## Results

Differences in percentage points against the frozen model on the same images.

Online gain on bursts (online bursts − frozen):

| h | KB | frozen % | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|---|---|
| 128 | 397.5 | 97.86 | +0.33 | +0.19 | -0.22 | -2.62 |
| 16 | 49.7 | 95.04 | +0.25 | +0.37 | +0.25 | -4.22 |
| 8 | 24.9 | 92.59 | +0.75 | +1.36 | +0.72 | -2.88 |
| 4 | 12.5 | 85.85 | +1.38 | +2.30 | +1.59 | -4.28 |
| 2 | 6.2 | 64.27 | +4.25 | +6.44 | +6.45 | -4.50 |

Locality effect (online bursts − online random):

| h | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|
| 128 | +0.16 | -0.05 | -0.15 | +0.87 |
| 16 | +0.05 | +0.19 | +1.38 | +2.49 |
| 8 | +0.48 | +1.49 | +2.35 | +4.30 |
| 4 | +1.49 | +3.69 | +5.55 | +12.58 |
| 2 | +2.98 | +6.99 | +10.93 | +11.70 |

Shuffled control (online random − frozen):

| h | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|
| 128 | +0.17 | +0.24 | -0.07 | -3.49 |
| 16 | +0.20 | +0.18 | -1.13 | -6.71 |
| 8 | +0.27 | -0.13 | -1.63 | -7.18 |
| 4 | -0.11 | -1.39 | -3.96 | -16.86 |
| 2 | +1.27 | -0.55 | -4.48 | -16.20 |

Plots: `plots/accuracy.*` and `plots/locality.*` (`plot_results.py 1`).

- With online SGD, small models beat their frozen offline version on bursts. At h = 2 the gain is
  +6.44 pp at lr 0.001 (64.27% → 70.71%), while on the shuffled stream the same run stays near
  the frozen model (63.72%).
- lr 0.001 gives the largest gain at h = 16, 8 and 4. At h = 2, lr 0.003 is level with it
  (+6.45 vs +6.44 pp). At h = 128, lr 0.0003 is ahead (+0.33 vs +0.19 pp).
- The locality effect grows as the model shrinks and as the learning rate rises, up to
  +12.58 pp (h = 4, lr 0.01).
- At lr 0.01 every model ends up below the frozen one: the locality effect is largest there
  only because accuracy on the shuffled stream collapses (-16 to -17 pp at h ≤ 4).
- The large model (h = 128) shows almost no locality effect at learning rates up to 0.003: with
  enough capacity, the frozen model already covers the whole distribution.

## Notes

The online runs in `results.csv` were made on a CPU (`get_device` falls back to it when there is
no CUDA GPU), from the saved offline weights. They match the earlier runs on this sequence exactly,
except at lr 0.01, where they differ by up to 0.3 pp: at that rate small floating-point
differences grow over the 10000 steps.
