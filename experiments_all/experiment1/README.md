# Experiment 2: more aggressive online learning

**Question.** Assuming experiment 1's locality effect holds, can different online learning rates
exploit it better?

## Setup

- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 16, 8, 4, 2.
- Offline: Adam, lr 0.001, 10 epochs, batch size 128, seed 0. The weights are the same as
  experiment 1's.
- Online: plain SGD (momentum 0), lr = 0.0003, 0.001, 0.003, 0.01.
- One configuration per size and learning rate: `mlp_<h>_lr<lr>` (20 in all). Configurations
  of the same size share copies of the same offline weights.

## Results

Online gain on bursts (online bursts − frozen), in percentage points:

| h | KB | frozen % | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|---|---|
| 128 | 397.5 | 97.86 | +0.33 | +0.19 | -0.22 | -2.35 |
| 16 | 49.7 | 95.04 | +0.25 | +0.37 | +0.25 | -4.43 |
| 8 | 24.9 | 92.59 | +0.75 | +1.36 | +0.72 | -2.87 |
| 4 | 12.5 | 85.85 | +1.38 | +2.30 | +1.59 | -4.37 |
| 2 | 6.2 | 64.27 | +4.25 | +6.44 | +6.45 | -4.50 |

Locality effect (online bursts − online random), in percentage points:

| h | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 |
|---|---|---|---|---|
| 128 | +0.16 | -0.05 | -0.15 | +1.13 |
| 16 | +0.05 | +0.19 | +1.38 | +2.36 |
| 8 | +0.48 | +1.49 | +2.35 | +4.31 |
| 4 | +1.49 | +3.69 | +5.55 | +12.28 |
| 2 | +2.98 | +6.99 | +10.93 | +11.69 |

- With online SGD, small models beat their frozen offline version on bursts. At h = 2 the gain is
  about +6.4 pp (lr 0.001–0.003).
- The locality effect grows as the model shrinks and as the learning rate rises.
- At lr 0.01 every model ends up below the frozen one: the locality effect is largest there, but
  only because accuracy on the shuffled stream collapses.
- The best trade-off is lr 0.001–0.003.

## Notes

These results come from an earlier version of the scripts (`results.csv` + `online_sweep.csv`)
and were converted without rerunning. The current code reproduces them exactly.
