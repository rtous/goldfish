# Experiment 1: vanilla temporal locality

**Question.** Do neural networks naturally exploit temporal locality when their capacity is
reduced and they are trained online?

## Setup

- Model: one-hidden-layer MLP, 784 → h → 10, h = 128, 64, 32, 16, 8, 4, 2 (`mlp_<h>`).
- Offline: Adam, lr 0.001, 10 epochs, batch size 128, seed 0.
- Online: Adam, lr 0.001 (the same optimizer and learning rate as offline).

## Results

Accuracy in %; the last two columns are differences in percentage points.

| config | KB | frozen | online bursts | online random | bursts − frozen | bursts − random |
|---|---|---|---|---|---|---|
| mlp_128 | 397.5 | 97.86 | 94.07 | 93.66 | -3.79 | +0.41 |
| mlp_64 | 198.8 | 97.44 | 94.23 | 93.90 | -3.21 | +0.33 |
| mlp_32 | 99.4 | 96.36 | 94.19 | 93.72 | -2.17 | +0.47 |
| mlp_16 | 49.7 | 95.04 | 92.89 | 92.46 | -2.15 | +0.43 |
| mlp_8 | 24.9 | 92.59 | 91.82 | 90.53 | -0.77 | +1.29 |
| mlp_4 | 12.5 | 85.85 | 84.53 | 83.69 | -1.32 | +0.84 |
| mlp_2 | 6.2 | 64.27 | 64.38 | 64.07 | +0.11 | +0.31 |

- Online training does better on bursts than on the shuffled stream at every size, so there is
  a locality effect, largest for small models (h = 8, 4).
- Online Adam at lr 0.001 hurts overall: it only matches the frozen model at h = 2. Experiment 2
  looks for better online settings.

## Notes

These results come from an earlier version of the scripts and were converted without rerunning.
The current code reproduces them exactly. That version also recorded the accuracy on the first
image of each burst, which is no longer stored; it is in the git history.
