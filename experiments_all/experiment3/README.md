# Experiment 3: a small modern CNN

**Question.** Does the locality effect found for MLPs (experiment 2) also hold for a small
modern CNN? (Formerly `experiment4_cnn`; an experiment in between was discarded.)

## Setup

- Model: `models/cnn.py` with width c = 8, 4, 2, 1 (`cnn_<c>`):

      [conv3x3(1->c) BN ReLU] [conv3x3(c->c) BN ReLU] maxpool2          # 28 -> 14
      [conv3x3(c->2c) BN ReLU] [conv3x3(2c->2c) BN ReLU] maxpool2       # 14 -> 7
      global average pool -> Linear(2c -> 10)

  The convolutions have no bias. Model size counts parameters and BN buffers.
- Offline: AdamW, lr 0.001, weight decay 1e-4, 15 epochs, batch size 128, cosine schedule,
  label smoothing 0.1, random shifts of up to 2 px, seed 0. Takes 1–4 min per configuration on
  the Jetson.
- Online: plain SGD, lr 0.0003, label smoothing 0.1, BN `frozen`: running statistics keep their
  offline values; BN weights are still trained. BN `update` (train mode on each single image)
  was tried first and did not work. One online run takes ~85 s.

## Results

Accuracy in %; the last two columns are differences in percentage points.

| config | KB | frozen | online bursts | online random | bursts − frozen | bursts − random |
|---|---|---|---|---|---|---|
| cnn_8 | 17.5 | 98.08 | 98.18 | 97.03 | +0.10 | +1.15 |
| cnn_4 | 4.8 | 94.36 | 93.27 | 89.76 | -1.09 | +3.51 |
| cnn_2 | 1.5 | 81.11 | 79.21 | 75.64 | -1.90 | +3.57 |
| cnn_1 | 0.5 | 42.14 | 31.35 | 13.14 | -10.79 | +18.21 |

- There is a locality effect at every width, growing as the model shrinks. At c = 1 it comes
  mostly from the collapse on the shuffled stream.
- Online learning does not beat the frozen model: about equal at c = 8, worse below. This is
  like experiment 1 rather than experiment 2. Only one online learning rate (0.0003) was tested.
- The earlier BN `update` runs collapsed: online accuracy fell 6.9 to 38.9 pp below the frozen
  model, with no locality effect for c ≤ 4.
