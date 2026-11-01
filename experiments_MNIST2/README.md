# experiments_MNIST2: online learning on context streams

Small MLPs (784 -> h -> 10, h = 8 and 4) see a stream of MNIST test images, one at a time. They
predict each image's digit, then may learn from it. The locality is in the inputs, not in the
labels: the stream comes in bursts, and every image of a burst is shown under the same context.
The only question for now is whether an online method beats the **frozen mixed model**.

**Contexts** (`contexts.py`): 8 rotations, by k * 22.5 degrees (0 to 157.5), bilinear.

**Streams** (`sequences/`, rebuilt exactly by `make_sequence.py`), each with a shuffled control
(`_random`: the same images and contexts in random order, so there is no locality):

| stream | positions | bursts |
|---|---|---|
| `short` | 10000: every test image once | 20-200 images (uniform), a random new context each |
| `L1000` | 80000: every test image once under every context | exactly 1000 images, 10 per context |

**Baseline** (`offline.py`, `weights/mixed_h<h>.pt`): the mixed model, trained offline on the
training set with a random context per sampled image (Adam, lr 0.001, batch 128, 50 epochs, seed
0). It is the frozen reference of every run, and the starting point of `sgd` and `dual`. For
scale: per-context models (one per rotation, same size) reached 93.26% (h = 8) and 87.31% (h = 4)
on the test set, against 74.05% and 64.78% for the mixed model (averaged over contexts), so there
is roughly +19 to +23 pp to gain.

| experiment | method | question |
|---|---|---|
| [experiment1](experiment1/README.md) | `sgd` | plain online SGD at a few learning rates |
| [experiment2](experiment2/README.md) | `dual` | slow + fast weights at a few settings |
| [experiment3_metaSGD](experiment3_metaSGD/README.md) | `metasgd` | meta-learned fast learning rates and decays |

## Configurations

Every experiment's `configurations.json` has the same schema:

```json
{
  "sizes": [8, 4],
  "streams": ["short", "L1000"],
  "seed": 0,
  "configurations": [
    {"name": "sgd_1e-4", "method": "sgd", "lr": 1e-4}
  ]
}
```

All methods run in one loop (`online.py`). The network predicts with S + F (F starts at 0). After
each image, with g the gradient of its loss at S + F: S <- S - lr * g, then
F <- decay * F - fast_lr * g on the fast parameters.

- `sgd` `{"lr"}`: no fast part.
- `dual` `{"lr", "fast_lr", "decay", "fast_layers"}`, with `fast_layers` either `"hidden"`
  (0.weight, 0.bias) or `"all"`.
- `metasgd` `{"meta": {...}}`: lr = 0, a fast part on every parameter, and S, fast_lr (alpha)
  and decay (lambda) are per-parameter tensors meta-learned by `meta_train.py`.

## Commands, in order (from this folder)

```
../venv/bin/python offline.py           # mixed models (--check: score the saved ones; 74.05 / 64.78)
../venv/bin/python meta_train.py 3      # Meta-SGD checkpoints (about 1 h; experiment 3 only)
../venv/bin/python online.py 1          # every config x size x stream, bursts and shuffled
../venv/bin/python show_results.py 1    # compact table
../venv/bin/python plot.py 1            # plots/summary, plots/burst_curve_<stream>
../venv/bin/python plot.py --compare 1 2 3   # plots/compare*: the best config of each experiment
```

## results.csv

One row per config x h x stream, the same columns in every experiment:

| column | meaning |
|---|---|
| `config`, `method`, `h`, `stream` | what ran |
| `frozen` | the mixed model on the stream (%) |
| `bursts`, `shuffled` | online accuracy on the stream and on its shuffled control (%) |
| `gain` | bursts - frozen (pp) |
| `locality` | bursts - shuffled (pp): how much of the result comes from the bursts |
| `pos1_10` ... `pos201_1000` | online accuracy by position within the burst (blank if no burst is that long) |
| `diverged` | true if a NaN or inf loss stopped a run |

`curves/<config>_h<h>_<stream>.npz` holds the per-position correctness on the bursts stream
(online and frozen), for the plots.
