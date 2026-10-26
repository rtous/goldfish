# Experiment 4: a modern CNN

Do neural networks exploit temporal locality when trained online? Experiment 2 showed it for a
one-hidden-layer MLP: below some model size, online SGD (predict, then one step per image) on a
bursty sequence beats the frozen batch-trained model, and the gain disappears on a shuffled
copy of the sequence. This experiment repeats it with a small modern CNN.

Models are trained offline on MNIST, then updated online one image at a time on a sequence of
test images: either in bursts of 3 digits (`sequence.csv`) or a shuffle of the same rows
(`sequence_random.csv`). Both files are copies of experiment 2's, so both experiments use the
same sequences.

Key metric, the **locality effect**:

    (online − frozen accuracy on bursts) − (online − frozen accuracy on random)

## Model

`models/cnn.py`, `build_model(width=c)` (no dropout; `dropout` in `model_params` adds it back, default 0):

    [conv3x3(1->c) BN ReLU] [conv3x3(c->c) BN ReLU] maxpool2          # 28 -> 14
    [conv3x3(c->2c) BN ReLU] [conv3x3(2c->2c) BN ReLU] maxpool2       # 14 -> 7
    global average pool -> Linear(2c -> 10)

Convolutions use padding=1 and bias=False. Model size counts parameters and BN buffers
(running mean/var and the batch counter).

| config | width | params | KB | test-set acc% (offline) |
|---|---|---|---|---|
| cnn_8 | 8 | 4370 | 17.5 | 97.31 |
| cnn_4 | 4 | 1182 | 4.8 | 93.80 |
| cnn_2 | 2 | 344 | 1.5 | 76.87 |
| cnn_1 | 1 | 114 | 0.5 | 41.42 |

Offline training takes 1-4 min per configuration on the Jetson; one online run (10000 images)
takes ~85 s, so the online phase takes ~3 min per configuration (two sequences).

## Configurations

Each entry in `configurations.json` is one full experiment: model, offline training and online
settings. Every configuration in the list gets one row in `results.csv`; to drop one, remove it
from the list (its row is dropped on the next write and is never shown). To try another online
learning rate, add a configuration (with a new name: weights are saved per name, so it is
trained again).

```json
{"name": "cnn_4", "model": "cnn", "seed": 0,
 "model_params": {"width": 4},
 "training_params": {"optimizer": "adamw", "learning_rate": 0.001, ...},
 "online_params": {"optimizer": "sgd", "learning_rate": 0.0003, "bn_mode": "frozen", "label_smoothing": 0.1}}
```

Current list: cnn_8, cnn_4, cnn_2, cnn_1, all with online SGD lr 0.0003, BN frozen, label smoothing 0.1.

## Offline training

`training_params` in `configurations.json` (optional keys default to the MLP behaviour):

| key | CNN | MLP / default |
|---|---|---|
| optimizer | adamw | adam |
| learning_rate | 0.001 | 0.001 |
| weight_decay | 1e-4 | 0 |
| epochs | 15 | 10 |
| batch_size | 128 | 128 |
| lr_schedule | cosine (per batch, down to 0) | none |
| label_smoothing | 0.1 | 0 |
| augment_shift | 2 (random shift up to 2 px in x and y, training only) | 0 |

A configuration's top-level `"seed"` (default 0) seeds `torch.manual_seed` for both the offline
training and the online runs.

## Online phase

Plain SGD, momentum 0, batch size 1, no weight decay, cross-entropy loss, no augmentation.
Settings from `online_params`:

| key | meaning |
|---|---|
| learning_rate | required |
| optimizer | `sgd` (the only option; default) |
| bn_mode | `frozen` (default) or `update`, see below |
| label_smoothing | label smoothing of the online loss (default 0) |

- Prediction: `model.eval()` (BN uses running statistics; dropout, if any, off).
- Training step: dropout (if any) on. BatchNorm follows `bn_mode`:
  - `frozen` (default): BN layers in eval mode, so running statistics stay at their offline
    values; BN weight and bias are still trained.
  - `update`: BN layers in train mode: the single image's own statistics normalize it and the
    running statistics are updated.
- `bn_mode` has no effect on the MLPs.

## Scripts

| script | what it does |
|---|---|
| `make_sequence.py` | builds `sequence.csv` and `sequence_random.csv` (not needed: copied from experiment 2) |
| `mnist_mlp.py` | trains the named configurations offline, saves `weights/<config>.pt`, scores the frozen model |
| `online_mlp.py` | online pass on both sequences: predict each image, then one training step on it |
| `show_results.py` | prints `results.csv`, one row per configuration |

Configuration names on the command line can be shell-style patterns, e.g. `'cnn_*'` (quoted);
with no names, all configurations are run.

## Usage

```bash
python3 mnist_mlp.py                     # offline training of all configurations
python3 online_mlp.py                    # online phase of all configurations, both sequences
python3 online_mlp.py cnn_2 --sequence bursts
python3 show_results.py                  # or: show_results.py bursts_accuracy (sort by a column)
```

The offline weights are only read by `online_mlp.py`, never modified.

## Results

`results.csv`, one row per configuration:

`configuration, model, model_params, training_params, params, size_bytes, offline_accuracy,
test_set_accuracy, online_params, bursts_accuracy, random_accuracy, updated`

- `offline_accuracy`: frozen model on the sequence images (the same for both sequences);
  `test_set_accuracy`: on the plain 10000-image test set.
- `bursts_accuracy` / `random_accuracy`: online accuracy on each sequence, or `diverged` if the
  loss became NaN/inf (checked every 100 images; the run stops early).
- `mnist_mlp.py` writes the offline columns and clears the online ones (new weights);
  `online_mlp.py` writes `online_params` and the accuracies. If `online_params` changed since the
  last online run, both accuracies are cleared before the new one is stored.
- `show_results.py` adds accuracy per KB and the relative changes vs batch and vs random, and
  marks with `*` a configuration whose `configurations.json` entry no longer matches its results.

`archive/` holds the outputs of the earlier layout (old `results.csv`, `online_sweep.csv`,
per-image `online_results_*.csv`, `online_runs/`, `sweep.log`) and the removed
`sweep_online.py` and `show_sweep.py`. Nothing reads it; delete it when no longer needed.
