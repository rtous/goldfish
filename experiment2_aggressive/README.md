# Experiment 2: more aggressive online learning

Do neural networks exploit temporal locality when trained online? MLPs are trained offline on
MNIST, then updated online one image at a time on a sequence of test images: either in bursts
of 3 digits (`sequence.csv`) or a shuffle of the same rows (`sequence_random.csv`).

Key metric, the **locality effect**:

    (online − frozen accuracy on bursts) − (online − frozen accuracy on random)

This experiment makes the online optimizer and learning rate configurable, separately from the
offline `training_params`, so the online phase can learn and forget more aggressively.

## Scripts

| script | what it does |
|---|---|
| `make_sequence.py` | builds `sequence.csv` and `sequence_random.csv` |
| `mnist_mlp.py` | trains every configuration in `configurations.json` offline, saves `weights/<config>.pt`, scores the frozen model on the sequences (`results.csv`) |
| `online_mlp.py` | online pass: predict each image, then one training step on it |
| `sweep_online.py` | runs `online_mlp.py`'s run for a grid of optimizers and learning rates |
| `show_results.py` | prints `results.csv`, plus each non-default online setting from `online_sweep.csv` as an extra row (`--no-sweep` to hide them) |
| `show_sweep.py` | prints the online sweep: online − frozen per sequence, locality effect, best lr |

## Usage

```bash
python3 mnist_mlp.py                       # offline training (already done: weights/)
python3 online_mlp.py                      # online, default settings: Adam, config's learning_rate
python3 online_mlp.py mlp_2 --sequence bursts --online-optimizer sgd --online-lr 0.1
python3 show_results.py

# Sweep: every configuration x both sequences x
#   adam: 0.001 0.003 0.01 0.03    sgd (momentum 0): 0.001 0.003 0.01 0.03 0.1 0.3
python3 sweep_online.py                    # 140 runs, roughly 80-90 min on the Jetson
python3 sweep_online.py mlp_2 mlp_4 --adam-lrs 0.001,0.03 --sgd-lrs 0.01,0.3
python3 sweep_online.py --sgd-lrs ""       # Adam only
python3 show_sweep.py                      # or: python3 show_sweep.py mlp_2 mlp_4
```

`online_mlp.py` options:

- `--online-lr FLOAT`: online learning rate (default: the configuration's `learning_rate`)
- `--online-optimizer adam|sgd`: default `adam`; `sgd` is plain SGD with momentum 0
- `--sequence bursts|random`: run only one sequence (default: both)

The offline weights are only read, never modified.

## Outputs

- `online_sweep.csv`: one row appended per online run (from `online_mlp.py` or the sweep):
  `configuration, sequence, optimizer, online_lr, frozen_accuracy, online_accuracy,
  first_in_burst_accuracy, rest_of_burst_accuracy, status, diverged_at, updated`.
  `frozen_accuracy` is the loaded model on the same sequence before any update. The burst
  columns are empty for the random sequence. `show_sweep.py` uses the latest row for each
  (configuration, sequence, optimizer, lr).
- Divergence: if the loss becomes NaN/inf (checked every 100 images) the run stops early and is
  recorded with `status = diverged`, `diverged_at` = sequence position (0-based) of the first
  bad loss, and empty online accuracies. The sweep carries on.
- `results.csv` and `online_results_<config>_<sequence>.csv` are updated only by runs with the
  default online settings (Adam, config `learning_rate`), so `show_results.py` keeps showing
  the original setup. Per-image results of other settings go to
  `online_runs/online_results_<config>_<sequence>_<optimizer>_lr<lr>.csv`.
