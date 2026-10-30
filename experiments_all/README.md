# experiments_all: code and experiments

The code, data and input sequences are shared; each experiment is a subfolder with its own
configurations and outputs. See the [root README](../README.md) for the research question and
each experiment's README for its setup and results.

```
experiments_all/
  mnist_mlp.py        offline training; scores the frozen model; shared helpers
  online_mlp.py       online phase: predict each image, then one training step on it
  show_results.py     prints an experiment's results.csv as a table
  plot_results.py     plots results against model size
  make_sequence.py    builds sequence.csv (bursts with intruders) and sequence_random.csv
                      (same rows shuffled) from sequence_v1.csv (the first version, no intruders)
  models/             mlp.py (784 -> h -> 10); cnn.py (small VGG-style CNN, used by no experiment)
  data/               MNIST (downloaded on first use)
  plots/              cross-experiment comparisons (plot_results.py with several experiments)
  experimentN/
    configurations.json   one entry per configuration
    results.csv           one row per configuration
    weights/<config>.pt   offline weights
    plots/                accuracy.*, locality.*
    README.md             question, setup, results
```

A new subfolder with a `configurations.json` is picked up as a new experiment.

## Usage

Every script takes the experiment first, as a folder name (`experiment2`) or a number (`2`).
Configuration names can follow, also as quoted shell-style patterns. With none, all of the
experiment's configurations are used.

```bash
python3 mnist_mlp.py experiment1                  # offline training of all configurations
python3 mnist_mlp.py experiment1 'mlp_2_*'        # only these
python3 mnist_mlp.py 2 --score-only               # rescore saved weights, no training
python3 online_mlp.py experiment1 'mlp_2_*'       # online phase, both sequences
python3 online_mlp.py 2 mlp_2_sgd --sequence bursts  # one sequence only
python3 show_results.py experiment1               # results table
python3 show_results.py 1 bursts_accuracy         # sorted by a column
python3 plot_results.py experiment1               # experiment1/plots/
python3 plot_results.py 1 2                       # plots/compare_*
python3 experiment2/plot_intruders.py             # experiment 2's extra figure
```

Training and online runs need a CUDA GPU. `make_sequence.py` does not need to be rerun: the
sequences are in the repository. `sequence.csv` is `sequence_v1.csv`
(the first version, without intruders) with 10% of its positions replaced by intruders, a digit that does not appear
in that position's burst (`--intrusion-rate`, default 0.1; `--v1` rebuilds the v1 sequence).
Its `intruder` column is 1 for them. All experiments use it; the v1 files are kept because
`sequence.csv` is built from them.

## configurations.json

```json
{"name": "mlp_2_lr0.001", "model": "mlp", "seed": 0,
 "model_params": {"hidden_size": 2},
 "training_params": {"learning_rate": 0.001, "epochs": 10, "batch_size": 128},
 "online_params": {"optimizer": "sgd", "learning_rate": 0.001}}
```

- `model`, `model_params`: `models/<model>.py`, built with `build_model(**model_params)`.
- `seed` (default 0): seeds both offline training and online runs.
- `training_params`: `learning_rate`, `epochs` and `batch_size` are required. Optional:
  `optimizer` (`adam` default, `adamw`), `weight_decay` (0), `lr_schedule` (`none`, `cosine`),
  `label_smoothing` (0), `augment_shift` (0; random shift of up to N pixels).
- `online_params`: batch size 1, cross-entropy loss. `learning_rate` is required. Optional:
  `optimizer` (`sgd` default, momentum 0; or `adam`), `label_smoothing` (0), and `bn_mode` for
  BatchNorm models: `frozen` (default; running statistics kept, BN weights still trained) or
  `update` (BN in train mode on the single image). `learning_rate` 0 freezes the weights.
  Two more options (see `online_mlp.py`): `fast` (`{"learning_rate", "decay", "layers"}`; the
  listed parameters get a decaying fast part added to them, experiment 2) and `label_prior`
  (rate of a running label prior added to the logits at prediction; learns no weights), with
  `label_prior_mix` m (0; predict with log((1 − m) p + m / 10)) so absent digits stay possible.

Weights are saved per configuration name. To try another online setting for the same model,
add a configuration with a new name, then train it or copy the weights of an identical model.

## results.csv

`configuration, model, model_params, training_params, params, size_bytes, offline_accuracy,
offline_intruder_accuracy, offline_regular_accuracy, test_set_accuracy, online_params,
bursts_accuracy, bursts_intruder_accuracy, bursts_regular_accuracy, random_accuracy,
random_intruder_accuracy, random_regular_accuracy, updated`

- `offline_accuracy`: the frozen model on the sequence images. It is the same for both
  sequences, which hold the same images. `test_set_accuracy`: the frozen model on the plain test set.
- `bursts_accuracy` / `random_accuracy`: online accuracy on each sequence, or `diverged` if the
  loss became NaN/inf.
- `*_intruder_accuracy` / `*_regular_accuracy`: the same accuracy on intruders only and on the
  other images only; empty when the sequence has no intruders. On the random sequence they are
  the same images out of their burst context.
- `mnist_mlp.py` writes the offline columns and clears the online ones (with `--score-only` it
  rewrites the offline columns from the saved weights and keeps the online ones); `online_mlp.py`
  writes the online ones.
- `show_results.py` adds accuracy per KB and the relative change vs batch and vs random. A `*`
  marks a configuration whose `configurations.json` entry changed since its results were measured.

## Plots

`plot_results.py` groups configurations into series by everything except model size (e.g. one
series per online learning rate). It plots them against model size (KB, log scale):

- `accuracy`: one panel per series with the frozen model, online on bursts and online on random.
- `locality`: online gain on bursts (bursts − frozen) and locality effect (bursts − random), in
  percentage points, one line per series.
