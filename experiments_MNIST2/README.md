# experiments_MNIST2: context streams (rotated and permuted MNIST)

In `experiments_MNIST/` the stream's locality was only in the labels (bursts of 3 digits), so a
running label prior that learns no weights beat every online method. Here the locality is in the
inputs. Each burst of the stream comes from a **context**, a fixed transformation of the images
(a rotation angle, or a pixel permutation), and holds all 10 digits in uniform proportions. A
label prior cannot help, and a model can only gain by adapting how it reads the images.

The folder is self-contained: it imports nothing from `experiments_MNIST/`, whose code it copies
and adapts. The code, data and input sequences are shared; each experiment is a subfolder with
its own configurations and outputs. See the [root README](../README.md) for the research question
and each experiment's README for its setup and results.

```
experiments_MNIST2/
  contexts.py         the context families and their deterministic transforms
  mnist_mlp.py        offline training on the mix of all contexts; scores the frozen model; helpers
  headroom.py         offline headroom check: mixed model vs one model per context (the ceiling)
  online_mlp.py       online phase: predict each image, then one training step on it
  tune.py             tunes the online settings on the tuning streams
  show_results.py     prints an experiment's results.csv as a table
  plot_results.py     plots results against model size, one set of figures per stream
  plot_sweep.py       plots a burst-length sweep (experiment 2) against the burst length
  make_sequence.py    builds the context-burst streams and their shuffled controls
  models/             mlp.py (784 -> h -> 10)
  data/               MNIST (copied from experiments_MNIST/data)
  sequence_<family>.csv, sequence_<family>_random.csv            evaluation stream and control
  sequence_<family>_tune.csv, sequence_<family>_tune_random.csv  tuning stream and control
  experimentN/
    configurations.json   one entry per configuration
    results.csv           one row per configuration
    headroom.csv          headroom.py: mixed vs per-context accuracy per family and size
    headroom_curves.csv   headroom.py: test accuracy after every epoch
    tuning.csv            tune.py: online gain of every tried setting on the tuning stream
    sequences/            the experiment's own streams, if its configurations name one
                          (experiment 2: sequence_L<L>*.csv)
    weights/<config>.pt   offline (mixed) weights
    weights/percontext/   headroom.py's per-context models (the oracle)
    plots/                contexts_*, headroom*, accuracy_<stream>.*, locality_<stream>.*
    README.md             question, setup, results
```

A new subfolder with a `configurations.json` is picked up as a new experiment.

## Contexts

`contexts.py` defines two families of `n_contexts` contexts each (default 8). Every context is
deterministic given the family, its index k and the seed:

- `rotated`: context k rotates the raw 28×28 image by k · 180 / n_contexts degrees
  counterclockwise (0, 22.5, …, 157.5 for 8), with `F.affine_grid` + `F.grid_sample`
  (bilinear, black outside the image), before normalization.
- `permuted`: context k reorders the 784 pixels with a fixed random permutation from
  `numpy.random.default_rng(seed + k)`. Context 0 is also a random permutation, not the identity.

`python3 contexts.py 1` saves a grid of a few test images under every context to
`experiment1/plots/contexts_<family>.png`.

## Streams

`make_sequence.py` builds, for each family, a stream of 10000 positions that uses every MNIST test
image exactly once, in a random order (no repeated images; uniform labels everywhere). The
positions are split into bursts of 20 to 200 images (uniform; the last burst is cut at the end).
Each burst gets a random context, never the previous burst's, so contexts recur. With the
defaults (8 contexts, seed 0) the stream has 88 bursts of mean length 113.6, and each context
starts 4 to 14 bursts. The bursts and contexts depend only on the seed and the number of
contexts, so both families get the same images, bursts and context indices.

- `sequence_<family>.csv`: the evaluation stream. Columns: `position, image_index, label, burst,
  context`.
- `sequence_<family>_random.csv`: the same rows randomly permuted, `burst` = -1. It is only a
  control: the same images under the same contexts, without temporal locality, to check that
  online gains come from the locality and not merely from training on more data.
- `sequence_<family>_tune.csv` and `sequence_<family>_tune_random.csv`: the same procedure with
  another seed (`--tune-seed`, default 1; 82 bursts). Used only by `tune.py`, so the online
  settings are not tuned on the stream they are scored on.

The sequences are in the repository and the defaults of `make_sequence.py` rebuild them exactly.

### Long streams (experiment 2)

`make_sequence.py --burst-lengths 100 500 1000 2500 5000 --out-dir experiment2/sequences` builds,
for each burst length L, streams of 80000 positions where every test image appears once under
every context: each context has its own random order of the 10000 test images and uses them in
that order whenever it is active. Every burst has exactly L images, so each context gets
10000 / L bursts, in random order and never twice in a row. An extra column `first_seen` is 1 at
an image's first appearance in the stream (under any context). Files: `sequence_L<L>.csv`,
`_random`, `_tune` and `_tune_random` (same seeds as above). They hold only context indices, so
one set serves both families; a configuration selects it with `"sequence": "L<L>"` and the family
with `"stream"`. The same options rebuild them exactly.

## Usage

Every script takes the experiment first, as a folder name (`experiment1`) or a number (`1`).
Configuration names can follow, also as quoted shell-style patterns. With none, all of the
experiment's configurations are used.

```bash
python3 make_sequence.py                          # rebuild the sequences (not needed)
python3 contexts.py 1                             # plots of the transformed images
python3 headroom.py 1                             # headroom check (both families, all sizes)
python3 headroom.py 1 --families rotated --sizes 8 4 2 --epochs 50
python3 mnist_mlp.py 1                            # offline training of all configurations
python3 mnist_mlp.py 1 'rotated_2_*'              # only these
python3 mnist_mlp.py 1 --score-only               # rescore saved weights (e.g. after headroom.py)
python3 tune.py 1 rotated sgd                     # tune online SGD's lr on the tuning stream
python3 tune.py 1 rotated fasthidden --slow-lr 0.003   # tune a fast variant's lr and decay
python3 tune.py 1 --report                        # tuning tables from tuning.csv
python3 online_mlp.py 1 'rotated_2_*'             # online phase, both sequences
python3 online_mlp.py 1 rotated_2_sgd --sequence bursts  # one sequence only
python3 show_results.py 1                         # results table
python3 plot_results.py 1                         # experiment1/plots/
python3 make_sequence.py --burst-lengths 100 500 1000 2500 5000 --out-dir experiment2/sequences
python3 tune.py 2 rotated_L1000 sgd --sizes 8 4 2 # tune per configuration prefix
python3 plot_sweep.py 2                           # experiment2/plots/sweep_*, burst_curves
```

Order: `headroom.py` (it saves the per-context models that `mnist_mlp.py` scores as the oracle),
`mnist_mlp.py`, `tune.py`, then `online_mlp.py`. Training and online runs use the CUDA GPU if
there is one, else the CPU (an online run takes about a second on a CPU; `headroom.py` trains 9 models per
family and size, tens of minutes in all at 50 epochs; `tune.py` always uses the CPU).

## configurations.json

```json
{"name": "rotated_2_fasthidden", "model": "mlp", "seed": 0, "stream": "rotated",
 "model_params": {"hidden_size": 2},
 "training_params": {"learning_rate": 0.001, "epochs": 50, "batch_size": 128},
 "online_params": {"optimizer": "sgd", "learning_rate": 1e-05,
                   "fast": {"learning_rate": 0.001, "decay": 0.99, "layers": ["0.weight", "0.bias"]}}}
```

- `stream` (required): the context family, `rotated` or `permuted`. It selects the contexts of
  offline training and the sequences used online. `n_contexts` (default 8) must match the
  sequence files.
- `sequence` (optional, e.g. `"L1000"`): use the experiment's own `sequences/sequence_L1000*.csv`
  instead of `sequence_<stream>*.csv` next to the scripts.
- `model`, `model_params`: `models/<model>.py`, built with `build_model(**model_params)`.
- `seed` (default 0): seeds both offline training and online runs.
- `training_params`: `learning_rate`, `epochs` and `batch_size` are required. Optional:
  `optimizer` (`adam` default, `adamw`), `weight_decay` (0), `lr_schedule` (`none`, `cosine`),
  `label_smoothing` (0). Training is always on the mix of the stream's contexts: each sampled
  training image is shown under a context drawn uniformly at random.
- `online_params`: batch size 1, cross-entropy loss. `learning_rate` is required (0 freezes the
  weights). Optional: `optimizer` (`sgd` default, momentum 0; or `adam`), `label_smoothing` (0),
  `fast` (`{"learning_rate", "decay", "layers"}`: the listed parameters get a decaying fast part
  added to them) and `label_prior` (rate of a running label prior added to the logits at
  prediction; learns no weights). See `online_mlp.py`.

Configurations with the same model, stream, seed and training_params get the same offline weights:
`mnist_mlp.py` trains the model once and saves it under each of their names.

## results.csv

`configuration, stream, model, model_params, training_params, params, size_bytes,
offline_accuracy, test_set_accuracy, oracle_accuracy, online_params, bursts_accuracy,
random_accuracy, bursts_pos1_10, bursts_pos11_50, bursts_pos51plus, bursts_seen_first20,
bursts_new_first20, bursts_retention, bursts_retention_slow, updated`

Offline columns (`mnist_mlp.py`):

- `offline_accuracy`: the frozen (mixed) model on the sequence, each image under its position's
  context. It is the same for both sequences, which hold the same images under the same contexts.
- `test_set_accuracy`: the frozen model on all 10000 test images under every context, averaged
  over the contexts.
- `oracle_accuracy`: the per-context ceiling of `headroom.py` (one model per context, same size)
  on the bursts sequence, each position scored by the model of its context. Blank until
  `headroom.py` has run (then `mnist_mlp.py --score-only`).

Online columns (`online_mlp.py`):

- `bursts_accuracy` / `random_accuracy`: online accuracy on each sequence, or `diverged`.
- `bursts_pos1_10`, `bursts_pos11_50`, `bursts_pos51plus`: online accuracy on bursts at positions
  1-10, 11-50 and 51+ within each burst (how fast the model adapts).
- `bursts_seen_first20`, `bursts_new_first20`: online accuracy on the first 20 images of bursts
  whose context occurred in an earlier burst, and of bursts whose context is new (8 bursts, 160
  images).
- `bursts_retention`: the weights at the end of the bursts stream (slow + fast parts), frozen, on
  all test images under every context, averaged; compare with `test_set_accuracy`.
  `bursts_retention_slow`: the slow part only.

Configurations with a `sequence` field add these columns (bursts sequence; blank when a bin is
empty): `sequence`; `bursts_bin1_100`, `bursts_bin101_500`, `bursts_bin501_1000`,
`bursts_bin1001_2500`, `bursts_bin2501plus` (online accuracy by position within the burst);
`bursts_first_seen`, `bursts_repeated` and their `_frozen` counterparts (online and frozen accuracy
on images at their first appearance in the stream vs images already seen under another context);
`bursts_first100_firstburst`, `bursts_first100_returning` (online accuracy on the first 100 images
of each context's first burst vs of bursts whose context has occurred before). A results.csv
without such configurations keeps exactly the columns above.

`mnist_mlp.py` writes the offline columns and clears the online ones (with `--score-only` it
rewrites the offline columns from the saved weights and keeps the online ones). `show_results.py`
adds accuracy per KB and the relative change vs batch and vs random. A `*` marks a configuration
whose `configurations.json` entry changed since its results were measured.

## Plots

`plot_results.py` groups configurations into series by everything except model size (e.g. one
series per online variant), with separate figures per stream, against model size (KB, log scale):

- `accuracy_<stream>`: one panel per series with the frozen model, the per-context oracle, online
  on bursts and online on random.
- `locality_<stream>`: online gain on bursts (bursts − frozen) and locality effect (bursts −
  random), in percentage points, one line per series.

`headroom.py` draws `headroom.*` (mixed vs per-context against model size) and
`headroom_curves.png` (test accuracy per epoch). `plot_sweep.py` draws, for a burst-length
sweep, `sweep_gain.*` and `sweep_locality.*` (against L, one panel per family and h, with the
per-context ceiling) and `burst_curves.*` (accuracy against position within the burst at the
longest L).
