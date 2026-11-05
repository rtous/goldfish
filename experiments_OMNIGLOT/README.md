# experiments_OMNIGLOT: online learning on alphabet bursts

Same setup as `experiments_MNIST2` (code copied and adapted, not imported): a small MLP sees a
stream of Omniglot test images one at a time, predicts each image's character, then may learn from
it. The question is whether an online method beats the frozen model, and, above all, whether it
beats simple controls that only exploit which labels are likely right now.

**Hypothesis.** A small model can't store every detail; here, 1623 character classes. If inputs
come with temporal locality (a burst uses one alphabet), an online learner can keep the general
knowledge (stroke features) and use spare capacity for the details of the current context. Unlike
rotation (MNIST2), the number of details can be raised beyond what the model can hold, so the gap
is not closed by a slightly bigger model.

## Data (`data.py`)

Omniglot background + evaluation sets merged: 50 alphabets, 1623 characters, 20 drawings each (one
per drawer). The two official zips are downloaded to `data/` (gitignored) if missing. Each
105x105 drawing is downscaled to 28x28 (area averaging), inverted (ink = 1), and normalized with
the mean and std of the fit split. Labels 0..1622 in (set, alphabet, character) order; `ALPHABET`
maps each class to its alphabet.

Split by drawer, every class in every split:

| split | drawers | images | used for |
|---|---|---|---|
| fit | 1-10 | 16230 | offline models (mixed, per-alphabet) |
| val | 11-12 | 3246 | tuning the label prior and the cache; Meta-SGD's episodes |
| test | 13-20 | 12984 | the evaluation stream only |

fit + val are the "training drawers" (1-12). The offline models train on fit only, so that tuning
and meta-training see training-drawer images the model hasn't memorized. (Trained on all 12
drawers, the mixed model is near 100% on them, and a prior or cache tuned there would look
useless.) Nothing is ever tuned on the test drawers except where marked "chosen on the evaluation
stream".

## Stream (`make_sequence.py`, `sequences/`)

| stream | positions | structure |
|---|---|---|
| `alphabets` | 12984: every test image once | one burst per alphabet (8 x its characters: 112-440 images), alphabets in random order, images shuffled within a burst (seed 0) |
| `alphabets_random` | 12984 | its shuffled control: the same images in random order |
| `val_alphabets` | 3246 | the same construction on the val drawers (bursts of 28-110), for tuning |

Columns: `position`, `image_index`, `label`, `burst`, `alphabet`, `occurrence` (times the class
was already seen in the current burst: 0 = first sighting, up to 7).

## Models (`models/mlp.py`) and offline training (`offline.py`)

| arch | layers | params per layer | total | KB (float32) |
|---|---|---|---|---|
| A | 784 -> 64 -> 64 -> 1623 | 50240, 4160, 105495 | 159895 | 625 |
| B | 784 -> 62 -> 62 -> 62 -> 1623 | 48670, 3906, 3906, 102249 | 158731 | 620 |

The output layer is two thirds of each model, as expected with 1623 classes.

Training: Adam lr 1e-3, batch 128, seed 0, light augmentation (random rotation +-10 degrees, scale
0.9-1.1, shift +-2 pixels; offline training only, identical for every model; stream images are
never augmented). Stopping by training loss: rounds of 25 epochs (at least 500 steps), stop when a
round's mean loss is above 95% of the previous one (A: 225 epochs, B: 250).

**Width sweep** (architecture A, 150 epochs with augmentation: `offline.py --sweep`; test
accuracy used only to pick a width in the requested 30-60% range). The no-augmentation column is
from a one-off scratch run (100 epochs, best of the epochs printed), not reproduced by the script:

| widths | params | no augmentation (best) | with augmentation |
|---|---|---|---|
| 32, 32 | 79735 | | 41.7 |
| 64, 64 | 159895 | 26.4 | 52.2 |
| 128, 128 | 326359 | 29.9 | 58.0 |
| 256, 256 | 683863 | 32.9 | 60.6 |

Without augmentation every width memorized the 10 training drawings (training loss near 0) and
stopped at 26-33%: the limit was data, not capacity, which would not test the hypothesis. With
augmentation accuracy grows with width, so w = 64 is capacity-limited: doubling it gains 6 pp. B's
w' = 62 matches A's parameter count.

**Offline results** (test drawers, averaged over images, i.e. weighted like the stream):

| arch | mixed | mixed, masked to the true alphabet | per-alphabet ceiling |
|---|---|---|---|
| A | 53.19 | 78.84 | 73.87 |
| B | 52.13 | 78.54 | 72.91 |

The per-alphabet ceiling is one model per alphabet (same architecture, output restricted to that
alphabet's classes), trained only on that alphabet's fit images, same settings. It lands 5 pp
*below* the masked mixed model: with 10 drawings per character, a model that knows the alphabet
but learns only from it is worse than the general model told which labels are possible. So the
part of the gap that label locality can't explain is negative here: on this data, the whole
context gain available is label locality.

## Commands, in order (from this folder)

```
../venv/bin/python make_sequence.py     # streams (data.py downloads the data on first use)
../venv/bin/python offline.py           # mixed models + per-alphabet ceilings (about 45 min)
../venv/bin/python tune.py 1            # label prior and cache, tuned on val_alphabets
../venv/bin/python meta_train.py 1      # Meta-SGD checkpoints (15 min cap per config and arch)
../venv/bin/python online.py 1          # every config x arch, bursts and shuffled
../venv/bin/python show_results.py 1    # table
../venv/bin/python plot.py 1            # plots/burst_curve, plots/occurrence
```

| experiment | question | answer |
|---|---|---|
| [experiment1](experiment1/README.md) | does any online method beat the label prior and the cache, and how close does it get to the alphabet mask and the per-alphabet ceiling? | no: the label prior (a decaying label count) beats every online method (+10.7 / +15.6 pp vs at most +8.0 / +7.2); online gains come from repeat sightings while first sightings get worse; the alphabet mask (+26 pp) is above even the per-alphabet ceiling, so the available context gain is label locality |
