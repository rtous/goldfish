# memory: online learning vs. temporal locality on MNIST

Does a small MLP trained online (one image at a time) exploit temporal locality in its
input stream? The stream is a sequence of 10000 MNIST test images arranged in bursts of
3 digits; a shuffled copy of the same sequence is the no-locality control.

## Run

```bash
python3 make_sequence.py                 # sequence.csv (bursts) + sequence_random.csv (shuffled)
python3 mnist_mlp.py                     # train offline, score the frozen model on both sequences
python3 online_mlp.py                    # online updates on the burst sequence
python3 online_mlp.py --sequence random  # online updates on the shuffled control
python3 show_results.py                  # results table (one row per configuration and sequence)
```

All scripts accept configuration names to run a subset, e.g. `python3 online_mlp.py mlp_2 mlp_4`.
MNIST is downloaded to `data/` on first use. A CUDA GPU is required.

## Files

| file | what it does |
|---|---|
| `make_sequence.py` | builds the burst sequence and its shuffled control |
| `mnist_mlp.py` | offline training; saves `weights/<config>.pt`; shared helpers |
| `online_mlp.py` | predict-then-train on each image, starting from the offline weights |
| `show_results.py` | prints `results.csv` |
| `models/mlp.py` | 784 → hidden_size → 10 MLP |
| `configurations.json` | the seven configurations, h = 128 … 2 |

## Provenance

`mnist_mlp.py`, `online_mlp.py` and `show_results.py` are the original scripts, with one
addition: a `sequence` dimension (`--sequence bursts|random`, a `sequence` column in
`results.csv`) and a fixed training seed in `mnist_mlp.py`.

`make_sequence.py`, `models/mlp.py` and `configurations.json` were reconstructed:

- `models/mlp.py` reproduces the original parameter counts exactly.
- `make_sequence.py` matches the original format (3 digits per burst, images drawn with
  replacement from the test set, ~52 images per burst on average), but not the exact sequence.
  The random control is now a permutation of the burst sequence, so both contain the same
  images and the frozen model scores identically on them.
