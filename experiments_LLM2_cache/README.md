# experiments_LLM2_cache: online learning as a cache

A small language model that keeps learning online during inference might use part of its weights
like a cache of the current context: material seen recently, beyond the attention window, is
predicted better, and new material gradually evicts old material. These tests measure that on
Pythia models and PG-19 books. They are an exploratory first look: one seed, quick learning-rate
picks, no sweeps.

## Data (`data.py`)

PG-19 test books. The Hugging Face dataset `deepmind/pg19` holds only a loading script, which
`datasets` 5 no longer runs, so `data.py` does what the script does for the test split only: it
reads the list of test files from the dataset repo, sorts it (the order the dataset gives), and
downloads only the books it needs from the bucket the script uses
(`storage.googleapis.com/deepmind-gutenberg`). The training set is never downloaded.

- Books under 20k tokens are skipped (12204, 26618 and 31974, single short stories). The first 25
  remaining books, in dataset order, form the stream; the next 2 are the **check books** used for
  every learning-rate pick.
- Each book is tokenized on its own (no special tokens) and cut to its first 20k tokens.
- Downloads, model files and token caches go to `data/` (gitignored; 264 MB with the three models).

## Models (`models.py`)

`EleutherAI/pythia-14m` (14067712 parameters, revision `main`, commit `cf967c0a9a04`) for tests 1-4
and most of 5-6; `EleutherAI/pythia-31m` (30494720) and `EleutherAI/pythia-70m` (70426624) in tests
5-6 (same tokenizer, checked). All three have 6 layers (width 128, 256, 512) and untied input and
output embeddings; float32, CPU, seed 0.

## Protocol (`online.py`)

- **Chunks:** the stream is cut into non-overlapping chunks of L = 256 tokens, each scored on its
  own (no KV cache or context from earlier chunks). Every token after a chunk's first is scored.
  Anything that helps beyond 256 tokens can only come through the weights.
- **Online (prequential):** for each chunk, compute the loss (that is what is scored), then one
  plain SGD step on that same loss (mean over the chunk), no momentum, no weight decay. By default
  all parameters; some runs update only a subset (output layer, MLPs, attention, single layers).
- **Frozen:** the same stream with no updates.
- **Learning rates:** picked on the 2 check books (lowest mean loss). Pythia-14M, all parameters:
  frozen 4.5474, lr 1e-4 **4.4775**, 3e-4 4.4787, 1e-3 4.7059 (`lr_check.csv`). From test 6 on, a
  pick at the edge of its grid extends the grid once by one step.

## Metrics

- **Gain** = frozen loss - run loss, in nats/token, over the same tokens (each model against its
  own frozen loss).
- **Distance bins:** for each scored token, the distance to the previous occurrence of the same
  token id anywhere earlier in the stream: *same chunk* (attention can see it), *earlier chunk*
  < 1k, 1k-10k, 10k-100k, >= 100k, or *never seen*. The 1000 most frequent token ids in the stream
  are left out of the bins; *all tokens* has no exclusion.
- **Same book / earlier book:** each earlier-chunk bin split by whether the previous occurrence is
  in the token's own book or in an earlier one (books are 20k tokens, so same book is never more
  than 20k back).
- **Locality** = gain in order - gain with the chunks shuffled (one fixed random order, seed 0):
  the part of the gain that needs recently seen material.
- **Memory of A** (eviction test): loss on a book's second half A2 after reading a different book's
  first half B1, minus the loss after reading A's own first half A1, with I tokens of other books in
  between.

Numbers in the READMEs are rounded from the CSV files: gains to 3 decimals, losses as saved
(4 decimals).

## Commands (from this folder)

```
../venv/bin/python data.py               # download + tokenize (also done on first use), list the books
../venv/bin/python online.py             # learning-rate check -> lr_check.csv (30 s)
../venv/bin/python test1_distance.py     # test 1 (2.5 min)
../venv/bin/python test2_eviction.py     # test 2 (4 min); `test2_eviction.py plot` redraws the plot
../venv/bin/python test3_baselines.py    # test 3, needs tests 1-2 (7 min); `report` redoes tables and plots
../venv/bin/python test4_checks.py       # test 4, needs tests 1 and 3 (4 min); `report` redoes table and plot
../venv/bin/python test5_capacity.py     # test 5, needs tests 1, 3 and 4 (11 min); `report` redoes tables and plots
../venv/bin/python test6.py              # test 6, needs tests 1, 4 and 5 (21 min); `report` redoes tables and plots
```

Files: `data.py` (PG-19), `models.py` (model loading), `online.py` (chunked frozen and online
scoring, parameter subsets, chunk order, learning-rate check), `plots.py` (plot style), one script
per test. Each test folder has a README, its CSV results, plots and `losses.npz` (per-token losses
of its new runs in stream order; token ids, books and positions are in
`test1_distance/losses.npz`): 5-21 MB each, not gitignored.

## Tests

| test | question | answer |
|---|---|---|
| [test1_distance](test1_distance/README.md) | does the online gain extend beyond the 256-token window, and how far? | yes, up to about 10k tokens: for rare tokens last seen in an earlier chunk, +0.301 under 1k tokens back, +0.143 at 1k-10k, +0.010 at 10k-100k, -0.035 beyond 100k; overall +0.099. Tokens repeated within the chunk get worse (-0.107) |
| [test2_eviction](test2_eviction/README.md) | does a book's memory get evicted by other books, and how fast? | slowly: memory of A is +0.038, +0.027 and +0.019 after 0, 10k and 100k tokens of other books (matched comparison). The A1 run's flat gain on A2 is consistent both with keeping A while the B1 run catches up and with losing A while gaining general adaptation; this test cannot tell them apart |
| [test3_baselines](test3_baselines/README.md) | is the cache in the weights more than a word-frequency cache, and which weights carry it? | yes beyond 1k tokens: +0.143 against +0.058 for a unigram cache at 1k-10k (the cache wins within 1k, +0.462 against +0.301). All but the output layer reproduces the full learner (+0.094 overall); output only is a short-range cache (+0.251 within 1k at lr 3e-3, +0.066 at 1k-10k) |
| [test4_checks](test4_checks/README.md) | how much of the gain is local, which layers carry it, does online + cache add up, and are the books in the training data? | 61% of the overall gain survives shuffling, but only 18% within 1k and 27% at 1k-10k (locality +0.247 and +0.105). MLP only keeps most of the long-range gain. Online + cache: +0.673 within 1k, +0.102 overall. The Pile's PG-19 component is very likely the training split only |
| [test5_capacity](test5_capacity/README.md) | is the long-range gain about recency or about the same book? (its size and smaller-part results are superseded by test 6) | mostly the same book: at 1k-10k, locality +0.117 for same-book tokens against +0.025 for earlier-book ones; same-book tokens 10k-20k back still have +0.038 |
| [test6](test6/README.md) | with tuned rates, which layers' MLPs hold the memory, and how does locality change from 14M to 31M to 70M? | every MLP but the last can hold it: a single layer-2 or layer-4 MLP gains +0.395 / +0.357 within 1k (full learner +0.301), but at lr 3e-3 (grid edge), with lower overall gains (+0.055 / +0.067 against +0.099) and larger losses on never-seen tokens. Locality within 1k: +0.247 (14M), +0.270 (31M), +0.016 (70M): flat, then a drop at 70M |
