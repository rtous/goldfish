# Goldfish experiments

Pythia-14M / 31M / 70M (CPU, float32, seed 0) learn PG-19 books online: each 256-token chunk is
scored on its own, then the model takes one update step on it. These experiments support claims
C1-C10 of the paper; each folder has a script, CSVs, plots, a README (question, setup, table,
findings, caveats) and `runtime.txt`.

## Layout

- `lib/`: data (books, streams, held-out text), models, learner (update rules, parameter subsets),
  metrics (distance bins, SE over books, locality, forgetting), baselines (unigram cache), plots.
- `runs.py`: the registry of every stream run (size, rule, lr, parameter subset, stream) and its
  runner; each run is saved once to `runs/<key>.npz` and read by the analyses.
- `e0_lr_picks` ... `e7_update_rules`: one experiment each.
- `REGRESSION.md`, `regression.csv`: saved output of a one-off comparison of old
  (`experiments_LLM2_cache/`, deleted; in the git history) against new.
- `runs/` (about 37 MB) is committed on purpose, so that every `report` works without rerunning.
- `data/`: cached tokenized books, models and WikiText (gitignored).

## How to run (from `experiments/`)

```
../venv/bin/python lib/learner.py                         # check: segmented == single run, every rule
../venv/bin/python e0_lr_picks/e0_lr_picks.py run <group>  # groups 14m, 14m_subsets, 31m, 70m; then: report
../venv/bin/python runs.py worker                         # all registry runs; start up to 3 in parallel
../venv/bin/python e4_eviction/e4_eviction.py run          # E4 has its own runs
../venv/bin/python e7_update_rules/e7_update_rules.py diag
../venv/bin/python eN_<name>/eN_<name>.py report          # each experiment's tables and plots
```

## Experiments

| experiment | claims | question | answer | runtime |
|---|---|---|---|---|
| [e0_lr_picks](e0_lr_picks/README.md) | all | which learning rate does each run use? | picked on 10 PG-19 validation books, every pick interior; the 9 (size, rule) picks equal the old ones | 137 min |
| [e1_locality](e1_locality/README.md) | C1-C4 | how far back does the gain reach, is it local, same book or recency, more than a unigram cache? | about 10k tokens (+0.307 within 1k, +0.144 at 1k-10k); 20% within 1k survives shuffling; same book (locality +0.120 against +0.035 at 1k-10k); beats a unigram cache beyond 1k (+0.144 against +0.059) | 36 min |
| [e2_where](e2_where/README.md) | C5 | which weights hold the memory? | the MLPs: MLP only beats the full learner (+0.415 against +0.307 within 1k) with more locality (+0.334 against +0.245 within 1k) but more WikiText forgetting (+0.235 against +0.196; exchange rate 0.45 against 0.51); every single-layer MLP keeps a large part, the last layer the least | 56 min |
| [e3_locality_levels](e3_locality_levels/README.md) | C6 | does the gain fall with locality? | yes, at every step: long +0.176 > K1 +0.100 > K5 +0.074 > K25 +0.065 > shuffled +0.062 | 8 min (+ E1's runs) |
| [e4_eviction](e4_eviction/README.md) | C7 | how fast is a book's memory evicted by other books? | halves in about 6k tokens (+0.097 to +0.044 at 10k), long tail (+0.016 ± 0.008 at 300k); a power law fits better than an exponential | 55 min |
| [e5_forgetting](e5_forgetting/README.md) | C8 | what does it cost in general knowledge? | WikiText forgetting about 2x the stream gain (exchange rate 0.51); nothing on held-out books read in order (-0.006 ± 0.010) | 59 min |
| [e6_long_books](e6_long_books/README.md) | C9 | over 200k-token books, does 14M online reach frozen 31M? | closes 49% of the gap (79% at its peak), never reaches it; 31M online gains as much (+0.168) | 60 min |
| [e7_update_rules](e7_update_rules/README.md) | C10 | does the update rule decide the cache, across sizes? | plain SGD starves 70M's MLPs; normalized SGD recovers part of the cache; Adam has none at any size; smaller models do not forget more per nat | 8.5 h (3 parallel) |

Runtimes are process time on the CPU, with shared runs counted once where they are first used.
