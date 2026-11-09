# goldfish: online learning as a cache of the current context

A language model's attention window forgets everything beyond its last few hundred or thousand
tokens. The text a model reads at inference time, though, has strong temporal locality: a book
keeps its characters, places and vocabulary for hundreds of thousands of tokens. A small language
model that keeps learning online during inference might use part of its weights as a cache of the
current context: material seen recently, beyond the attention window, would be predicted better,
new material would gradually evict old material, and general knowledge would stay in the weights.
This repository tests that idea.

## Setup in brief

- **Models:** Pythia-14M, 31M and 70M (`EleutherAI/pythia-*`), CPU, float32, seed 0.
- **Data:** 25 PG-19 test books cut to their first 20k tokens (the main set, 500k tokens); longer
  streams of 5 books x 100k and 4 books x 200k tokens. Learning rates are picked on 10 PG-19
  validation books; no test book is used for tuning.
- **Protocol:** non-overlapping chunks of 256 tokens, each scored with no context from earlier
  chunks; online, the model takes one update step on each chunk after scoring it. Anything that
  helps beyond 256 tokens can only come through the weights. The frozen model is the reference.
- **Controls:** the same chunks shuffled or interleaved across books (local against general
  adaptation), a unigram (token-count) cache on the frozen model, updates restricted to parts of the
  network, and held-out text never trained on (detokenized WikiText-103, unseen PG-19 books).
  Gains are per token, binned by the distance to the token's previous occurrence and by whether it
  was in the same book, with standard errors over books.

Details, layout and commands: [`experiments/README.md`](experiments/README.md).

## How to run

Python 3.13 in `venv/` (see `CLAUDE.md`); dependencies in `requirements.txt`.

```
python3.13 -m venv venv && venv/bin/pip install -r requirements.txt
cd experiments
../venv/bin/python e0_lr_picks/e0_lr_picks.py run 14m     # and 14m_subsets, 31m, 70m; then: report
../venv/bin/python runs.py worker                         # every stream run (start up to 3 in parallel)
../venv/bin/python e4_eviction/e4_eviction.py run
../venv/bin/python e7_update_rules/e7_update_rules.py diag
../venv/bin/python e1_locality/e1_locality.py report      # and the other experiments' report
```

Paper tables and figures (read only the saved results): `venv/bin/python paper/make_tables.py` and
`venv/bin/python paper/make_figures.py`; the paper is `paper/main.tex`, built with
`latexmk -pdf main.tex` in `paper/`.

## Experiments

| experiment | claims | question | current answer |
|---|---|---|---|
| [E0: learning-rate picks](experiments/e0_lr_picks/README.md) | all | which rate does each run use? | picked on validation books, every pick inside its grid |
| [E1: locality](experiments/e1_locality/README.md) | C1-C4 | how far back does the online gain reach, is it local, about the same book, more than a unigram cache? | up to about 10k tokens (+0.308 nats/token within 1k, +0.144 at 1k-10k); only 20% of it within 1k survives shuffling; tied to the same book (locality +0.120 against +0.035 at 1k-10k); beats a unigram cache beyond 1k (+0.144 against +0.059) though the cache wins within 1k |
| [E2: where](experiments/e2_where/README.md) | C5 | which weights hold the memory? | the MLPs below the output layer: updating only the MLPs beats the full learner (+0.415 against +0.308 within 1k); each single-layer MLP keeps a large part, the last layer the least; the output layer alone is a short-range cache |
| [E3: locality levels](experiments/e3_locality_levels/README.md) | C6 | does the gain fall with locality? | at every step: long books +0.176 > K = 1 +0.100 > K = 5 +0.074 > K = 25 +0.065 > shuffled +0.062 |
| [E4: eviction](experiments/e4_eviction/README.md) | C7 | how fast do other books evict a book's memory? | it halves in about 6k tokens (+0.097 to +0.044 at 10k) with a long tail (+0.016 ± 0.008 at 300k); a power law fits better than an exponential |
| [E5: forgetting](experiments/e5_forgetting/README.md) | C8 | what does it cost in general knowledge? | outside the domain, a lot: WikiText forgetting is about twice the stream gain (exchange rate 0.51); nothing on unseen PG-19 books read in order (-0.006 ± 0.010) |
| [E6: long books](experiments/e6_long_books/README.md) | C9 | over 200k-token books, does 14M online reach frozen 31M? | it closes 49% of the gap (up to 79% late in a book) but never reaches it; 31M online gains as much (+0.168 against +0.170) |
| [E7: update rules](experiments/e7_update_rules/README.md) | C10 | does the update rule decide the cache, across sizes? | plain SGD moves 70M's MLPs 100x less per step and leaves almost no cache; normalized SGD recovers part of it; Adam gives none at any size; smaller models do not forget more per nat of gain |

## Status

Preliminary results: one seed (error bars are across books, not seeds), PG-19 books only, three
model sizes, every learning rate picked the same way on validation books. The comparison with the
earlier, incremental experiments is in [`experiments/REGRESSION.md`](experiments/REGRESSION.md).

## Open questions

- **A slow + fast "cache" part and its knobs:** a decaying fast part on one MLP on top of the slow
  learner gave +0.012 (K = 1) and +0.021 (long books), but did not reduce forgetting.
- **Aggressive single-layer caches:** one MLP at a high rate gains more on recent material but loses
  it through forgetting; whether some setting avoids that is open.
- **Why Adam removes the cache:** ablate its momentum and its per-weight normalization.
- **Capacity:** interleave many books at once, and relate the adapter size to the eviction half-life.
- **Baselines:** the full 2k-token context, and baselines matched in compute.
- **More seeds and sizes:** PolyPythias (several pre-training seeds per size), and 160M.
- **Human-memory tests:** savings on re-reading, interference by similarity, spacing effects,
  planted facts.

## Folders

| folder | contents |
|---|---|
| [`experiments/`](experiments/README.md) | library, run registry, one folder per experiment, regression report |
| [`paper/`](paper/) | preliminary-results paper (`main.tex`, `references.bib`, `tables/`, `figures/`, `make_tables.py`, `make_figures.py`) |
