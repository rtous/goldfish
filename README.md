# goldfish: online learning as a cache of the current context

A language model's attention window forgets everything beyond its last few hundred or thousand
tokens. The text a model reads at inference time, though, has strong temporal locality: a book
keeps its characters, places and vocabulary for hundreds of thousands of tokens. A small language
model that keeps learning online during inference might use part of its weights as a cache of the
current context: material seen recently, beyond the attention window, would be predicted better,
new material would gradually evict old material, and general knowledge would stay in the weights.
This repository tests that idea.

## Setup in brief

- **Models:** Pythia-14M, 31M and 70M (`EleutherAI/pythia-*`), CPU, float32.
- **Data:** 25 PG-19 test books, each cut to its first 20k tokens and concatenated (500k tokens),
  plus 2 more books for learning-rate picks.
- **Protocol:** non-overlapping chunks of 256 tokens, each scored with no context from earlier
  chunks; online, the model takes one plain SGD step on each chunk after scoring it. Anything that
  helps beyond 256 tokens can only come through the weights. The frozen model is the reference.
- **Controls:** the same chunks in a shuffled order (separates local from general adaptation), a
  unigram (token-count) cache on the frozen model, and online updates restricted to parts of the
  network. Gains are measured per token, binned by the distance to the token's previous occurrence
  and by whether that occurrence was in the same book.

Details, metrics and commands: [`experiments_LLM2_cache/README.md`](experiments_LLM2_cache/README.md).

## How to run

Python 3.13 in `venv/` (see `CLAUDE.md`); dependencies in `requirements.txt`.

```
python3.13 -m venv venv && venv/bin/pip install -r requirements.txt
cd experiments_LLM2_cache
../venv/bin/python online.py            # learning-rate check (downloads the books and Pythia-14M)
../venv/bin/python test1_distance.py    # then test2_eviction.py, test3_baselines.py, ... in order
```

Paper figures: `venv/bin/python paper/make_figures.py` (reads only the saved results); the paper is
`paper/main.tex`, built as `paper/main.pdf` with `latexmk -pdf main.tex` in `paper/`.

## Tests

| test | question | current answer |
|---|---|---|
| [1: distance](experiments_LLM2_cache/test1_distance/README.md) | does the online gain extend beyond the 256-token window, and how far? | yes, up to about 10k tokens: for rare tokens last seen in an earlier chunk, +0.301 nats/token under 1k tokens back, +0.143 at 1k-10k, +0.010 at 10k-100k; overall +0.099. Tokens repeated within the chunk get worse (-0.107) |
| [2: eviction](experiments_LLM2_cache/test2_eviction/README.md) | does a book's memory get evicted by other books, and how fast? | slowly: memory of A is +0.038, +0.027 and +0.019 after 0, 10k and 100k tokens of other books (matched comparison). Whether the A1 run keeps A and the comparison run catches up, or the A1 run loses A while gaining general adaptation, this test cannot tell |
| [3: baselines](experiments_LLM2_cache/test3_baselines/README.md) | is it more than a word-frequency cache, and which weights carry it? | yes beyond 1k tokens (+0.143 against +0.058 for a unigram cache at 1k-10k); the cache wins within 1k (+0.462 against +0.301). The long-range gain lives below the output layer; output only is a short-range cache |
| [4: checks](experiments_LLM2_cache/test4_checks/README.md) | how much is local, which layers, does online + cache add up, is there contamination? | 61% of the overall gain survives shuffling, but only 18% within 1k and 27% at 1k-10k. MLP only keeps most of the long-range gain. Online + cache: +0.673 within 1k. The books are very likely not in Pythia's training data |
| [5: capacity](experiments_LLM2_cache/test5_capacity/README.md) | recency of the token or the same book? | mostly the same book: at 1k-10k, locality +0.117 for same-book tokens against +0.025 for earlier-book ones (its size and smaller-part results are superseded by test 6) |
| [6: layers and size](experiments_LLM2_cache/test6/README.md) | which layers' MLPs hold the memory, and how does locality change with model size? | every MLP but the last; a single middle-layer MLP beats the full learner within 1k tokens (layer 2: +0.395 against +0.301), but at a higher learning rate (at its grid edge), with a lower overall gain (+0.055 against +0.099) and a larger loss on never-seen tokens (-0.129 against -0.062). Locality within 1k: +0.247 (14M), +0.270 (31M), +0.016 (70M) |

## Status and open questions

Preliminary results: one seed, plain SGD with a learning rate picked per run on two held-out books
(several picks at grid edges), short books, three model sizes. What is established so far:

- Online learning gives a per-token gain beyond the attention window that is mostly local (it
  needs the recent material) and mostly tied to the current book.
- It is not just a token-count cache: a unigram cache is better within 1k tokens, but the online
  learner keeps gaining up to about 10k tokens, and the two combine within 10k.
- The memory sits in the MLPs of the layers below the output; a single middle-layer MLP can carry much of it, at a higher learning rate and a larger cost on tokens not seen recently.

Open questions and next steps:

- **Capacity and interleaving:** the eviction test cannot separate forgetting a book from adapting
  to books in general. Interleaving K books chunk by chunk, with K growing, would measure how many
  contexts the weights can hold at once.
- **Model-size trend with a normalized update:** locality is flat from 14M to 31M and collapses at
  70M, where online SGD barely helps at any learning rate. Plain SGD with a per-model learning rate
  may not be comparable across sizes; an update normalized per layer or per parameter (or a
  constant relative weight change) would make the comparison fairer, and more sizes would show
  whether the drop is a trend.
- **More seeds:** PolyPythias (several pre-training seeds per size) would give error bars across
  models, not only across books.

## Folders

| folder | contents |
|---|---|
| [`experiments_LLM2_cache/`](experiments_LLM2_cache/README.md) | code, results and one subfolder per test |
| [`paper/`](paper/) | preliminary-results paper (`main.tex`, `references.bib`, `figures/`, `make_figures.py`), built as `main.pdf` |
