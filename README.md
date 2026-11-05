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
  plus 2 more books for the first learning-rate picks; longer streams of 5 books x 100k and
  4 books x 200k tokens. Later tuning and meta-training use only PG-19 validation books.
- **Protocol:** non-overlapping chunks of 256 tokens, each scored with no context from earlier
  chunks; online, the model takes one plain SGD step on each chunk after scoring it. Anything that
  helps beyond 256 tokens can only come through the weights. The frozen model is the reference.
- **Controls:** the same chunks in a shuffled order (separates local from general adaptation), a
  unigram (token-count) cache on the frozen model, and online updates restricted to parts of the
  network. Gains are measured per token, binned by the distance to the token's previous occurrence
  and by whether that occurrence was in the same book, with standard errors over books.
- **Slow and fast weights** (tests 8-10): the model predicts with S + F; S takes the plain SGD step,
  and a fast part F on one MLP decays back to zero (F <- lambda F - alpha g).

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
| [4: checks](experiments_LLM2_cache/test4_checks/README.md) | how much is local, which layers, does online + cache add up, is there contamination? | 61% of the overall gain survives shuffling, but only 18% within 1k and 27% at 1k-10k. MLP only keeps most of the long-range gain. Online + cache: +0.673 within 1k, but overall no better than online alone (+0.002 ± 0.002). The books are very likely not in Pythia's training data |
| [5: capacity](experiments_LLM2_cache/test5_capacity/README.md) | recency of the token or the same book? | mostly the same book: at 1k-10k, locality +0.117 for same-book tokens against +0.025 for earlier-book ones (its size and smaller-part results are superseded by test 6) |
| [6: layers and size](experiments_LLM2_cache/test6/README.md) | which layers' MLPs hold the memory, and how does locality change with model size? | every MLP but the last; a single middle-layer MLP beats the full learner within 1k tokens (layer 2: +0.395 against +0.301), but at a higher learning rate (at its grid edge), with a lower overall gain (+0.055 against +0.099) and a larger loss on never-seen tokens (-0.129 against -0.062). Locality within 1k: +0.247 (14M), +0.270 (31M), +0.016 (70M) |
| [7: aggressive cache](experiments_LLM2_cache/test7_aggressive/README.md) | does a more aggressive cache pay off with strong locality, and can it close the gap to larger models? | no crossover: full online learning at lr 1e-4 gains most on every stream (+0.176 on long books, +0.062 shuffled). On long books it closes 50% of the 14M-31M gap (80% by the end of a book). A decaying fast part makes weak locality costlier |
| [8: dual weights](experiments_LLM2_cache/test8_dual/README.md) | does a fast decaying part on top of the slow learner add anything? | a little, with locality: +0.012 over plain SGD on K = 1 and +0.021 on long books, -0.008 shuffled (fast part on layer 4's MLP) |
| [9: Meta-SGD lite](experiments_LLM2_cache/test9_metasgd_lite/README.md) | does a learned per-tensor fast rate and decay beat it? | no: the learned values stayed near their start; ties the best dual on K = 1 and long, loses more on shuffled |
| [10: follow-ups](experiments_LLM2_cache/test10_followups/README.md) | error bars; a wider dual grid; can 14M learning online reach frozen 31M over 200k-token books? | most effects hold beyond 2 SE; layers 2 and 4 are equivalent for the fast part. 14M learning online closes about half of the gap to frozen 31M (49%, dual 51%; up to 79-84% late in a book) but never reaches it; 31M learning online gains as much (+0.168 against +0.170) |

## Status and open questions

Preliminary results: one seed (error bars are across books, not seeds), plain SGD with learning
rates picked per run on few books (several at grid edges), PG-19 books only, three model sizes.
What the tests show so far:

1. **Online learning gives a small LM a memory of its current context beyond the attention
   window** (tests 1-6). It is mostly local (it disappears when the same chunks are shuffled),
   mostly tied to the current book rather than to the recency of single tokens, more than a
   token-count cache from 1k tokens back, and stored in the MLPs below the output layer.
2. **Making the base weights more aggressive does not pay off** (test 7). Higher learning rates, a
   single layer, or a decaying update on its own gain more on recent material but lose much more
   on everything else; there is no crossover across locality levels.
3. **A small, decaying fast part on one middle MLP, on top of the slow learner, adds a short-term
   cache with a good exchange rate** (tests 8, 10B): small but robust gains with strong locality
   (+0.012 ± 0.001 nats/token on K = 1, +0.021 ± 0.003 on long books), a small cost without it
   (-0.008 on shuffled); layers 2 and 4 are equivalent. Meta-learning per-tensor rates and decays
   learned almost nothing beyond its starting values (test 9).
4. **With long contexts, online learning is worth a large part of a model-size step** (tests 7, 8,
   10C). Over 200k-token books, 14M learning online closes about half of the gap to a frozen 31M
   (up to 79-84% late in a book). But 31M learning online gains about as much (+0.168 against
   +0.170) and beats a frozen 70M in some late bins. Online learning does not favor the smaller
   model: both gain about the same, and the locality effect is the same at 14M and 31M overall
   (+0.000 ± 0.001; +0.018 ± 0.004 on recent same-book tokens). At these sizes this does not
   support the hypothesis that smaller models rely more on the cache.

Open questions and next steps:

- **70M:** plain SGD barely helps Pythia-70M at any learning rate tried (locality +0.004 overall).
  Whether that is about size or about the optimizer is open; an update normalized per layer or
  per parameter, and more sizes, would tell.
- **Capacity and interleaving:** the eviction test cannot separate forgetting a book from adapting
  to books in general. Interleaving K books chunk by chunk, with K growing, would measure how many
  contexts the weights can hold at once.
- **More seeds:** PolyPythias (several pre-training seeds per size) would give error bars across
  models, not only across books.
- **Meta-learning the update:** if pursued, start the other tensors with no fast part, use longer
  episodes or more meta-training data, and check that the meta-gradient can move the rates at all.

## Folders

| folder | contents |
|---|---|
| [`experiments_LLM2_cache/`](experiments_LLM2_cache/README.md) | code, results and one subfolder per test |
| [`paper/`](paper/) | preliminary-results paper (`main.tex`, `references.bib`, `figures/`, `make_figures.py`), built as `main.pdf` |
