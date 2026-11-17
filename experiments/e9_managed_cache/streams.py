"""Streams for E9: the registry main streams and two mixed streams (books with WikiText blocks).

Mixed layout: every book in book order (78 chunks), each followed by a block of WIKI_PER_BOOK = 4
consecutive WikiText chunks (1,024 tokens); block j is chunks 4j..4j+3 of one contiguous WikiText text.
    mixed      the 25 main books + 25 WikiText blocks from WikiText-103 *validation* (2,050 chunks)
    valmixed   the 10 val-pick books + 10 WikiText blocks from WikiText-103 *train* (820 chunks), for picks
The held-out WikiText of the snapshots is WikiText-103 *test* (data.heldout), so the three are disjoint.
WikiText is detokenized as in data.py (lm-evaluation-harness), tokenized with Pythia's tokenizer.
Order rows are (unit, chunk): unit < n_books is a book, unit = n_books + j is WikiText block j.
"""
import os

import numpy as np

from data import DATA, DEFAULT, L, _tokenizer, chunk_set, detokenize, stream

WIKI_PER_BOOK = 4
CHARS = 300000
WIKI_SPLIT = {"mixed": "validation", "valmixed": "train"}
BOOKS = {"mixed": "main", "valmixed": "val_pick"}


def wiki_chunks(split, n_chunks):
    """[n_chunks, L] token ids: the start of a WikiText-103 split, detokenized (cached in data/)."""
    path = os.path.join(DATA, "e9_wikitext103_%s_detok_%dx%d_%s.npy" % (split, n_chunks, L, DEFAULT.replace("/", "__")))
    if not os.path.exists(path):
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, cache_dir=os.path.join(DATA, "hf"))
        text = "".join(ds["text"])[:CHARS]
        ids = _tokenizer(None)(detokenize(text), add_special_tokens=False)["input_ids"]
        assert len(ids) >= n_chunks * L, (split, len(ids))
        np.save(path, np.array(ids[:n_chunks * L], dtype=np.int64).reshape(n_chunks, L))
    return np.load(path)


def get(name):
    """(ids [chunks, L] in reading order, order [(unit, chunk)], n_books)."""
    if name in WIKI_SPLIT:
        books = chunk_set(BOOKS[name])
        n_books = books.shape[0]
        wiki = wiki_chunks(WIKI_SPLIT[name], n_books * WIKI_PER_BOOK)
        ids, order = [], []
        for b in range(n_books):
            ids += list(books[b])
            order += [(b, k) for k in range(books.shape[1])]
            ids += list(wiki[b * WIKI_PER_BOOK:(b + 1) * WIKI_PER_BOOK])
            order += [(n_books + b, k) for k in range(WIKI_PER_BOOK)]
        return np.stack(ids), order, n_books
    chunks, order = stream(name)
    return np.stack([chunks[b, k] for b, k in order]), order, chunks.shape[0]
