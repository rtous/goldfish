"""Data: PG-19 books, held-out text, chunking and stream orders.

PG-19 comes from the Hugging Face dataset `deepmind/pg19`, which holds only a loading script that
`datasets` 5 no longer runs. So this file does what the script does: read the list of files of a
split from the dataset repo (data/<split>_files.txt), sort it (the dataset's order), and download
each book needed from the bucket the script uses (storage.googleapis.com/deepmind-gutenberg) plus
metadata.csv for the titles. The training set is never downloaded. Everything goes to data/
(gitignored). Each book is tokenized on its own (no special tokens).

Sets (all tokenized with Pythia's tokenizer, shared by 14M / 31M / 70M):
    main        the first 25 test books with at least 20k tokens, dataset order, cut to 20k tokens.
                books() also keeps the next 2 (the old "check books", no longer used for tuning) so
                that long_books() skips exactly the same books as before.
    long        the first 5 test books (check books excluded) with at least 100k tokens, cut to 100k
    very long   the first 4 such books with at least 200k tokens, cut to 200k
    val-pick    the first 10 PG-19 validation books with at least 20k tokens (all learning-rate picks)
    held-out    the last 6 of those 42 validation books, tokens 0-3839 of each (15 chunks): forgetting
    WikiText    WikiText-103 test split, the first 20480 raw tokens' text, detokenized as in
                lm-evaluation-harness, re-tokenized and cut to a multiple of L

Chunking (one chunking everywhere): each book is chunked on its own into L = 256-token chunks, the
remainder dropped (78 chunks per main book). A stream is a list of (book, chunk) in reading order:
    K1        book after book
    K5        groups of 5 books (book order); inside a group, chunks round-robin
    K25       round-robin across all 25 books
    shuffled  all chunks in one random order (numpy default_rng(0) permutation of the K1 list)
"""
import csv
import io
import json
import os
import re
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
DATA = os.path.join(HERE, "data")
REPO = "deepmind/pg19"
BUCKET = "https://storage.googleapis.com/deepmind-gutenberg/"
DEFAULT = "EleutherAI/pythia-14m"   # the tokenizer used for every set
L = 256
N_BOOKS = 27            # 25 main books + the 2 old check books (kept for long_books' skip list)
N_MAIN = 25
MAX_TOKENS = 20000
CHARS = 200000          # tokenize only the start of a book (about 4 characters per token)
N_LONG, LONG_TOKENS = 5, 100000
N_VERY_LONG, VERY_LONG_TOKENS = 4, 200000
N_PICK, TRAIN_SHARE = 10, 0.8   # val-pick books; the held-out 6 are the last 20% of the rest
N_WIKI, HELD_BOOK_TOKENS = 80 * L, 15 * L


def _tokenizer(tokenizer):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT, cache_dir=os.path.join(DATA, "hf"))
    return tokenizer


def test_files(split="test"):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "data/%s_files.txt" % split, repo_type="dataset", cache_dir=os.path.join(DATA, "hf"))
    with open(path) as f:
        return sorted(f.read().splitlines())


def fetch(name):
    path = os.path.join(DATA, "pg19", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(BUCKET + name) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load(path):
    d = np.load(path)
    return {"ids": d["ids"], "book_ids": [str(b) for b in d["book_ids"]], "titles": json.loads(str(d["titles"])),
            "skipped": [str(b) for b in d["skipped"]]}


def _short_books(path, split, n_books, tokenizer):
    """First n_books books of a split with at least MAX_TOKENS tokens, cut to MAX_TOKENS (cached)."""
    if not os.path.exists(path):
        tokenizer = _tokenizer(tokenizer)
        titles = {r[0]: r[1] for r in csv.reader(io.StringIO(fetch("metadata.csv")))}
        ids, book_ids, skipped = [], [], []
        for name in test_files(split):
            if len(ids) == n_books:
                break
            book = os.path.splitext(os.path.basename(name))[0]
            text = fetch(name)
            t = tokenizer(text[:CHARS], add_special_tokens=False)["input_ids"]
            # +10: the cut must not fall on a token split by the character cut
            if len(t) < MAX_TOKENS + 10:
                if len(text) > CHARS:
                    raise SystemExit("%s: only %d tokens in %d characters; raise CHARS" % (name, len(t), CHARS))
                skipped.append(book)
                continue
            ids.append(t[:MAX_TOKENS])
            book_ids.append(book)
        np.savez_compressed(path, ids=np.array(ids, dtype=np.int64), book_ids=np.array(book_ids),
                            titles=np.array(json.dumps([titles[b] for b in book_ids])), skipped=np.array(skipped))
    return _load(path)


def books(tokenizer=None):
    """Test books: ids [27, 20000] (25 main + 2 old check books), book_ids, titles, skipped."""
    return _short_books(os.path.join(DATA, "books_%s.npz" % DEFAULT.replace("/", "__")), "test", N_BOOKS, tokenizer)


def val_books(tokenizer=None):
    """Validation books with at least 20k tokens (42), cut to 20k; never test books."""
    return _short_books(os.path.join(DATA, "val_books_%s.npz" % DEFAULT.replace("/", "__")), "validation", 10 ** 9,
                        tokenizer)


def long_books(n_books=N_LONG, n_tokens=LONG_TOKENS, tokenizer=None):
    """Test books (old check books excluded) with at least n_tokens tokens (whole book tokenized), cut."""
    suffix = "" if (n_books, n_tokens) == (N_LONG, LONG_TOKENS) else "_%dx%d" % (n_books, n_tokens)
    path = os.path.join(DATA, "long_books%s_%s.npz" % (suffix, DEFAULT.replace("/", "__")))
    if not os.path.exists(path):
        tokenizer = _tokenizer(tokenizer)
        check = set(books(tokenizer)["book_ids"][N_MAIN:])
        titles = {r[0]: r[1] for r in csv.reader(io.StringIO(fetch("metadata.csv")))}
        ids, book_ids, skipped = [], [], []
        for name in test_files():
            if len(ids) == n_books:
                break
            book = os.path.splitext(os.path.basename(name))[0]
            if book in check:
                continue
            t = tokenizer(fetch(name), add_special_tokens=False)["input_ids"]
            if len(t) < n_tokens:
                skipped.append(book)
                continue
            ids.append(t[:n_tokens])
            book_ids.append(book)
        np.savez_compressed(path, ids=np.array(ids, dtype=np.int64), book_ids=np.array(book_ids),
                            titles=np.array(json.dumps([titles[b] for b in book_ids])), skipped=np.array(skipped))
    return _load(path)


# ---------------------------------------------------------------- chunking and streams


def chunk(ids):
    """[books, tokens] -> [books, chunks, L], remainder dropped."""
    n = ids.shape[1] // L
    return ids[:, :n * L].reshape(ids.shape[0], n, L)


def orders(n_books, n_chunks):
    """Stream name -> list of (book, chunk) in reading order."""
    k1 = [(b, c) for b in range(n_books) for c in range(n_chunks)]
    k5 = [(b, c) for g in range(0, n_books, 5) for c in range(n_chunks) for b in range(g, g + 5)]
    k25 = [(b, c) for c in range(n_chunks) for b in range(n_books)]
    perm = np.random.default_rng(0).permutation(len(k1))
    return {"K1": k1, "K5": k5, "K25": k25, "shuffled": [k1[i] for i in perm]}


def chunk_set(name):
    """Chunk set [books, chunks, L]: main, long, verylong, val_pick."""
    if name == "main":
        return chunk(books()["ids"][:N_MAIN])
    if name == "long":
        return chunk(long_books()["ids"])
    if name == "verylong":
        return chunk(long_books(N_VERY_LONG, VERY_LONG_TOKENS)["ids"])
    if name == "val_pick":
        return chunk(val_books()["ids"][:N_PICK])
    raise ValueError(name)


STREAMS = {"main_K1": ("main", "K1"), "main_K5": ("main", "K5"), "main_K25": ("main", "K25"),
           "main_shuffled": ("main", "shuffled"), "long": ("long", "K1"), "verylong": ("verylong", "K1"),
           "val_pick": ("val_pick", "K1")}


def stream(name):
    """(chunk set [books, chunks, L], order: list of (book, chunk)) for a stream name in STREAMS."""
    set_name, kind = STREAMS[name]
    c = chunk_set(set_name)
    return c, orders(c.shape[0], c.shape[1])[kind]


# ---------------------------------------------------------------- held-out text


def detokenize(t):
    """WikiText detokenizer of lm-evaluation-harness (wikitext_detokenizer)."""
    t = t.replace("s '", "s'")
    t = re.sub(r"/' [0-9]/", r"/'[0-9]/", t)
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for p in (":", ";", ".", "!", "?", ","):
        t = t.replace(" %s " % p, "%s " % p)
    t = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", t)
    t = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", t)
    t = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", t)
    t = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', t)
    t = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", t)
    t = t.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    t = t.replace(" " + chr(176) + " ", chr(176))
    t = t.replace(" \n", "\n").replace("\n ", "\n")
    t = t.replace(" N ", " 1 ").replace(" 's", "'s")
    return t


def wikitext_detok_ids():
    """WikiText-103 test (wikitext-103-raw-v1): the articles joined in order, the text behind the first
    N_WIKI raw tokens, detokenized, tokenized and cut to a multiple of L (cached)."""
    path = os.path.join(DATA, "wikitext103_test_detok_%s.npy" % DEFAULT.replace("/", "__"))
    if not os.path.exists(path):
        from datasets import load_dataset
        tok = _tokenizer(None)
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", cache_dir=os.path.join(DATA, "hf"))
        text = "".join(ds["text"])[:200000]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert len(enc["input_ids"]) > N_WIKI + 10
        raw = text[:enc["offset_mapping"][N_WIKI - 1][1]]
        ids = tok(detokenize(raw), add_special_tokens=False)["input_ids"]
        np.save(path, np.array(ids[:len(ids) // L * L], dtype=np.int64))
    return np.load(path)


def heldout_books():
    """The 6 held-out validation books, tokens 0-3839 each: [6, 3840]."""
    ids = val_books()["ids"]
    n_rest = len(ids) - N_PICK
    held = ids[N_PICK + round(TRAIN_SHARE * n_rest):]
    assert len(held) == 6
    return held[:, :HELD_BOOK_TOKENS]


def heldout():
    """(token ids: detokenized WikiText then the 6 held-out books, number of WikiText chunks)."""
    wiki = wikitext_detok_ids()
    return np.concatenate([wiki, heldout_books().reshape(-1)]), len(wiki) // L


if __name__ == "__main__":
    d = books()
    for i, (b, t) in enumerate(zip(d["book_ids"][:N_MAIN], d["titles"])):
        print("%2d  %6s  %s" % (i, b, t))
    for name in ("main", "long", "verylong", "val_pick"):
        print(name, chunk_set(name).shape)
    h, n_wiki = heldout()
    print("held-out: %d tokens, %d WikiText chunks + %d book chunks" % (len(h), n_wiki, len(h) // L - n_wiki))
