"""PG-19 test books, tokenized for a model and cut to their first 20k tokens.

Source: the Hugging Face dataset `deepmind/pg19`. It holds only a loading script (pg19.py, no
parquet files), which `datasets` 5 no longer runs. So this file does what the script does, for the
test split only: read the list of test files from the dataset repo (data/test_files.txt), sort it
(the order the dataset gives the books), and download each book needed from the bucket the script
uses (https://storage.googleapis.com/deepmind-gutenberg/test/<id>.txt) plus metadata.csv for the
titles. The training set is never downloaded. Everything goes to data/ (gitignored).

Each book is tokenized on its own (no special tokens) and cut to its first MAX_TOKENS tokens.
Books shorter than MAX_TOKENS (some test "books" are single short stories) are skipped: we keep the
first N_BOOKS books, in dataset order, that have at least MAX_TOKENS tokens. Cached as
data/books_<model>.npz: ids [N_BOOKS, MAX_TOKENS] in dataset order, book ids, titles, and the ids of
the skipped books.

Usage:
    python data.py      # download + tokenize, print the books
"""
import csv
import io
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = "deepmind/pg19"
BUCKET = "https://storage.googleapis.com/deepmind-gutenberg/"
N_BOOKS = 27            # 25 for the stream + 2 for the learning-rate check
MAX_TOKENS = 20000
CHARS = 200000          # tokenize only the start of a book (about 4 characters per token)


def test_files():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "data/test_files.txt", repo_type="dataset", cache_dir=os.path.join(DATA, "hf"))
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


def books(model_name, tokenizer=None):
    """Dict: ids [N_BOOKS, MAX_TOKENS] (int64), book_ids (Gutenberg ids), titles."""
    path = os.path.join(DATA, "books_%s.npz" % model_name.replace("/", "__"))
    if not os.path.exists(path):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(DATA, "hf"))
        titles = {r[0]: r[1] for r in csv.reader(io.StringIO(fetch("metadata.csv")))}
        ids, book_ids, skipped = [], [], []
        for name in test_files():
            if len(ids) == N_BOOKS:
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
    d = np.load(path)
    return {"ids": d["ids"], "book_ids": [str(b) for b in d["book_ids"]], "titles": json.loads(str(d["titles"])),
            "skipped": [str(b) for b in d["skipped"]]}


def main():
    from models import DEFAULT
    d = books(DEFAULT)
    for i, (b, t) in enumerate(zip(d["book_ids"], d["titles"])):
        print("%2d  %6s  %s" % (i, b, t))
    print("%d books x %d tokens; skipped (under %d tokens): %s" % (d["ids"].shape + (MAX_TOKENS, ", ".join(d["skipped"]))))


if __name__ == "__main__":
    main()
