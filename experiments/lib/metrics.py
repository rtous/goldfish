"""Metrics shared by the experiments.

Gain = frozen loss - run loss (nats/token) on the same tokens, each model against its own frozen loss.
All means are token-level; their SE is over books: the SE of the per-book means (or of the paired
per-book differences when two runs read the same chunks), books with fewer than MIN_BOOK tokens in a
bin left out. A value with |mean| < 2 SE is flagged ("under_2se").

Distance bins (main set, K1 order = book order): for each scored token, the distance to the previous
occurrence of the same token id anywhere earlier in the book-order stream:
    same chunk        the previous occurrence is in the token's own chunk (attention sees it)
    earlier chunk     < 1k, 1k-10k, 10k-100k, >= 100k (the first includes the few tokens whose previous
                      occurrence is in an earlier chunk less than 256 tokens back)
    never seen        first occurrence in the stream
The TOP = 1000 most frequent token ids of the stream are left out of the bins ("all tokens" has no
exclusion). Earlier-chunk bins are split by whether the previous occurrence is in the token's own book
("same book") or an earlier one ("earlier book"). A token keeps its bin in every stream (shuffled
losses are put back in book order), so the same tokens are compared everywhere.
"""
import numpy as np

from data import L

TOP = 1000
MIN_BOOK = 50
BINS = ["same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
        "earlier chunk, >= 100k", "never seen"]


def prev_occurrence(ids):
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    return prev


def distance_bins(ids):
    """Bin index (into BINS) per token, -1 for the TOP most frequent ids."""
    prev = prev_occurrence(ids)
    t = np.arange(len(ids))
    dist = t - prev
    b = np.select([prev < 0, prev >= t - t % L, dist < 1000, dist < 10000, dist < 100000], [5, 0, 1, 2, 3], 4)
    frequent = np.argsort(-np.bincount(ids))[:TOP]
    b[np.isin(ids, frequent)] = -1
    return b


def bin_masks(chunks, split=True):
    """(book per token, {bin name: mask}) over the flattened chunk set [books, chunks, L] in book order.
    Names: "all tokens", the BINS, and with split "<earlier-chunk bin>, same book" / ", earlier book"."""
    ids = chunks.reshape(-1)
    book = np.repeat(np.arange(chunks.shape[0]), chunks.shape[1] * chunks.shape[2])
    prev = prev_occurrence(ids)
    same_book = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    bins = distance_bins(ids)
    masks = {"all tokens": np.ones(len(ids), bool)}
    for k, name in enumerate(BINS):
        masks[name] = bins == k
        if split and name.startswith("earlier chunk"):
            masks[name + ", same book"] = (bins == k) & same_book
            masks[name + ", earlier book"] = (bins == k) & ~same_book
    return book, masks


def se(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def per_book(x, book, mask):
    """Per-book means of x over masked finite tokens; books with fewer than MIN_BOOK such tokens dropped."""
    m = mask & np.isfinite(x)
    out = []
    for b in np.unique(book):
        mb = m & (book == b)
        if mb.sum() >= MIN_BOOK:
            out.append(float(x[mb].mean()))
    return np.array(out)


def summary(x, book, mask):
    """{tokens, mean (token level), se (over books), books, under_2se} of x on the mask."""
    m = mask & np.isfinite(x)
    pb = per_book(x, book, m)
    mean, s = float(x[m].mean()) if m.any() else float("nan"), se(pb)
    return {"tokens": int(m.sum()), "mean": mean, "se": s, "books": len(pb),
            "under_2se": bool(abs(mean) < 2 * s) if np.isfinite(s) else ""}


def position_bins(x, book, n_per_book, width=1000, books=None):
    """Per position bin in the book: (mean over books of per-book means, SE over books). books: which books
    (default all)."""
    xb = x.reshape(-1, n_per_book)
    if books is not None:
        xb = xb[books]
    n_bins = (n_per_book + width - 1) // width
    per = np.array([[np.nanmean(row[j * width:(j + 1) * width]) for j in range(n_bins)] for row in xb])
    return per.mean(0), per.std(0, ddof=1) / np.sqrt(len(per))


def locality(gain_in_order, gain_shuffled):
    """Locality = gain in order - gain shuffled (per token; summarize with summary(), paired by book)."""
    return gain_in_order - gain_shuffled


def forgetting(snaps, positions, n_wiki, n_books=6):
    """Forgetting on held-out text from snapshot chunk means snaps [n_snap, n_wiki + book chunks]
    (snapshot 0 = pretrained). Returns {wiki/books: {last_half, last_half_se, final, final_se}}: the mean
    over the snapshots in the last half of the stream (position > half the stream) and the last snapshot,
    of snapshot - pretrained; SE over WikiText chunks, or over the held-out books."""
    snaps = np.asarray(snaps, dtype=np.float64)
    half = [k for k in range(1, len(positions)) if positions[k] > positions[-1] / 2]
    per_chunk_book = (snaps.shape[1] - n_wiki) // n_books
    out = {}
    for label, d in (("last_half", np.mean([snaps[k] - snaps[0] for k in half], axis=0)), ("final", snaps[-1] - snaps[0])):
        wiki = d[:n_wiki]
        bk = [d[n_wiki + per_chunk_book * b:n_wiki + per_chunk_book * (b + 1)].mean() for b in range(n_books)]
        out.setdefault("wiki", {}).update({label: float(wiki.mean()), label + "_se": se(wiki)})
        out.setdefault("books", {}).update({label: float(np.mean(bk)), label + "_se": se(bk)})
    return out


def forgetting_curve(snaps, n_wiki, n_books=6):
    """Per snapshot: (WikiText forgetting, held-out books forgetting), means of snapshot - pretrained."""
    snaps = np.asarray(snaps, dtype=np.float64)
    d = snaps - snaps[0]
    return d[:, :n_wiki].mean(1), d[:, n_wiki:].mean(1)


def exchange_rate(stream_gain, wiki_forgetting_last_half):
    """Stream gain per nat of last-half mean WikiText forgetting (None if there is no forgetting)."""
    return stream_gain / wiki_forgetting_last_half if wiki_forgetting_last_half > 0 else None
