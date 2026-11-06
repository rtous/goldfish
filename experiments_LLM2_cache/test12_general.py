"""Test 12: forgetting of general knowledge (a "thermometer").

While a model learns a stream online, snapshots of its weights are scored frozen on fixed held-out
text that is never trained on:
    wikitext  WikiText-103 test split (Salesforce/wikitext, wikitext-103-raw-v1, via datasets), the
              articles joined in order, tokenized, first 20480 tokens (80 chunks)
    books     the 6 val-meta-heldout PG-19 validation books (dual.val_sets), tokens 0-3839 of each
              (15 chunks each, 90 chunks)
Snapshot every EVERY chunks (and at the end); snapshot 0 is the pretrained model.
    forgetting(t) = loss_snapshot(held-out) - loss_frozen(held-out)   (positive: knowledge lost)
SE: paired over the 80 WikiText chunks, or over the 6 held-out books.

Streams: test 7's 25-book chunk set (78 chunks per book), K1 (book after book) and shuffled (test 7's
order, seed 0); and test 10's very long set (4 books x 200k, K1). Runs:
    Pythia-14M: full_1e-4, mlp4_3e-3, mlp4_1e-2, fast4_d0.99 (test 7) and dual (test 10's pick: fast
                part on layer 2's MLP, alpha 1e-3, lambda 0.99), each on K1 and shuffled
    Pythia-31M: full online learning at lr 3e-4, K1 and shuffled
    Pythia-14M full_1e-4 on the very long books
Each run's per-token stream losses are compared with the saved losses of the same run in tests 7 and
10 (float16), which checks that the learner is the same; the stream gain uses the saved frozen losses.
Exchange rate = stream gain / mean WikiText forgetting over the snapshots in the last half of the
stream.

Writes test12_general/: losses_<run>.npz (stream losses float16 [book, chunk, L]; per-chunk mean
held-out losses per snapshot), results.csv, summary.csv, k1_vs_shuffled.csv, runtime.txt, plots/.
Usage:
    python test12_general.py run <run> [<run> ...]   # runs (names in RUNS)
    python test12_general.py smoke                   # timing on the first 80 chunks
    python test12_general.py report                  # tables and plots from the saved files
    python test12_general.py rescore <run> [...]     # rerun, scoring detokenized WikiText (and raw per token)
    python test12_general.py detok_report            # raw vs detokenized tables, token-type breakdown
"""
import csv
import json
import os
import sys
import time

import numpy as np

from data import DATA, HERE, books, long_books, val_books
from dual import N_PICK, TRAIN_SHARE, mlp_weights
from models import DEFAULT
from online import L, setup
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from snapshot import Learner, score
from test7_aggressive import chunk, orders

OUT = os.path.join(HERE, "test12_general")
EVERY = 40
N_WIKI, BOOK_TOKENS = 80 * L, 15 * L
MID = "EleutherAI/pythia-31m"
SETTINGS = {"full_1e-4": dict(rule="sgd", lr=1e-4), "mlp4_3e-3": dict(rule="sgd", lr=3e-3, params="mlp@3"),
            "mlp4_1e-2": dict(rule="sgd", lr=1e-2, params="mlp@3"),
            "fast4_d0.99": dict(rule="decay", lr=1e-2, params="mlp@3", decay=0.99),
            "dual": dict(rule="dual", fast={n: (1e-3, 0.99) for n in mlp_weights([1])}),
            "full_3e-4": dict(rule="sgd", lr=3e-4)}
# run name -> (size, setting, stream, saved losses (file, key) of the same run, saved frozen losses)
T7, T10 = os.path.join(HERE, "test7_aggressive", "losses.npz"), os.path.join(HERE, "test10_followups")
RUNS = {}
for s in ("full_1e-4", "mlp4_3e-3", "mlp4_1e-2", "fast4_d0.99", "dual"):
    for st in ("K1", "shuffled"):
        ref = (os.path.join(T10, "losses_B.npz"), "mlp2_a0.001_l0.99_" + st) if s == "dual" else (T7, "%s_%s" % (s, st))
        RUNS["14m_%s_%s" % (s, st)] = ("14m", s, st, ref, (T7, "frozen14m_books25"))
for st in ("K1", "shuffled"):
    RUNS["31m_full_3e-4_" + st] = ("31m", "full_3e-4", st, None, (T7, "frozen31m_books25"))
RUNS["14m_full_1e-4_verylong"] = ("14m", "full_1e-4", "verylong", (os.path.join(T10, "losses_C.npz"), "14m_full_1e-4"),
                                  (os.path.join(T10, "losses_C.npz"), "frozen14m"))


def wikitext_ids():
    path = os.path.join(DATA, "wikitext103_test_%s.npy" % DEFAULT.replace("/", "__"))
    if not os.path.exists(path):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(DEFAULT, cache_dir=os.path.join(DATA, "hf"))
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", cache_dir=os.path.join(DATA, "hf"))
        text = "".join(ds["text"])
        ids = tok(text[:200000], add_special_tokens=False)["input_ids"]
        assert len(ids) > N_WIKI + 10
        np.save(path, np.array(ids[:N_WIKI], dtype=np.int64))
    return np.load(path)


RAW_TOKEN_RUNS = ("14m_full_1e-4_K1", "14m_full_1e-4_shuffled", "31m_full_3e-4_K1", "31m_full_3e-4_shuffled")


def detokenize(t):
    """WikiText detokenizer of lm-evaluation-harness (wikitext_detokenizer), as used for WikiText
    perplexity there. "= Heading =" lines are kept and detokenized ("= = Career = =" -> "== Career ==")."""
    import re
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
    """The same WikiText text as wikitext_ids() (the characters behind its 20480 tokens), detokenized,
    tokenized and cut to a multiple of L."""
    path = os.path.join(DATA, "wikitext103_test_detok_%s.npy" % DEFAULT.replace("/", "__"))
    if not os.path.exists(path):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(DEFAULT, cache_dir=os.path.join(DATA, "hf"))
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", cache_dir=os.path.join(DATA, "hf"))
        text = "".join(ds["text"])[:200000]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert (np.array(enc["input_ids"][:N_WIKI]) == wikitext_ids()).all()
        raw = text[:enc["offset_mapping"][N_WIKI - 1][1]]
        ids = tok(detokenize(raw), add_special_tokens=False)["input_ids"]
        np.save(path, np.array(ids[:len(ids) // L * L], dtype=np.int64))
    return np.load(path)


def token_types(ids):
    """Type per raw WikiText token: markup (@-@ style markers, '=' heading markers, a space before
    punctuation), whitespace (spaces and newlines only), digits (contains a digit), other."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DEFAULT, cache_dir=os.path.join(DATA, "hf"))
    out = []
    for i in ids.tolist():
        s = tok.decode([i])
        core = s.strip()
        if "@" in s or (core and set(core) <= {"="}) or (s.startswith(" ") and core and all(not c.isalnum() for c in core)):
            out.append("markup")
        elif not core:
            out.append("whitespace")
        elif any(c.isdigit() for c in s):
            out.append("digits")
        else:
            out.append("other")
    return np.array(out)


def rescore(name):
    """Rerun a run and score, at the same snapshots, the detokenized WikiText (per-chunk means) and, for
    RAW_TOKEN_RUNS, the raw WikiText per token. losses_detok_<run>.npz."""
    size, setting, stream, _, _ = RUNS[name]
    model, init, info, _ = setup(DEFAULT if size == "14m" else MID)
    chunks, order = stream_of(stream)
    ids = np.concatenate([chunks[b, k] for b, k in order])
    detok, raw = wikitext_detok_ids(), wikitext_ids()
    keep_raw = name in RAW_TOKEN_RUNS
    t0 = time.time()
    lrn = Learner(model, init, **SETTINGS[setting])
    det, rawtok, pos = [chunk_means(score(model, init, detok))], [], [0]
    if keep_raw:
        rawtok.append(score(model, init, raw))
    for s in range(0, len(order), EVERY):
        e = min(s + EVERY, len(order))
        lrn.read(ids[s * L:e * L])
        det.append(chunk_means(lrn.score(detok)))
        if keep_raw:
            rawtok.append(lrn.score(raw))
        pos.append(e)
    extra = {}
    if keep_raw:
        old = np.load(os.path.join(OUT, "losses_%s.npz" % name))["snaps"][:, :N_WIKI // L]
        new = np.array([chunk_means(r) for r in rawtok], dtype=np.float32)
        extra["raw_reproduces_max_abs_diff"] = float(np.abs(old - new).max())
        print(name, "raw WikiText chunk means vs saved: max abs diff %.2e" % extra["raw_reproduces_max_abs_diff"], flush=True)
    meta = {"run": name, "seconds": round(time.time() - t0), "detok_tokens": int(len(detok)), **extra}
    np.savez_compressed(os.path.join(OUT, "losses_detok_%s.npz" % name), detok=np.array(det, dtype=np.float32),
                        positions=np.array(pos) * L, meta=json.dumps(meta),
                        **({"raw_tokens": np.array(rawtok, dtype=np.float16)} if keep_raw else {}))
    print(meta, flush=True)


def detok_report():
    """Raw against detokenized WikiText forgetting per run (detok.csv), raw forgetting by token type for
    RAW_TOKEN_RUNS (breakdown.csv), plots/raw_vs_detok.png."""
    rows, brk = [], []
    raw_ids = wikitext_ids()
    types = token_types(raw_ids)
    for name in RUNS:
        p, pd = os.path.join(OUT, "losses_%s.npz" % name), os.path.join(OUT, "losses_detok_%s.npz" % name)
        if not (os.path.exists(p) and os.path.exists(pd)):
            continue
        z, zd = np.load(p), np.load(pd)
        assert (z["positions"] == zd["positions"]).all()
        pos = z["positions"]
        half = [k for k in range(1, len(pos)) if pos[k] > pos[-1] / 2]
        r = {"run": name}
        gain = None
        summ = os.path.join(OUT, "summary.csv")
        if os.path.exists(summ):
            gain = next((float(x["stream_gain"]) for x in csv.DictReader(open(summ)) if x["run"] == name), None)
        for which, snaps in (("raw", z["snaps"][:, :N_WIKI // L].astype(np.float64)), ("detok", zd["detok"].astype(np.float64))):
            lh = np.mean([snaps[k] - snaps[0] for k in half], axis=0)
            end = snaps[-1] - snaps[0]
            r.update({"frozen_%s" % which: "%.4f" % snaps[0].mean(), "last_half_%s" % which: "%.4f" % lh.mean(),
                      "last_half_%s_se" % which: "%.4f" % se(lh), "end_%s" % which: "%.4f" % end.mean(),
                      "end_%s_se" % which: "%.4f" % se(end),
                      "exchange_rate_%s" % which: ("%.2f" % (gain / lh.mean())) if gain is not None and lh.mean() > 0 else ""})
        lh_diff = np.mean([(z["snaps"][k, :N_WIKI // L] - z["snaps"][0, :N_WIKI // L]).mean() - (zd["detok"][k] - zd["detok"][0]).mean()
                           for k in half])
        r["raw_minus_detok_last_half"] = "%.4f" % lh_diff
        rows.append(r)
        if "raw_tokens" in zd.files:
            rt = zd["raw_tokens"].astype(np.float64)
            assert len(rt) == len(pos)
            for label, d in (("last_half", np.mean([rt[k] - rt[0] for k in half], axis=0)), ("end", rt[-1] - rt[0])):
                ok = np.isfinite(d)
                total = d[ok].sum() / ok.sum()
                for t in ("markup", "whitespace", "digits", "other"):
                    m = ok & (types == t)
                    brk.append({"run": name, "when": label, "type": t, "token_share": "%.3f" % (m.sum() / ok.sum()),
                                "frozen_loss": "%.4f" % rt[0][m].mean(), "forgetting_per_token": "%.4f" % d[m].mean(),
                                "contribution": "%.4f" % (d[m].sum() / ok.sum()),
                                "share_of_forgetting": "%.3f" % (d[m].sum() / ok.sum() / total)})
    write_csv("detok.csv", rows)
    if brk:
        write_csv("breakdown.csv", brk)
    for r in rows + brk:
        print(r)
    style()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for r in rows:
        size, setting, stream, _, _ = RUNS[r["run"]]
        x, y = float(r["last_half_raw"]), float(r["last_half_detok"])
        ax.scatter(x, y, color=COLORS[setting], marker={"K1": "o", "shuffled": "s", "verylong": "^"}[stream],
                   s=28 if size == "14m" else 50, edgecolor=INK if size == "31m" else "none")
        ax.annotate("%s %s %s" % (size.upper(), setting, stream), (x, y), xytext=(4, 2), textcoords="offset points",
                    fontsize=5.5, color=MUTED)
    lim = max(max(float(r["last_half_raw"]), float(r["last_half_detok"])) for r in rows) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("raw WikiText forgetting (last-half mean)")
    ax.set_ylabel("detokenized WikiText forgetting")
    ax.set_title("Same articles, raw against detokenized (dotted: equal)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "raw_vs_detok.png"), dpi=160)


def heldout():
    """(token ids [80 + 90 chunks * L], set name per chunk, held-out book per chunk or -1)."""
    ids = val_books(DEFAULT)["ids"]
    n_rest = len(ids) - N_PICK
    held_books = ids[N_PICK + round(TRAIN_SHARE * n_rest):]
    assert len(held_books) == 6
    toks = np.concatenate([wikitext_ids()] + [b[:BOOK_TOKENS] for b in held_books])
    n_b = BOOK_TOKENS // L
    sets = np.array(["wikitext"] * (N_WIKI // L) + ["books"] * (6 * n_b))
    book = np.r_[np.full(N_WIKI // L, -1), np.repeat(np.arange(6), n_b)]
    return toks, sets, book


def stream_of(stream):
    if stream == "verylong":
        c = chunk(long_books(DEFAULT, n_books=4, n_tokens=200000)["ids"])
        order = [(b, k) for b in range(c.shape[0]) for k in range(c.shape[1])]
    else:
        c = chunk(books(DEFAULT)["ids"][:25])
        order = orders(25, c.shape[1])[stream]
    return c, order


def chunk_means(loss):
    return np.nanmean(loss.reshape(-1, L), axis=1)


def run(name, max_chunks=None):
    size, setting, stream, ref, _ = RUNS[name]
    model, init, info, _ = setup(DEFAULT if size == "14m" else MID)
    chunks, order = stream_of(stream)
    if max_chunks:
        order = order[:max_chunks]
    ids = np.concatenate([chunks[b, k] for b, k in order])
    held, _, _ = heldout()
    t0 = time.time()
    lrn = Learner(model, init, **SETTINGS[setting])
    snaps, pos, losses = [chunk_means(score(model, init, held))], [0], []
    t_online = t_snap = 0.0
    for s in range(0, len(order), EVERY):
        e = min(s + EVERY, len(order))
        t = time.time()
        losses.append(lrn.read(ids[s * L:e * L]))
        t_online += time.time() - t
        t = time.time()
        snaps.append(chunk_means(lrn.score(held)))
        t_snap += time.time() - t
        pos.append(e)
    loss = np.concatenate(losses).reshape(-1, L)
    out = np.full(chunks.shape, np.nan, dtype=np.float32)
    for i, (b, k) in enumerate(order):
        out[b, k] = loss[i]
    check = {}
    if ref is not None and not max_chunks:
        saved = np.load(ref[0])[ref[1]]
        mine = out.astype(np.float16)
        m = np.isfinite(saved)
        check = {"reference": "%s:%s" % (os.path.relpath(ref[0], HERE), ref[1]),
                 "identical_share": float((mine[m] == saved[m]).mean()),
                 "max_abs_diff": float(np.abs(mine[m].astype(np.float32) - saved[m].astype(np.float32)).max())}
        print(name, "reproduces saved run:", check, flush=True)
    meta = {"run": name, "size": size, "setting": setting, "stream": stream, "seconds_online": round(t_online),
            "seconds_snapshots": round(t_snap), "snapshots": len(pos), "device": info["device"], **check}
    if max_chunks:
        return meta
    np.savez_compressed(os.path.join(OUT, "losses_%s.npz" % name), stream=out.astype(np.float16),
                        snaps=np.array(snaps, dtype=np.float32), positions=np.array(pos) * L, meta=json.dumps(meta))
    print(meta, "total %.0f s" % (time.time() - t0), flush=True)


def smoke():
    for name in ("14m_full_1e-4_K1", "31m_full_3e-4_K1"):
        m = run(name, max_chunks=2 * EVERY)
        full = 3124 if "verylong" in name else 1950
        n_seg = full / EVERY
        est = (m["seconds_online"] / (2 * EVERY)) * full + (m["seconds_snapshots"] / 2) * (n_seg + 1)
        print("%s: %s; estimated full run %.0f min" % (name, m, est / 60), flush=True)


def se(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def forgetting(snaps, sets, hbook, k, which):
    """(mean, SE) of snapshot k minus snapshot 0: paired over WikiText chunks, or over the 6 books."""
    d = snaps[k] - snaps[0]
    if which == "wikitext":
        x = d[sets == "wikitext"]
        return float(x.mean()), se(x)
    per = [d[hbook == b].mean() for b in range(6)]
    return float(np.mean(per)), se(per)


def load_runs():
    out = {}
    for name in RUNS:
        p = os.path.join(OUT, "losses_%s.npz" % name)
        if os.path.exists(p):
            z = np.load(p)
            out[name] = {"stream": z["stream"].astype(np.float64), "snaps": z["snaps"].astype(np.float64),
                         "positions": z["positions"], "meta": json.loads(str(z["meta"]))}
    return out


def report():
    _, sets, hbook = heldout()
    runs = load_runs()
    rows, summary = [], []
    for name, r in runs.items():
        size, setting, stream, ref, fref = RUNS[name]
        snaps, pos = r["snaps"], r["positions"]
        for k in range(len(pos)):
            for which in ("wikitext", "books"):
                m = sets == which
                f, s = forgetting(snaps, sets, hbook, k, which)
                rows.append({"run": name, "position": int(pos[k]), "set": which, "frozen_loss": "%.4f" % snaps[0][m].mean(),
                             "snapshot_loss": "%.4f" % snaps[k][m].mean(), "forgetting": "%.4f" % f, "se": "%.4f" % s})
        frozen = np.load(fref[0])[fref[1]].astype(np.float64)
        diff = frozen - r["stream"]
        per = [np.nanmean(diff[b]) for b in range(diff.shape[0])]
        gain, gain_se = float(np.nanmean(diff)), se(per)
        row = {"run": name, "size": size, "setting": setting, "stream": stream, "stream_gain": "%.4f" % gain,
               "stream_gain_se": "%.4f" % gain_se}
        last = len(pos) - 1
        half = [k for k in range(1, len(pos)) if pos[k] > pos[-1] / 2]
        for which in ("wikitext", "books"):
            m = sets == which
            f, s = forgetting(snaps, sets, hbook, last, which)
            spread = np.std([forgetting(snaps, sets, hbook, k, which)[0] for k in range(last - 4, last + 1)], ddof=1)
            mh = float(np.mean([forgetting(snaps, sets, hbook, k, which)[0] for k in half]))
            peak = max(forgetting(snaps, sets, hbook, k, which)[0] for k in range(1, len(pos)))
            row.update({"frozen_%s" % which: "%.4f" % snaps[0][m].mean(), "final_forgetting_%s" % which: "%.4f" % f,
                        "final_forgetting_%s_se" % which: "%.4f" % s, "last5_sd_%s" % which: "%.4f" % spread,
                        "mean_forgetting_last_half_%s" % which: "%.4f" % mh, "max_forgetting_%s" % which: "%.4f" % peak,
                        "exchange_rate_%s" % which: ("%.2f" % (gain / mh)) if mh > 0 else "no forgetting"})
        row["reproduces_saved"] = r["meta"].get("identical_share", "")
        row["max_abs_diff_saved"] = r["meta"].get("max_abs_diff", "")
        row["seconds"] = r["meta"]["seconds_online"] + r["meta"]["seconds_snapshots"]
        summary.append(row)
    write_csv("results.csv", rows)
    write_csv("summary.csv", summary)
    comp = []
    for name, r in runs.items():  # K1 vs shuffled, paired over held-out chunks / books at matching positions
        if not name.endswith("_K1") or name.replace("_K1", "_shuffled") not in runs:
            continue
        sh = runs[name.replace("_K1", "_shuffled")]
        k1s = next(x for x in summary if x["run"] == name)
        shs = next(x for x in summary if x["run"] == name.replace("_K1", "_shuffled"))
        row = {"setting": name[:-3], "gain_K1": k1s["stream_gain"], "gain_shuffled": shs["stream_gain"]}
        for which in ("wikitext", "books"):
            m = sets == which
            half = [k for k in range(1, len(r["positions"])) if r["positions"][k] > r["positions"][-1] / 2]
            d = np.mean([sh["snaps"][k] - r["snaps"][k] for k in half], axis=0)  # shuffled - K1, per chunk
            if which == "wikitext":
                x = d[m]
            else:
                x = [d[hbook == b].mean() for b in range(6)]
            row.update({"forgetting_last_half_K1_%s" % which: k1s["mean_forgetting_last_half_%s" % which],
                        "forgetting_last_half_shuffled_%s" % which: shs["mean_forgetting_last_half_%s" % which],
                        "shuffled_minus_K1_%s" % which: "%.4f" % np.mean(x), "shuffled_minus_K1_%s_se" % which: "%.4f" % se(x)})
        comp.append(row)
    write_csv("k1_vs_shuffled.csv", comp)
    for r in summary:
        print({k: v for k, v in r.items() if k not in ("size", "setting", "stream")})
    for r in comp:
        print(r)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for r in runs.values():
            f.write(json.dumps(r["meta"]) + "\n")
    plot(runs, sets, hbook, summary)


COLORS = {"full_1e-4": BLUE, "mlp4_3e-3": ORANGE, "mlp4_1e-2": "#b8442a", "fast4_d0.99": "#8a5cc2", "dual": "#1baa8a",
          "full_3e-4": INK}


def plot(runs=None, sets=None, hbook=None, summary=None):
    if runs is None:
        _, sets, hbook = heldout()
        runs = load_runs()
        summary = list(csv.DictReader(open(os.path.join(OUT, "summary.csv"))))
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
    for name, r in runs.items():
        size, setting, stream, _, _ = RUNS[name]
        if stream == "verylong":
            continue
        col = 0 if stream == "K1" else 1
        for row, which in enumerate(("wikitext", "books")):
            ax = axes[row, col]
            y = [forgetting(r["snaps"], sets, hbook, k, which)[0] for k in range(len(r["positions"]))]
            ax.plot(r["positions"] / 1000, y, color=COLORS[setting], lw=1.8 if size == "14m" else 2.2,
                    ls="-" if size == "14m" else "--", label="%s %s" % (size.upper(), setting))
    for (row, col), ax in np.ndenumerate(axes):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title("%s stream, held-out %s" % (("K = 1", "shuffled")[col], ("WikiText", "PG-19 books")[row]),
                     fontsize=8.5, color=INK, loc="left")
        if row == 1:
            ax.set_xlabel("position in the stream (thousands of tokens)")
        if col == 0:
            ax.set_ylabel("forgetting (nats/token)")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Snapshot loss - pretrained loss on held-out text (positive: knowledge lost)", fontsize=9, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "forgetting_vs_position.png"), dpi=160)

    if "14m_full_1e-4_verylong" in runs:
        r = runs["14m_full_1e-4_verylong"]
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        for which, ls in (("wikitext", "-"), ("books", "--")):
            y = [forgetting(r["snaps"], sets, hbook, k, which)[0] for k in range(len(r["positions"]))]
            ax.plot(r["positions"] / 1000, y, color=BLUE, ls=ls, lw=1.8, label="held-out " + which)
        for b in range(1, 4):
            ax.axvline(b * 781 * L / 1000, color="#c8c7c3", lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("position in the stream (thousands of tokens; grey lines: book boundaries)")
        ax.set_ylabel("forgetting (nats/token)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Pythia-14M full_1e-4 on 4 books x 200k", fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "forgetting_verylong.png"), dpi=160)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, which in zip(axes, ("wikitext", "books")):
        for r in summary:
            if r["stream"] == "verylong":
                continue
            x, y = float(r["mean_forgetting_last_half_%s" % which]), float(r["stream_gain"])
            ax.scatter(x, y, color=COLORS[r["setting"]], marker="o" if r["stream"] == "K1" else "s",
                       s=28 if r["size"] == "14m" else 50, edgecolor=INK if r["size"] == "31m" else "none", lw=0.8)
            ax.annotate("%s %s %s" % (r["size"].upper(), r["setting"], r["stream"]), (x, y), xytext=(4, 2),
                        textcoords="offset points", fontsize=6, color=MUTED)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("forgetting on held-out %s (mean over the last half of the stream)" % which)
        ax.set_ylabel("stream gain over frozen (nats/token)")
    axes[0].set_title("circles: K = 1, squares: shuffled; outlined: 31M", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_forgetting.png"), dpi=160)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "run":
        for n in sys.argv[2:]:
            run(n)
    elif cmd == "rescore":
        for n in sys.argv[2:]:
            rescore(n)
    elif cmd == "detok_report":
        detok_report()
    elif cmd == "smoke":
        smoke()
    elif cmd == "report":
        report()
    elif cmd == "plot":
        plot()
