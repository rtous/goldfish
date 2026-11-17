"""E8: how much of the online learner's gain does the frozen model get for free from its attention window?

The other experiments score each 256-token chunk with no earlier context, so any help from earlier
chunks must come through the weights. Here each chunk (book b, chunk k) is scored with the C tokens
that precede it in the same book (chunks[b, :k] concatenated, last C tokens; never across a book
boundary, so the first chunks of a book get less) in the input. Only the chunk's tokens are scored, and
its first token stays unscored (NaN), so the scored tokens are exactly those of the registry runs and
every comparison is paired per token. C in CONTEXTS; C + L <= max_position_embeddings (2048 in Pythia).

Runs (saved in e8_full_context/runs/<key>.npz, the layout of runs.run: loss, order, meta):
    part A  frozen, main set in book order (main_K1), 14M / 31M / 70M, each C; frozen, very long set,
            14M / 31M, C = 1792
    part B  14M online with context: plain SGD on all parameters at the E0 pick (picked without
            context), main_K1, C = 1792. Each chunk is scored with its context, then one SGD step on the
            mean loss of the chunk's scored tokens (context tokens are input, not loss). Reset at the start.
Frozen chunks are batched by context length (all inputs in a batch have the same length, no padding).
Checks: C = 0 must reproduce the registry frozen run (runs.compare) before anything else runs; batched
scoring against chunks scored one at a time (a few chunks per run, max abs diff in meta); part B at
C = 0 on the first chunks against the registry online run.

Tables (report; only saved runs; gains = frozen no-context loss - run loss, metrics.summary, SE over
books): results.csv, distance.csv (14M, E1 bins), window.csv (14M, earlier-chunk tokens split at the
2048-token window edge, same book / earlier book), verylong.csv (share of the 14M-31M gap closed, with
and without context), plots/gain_vs_context.png, plots/gain_by_distance.png, runtime.txt.
Usage (from experiments/):
    ../venv/bin/python e8_full_context/e8_full_context.py run a      # sanity check, then part A
    ../venv/bin/python e8_full_context/e8_full_context.py run b      # part B
    ../venv/bin/python e8_full_context/e8_full_context.py run <key> [...]
    ../venv/bin/python e8_full_context/e8_full_context.py report
"""
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import runs  # noqa: E402
from data import L, chunk_set, stream  # noqa: E402
from metrics import BINS, bin_masks, distance_bins, prev_occurrence, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(OUT, "runs")
CONTEXTS = (0, 256, 768, 1792)
WINDOW = 2048        # 1792 + L: the farthest token the input at C = 1792 reaches
TOKENS_PER_BATCH = 8192
N_SINGLE = 4          # chunks rescored one at a time per frozen run (the batching check)
N_ONLINE_CHECK = 20   # chunks of the part B code at C = 0 compared with the registry online run


def key_of(size, rule, stream_name, ctx):
    if rule == "frozen":
        return "%s__%s_frozen_ctx%d" % (stream_name, size, ctx)
    return "%s__%s_sgd_%g_all_ctx%d" % (stream_name, size, runs.picks()[(size, "sgd", "all")], ctx)


def specs(part):
    if part == "a":
        out = [(s, "frozen", "main_K1", c) for s in runs.SIZES for c in CONTEXTS]
        return out + [(s, "frozen", "verylong", 1792) for s in ("14m", "31m")]
    if part == "b":
        return [("14m", "sgd", "main_K1", 1792)]
    raise ValueError(part)


def path_of(key):
    return os.path.join(RUNS, key + ".npz")


# ---------------------------------------------------------------- scoring with context


def inputs(chunks, b, k, ctx):
    """Context (last ctx tokens of chunks[b, :k]) followed by chunk (b, k): int64 array."""
    n = min(k, ctx // L)
    return np.concatenate([chunks[b, k - n:k].reshape(-1), chunks[b, k]])


def chunk_nll(model, xb):
    """Per-token loss [B, L - 1] of the last L tokens of each row of xb (the chunk), each predicted from
    everything before it in the row; the chunk's first token is not scored."""
    logits = model(input_ids=xb, logits_to_keep=L).logits
    return F.cross_entropy(logits[:, :-1].transpose(1, 2), xb[:, -L + 1:], reduction="none")


def frozen_losses(model, chunks, order, ctx, batch=None):
    """[len(order), L] float32, NaN in column 0. Chunks grouped by context length, batched."""
    if batch is None:
        batch = 16 if ctx == 0 else max(1, TOKENS_PER_BATCH // (ctx + L))  # 16 = stream_losses' batch
    out = np.full((len(order), L), np.nan, dtype=np.float32)
    by_len = {}
    for i, (b, k) in enumerate(order):
        by_len.setdefault(min(k, ctx // L), []).append(i)
    with torch.no_grad():
        for idx in by_len.values():
            for a in range(0, len(idx), batch):
                rows = idx[a:a + batch]
                xb = torch.from_numpy(np.stack([inputs(chunks, *order[i], ctx) for i in rows]))
                out[rows, 1:] = chunk_nll(model, xb).numpy()
    return out


def online_losses(model, init, chunks, order, ctx, lr, n=None):
    """Prequential plain SGD on all parameters, from init: score each chunk with its context, then one
    step on the mean loss of its scored tokens. n: only the first n chunks (the check)."""
    model.load_state_dict(init)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    order = order[:n] if n else order
    out = np.full((len(order), L), np.nan, dtype=np.float32)
    for i, (b, k) in enumerate(order):
        # 2-D cross entropy, as stream_losses' SGD path: the batched [B, V, L] layout differs by ~2e-4,
        # which SGD amplifies to ~0.9 nats within 20 chunks
        xb = torch.from_numpy(inputs(chunks, b, k, ctx)).unsqueeze(0)
        nll = F.cross_entropy(model(input_ids=xb, logits_to_keep=L).logits[0, :-1], xb[0, -L + 1:], reduction="none")
        out[i, 1:] = nll.detach().numpy()
        opt.zero_grad(set_to_none=True)
        nll.mean().backward()
        opt.step()
    return out


def sanity(size, loss):
    """C = 0 against the registry frozen run; stops if neither identical nor within float16 rounding."""
    old = runs.load(runs.frozen_key(size, "main_K1"))["loss"]
    chunks, order = stream("main_K1")
    new = np.full(old.shape, np.nan)
    o = np.array(order)
    new[o[:, 0], o[:, 1]] = loss
    c = runs.compare(new, old)
    same_nan = bool((np.isnan(new) == np.isnan(old)).all())
    print("sanity %s C=0 vs %s: %s, same NaNs %s" % (size, runs.frozen_key(size, "main_K1"), c, same_nan), flush=True)
    # float16 rounding: half an ulp at loss ~ 8-16 is 2^-8; allow one ulp
    if not same_nan or c["max_abs_diff"] > 2 ** -7:
        raise SystemExit("SANITY CHECK FAILED for %s: %s" % (size, c))
    return c


def run(spec):
    from learner import setup
    size, rule, stream_name, ctx = spec
    key = key_of(*spec)
    model, init, info = setup(size)
    assert ctx + L <= model.config.max_position_embeddings, (ctx, model.config.max_position_embeddings)
    chunks, order = stream(stream_name)
    extra = {}
    t0 = time.time()
    if rule == "frozen":
        loss = frozen_losses(model, chunks, order, ctx)
    else:
        lr = runs.picks()[(size, "sgd", "all")]
        loss = online_losses(model, init, chunks, order, ctx, lr)
    sec = time.time() - t0
    if rule == "frozen":
        if ctx == 0 and stream_name == "main_K1":
            extra["sanity"] = sanity(size, loss)
        # batching check: a few chunks (largest context, and one with a short context) one at a time
        pick = sorted(set([len(order) - 1, len(order) // 2, 1, min(2, len(order) - 1)]))[:N_SINGLE]
        single = frozen_losses(model, chunks, [order[i] for i in pick], ctx, batch=1)
        extra["batch_check_max_abs_diff"] = float(np.nanmax(np.abs(single - loss[pick])))
        print("%s: batched vs one at a time, max abs diff %.2e" % (key, extra["batch_check_max_abs_diff"]))
    else:
        lr = runs.picks()[(size, "sgd", "all")]
        ref = runs.load(runs.find(size, "sgd", stream_name))["loss"]
        chk = online_losses(model, init, chunks, order, 0, lr, n=N_ONLINE_CHECK)
        o = np.array(order[:N_ONLINE_CHECK])
        extra["online_c0_check"] = runs.compare(chk, ref[o[:, 0], o[:, 1]])
        print("%s: part B code at C=0, first %d chunks vs registry online run: %s"
              % (key, N_ONLINE_CHECK, extra["online_c0_check"]))
    meta = {"size": size, "rule": rule, "stream": stream_name, "context": ctx,
            "lr": 0.0 if rule == "frozen" else lr, "params": "all", "seconds": round(sec), "device": info["device"],
            "commit": info["commit"], "threads": torch.get_num_threads(), **extra}
    os.makedirs(RUNS, exist_ok=True)
    np.savez_compressed(path_of(key), loss=loss.astype(np.float16), order=np.array(order, dtype=np.int16),
                        meta=json.dumps(meta))
    print("%s: mean loss %.4f, %.0f s" % (key, np.nanmean(loss), sec), flush=True)


def run_cmd(args):
    todo = []
    for a in args:
        todo += specs(a) if a in ("a", "b") else [s for p in "ab" for s in specs(p) if key_of(*s) == a]
    # the sanity checks (C = 0) first: a failure stops everything
    todo.sort(key=lambda s: (s[3] != 0, 0))
    for s in todo:
        if os.path.exists(path_of(key_of(*s))):
            continue
        run(s)


# ---------------------------------------------------------------- report


def load(key):
    """Like runs.load, from this experiment's runs/."""
    z = np.load(path_of(key))
    order = z["order"].astype(np.int64)
    loss = np.full((order[:, 0].max() + 1, order[:, 1].max() + 1, L), np.nan)
    loss[order[:, 0], order[:, 1]] = z["loss"].astype(np.float64)
    return {"loss": loss, "meta": json.loads(str(z["meta"]))}


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def fmt(s, prefix):
    return {prefix: "%.4f" % s["mean"], prefix + "_se": "%.4f" % s["se"], prefix + "_under_2se": s["under_2se"]}


def report():
    from plots import plt, style, BLUE, ORANGE, AQUA, INK, MUTED
    have = {key_of(*s): s for p in "ab" for s in specs(p) if os.path.exists(path_of(key_of(*s)))}
    flat = {k: load(k)["loss"].reshape(-1) for k in have}
    base, online, rows = {}, {}, []
    for k, (size, rule, st, ctx) in sorted(have.items(), key=lambda kv: (kv[1][2], kv[1][0], kv[1][1], kv[1][3])):
        if (size, st) not in base:
            base[size, st] = runs.load(runs.frozen_key(size, st))["loss"].reshape(-1)
            ok = runs.find(size, "sgd", st)
            online[size, st] = (ok, runs.load(ok)["loss"].reshape(-1))
        f0 = base[size, st]
        n_books = len(f0) // (chunk_set("main" if st == "main_K1" else st).shape[1] * L)
        book = np.repeat(np.arange(n_books), len(f0) // n_books)
        allm = np.ones(len(f0), bool)
        r = {"size": size, "stream": st, "context": ctx, "rule": rule, "key": k, "loss": "%.4f" % np.nanmean(flat[k]),
             "frozen_c0_loss": "%.4f" % np.nanmean(f0)}
        r.update(fmt(summary(f0 - flat[k], book, allm), "gain"))
        r.update(fmt(summary(f0 - online[size, st][1], book, allm), "online_c0_gain"))
        r["online_c0_key"] = online[size, st][0]
        r["books"] = summary(f0 - flat[k], book, allm)["books"]
        rows.append(r)
        print({kk: v for kk, v in r.items() if not kk.endswith("under_2se") and kk != "key"})
    write_csv("results.csv", rows)

    # distance bins, 14M, main_K1
    chunks = chunk_set("main")
    book, masks = bin_masks(chunks)
    f0 = base["14m", "main_K1"]
    series = {"frozen_ctx%d" % c: f0 - flat[key_of("14m", "frozen", "main_K1", c)] for c in CONTEXTS[1:]}
    series["online_ctx0"] = f0 - online["14m", "main_K1"][1]
    kb = key_of("14m", "sgd", "main_K1", 1792)
    if kb in flat:
        series["online_ctx1792"] = f0 - flat[kb]
        series["online_ctx1792_minus_frozen_ctx1792"] = series["online_ctx1792"] - series["frozen_ctx1792"]
    drows = []
    for name, m in masks.items():
        m = m & np.isfinite(f0)
        r = {"bin": name, "tokens": int(m.sum()), "frozen_ctx0_loss": "%.4f" % f0[m].mean()}
        for sk, x in series.items():
            r.update(fmt(summary(x, book, m), sk if "_minus_" in sk else "gain_" + sk))
        drows.append(r)
        print({kk: v for kk, v in r.items() if not kk.endswith("under_2se")})
    write_csv("distance.csv", drows)

    # the window edge: earlier-chunk bins (same exclusions as E1) split at WINDOW tokens back, the most the
    # input at C = 1792 can reach (1792 of context + 256 of chunk)
    ids = chunks.reshape(-1)
    prev = prev_occurrence(ids)
    dist = np.arange(len(ids)) - prev
    bins = distance_bins(ids)
    earlier = (bins >= 1) & (bins <= 4)
    same = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    wrows = []
    for name, mk in (("same book, 1k-%d" % WINDOW, earlier & same & (dist >= 1000) & (dist < WINDOW)),
                     ("same book, >= %d" % WINDOW, earlier & same & (dist >= WINDOW)),
                     ("earlier book, >= %d" % WINDOW, earlier & ~same & (dist >= WINDOW))):
        mk = mk & np.isfinite(f0)
        r = {"bin": name, "tokens": int(mk.sum())}
        for sk, x in series.items():
            r.update(fmt(summary(x, book, mk), sk if "_minus_" in sk else "gain_" + sk))
        wrows.append(r)
        print({kk: v for kk, v in r.items() if not kk.endswith("under_2se")})
    write_csv("window.csv", wrows)

    # very long books
    vl = {"frozen14m_ctx0": runs.load(runs.frozen_key("14m", "verylong"))["loss"].reshape(-1),
          "frozen14m_ctx1792": flat[key_of("14m", "frozen", "verylong", 1792)],
          "online14m_ctx0": runs.load(runs.find("14m", "sgd", "verylong"))["loss"].reshape(-1),
          "frozen31m_ctx0": runs.load(runs.frozen_key("31m", "verylong"))["loss"].reshape(-1),
          "frozen31m_ctx1792": flat[key_of("31m", "frozen", "verylong", 1792)]}
    vbook = np.repeat(np.arange(4), len(vl["frozen14m_ctx0"]) // 4)
    m = {n: np.nanmean(v) for n, v in vl.items()}
    vrows = []
    for n, v in vl.items():
        vrows.append({"quantity": n, "loss": "%.4f" % m[n],
                      **fmt(summary(vl["frozen14m_ctx0"] - v, vbook, np.ones(len(v), bool)), "gain_over_frozen14m_ctx0")})
    fracs = [("gap closed by 14M online, no context: (f14 - on14) / (f14 - f31), all at C=0",
              (m["frozen14m_ctx0"] - m["online14m_ctx0"]) / (m["frozen14m_ctx0"] - m["frozen31m_ctx0"])),
             ("gap closed by 14M online (C=0) against the context baselines: (f14c - on14) / (f14c - f31c)",
              (m["frozen14m_ctx1792"] - m["online14m_ctx0"]) / (m["frozen14m_ctx1792"] - m["frozen31m_ctx1792"])),
             ("gap closed by context alone: (f14 - f14c) / (f14 - f31), gap at C=0",
              (m["frozen14m_ctx0"] - m["frozen14m_ctx1792"]) / (m["frozen14m_ctx0"] - m["frozen31m_ctx0"])),
             ("14M-31M gap, C=0 (nats)", m["frozen14m_ctx0"] - m["frozen31m_ctx0"]),
             ("14M-31M gap, C=1792 (nats)", m["frozen14m_ctx1792"] - m["frozen31m_ctx1792"])]
    for name, val in fracs:  # ratios of 4-book means; no SE
        vrows.append({"quantity": name, "value": "%.4f" % val})
    for name, a, b in (("paired: frozen14m_ctx1792 - online14m_ctx0 (frozen with context ahead)", "online14m_ctx0",
                        "frozen14m_ctx1792"),
                       ("paired: frozen31m_ctx0 - frozen14m_ctx1792 (31M without context ahead)", "frozen14m_ctx1792",
                        "frozen31m_ctx0")):
        vrows.append({"quantity": name, **fmt(summary(vl[a] - vl[b], vbook, np.ones(len(vbook), bool)), "value")})
    write_csv("verylong.csv", vrows)
    for r in vrows:
        print(r)

    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for k in have:
            md = load(k)["meta"]
            f.write("%s %d s (%d threads)\n" % (k, md["seconds"], md["threads"]))

    style()
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    colors = {"14m": BLUE, "31m": AQUA, "70m": ORANGE}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for size in runs.SIZES:
        rs = [r for r in rows if r["size"] == size and r["stream"] == "main_K1" and r["rule"] == "frozen"]
        x = [r["context"] for r in rs]
        ax.errorbar(x, [float(r["gain"]) for r in rs], yerr=[2 * float(r["gain_se"]) for r in rs], color=colors[size],
                    lw=1.8, marker="o", ms=4, capsize=2, label="%s frozen with context" % size.upper())
        ax.axhline(float(rs[0]["online_c0_gain"]), color=colors[size], ls="--", lw=1.2,
                   label="%s online, no context" % size.upper())
    rb = [r for r in rows if r["rule"] == "sgd"]
    if rb:
        ax.errorbar([1792], [float(rb[0]["gain"])], yerr=[2 * float(rb[0]["gain_se"])], color=BLUE, marker="*", ms=9,
                    capsize=2, ls="none", label="14M online with context")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(CONTEXTS)
    ax.set_xlabel("context C (tokens before the chunk, same book)")
    ax.set_ylabel("gain over frozen, no context (nats/token)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Main set, book order: gain against context (bars +- 2 SE)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_context.png"), dpi=160)

    main_rows = [r for r in drows if r["bin"] in BINS]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(main_rows))
    lines = [("frozen_ctx256", INK, ":", "frozen, C = 256"), ("frozen_ctx768", INK, "--", "frozen, C = 768"),
             ("frozen_ctx1792", INK, "-", "frozen, C = 1792"), ("online_ctx0", BLUE, "--", "online, no context"),
             ("online_ctx1792", BLUE, "-", "online, C = 1792")]
    for sk, color, ls, label in lines:
        if "gain_" + sk not in main_rows[0]:
            continue
        y = np.array([float(r["gain_" + sk]) for r in main_rows])
        s = np.array([float(r["gain_%s_se" % sk]) for r in main_rows])
        ax.errorbar(x, y, yerr=2 * s, color=color, ls=ls, lw=1.6, marker="o", ms=3.5, capsize=2, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [r["bin"].replace(", ", "\n") for r in main_rows], fontsize=7)
    ax.set_ylabel("gain over frozen 14M, no context (nats/token)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("14M: gain by distance to the previous occurrence (bars +- 2 SE)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_by_distance.png"), dpi=160)


if __name__ == "__main__":
    {"run": lambda: run_cmd(sys.argv[2:]), "report": report}[sys.argv[1]]()
