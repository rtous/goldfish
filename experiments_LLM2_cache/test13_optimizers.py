"""Test 13: update rules that behave the same across sizes. Is 70M's weak online learning about size or
about plain SGD?

Step 1 (diag): plain SGD at the old picks (14M 1e-4, 31M 3e-4, 70M 1e-5) on the first 40 chunks of the
main stream (test 7's chunk set, K1); per tensor and chunk: gradient norm, weight norm and relative
update ||dW|| / ||W|| = lr ||g|| / ||W||, summarized by tensor type and layer. For the input embedding,
also relative to the rows the chunk touches (its gradient is zero elsewhere).

Step 2: three update rules (snapshot.Learner), one update per chunk after scoring it:
    sgd   plain SGD, all parameters
    adam  torch.optim.Adam (0.9, 0.999, eps 1e-8, no weight decay), fresh per stream
    nsgd  normalized SGD per tensor, W <- W - eta n(W) g / ||g||, n(W) = ||W|| (+1e-3 for 1-D tensors)
Rates are picked per (size, rule) on the val-pick validation stream (dual.val_sets: 10 books, K1,
780 chunks): lowest mean online loss, with the one-step grid extension at an edge (1-3-10 ladder).
Runs at the picked rate on test 7's 25-book chunk set, K1 and shuffled (test 7's order), with
snapshots every EVERY chunks scored frozen on the detokenized WikiText and the 6 held-out books
(test 12). Runs identical to test 12 runs (same size, rule sgd and rate) reuse test 12's files.

Metrics (as test 6, on the K1 order's distance bins; 1000 most frequent ids excluded from the bins):
gain over the model's own frozen loss and locality (K1 - shuffled gain) for all tokens, earlier chunk
< 1k, 1k-10k same book, never seen; SE over the 25 books (paired for locality). Locality relative to
the frozen loss on those tokens and as a share of the in-order gain. Forgetting on held-out text and
exchange rate (stream gain / last-half mean WikiText forgetting) as in test 12.

Writes test13_optimizers/: diagnostics.csv, diagnostics_summary.csv, picks.csv, losses_<size>_<rule>_<stream>.npz,
results.csv, exchange.csv, runtime.txt, plots/. Usage:
    python test13_optimizers.py diag
    python test13_optimizers.py pick <size>
    python test13_optimizers.py run <size> <rule> <stream> [...]   (triples)
    python test13_optimizers.py report
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F_

from data import HERE, books
from dual import val_sets
from models import DEFAULT
from online import L, setup
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from snapshot import Learner, score
from test1_distance import distance_bins
from test12_general import BOOK_TOKENS, OUT as OUT12, chunk_means, heldout, se, wikitext_detok_ids
from test7_aggressive import chunk, orders

OUT = os.path.join(HERE, "test13_optimizers")
SIZES = {"14m": DEFAULT, "31m": "EleutherAI/pythia-31m", "70m": "EleutherAI/pythia-70m",
         "160m": "EleutherAI/pythia-160m"}
PARAMS = {"14m": 14067712, "31m": 30494720, "70m": 70426624}  # models.py / test 6
OLD_SGD = {"14m": 1e-4, "31m": 3e-4, "70m": 1e-5}
LADDER = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
GRIDS = {"adam": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4], "nsgd": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]}
SGD_GRIDS = {"14m": [3e-5, 1e-4, 3e-4], "31m": [1e-4, 3e-4, 1e-3], "70m": [3e-6, 1e-5, 3e-5], "160m": [1e-6, 3e-6, 1e-5]}
RULES = ("sgd", "adam", "nsgd")
EVERY = 80
DIAG_CHUNKS = 40
FROZEN7 = os.path.join(HERE, "test7_aggressive", "losses.npz")
T12_RUNS = {("14m", 1e-4): "14m_full_1e-4", ("31m", 3e-4): "31m_full_3e-4"}


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def tensor_type(name):
    if "embed_in" in name:
        return "input embedding"
    if name.startswith("lm_head") or "embed_out" in name:
        return "output embedding"
    if "layernorm" in name or "layer_norm" in name:
        return "layer norm"
    if ".attention." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "other"


def layer_of(name):
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1]) if "layers" in parts else ""


# ---------------------------------------------------------------- step 1


def diag():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    c = chunk(books(DEFAULT)["ids"][:25])
    xs = [torch.from_numpy(c[0, k]) for k in range(DIAG_CHUNKS)]  # book 0, first 40 chunks (K1 start)
    rows = []
    for size, lr in OLD_SGD.items():
        model, init, info, _ = setup(SIZES[size])
        params = dict(model.named_parameters())
        opt = torch.optim.SGD(params.values(), lr=lr, momentum=0, weight_decay=0)
        for k, x in enumerate(xs):
            loss = F_.cross_entropy(model(input_ids=x[None]).logits[0, :-1], x[1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            touched = torch.unique(x)
            for n, p in params.items():
                g, w = p.grad.norm().item(), p.detach().norm().item()
                r = {"size": size, "lr": lr, "chunk": k, "tensor": n, "type": tensor_type(n), "layer": layer_of(n),
                     "numel": p.numel(), "grad_norm": g, "weight_norm": w, "update_norm": lr * g,
                     "rel_update": lr * g / w if w > 0 else float("nan")}
                if "embed_in" in n:
                    r["rel_update_touched_rows"] = lr * g / p.detach()[touched].norm().item()
                rows.append(r)
            opt.step()
        print(size, "done", flush=True)
    write_csv("diagnostics.csv", [{k: ("%.4g" % v if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    diag_report(rows)


def diag_report(rows=None):
    if rows is None:
        rows = list(csv.DictReader(open(os.path.join(OUT, "diagnostics.csv"))))
    out = []
    groups = {}
    for r in rows:  # per size, type (and layer for attention / mlp), chunk: aggregate norms over tensors
        for key in [(r["size"], r["type"], "all layers")] + ([(r["size"], r["type"], "layer %s" % r["layer"])]
                                                              if r["type"] in ("attention", "mlp") else []):
            d = groups.setdefault(key, {})
            c = d.setdefault(int(r["chunk"]), [0.0, 0.0, 0.0])
            c[0] += float(r["update_norm"]) ** 2
            c[1] += float(r["weight_norm"]) ** 2
            c[2] += float(r["grad_norm"]) ** 2
    totals = {}
    for (size, typ, layer), d in groups.items():
        if layer == "all layers":
            for k, v in d.items():
                totals.setdefault((size, k), 0.0)
                totals[(size, k)] += v[0]
    for (size, typ, layer), d in sorted(groups.items()):
        rel = [np.sqrt(v[0] / v[1]) for v in d.values()]
        share = [v[0] / totals[(size, k)] for k, v in d.items()]
        out.append({"size": size, "type": typ, "layer": layer, "grad_norm": "%.4g" % np.mean([np.sqrt(v[2]) for v in d.values()]),
                    "weight_norm": "%.4g" % np.sqrt(np.mean([v[1] for v in d.values()])),
                    "rel_update": "%.3g" % np.mean(rel), "share_of_update_sq": "%.3f" % np.mean(share)})
    emb = {}
    for r in rows:
        if r["type"] == "input embedding":
            emb.setdefault(r["size"], []).append(float(r["rel_update_touched_rows"]))
    for size, v in emb.items():
        out.append({"size": size, "type": "input embedding, touched rows", "layer": "all layers",
                    "rel_update": "%.3g" % np.mean(v)})
    write_csv("diagnostics_summary.csv", out)
    for r in out:
        if r["layer"] == "all layers":
            print(r)
    style()
    types = ["input embedding", "input embedding, touched rows", "attention", "mlp", "layer norm", "output embedding"]
    sizes = [s for s in OLD_SGD if any(r["size"] == s for r in out)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    w = 0.8 / len(sizes)
    colors = {"14m": BLUE, "31m": ORANGE, "70m": INK}
    for i, size in enumerate(sizes):
        y = [float(next(r["rel_update"] for r in out if r["size"] == size and r["type"] == t and r["layer"] == "all layers"))
             for t in types]
        ax.bar(np.arange(len(types)) + (i - (len(sizes) - 1) / 2) * w, y, width=w, color=colors[size],
               label="%s (lr %g)" % (size.upper(), OLD_SGD[size]))
    ax.set_yscale("log")
    short = ["input\nembedding", "input emb.,\ntouched rows", "attention", "MLP", "layer\nnorm", "output\nembedding"]
    ax.set_xticks(np.arange(len(types)), short, fontsize=7)
    ax.set_ylabel("relative update ||dW|| / ||W|| per step")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Plain SGD, first 40 chunks: relative update by tensor type", fontsize=8.5, color=INK, loc="left")
    ax = axes[1]
    for size in sizes:
        for typ, ls in (("mlp", "-"), ("attention", "--")):
            y = [float(next(r["rel_update"] for r in out if r["size"] == size and r["type"] == typ and
                            r["layer"] == "layer %d" % i)) for i in range(6)]
            ax.plot(np.arange(1, 7), y, color=colors[size], ls=ls, marker="o", ms=3,
                    label="%s %s" % (size.upper(), typ))
    ax.set_yscale("log")
    ax.set_xlabel("layer (1 = first)")
    ax.set_ylabel("relative update per step")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("MLP (solid) and attention (dashed) by layer", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "update_sizes.png"), dpi=160)


# ---------------------------------------------------------------- picks


def val_stream():
    sets, streams = val_sets()
    name, order = streams["val_pick_K1"]
    c = sets[name]
    return np.concatenate([c[b, k] for b, k in order])


def pick(size):
    os.makedirs(OUT, exist_ok=True)
    model, init, info, _ = setup(SIZES[size])
    ids = val_stream()
    rows = [{"size": size, "rule": "frozen", "lr": 0, "val_pick_loss": "%.4f" % np.nanmean(score(model, init, ids))}]
    print(rows[-1], flush=True)
    for rule in RULES:
        grid = SGD_GRIDS[size] if rule == "sgd" else GRIDS[rule]
        res = {}

        def evaluate(lr):
            t = time.time()
            res[lr] = float(np.nanmean(Learner(model, init, rule, lr).read(ids)))
            rows.append({"size": size, "rule": rule, "lr": lr, "val_pick_loss": "%.4f" % res[lr],
                         "extension": lr not in grid, "seconds": round(time.time() - t)})
            print(rows[-1], flush=True)

        for lr in grid:
            evaluate(lr)
        best = min(res, key=res.get)
        if best in (min(grid), max(grid)):
            evaluate(LADDER[LADDER.index(best) + (-1 if best == min(grid) else 1)])
            best = min(res, key=res.get)
        for r in rows:
            if r["rule"] == rule:
                r["picked"] = r["lr"] == best
    path = os.path.join(OUT, "picks_%s.csv" % size)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def picks():
    out = {}
    for size in SIZES:
        p = os.path.join(OUT, "picks_%s.csv" % size)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if r.get("picked") == "True":
                    out[size, r["rule"]] = float(r["lr"])
    return out


# ---------------------------------------------------------------- runs


def run(size, rule, stream):
    lr = picks()[size, rule]
    if (size, lr) in T12_RUNS and rule == "sgd":
        print("%s %s %s: lr %g is test 12's run, reused" % (size, rule, stream, lr), flush=True)
        return
    model, init, info, _ = setup(SIZES[size])
    c = chunk(books(DEFAULT)["ids"][:25])
    order = orders(25, c.shape[1])[stream]
    ids = np.concatenate([c[b, k] for b, k in order])
    held_books = heldout()[0][-6 * BOOK_TOKENS:]
    held = np.concatenate([wikitext_detok_ids(), held_books])
    t0 = time.time()
    extra = {}
    if size not in ("14m", "31m", "70m") and stream == "K1":
        extra["frozen"] = score(model, init, c.reshape(-1)).reshape(c.shape).astype(np.float16)
    lrn = Learner(model, init, rule, lr)
    snaps, pos, losses = [chunk_means(score(model, init, held))], [0], []
    for s in range(0, len(order), EVERY):
        e = min(s + EVERY, len(order))
        losses.append(lrn.read(ids[s * L:e * L]))
        snaps.append(chunk_means(lrn.score(held)))
        pos.append(e)
    loss = np.concatenate(losses).reshape(-1, L)
    out = np.full(c.shape, np.nan, dtype=np.float32)
    for i, (b, k) in enumerate(order):
        out[b, k] = loss[i]
    meta = {"size": size, "rule": rule, "stream": stream, "lr": lr, "seconds": round(time.time() - t0),
            "device": info["device"], "params": info["params"]}
    np.savez_compressed(os.path.join(OUT, "losses_%s_%s_%s.npz" % (size, rule, stream)), stream=out.astype(np.float16),
                        snaps=np.array(snaps, dtype=np.float32), positions=np.array(pos) * L, meta=json.dumps(meta), **extra)
    print(meta, "mean loss %.4f" % np.nanmean(out), flush=True)


def load(size, rule, stream):
    """(stream losses [25, 78, L], snapshot chunk means [n, 76 detok WikiText + 90 book chunks], positions, meta)."""
    lr = picks()[size, rule]
    if rule == "sgd" and (size, lr) in T12_RUNS:
        name = "%s_%s" % (T12_RUNS[size, lr], stream)
        z, zd = np.load(os.path.join(OUT12, "losses_%s.npz" % name)), np.load(os.path.join(OUT12, "losses_detok_%s.npz" % name))
        snaps = np.concatenate([zd["detok"], z["snaps"][:, -90:]], axis=1)
        meta = dict(json.loads(str(z["meta"])), lr=lr, reused="test 12 " + name)
        return z["stream"].astype(np.float64), snaps.astype(np.float64), z["positions"], meta
    p = os.path.join(OUT, "losses_%s_%s_%s.npz" % (size, rule, stream))
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["stream"].astype(np.float64), z["snaps"].astype(np.float64), z["positions"], json.loads(str(z["meta"]))


def frozen_of(size):
    if size in ("14m", "31m", "70m"):
        return np.load(FROZEN7)["frozen%s_books25" % size].astype(np.float64)
    return np.load(os.path.join(OUT, "losses_%s_sgd_K1.npz" % size))["frozen"].astype(np.float64)


# ---------------------------------------------------------------- report


def bins():
    c = chunk(books(DEFAULT)["ids"][:25])
    ids = c.reshape(-1)
    book = np.repeat(np.arange(25), c.shape[1] * L)
    prev = np.full(len(ids), -1)
    last = {}
    for t, i in enumerate(ids.tolist()):
        prev[t] = last.get(i, -1)
        last[i] = t
    same_book = (prev >= 0) & (book[np.maximum(prev, 0)] == book)
    b = distance_bins(ids)
    return book, {"all tokens": np.ones(len(ids), bool), "earlier chunk < 1k": b == 1,
                  "1k-10k same book": (b == 2) & same_book, "never seen": b == 5}


def per_book(x, book, m):
    return np.array([np.nanmean(x[m & (book == k)]) for k in range(25) if (m & (book == k) & np.isfinite(x)).sum() >= 50])


def held_forgetting(snaps, pos, n_detok):
    """Last-half mean forgetting: WikiText (mean, SE over chunks), books (mean, SE over 6 books)."""
    ks = [k for k in range(1, len(pos)) if pos[k] > pos[-1] / 2]
    d = np.mean([snaps[k] - snaps[0] for k in ks], axis=0)
    wiki = d[:n_detok]
    bk = [d[n_detok + 15 * b:n_detok + 15 * (b + 1)].mean() for b in range(6)]
    return float(wiki.mean()), se(wiki), float(np.mean(bk)), se(bk)


def report():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    book, masks = bins()
    n_detok = len(wikitext_detok_ids()) // L
    pk = picks()
    rows, xr = [], []
    for size in SIZES:
        for rule in RULES:
            if (size, rule) not in pk:
                continue
            k1, sh = load(size, rule, "K1"), load(size, rule, "shuffled")
            if k1 is None or sh is None:
                continue
            frozen = frozen_of(size).reshape(-1)
            g1, gs = frozen - k1[0].reshape(-1), frozen - sh[0].reshape(-1)
            scored = np.isfinite(frozen)
            base = {"size": size, "rule": rule, "lr": pk[size, rule], "params": PARAMS.get(size, k1[3].get("params", ""))}
            for bname, m in masks.items():
                m = m & scored
                a, b, loc = per_book(g1, book, m), per_book(gs, book, m), per_book(g1 - gs, book, m)
                fz = float(frozen[m].mean())
                l_mean = float((g1 - gs)[m].mean())
                rows.append({**base, "bin": bname, "tokens": int(m.sum()), "frozen_loss": "%.4f" % fz,
                             "gain_K1": "%.4f" % g1[m].mean(), "gain_K1_se": "%.4f" % se(a),
                             "gain_shuffled": "%.4f" % gs[m].mean(), "gain_shuffled_se": "%.4f" % se(b),
                             "locality": "%.4f" % l_mean, "locality_se": "%.4f" % se(loc),
                             "locality_under_2se": bool(abs(l_mean) < 2 * se(loc)),
                             "locality_rel_frozen": "%.4f" % (l_mean / fz),
                             "locality_share_of_gain": "%.3f" % (l_mean / g1[m].mean()) if g1[m].mean() > 0 else ""})
            r = {**base, "gain_K1": "%.4f" % np.nanmean(g1), "gain_shuffled": "%.4f" % np.nanmean(gs)}
            for name, run_ in (("K1", k1), ("shuffled", sh)):
                fw, fws, fb, fbs = held_forgetting(run_[1], run_[2], n_detok)
                gain = float(np.nanmean(frozen - run_[0].reshape(-1)))
                r.update({"wiki_forgetting_%s" % name: "%.4f" % fw, "wiki_forgetting_%s_se" % name: "%.4f" % fws,
                          "books_forgetting_%s" % name: "%.4f" % fb, "books_forgetting_%s_se" % name: "%.4f" % fbs,
                          "exchange_rate_%s" % name: ("%.2f" % (gain / fw)) if fw > 0 else "no forgetting",
                          "frozen_wiki": "%.4f" % run_[1][0][:n_detok].mean(),
                          "reused": run_[3].get("reused", "")})
            xr.append(r)
    write_csv("results.csv", rows)
    write_csv("exchange.csv", xr)
    for r in xr:
        print(r)
    for r in rows:
        if r["bin"] in ("all tokens", "earlier chunk < 1k"):
            print(r["size"], r["rule"], r["lr"], r["bin"], "gain", r["gain_K1"], "loc", r["locality"], "±", r["locality_se"],
                  "rel", r["locality_rel_frozen"], "share", r["locality_share_of_gain"])
    plot(rows)


def plot(rows=None):
    if rows is None:
        rows = list(csv.DictReader(open(os.path.join(OUT, "results.csv"))))
    style()
    colors = {"sgd": INK, "adam": BLUE, "nsgd": ORANGE}
    labels = {"sgd": "plain SGD", "adam": "Adam", "nsgd": "normalized SGD"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (bname, key, title) in zip(axes, (("earlier chunk < 1k", "locality", "locality, earlier chunk < 1k"),
                                              ("all tokens", "locality", "locality, all tokens"),
                                              ("all tokens", "gain_K1", "gain in order, all tokens"))):
        for rule in RULES:
            r = [x for x in rows if x["rule"] == rule and x["bin"] == bname]
            if not r:
                continue
            x = [int(v["params"]) / 1e6 for v in r]
            y = np.array([float(v[key]) for v in r])
            s = np.array([float(v[key + "_se"]) for v in r])
            ax.fill_between(x, y - s, y + s, color=colors[rule], alpha=0.15, lw=0)
            ax.plot(x, y, color=colors[rule], marker="o", ms=4, lw=1.8, label=labels[rule])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (millions, log scale)")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.minorticks_off()
        xs = sorted({int(v["params"]) / 1e6 for v in rows})
        names = {round(v / 1e6, 3): k.upper() for k, v in PARAMS.items()}
        ax.set_xticks(xs, [names.get(round(v, 3), "%.0fM" % v) for v in xs])
    axes[0].set_ylabel("nats/token (band +- 1 SE over books)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "locality_vs_size.png"), dpi=160)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "diag":
        diag()
    elif cmd == "diag_report":
        diag_report()
    elif cmd == "pick":
        for s in sys.argv[2:]:
            pick(s)
    elif cmd == "run":
        a = sys.argv[2:]
        for i in range(0, len(a), 3):
            run(*a[i:i + 3])
    elif cmd == "report":
        report()
    elif cmd == "plot":
        plot()
