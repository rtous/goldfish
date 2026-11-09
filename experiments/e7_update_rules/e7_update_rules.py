"""E7 (claim C10): does the update rule decide whether there is a cache, and how does it scale?

Step 1 (diag): plain SGD at the 14M / 31M / 70M sgd picks on the first 40 chunks of main book 0 (the
start of K = 1); per tensor and chunk the relative update ||dW|| / ||W|| = lr ||g|| / ||W||, summarized
by tensor type (and by layer for MLP / attention). For the input embedding also relative to the rows
the chunk touches (its gradient is zero elsewhere).

Step 2 (registry runs, snapshots on): 14M / 31M / 70M x {sgd, adam, nsgd} (lib/learner.py), each at
its val-pick rate, main set K = 1 and shuffled. Per run and bin (all tokens, earlier chunk < 1k,
1k-10k same book, never seen; lib/metrics.py bins): gain over the model's own frozen loss in order and
shuffled, locality (paired), locality relative to the frozen loss on those tokens and as a share of the
in-order gain, and paired differences against sgd of the same size. Forgetting (last-half mean,
held-out WikiText and books) and exchange rate as in E5.

Writes diagnostics.csv, diagnostics_summary.csv, results.csv, exchange.csv, runtime.txt, plots/.
Usage (from experiments/):
    ../venv/bin/python e7_update_rules/e7_update_rules.py diag      # about 2 min
    ../venv/bin/python e7_update_rules/e7_update_rules.py run       # the registry runs, if missing
    ../venv/bin/python e7_update_rules/e7_update_rules.py report
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from data import chunk_set  # noqa: E402
from metrics import bin_masks, exchange_rate, forgetting, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SIZES, RULES = ("14m", "31m", "70m"), ("sgd", "adam", "nsgd")
BINS = {"all tokens": "all tokens", "earlier chunk < 1k": "earlier chunk, < 1k",
        "1k-10k same book": "earlier chunk, 1k-10k, same book", "never seen": "never seen"}
DIAG_CHUNKS = 40


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- step 1


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


def diag():
    import torch
    import torch.nn.functional as F
    from learner import setup
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    pk = runs.picks()
    c = chunk_set("main")
    xs = [torch.from_numpy(c[0, k]) for k in range(DIAG_CHUNKS)]
    rows, t0 = [], time.time()
    for size in SIZES:
        lr = pk[size, "sgd", "all"]
        model, init, info = setup(size)
        params = dict(model.named_parameters())
        opt = torch.optim.SGD(params.values(), lr=lr, momentum=0, weight_decay=0)
        for k, x in enumerate(xs):
            loss = F.cross_entropy(model(input_ids=x[None]).logits[0, :-1], x[1:])
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
    with open(os.path.join(OUT, "runtime_diag.txt"), "w") as f:
        f.write("diag %.0f s\n" % (time.time() - t0))
    diag_report()


def diag_report():
    from plots import plt, style, BLUE, ORANGE, INK
    rows = list(csv.DictReader(open(os.path.join(OUT, "diagnostics.csv"))))
    groups = {}
    for r in rows:  # per size, type (and layer for attention / mlp), chunk: aggregate norms over tensors
        for key in [(r["size"], r["type"], "all layers")] + ([(r["size"], r["type"], "layer %s" % r["layer"])]
                                                              if r["type"] in ("attention", "mlp") else []):
            c = groups.setdefault(key, {}).setdefault(int(r["chunk"]), [0.0, 0.0, 0.0])
            c[0] += float(r["update_norm"]) ** 2
            c[1] += float(r["weight_norm"]) ** 2
            c[2] += float(r["grad_norm"]) ** 2
    totals = {}
    for (size, typ, layer), d in groups.items():
        if layer == "all layers":
            for k, v in d.items():
                totals[size, k] = totals.get((size, k), 0.0) + v[0]
    out = []
    for (size, typ, layer), d in sorted(groups.items()):
        out.append({"size": size, "lr": next(r["lr"] for r in rows if r["size"] == size), "type": typ, "layer": layer,
                    "grad_norm": "%.4g" % np.mean([np.sqrt(v[2]) for v in d.values()]),
                    "weight_norm": "%.4g" % np.sqrt(np.mean([v[1] for v in d.values()])),
                    "rel_update": "%.3g" % np.mean([np.sqrt(v[0] / v[1]) for v in d.values()]),
                    "share_of_update_sq": "%.3f" % np.mean([v[0] / totals[size, k] for k, v in d.items()])})
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
    colors = {"14m": BLUE, "31m": ORANGE, "70m": INK}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax, w = axes[0], 0.8 / len(SIZES)
    for i, size in enumerate(SIZES):
        y = [float(next(r["rel_update"] for r in out if r["size"] == size and r["type"] == t and r["layer"] == "all layers"))
             for t in types]
        lr = next(r["lr"] for r in out if r["size"] == size)
        ax.bar(np.arange(len(types)) + (i - 1) * w, y, width=w, color=colors[size], label="%s (lr %s)" % (size.upper(), lr))
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(types)), ["input\nembedding", "input emb.,\ntouched rows", "attention", "MLP", "layer\nnorm",
                                          "output\nembedding"], fontsize=7)
    ax.set_ylabel("relative update ||dW|| / ||W|| per step")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Plain SGD, first 40 chunks: relative update by tensor type", fontsize=8.5, color=INK, loc="left")
    ax = axes[1]
    for size in SIZES:
        for typ, ls in (("mlp", "-"), ("attention", "--")):
            y = [float(next(r["rel_update"] for r in out if r["size"] == size and r["type"] == typ and
                            r["layer"] == "layer %d" % i)) for i in range(6)]
            ax.plot(np.arange(1, 7), y, color=colors[size], ls=ls, marker="o", ms=3, label="%s %s" % (size.upper(), typ))
    ax.set_yscale("log")
    ax.set_xlabel("layer (1 = first)")
    ax.set_ylabel("relative update per step")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("MLP (solid) and attention (dashed) by layer", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "update_sizes.png"), dpi=160)


# ---------------------------------------------------------------- step 2


def keys():
    return {(size, rule, st): runs.find(size, rule, st) for size in SIZES for rule in RULES
            for st in ("main_K1", "main_shuffled")}


def run():
    for k in list(keys().values()) + [runs.frozen_key(s, "main_K1") for s in SIZES]:
        if not os.path.exists(runs.path_of(k)):
            runs.run(next(s for s in runs.registry() if s["key"] == k))


def report():
    from plots import plt, style, BLUE, ORANGE, INK, MUTED
    k = keys()
    book, masks = bin_masks(chunk_set("main"))
    rows, xr, gains = [], [], {}
    for size in SIZES:
        frozen = runs.load(runs.frozen_key(size, "main_K1"))["loss"].reshape(-1)
        scored = np.isfinite(frozen)
        for rule in RULES:
            r1, rs = runs.load(k[size, rule, "main_K1"]), runs.load(k[size, rule, "main_shuffled"])
            g1, gs = frozen - r1["loss"].reshape(-1), frozen - rs["loss"].reshape(-1)
            gains[size, rule] = (g1, gs)
            base = {"size": size, "rule": rule, "lr": "%g" % r1["meta"]["lr"]}
            for bname, mname in BINS.items():
                m = masks[mname] & scored
                a, b, loc = summary(g1, book, m), summary(gs, book, m), summary(g1 - gs, book, m)
                fz = float(frozen[m].mean())
                row = {**base, "bin": bname, "tokens": a["tokens"], "frozen_loss": "%.4f" % fz,
                       "gain_K1": "%.4f" % a["mean"], "gain_K1_se": "%.4f" % a["se"],
                       "gain_shuffled": "%.4f" % b["mean"], "gain_shuffled_se": "%.4f" % b["se"],
                       "locality": "%.4f" % loc["mean"], "locality_se": "%.4f" % loc["se"],
                       "locality_under_2se": loc["under_2se"], "locality_rel_frozen": "%.4f" % (loc["mean"] / fz),
                       "locality_share_of_gain": "%.3f" % (loc["mean"] / a["mean"]) if a["mean"] > 0 else ""}
                if rule != "sgd":
                    s1, ss = gains[size, "sgd"]
                    dg, dl = summary(g1 - s1, book, m), summary((g1 - gs) - (s1 - ss), book, m)
                    row.update({"gain_K1_minus_sgd": "%.4f" % dg["mean"], "gain_K1_minus_sgd_se": "%.4f" % dg["se"],
                                "gain_K1_minus_sgd_under_2se": dg["under_2se"],
                                "locality_minus_sgd": "%.4f" % dl["mean"], "locality_minus_sgd_se": "%.4f" % dl["se"],
                                "locality_minus_sgd_under_2se": dl["under_2se"]})
                rows.append(row)
            x = {**base}
            for st, rr, g in (("K1", r1, g1), ("shuffled", rs, gs)):
                f = forgetting(rr["snaps"], rr["positions"], rr["n_wiki"])
                gm = float(np.nanmean(g))
                xrate = exchange_rate(gm, f["wiki"]["last_half"])
                x.update({"gain_%s" % st: "%.4f" % gm,
                          "wiki_forgetting_%s" % st: "%.4f" % f["wiki"]["last_half"],
                          "wiki_forgetting_%s_se" % st: "%.4f" % f["wiki"]["last_half_se"],
                          "books_forgetting_%s" % st: "%.4f" % f["books"]["last_half"],
                          "books_forgetting_%s_se" % st: "%.4f" % f["books"]["last_half_se"],
                          "exchange_rate_%s" % st: "%.2f" % xrate if xrate is not None else "no forgetting"})
            xr.append(x)
    write_csv("results.csv", rows)
    write_csv("exchange.csv", xr)
    for r in xr:
        print(r)
    for r in rows:
        if r["bin"] in ("all tokens", "earlier chunk < 1k"):
            print(r["size"], r["rule"], r["lr"], r["bin"], "gain", r["gain_K1"], "+-", r["gain_K1_se"], "loc", r["locality"],
                  "+-", r["locality_se"], "rel", r["locality_rel_frozen"], "share", r["locality_share_of_gain"])
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for kk in k.values():
            f.write("%s %d s\n" % (kk, runs.seconds(kk)))
        if os.path.exists(os.path.join(OUT, "runtime_diag.txt")):
            f.write(open(os.path.join(OUT, "runtime_diag.txt")).read())

    style()
    colors = {"sgd": INK, "adam": BLUE, "nsgd": ORANGE}
    labels = {"sgd": "plain SGD", "adam": "Adam", "nsgd": "normalized SGD"}
    params = {"14m": 14.07, "31m": 30.49, "70m": 70.43}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (bname, key, title) in zip(axes, (("earlier chunk < 1k", "locality", "locality, earlier chunk < 1k"),
                                              ("all tokens", "locality", "locality, all tokens"),
                                              ("all tokens", "gain_K1", "gain in order, all tokens"))):
        for rule in RULES:
            r = [x for x in rows if x["rule"] == rule and x["bin"] == bname]
            xs = [params[v["size"]] for v in r]
            y, s = np.array([float(v[key]) for v in r]), np.array([float(v[key + "_se"]) for v in r])
            ax.fill_between(xs, y - s, y + s, color=colors[rule], alpha=0.15, lw=0)
            ax.plot(xs, y, color=colors[rule], marker="o", ms=4, lw=1.8, label=labels[rule])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(list(params.values()), ["14M", "31M", "70M"])
        ax.set_xlabel("parameters (log scale)")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].set_ylabel("nats/token (band +- 1 SE over books)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "locality_vs_size.png"), dpi=160)


if __name__ == "__main__":
    {"diag": diag, "diag_report": diag_report, "run": run, "report": report}[sys.argv[1]]()
