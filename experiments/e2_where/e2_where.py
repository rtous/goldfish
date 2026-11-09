"""E2 (claim C5): which parameters hold the online memory?

Runs (registry): Pythia-14M, plain SGD, main set, K = 1, updating only one parameter subset, each at
its own val-pick rate (e0_lr_picks): all parameters (E1's run), output embedding only, all but the
output embedding, MLPs only, attention only, the MLP of a single layer (1 ... 6, 1 = first).

Table (results.csv, long format): per subset and E1 bin, gain over frozen 14M with SE over books, and
the paired difference to the full learner (subset - all parameters). Plot: single-layer MLPs next to
the full learner.

MLP only against all parameters (mlp_vs_full.csv): both runs also on the shuffled stream, with snapshots
(as E5). Per bin: gain in order, gain shuffled, locality (in order - shuffled, paired per token), SE over
books, and the paired difference MLP only - all parameters. Forgetting on detokenized WikiText and the 6
held-out books (last-half mean and final snapshot - pretrained; SE over WikiText chunks or held-out
books), K = 1 and shuffled, and its paired difference (same snapshot positions); exchange rate = stream
gain / last-half WikiText forgetting. Usage (from experiments/):
    ../venv/bin/python e2_where/e2_where.py run | report
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from data import chunk_set  # noqa: E402
from metrics import bin_masks, exchange_rate, forgetting, se, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SUBSETS = ["all", "output", "no_output", "mlp", "attention"] + ["mlp@%d" % i for i in range(6)]
LABELS = {"all": "all parameters", "output": "output only", "no_output": "all but output", "mlp": "MLP only",
          "attention": "attention only", **{"mlp@%d" % i: "MLP of layer %d" % (i + 1) for i in range(6)}}
TABLE_BINS = ["all tokens", "same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
              "earlier chunk, >= 100k", "never seen", "earlier chunk, 1k-10k, same book"]


def keys():
    return {p: runs.find("14m", "sgd", "main_K1", p) for p in SUBSETS}


COMPARE_BINS = ["all tokens", "earlier chunk, < 1k", "earlier chunk, 1k-10k, same book", "never seen"]


def mlp_keys():
    """{stream: {"mlp": key, "all": key}} for MLP only and all parameters, K = 1 and shuffled."""
    return {st: {p: runs.find("14m", "sgd", st, p) for p in ("mlp", "all")} for st in ("main_K1", "main_shuffled")}


def held_out_diff(r, k, n_wiki, n_books=6):
    """Per held-out unit (WikiText chunk, or held-out book) snapshot - pretrained: last-half mean and final."""
    snaps, pos = r["snaps"], r["positions"]
    half = [i for i in range(1, len(pos)) if pos[i] > pos[-1] / 2]
    d = {"last_half": np.mean([snaps[i] - snaps[0] for i in half], axis=0), "final": snaps[-1] - snaps[0]}[k]
    per = (len(d) - n_wiki) // n_books
    return {"wiki": d[:n_wiki], "books": np.array([d[n_wiki + per * b:n_wiki + per * (b + 1)].mean()
                                                    for b in range(n_books)])}


def mlp_vs_full(book, masks, frozen):
    """Rows of mlp_vs_full.csv."""
    k = mlp_keys()
    pk = runs.picks()
    r = {(st, p): runs.load(k[st][p]) for st in k for p in ("mlp", "all")}
    lo = {key: v["loss"].reshape(-1) for key, v in r.items()}
    rows = []

    def row(section, quantity, stream, x, y, diff):
        """x, y: dicts with mean, se (MLP only, all parameters); diff: paired MLP - all."""
        out = {"section": section, "quantity": quantity, "stream": stream}
        for name, v in (("mlp", x), ("full", y), ("mlp_minus_full", diff)):
            out[name] = "%.4f" % v["mean"]
            out[name + "_se"] = "%.4f" % v["se"] if v.get("se") is not None else ""
            out[name + "_under_2se"] = bool(abs(v["mean"]) < 2 * v["se"]) if v.get("se") is not None else ""
        rows.append(out)

    for b in COMPARE_BINS:
        m = masks[b] & np.isfinite(frozen)
        for st, label in (("main_K1", "gain in order"), ("main_shuffled", "gain shuffled")):
            row("stream", label, b, summary(frozen - lo[st, "mlp"], book, m), summary(frozen - lo[st, "all"], book, m),
                summary(lo[st, "all"] - lo[st, "mlp"], book, m))
        loc = {p: lo["main_shuffled", p] - lo["main_K1", p] for p in ("mlp", "all")}
        row("stream", "locality", b, summary(loc["mlp"], book, m), summary(loc["all"], book, m),
            summary(loc["mlp"] - loc["all"], book, m))
    allm = np.isfinite(frozen)
    for st, label in (("main_K1", "K = 1"), ("main_shuffled", "shuffled")):
        assert (r[st, "mlp"]["positions"] == r[st, "all"]["positions"]).all()
        n_wiki = r[st, "mlp"]["n_wiki"]
        for which, wl in (("wiki", "WikiText forgetting"), ("books", "held-out books forgetting")):
            for kind, kl in (("last_half", "last half"), ("final", "final")):
                a = held_out_diff(r[st, "mlp"], kind, n_wiki)[which]
                c = held_out_diff(r[st, "all"], kind, n_wiki)[which]
                row("forgetting", "%s, %s" % (wl, kl), label, {"mean": a.mean(), "se": se(a)},
                    {"mean": c.mean(), "se": se(c)}, {"mean": (a - c).mean(), "se": se(a - c)})
        ex = {}
        for p in ("mlp", "all"):
            g = summary(frozen - lo[st, p], book, allm)["mean"]
            f = forgetting(r[st, p]["snaps"], r[st, p]["positions"], n_wiki)["wiki"]["last_half"]
            ex[p] = exchange_rate(g, f)
        rows.append({"section": "exchange rate", "quantity": "stream gain / WikiText forgetting (last half)",
                     "stream": label, "mlp": "%.2f" % ex["mlp"], "full": "%.2f" % ex["all"],
                     "mlp_minus_full": "%.2f" % (ex["mlp"] - ex["all"])})
    for x in rows:
        x["lr_mlp"], x["lr_full"] = "%g" % pk["14m", "sgd", "mlp"], "%g" % pk["14m", "sgd", "all"]
    return rows


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def run():
    for k in list(keys().values()) + [runs.frozen_key("14m", "main_K1")] + list(mlp_keys()["main_shuffled"].values()):
        if not os.path.exists(runs.path_of(k)):
            runs.run(next(s for s in runs.registry() if s["key"] == k))


def n_params():
    from learner import setup, trained_params
    model, _, _ = setup()
    return {p: sum(t.numel() for t in trained_params(model, p)) for p in SUBSETS}


def report():
    from plots import plt, style, BLUE, ORANGE, INK, MUTED
    k = keys()
    book, masks = bin_masks(chunk_set("main"))
    frozen = runs.load(runs.frozen_key("14m", "main_K1"))["loss"].reshape(-1)
    loss = {p: runs.load(k[p])["loss"].reshape(-1) for p in SUBSETS}
    pk, npar = runs.picks(), n_params()
    rows = []
    for p in SUBSETS:
        for b in TABLE_BINS:
            m = masks[b] & np.isfinite(frozen)
            g, d = summary(frozen - loss[p], book, m), summary(loss["all"] - loss[p], book, m)
            rows.append({"subset": p, "label": LABELS[p], "lr": "%g" % pk["14m", "sgd", p], "params": npar[p], "bin": b,
                         "tokens": g["tokens"], "gain": "%.4f" % g["mean"], "gain_se": "%.4f" % g["se"],
                         "gain_under_2se": g["under_2se"], "minus_full": "%.4f" % d["mean"],
                         "minus_full_se": "%.4f" % d["se"], "minus_full_under_2se": d["under_2se"],
                         "share_of_full": "%.2f" % (g["mean"] / float(next(r["gain"] for r in rows if r["subset"] == "all"
                                                                           and r["bin"] == b)))
                         if p != "all" else "1.00"})
    write_csv("results.csv", rows)
    cmp_rows = mlp_vs_full(book, masks, frozen)
    write_csv("mlp_vs_full.csv", cmp_rows)
    for r in cmp_rows:
        print("%-13s %-45s %-34s mlp %s +- %s  full %s +- %s  diff %s +- %s" % (
            r["section"], r["quantity"], r["stream"], r["mlp"], r.get("mlp_se", ""), r["full"], r.get("full_se", ""),
            r["mlp_minus_full"], r.get("mlp_minus_full_se", "")))
    for r in rows:
        if r["bin"] in ("all tokens", "earlier chunk, < 1k", "earlier chunk, 1k-10k, same book"):
            print("%-16s lr %-7s %-34s gain %s +- %s  share %s" % (r["label"], r["lr"], r["bin"], r["gain"], r["gain_se"],
                                                                 r["share_of_full"]))
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for p in SUBSETS:
            f.write("%s %d s\n" % (k[p], runs.seconds(k[p])))
        f.write("%s %d s\n" % (mlp_keys()["main_shuffled"]["mlp"], runs.seconds(mlp_keys()["main_shuffled"]["mlp"])))

    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(1, 7)
    for b, color, label in (("earlier chunk, < 1k", BLUE, "earlier chunk, < 1k"),
                            ("earlier chunk, 1k-10k, same book", ORANGE, "earlier chunk, 1k-10k, same book"),
                            ("all tokens", INK, "all tokens")):
        r = {q["subset"]: q for q in rows if q["bin"] == b}
        y = np.array([float(r["mlp@%d" % (i - 1)]["gain"]) for i in x])
        s = np.array([float(r["mlp@%d" % (i - 1)]["gain_se"]) for i in x])
        ax.errorbar(x, y, yerr=2 * s, color=color, lw=1.8, marker="o", ms=4, capsize=2, label=label)
        ax.axhline(float(r["all"]["gain"]), color=color, lw=1, ls="--")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["%d\nlr %s" % (i, next(q["lr"] for q in rows if q["subset"] == "mlp@%d" % (i - 1))) for i in x])
    ax.set_xlabel("layer whose MLP learns online (1 = first); dashed: all parameters")
    ax.set_ylabel("gain over frozen 14M (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pythia-14M, one layer's MLP learning online (bars +- 2 SE)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "gain_vs_layer.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
