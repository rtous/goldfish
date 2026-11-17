"""E9 report: tables and plots from the saved runs only (e9_managed_cache/runs/ and the registry).

table.csv     per configuration and stream: gain over frozen on the book chunks (SE over books), locality
              (main_K1 row: in order - shuffled, paired), loss change on the WikiText blocks of the mixed
              stream (run - frozen; positive = forgetting while reading; SE over the 25 blocks), held-out
              forgetting as E5 (snapshot - pretrained, last half and final; WikiText and held-out books),
              exchange rate (stream gain / last-half held-out WikiText forgetting), flushes, mixer weight.
distance.csv  configurations 2, 3, 4, 7 on main_K1: gain by E1 distance bin.
mixer.csv     configurations 6 and 7 on the mixed stream: per chunk, mean mixer weight on the store, unit
              kind, flush; mixer_profile.csv: mean weight by position inside a book / WikiText block.
plots/        tradeoff.png (stream gain against held-out WikiText forgetting, main_K1, with E2 / E7 runs),
              mixer.png (mixer weight along the mixed stream, WikiText blocks shaded, flushes marked).
"""
import csv
import json
import os

import numpy as np

import runs
from data import L, chunk_set
from metrics import bin_masks, exchange_rate, forgetting, summary

OUT = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(OUT)
CONFIGS = [("1", "frozen", "frozen"), ("2", "all", "online, all parameters"), ("3", "store", "store only"),
           ("4", "decay", "store + decay"), ("5", "decay_flush", "store + decay + flush"),
           ("6", "store_fallback", "store + fallback"), ("7", "full", "store + decay + flush + fallback")]
STREAMS = ["main_K1", "main_shuffled", "mixed"]


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def get(cfg, st):
    """Run dict (loss in unit layout, snaps, ...) or None: registry runs for 1 and 2 on the main streams."""
    from e9_managed_cache import load, path_of
    if st != "mixed" and cfg == "frozen":
        return runs.load(runs.frozen_key("14m", st))
    if st != "mixed" and cfg == "all":
        return runs.load(runs.find("14m", "sgd", st))
    key = "%s__%s" % (st, cfg)
    return load(key) if os.path.exists(path_of(key)) else None


def fmt(s, name):
    return {name: "%.4f" % s["mean"], name + "_se": "%.4f" % s["se"], name + "_under_2se": s["under_2se"]}


def report():
    from plots import plt, style, BLUE, ORANGE, AQUA, INK, MUTED
    n_books = 25
    rows, gains = [], {}
    for st in STREAMS:
        fz = get("frozen", st)["loss"]
        fb = fz[:n_books].reshape(-1)
        book = np.repeat(np.arange(n_books), fz.shape[1] * L)
        allm = np.isfinite(fb)
        for num, cfg, label in CONFIGS:
            r = get(cfg, st)
            if r is None:
                continue
            lb = r["loss"][:n_books].reshape(-1)
            g = summary(fb - lb, book, allm)
            gains[cfg, st] = g["mean"]
            row = {"config": num, "name": cfg, "label": label, "stream": st, "loss_books": "%.4f" % np.nanmean(lb)}
            row.update(fmt(g, "gain"))
            if st == "main_K1":
                sh = get(cfg, "main_shuffled")
                if sh is not None:
                    row.update(fmt(summary(sh["loss"][:n_books].reshape(-1) - lb, book, allm), "locality"))
            if st == "mixed":
                wz, wl = fz[n_books:].reshape(-1), r["loss"][n_books:].reshape(-1)
                wblock = np.repeat(np.arange(fz.shape[0] - n_books), fz.shape[1] * L)
                row["frozen_wiki_blocks"] = "%.4f" % np.nanmean(wz)
                row.update(fmt(summary(wl - wz, wblock, np.isfinite(wz)), "wiki_block_change"))
                if "weight" in r:
                    row["weight_books"] = "%.3f" % np.nanmean(r["weight"][:n_books])
                    row["weight_wiki"] = "%.3f" % np.nanmean(r["weight"][n_books:])
            if "snaps" in r:
                f = forgetting(r["snaps"], r["positions"], r["n_wiki"])
                for which in ("wiki", "books"):
                    for k in ("last_half", "final"):
                        row["heldout_%s_%s" % (which, k)] = "%.4f" % f[which][k]
                        row["heldout_%s_%s_se" % (which, k)] = "%.4f" % f[which][k + "_se"]
                if f["wiki"]["last_half"] > 2 * f["wiki"]["last_half_se"]:
                    row["exchange_rate"] = "%.2f" % exchange_rate(g["mean"], f["wiki"]["last_half"])
                else:
                    row["exchange_rate"] = "n/a (forgetting under 2 SE)"
            elif cfg == "frozen":
                row.update(heldout_wiki_last_half="0", heldout_books_last_half="0")
            if "flushes" in r:
                row["flushes"] = len(r["flushes"])
            rows.append(row)
            print({k: v for k, v in row.items() if not k.endswith("under_2se") and k != "label"})
    write_csv("table.csv", rows)

    # distance bins, main_K1
    chunks = chunk_set("main")
    book, masks = bin_masks(chunks)
    fz = get("frozen", "main_K1")["loss"].reshape(-1)
    drows = []
    for name, m in masks.items():
        m = m & np.isfinite(fz)
        r = {"bin": name, "tokens": int(m.sum())}
        for num, cfg in (("2", "all"), ("3", "store"), ("4", "decay"), ("7", "full")):
            x = get(cfg, "main_K1")
            if x is not None:
                r.update(fmt(summary(fz - x["loss"].reshape(-1), book, m), "gain_%s_%s" % (num, cfg)))
        drows.append(r)
    write_csv("distance.csv", drows)

    # mixer along the mixed stream
    mrows, prof = [], []
    traces = {}
    for num, cfg in (("6", "store_fallback"), ("7", "full")):
        r = get(cfg, "mixed")
        if r is None:
            continue
        order = r["order"]
        flushes = set(r["flushes"].tolist()) if "flushes" in r else set()
        wt = np.array([np.nanmean(r["weight"][u, k]) for u, k in order])
        traces[cfg] = (wt, sorted(flushes), [u >= n_books for u, _ in order])
        for i, (u, k) in enumerate(order):
            mrows.append({"config": num, "name": cfg, "position": i, "unit": int(u), "chunk": int(k),
                          "kind": "wiki" if u >= n_books else "book", "weight": "%.4f" % wt[i], "flush": i in flushes})
        for kind, sl, n in (("book", slice(0, n_books), r["weight"].shape[1]), ("wiki", slice(n_books, None), 4)):
            for k in range(n):
                prof.append({"config": num, "name": cfg, "kind": kind, "chunk_in_unit": k,
                             "weight": "%.4f" % np.nanmean(r["weight"][sl, k])})
    write_csv("mixer.csv", mrows)
    write_csv("mixer_profile.csv", prof)

    # runtimes
    from e9_managed_cache import RUNS
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for fn in sorted(os.listdir(RUNS)):
            if fn.endswith(".npz"):
                md = json.loads(str(np.load(os.path.join(RUNS, fn))["meta"]))
                f.write("%s %d s%s\n" % (fn[:-4], md["seconds"], " (same run as %s)" % md["shared_run"]
                                         if "shared_run" in md else ""))
        pk = json.load(open(os.path.join(OUT, "picks.json")))
        f.write("pick (validation grids, 1 + 10 + 12 store runs on valmixed) %d s\n" % pk["seconds"])

    # plots
    style()
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in csv.DictReader(open(os.path.join(HERE, "e7_update_rules", "exchange.csv"))):
        ax.scatter(float(r["wiki_forgetting_K1"]), float(r["gain_K1"]), color="#c8c7c3", s=18, zorder=2)
        ax.annotate("%s %s" % (r["size"].upper(), r["rule"]), (float(r["wiki_forgetting_K1"]), float(r["gain_K1"])),
                    fontsize=5.5, color=MUTED, xytext=(3, -3), textcoords="offset points")
    mvf = {(r["quantity"], r["stream"]): r for r in csv.DictReader(open(os.path.join(HERE, "e2_where", "mlp_vs_full.csv")))}
    ax.scatter(float(mvf["WikiText forgetting, last half", "K = 1"]["mlp"]),
               float(mvf["gain in order", "all tokens"]["mlp"]), color="#c8c7c3", marker="s", s=18, zorder=2)
    ax.annotate("14M sgd MLPs only (E2)", (float(mvf["WikiText forgetting, last half", "K = 1"]["mlp"]),
                                           float(mvf["gain in order", "all tokens"]["mlp"])),
                fontsize=5.5, color=MUTED, xytext=(3, 3), textcoords="offset points")
    colors = {"all": INK, "store": BLUE, "decay": BLUE, "decay_flush": BLUE, "store_fallback": ORANGE, "full": ORANGE}
    markers = {"all": "o", "store": "o", "decay": "s", "decay_flush": "^", "store_fallback": "o", "full": "^"}
    for row in rows:
        if row["stream"] != "main_K1" or "heldout_wiki_last_half_se" not in row:
            continue
        x, y = float(row["heldout_wiki_last_half"]), float(row["gain"])
        ax.errorbar(x, y, xerr=float(row["heldout_wiki_last_half_se"]), yerr=float(row["gain_se"]),
                    color=colors[row["name"]], marker=markers[row["name"]], ms=6, capsize=2, ls="none", zorder=3,
                    label="%s %s" % (row["config"], row["label"]))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("collateral forgetting, held-out WikiText (last half, nats/token)")
    ax.set_ylabel("gain over frozen, main K = 1 (nats/token)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("E9 configurations (colour) against E2 / E7 runs (grey); bars +- 1 SE", fontsize=8.5, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "tradeoff.png"), dpi=160)

    if traces:
        fig, axes = plt.subplots(len(traces), 1, figsize=(7.4, 1.9 * len(traces) + 0.4), sharex=True, squeeze=False)
        for ax, (cfg, (wt, fl, wiki)) in zip(axes[:, 0], traces.items()):
            wiki = np.array(wiki)
            for i in np.nonzero(wiki)[0]:
                ax.axvspan(i - 0.5, i + 0.5, color=ORANGE, alpha=0.25, lw=0)
            ax.plot(np.arange(len(wt)), wt, color=BLUE, lw=0.7)
            for i in fl:
                ax.axvline(i, color=INK, lw=0.8, ls=":")
            ax.set_ylim(0, 1)
            ax.set_ylabel("weight on store")
            ax.set_title("%s (orange: WikiText blocks; dotted: flushes, %d)" % (dict((c, l) for _, c, l in CONFIGS)[cfg],
                                                                              len(fl)), fontsize=8, color=INK, loc="left")
        axes[-1, 0].set_xlabel("chunk of the mixed stream")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "mixer.png"), dpi=160)
