"""E9 offline follow-up: the fallback on top of other online learners, from saved per-token losses.

The fixed-share mixer (mixer.py, unchanged; initial weight 0.5, state carried across chunks, books and
WikiText blocks, each chunk's unscored first token skipped) only needs each expert's per-token loss in
stream order, so it is applied here to saved runs (float16 losses):
    2F  all-parameters online learner (registry main_K1 / main_shuffled, E9's mixed__all) + fallback to frozen
    4F  store + decay, no flush (E9's <stream>__decay) + fallback to frozen
The frozen expert is the frozen run of each stream (registry main_K1 frozen, mapped to the reading order of
each main stream; E9's mixed__frozen). Scored: -log(w p_online + (1 - w) p_frozen).
check:  recompute E9's configurations 6 (store + fallback) and 7 (full) from their saved store runs and
        compare with the saved mixture losses and weights, per token, and the means E9 reports. E9 mixed
        float32 losses; only float16 are saved, and the rounding (up to ~0.004 nats per input) is carried by
        the mixer weight across tokens, so per-token differences reach ~0.02 nats (the same as adding noise
        of half a float16 ulp to the inputs) while the means agree to ~4e-5. Pass: means within 1e-4.
pick:   alpha for 2F and 4F on E9's validation mixed stream (valmixed) with E9's grid and objective (mean
        loss over all scored tokens). E9 saved no validation runs, so two small runs are made here and
        saved as runs/offline_valmixed__{frozen,all,decay}.npz: frozen, plain SGD on all parameters at the
        E0 pick (lr 1e-4), and the store with decay at E9's picks (layer 2 MLP, lr 1e-3, lam 1e-3).
report: offline_table.csv, offline_paired.csv, offline_distance.csv, offline_mixer.csv,
        plots/offline_tradeoff.png, offline_runtime.txt (only saved runs).
Usage (from experiments/):
    ../venv/bin/python e9_managed_cache/e9_offline.py check | pick | report
"""
import csv
import json
import os
import sys
import time

OUT = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(OUT)
sys.path[:0] = [HERE, os.path.join(HERE, "lib"), OUT]

import numpy as np  # noqa: E402

import runs  # noqa: E402
import streams  # noqa: E402
from data import L, chunk_set  # noqa: E402
from e9_managed_cache import GRID_ALPHA, LAYER, RUNS, path_of, store_lr  # noqa: E402
from metrics import bin_masks, se, summary  # noqa: E402
from mixer import fixed_share  # noqa: E402

N_BOOKS = 25
STREAMS = ["main_K1", "main_shuffled", "mixed"]


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def reading(path, name="loss"):
    """(array [chunks, L] float64 in reading order, order [chunks, 2]) from an npz in runs.run's layout."""
    z = np.load(path)
    return z[name].astype(np.float64), z["order"].astype(np.int64)


def layout(x, order):
    out = np.full((order[:, 0].max() + 1, order[:, 1].max() + 1, L), np.nan)
    out[order[:, 0], order[:, 1]] = x
    return out


def frozen_reading(st, order):
    """Frozen per-token losses in the reading order of stream st."""
    if st == "mixed":
        f, o = reading(path_of("mixed__frozen"))
        fl = layout(f, o)
    else:
        fl = runs.load(runs.frozen_key("14m", st))["loss"]
    return fl[order[:, 0], order[:, 1]]


def run_path(cfg, st):
    if cfg == "all" and st != "mixed":
        return runs.path_of(runs.find("14m", "sgd", st))
    return path_of("%s__%s" % (st, cfg))


def picks():
    p = os.path.join(OUT, "offline_picks.json")
    return json.load(open(p)) if os.path.exists(p) else None


def config(name, st):
    """{loss [chunks, L] reading order, order, weight or None} for a configuration on a stream."""
    if name in ("2F", "4F"):
        base = {"2F": "all", "4F": "decay"}[name]
        x, order = reading(run_path(base, st))
        mix, w = fixed_share(x, frozen_reading(st, order), picks()["alpha"][name])
        return {"loss": mix, "order": order, "weight": w}
    cfg = {"1": "frozen", "2": "all", "3": "store", "4": "decay", "6": "store_fallback", "7": "full"}[name]
    if cfg == "frozen":
        x, order = reading(run_path("store", st))  # any run of the stream, for its order
        return {"loss": frozen_reading(st, order), "order": order, "weight": None}
    x, order = reading(run_path(cfg, st))
    z = np.load(run_path(cfg, st))
    return {"loss": x, "order": order, "weight": z["weight"].astype(np.float64) if "weight" in z.files else None}


# ---------------------------------------------------------------- check


def check():
    lines, ok_all = [], True
    e9 = {(r["config"], r["stream"]): r for r in csv.DictReader(open(os.path.join(OUT, "table.csv")))}
    alpha = json.load(open(os.path.join(OUT, "picks.json")))["alpha"]
    for num, cfg, base, a in (("6", "store_fallback", "store", alpha["store_fallback"]),
                              ("7", "full", "decay_flush", alpha["full"])):
        for st in STREAMS:
            x, order = reading(run_path(base, st))
            fz = frozen_reading(st, order)
            mix, w = fixed_share(x, fz, a)
            saved, _ = reading(run_path(cfg, st))
            wsaved, _ = reading(run_path(cfg, st), "weight")
            m = np.isfinite(saved)
            d = np.abs(mix[m] - saved[m])
            # float16 tolerance: two ulps of the saved value (inputs and output were each rounded to float16)
            ulp = np.spacing(saved[m].astype(np.float16)).astype(np.float64)
            within = bool((d <= 2 * ulp).all())
            fl, ml = layout(fz, order)[:N_BOOKS].reshape(-1), layout(mix, order)[:N_BOOKS].reshape(-1)
            book = np.repeat(np.arange(N_BOOKS), len(fl) // N_BOOKS)
            gain = summary(fl - ml, book, np.isfinite(fl))["mean"]
            dg = gain - float(e9[num, st]["gain"])
            msg = "config %s on %s: max per-token |offline - saved| %.2e (%s 2 float16 ulps), weight max diff %.2e; " \
                  "gain %.4f vs E9 %s (diff %+.1e)" % (num, st, d.max(), "all within" if within else "NOT within",
                                                       float(np.nanmax(np.abs(w - wsaved))), gain, e9[num, st]["gain"], dg)
            ok = abs(dg) < 1e-4  # agreed criterion: the reported means; per-token differences only reported
            if st == "mixed":
                wf, wm = layout(fz, order)[N_BOOKS:].reshape(-1), layout(mix, order)[N_BOOKS:].reshape(-1)
                wb = float(np.nanmean(wm - wf))
                dw = wb - float(e9[num, st]["wiki_block_change"])
                msg += "; WikiText blocks %+.4f vs E9 %s (diff %+.1e)" % (wb, e9[num, st]["wiki_block_change"], dw)
                ok = ok and abs(dw) < 1e-4
            ok_all &= ok
            lines.append(("PASS " if ok else "FAIL ") + msg)
            print(lines[-1], flush=True)
    lines.append("overall: %s (criterion: gains and WikiText-block means within 1e-4 of E9's table.csv; per-token "
                 "differences come from float16 inputs carried through the mixer weight and are reported, not "
                 "required to be within float16 rounding)" % ("PASS" if ok_all else "FAIL"))
    print(lines[-1])
    with open(os.path.join(OUT, "offline_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- pick (validation only)


def pick():
    from learner import Learner, score, setup
    from store import run_store
    model, init, info = setup("14m")
    ids, order, n_books = streams.get("valmixed")
    wiki = np.array([u >= n_books for u, _ in order])
    e9p = json.load(open(os.path.join(OUT, "picks.json")))
    sec = {}
    t0 = time.time()
    frozen = score(model, init, ids.reshape(-1)).reshape(len(ids), L)
    sec["frozen"] = time.time() - t0
    t0 = time.time()
    lr_all = runs.picks()[("14m", "sgd", "all")]
    online = Learner(model, init, "sgd", lr_all, "all").read(ids.reshape(-1)).reshape(len(ids), L)
    sec["all"] = time.time() - t0
    t0 = time.time()
    decay = run_store(model, init, ids, e9p["decay"]["lr"], LAYER, lam=e9p["decay"]["lam"])["loss"]
    sec["decay"] = time.time() - t0
    for name, x, meta in (("frozen", frozen, {}), ("all", online, {"rule": "sgd", "lr": lr_all, "params": "all"}),
                          ("decay", decay, {"layer": LAYER, "lr": e9p["decay"]["lr"], "lam": e9p["decay"]["lam"]})):
        np.savez_compressed(path_of("offline_valmixed__%s" % name), loss=x.astype(np.float16),
                            order=np.array(order, dtype=np.int16),
                            meta=json.dumps({"stream": "valmixed", "config": name, "seconds": round(sec[name]), **meta}))
        print("valmixed %s: mean loss %.4f, %.0f s" % (name, np.nanmean(x), sec[name]), flush=True)
    rows, pk = [], {}
    for name, x in (("2F", online), ("4F", decay)):
        scores = []
        for a in GRID_ALPHA:
            mix, w = fixed_share(x, frozen, a)
            rows.append({"config": name, "alpha": a, "val_loss": "%.4f" % np.nanmean(mix),
                         "val_loss_books": "%.4f" % np.nanmean(mix[~wiki]), "val_loss_wiki": "%.4f" % np.nanmean(mix[wiki]),
                         "frozen_wiki": "%.4f" % np.nanmean(frozen[wiki]), "expert_val_loss": "%.4f" % np.nanmean(x),
                         "mean_weight_books": "%.3f" % np.nanmean(w[~wiki]), "mean_weight_wiki": "%.3f" % np.nanmean(w[wiki])})
            scores.append(np.nanmean(mix))
        pk[name] = GRID_ALPHA[int(np.argmin(scores))]
    write_csv("offline_grid_alpha.csv", rows)
    out = {"alpha": pk, "grid": GRID_ALPHA, "objective": "mean loss over all scored tokens of valmixed",
           "decay_val_loss_check": "%.4f (E9 grid_decay.csv, lr %g lam %g)" % (np.nanmean(decay), e9p["decay"]["lr"],
                                                                                e9p["decay"]["lam"]),
           "seconds": {k: round(v) for k, v in sec.items()}}
    with open(os.path.join(OUT, "offline_picks.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out)


# ---------------------------------------------------------------- report


def fmt(s, name):
    return {name: "%.4f" % s["mean"], name + "_se": "%.4f" % s["se"], name + "_under_2se": s["under_2se"]}


def report():
    from plots import plt, style, BLUE, ORANGE, AQUA, INK, MUTED
    names = {"2": "online, all parameters", "3": "store only", "4": "store + decay", "6": "store + fallback",
             "7": "store + decay + flush + fallback", "2F": "online, all parameters + fallback",
             "4F": "store + decay + fallback (no flush)"}
    cfgs = ["2", "4", "6", "7", "2F", "4F"]
    data = {}
    for st in STREAMS:
        for c in cfgs + ["1", "3"]:
            r = config(c, st)
            data[c, st] = {"loss": layout(r["loss"], r["order"]), "weight": None if r["weight"] is None else
                           layout(r["weight"], r["order"]), "order": r["order"]}
    book_l = lambda c, st: data[c, st]["loss"][:N_BOOKS].reshape(-1)
    wiki_l = lambda c: data[c, "mixed"]["loss"][N_BOOKS:].reshape(-1)
    n_tok = data["1", "main_K1"]["loss"][:N_BOOKS].reshape(-1).shape[0]
    book = np.repeat(np.arange(N_BOOKS), n_tok // N_BOOKS)
    allm = np.isfinite(book_l("1", "main_K1"))
    wz = wiki_l("1")
    shp = data["1", "mixed"]["loss"][N_BOOKS:].shape  # WikiText units in the 78-chunk layout (NaN past chunk 3)
    wblock = np.repeat(np.arange(shp[0]), shp[1] * L)
    wm = np.isfinite(wz)
    rows = []
    for c in cfgs:
        for st in STREAMS:
            fz = book_l("1", st)
            r = {"config": c, "label": names[c], "stream": st, "loss_books": "%.4f" % np.nanmean(book_l(c, st))}
            r.update(fmt(summary(fz - book_l(c, st), book, allm), "gain"))
            if st == "main_K1":
                r.update(fmt(summary(book_l(c, "main_shuffled") - book_l(c, "main_K1"), book, allm), "locality"))
            if st == "mixed":
                r.update(fmt(summary(wiki_l(c) - wz, wblock, wm), "wiki_block_change"))
                w = data[c, st]["weight"]
                if w is not None:
                    r["weight_books"], r["weight_wiki"] = "%.3f" % np.nanmean(w[:N_BOOKS]), "%.3f" % np.nanmean(w[N_BOOKS:])
            rows.append(r)
            print({k: v for k, v in r.items() if not k.endswith("under_2se") and k != "label"})
    write_csv("offline_table.csv", rows)

    prs = []
    for a, b in (("2F", "7"), ("2F", "4F"), ("4F", "7"), ("2F", "2"), ("4F", "4")):
        r = {"comparison": "%s - %s" % (a, b)}
        r.update(fmt(summary(book_l(b, "main_K1") - book_l(a, "main_K1"), book, allm), "gain_K1"))
        r.update(fmt(summary(book_l(b, "mixed") - book_l(a, "mixed"), book, allm), "gain_mixed_books"))
        r.update(fmt(summary(wiki_l(a) - wiki_l(b), wblock, wm), "wiki_block_change"))
        prs.append(r)
        print({k: v for k, v in r.items() if not k.endswith("under_2se")})
    write_csv("offline_paired.csv", prs)

    chunks = chunk_set("main")
    bk, masks = bin_masks(chunks)
    fz = data["1", "main_K1"]["loss"].reshape(-1)
    drows = []
    for name, m in masks.items():
        m = m & np.isfinite(fz)
        r = {"bin": name, "tokens": int(m.sum())}
        for c in ("2", "2F", "4F", "7"):
            r.update(fmt(summary(fz - data[c, "main_K1"]["loss"].reshape(-1), bk, m), "gain_" + c))
        drows.append(r)
    write_csv("offline_distance.csv", drows)

    mrows = []
    for c in ("2F", "4F", "6", "7"):
        w = data[c, "mixed"]["weight"]
        stats = [("books, all chunks", w[:N_BOOKS]), ("WikiText blocks, all chunks", w[N_BOOKS:]),
                 ("books, first 4 chunks", w[:N_BOOKS, :4]), ("books, last 4 chunks", w[:N_BOOKS, -4:])]
        stats += [("WikiText blocks, chunk %d" % k, w[N_BOOKS:, k]) for k in range(4)]
        per_chunk = np.nanmean(w[:N_BOOKS], axis=2)
        stats.append(("books, chunk-to-chunk SD of the chunk mean", None))
        for name, x in stats:
            v = np.nanstd(per_chunk) if x is None else np.nanmean(x)
            mrows.append({"config": c, "label": names[c], "statistic": name, "weight_on_online_expert": "%.3f" % v})
    write_csv("offline_mixer.csv", mrows)

    with open(os.path.join(OUT, "offline_runtime.txt"), "w") as f:
        pk = picks()
        for k, v in pk["seconds"].items():
            f.write("offline_valmixed__%s %d s\n" % (k, v))
        f.write("check, report: offline (mixer over saved losses), well under a minute each\n")

    style()
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    tab = {(r["config"], r["stream"]): r for r in rows}
    extra = {}
    for c in ("3",):
        g = summary(book_l("1", "mixed") - book_l(c, "mixed"), book, allm)
        wb = summary(wiki_l(c) - wz, wblock, wm)
        extra[c] = (g, wb)
    style_of = {"2": (INK, "o"), "3": (BLUE, "o"), "4": (BLUE, "s"), "6": (ORANGE, "o"), "7": (ORANGE, "^"),
                "2F": (AQUA, "o"), "4F": (AQUA, "s")}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for c in ("2", "3", "4", "6", "7", "2F", "4F"):
        if c in extra:
            g, wb = extra[c]
            x, xs, y, ys = wb["mean"], wb["se"], g["mean"], g["se"]
        else:
            r = tab[c, "mixed"]
            x, xs, y, ys = (float(r["wiki_block_change"]), float(r["wiki_block_change_se"]), float(r["gain"]),
                            float(r["gain_se"]))
        color, marker = style_of[c]
        ax.errorbar(x, y, xerr=xs, yerr=ys, color=color, marker=marker, ms=6, capsize=2, ls="none",
                    label="%s %s" % (c, names[c]))
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("loss - frozen on the WikiText blocks (nats/token; > 0: forgetting)")
    ax.set_ylabel("gain over frozen on the books (nats/token)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Mixed stream (25 books + 25 WikiText blocks); bars +- 1 SE", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "offline_tradeoff.png"), dpi=160)


if __name__ == "__main__":
    {"check": check, "pick": pick, "report": report}[sys.argv[1]]()
