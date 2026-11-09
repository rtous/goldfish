"""Regression report (run once, before experiments_LLM2_cache/ was deleted; it needs that folder, which is
in the git history): headline numbers of claims C1-C10, old (experiments_LLM2_cache/) against new
(experiments/), written to REGRESSION.md.

Kinds:
    exact           same run as before (same chunks, order, rule, rate): |new - old| must be < 1e-3
    expected shift  changed chunking / order (tests 1-6 used one stream chunked across book
                    boundaries and another shuffle) or a changed learning-rate pick
Shifts are flagged when |new - old| > 2 SE (SE of the new value, over books) or the sign changes.
Also per-token identity of the exact runs (float16 losses, same layout) and of their snapshots.
Usage (from experiments/): ../venv/bin/python regression.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from metrics import forgetting, se  # noqa: E402

OLD = os.path.join(HERE, "..", "experiments_LLM2_cache")
rows, ident = [], []


def rd(path):
    return list(csv.DictReader(open(path)))


def one(table, **kw):
    r = [x for x in table if all(str(x[k]) == str(v) for k, v in kw.items())]
    assert len(r) == 1, (kw, len(r))
    return r[0]


def add(claim, quantity, old, new, s, kind, note=""):
    old, new = float(old), float(new)
    s = float(s) if s not in ("", None) and np.isfinite(float(s)) else float("nan")
    d = new - old
    flag = ""
    if kind == "exact":
        flag = "MISMATCH" if abs(d) >= 1e-3 else "ok"
    else:
        f = []
        if np.isfinite(s) and abs(d) > 2 * s:
            f.append("> 2 SE")
        if np.sign(old) != np.sign(new) and max(abs(old), abs(new)) >= 5e-4:
            f.append("sign change")
        flag = ", ".join(f) or "ok"
    rows.append({"claim": claim, "quantity": quantity, "old": old, "new": new, "diff": d, "se": s, "kind": kind,
                 "flag": flag, "note": note})


def identity(name, new, old):
    c = runs.compare(new, old)
    ident.append({"what": name, **c})


def main():
    pk = runs.picks()
    # ---------------------------------------------------------------- C1-C4: E1 vs tests 1, 3, 4, 5
    e1 = rd(os.path.join(HERE, "e1_locality", "distance.csv"))
    t1, t3, t4, t5 = (rd(os.path.join(OLD, p)) for p in ("test1_distance/results.csv", "test3_baselines/distance.csv",
                                                        "test4_checks/results.csv", "test5_capacity/results.csv"))
    shift = "expected shift"
    for b in ("all tokens", "same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k",
              "earlier chunk, >= 100k", "never seen"):
        n = one(e1, bin=b)
        add("C1", "gain in order, " + b, one(t1, bin=b)["gain"], n["gain_in_order"], n["gain_in_order_se"], shift, "test 1")
    for b in ("all tokens", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k"):
        n = one(e1, bin=b)
        add("C2", "gain shuffled, " + b, one(t4, bin=b)["gain_shuffled"], n["gain_shuffled"], n["gain_shuffled_se"], shift,
            "test 4")
        add("C2", "locality, " + b, one(t4, bin=b)["locality"], n["locality"], n["locality_se"], shift, "test 4")
    for b in ("earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k"):
        for split in ("same book", "earlier book"):
            n = one(e1, bin="%s, %s" % (b, split))
            o = [x for x in t5 if x["bin"] == b and x["book"] == split]
            if o:
                add("C3", "locality, %s, %s" % (b, split), o[0]["locality14"], n["locality"], n["locality_se"], shift,
                    "test 5")
                add("C3", "gain in order, %s, %s" % (b, split), o[0]["online14"], n["gain_in_order"],
                    n["gain_in_order_se"], shift, "test 5")
    for b in ("all tokens", "same chunk", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "earlier chunk, 10k-100k"):
        n = one(e1, bin=b)
        add("C4", "unigram cache gain, " + b, one(t3, bin=b)["gain_unigram"], n["gain_unigram"], n["gain_unigram_se"],
            shift, "test 3")
        add("C4", "online - unigram, " + b, float(one(t3, bin=b)["gain_online"]) - float(one(t3, bin=b)["gain_unigram"]),
            n["in_order_minus_unigram"], n["in_order_minus_unigram_se"], shift, "test 3")

    # ---------------------------------------------------------------- C5: E2 vs tests 3, 4, 6
    e2 = rd(os.path.join(HERE, "e2_where", "results.csv"))
    t6 = rd(os.path.join(OLD, "test6", "results.csv"))
    old_lr = {"output": 3e-3, "no_output": 1e-4, "mlp": 1e-4, "attention": 1e-4,
              **dict(zip(["mlp@%d" % i for i in range(6)], [1e-3, 3e-3, 3e-4, 3e-3, 3e-3, 1e-3]))}
    for sub, src in (("output", ("t4", "gain_output_lr3e-3")), ("no_output", ("t3", "gain_no_output")),
                     ("mlp", ("t4", "gain_mlp")), ("attention", ("t4", "gain_attention")),
                     *[("mlp@%d" % i, ("t6", "mlp_l%d" % (i + 1))) for i in range(6)]):
        for b in ("all tokens", "earlier chunk, < 1k", "earlier chunk, 1k-10k", "never seen"):
            tab = {"t3": t3, "t4": t4, "t6": [x for x in t6 if x["book"] == "all"]}[src[0]]
            n = one(e2, subset=sub, bin=b)
            note = "test %s, old lr %g, new lr %s" % (src[0][1], old_lr[sub], n["lr"])
            add("C5", "%s, %s" % (sub, b), one(tab, bin=b)[src[1]], n["gain"], n["gain_se"], shift, note)

    # ---------------------------------------------------------------- C6: E3 vs test 7
    e3 = rd(os.path.join(HERE, "e3_locality_levels", "results.csv"))
    t7 = rd(os.path.join(OLD, "test7_aggressive", "results.csv"))
    for st, old_st in (("long", "long"), ("main_K1", "K1"), ("main_K5", "K5"), ("main_K25", "K25"),
                       ("main_shuffled", "shuffled")):
        n = one(e3, stream=st)
        add("C6", "gain, " + old_st, one(t7, setting="full_1e-4", stream=old_st)["gain"], n["gain"], n["gain_se"], "exact",
            "test 7")
    z7 = np.load(os.path.join(OLD, "test7_aggressive", "losses.npz"))
    for s in ("14m", "31m", "70m"):
        identity("frozen %s main vs test 7" % s, runs.load(runs.frozen_key(s, "main_K1"))["loss"], z7["frozen%s_books25" % s])
    identity("frozen 14m long vs test 7", runs.load(runs.frozen_key("14m", "long"))["loss"], z7["frozen14m_long"])
    for st, o in (("main_K1", "K1"), ("main_shuffled", "shuffled"), ("long", "long")):
        identity("14m sgd %s vs test 7 full_1e-4_%s" % (st, o), runs.load(runs.find("14m", "sgd", st))["loss"],
                 z7["full_1e-4_%s" % o])

    # ---------------------------------------------------------------- C7: E4 vs test 11
    e4m, e4h = (rd(os.path.join(HERE, "e4_eviction", f)) for f in ("means.csv", "halving.csv"))
    o11m, o11h = (rd(os.path.join(OLD, "test11_probe_eviction", f)) for f in ("means.csv", "halving.csv"))
    for I in [r["I"] for r in o11m if r["setting"] == "full_1e-4" and r["I"] != "pre"]:
        n, o = one(e4m, I=I), one(o11m, setting="full_1e-4", I=I)
        for nk, ok in (("memory_A", "memory_A"), ("gain_A2", "gain_A2"), ("gain_C2", "gain_C2"),
                       ("memory_rare_in_A1", "memory_rare_in_first_half"),
                       ("memory_rare_not_in_A1", "memory_rare_not_in_first_half")):
            add("C7", "%s, I = %s" % (nk, I), o[ok], n[nk], n[nk + "_se"], "exact", "test 11")
    for q, oq in (("memory_A", "memory_A"), ("memory_rare_in_A1", "memory_rare_in_first_half"), ("gain_A2", "gain_A2")):
        n, o = one(e4h, quantity=q), one(o11h, setting="full_1e-4", quantity=oq)
        if n["halves_at_tokens"][0].isdigit() and o["halves_at_tokens"][0].isdigit():
            add("C7", "%s halves at (tokens)" % q, o["halves_at_tokens"], n["halves_at_tokens"], "", "exact", "test 11; "
                "|diff| < 1 token required")
            rows[-1]["flag"] = "ok" if abs(rows[-1]["diff"]) < 1 else "MISMATCH"
    z = np.load(os.path.join(HERE, "e4_eviction", "losses.npz"))
    zo = np.load(os.path.join(OLD, "test11_probe_eviction", "losses_full_1e-4.npz"))
    identity("E4 probe losses vs test 11 full_1e-4", z["probe"], zo["probe"])

    # ---------------------------------------------------------------- C8: E5 vs test 12 (detokenized WikiText)
    e5, e5c = (rd(os.path.join(HERE, "e5_forgetting", f)) for f in ("summary.csv", "k1_vs_shuffled.csv"))
    d12, s12 = (rd(os.path.join(OLD, "test12_general", f)) for f in ("detok.csv", "summary.csv"))
    for st, old in (("main_K1", "14m_full_1e-4_K1"), ("main_shuffled", "14m_full_1e-4_shuffled"),
                    ("verylong", "14m_full_1e-4_verylong")):
        n, od, osu = one(e5, stream=st), one(d12, run=old), one(s12, run=old)
        add("C8", "%s stream gain" % st, osu["stream_gain"], n["stream_gain"], n["stream_gain_se"], "exact", "test 12")
        add("C8", "%s WikiText last half" % st, od["last_half_detok"], n["wiki_last_half"], n["wiki_last_half_se"], "exact",
            "test 12")
        add("C8", "%s WikiText final" % st, od["end_detok"], n["wiki_final"], n["wiki_final_se"], "exact", "test 12")
        add("C8", "%s books last half" % st, osu["mean_forgetting_last_half_books"], n["books_last_half"],
            n["books_last_half_se"], "exact", "test 12")
        add("C8", "%s books final" % st, osu["final_forgetting_books"], n["books_final"], n["books_final_se"], "exact",
            "test 12")
        add("C8", "%s exchange rate" % st, od["exchange_rate_detok"], n["exchange_rate"], "", "exact",
            "test 12; 2 decimals, |diff| < 0.01 required")
        rows[-1]["flag"] = "ok" if abs(rows[-1]["diff"]) < 0.01 else "MISMATCH"
        r = runs.load(runs.find("14m", "sgd", st))
        zo, zd = (np.load(os.path.join(OLD, "test12_general", "losses%s_%s.npz" % (p, old))) for p in ("", "_detok"))
        old_snaps = np.concatenate([zd["detok"], zo["snaps"][:, -90:]], 1)
        identity("14m sgd %s snapshots vs test 12 (WikiText detok + books)" % st, r["snaps"], old_snaps)
        if st == "verylong":
            identity("14m sgd verylong stream vs test 12", r["loss"], zo["stream"])
    o_k1 = {p: np.load(os.path.join(OLD, "test12_general", "losses_detok_14m_full_1e-4_%s.npz" % p)) for p in ("K1", "shuffled")}
    pos = o_k1["K1"]["positions"]
    half = [k for k in range(1, len(pos)) if pos[k] > pos[-1] / 2]
    dw = np.mean([o_k1["shuffled"]["detok"][k] - o_k1["K1"]["detok"][k] for k in half], axis=0)
    add("C8", "shuffled - K1, WikiText last half", dw.mean(), e5c[0]["shuffled_minus_K1_wiki"],
        e5c[0]["shuffled_minus_K1_wiki_se"], "exact", "test 12 detok files")
    add("C8", "shuffled - K1, books last half", one(rd(os.path.join(OLD, "test12_general", "k1_vs_shuffled.csv")),
                                                    setting="14m_full_1e-4")["shuffled_minus_K1_books"],
        e5c[0]["shuffled_minus_K1_books"], e5c[0]["shuffled_minus_K1_books_se"], "exact", "test 12")

    # ---------------------------------------------------------------- C9: E6 vs test 10C
    e6, e6p, e6pos = (rd(os.path.join(HERE, "e6_long_books", f)) for f in ("results.csv", "pairs.csv", "position.csv"))
    c, cp, cpos = (rd(os.path.join(OLD, "test10_followups", f)) for f in ("partC.csv", "partC_pairs.csv", "partC_position.csv"))
    names = {"frozen14m": "frozen14m", "frozen31m": "frozen31m", "frozen70m": "frozen70m", "online14m": "14m full_1e-4",
             "online31m": "31m online lr3e-4"}
    for n_, o_ in names.items():
        n, o = one(e6, run=n_), one(c, run=o_)
        add("C9", "%s loss" % n_, o["loss"], n["loss"], n["se"], "exact", "test 10C")
        add("C9", "%s share of 14M-31M gap closed" % n_, o["gap31_closed"], n["gap31_closed"], "", "exact", "test 10C")
    for name, old_name in (("14M online - frozen 14M", "14M full_1e-4 - frozen 14M"),
                           ("31M online - frozen 31M", "31M online - frozen 31M"),
                           ("31M online - 14M online", "31M online - 14M full_1e-4"),
                           ("frozen 31M - 14M online", "frozen 31M - 14M full_1e-4"),
                           ("31M online - frozen 70M", "31M online - frozen 70M")):
        n, o = one(e6p, comparison=name), one(cp, comparison=old_name)
        add("C9", name, o["token_mean"], n["mean"], n["se"], "exact", "test 10C pairs")
    for j, (n, o) in enumerate(zip(e6pos, cpos)):
        f14, f31, o14 = float(o["frozen14m"]), float(o["frozen31m"]), float(o["14m full_1e-4"])
        if j % 5 == 0 or j == len(e6pos) - 1:
            add("C9", "gap closed, %s" % n["position"], (f14 - o14) / (f14 - f31), n["gap31_closed_by_online14m"], "",
                "exact", "test 10C positions (old: ratio of rounded book means)")
            rows[-1]["flag"] = "ok" if abs(rows[-1]["diff"]) < 5e-3 else "MISMATCH"
    zc = np.load(os.path.join(OLD, "test10_followups", "losses_C.npz"))
    for n_, o_ in (("frozen14m", "frozen14m"), ("frozen31m", "frozen31m"), ("frozen70m", "frozen70m"),
                   ("online14m", "14m_full_1e-4"), ("online31m", "31m_online_lr3e-4")):
        k = runs.frozen_key(n_[-3:], "verylong") if n_.startswith("frozen") else runs.find(n_[-3:], "sgd", "verylong")
        identity("verylong %s vs test 10C %s" % (n_, o_), runs.load(k)["loss"], zc[o_])

    # ---------------------------------------------------------------- C10: E7 vs test 13
    e7, e7x = (rd(os.path.join(HERE, "e7_update_rules", f)) for f in ("results.csv", "exchange.csv"))
    o13, o13x = (rd(os.path.join(OLD, "test13_optimizers", f)) for f in ("results.csv", "exchange.csv"))
    old_picks = {(r["size"], r["rule"]): float(r["lr"]) for r in o13x}
    for size in ("14m", "31m", "70m"):
        for rule in ("sgd", "adam", "nsgd"):
            new_lr = pk[size, rule, "all"]
            kind = "exact" if abs(new_lr - old_picks[size, rule]) < 1e-15 else shift
            note = "test 13, lr %g" % new_lr + ("" if kind == "exact" else " (old %g)" % old_picks[size, rule])
            for b in ("all tokens", "earlier chunk < 1k", "1k-10k same book", "never seen"):
                n, o = one(e7, size=size, rule=rule, bin=b), one(o13, size=size, rule=rule, bin=b)
                add("C10", "%s %s gain K1, %s" % (size, rule, b), o["gain_K1"], n["gain_K1"], n["gain_K1_se"], kind, note)
                add("C10", "%s %s locality, %s" % (size, rule, b), o["locality"], n["locality"], n["locality_se"], kind, note)
            # forgetting on the snapshots test 13 had (every 80 chunks, or test 12's every 40 for reused sgd runs)
            ox = one(o13x, size=size, rule=rule)
            for st in ("K1", "shuffled"):
                r = runs.load(runs.find(size, rule, "main_" + st))
                if ox["reused"]:
                    old_pos = r["positions"]
                else:
                    old_pos = np.load(os.path.join(OLD, "test13_optimizers", "losses_%s_%s_%s.npz" % (size, rule, st)))["positions"] // 256
                idx = [i for i, p in enumerate(r["positions"]) if p in set(old_pos.tolist())]
                f = forgetting(r["snaps"][idx], r["positions"][idx], r["n_wiki"])
                if st == "K1":
                    f_k1_old_grid = f["wiki"]["last_half"]
                add("C10", "%s %s WikiText forgetting %s (old snapshot grid)" % (size, rule, st), ox["wiki_forgetting_%s" % st],
                    f["wiki"]["last_half"], f["wiki"]["last_half_se"], kind, note)
                add("C10", "%s %s books forgetting %s (old snapshot grid)" % (size, rule, st), ox["books_forgetting_%s" % st],
                    f["books"]["last_half"], f["books"]["last_half_se"], kind, note)
                if kind == "exact":
                    if ox["reused"]:
                        name = "31m_full_3e-4" if size == "31m" else "14m_full_1e-4"
                        old_stream = np.load(os.path.join(OLD, "test12_general", "losses_%s_%s.npz" % (name, st)))["stream"]
                    else:
                        zo = np.load(os.path.join(OLD, "test13_optimizers", "losses_%s_%s_%s.npz" % (size, rule, st)))
                        old_stream = zo["stream"]
                        identity("%s %s %s snapshots vs test 13 (every 80)" % (size, rule, st), r["snaps"][idx], zo["snaps"])
                    identity("%s %s %s stream vs old" % (size, rule, st), r["loss"], old_stream)
            n = one(e7x, size=size, rule=rule)
            # exact check on the old grid (gain rounded as in the old table), then the reported new-grid value
            add("C10", "%s %s exchange rate K1 (old snapshot grid)" % (size, rule), ox["exchange_rate_K1"],
                round(float(n["gain_K1"]) / f_k1_old_grid, 2), "", kind, note)
            if kind == "exact":
                rows[-1]["flag"] = "ok" if abs(rows[-1]["diff"]) <= 0.01 + 1e-9 else "MISMATCH"
            add("C10", "%s %s exchange rate K1 (new snapshot grid, every 40)" % (size, rule), ox["exchange_rate_K1"],
                n["exchange_rate_K1"], "", "grid change" if not ox["reused"] else kind, note + "; old grid every 80 unless reused")
    d_new = rd(os.path.join(HERE, "e7_update_rules", "diagnostics_summary.csv"))
    d_old = rd(os.path.join(OLD, "test13_optimizers", "diagnostics_summary.csv"))
    for size in ("14m", "31m", "70m"):
        for typ in ("attention", "mlp", "output embedding"):
            n, o = one(d_new, size=size, type=typ, layer="all layers"), one(d_old, size=size, type=typ, layer="all layers")
            add("C10", "diag %s %s relative update" % (size, typ), o["rel_update"], n["rel_update"], "", "exact",
                "test 13 step 1; relative |diff| < 1% required")
            rows[-1]["flag"] = "ok" if abs(rows[-1]["diff"]) <= 0.01 * abs(float(o["rel_update"])) else "MISMATCH"
    write()


def write():
    with open(os.path.join(HERE, "regression.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = ["# Regression: old (experiments_LLM2_cache/) against new (experiments/)", "",
           "Generated by `regression.py` (full table also in `regression.csv`). Kinds: **exact** = same run as before, "
           "must match to float16 precision (|diff| < 1e-3; exchange rates < 0.01; half-lives < 1 token); **expected shift** "
           "= changed chunking / order (E1, E2 against tests 1-6) or changed learning-rate pick. Shifts are flagged when "
           "|diff| > 2 SE (SE of the new value, over books) or the sign changes; **grid change** = the same run summarized on "
           "the new snapshot grid (every 40 chunks instead of test 13's every 80), for information.", ""]
    mism = [r for r in rows if r["kind"] == "exact" and r["flag"].startswith("MISMATCH")]
    flagged = [r for r in rows if r["kind"] != "exact" and r["flag"] != "ok"]
    bad_ident = [r for r in ident if r["max_abs_diff"] > 0]
    out += ["**Summary:** %d numbers compared, %d exact (%d mismatches), %d expected shifts (%d flagged). "
            "Per-token identity: %d arrays compared, %d not bit-identical in float16." % (
                len(rows), sum(r["kind"] == "exact" for r in rows), len(mism), sum(r["kind"] != "exact" for r in rows),
                len(flagged), len(ident), len(bad_ident)), ""]
    out += ["## Wording changes", "",
            "All flagged shifts are in C5 and come from the new subset learning-rate picks (val-pick instead of the "
            "old check books; MLP only and attention only 1e-4 -> 3e-4, layer-2 and layer-4 MLP 3e-3 -> 1e-3).", "",
            "- **C5, MLP only:** was \"keeps most of the long-range gain (+0.216 within 1k, +0.123 at 1k-10k)\"; now it "
            "**beats** the full learner: +0.415 against +0.308 within 1k, +0.158 against +0.144 at 1k-10k, +0.107 against "
            "+0.100 overall (paired differences beyond 2 SE).",
            "- **C5, attention only:** +0.142 / +0.092 -> +0.259 / +0.102 (still below MLP only and the full learner).",
            "- **C5, single layers:** was \"the MLP of layer 2 or 4 alone gains +0.395 / +0.357 within 1k, more than the "
            "full learner\" (at lr 3e-3, grid edge); now +0.242 / +0.187, and no single layer beats the full learner "
            "(best: layer 5, +0.285). \"Every MLP but the last holds a large part; the last is the weakest\" still holds. "
            "\"Several picks at the edge of the grid\" no longer applies (all interior). Layer 2's overall gain and "
            "never-seen loss: +0.055 / -0.129 -> +0.069 / -0.051. \"Layer 4 level with the full learner on recent "
            "same-book tokens\" no longer holds (-0.053 ± 0.004 at 1k-10k same book).",
            "- **C2, numbers only:** share of the gain surviving shuffling 18% -> 20% within 1k, 27% -> 24% at 1k-10k, "
            "61% -> 62% overall (each shift under 2 SE).",
            "- **C10, numbers only:** exchange rates on the new snapshot grid (every 40 chunks) move by 0.01, and 70M sgd "
            "2.68 -> 2.57; the conclusions are unchanged.",
            "- C1, C3, C4, C6, C7, C8, C9: no wording change (C6-C9 are the same runs, exact).", ""]
    out += ["## Per-token identity of the exact runs", "", "| arrays | identical share | max abs diff |", "|---|---|---|"]
    out += ["| %s | %.4f | %.2e |" % (r["what"], r["identical_share"], r["max_abs_diff"]) for r in ident]
    for claim in ["C%d" % i for i in range(1, 11)]:
        rs = [r for r in rows if r["claim"] == claim]
        out += ["", "## %s" % claim, "", "| quantity | old | new | diff | SE | kind | flag | note |", "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            fmt = "%.0f" if "halves" in r["quantity"] else "%.4f" if "relative update" not in r["quantity"] else "%.3g"
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["quantity"], fmt % r["old"], fmt % r["new"], ("%+.4f" % r["diff"]) if fmt != "%.3g" else "%+.3g" % r["diff"],
                "%.4f" % r["se"] if np.isfinite(r["se"]) else "", r["kind"], r["flag"], r["note"]))
    with open(os.path.join(HERE, "REGRESSION.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[:8]))
    for r in mism + flagged:
        print(r)


if __name__ == "__main__":
    main()
