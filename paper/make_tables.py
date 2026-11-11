"""Result tables and quoted numbers for the paper, generated from the saved results in ../experiments/
(no runs).

Gains are token-level means from each experiment's CSVs; standard errors are over books (paired per
book for differences), or over held-out chunks / books for forgetting. A dagger marks a value under 2 SE.

Writes paper/tables/*.tex (included by main.tex) and paper/numbers.tex: one macro per number quoted in
the text, rounded here (gains, locality and forgetting to 3 decimals with sign, SE to 3 decimals,
percentages to integers, exchange rates to 2 decimals, half-lives to 0.1k tokens).

Usage (from the repo root or from paper/):
    venv/bin/python paper/make_tables.py
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")
OUT = os.path.join(HERE, "tables")


def read(path):
    with open(os.path.join(EXP, path)) as f:
        return list(csv.DictReader(f))


def pm(mean, se=None):
    """$+0.301_{\\pm.011}$; a dagger when |mean| < 2 se."""
    m = float(mean)
    if se is None or se == "" or float(se) != float(se):
        return "$%+.3f$" % m
    s = float(se)
    dagger = "^{\\dagger}" if abs(m) < 2 * s else ""
    return "$%+.3f_{\\pm %s}%s$" % (m, ("%.3f" % s)[1:], dagger)


def tex_lr(v):
    m, e = ("%.0e" % float(v)).split("e")
    return "$%s10^{%d}$" % ("" if m == "1" else m + "\\!\\times\\!", int(e))


def write(name, header, rows, colspec):
    lines = ["\\begin{tabular}{%s}" % colspec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


BIN_TEX = {"all tokens": "all tokens", "same chunk": "same chunk", "earlier chunk, < 1k": "earlier, $<$ 1k",
           "earlier chunk, 1k-10k": "earlier, 1k-10k", "earlier chunk, 10k-100k": "earlier, 10k-100k",
           "earlier chunk, >= 100k": "earlier, $\\geq$ 100k", "never seen": "never seen"}
BINS = list(BIN_TEX)
E1 = {r["bin"]: r for r in read("e1_locality/distance.csv")}
E2 = {(r["subset"], r["bin"]): r for r in read("e2_where/results.csv")}
MVF = {(r["quantity"], r["stream"]): r for r in read("e2_where/mlp_vs_full.csv")}
E7 = {(r["size"], r["rule"], r["bin"]): r for r in read("e7_update_rules/results.csv")}
E7X = {(r["size"], r["rule"]): r for r in read("e7_update_rules/exchange.csv")}


def e1(b, key):
    return pm(E1[b][key], E1[b][key + "_se"])


def e2(subset, b):
    r = E2[subset, b]
    return pm(r["gain"], r["gain_se"])


# ---------------------------------------------------------------- tables

def distance():
    """E1: gain by distance, in order and shuffled, locality, unigram cache (C1, C2, C4). Where the unigram
    cache has no count for a token, mixing it in costs exactly -log(0.95) = 0.051 nats, so those bins have a
    constant loss and an SE of 0."""
    rows = [[BIN_TEX[b], "{:,}".format(int(E1[b]["tokens"])), "%.3f" % float(E1[b]["frozen"]),
             e1(b, "gain_in_order"), e1(b, "gain_shuffled"), e1(b, "locality"), e1(b, "gain_unigram")] for b in BINS]
    write("distance", ["bin (previous occurrence)", "tokens", "frozen loss", "in order", "shuffled", "locality gain",
                       "unigram cache"], rows, "lrrrrrr")


def samebook():
    """E1: same book against earlier book at the same distance (C3)."""
    rows = []
    for b, split in (("earlier chunk, < 1k", "same book"), ("earlier chunk, 1k-10k", "same book"),
                     ("earlier chunk, 1k-10k", "earlier book"), ("earlier chunk, 10k-100k", "same book"),
                     ("earlier chunk, 10k-100k", "earlier book")):
        name = "%s, %s" % (b, split)
        rows.append([BIN_TEX[b], split, "{:,}".format(int(E1[name]["tokens"])), e1(name, "gain_in_order"),
                     e1(name, "gain_shuffled"), e1(name, "locality"), e1(name, "gain_unigram")])
    write("samebook", ["bin", "previous occurrence", "tokens", "in order", "shuffled", "locality gain", "unigram cache"],
          rows, "llrrrrr")


def layers():
    """E2: every parameter subset, each at its own rate, in order (C5)."""
    subsets = [("all", "all parameters"), ("output", "output embedding only"), ("no_output", "all but output embedding"),
               ("mlp", "MLPs only"), ("attention", "attention only")] + \
              [("mlp@%d" % i, "MLP of layer %d only" % (i + 1)) for i in range(6)]
    bins = ["all tokens", "earlier chunk, < 1k", "earlier chunk, 1k-10k, same book", "never seen"]
    rows = [[label, "{:,}".format(int(E2[s, "all tokens"]["params"])), tex_lr(E2[s, "all tokens"]["lr"])] +
            [e2(s, b) for b in bins] for s, label in subsets]
    write("layers", ["parameters updated", "parameters", "lr", "all tokens", "earlier, $<$ 1k",
                     "1k-10k, same book", "never seen"], rows, "lrrrrrr")


MVF_ORDER = ["gain shuffled", "locality", "WikiText forgetting, last half", "held-out books forgetting, last half",
             "stream gain / WikiText forgetting (last half)"]


def mlp_vs_full():
    """E2: MLPs only (its own val-pick rate) against all parameters: gain shuffled, locality, forgetting
    (last half), exchange rate (C5). The in-order gains are in the layers table."""
    rows_in = sorted([r for r in read("e2_where/mlp_vs_full.csv") if r["quantity"] in MVF_ORDER],
                     key=lambda r: MVF_ORDER.index(r["quantity"]))
    bin_tex = {"all tokens": "all tokens", "earlier chunk, < 1k": "earlier, $<$ 1k",
               "earlier chunk, 1k-10k, same book": "1k-10k, same book", "never seen": "never seen",
               "K = 1": "$K=1$", "shuffled": "shuffled"}
    names = {"locality": "locality gain", "WikiText forgetting, last half": "WikiText forgetting", "held-out books forgetting, last half":
             "held-out books forgetting", "stream gain / WikiText forgetting (last half)": "exchange rate"}
    rows, last = [], None
    for r in rows_in:
        label = names.get(r["quantity"], r["quantity"]) if r["quantity"] != last else ""
        last = r["quantity"]
        if r["section"] == "exchange rate":
            rows.append([label, bin_tex[r["stream"]], r["mlp"], r["full"], "$%+.2f$" % float(r["mlp_minus_full"])])
        else:
            rows.append([label, bin_tex[r["stream"]], pm(r["mlp"], r["mlp_se"]), pm(r["full"], r["full_se"]),
                         pm(r["mlp_minus_full"], r["mlp_minus_full_se"])])
    write("mlp_vs_full", ["quantity", "bin / stream", "MLPs only, lr %s" % tex_lr(rows_in[0]["lr_mlp"]),
                          "all, lr %s" % tex_lr(rows_in[0]["lr_full"]), "MLPs only $-$ all"], rows, "llrrr")


def verylong():
    """E6: 200k-token books, 14M and 31M online against frozen 14M / 31M / 70M (C9)."""
    res = {r["run"]: r for r in read("e6_long_books/results.csv")}
    rows = []
    for key, label in (("frozen14m", "frozen 14M"), ("online14m", "14M dyn.\\ eval., lr $10^{-4}$"), ("frozen31m", "frozen 31M"),
                       ("online31m", "31M dyn.\\ eval., lr $3\\!\\times\\!10^{-4}$"), ("frozen70m", "frozen 70M")):
        r = res[key]
        rows.append([label, "%.3f" % float(r["loss"]), pm(r["gain_over_frozen14m"], r["se"]) if key != "frozen14m" else "",
                     "%.0f\\%%" % (100 * float(r["gap31_closed"])) if key != "frozen14m" else ""])
    write("verylong", ["run", "loss", "gain over frozen 14M", "share of 14M-31M gap"], rows, "lrrr")


def forgetting_table():
    """E5: stream gain, last-half forgetting on detokenized WikiText and held-out books, exchange rate (C8)."""
    rows = []
    for st, label in (("main_K1", "$K=1$"), ("main_shuffled", "shuffled"), ("verylong", "very long (4 $\\times$ 200k)")):
        r = next(x for x in read("e5_forgetting/summary.csv") if x["stream"] == st)
        rows.append([label, pm(r["stream_gain"], r["stream_gain_se"]), pm(r["wiki_last_half"], r["wiki_last_half_se"]),
                     pm(r["books_last_half"], r["books_last_half_se"]), r["exchange_rate"]])
    write("forgetting", ["stream", "stream gain", "WikiText forgetting", "held-out books forgetting",
                         "exchange rate"], rows, "lrrrr")


def rules():
    """E7: per size and update rule, picked rate, gain, locality, WikiText forgetting, exchange rates (C10)."""
    names = {"sgd": "plain SGD", "adam": "Adam", "nsgd": "normalized SGD"}
    rows = []
    for size in ("14m", "31m", "70m"):
        for rule in ("sgd", "adam", "nsgd"):
            a, n, e = E7[size, rule, "all tokens"], E7[size, rule, "earlier chunk < 1k"], E7X[size, rule]
            rows.append([size.upper() if rule == "sgd" else "", names[rule], tex_lr(e["lr"]),
                         pm(a["gain_K1"], a["gain_K1_se"]), pm(n["locality"], n["locality_se"]),
                         pm(a["locality"], a["locality_se"]), pm(e["wiki_forgetting_K1"], e["wiki_forgetting_K1_se"]),
                         e["exchange_rate_K1"], e["exchange_rate_shuffled"]])
    write("rules", ["size", "rule", "lr", "gain, $K=1$", "loc. gain, $<$ 1k", "loc. gain, all", "WikiText forgetting",
                    "exch. rate, $K=1$", "shuffled"], rows, "llrrrrrrr")


# ---------------------------------------------------------------- numbers.tex

MACROS = []


def m(name, text):
    assert name.isalpha(), name
    assert not name.endswith("PM") or text.startswith("\\ensuremath"), name
    assert name not in [n for n, _ in MACROS], name
    MACROS.append((name, text))


def g(name, value, se=None):
    """Signed 3-decimal value; with an SE also name + "SE" (the SE) and name + "PM" (value +- SE, with a
    dagger when under 2 SE)."""
    v = float(value)
    m(name, "\\ensuremath{%+.3f}" % v)
    if se is not None:
        s_ = float(se)
        m(name + "SE", "\\ensuremath{%.3f}" % s_)
        m(name + "PM", "\\ensuremath{%+.3f \\pm %.3f%s}" % (v, s_, "^{\\dagger}" if abs(v) < 2 * s_ else ""))


def pct(name, x):
    m(name, "%d\\%%" % round(100 * float(x)))


def rate(name, x):
    m(name, "%.2f" % float(x))


def sci(name, x):
    mant, e = ("%.1e" % float(x)).split("e")
    m(name, "\\ensuremath{%s\\times10^{%d}}" % (mant, int(e)))


def kilo(name, tokens):
    m(name, "%.1fk" % (float(tokens) / 1000))


def numbers():
    # ---- E1 (C1-C4)
    for key, b in (("All", "all tokens"), ("SameChunk", "same chunk"), ("OneK", "earlier chunk, < 1k"),
                   ("TenK", "earlier chunk, 1k-10k"), ("HundredK", "earlier chunk, 10k-100k"),
                   ("Far", "earlier chunk, >= 100k"), ("Never", "never seen")):
        r = E1[b]
        g("gainInOrder" + key, r["gain_in_order"], r["gain_in_order_se"])
        g("gainShuffled" + key, r["gain_shuffled"], r["gain_shuffled_se"])
        g("locality" + key, r["locality"], r["locality_se"])
        g("gainUnigram" + key, r["gain_unigram"], r["gain_unigram_se"])
        g("inOrderMinusUnigram" + key, r["in_order_minus_unigram"], r["in_order_minus_unigram_se"])
        pct("shuffledShare" + key, float(r["gain_shuffled"]) / float(r["gain_in_order"])) \
            if key in ("All", "OneK", "TenK") else None
    m("frozenLossMain", "%.3f" % float(E1["all tokens"]["frozen"]))
    m("tokensOneKEarlierBook", "{:,}".format(int(E1["earlier chunk, < 1k, earlier book"]["tokens"])).replace(",", "{,}"))
    for key, b in (("TenKSame", "earlier chunk, 1k-10k, same book"), ("TenKEarlier", "earlier chunk, 1k-10k, earlier book"),
                   ("HundredKSame", "earlier chunk, 10k-100k, same book"),
                   ("HundredKEarlier", "earlier chunk, 10k-100k, earlier book")):
        r = E1[b]
        g("gainInOrder" + key, r["gain_in_order"], r["gain_in_order_se"])
        g("locality" + key, r["locality"], r["locality_se"])
        g("gainUnigram" + key, r["gain_unigram"], r["gain_unigram_se"])
    pos = read("e1_locality/position.csv")
    g("posGainFirst", pos[0]["gain_in_order"], pos[0]["gain_in_order_se"])
    g("posGainLast", pos[-1]["gain_in_order"], pos[-1]["gain_in_order_se"])
    sh = [float(r["gain_shuffled"]) for r in pos]
    g("posShuffledMin", min(sh))
    g("posShuffledMax", max(sh))

    # ---- E2 (C5)
    for key, s in (("Output", "output"), ("NoOutput", "no_output"), ("Attention", "attention"), ("Mlp", "mlp")):
        for bk, b in (("All", "all tokens"), ("OneK", "earlier chunk, < 1k"), ("TenKSame", "earlier chunk, 1k-10k, same book"),
                      ("Never", "never seen")):
            r = E2[s, b]
            g("gain" + key + bk, r["gain"], r["gain_se"])
            g("minusFull" + key + bk, r["minus_full"], r["minus_full_se"])
    g("gainAllTenKSame", E2["all", "earlier chunk, 1k-10k, same book"]["gain"],
      E2["all", "earlier chunk, 1k-10k, same book"]["gain_se"])
    one = {i: E2["mlp@%d" % i, "earlier chunk, < 1k"] for i in range(6)}
    ten = {i: E2["mlp@%d" % i, "earlier chunk, 1k-10k, same book"] for i in range(6)}
    shares1 = [float(one[i]["share_of_full"]) for i in range(6)]
    shares10 = [float(ten[i]["share_of_full"]) for i in range(6)]
    pct("layerShareOneKMin", min(shares1))
    pct("layerShareOneKMax", max(shares1))
    pct("layerShareTenKMin", min(shares10))
    pct("layerShareTenKMax", max(shares10))
    best = max(range(6), key=lambda i: float(one[i]["gain"]))
    m("layerBest", str(best + 1))
    g("layerBestOneK", one[best]["gain"], one[best]["gain_se"])
    g("layerBestMinusFull", one[best]["minus_full"], one[best]["minus_full_se"])
    g("layerLastOneK", one[5]["gain"], one[5]["gain_se"])
    g("layerLastTenK", ten[5]["gain"], ten[5]["gain_se"])
    for key, q, st in (("ShuffledAll", "gain shuffled", "all tokens"), ("LocAll", "locality", "all tokens"),
                       ("LocOneK", "locality", "earlier chunk, < 1k"), ("LocTenK", "locality", "earlier chunk, 1k-10k, same book"),
                       ("Wiki", "WikiText forgetting, last half", "K = 1"), ("WikiSh", "WikiText forgetting, last half", "shuffled"),
                       ("Books", "held-out books forgetting, last half", "K = 1"),
                       ("BooksSh", "held-out books forgetting, last half", "shuffled")):
        r = MVF[q, st]
        g("mvfMlp" + key, r["mlp"], r["mlp_se"])
        g("mvfFull" + key, r["full"], r["full_se"])
        g("mvfDiff" + key, r["mlp_minus_full"], r["mlp_minus_full_se"])
    for key, st in (("", "K = 1"), ("Sh", "shuffled")):
        r = MVF["stream gain / WikiText forgetting (last half)", st]
        rate("mvfMlpExchange" + key, r["mlp"])
        rate("mvfFullExchange" + key, r["full"])
    pk = {(r["size"], r["rule"], r["params"], r["lr"]): r for r in read("e0_lr_picks/picks.csv")}
    m("valPickMlp", pk["14m", "sgd", "mlp", "0.0003"]["val_pick_loss"])
    m("valPickAll", pk["14m", "sgd", "all", "0.0001"]["val_pick_loss"])
    m("valPickAllHigh", pk["14m", "sgd", "all", "0.0003"]["val_pick_loss"])
    diag = {(r["size"], r["type"]): r for r in read("e7_update_rules/diagnostics_summary.csv") if r["layer"] == "all layers"}
    for size, key in (("14m", "Fourteen"), ("31m", "ThirtyOne"), ("70m", "Seventy")):
        pct("shareAttn" + key, diag[size, "attention"]["share_of_update_sq"])
        pct("shareMlp" + key, diag[size, "mlp"]["share_of_update_sq"])
        sci("relMlp" + key, diag[size, "mlp"]["rel_update"])
        sci("relAttn" + key, diag[size, "attention"]["rel_update"])
    m("mlpStepRatio", "%d" % round(float(diag["14m", "mlp"]["rel_update"]) / float(diag["70m", "mlp"]["rel_update"])))

    # ---- E3 (C6)
    e3 = {r["stream"]: r for r in read("e3_locality_levels/results.csv")}
    for key, st in (("Long", "long"), ("KOne", "main_K1"), ("KFive", "main_K5"), ("KTwentyFive", "main_K25"),
                    ("Shuffled", "main_shuffled")):
        g("level" + key, e3[st]["gain"], e3[st]["gain_se"])
        if e3[st].get("drop_from_previous"):
            g("drop" + key, e3[st]["drop_from_previous"], e3[st]["drop_se"])

    # ---- E4 (C7)
    means = {r["I"]: r for r in read("e4_eviction/means.csv")}
    for key, i in (("Zero", "0"), ("TenK", "9984"), ("HundredK", "100096"), ("ThreeHundredK", "300032")):
        g("memory" + key, means[i]["memory_A"], means[i]["memory_A_se"])
    g("gainAZero", means["0"]["gain_A2"], means["0"]["gain_A2_se"])
    g("gainCZero", means["0"]["gain_C2"], means["0"]["gain_C2_se"])
    c2 = [float(r["gain_C2"]) for k, r in means.items() if k != "pre"]
    g("gainCMax", max(c2))
    g("gainCMin", min(c2))
    g("memoryRareIn", means["0"]["memory_rare_in_A1"], means["0"]["memory_rare_in_A1_se"])
    g("memoryRareNotIn", means["0"]["memory_rare_not_in_A1"], means["0"]["memory_rare_not_in_A1_se"])
    halv = {r["quantity"]: r for r in read("e4_eviction/halving.csv")}
    kilo("evictHalfLife", halv["memory_A"]["halves_at_tokens"])
    kilo("evictHalfLifeRare", halv["memory_rare_in_A1"]["halves_at_tokens"])
    kilo("evictHalfLifeGainA", halv["gain_A2"]["halves_at_tokens"])
    fits = {r["model"]: r for r in read("e4_eviction/fits.csv")}
    m("fitChiPower", "%.1f" % float(fits["power law"]["chi2"]))
    m("fitChiExp", "%.1f" % float(fits["exponential + floor"]["chi2"]))
    m("fitPowerExponent", "%.2f" % float(fits["power law"]["params"].split(" b ")[1]))

    # ---- E5 (C8)
    summ = {r["stream"]: r for r in read("e5_forgetting/summary.csv")}
    for key, st in (("KOne", "main_K1"), ("Shuffled", "main_shuffled"), ("VeryLong", "verylong")):
        r = summ[st]
        g("streamGain" + key, r["stream_gain"], r["stream_gain_se"])
        g("forgetWiki" + key, r["wiki_last_half"], r["wiki_last_half_se"])
        g("forgetWikiFinal" + key, r["wiki_final"], r["wiki_final_se"])
        g("forgetBooks" + key, r["books_last_half"], r["books_last_half_se"])
        rate("exchange" + key, r["exchange_rate"])
        m("wikiOverGain" + key, "%.1f" % float(r["wiki_over_stream_gain"]))
    m("frozenWiki", "%.3f" % float(summ["main_K1"]["frozen_wiki"]))
    m("frozenHeldOutBooks", "%.3f" % float(summ["main_K1"]["frozen_books"]))
    k1s = read("e5_forgetting/k1_vs_shuffled.csv")[0]
    g("shMinusKOneWiki", k1s["shuffled_minus_K1_wiki"], k1s["shuffled_minus_K1_wiki_se"])
    g("shMinusKOneBooks", k1s["shuffled_minus_K1_books"], k1s["shuffled_minus_K1_books_se"])
    curve = [r for r in read("e5_forgetting/curve.csv") if r["stream"] == "verylong"]
    for j, key in enumerate(("One", "Two", "Three", "Four")):  # mean WikiText forgetting of the snapshots in each book
        w = [float(r["wiki"]) for r in curve if j * 781 < int(r["chunks_read"]) <= (j + 1) * 781]
        g("veryLongWikiBook" + key, sum(w) / len(w))

    # ---- E7 (C10)
    sizes = (("14m", "Fourteen"), ("31m", "ThirtyOne"), ("70m", "Seventy"))
    rules_ = (("sgd", "Sgd"), ("adam", "Adam"), ("nsgd", "Nsgd"))
    for size, sk in sizes:
        for rule, rk in rules_:
            a, n, e = E7[size, rule, "all tokens"], E7[size, rule, "earlier chunk < 1k"], E7X[size, rule]
            g("rGain" + rk + sk, a["gain_K1"], a["gain_K1_se"])
            g("rGainSh" + rk + sk, a["gain_shuffled"], a["gain_shuffled_se"])
            g("rLocAll" + rk + sk, a["locality"], a["locality_se"])
            g("rLocOneK" + rk + sk, n["locality"], n["locality_se"])
            g("rWiki" + rk + sk, e["wiki_forgetting_K1"], e["wiki_forgetting_K1_se"])
            rate("rExchange" + rk + sk, e["exchange_rate_K1"])
            rate("rExchangeSh" + rk + sk, e["exchange_rate_shuffled"])
    r = E7["70m", "nsgd", "all tokens"]
    g("nsgdMinusSgdGainSeventy", r["gain_K1_minus_sgd"], r["gain_K1_minus_sgd_se"])
    for key, b in (("All", "all tokens"), ("OneK", "earlier chunk < 1k")):  # 31M - 14M plain SGD, unpaired SE
        a, c = E7["31m", "sgd", b], E7["14m", "sgd", b]
        g("locDiffThirtyOneFourteen" + key, float(a["locality"]) - float(c["locality"]),
          math.sqrt(float(a["locality_se"]) ** 2 + float(c["locality_se"]) ** 2))

    # ---- E6 (C9)
    res = {r["run"]: r for r in read("e6_long_books/results.csv")}
    pct("gapClosedFourteen", res["online14m"]["gap31_closed"])
    g("vlGainOnlineFourteen", res["online14m"]["gain_over_frozen14m"], res["online14m"]["se"])
    prs = {r["comparison"]: r for r in read("e6_long_books/pairs.csv")}
    g("vlOnlineMinusFrozenThirtyOne", prs["31M online - frozen 31M"]["mean"], prs["31M online - frozen 31M"]["se"])
    g("vlFrozenThirtyOneAhead", prs["frozen 31M - 14M online"]["mean"], prs["frozen 31M - 14M online"]["se"])
    g("vlOnlineThirtyOneAhead", prs["31M online - 14M online"]["mean"], prs["31M online - 14M online"]["se"])
    d = prs["31M online - frozen 70M"]
    g("vlFrozenSeventyAhead", -float(d["mean"]), d["se"])
    gap = [(r["position"], float(r["gap31_closed_by_online14m"])) for r in read("e6_long_books/position.csv")]
    pct("gapFirstBin", gap[0][1])
    peak = max(gap, key=lambda x: x[1])
    pct("gapPeak", peak[1])
    m("gapPeakBin", peak[0].replace("k-", "k and ").replace("k", "k"))
    pct("gapLastBin", gap[-1][1])

    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        f.write("% Generated by make_tables.py from the CSVs in ../experiments/. Do not edit.\n")
        for name, text in MACROS:
            f.write("\\newcommand{\\%s}{%s}\n" % (name, text))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for old in ("baselines.tex",):  # merged into distance.tex
        if os.path.exists(os.path.join(OUT, old)):
            os.remove(os.path.join(OUT, old))
    distance()
    samebook()
    layers()
    mlp_vs_full()
    verylong()
    forgetting_table()
    rules()
    numbers()
    print(sorted(os.listdir(OUT)), len(MACROS), "macros")
