"""Result tables for the paper, generated from the saved results in ../experiments/ (no runs).

Gains are token-level means from each experiment's CSVs; standard errors are over books (paired per
book for differences). A dagger marks a value under 2 SE.

Usage (from the repo root or from paper/):
    venv/bin/python paper/make_tables.py
Writes paper/tables/*.tex, included by main.tex.
"""
import csv
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
           "earlier chunk, 1k-10k": "earlier, 1k--10k", "earlier chunk, 10k-100k": "earlier, 10k--100k",
           "earlier chunk, >= 100k": "earlier, $\\geq$ 100k", "never seen": "never seen"}
BINS = list(BIN_TEX)
E1 = {r["bin"]: r for r in read("e1_locality/distance.csv")}
E2 = {(r["subset"], r["bin"]): r for r in read("e2_where/results.csv")}


def e1(b, key):
    return pm(E1[b][key], E1[b][key + "_se"])


def e2(subset, b):
    r = E2[subset, b]
    return pm(r["gain"], r["gain_se"])


def distance():
    """E1: gain by distance, in order and shuffled, locality (C1, C2)."""
    rows = [[BIN_TEX[b], E1[b]["tokens"], E1[b]["frozen"], e1(b, "gain_in_order"), e1(b, "gain_shuffled"),
             e1(b, "locality")] for b in BINS]
    write("distance", ["bin (previous occurrence)", "tokens", "frozen loss", "online, in order", "online, shuffled",
                       "locality"], rows, "lrrrrr")


def samebook():
    """E1: same book against earlier book at the same distance (C3)."""
    rows = []
    for b, split in (("earlier chunk, < 1k", "same book"), ("earlier chunk, 1k-10k", "same book"),
                     ("earlier chunk, 1k-10k", "earlier book"), ("earlier chunk, 10k-100k", "same book"),
                     ("earlier chunk, 10k-100k", "earlier book")):
        name = "%s, %s" % (b, split)
        rows.append([BIN_TEX[b], split, E1[name]["tokens"], e1(name, "gain_in_order"), e1(name, "gain_shuffled"),
                     e1(name, "locality"), e1(name, "gain_unigram")])
    write("samebook", ["bin", "previous occurrence", "tokens", "in order", "shuffled", "locality", "unigram cache"],
          rows, "llrrrrr")


def baselines():
    """E1 (unigram cache) and E2 (output only, all but output) (C4, C5)."""
    lr_out = tex_lr(E2["output", "all tokens"]["lr"])
    rows = [[BIN_TEX[b], e1(b, "gain_in_order"), e1(b, "gain_unigram"), e2("output", b), e2("no_output", b)]
            for b in BINS]
    write("baselines", ["bin", "online, all", "unigram cache", "output only, lr %s" % lr_out, "all but output"],
          rows, "lrrrr")


def layers():
    """E2: all parameters, MLPs only, attention only, one layer's MLP; each at its own rate (C5)."""
    pick = [("all tokens", "all tokens"), ("same chunk", "same chunk"), ("earlier, $<$ 1k", "earlier chunk, < 1k"),
            ("1k--10k, same book", "earlier chunk, 1k-10k, same book"), ("never seen", "never seen")]
    subsets = ["all", "mlp", "attention"] + ["mlp@%d" % i for i in range(6)]
    header = ["bin", "all", "MLP 1--6", "attention"] + ["L%d" % i for i in range(1, 7)]
    lrs = ["lr"] + [tex_lr(E2[s, "all tokens"]["lr"]) for s in subsets]
    rows = [[label] + [e2(s, b) for s in subsets] for label, b in pick]
    write("layers", header, [lrs] + rows, "l" + "r" * len(subsets))


def verylong():
    """E6: 200k-token books, 14M and 31M online against frozen 14M / 31M / 70M (C9)."""
    res = {r["run"]: r for r in read("e6_long_books/results.csv")}
    secs = dict(line.rsplit(" ", 2)[:2] for line in open(os.path.join(EXP, "e6_long_books", "runtime.txt")))
    rows = []
    for key, label in (("frozen14m", "frozen 14M"), ("online14m", "14M online, lr $10^{-4}$"), ("frozen31m", "frozen 31M"),
                       ("online31m", "31M online, lr $3\\!\\times\\!10^{-4}$"), ("frozen70m", "frozen 70M")):
        r = res[key]
        rows.append([label, "%.3f" % float(r["loss"]), pm(r["gain_over_frozen14m"], r["se"]) if key != "frozen14m" else "",
                     "%.0f\\%%" % (100 * float(r["gap31_closed"])) if key != "frozen14m" else "", secs[r["key"]]])
    write("verylong", ["run", "loss", "gain over frozen 14M", "share of 14M--31M gap", "seconds"], rows, "lrrrr")


def forgetting():
    """E5: stream gain, last-half forgetting on detokenized WikiText and held-out books, exchange rate (C8)."""
    rows = []
    for st, label in (("main_K1", "$K=1$"), ("main_shuffled", "shuffled"), ("verylong", "very long (4 $\\times$ 200k)")):
        r = next(x for x in read("e5_forgetting/summary.csv") if x["stream"] == st)
        rows.append([label, pm(r["stream_gain"], r["stream_gain_se"]), pm(r["wiki_last_half"], r["wiki_last_half_se"]),
                     pm(r["books_last_half"], r["books_last_half_se"]), r["exchange_rate"]])
    write("forgetting", ["stream (14M, all, lr $10^{-4}$)", "stream gain", "WikiText", "held-out books",
                         "gain / forgetting"], rows, "lrrrr")


def rules():
    """E7: per size and update rule, picked rate, gain, locality, WikiText forgetting, exchange rates (C10)."""
    res = read("e7_update_rules/results.csv")
    ex = {(r["size"], r["rule"]): r for r in read("e7_update_rules/exchange.csv")}
    names = {"sgd": "plain SGD", "adam": "Adam", "nsgd": "normalized SGD"}
    rows = []
    for size in ("14m", "31m", "70m"):
        for rule in ("sgd", "adam", "nsgd"):
            a = next(r for r in res if r["size"] == size and r["rule"] == rule and r["bin"] == "all tokens")
            n = next(r for r in res if r["size"] == size and r["rule"] == rule and r["bin"] == "earlier chunk < 1k")
            e = ex[size, rule]
            rows.append([size.upper() if rule == "sgd" else "", names[rule], tex_lr(e["lr"]),
                         pm(a["gain_K1"], a["gain_K1_se"]), pm(n["locality"], n["locality_se"]),
                         pm(a["locality"], a["locality_se"]), pm(e["wiki_forgetting_K1"], e["wiki_forgetting_K1_se"]),
                         e["exchange_rate_K1"], e["exchange_rate_shuffled"]])
    write("rules", ["size", "rule", "lr", "gain, $K=1$", "locality $<$1k", "locality, all", "WikiText forgetting",
                    "gain / forg., $K=1$", "shuffled"], rows, "llrrrrrrr")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    distance()
    samebook()
    baselines()
    layers()
    verylong()
    forgetting()
    rules()
    print(sorted(os.listdir(OUT)))
