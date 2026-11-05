"""Result tables for the paper, generated from the saved results in ../experiments_LLM2_cache/ (no runs).

Gains are token-level means (from each test's CSVs); standard errors are over books (test 10 part A:
errorbars.csv, errorbars_distance.csv, and parts B and C), paired per-book differences for comparisons.
A dagger marks a difference under 2 SE.

Usage (from the repo root or from paper/):
    venv/bin/python paper/make_tables.py
Writes paper/tables/*.tex, included by main.tex.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments_LLM2_cache")
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


def write(name, header, rows, colspec):
    lines = ["\\begin{tabular}{%s}" % colspec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


D = {(r["comparison"], r["bin"]): r for r in read("test10_followups/errorbars_distance.csv")}
EB = {(r["test"], r["stream"], r["comparison"], r["bin"]): r for r in read("test10_followups/errorbars.csv")}


def d(comp, b):
    r = D[comp, b]
    return pm(r["token_mean"], r["se"])


def eb(mean, test, stream, comp, b="all"):
    r = EB.get((test, stream, comp, b))
    return pm(mean, r["se"] if r else None)


BIN_TEX = {"all tokens": "all tokens", "same chunk": "same chunk", "earlier chunk, < 1k": "earlier, $<$ 1k",
           "earlier chunk, 1k-10k": "earlier, 1k--10k", "earlier chunk, 10k-100k": "earlier, 10k--100k",
           "earlier chunk, >= 100k": "earlier, $\\geq$ 100k", "never seen": "never seen"}
BINS = list(BIN_TEX)


def distance():
    """Test 1 and 4: gain by distance, in order and shuffled, locality."""
    t1 = {r["bin"]: r for r in read("test1_distance/results.csv")}
    rows = [[BIN_TEX[b], t1[b]["tokens"], t1[b]["frozen"], d("online - frozen", b), d("shuffled - frozen", b),
             d("locality 14M", b)] for b in BINS]
    write("distance", ["bin (previous occurrence)", "tokens", "frozen loss", "online, in order", "online, shuffled",
                       "locality"], rows, "lrrrrr")


def samebook():
    """Test 5 part A."""
    rows = []
    for b, split in (("earlier chunk, < 1k", "same book"), ("earlier chunk, 1k-10k", "same book"),
                     ("earlier chunk, 1k-10k", "earlier book"), ("earlier chunk, 10k-100k", "same book"),
                     ("earlier chunk, 10k-100k", "earlier book")):
        name = "%s, %s" % (b, split)
        rows.append([BIN_TEX[b], split, D["online - frozen", name]["tokens"], d("online - frozen", name),
                     d("shuffled - frozen", name), d("locality 14M", name), d("unigram - frozen", name)])
    write("samebook", ["bin", "previous occurrence", "tokens", "in order", "shuffled", "locality", "unigram cache"],
          rows, "llrrrrr")


def baselines():
    """Tests 3 and 4: unigram cache, online + cache, output only, all but output."""
    rows = [[BIN_TEX[b]] + [d("%s - frozen" % k, b) for k in ("online", "unigram", "online_cache", "output_lr1e-3",
                                                             "output_lr3e-3", "no_output")] for b in BINS]
    write("baselines", ["bin", "online, all", "unigram", "online + cache", "output, lr $10^{-3}$",
                        "output, lr $3\\!\\times\\!10^{-3}$", "all but output"], rows, "lrrrrrr")


def layers():
    """Tests 4 and 6: MLP only, fewer layers, single layers."""
    pick = [("all tokens", "all tokens"), ("same chunk", "same chunk"), ("earlier, $<$ 1k", "earlier chunk, < 1k"),
            ("1k--10k, same book", "earlier chunk, 1k-10k, same book"),
            ("10k--100k, same book", "earlier chunk, 10k-100k, same book"), ("never seen", "never seen")]
    comps = ["online - frozen", "mlp - frozen", "mlp_last2 - frozen"] + ["mlp_l%d - frozen" % i for i in range(1, 7)]
    rows = [[label] + [d(c, b) for c in comps] for label, b in pick]
    lr = json.loads(open(os.path.join(EXP, "test6", "runtime.txt")).readline().split(" ", 2)[2])

    def tex_lr(v):
        m, e = ("%.0e" % v).split("e")
        return "$%s\\!\\times\\!10^{%d}$" % (m, int(e)) if m != "1" else "$10^{%d}$" % int(e)
    header = ["bin", "all", "MLP 1--6", "MLP 5--6"] + ["L%d" % i for i in range(1, 7)]
    lrs = ["lr", tex_lr(1e-4), tex_lr(1e-4), tex_lr(lr["mlp_last2"])] + [tex_lr(lr["mlp_l%d" % i]) for i in range(1, 7)]
    write("layers", header, [lrs] + rows, "l" + "r" * 9)


def aggressive():
    """Test 7: settings across streams; K = 1 bins."""
    res = {(r["setting"], r["stream"]): r for r in read("test7_aggressive/results.csv")}
    bins = {(r["setting"], r["stream"]): r for r in read("test7_aggressive/bins.csv")}
    rows = []
    for s, label in (("full_1e-4", "all parameters, lr $10^{-4}$"), ("mlp4_3e-3", "MLP 4, lr $3\\!\\times\\!10^{-3}$"),
                     ("mlp4_1e-2", "MLP 4, lr $10^{-2}$"), ("fast4_d0.99", "MLP 4, decay 0.99"),
                     ("fast4_d0.9", "MLP 4, decay 0.9")):
        row = [label] + [eb(res[s, st]["gain"], "7", st, "%s - frozen" % s) for st in ("long", "K1", "K5", "K25", "shuffled")]
        row += [eb(bins[s, "K1"]["gain_same_book_lt10k"], "7", "K1", "%s - frozen" % s, "same book < 10k"),
                eb(bins[s, "K1"]["gain_never_seen"], "7", "K1", "%s - frozen" % s, "never seen")]
        rows.append(row)
    write("aggressive", ["setting", "long", "$K=1$", "$K=5$", "$K=25$", "shuffled", "$K=1$: same book $<$10k",
                         "$K=1$: never seen"], rows, "lrrrrrrr")


def dual():
    """Tests 8, 9 and 10B: gains over full_1e-4 on the test streams."""
    pick8 = json.loads(open(os.path.join(EXP, "test8_dual", "runtime.txt")).readline())["pick_val_K1"]
    r8 = {(r["config"], r["stream"]): r for r in read("test8_dual/results.csv") if r["phase"] == "test"}
    r9 = {(r["config"], r["stream"]): r for r in read("test9_metasgd_lite/results.csv")}
    b8 = {(r["config"], r["stream"]): r for r in read("test8_dual/bins.csv")}
    b9 = {(r["config"], r["stream"]): r for r in read("test9_metasgd_lite/bins.csv")}
    rb = {(r["stream"], r["comparison"]): r for r in read("test10_followups/partB_test.csv")}
    rows = [["all parameters, lr $10^{-4}$ (gain over frozen)"] +
            [eb(r8["full_1e-4", s]["gain_frozen"], "8", s, "full_1e-4 - frozen") for s in ("K1", "shuffled", "long")] +
            [eb(r8["full_1e-4", "locality (K1 - shuffled)"]["gain_frozen"], "8", "K1 vs shuffled", "full_1e-4 locality"),
             eb(b8["full_1e-4", "K1"]["gain_same_book_lt10k"], "8", "K1", "full_1e-4 - frozen", "same book < 10k")]]
    rows.append(["+ fast part, MLP 4 (test 8)"] +
                [eb(r8[pick8, s]["gain_full"], "8", s, "%s - full_1e-4" % pick8) for s in ("K1", "shuffled", "long")] +
                [eb(r8[pick8, "locality (K1 - shuffled)"]["gain_frozen"], "8", "K1 vs shuffled", "%s locality" % pick8),
                 eb(b8[pick8, "K1"]["vs_full_same_book_lt10k"], "8", "K1", "%s - full_1e-4" % pick8, "same book < 10k")])
    rows.append(["+ fast part, MLP 2 (test 10)"] +
                [pm(rb[s, "vs full_1e-4"]["token_mean"], rb[s, "vs full_1e-4"]["se"]) for s in ("K1", "shuffled", "long")] +
                ["", ""])  # part B saved no token-level bin means or paired locality
    rows.append(["Meta-SGD lite (test 9)"] +
                [eb(r9["metasgd_lite", s]["gain_full"], "9", s, "metasgd_lite - full_1e-4") for s in ("K1", "shuffled", "long")] +
                [eb(r9["metasgd_lite", "locality (K1 - shuffled)"]["gain_frozen"], "9", "K1 vs shuffled", "metasgd_lite locality"),
                 eb(b9["metasgd_lite", "K1"]["vs_full_same_book_lt10k"], "9", "K1", "metasgd_lite - full_1e-4", "same book < 10k")])
    write("dual", ["run", "$K=1$", "shuffled", "long", "locality", "$K=1$: same book $<$10k"], rows, "lrrrrr")


def size():
    """Test 6 (sizes, locality) and test 10C (200k-token books)."""
    rows = []
    for label, b in (("all tokens", "all tokens"), ("earlier, $<$ 1k", "earlier chunk, < 1k"),
                     ("1k--10k, same book", "earlier chunk, 1k-10k, same book"), ("never seen", "never seen")):
        rows.append([label, d("online - frozen", b), d("locality 14M", b), d("31M online - frozen 31M", b),
                     d("locality 31M", b), d("70M online - frozen 70M", b), d("locality 70M", b)])
    write("size", ["bin", "14M gain", "14M locality", "31M gain", "31M locality", "70M gain", "70M locality"], rows,
          "lrrrrrr")
    c = {r["run"]: r for r in read("test10_followups/partC.csv")}
    dual_key = next(k for k in c if k.startswith("14m dual"))
    rows = []
    for key, label in (("frozen14m", "frozen 14M"), ("14m full_1e-4", "14M online, lr $10^{-4}$"),
                       (dual_key, "14M online + fast part (MLP 2)"), ("frozen31m", "frozen 31M"),
                       ("31m online lr3e-4", "31M online, lr $3\\!\\times\\!10^{-4}$"), ("frozen70m", "frozen 70M")):
        r = c[key]
        rows.append([label, r["loss"], pm(r["gain_over_frozen14m"], r["se"]) if key != "frozen14m" else "",
                     "%.0f\\%%" % (100 * float(r["gap31_closed"])) if key != "frozen14m" else "", r["seconds"]])
    write("verylong", ["run", "loss", "gain over frozen 14M", "share of 14M--31M gap", "seconds"], rows, "lrrrr")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    distance()
    samebook()
    baselines()
    layers()
    aggressive()
    dual()
    size()
    print(sorted(os.listdir(OUT)))
