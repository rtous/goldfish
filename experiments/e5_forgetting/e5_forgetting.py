"""E5 (claim C8): what does online learning cost in general knowledge?

Runs (registry, snapshots on): Pythia-14M plain SGD on all parameters at its val-pick rate, on main
K = 1, main shuffled and very long (4 books x 200k). Every 40 chunks a snapshot of the weights is
scored frozen on held-out text never trained on: detokenized WikiText-103 (outside the domain) and the
6 held-out PG-19 validation books (inside the domain, unseen books).
    forgetting(t) = loss_snapshot(held-out) - loss_pretrained(held-out)   (positive: knowledge lost)
Last-half mean: over the snapshots after half of the stream; final: the last snapshot. SE over the
WikiText chunks, or over the 6 held-out books. Exchange rate = stream gain / last-half mean WikiText
forgetting (nats gained on the stream per nat lost on WikiText). K = 1 against shuffled: paired per
held-out chunk (or book), averaged over the last-half snapshots (same positions in both streams).

Writes summary.csv, k1_vs_shuffled.csv, curve.csv (all snapshots), plots/forgetting_verylong.png.
Usage (from experiments/):
    ../venv/bin/python e5_forgetting/e5_forgetting.py run | report
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

import numpy as np  # noqa: E402

import runs  # noqa: E402
from metrics import exchange_rate, forgetting, forgetting_curve, se, summary  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
STREAMS = ["main_K1", "main_shuffled", "verylong"]


def keys():
    return {st: runs.find("14m", "sgd", st) for st in STREAMS}


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def run():
    for k in list(keys().values()) + [runs.frozen_key("14m", st) for st in STREAMS]:
        if not os.path.exists(runs.path_of(k)):
            runs.run(next(s for s in runs.registry() if s["key"] == k))


def stream_gain(key, size, st):
    frozen, loss = runs.load(runs.frozen_key(size, st))["loss"], runs.load(key)["loss"]
    book = np.repeat(np.arange(frozen.shape[0]), frozen.shape[1] * frozen.shape[2])
    return summary((frozen - loss).reshape(-1), book, np.ones(book.shape, bool))


def summarize(key, size, st):
    """One row: stream gain, forgetting (last half, final; WikiText and books), exchange rate."""
    r = runs.load(key)
    g = stream_gain(key, size, st)
    f = forgetting(r["snaps"], r["positions"], r["n_wiki"])
    row = {"run": key, "size": size, "rule": r["meta"]["rule"], "lr": "%g" % r["meta"]["lr"], "stream": st,
           "stream_gain": "%.4f" % g["mean"], "stream_gain_se": "%.4f" % g["se"],
           "frozen_wiki": "%.4f" % r["snaps"][0][:r["n_wiki"]].mean(),
           "frozen_books": "%.4f" % r["snaps"][0][r["n_wiki"]:].mean()}
    for which in ("wiki", "books"):
        for k in ("last_half", "final"):
            row["%s_%s" % (which, k)] = "%.4f" % f[which][k]
            row["%s_%s_se" % (which, k)] = "%.4f" % f[which][k + "_se"]
            row["%s_%s_under_2se" % (which, k)] = bool(abs(f[which][k]) < 2 * f[which][k + "_se"])
    x = exchange_rate(g["mean"], f["wiki"]["last_half"])
    row["exchange_rate"] = "%.2f" % x if x is not None else "no forgetting"
    row["wiki_over_stream_gain"] = "%.2f" % (f["wiki"]["last_half"] / g["mean"]) if g["mean"] > 0 else ""
    return row


def k1_vs_shuffled(k1, sh):
    """Shuffled - K1 forgetting, last-half snapshots, paired per held-out chunk (WikiText) or book."""
    a, b = runs.load(k1), runs.load(sh)
    assert (a["positions"] == b["positions"]).all()
    pos, n_wiki = a["positions"], a["n_wiki"]
    half = [k for k in range(1, len(pos)) if pos[k] > pos[-1] / 2]
    d = np.mean([b["snaps"][k] - a["snaps"][k] for k in half], axis=0)
    per_book = (len(d) - n_wiki) // 6
    bk = [d[n_wiki + per_book * j:n_wiki + per_book * (j + 1)].mean() for j in range(6)]
    out = {}
    for which, x in (("wiki", d[:n_wiki]), ("books", bk)):
        m, s = float(np.mean(x)), se(x)
        out.update({"shuffled_minus_K1_%s" % which: "%.4f" % m, "shuffled_minus_K1_%s_se" % which: "%.4f" % s,
                    "shuffled_minus_K1_%s_under_2se" % which: bool(abs(m) < 2 * s)})
    return out


def report():
    from plots import plt, style, BLUE, MUTED, INK
    k = keys()
    rows = [summarize(k[st], "14m", st) for st in STREAMS]
    write_csv("summary.csv", rows)
    comp = [{"run": "14m sgd", **k1_vs_shuffled(k["main_K1"], k["main_shuffled"])}]
    write_csv("k1_vs_shuffled.csv", comp)
    curve = []
    for st in STREAMS:
        r = runs.load(k[st])
        w, b = forgetting_curve(r["snaps"], r["n_wiki"])
        curve += [{"stream": st, "chunks_read": int(p), "tokens_read": int(p) * 256, "wiki": "%.4f" % w[i],
                   "books": "%.4f" % b[i]} for i, p in enumerate(r["positions"])]
    write_csv("curve.csv", curve)
    for r in rows + comp:
        print(r)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        for st in STREAMS:
            f.write("%s %d s\n" % (k[st], runs.seconds(k[st])))

    style()
    r = runs.load(k["verylong"])
    w, b = forgetting_curve(r["snaps"], r["n_wiki"])
    x = r["positions"] * 256 / 1000
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(x, w, color=BLUE, lw=1.8, label="held-out WikiText (detokenized)")
    ax.plot(x, b, color=BLUE, lw=1.8, ls="--", label="held-out PG-19 books")
    for j in range(1, 4):
        ax.axvline(j * 781 * 256 / 1000, color="#c8c7c3", lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("position in the stream (thousands of tokens; grey lines: book boundaries)")
    ax.set_ylabel("forgetting (nats/token)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pythia-14M, plain SGD lr %s, 4 books x 200k" % rows[2]["lr"], fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "forgetting_verylong.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
