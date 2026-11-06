"""Test 11: eviction with a probe. How much book-specific memory of A is left after I tokens of other
books, with general adaptation removed?

A and C books: books 0-19 of the main stream, paired (i, i + 10): A = i has control C = i + 10 and
the other way round (20 A books). A1 = tokens 0-10239 (40 chunks), A2 = tokens 10240-19999; C2 = the
same range of C, never read in that run.
Interlude: PG-19 validation books (data.val_books(), 42 x 20k) concatenated in dataset order, cyclic,
starting at validation book 2 * A (mod 42), so each A gets a different interlude.
Stream per A: A1 + interlude, read online from the pretrained weights (snapshot.Learner).
Checkpoints: before A1 ("pre"), and after A1 + I tokens, I in INTERLUDES (multiples of 256).
Probe: A2 and C2 scored frozen with the snapshot; also once with the pretrained model.
    gain_A2(I) = loss_frozen(A2) - loss_snap(A2);  gain_C2 likewise;  memory_A = gain_A2 - gain_C2
Token subsets (1000 most frequent ids of the 25-book stream excluded): A2 tokens whose id occurs in A1
against those that do not; the control is C2 tokens whose id occurs (or not) in C1, never read.
Settings: full_1e-4 (all parameters, lr 1e-4); mlp4_3e-3 (layer 4's MLP, lr 3e-3; test 6's pick);
dual (slow full_1e-4 + fast part on layer 2's MLP, alpha 1e-3, lambda 0.99; test 10's pick).
Test 2's old metric (full_1e-4, A = 0, 1, 2): loss(A2 | snapshot after C1 + I) - loss(A2 | snapshot
after A1 + I), I = 0, 10k, 100k, with C1 = C's first 40 chunks in place of A1 (same interlude).

Writes test11_probe_eviction/: losses_<setting>.npz (probe losses, float16), results.csv, means.csv,
test2_metric.csv, pairs.csv (paired differences between settings), halving.csv, runtime.txt (and\nruntime_<setting>.txt), plots/memory_vs_interlude.png. About 55 min with the three settings run as\nthree parallel processes (8-core CPU). Usage:
    python test11_probe_eviction.py run <setting> [n_books]   # one setting (n_books: smoke test)
    python test11_probe_eviction.py report                    # tables and plot from the saved files
"""
import csv
import os
import sys
import time

import numpy as np

from data import HERE, val_books
from dual import mlp_weights
from models import DEFAULT
from online import L, setup
from plots import plt, style, BLUE, ORANGE, INK, MUTED
from snapshot import Learner, score
from test1_distance import TOP

OUT = os.path.join(HERE, "test11_probe_eviction")
N_A, HALF, END = 20, 40 * L, 20000
INTERLUDES = [int(round(i / L)) * L for i in (0, 1000, 3000, 10000, 30000, 100000, 300000)]
CKPTS = ["pre"] + INTERLUDES
SETTINGS = {"full_1e-4": dict(rule="sgd", lr=1e-4, params="all"),
            "mlp4_3e-3": dict(rule="sgd", lr=3e-3, params="mlp@3"),
            "dual": dict(rule="dual", fast={n: (1e-3, 0.99) for n in mlp_weights([1])})}
OLD_BOOKS, OLD_I = (0, 1, 2), [INTERLUDES[i] for i in (0, 3, 5)]


def pairs():
    """A book -> control book C."""
    return {a: (a + 10) % N_A if a < 10 else a - 10 for a in range(N_A)}


def interlude(val, a, n):
    k = len(val)
    order = [(2 * a + j) % k for j in range(n // val.shape[1] + 2)]
    return np.concatenate([val[j] for j in order])[:n]


def run(setting, n_books=N_A, ckpts=CKPTS):
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    model, init, info, d = setup()
    ids, val = d["ids"], val_books(DEFAULT)["ids"]
    cfg = SETTINGS[setting]
    interludes = [c for c in ckpts if c != "pre"]
    t0 = time.time()
    frozen = np.stack([score(model, init, ids[b, HALF:END]) for b in range(N_A)])  # [book, A2 tokens]
    probe = np.full((n_books, len(ckpts), 2, END - HALF), np.nan, dtype=np.float32)
    for a in range(n_books):
        c, t = pairs()[a], time.time()
        mid = interlude(val, a, max(interludes))
        lrn = Learner(model, init, **cfg)
        pos = None
        for k, ck in enumerate(ckpts):
            if ck != "pre":
                lrn.read(ids[a, :HALF]) if pos is None else lrn.read(mid[pos:ck])
                pos = ck
            probe[a, k, 0], probe[a, k, 1] = lrn.score(ids[a, HALF:END]), lrn.score(ids[c, HALF:END])
        print("%s A %d: %.0f s; memory by I %s" % (setting, a, time.time() - t, " ".join(
            "%.3f" % ((np.nanmean(frozen[a] - probe[a, k, 0])) - np.nanmean(frozen[c] - probe[a, k, 1]))
            for k in range(len(ckpts)))), flush=True)
    seconds = {setting: time.time() - t0}
    extra = {}
    if setting == "full_1e-4" and n_books == N_A:  # test 2's metric: C1 in place of A1
        t = time.time()
        for a in OLD_BOOKS:
            c, mid, pos = pairs()[a], interlude(val, a, max(OLD_I)), None
            lrn = Learner(model, init, **cfg)
            for I in OLD_I:
                lrn.read(ids[c, :HALF]) if pos is None else lrn.read(mid[pos:I])
                pos = I
                extra["oldC1_A%d_I%d" % (a, I)] = lrn.score(ids[a, HALF:END]).astype(np.float16)
        seconds["test2_metric"] = time.time() - t
    np.savez_compressed(os.path.join(OUT, "losses_%s.npz" % setting), probe=probe.astype(np.float16),
                        frozen=frozen, ckpts=np.array([str(c) for c in ckpts]), **extra)
    with open(os.path.join(OUT, "runtime_%s.txt" % setting), "w") as f:
        f.write("%s; %d A books; %s\n" % (info["device"], n_books, ", ".join("%s %.0f s" % kv for kv in seconds.items())))
    print(seconds)


def masks(d):
    """Per book: (rare & id in first half, rare & not in first half) over the second half's tokens."""
    ids = d["ids"]
    frequent = np.argsort(-np.bincount(ids[:25].reshape(-1)))[:TOP]
    out = []
    for b in range(N_A):
        second = ids[b, HALF:END]
        rare = ~np.isin(second, frequent)
        seen = np.isin(second, ids[b, :HALF])
        out.append((rare & seen, rare & ~seen))
    return out


def se(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def report():
    from data import books
    d = books(DEFAULT)
    mk = masks(d)
    rows, means, saved = [], [], {}
    for setting in SETTINGS:
        path = os.path.join(OUT, "losses_%s.npz" % setting)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        probe, frozen, ckpts = z["probe"].astype(np.float64), z["frozen"].astype(np.float64), list(z["ckpts"])
        saved[setting] = z
        for k, ck in enumerate(ckpts):
            per = []
            for a in range(probe.shape[0]):
                c = pairs()[a]
                gA, gC = frozen[a] - probe[a, k, 0], frozen[c] - probe[a, k, 1]
                r = {"setting": setting, "A": a, "C": c, "I": ck, "frozen_A2": "%.4f" % np.nanmean(frozen[a]),
                     "frozen_C2": "%.4f" % np.nanmean(frozen[c]), "gain_A2": np.nanmean(gA),
                     "gain_C2": np.nanmean(gC)}
                r["memory_A"] = r["gain_A2"] - r["gain_C2"]
                for j, sub in enumerate(("in_first_half", "not_in_first_half")):
                    r["gain_A2_rare_" + sub] = np.nanmean(gA[mk[a][j]])
                    r["gain_C2_rare_" + sub] = np.nanmean(gC[mk[c][j]])
                    r["memory_rare_" + sub] = r["gain_A2_rare_" + sub] - r["gain_C2_rare_" + sub]
                per.append(r)
            m = {"setting": setting, "I": ck, "books": len(per)}
            for key in [k_ for k_ in per[0] if k_.startswith(("gain", "memory"))]:
                x = [r[key] for r in per]
                m[key], m[key + "_se"] = "%.4f" % np.mean(x), "%.4f" % se(x)
                m[key + "_under_2se"] = bool(abs(np.mean(x)) < 2 * se(x)) if se(x) > 0 else ""
            means.append(m)
            rows += [{k_: ("%.4f" % v if isinstance(v, float) else v) for k_, v in r.items()} for r in per]
    write_csv("results.csv", rows)
    write_csv("means.csv", means)
    for m in means:
        print("%-10s I %-6s memory %s ± %s  gain_A2 %s  gain_C2 %s ± %s  | rare in A1 %s ± %s, not %s ± %s" % (
            m["setting"], m["I"], m["memory_A"], m["memory_A_se"], m["gain_A2"], m["gain_C2"], m["gain_C2_se"],
            m["memory_rare_in_first_half"], m["memory_rare_in_first_half_se"], m["memory_rare_not_in_first_half"],
            m["memory_rare_not_in_first_half_se"]))
    if "full_1e-4" in saved and "oldC1_A0_I0" in saved["full_1e-4"].files:  # test 2's metric
        z = saved["full_1e-4"]
        ckpts = list(z["ckpts"])
        old = []
        for I in OLD_I:
            per_old, per_new = [], []
            for a in OLD_BOOKS:
                snapA = z["probe"][a, ckpts.index(str(I)), 0].astype(np.float64)
                snapC1 = z["oldC1_A%d_I%d" % (a, I)].astype(np.float64)
                per_old.append(np.nanmean(snapC1) - np.nanmean(snapA))
                per_new.append(next(float(r["memory_A"]) for r in rows if r["setting"] == "full_1e-4" and
                                    r["A"] == a and r["I"] == str(I)))
            old.append({"I": I, "books": len(OLD_BOOKS), "test2_metric": "%.4f" % np.mean(per_old),
                        "test2_metric_se": "%.4f" % se(per_old), "memory_A_same_books": "%.4f" % np.mean(per_new),
                        "memory_A_same_books_se": "%.4f" % se(per_new)})
            print(old[-1])
        write_csv("test2_metric.csv", old)
    extra_tables(rows, means)
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:  # one file for all settings
        for s_ in SETTINGS:
            p = os.path.join(OUT, "runtime_%s.txt" % s_)
            if os.path.exists(p):
                f.write(open(p).read())
    plot(means)


def half_point(I, y):
    """Interlude at which y first falls to half of y[0], by linear interpolation in log I (I[0] = 0 taken
    as 256, one chunk); None if it never does."""
    x = np.log(np.maximum(I, L))
    for k in range(1, len(y)):
        if y[k] <= y[0] / 2:
            f = (y[k - 1] - y[0] / 2) / (y[k - 1] - y[k])
            return float(np.exp(x[k - 1] + f * (x[k] - x[k - 1])))
    return None


def extra_tables(rows, means):
    """pairs.csv: paired per-book differences between settings; halving.csv: where memory halves;
    subset shares of A2's scored tokens."""
    out = []
    settings = [s for s in SETTINGS if any(r["setting"] == s for r in rows)]
    Is = [m["I"] for m in means if m["setting"] == settings[0]]
    for s in settings[1:]:
        for I in Is:
            a = {int(r["A"]): r for r in rows if r["setting"] == s and str(r["I"]) == str(I)}
            b = {int(r["A"]): r for r in rows if r["setting"] == "full_1e-4" and str(r["I"]) == str(I)}
            row = {"comparison": "%s - full_1e-4" % s, "I": I}
            for key in ("memory_A", "gain_A2", "gain_C2", "memory_rare_in_first_half"):
                x = [float(a[k][key]) - float(b[k][key]) for k in sorted(a)]
                row[key], row[key + "_se"] = "%.4f" % np.mean(x), "%.4f" % se(x)
                row[key + "_under_2se"] = bool(abs(np.mean(x)) < 2 * se(x)) if se(x) > 0 else ""
            out.append(row)
    write_csv("pairs.csv", out)
    half = []
    for s in settings:
        m = [x for x in means if x["setting"] == s and x["I"] != "pre"]
        I = np.array([int(x["I"]) for x in m])
        for key in ("memory_A", "memory_rare_in_first_half", "gain_A2"):
            y = np.array([float(x[key]) for x in m])
            h = half_point(I, y)
            half.append({"setting": s, "quantity": key, "at_I0": "%.4f" % y[0],
                         "halves_at_tokens": "%.0f" % h if h else "not within 300k",
                         "at_300k": "%.4f" % y[-1], "at_300k_se": next(x[key + "_se"] for x in m[-1:])})
    write_csv("halving.csv", half)
    for r in out + half:
        print(r)
    from data import books
    mk = masks(books(DEFAULT))
    n = np.array([[m.sum() for m in mk[b]] for b in range(N_A)])
    print("rare in A1: %.1f%% of A2 tokens; rare not in A1: %.1f%%" % tuple(100 * n.sum(0) / (N_A * (END - HALF))))


def plot(means=None):
    if means is None:
        means = list(csv.DictReader(open(os.path.join(OUT, "means.csv"))))
    style()
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    colors = {"full_1e-4": BLUE, "mlp4_3e-3": ORANGE, "dual": "#1baa8a"}
    labels = {"full_1e-4": "full_1e-4", "mlp4_3e-3": "layer-4 MLP, lr 3e-3", "dual": "dual (fast part on layer 2)"}
    xs_all = None
    for setting in colors:
        r = [m for m in means if m["setting"] == setting and m["I"] != "pre"]
        if not r:
            continue
        I = np.array([int(m["I"]) for m in r], dtype=float)
        x = np.where(I == 0, 300, I)  # I = 0 drawn at the left edge
        xs_all = (I, x)
        y, s = np.array([float(m["memory_A"]) for m in r]), np.array([float(m["memory_A_se"]) for m in r])
        ax.fill_between(x, y - s, y + s, color=colors[setting], alpha=0.15, lw=0)
        ax.plot(x, y, color=colors[setting], lw=2, marker="o", ms=3.5, label="memory of A, " + labels[setting])
        g = np.array([float(m["gain_C2"]) for m in r])
        ax.plot(x, g, color=colors[setting], lw=1.2, ls="--", label="gain on C2 (general), " + labels[setting])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    I, x = xs_all
    ax.set_xticks(x, ["0" if i == 0 else "%gk" % (i / 1000) for i in np.round(I / 1000, 0) * 1000])
    ax.minorticks_off()
    ax.set_xlabel("interlude after A1 (tokens of other books; log scale, 0 at the left)")
    ax.set_ylabel("nats/token")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Pythia-14M: memory of A = gain on A2 - gain on C2 (same snapshot); bands +- 1 SE, 20 books",
                 fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "memory_vs_interlude.png"), dpi=160)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else N_A
        run(sys.argv[2], n, CKPTS if n == N_A else CKPTS[:5])
    elif sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "plot":
        plot()
