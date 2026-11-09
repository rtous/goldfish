"""E4 (claim C7): how fast is a book's memory evicted by other text?

A and C books: main books 0-19, paired (i, i + 10): A = i has control C = i + 10 and the other way
round (20 A books). A1 = tokens 0-10239 (40 chunks), A2 = tokens 10240-19999; C2 = the same range of
C, never read in that run.
Interlude: the 42 PG-19 validation books concatenated in dataset order, cyclic, starting at
validation book 2 * A (mod 42), so each A gets a different interlude.
Stream per A: A1 + interlude, read online from the pretrained weights (plain SGD, all parameters,
lr 1e-4). Checkpoints: before A1 ("pre") and after A1 + I tokens, I in INTERLUDES (multiples of 256).
Probe: A2 and C2 scored frozen with each snapshot, and once with the pretrained model.
    gain_A2(I) = loss_frozen(A2) - loss_snap(A2);  gain_C2 likewise;  memory_A = gain_A2 - gain_C2
Rare-token subsets (1000 most frequent ids of the main 25-book stream excluded): A2 tokens whose id
occurs in A1 against those that do not; the control is the same split of C2 against C1 (never read).
Means and SE over the 20 A books. Half-life: interlude at which a quantity first falls to half its
I = 0 value, by linear interpolation in log I (I = 0 taken as 256).
Fits (optional extra): memory_A(I) against I, weighted by 1 / SE^2, with an exponential with a floor
a exp(-I / tau) + c and a power law a (1 + I / I0)^(-b); both fit errors reported. The 7 points share
the same 20 books, so they are not independent.

Writes e4_eviction/: losses.npz (probe losses, float16), results.csv (per book), means.csv,
halving.csv, fits.csv, runtime.txt, plots/memory_vs_interlude.png. Usage (from experiments/):
    ../venv/bin/python e4_eviction/e4_eviction.py run       # about 1 hour
    ../venv/bin/python e4_eviction/e4_eviction.py report
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))

import numpy as np  # noqa: E402

from data import L, N_MAIN, books, val_books  # noqa: E402
from metrics import TOP, se  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
N_A, HALF, END = 20, 40 * L, 20000
INTERLUDES = [int(round(i / L)) * L for i in (0, 1000, 3000, 10000, 30000, 100000, 300000)]
CKPTS = ["pre"] + INTERLUDES
LR = 1e-4


def pairs():
    """A book -> control book C."""
    return {a: (a + 10) % N_A if a < 10 else a - 10 for a in range(N_A)}


def interlude(val, a, n):
    k = len(val)
    order = [(2 * a + j) % k for j in range(n // val.shape[1] + 2)]
    return np.concatenate([val[j] for j in order])[:n]


def run():
    from learner import Learner, score, setup
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    model, init, info = setup()
    ids, val = books()["ids"], val_books()["ids"]
    t0 = time.time()
    frozen = np.stack([score(model, init, ids[b, HALF:END]) for b in range(N_A)])  # [book, A2 tokens]
    probe = np.full((N_A, len(CKPTS), 2, END - HALF), np.nan, dtype=np.float32)
    for a in range(N_A):
        c, t = pairs()[a], time.time()
        mid = interlude(val, a, max(INTERLUDES))
        lrn = Learner(model, init, "sgd", LR)
        pos = None
        for k, ck in enumerate(CKPTS):
            if ck != "pre":
                lrn.read(ids[a, :HALF]) if pos is None else lrn.read(mid[pos:ck])
                pos = ck
            probe[a, k, 0], probe[a, k, 1] = lrn.score(ids[a, HALF:END]), lrn.score(ids[c, HALF:END])
        print("A %d: %.0f s; memory by I %s" % (a, time.time() - t, " ".join(
            "%.3f" % ((np.nanmean(frozen[a] - probe[a, k, 0])) - np.nanmean(frozen[c] - probe[a, k, 1]))
            for k in range(len(CKPTS)))), flush=True)
    sec = time.time() - t0
    np.savez_compressed(os.path.join(OUT, "losses.npz"), probe=probe.astype(np.float16), frozen=frozen,
                        ckpts=np.array([str(c) for c in CKPTS]))
    with open(os.path.join(OUT, "runtime.txt"), "w") as f:
        f.write("%s; 20 A books, plain SGD lr 1e-4; %.0f s (%.0f min)\n" % (info["device"], sec, sec / 60))


def masks():
    """Per book: (rare & id in first half, rare & not in first half) over the second half's tokens."""
    ids = books()["ids"]
    frequent = np.argsort(-np.bincount(ids[:N_MAIN].reshape(-1)))[:TOP]
    out = []
    for b in range(N_A):
        second = ids[b, HALF:END]
        rare = ~np.isin(second, frequent)
        seen = np.isin(second, ids[b, :HALF])
        out.append((rare & seen, rare & ~seen))
    return out


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def half_point(I, y):
    """Interlude at which y first falls to half of y[0], by linear interpolation in log I (I[0] = 0 taken
    as 256, one chunk); None if it never does."""
    x = np.log(np.maximum(I, L))
    for k in range(1, len(y)):
        if y[k] <= y[0] / 2:
            f = (y[k - 1] - y[0] / 2) / (y[k - 1] - y[k])
            return float(np.exp(x[k - 1] + f * (x[k] - x[k - 1])))
    return None


def fits(I, y, s):
    """Weighted least squares (1 / SE^2) of memory_A against I, numpy only: the nonlinear parameters on a
    fine log grid, the linear ones solved exactly at each grid point.
        exponential + floor  a exp(-I / tau) + c     (grid: tau; linear: a, c)
        power law            a (1 + I / I0)^(-b)     (grid: I0, b; linear: a)"""
    w = 1 / s

    def solve(cols):
        A = np.stack(cols, 1)
        coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        return coef, float((((A @ coef) - y) * w) ** 2 @ np.ones(len(y)))

    best = {}
    for tau in np.geomspace(10, 1e7, 4000):
        coef, chi2 = solve([np.exp(-I / tau), np.ones_like(I)])
        if chi2 < best.get("exp", (np.inf,))[0]:
            best["exp"] = (chi2, "a %.4g, tau %.4g, c %.4g" % (coef[0], tau, coef[1]), coef[0] * np.exp(-I / tau) + coef[1])
    for i0 in np.geomspace(1, 1e7, 400):
        for b in np.linspace(0.01, 3, 300):
            col = (1 + I / i0) ** (-b)
            coef, chi2 = solve([col])
            if chi2 < best.get("pow", (np.inf,))[0]:
                best["pow"] = (chi2, "a %.4g, I0 %.4g, b %.4g" % (coef[0], i0, b), coef[0] * col)
    out = []
    for key, name, n_par in (("exp", "exponential + floor", 3), ("pow", "power law", 3)):
        chi2, params, fit = best[key]
        out.append({"model": name, "params": params, "chi2": "%.2f" % chi2, "dof": len(I) - n_par,
                    "rmse_nats": "%.4f" % float(np.sqrt(np.mean((y - fit) ** 2)))})
    return out


def report():
    mk = masks()
    z = np.load(os.path.join(OUT, "losses.npz"))
    probe, frozen, ckpts = z["probe"].astype(np.float64), z["frozen"].astype(np.float64), list(z["ckpts"])
    rows, means = [], []
    for k, ck in enumerate(ckpts):
        per = []
        for a in range(N_A):
            c = pairs()[a]
            gA, gC = frozen[a] - probe[a, k, 0], frozen[c] - probe[a, k, 1]
            r = {"A": a, "C": c, "I": ck, "gain_A2": np.nanmean(gA), "gain_C2": np.nanmean(gC)}
            r["memory_A"] = r["gain_A2"] - r["gain_C2"]
            for j, sub in enumerate(("in_A1", "not_in_A1")):
                r["memory_rare_" + sub] = np.nanmean(gA[mk[a][j]]) - np.nanmean(gC[mk[c][j]])
            per.append(r)
        m = {"I": ck, "books": len(per)}
        for key in ("gain_A2", "gain_C2", "memory_A", "memory_rare_in_A1", "memory_rare_not_in_A1"):
            x = [r[key] for r in per]
            m[key], m[key + "_se"] = "%.4f" % np.mean(x), "%.4f" % se(x)
            m[key + "_under_2se"] = bool(abs(np.mean(x)) < 2 * se(x))
        means.append(m)
        rows += [{k_: ("%.4f" % v if isinstance(v, float) else v) for k_, v in r.items()} for r in per]
    write_csv("results.csv", rows)
    write_csv("means.csv", means)
    m = [x for x in means if x["I"] != "pre"]
    I = np.array([int(x["I"]) for x in m], dtype=float)
    half = []
    for key in ("memory_A", "memory_rare_in_A1", "gain_A2"):
        y = np.array([float(x[key]) for x in m])
        h = half_point(I, y)
        half.append({"quantity": key, "at_I0": "%.4f" % y[0], "halves_at_tokens": "%.0f" % h if h else "not within 300k",
                     "at_300k": "%.4f" % y[-1], "at_300k_se": m[-1][key + "_se"]})
    write_csv("halving.csv", half)
    y, s = np.array([float(x["memory_A"]) for x in m]), np.array([float(x["memory_A_se"]) for x in m])
    fr = fits(I, y, s)
    write_csv("fits.csv", fr)
    for r in means + half + fr:
        print(r)
    plot(means)


def plot(means):
    from plots import plt, style, BLUE, INK, MUTED
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    r = [m for m in means if m["I"] != "pre"]
    I = np.array([int(m["I"]) for m in r], dtype=float)
    x = np.where(I == 0, 300, I)
    for key, ls, label in (("memory_A", "-", "memory of A (gain on A2 - gain on C2)"),
                           ("gain_C2", "--", "gain on C2 (general adaptation)")):
        y, s = np.array([float(m[key]) for m in r]), np.array([float(m[key + "_se"]) for m in r])
        ax.fill_between(x, y - s, y + s, color=BLUE, alpha=0.15, lw=0)
        ax.plot(x, y, color=BLUE, ls=ls, lw=2 if ls == "-" else 1.2, marker="o" if ls == "-" else None, ms=3.5, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(x, ["0" if i == 0 else "%gk" % round(i / 1000) for i in I])
    ax.minorticks_off()
    ax.set_xlabel("interlude after A1 (tokens of other books; log scale, 0 at the left)")
    ax.set_ylabel("nats/token")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pythia-14M, plain SGD lr 1e-4: memory of A (bands +- 1 SE, 20 books)", fontsize=8.5, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "memory_vs_interlude.png"), dpi=160)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
