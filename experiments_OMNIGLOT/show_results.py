"""Print an experiment's results.csv as one compact table per architecture, sorted by accuracy
on the bursts stream:
    bursts / shuffled     online accuracy on the stream and on its shuffled control (%)
    gain, vs prior, vs mask   pp over frozen, over control 2a (label prior), over 2b (alphabet mask)
    occ0 / occ1 / occ4_7  accuracy at the first sighting of a class in the burst, the second, the 5th-8th
    KB                    model + extra online state (F, alpha, lambda, label counts or cache)
On a terminal, rows that beat both the label prior and the cache are green.

Usage:
    python show_results.py 1
"""
import csv
import os
import sys

from data import experiment_dir, experiments

COLUMNS = ["config", "bursts", "shuffled", "gain", "gain_vs_prior", "gain_vs_mask", "locality", "occ0", "occ1",
           "occ4_7", "KB"]
HEADERS = ["config", "bursts", "shuffled", "gain", "vs prior", "vs mask", "locality", "occ0", "occ1", "occ4_7", "KB"]
GREEN, RESET = "\033[32m", "\033[0m"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip() + "\n\nExperiments: " + ", ".join(experiments()))
    exp_dir = experiment_dir(sys.argv[1])
    path = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(path):
        raise SystemExit("No %s: run online.py first." % path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    color = sys.stdout.isatty()
    for arch in sorted({r["arch"] for r in rows}):
        sel = [r for r in rows if r["arch"] == arch]
        for r in sel:
            r["KB"] = "%.0f" % (float(r["model_kb"]) + float(r["extra_kb"]))
        sel.sort(key=lambda r: -float(r["bursts"] or "-1"))
        best_cache = max([float(r["bursts"]) for r in sel if r["method"] == "cache" and r["bursts"]] or [0])
        prior = next((float(r["bursts"]) for r in sel if r["config"] == "prior"), 0)
        table = [HEADERS] + [[r[c] or ("diverged" if r["diverged"] == "true" else "-") for c in COLUMNS] for r in sel]
        widths = [max(len(row[i]) for row in table) for i in range(len(COLUMNS))]
        print("%s, architecture %s" % (os.path.basename(exp_dir), arch))
        for n, row in enumerate(table):
            line = "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths)))
            r = sel[n - 1] if n else None
            if color and r and r["bursts"] and r["method"] not in ("mask",) and \
                    float(r["bursts"]) > max(prior, best_cache):
                line = GREEN + line + RESET
            print(line)
            if n == 0:
                print("  ".join("-" * w for w in widths))
        print()


if __name__ == "__main__":
    main()
