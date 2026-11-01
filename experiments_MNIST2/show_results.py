"""Print an experiment's results.csv as one compact table, sorted by stream, h (descending), then
gain over the offline model (descending):
    %acc offline             the frozen mixed model on the stream
    %acc online              online learning on the bursts stream
    %acc shuffled(control)   online learning on the shuffled control
    acc/KB                   online accuracy per KB of model weights (784 -> h -> 10 in float32)
On a terminal, %acc online is bold, and green when it beats both offline and the control.

Usage:
    python show_results.py 1
"""
import csv
import os
import sys

from data import experiment_dir, experiments, load_config

COLUMNS = ["config", "h", "stream", "frozen", "bursts", "shuffled", "acc/KB"]
HEADERS = ["config", "h", "stream", "%acc offline", "%acc online", "%acc shuffled(control)", "acc/KB"]
NUMERIC = COLUMNS[1:2] + COLUMNS[3:]
BOLD, GREEN, RESET = "\033[1m", "\033[32m", "\033[0m"


def kb(h):
    return (784 * h + h + h * 10 + 10) * 4 / 1024


def beats_both(r):
    return bool(r["bursts"] and r["shuffled"]) and float(r["bursts"]) > max(float(r["frozen"]), float(r["shuffled"]))


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip() + "\n\nExperiments: " + ", ".join(experiments()))
    exp_dir = experiment_dir(sys.argv[1])
    path = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(path):
        raise SystemExit("No %s: run online.py first." % path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["acc/KB"] = "%.2f" % (float(r["bursts"]) / kb(int(r["h"]))) if r["bursts"] else ""
    streams = load_config(exp_dir)["streams"]
    gain = lambda r: float(r["gain"]) if r["gain"] else float("-inf")
    rows.sort(key=lambda r: (streams.index(r["stream"]) if r["stream"] in streams else 9, -int(r["h"]), -gain(r)))
    table = [HEADERS] + [[r[c] or ("diverged" if r["diverged"] == "true" else "-") for c in COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(COLUMNS))]
    color = sys.stdout.isatty()
    print(os.path.basename(exp_dir))
    for n, row in enumerate(table):
        cells = [v.rjust(w) if c in NUMERIC else v.ljust(w) for v, w, c in zip(row, widths, COLUMNS)]
        if color:
            i = COLUMNS.index("bursts")
            cells[i] = BOLD + (GREEN if n > 0 and beats_both(rows[n - 1]) else "") + cells[i] + RESET
        print("  ".join(cells))
        if n == 0:
            print("  ".join("-" * w for w in widths))
        elif n < len(rows) and (rows[n]["stream"], rows[n]["h"]) != (rows[n - 1]["stream"], rows[n - 1]["h"]):
            print()


if __name__ == "__main__":
    main()
