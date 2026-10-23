"""Show all results stored in results.csv (written by mnist_mlp.py and online_mlp.py),
together with the online optimizer/lr runs in online_sweep.csv (online_mlp.py, sweep_online.py).

One row per configuration and online setting, with the bursts and random sequences side by side.
The first row of each configuration is the default online setting (Adam, the configuration's
learning_rate); every other setting in online_sweep.csv (latest run of each) follows it, with
its offline columns left blank since they are the same. The offline (frozen) accuracy is the
same on both sequences, as they hold the same images.

The online bursts group also shows the relative accuracy change (%) of the bursts run,
(acc_bursts - acc_ref) / acc_ref * 100 (e.g. 25% against a reference of 50% gives -50.0):
  vs batch%   reference: the frozen batch model on the same sequence (offline acc%)
  vs random%  reference: the online run on the random sequence with the same optimizer and lr
"-" when the reference is missing (or a run diverged).

Usage:
    python3 show_results.py                    # in file order
    python3 show_results.py bursts_accuracy    # sorted by a column, highest first
    python3 show_results.py --no-sweep         # only the default online setting
"""
import csv
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(HERE, "results.csv")
SWEEP_FILE = os.path.join(HERE, "online_sweep.csv")
SEQUENCES = ("bursts", "random")

# (column, header shown, drop priority). When the table is wider than the terminal,
# columns are dropped highest priority number first (0 = never).
COLUMNS = [
    ("configuration", "config", 0),
    ("model", "model", 3),
    ("model_params", "model params", 0),
    ("training_params", "training", 0),
    ("params", "params", 0),
    ("size_bytes", "KB", 0),
    ("test_accuracy", "acc%", 0),
    ("test_efficiency", "acc/KB", 0),
    ("test_set_accuracy", "set%", 1),
    ("online_setting", "opt lr", 0),
    ("bursts_accuracy", "acc%", 0),
    ("bursts_efficiency", "acc/KB", 1),
    ("bursts_vs_batch", "vs batch%", 0),
    ("bursts_vs_random", "vs random%", 0),
    ("random_accuracy", "acc%", 0),
    ("random_efficiency", "acc/KB", 1),
]

# Header shown above each group of columns, by column-name prefix.
GROUPS = {"test_": "offline", "bursts_": "online bursts", "random_": "online random"}

# Short names for common parameter keys.
ABBREV = {
    "hidden_size": "h",
    "learning_rate": "lr",
    "epochs": "ep",
    "batch_size": "bs",
}

NUMERIC = {"params", "size_bytes", "test_accuracy", "test_efficiency", "test_set_accuracy",
           "bursts_accuracy", "bursts_efficiency", "bursts_vs_batch", "bursts_vs_random",
           "random_accuracy", "random_efficiency"}

# Columns left blank in the extra sweep rows (they repeat the row above).
REPEATED = {"model", "model_params", "training_params", "params", "size_bytes",
            "test_accuracy", "test_efficiency", "test_set_accuracy"}


def format_cell(column, value):
    if not value:
        return "-"
    if column.endswith("_params"):
        try:
            d = json.loads(value)
            return " ".join("%s=%s" % (ABBREV.get(k, k), v) for k, v in d.items()) or "-"
        except (ValueError, AttributeError):
            return value
    if column == "size_bytes":
        try:
            return "%.1f" % (float(value) / 1024)
        except ValueError:
            return value
    if column == "updated":
        return value[5:] if len(value) == 16 else value  # drop the year
    return value


# Online columns compared against their offline (frozen) counterpart in the same row.
COMPARE = {"bursts_accuracy": "test_accuracy", "bursts_efficiency": "test_efficiency",
           "random_accuracy": "test_accuracy", "random_efficiency": "test_efficiency"}
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


# Relative accuracy change columns: green when positive (the bursts run is more accurate), red otherwise.
SIGNED = {"bursts_vs_batch", "bursts_vs_random"}


def relative_change(acc, ref_acc):
    """Relative accuracy change in %, (acc - acc_ref) / acc_ref * 100, as "+6.5"; "" if unavailable."""
    try:
        acc, ref_acc = float(acc), float(ref_acc)
        return "%+.1f" % ((acc - ref_acc) / ref_acc * 100)
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def colorize(column, cell, row):
    """Green if the online value beats the offline one, red otherwise; unchanged if either is missing."""
    if column in SIGNED:
        value = as_number(row.get(column) or "")
        if value == float("-inf"):
            return cell
        return (GREEN if value > 0 else RED) + cell + RESET
    if column not in COMPARE:
        return cell
    try:
        online, batch = float(row[column]), float(row[COMPARE[column]])
    except (KeyError, TypeError, ValueError):
        return cell
    return (GREEN if online > batch else RED) + cell + RESET


def as_number(value):
    try:
        return float(value)
    except ValueError:
        return float("-inf")  # empty cells go last


def default_lr(row):
    try:
        return float(json.loads(row["training_params"])["learning_rate"])
    except (KeyError, TypeError, ValueError):
        return None


def load_sweep():
    """Latest online_sweep.csv row per (configuration, sequence, optimizer, lr)."""
    latest = {}
    if os.path.exists(SWEEP_FILE):
        with open(SWEEP_FILE) as f:
            for r in csv.DictReader(f):  # later rows overwrite earlier ones
                latest[(r["configuration"], r["sequence"], r["optimizer"], float(r["online_lr"]))] = r
    return latest


def build_rows(results, sweep):
    """One row per (configuration, online setting), bursts and random side by side."""
    by_config = {}
    for r in results:
        by_config.setdefault(r["configuration"], {})[r.get("sequence") or "bursts"] = r

    rows = []
    for name, per_seq in by_config.items():
        base = per_seq.get("bursts") or next(iter(per_seq.values()))
        lr0 = default_lr(base)
        # online[(optimizer, lr)][sequence] = accuracy string (or "diverged")
        online = {("adam", lr0): {seq: r["online_accuracy"] for seq, r in per_seq.items()
                                  if r.get("online_accuracy")}}
        for (cfg, seq, opt, lr), s in sweep.items():
            if cfg == name:
                online.setdefault((opt, lr), {})[seq] = (
                    s["online_accuracy"] if s["status"] == "ok" else "diverged")
        settings = [("adam", lr0)] + sorted(k for k in online if k != ("adam", lr0))
        for opt, lr in settings:
            if not online[(opt, lr)] and (opt, lr) != ("adam", lr0):
                continue
            row = {c: base.get(c, "") for c in REPEATED}
            row.update({"configuration": name, "_sub": (opt, lr) != ("adam", lr0),
                        "online_setting": "%s %g" % (opt, lr) if lr is not None else opt})
            for seq in SEQUENCES:
                acc = online[(opt, lr)].get(seq, "")
                row[seq + "_accuracy"] = acc
                try:
                    row[seq + "_efficiency"] = "%.4f" % (float(acc) / (float(base["size_bytes"]) / 1024))
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    row[seq + "_efficiency"] = ""
            row["bursts_vs_batch"] = relative_change(row["bursts_accuracy"], base.get("test_accuracy"))
            row["bursts_vs_random"] = relative_change(row["bursts_accuracy"], row["random_accuracy"])
            rows.append(row)
    return rows


def main():
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit("No results yet: run mnist_mlp.py or online_mlp.py first.")
    with open(RESULTS_FILE) as f:
        results = list(csv.DictReader(f))

    args = sys.argv[1:]
    if "--no-sweep" in args:
        args.remove("--no-sweep")
        sweep = {}
    else:
        sweep = load_sweep()
    rows = build_rows(results, sweep)
    blank_repeated = not args  # keep every cell filled when sorted, rows lose their neighbours

    if args:
        key = args[0]
        if rows and key not in rows[0]:
            raise SystemExit("Unknown column '%s'. Columns: %s"
                             % (key, ", ".join(k for k in rows[0] if not k.startswith("_"))))
        rows.sort(key=lambda r: as_number(r[key]), reverse=True)

    columns = list(COLUMNS)
    term_width = shutil.get_terminal_size((120, 24)).columns
    dropped = []
    while True:
        table = [[h for _, h, _ in columns]] + [
            ["" if blank_repeated and r.get("_sub") and c in REPEATED else format_cell(c, r.get(c))
             for c, _, _ in columns] for r in rows]
        widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
        droppable = [col for col in columns if col[2] > 0]
        if sum(widths) + 2 * (len(widths) - 1) <= term_width or not droppable:
            break
        victim = max(droppable, key=lambda col: col[2])
        columns.remove(victim)
        dropped.append(victim[1])

    # Group header above the offline and online column groups.
    groups = [next((g for p, g in GROUPS.items() if c.startswith(p)), "") for c, _, _ in columns]
    line = ""
    for i, w in enumerate(widths):
        if groups[i] and (i == 0 or groups[i - 1] != groups[i]):
            span = [j for j in range(i, len(widths)) if groups[j] == groups[i]]
            span_w = sum(widths[j] for j in span) + 2 * (len(span) - 1)
            line += (" %s " % groups[i]).center(span_w, "-")
        elif not groups[i]:
            line += " " * w
        else:
            continue
        line += "  "
    print(line.rstrip())
    color = sys.stdout.isatty()
    for n, row in enumerate(table):
        cells = [cell.rjust(w) if c in NUMERIC and n > 0 else cell.ljust(w)
                 for cell, w, (c, _, _) in zip(row, widths, columns)]
        if color and n > 0:
            cells = [colorize(c, cell, rows[n - 1]) for cell, (c, _, _) in zip(cells, columns)]
        print("  ".join(cells).rstrip())
        if n == 0:
            print("  ".join("-" * w for w in widths))
    if dropped:
        print("(hidden to fit terminal: %s)" % ", ".join(dropped))

if __name__ == "__main__":
    main()