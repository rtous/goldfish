"""Show all results stored in results.csv (written by mnist_mlp.py and online_mlp.py).

Usage:
    python3 show_results.py                    # in file order
    python3 show_results.py online_accuracy    # sorted by a column, highest first
"""
import csv
import json
import os
import shutil
import sys

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

# (column in results.csv, header shown, drop priority). When the table is wider
# than the terminal, columns are dropped highest priority number first (0 = never).
COLUMNS = [
    ("configuration", "config", 0),
    ("sequence", "sequence", 0),
    ("model", "model", 3),
    ("model_params", "model params", 0),
    ("training_params", "training", 0),
    ("params", "params", 0),
    ("size_bytes", "KB", 0),
    ("test_accuracy", "acc%", 0),
    ("test_efficiency", "acc/KB", 0),
    ("test_set_accuracy", "set%", 1),
    ("online_accuracy", "acc%", 0),
    ("online_first_in_burst_accuracy", "1st%", 0),
    ("online_efficiency", "acc/KB", 0),
    ("updated", "updated", 2),
]

# Short names for common parameter keys.
ABBREV = {
    "hidden_size": "h",
    "learning_rate": "lr",
    "epochs": "ep",
    "batch_size": "bs",
}

NUMERIC = {"params", "size_bytes", "test_accuracy", "test_efficiency", "test_set_accuracy",
           "online_accuracy",
           "online_first_in_burst_accuracy", "online_efficiency"}


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


# Online columns compared against their batch (offline) counterpart in the same row.
COMPARE = {"online_accuracy": "test_accuracy", "online_efficiency": "test_efficiency"}
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def colorize(column, cell, row):
    """Green if the online value beats the batch one, red otherwise; unchanged if either is missing."""
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


def main():
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit("No results yet: run mnist_mlp.py or online_mlp.py first.")
    with open(RESULTS_FILE) as f:
        rows = list(csv.DictReader(f))

    if len(sys.argv) > 1:
        key = sys.argv[1]
        if rows and key not in rows[0]:
            raise SystemExit("Unknown column '%s'. Columns: %s" % (key, ", ".join(rows[0])))
        rows.sort(key=lambda r: as_number(r[key]), reverse=True)

    columns = list(COLUMNS)
    term_width = shutil.get_terminal_size((120, 24)).columns
    dropped = []
    while True:
        table = [[h for _, h, _ in columns]] + [[format_cell(c, r.get(c)) for c, _, _ in columns] for r in rows]
        widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
        droppable = [col for col in columns if col[2] > 0]
        if sum(widths) + 2 * (len(widths) - 1) <= term_width or not droppable:
            break
        victim = max(droppable, key=lambda col: col[2])
        columns.remove(victim)
        dropped.append(victim[1])

    # Group header above the test_* and online_* columns.
    groups = ["offline" if c.startswith("test_") else "online" if c.startswith("online_") else ""
              for c, _, _ in columns]
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