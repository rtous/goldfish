"""Show results.csv (written by mnist_mlp.py and online_mlp.py): one row per configuration in
configurations.json, in file order, with the offline (frozen) model and the online runs on the
bursts and random sequences side by side. The offline accuracy is the same on both sequences,
as they hold the same images.

The online bursts group also shows the relative accuracy change (%) of the bursts run,
(acc_bursts - acc_ref) / acc_ref * 100 (e.g. 25% against a reference of 50% gives -50.0):
  vs batch%   reference: the frozen batch model on the same sequence (offline acc%)
  vs random%  reference: the online run on the random sequence
"-" when the reference is missing (or a run diverged).

A "*" after the configuration name means configurations.json changed since the results were
measured (rerun mnist_mlp.py for model/training changes, online_mlp.py for online changes).

The first argument is the experiment (experiment1, experiment2, ...; or just its number).

Usage:
    python3 show_results.py experiment2                    # in configurations.json order
    python3 show_results.py experiment2 bursts_accuracy    # sorted by a column, highest first
"""
import json
import shutil
import sys

from mnist_mlp import fast_layers, list_experiments, load_configurations, params_json, read_results, set_experiment

# (column, header shown, drop priority). When the table is wider than the terminal,
# columns are dropped highest priority number first (0 = never).
COLUMNS = [
    ("configuration", "config", 0),
    ("model", "model", 3),
    ("model_params", "model params", 0),
    ("training_params", "training", 2),
    ("params", "params", 0),
    ("size_bytes", "KB", 0),
    ("offline_accuracy", "acc%", 0),
    ("offline_efficiency", "acc/KB", 0),
    ("test_set_accuracy", "set%", 1),
    ("online_params", "online", 0),
    ("bursts_accuracy", "acc%", 0),
    ("bursts_efficiency", "acc/KB", 1),
    ("bursts_vs_batch", "vs batch%", 0),
    ("bursts_vs_random", "vs random%", 0),
    ("random_accuracy", "acc%", 0),
    ("random_efficiency", "acc/KB", 1),
]

# Header shown above each group of columns, by column name.
GROUPS = {"offline_accuracy": "offline", "offline_efficiency": "offline", "test_set_accuracy": "offline",
          "bursts_accuracy": "online bursts", "bursts_efficiency": "online bursts",
          "bursts_vs_batch": "online bursts", "bursts_vs_random": "online bursts",
          "random_accuracy": "online random", "random_efficiency": "online random"}

# Short names for common parameter keys.
ABBREV = {
    "hidden_size": "h",
    "learning_rate": "lr",
    "epochs": "ep",
    "batch_size": "bs",
    "width": "c",
    "dropout": "do",
    "optimizer": "opt",
    "weight_decay": "wd",
    "lr_schedule": "sched",
    "label_smoothing": "ls",
    "augment_shift": "shift",
}

NUMERIC = {"params", "size_bytes", "offline_accuracy", "offline_efficiency", "test_set_accuracy",
           "bursts_accuracy", "bursts_efficiency", "bursts_vs_batch", "bursts_vs_random",
           "random_accuracy", "random_efficiency"}

# Online columns compared against their offline (frozen) counterpart in the same row.
COMPARE = {"bursts_accuracy": "offline_accuracy", "bursts_efficiency": "offline_efficiency",
           "random_accuracy": "offline_accuracy", "random_efficiency": "offline_efficiency"}
# Relative change columns: green when positive (the bursts run is more accurate), red otherwise.
SIGNED = {"bursts_vs_batch", "bursts_vs_random"}
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def format_online(value):
    """online_params as "sgd 0.0003" (plus " bn=update" / " ls=0.1" when not the default, and
    " fast=<lr>/<decay>(<fast layers>)" / " prior=<rate>" when set)."""
    p = json.loads(value)
    text = "%s %g" % (p.get("optimizer", "sgd"), p["learning_rate"])
    if p.get("bn_mode", "frozen") != "frozen":
        text += " bn=" + p["bn_mode"]
    if p.get("label_smoothing", 0):
        text += " ls=%g" % p["label_smoothing"]
    if "fast" in p:
        text += " fast=%g/%g(%s)" % (p["fast"]["learning_rate"], p["fast"]["decay"], fast_layers(p["fast"]["layers"]))
    if "label_prior" in p:
        text += " prior=%g" % p["label_prior"]
    return text


def format_cell(column, value):
    if not value:
        return "-"
    try:
        if column == "online_params":
            return format_online(value)
        if column.endswith("_params"):
            return " ".join("%s=%s" % (ABBREV.get(k, k), v) for k, v in json.loads(value).items()) or "-"
        if column == "size_bytes":
            return "%.1f" % (float(value) / 1024)
    except (ValueError, KeyError, AttributeError):
        pass
    return value


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")  # empty cells go last


def efficiency(acc, size_bytes):
    """Accuracy points per KB, as "5.4421"; "" if unavailable."""
    try:
        return "%.4f" % (float(acc) / (float(size_bytes) / 1024))
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


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
        value = as_number(row.get(column))
        return cell if value == float("-inf") else (GREEN if value > 0 else RED) + cell + RESET
    if column not in COMPARE:
        return cell
    online, batch = as_number(row.get(column)), as_number(row.get(COMPARE[column]))
    if float("-inf") in (online, batch):
        return cell
    return (GREEN if online > batch else RED) + cell + RESET


def build_rows(configs, results):
    """One row per configuration (those without results are skipped), with the derived columns."""
    rows = []
    for config in configs:
        r = results.get(config["name"])
        if r is None:
            continue
        row = dict(r)
        offline_changed = (r.get("model") != config["model"]
                           or r.get("model_params") != params_json(config, "model_params")
                           or r.get("training_params") != params_json(config, "training_params"))
        online_changed = bool(r.get("online_params")) and r["online_params"] != params_json(config, "online_params")
        if offline_changed or online_changed:
            row["configuration"] += "*"
        for prefix in ("offline", "bursts", "random"):
            row[prefix + "_efficiency"] = efficiency(r.get(prefix + "_accuracy"), r.get("size_bytes"))
        row["bursts_vs_batch"] = relative_change(r.get("bursts_accuracy"), r.get("offline_accuracy"))
        row["bursts_vs_random"] = relative_change(r.get("bursts_accuracy"), r.get("random_accuracy"))
        rows.append(row)
    return rows


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__.strip() + "\n\nExperiments: " + ", ".join(list_experiments()))
    set_experiment(sys.argv[1])
    configs = load_configurations()
    results = read_results()
    if not results:
        raise SystemExit("No results yet: run mnist_mlp.py and online_mlp.py first.")
    rows = build_rows(configs, results)
    missing = [c["name"] for c in configs if c["name"] not in results]

    args = sys.argv[2:]
    if args:
        key = args[0]
        valid = [c for c, _, _ in COLUMNS]
        if key not in valid:
            raise SystemExit("Unknown column '%s'. Columns: %s" % (key, ", ".join(valid)))
        rows.sort(key=lambda r: as_number(r.get(key)), reverse=True)

    columns = list(COLUMNS)
    term_width = shutil.get_terminal_size((120, 24)).columns
    dropped = []
    while True:
        table = [[h for _, h, _ in columns]] + [[format_cell(c, r.get(c)) for c, _, _ in columns]
                                                for r in rows]
        widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
        droppable = [col for col in columns if col[2] > 0]
        if sum(widths) + 2 * (len(widths) - 1) <= term_width or not droppable:
            break
        victim = max(droppable, key=lambda col: col[2])
        columns.remove(victim)
        dropped.append(victim[1])

    # Group header above the offline and online column groups.
    groups = [GROUPS.get(c, "") for c, _, _ in columns]
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
    if any(r["configuration"].endswith("*") for r in rows):
        print("* configurations.json changed since these results were measured.")
    if missing:
        print("No results yet: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
