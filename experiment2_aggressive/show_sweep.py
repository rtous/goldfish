"""Summarize online_sweep.csv (written by online_mlp.py and sweep_online.py).

For each configuration and online optimizer/lr it prints the online - frozen accuracy
difference on the bursts and random sequences, and the locality effect (bursts - random),
then the best lr per configuration by locality effect and by online accuracy on bursts.

For the bursts run it also shows the relative accuracy change (%),
(acc_bursts - acc_ref) / acc_ref * 100 (e.g. 25% against a reference of 50% gives -50.0):
  vs batch%   reference: the frozen batch model on the bursts sequence
  vs random%  reference: the online run on the random sequence with the same optimizer and lr
"-" when the reference is missing or a run diverged.
When a (configuration, sequence, optimizer, lr) combination was run several times,
the latest row (last in the file) is used.

Usage:
    python3 show_sweep.py              # all configurations
    python3 show_sweep.py mlp_2 mlp_4  # only these
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_FILE = os.path.join(HERE, "online_sweep.csv")
CONFIG_FILE = os.path.join(HERE, "configurations.json")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def config_order():
    try:
        with open(CONFIG_FILE) as f:
            return [c["name"] for c in json.load(f)]
    except (OSError, ValueError, KeyError):
        return []


def relative_change(acc, ref_acc):
    """Relative accuracy change in %, (acc - acc_ref) / acc_ref * 100; None if unavailable."""
    if acc is None or not ref_acc:
        return None
    return (acc - ref_acc) / ref_acc * 100


def fmt1(value):
    return "-" if value is None else "%+.1f" % value


def fmt(value, signed=True):
    if value is None:
        return "-"
    return ("%+.2f" if signed else "%.2f") % value


def main():
    if not os.path.exists(SWEEP_FILE):
        raise SystemExit("No sweep results yet: run sweep_online.py or online_mlp.py first.")
    with open(SWEEP_FILE) as f:
        rows = list(csv.DictReader(f))

    # Latest row per (configuration, sequence, optimizer, lr): later rows overwrite earlier ones.
    latest = {}
    for r in rows:
        latest[(r["configuration"], r["sequence"], r["optimizer"], number(r["online_lr"]))] = r

    names = sys.argv[1:] or sorted({k[0] for k in latest},
                                   key=lambda n: (config_order() + [n]).index(n))
    best_rows = []
    for name in names:
        settings = sorted({(k[2], k[3]) for k in latest if k[0] == name})
        if not settings:
            print("\n%s: no runs" % name)
            continue
        frozen = None
        print("\n=== %s ===" % name)
        print("%-5s %7s  %8s %8s %8s  %8s %8s  %9s %10s  %-s" % (
            "opt", "lr", "bursts", "random", "locality", "acc_burst", "rest%",
            "vs batch%", "vs random%", "notes"))
        print("%-5s %7s  %8s %8s %8s  %8s %8s" % ("", "", "on-fz", "on-fz", "b-r", "", ""))
        results = []
        for opt, lr in settings:
            diffs, notes = {}, []
            b = latest.get((name, "bursts", opt, lr))
            for seq in ("bursts", "random"):
                r = latest.get((name, seq, opt, lr))
                if r is None:
                    diffs[seq] = None
                    notes.append("no %s run" % seq)
                    continue
                frozen = frozen if frozen is not None else number(r["frozen_accuracy"])
                if r["status"] == "diverged":
                    diffs[seq] = None
                    notes.append("%s diverged at image %s" % (seq, r["diverged_at"]))
                    continue
                diffs[seq] = number(r["online_accuracy"]) - number(r["frozen_accuracy"])
            locality = (diffs["bursts"] - diffs["random"]
                        if diffs["bursts"] is not None and diffs["random"] is not None else None)
            acc_b = number(b["online_accuracy"]) if b and b["status"] == "ok" else None
            rest = number(b["rest_of_burst_accuracy"]) if b and b["status"] == "ok" else None
            r = latest.get((name, "random", opt, lr))
            acc_r = number(r["online_accuracy"]) if r and r["status"] == "ok" else None
            vs_batch = relative_change(acc_b, number(b["frozen_accuracy"]) if b else None)
            vs_random = relative_change(acc_b, acc_r)
            results.append((opt, lr, locality, acc_b))
            print("%-5s %7g  %8s %8s %8s  %8s %8s  %9s %10s  %s" % (
                opt, lr, fmt(diffs["bursts"]), fmt(diffs["random"]), fmt(locality),
                fmt(acc_b, False), fmt(rest, False), fmt1(vs_batch), fmt1(vs_random),
                ", ".join(notes)))
        print("frozen accuracy: %s%%" % fmt(frozen, False))

        def best(i):
            valid = [r for r in results if r[i] is not None]
            return max(valid, key=lambda r: r[i]) if valid else None
        best_rows.append((name, frozen, best(2), best(3)))

    print("\n=== Best online setting per configuration ===")
    print("%-10s %7s  %-24s  %-24s" % ("config", "frozen", "by locality effect", "by online acc on bursts"))
    for name, frozen, by_loc, by_acc in best_rows:
        loc = "%s lr=%g (%+.2f)" % (by_loc[0], by_loc[1], by_loc[2]) if by_loc else "-"
        acc = "%s lr=%g (%.2f%%)" % (by_acc[0], by_acc[1], by_acc[3]) if by_acc else "-"
        print("%-10s %6s%%  %-24s  %-24s" % (name, fmt(frozen, False), loc, acc))


if __name__ == "__main__":
    main()
