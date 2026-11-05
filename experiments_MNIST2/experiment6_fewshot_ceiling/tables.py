"""Print part 2's tables (one per budget, markdown) from results.csv, and apply the decision rule.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment6_fewshot_ceiling/tables.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHS = ("mlp1", "mlp2", "mlp3", "cnn")
RS = ("1", "2", "4", "8")
NS = ("10", "20", "50", "100", "200", "1000")


def main():
    with open(os.path.join(HERE, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    get = {(r["budget"], r["arch"], r["r"], r["measure"], r["N"]): r for r in rows}

    def acc(*key):
        r = get.get(key)
        return float(r["acc"]) if r else None

    def fmt(x, sign=False):
        return "" if x is None else ("%+.2f" % x if sign else "%.2f" % x)

    passed = []
    for b in ("8", "4"):
        print("\nBudget %s (accuracy %%, gains in pp over the architecture's mixed model)\n" % b)
        print("| arch | params (r = 0 / 8) | mixed | per-context | gap | " +
              " | ".join("S / S+c_k r=%s" % r for r in RS) + " |")
        print("|---|---|---|---|---|" + "---|" * len(RS))
        for a in ARCHS:
            part = "1" if a == "mlp1" else "2"
            m = get.get((b, a, "0", "mixed", ""))
            if m is None:
                continue
            mixed = float(m["acc"])
            pc = acc(b, a, "0", "per_context", "") if part == "2" else {"8": 93.26, "4": 87.31}[b]
            p8 = get.get((b, a, "8", "joint_S", ""))
            joint = []
            for r in RS:
                s, sc = acc(b, a, r, "joint_S", ""), acc(b, a, r, "joint_Sc", "")
                joint.append("" if s is None else "%.2f / %.2f" % (s, sc))
            print("| %s | %s / %s | %.2f | %s | %s | %s |" % (
                a, m["params"], p8["params"] if p8 else "", mixed, fmt(pc),
                fmt(None if pc is None else pc - mixed, True), " | ".join(joint)))
        print("\nFew-shot oracle, gain over mixed (pp); c-only: best r (lr up to 1e-1)\n")
        print("| arch | oracle | " + " | ".join("N=%s" % n for n in NS) + " |")
        print("|---|---|" + "---|" * len(NS))
        for a in ARCHS:
            m = get.get((b, a, "0", "mixed", ""))
            if m is None:
                continue
            mixed = float(m["acc"])
            measures = [("oracle_all", "all")] + ([("oracle_first", "first layer")] if a == "mlp1" else []) + \
                ([("oracle_c_ext", "c only")] if a != "mlp1" else [])
            for meas, label in measures:
                cells = []
                for n in NS:
                    if meas == "oracle_c_ext":
                        cand = [(acc(b, a, r, meas, n), r) for r in RS if acc(b, a, r, meas, n) is not None]
                        cells.append("%+.2f (r=%s)" % (max(cand)[0] - mixed, max(cand)[1]) if cand else "")
                    else:
                        x = acc(b, a, "0", meas, n)
                        cells.append(fmt(None if x is None else x - mixed, True))
                print("| %s | %s | %s |" % (a, label, " | ".join(cells)))
            if a != "mlp1":
                pc = acc(b, a, "0", "per_context", "")
                c50 = [acc(b, a, r, "oracle_c_ext", "50") for r in RS]
                c50 = [x for x in c50 if x is not None]
                gap, gain = pc - mixed, (max(c50) - mixed if c50 else None)
                ok = gap >= 10 and gain is not None and gain >= 5
                passed.append((b, a, gap, gain, ok))
    print("\nDecision rule (gap >= 10 pp and c-only gain at N = 50 >= 5 pp):")
    for b, a, gap, gain, ok in passed:
        print("  budget %s %-5s gap %5.2f  c-only N=50 %s  -> %s" % (b, a, gap, fmt(gain, True) or "pending",
                                                                    "PASS" if ok else "fail"))


if __name__ == "__main__":
    main()
