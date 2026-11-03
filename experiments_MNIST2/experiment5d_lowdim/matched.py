"""Experiment 5d, part 2: matched-size mixed models. A model with r context directions stores
S1, the head and r directions: (795 h + 10) + r * 785 h parameters. For each base h and r in
{1, 2, 4}, train a mixed model of the width H whose plain MLP is closest to that, exactly as
offline.py does (its train function and settings, seed 0), and score it frozen on the test set
averaged over contexts. Existing weights/mixed_h<H>.pt are reused (H = 8, and r = 0: the mixed
model of width h itself). New models go to experiment5d_lowdim/weights/mixed_h<H>.pt. Writes the
matched columns of results.csv.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment5d_lowdim/matched.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import HERE, MATCHED_RANKS, SIZES, matched_width, update_results  # noqa: E402
from contexts import under_every_context  # noqa: E402
from data import load_split, mixed_path  # noqa: E402
from models.mlp import build_model  # noqa: E402
from offline import test_accuracy, train  # noqa: E402


def main():
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    cache = under_every_context(x_test)
    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    scores = {}
    for H in sorted({matched_width(h, r) for h in SIZES for r in MATCHED_RANKS}):
        model = build_model(H)
        path = mixed_path(H)
        if not os.path.exists(path):
            path = os.path.join(HERE, "weights", "mixed_h%d.pt" % H)
            if not os.path.exists(path):
                trained = train(H, x_train, y_train)
                torch.save({"hidden_size": H, "state_dict": trained.state_dict()}, path)
        model.load_state_dict(torch.load(path)["state_dict"])
        scores[H] = test_accuracy(model, cache, y_test)
        print("H=%d (%s): %.2f%%" % (H, os.path.relpath(path), scores[H]), flush=True)
    update_results({(h, r): {"matched_H": matched_width(h, r), "matched_acc": "%.2f" % scores[matched_width(h, r)]}
                    for h in SIZES for r in MATCHED_RANKS})


if __name__ == "__main__":
    main()
