"""Experiment 6, part 2: ceiling map for deeper architectures with a FiLM context vector.

For one budget (8: about 6.4k parameters, 4: about 3.2k) and one architecture:
    mixed        trained like offline.py with c = 0 (50 epochs)       weights/<arch>_b<budget>_mixed.pt
    per_context  one model per rotation, from scratch (50 epochs)    weights/<arch>_b<budget>_ctx<k>.pt
    joint        for r in {1, 2, 4, 8}: shared weights + FiLM + c_k, from the mixed model, loss_ctx +
                 loss_gen as in 5c (10 epochs)                        weights/<arch>_b<budget>_joint_r<r>.pt
    oracle_all   few-shot oracle on every parameter, from the mixed model
    oracle_c     few-shot oracle on c only (from 0), from the joint model of each r; on the spec
                 learning rates and on larger ones (oracle_c_ext: 1e-2, 3e-2, 1e-1 added)
Saved weights are reused on a rerun. Writes results.csv rows and grids/part2_*.csv.

Usage (from experiments_MNIST2):
    ../venv/bin/python experiment6_fewshot_ceiling/part2.py <budget> <arch>
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HERE, LRS, WEIGHTS, Net, accuracy, best_per_n, fewshot, n_params, save_grid,  # noqa: E402
                    train_joint, train_offline, train_per_context_batched, update_results)
from contexts import N_CONTEXTS, under_every_context  # noqa: E402
from data import load_split  # noqa: E402

ARCHS = {  # budget -> arch -> (kind, hidden widths or channels)
    8: {"mlp2": ("mlp", (7, 24)), "mlp3": ("mlp", (7, 12, 12)), "cnn": ("cnn", (8, 10))},
    4: {"mlp2": ("mlp", (3, 20)), "mlp3": ("mlp", (3, 12, 12)), "cnn": ("cnn", (6, 5))},
}
RS = (1, 2, 4, 8)
DEVICE = [None]


def to_cpu(obj):
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj.cpu()
EXT_LRS = LRS + (1e-2, 3e-2, 1e-1)


def cached(path, build):
    """Load a saved state dict (and extra tensors) or build and save it."""
    if os.path.exists(path):
        return torch.load(path, map_location=DEVICE[0])
    torch.save(to_cpu(build()), path)
    return torch.load(path, map_location=DEVICE[0])


def main():
    budget, arch = int(sys.argv[1]), sys.argv[2]
    kind, hidden = ARCHS[budget][arch]
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    dev = torch.device(os.environ.get("DEVICE", "cpu"))
    DEVICE[0] = dev
    one_seed = (1000,) if kind == "cnn" else ()  # the CNN's N = 1000 runs are the slowest
    os.makedirs(WEIGHTS, exist_ok=True)
    os.makedirs(os.path.join(HERE, "grids"), exist_ok=True)
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    cache, y_test = under_every_context(x_test).to(dev), y_test.to(dev)
    name = "%s_b%d" % (arch, budget)
    # The CNN (on the GPU) trains from the training set rotated once under every context, and its
    # 8 per-context models at once; the MLPs rotate each batch, as offline.py does.
    x_rot = under_every_context(x_train).to(dev) if kind == "cnn" else None
    base = dict(part=2, budget=budget, arch=arch)

    def plain():
        return Net(kind, hidden).to(dev)

    mixed = plain()
    mixed.load_state_dict(cached(os.path.join(WEIGHTS, name + "_mixed.pt"),
                                 lambda: train_offline(plain(), x_train, y_train, log=name + " mixed", x_rot=x_rot).state_dict()))
    mixed_acc = accuracy(mixed, cache, y_test)
    rows = [dict(base, r=0, params=n_params(mixed), measure="mixed", acc="%.2f" % mixed_acc)]
    print("%s mixed %.2f (%d params)" % (name, mixed_acc, n_params(mixed)), flush=True)
    update_results(rows)

    per_ctx = []
    paths = [os.path.join(WEIGHTS, "%s_ctx%d.pt" % (name, k)) for k in range(N_CONTEXTS)]
    if x_rot is not None and not all(os.path.exists(p) for p in paths):
        for p, sd in zip(paths, train_per_context_batched([plain() for _ in range(N_CONTEXTS)], x_rot, y_train,
                                                          log=name + " per-context")):
            torch.save(to_cpu(sd), p)
    for k in range(N_CONTEXTS):
        m = plain()
        m.load_state_dict(cached(os.path.join(WEIGHTS, "%s_ctx%d.pt" % (name, k)), lambda: train_offline(
            plain(), x_train, y_train, context=k, log="%s ctx%d" % (name, k)).state_dict()))
        with torch.no_grad():
            per_ctx.append((m(cache[k]).argmax(1) == y_test).float().mean().item() * 100)
    pc = sum(per_ctx) / N_CONTEXTS
    print("%s per-context %.2f" % (name, pc), flush=True)
    rows.append(dict(base, r=0, params=n_params(mixed), measure="per_context", acc="%.2f" % pc,
                     gain="%.2f" % (pc - mixed_acc)))
    update_results(rows)

    grid = fewshot(mixed, [n for n, _ in mixed.named_parameters()], x_train, y_train, cache, y_test,
                   one_seed=one_seed, log=name + " oracle_all")
    save_grid("part2_%s_all" % name, grid)
    for N, (a, lr, s) in best_per_n(grid).items():
        rows.append(dict(base, r=0, params=n_params(mixed), measure="oracle_all", N=N, acc="%.2f" % a,
                         gain="%.2f" % (a - mixed_acc), lr=lr, steps=s))
    update_results(rows)

    for r in RS:
        model = Net(kind, hidden, r).to(dev)

        def build():
            torch.manual_seed(r)
            m = Net(kind, hidden, r).to(dev)
            m.load_state_dict(mixed.state_dict(), strict=False)  # FiLM keeps its random init
            ck = train_joint(m, x_train, y_train, log="%s r=%d" % (name, r), x_rot=x_rot)
            return {"state_dict": m.state_dict(), "c": ck}

        saved = cached(os.path.join(WEIGHTS, "%s_joint_r%d.pt" % (name, r)), build)
        model.load_state_dict(saved["state_dict"])
        s_acc = accuracy(model, cache, y_test, torch.zeros(N_CONTEXTS, r, device=dev))
        sc_acc = accuracy(model, cache, y_test, saved["c"])
        norm = saved["c"].norm(dim=1).mean().item()
        print("%s r=%d joint: S %.2f  S+c_k %.2f  mean |c_k| %.2f" % (name, r, s_acc, sc_acc, norm), flush=True)
        p = n_params(model)
        rows += [dict(base, r=r, params=p, measure="joint_S", acc="%.2f" % s_acc, gain="%.2f" % (s_acc - mixed_acc)),
                 dict(base, r=r, params=p, measure="joint_Sc", acc="%.2f" % sc_acc, gain="%.2f" % (sc_acc - mixed_acc),
                      steps="", lr="", N="")]
        update_results(rows)
        grid = fewshot(model, "c", x_train, y_train, cache, y_test, lrs=EXT_LRS, one_seed=one_seed, log="%s r=%d oracle_c" % (name, r))
        save_grid("part2_%s_c_r%d" % (name, r), grid)
        for measure, lrs in (("oracle_c", LRS), ("oracle_c_ext", EXT_LRS)):
            for N, (a, lr, s) in best_per_n(grid, lrs).items():
                rows.append(dict(base, r=r, params=p, measure=measure, N=N, acc="%.2f" % a,
                                 gain="%.2f" % (a - mixed_acc), lr=lr, steps=s))
        update_results(rows)


if __name__ == "__main__":
    main()
