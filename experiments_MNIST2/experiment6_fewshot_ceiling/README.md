# Experiment 6: few-shot ceilings, and does depth plus a context vector make rotation low-dimensional?

**Question.** Experiment 5d concluded that rotation is too high-dimensional as a context. A
narrower reading is that rotation is high-dimensional only as a change to the first layer of
784 -> h -> 10, the only place that network can store it. Before changing the kind of locality,
this experiment checks two things offline, with no online runs:

1. **Part 1.** How much can any learner gain from N images of one context, with N matched to
   burst positions? (The per-context models, 93.26 / 87.31%, had 60k images per context.)
2. **Part 2.** With depth and a small FiLM context vector c (r numbers), at the same parameter
   budget, is the context low-dimensional: can c alone, fitted on few images, recover a large part
   of a large mixed-to-per-context gap?

**Decision rule** (fixed before the runs). If some architecture has a mixed-to-per-context gap of
at least 10 pp AND its context-only oracle gains at least 5 pp over its mixed model at N = 50,
proceed to online runs with it (Meta-SGD / CAVIA-style, adapting only c online, with S, the
per-parameter learning rates and the decays meta-learned as in experiment 3). If not, conclude
that rotation is the wrong kind of locality at this budget, and switch.

**Caveats.** Every grid (learning rate x steps) is chosen on the test set, so every oracle
number is optimistic. Training runs (mixed, per-context, joint) use one seed. The few-shot oracle
averages 8 contexts x 3 seeds (one seed for the CNN at N = 1000, to save time).

## Few-shot oracle (both parts, `common.fewshot`)

For each N in {10, 20, 50, 100, 200, 1000}, context k and seed s: a sample of N training images
(nested across N), rotated under k. Start from the given weights and fine-tune full-batch with
Adam, lr in {1e-4, 3e-4, 1e-3, 3e-3}, scoring on the whole test set under k after 10, 30, 100 and
300 steps. Average over contexts and seeds; report the best (lr, steps) cell per N. The whole
grid is in `grids/`.

## Part 1: few-shot oracle on the current MLP (`part1_mlp.py`, `compare_online.py`)

Starts from `weights/mixed_h<h>.pt` and fine-tunes either every parameter (`all`) or the first
layer only (`first`). Gain = accuracy minus the mixed model (74.05 / 64.78%). Runs in about a
minute.

| N | h = 8 `all` | gain | h = 8 `first` | gain | h = 4 `all` | gain | h = 4 `first` | gain |
|---|---|---|---|---|---|---|---|---|
| 10 | 73.81 | -0.24 | 73.91 | -0.15 | 64.78 | +0.01 | 64.83 | +0.05 |
| 20 | 73.97 | -0.08 | 74.00 | -0.06 | 64.86 | +0.09 | 64.89 | +0.11 |
| 50 | 74.25 | +0.20 | 74.49 | +0.44 | 65.49 | +0.71 | 65.89 | +1.11 |
| 100 | 76.68 | +2.63 | 77.30 | +3.24 | 69.29 | +4.52 | 68.48 | +3.70 |
| 200 | 80.05 | +6.00 | 79.97 | +5.92 | 73.29 | +8.51 | 71.41 | +6.64 |
| 1000 | 85.70 | +11.65 | 85.00 | +10.95 | 80.29 | +15.52 | 76.55 | +11.77 |

At N <= 50 the best cell is the smallest step (lr 1e-4, 10-100 steps): fine-tuning on so few
images barely helps and more steps hurt. The h = 8 `all` numbers land close to the quick numpy
check (+0 / +2.6 / +5.4 / +10.9 pp at N <= 50 / 100 / 200 / 1000): within 0.3 pp up to N = 100,
and 0.6-0.8 pp higher at N = 200 and 1000.

**Online methods at matching burst positions** (online minus frozen at those positions, pp, from
the saved per-position curves; at position p a method has seen p - 1 images of the burst):

| positions | oracle N | oracle `all` h = 8 / h = 4 | Meta-SGD (3) short | Meta-SGD (3) L1000 | dual_shared (5b) L1000 |
|---|---|---|---|---|---|
| 11-50 | 10-50 | -0.2 to +0.2 / 0.0 to +0.7 | +2.59 / +2.17 | +1.66 / +1.94 | -13.44 / -12.25 |
| 51-200 | 100-200 | +2.6 to +6.0 / +4.5 to +8.5 | +3.33 / +3.00 | +3.43 / +4.18 | -1.38 / +1.57 |
| 201-1000 | 200-1000 | +6.0 to +11.7 / +8.5 to +15.5 | | +3.65 / +3.86 | +7.29 / +10.35 |

- **Meta-SGD beats the oracle on short bursts.** At positions 11-50 it gains +2.2 to +2.6 pp,
  where fine-tuning the mixed model on 10-50 images gains nothing. So the oracle bounds
  fine-tuning *from the mixed model*, not meta-learning: Meta-SGD's starting weights S and
  per-parameter rates are trained for fast adaptation, which is worth more than the best fixed
  grid cell from the mixed start.
- **Past ~100 images, the oracle pulls away from Meta-SGD**, whose gain stays flat at +3.4 to
  +4.2 pp; dual_shared on L1000 (+7.3 / +10.4 pp at 201-1000) sits between the oracle at N = 200
  and N = 1000, i.e. roughly at the oracle for the number of images it has seen.
- Even the oracle at N = 1000 stays 7.6 pp (h = 8) and 7.0 pp (h = 4) below the per-context
  models: bursts of this length can't reach the 60k-image ceiling.

**Equivalent width** (for the paper): the width of a frozen mixed model with the same accuracy,
interpolated linearly between the mixed models of width 4 (64.78%), 8 (74.05), 12 (82.72), 16
(83.80), 20 (87.46), 24 (89.19) and 40 (92.82) from offline.py and 5d. A method's accuracy is the
mixed model's test accuracy plus its gain on the stream (the `short` stream's frozen accuracy is
73.25 / 64.41%, not the test mean). Best config of each experiment by gain:

| experiment | h = 8 short | h = 8 L1000 | h = 4 short | h = 4 L1000 |
|---|---|---|---|---|
| 1 (`sgd`) | 8.1 | 8.3 | 4.0 | 4.7 |
| 2 (`dual`) | 8.1 | 9.9 | 4.0 | 6.6 |
| 3 (`metasgd`) | 9.2 | 9.6 | 5.0 | 5.6 |
| 4 (`metasgd_reset` short, `dual_..._reset` L1000) | 9.3 | 10.4 | 5.2 | 7.1 |
| 5b (`metasgd_shared` short, `dual_shared` L1000) | 8.6 | 10.3 | 4.6 | 7.4 |
| 5c (`metasgd_general` short, `dual_general` L1000) | 9.0 | 10.0 | 5.1 | 6.8 |
| oracle `all`, N = 50 / 200 / 1000 | 8.1 / 10.8 / 18.1 | | 4.3 / 7.7 / 10.9 | |

The best online method is worth at most 1.3 (h = 8) or 1.2 (h = 4) extra hidden units on short
bursts and 2.4 / 3.4 on L1000 (15-30% more width, 85% at h = 4 on L1000). A mixed model 1.5x
wider (h = 12, 82.72%) beats every online result at h = 8. The configs are chosen on the stream they are scored on, so
these are optimistic too.

## Part 2: deeper architectures with a FiLM context vector (`part2.py`, `tables.py`)

**Architectures** (`common.Net`, `part2.ARCHS`), sized to the budgets of the current MLP (6370
parameters at h = 8, 3190 at h = 4), FiLM parameters included:

| arch | budget 8 | budget 4 |
|---|---|---|
| `mlp2` (2 hidden layers) | 784 -> 7 -> 24 -> 10 | 784 -> 3 -> 20 -> 10 |
| `mlp3` (3 hidden layers) | 784 -> 7 -> 12 -> 12 -> 10 | 784 -> 3 -> 12 -> 12 -> 10 |
| `cnn` (3x3 conv + ReLU + 2x2 max pool, twice, then linear) | 8, 10 channels | 6, 5 channels |

**FiLM.** On every hidden layer (unit or channel), before the ReLU: h <- h * (1 + A_l c) + B_l c,
with A_l, B_l a bias-free linear map from c (r numbers) to that layer's units. So c = 0 gives
gamma = 1 and beta = 0 for any FiLM weights. A_l, B_l start at N(0, 1/r) so that c gets a
gradient from c = 0. The FiLM weights count in the budget (2 r per hidden unit or channel); the
per-context c_k are not stored.

**Measures**, for each architecture and budget:
1. `mixed`: trained like offline.py with c = 0 (no FiLM; 50 epochs, Adam 1e-3, batch 128,
   seed 0), frozen, averaged over the 8 contexts.
2. `per_context`: one model of the same architecture per rotation, from scratch, same settings.
3. `joint` (r in {1, 2, 4, 8}): from the mixed model, shared weights + FiLM + one c_k per
   context trained together on loss_ctx (c = c_k) + loss_gen (c = 0), 10 epochs as in 5c.
   `S` = c = 0, `S + c_k` = each context's own c_k.
4. `oracle_all`: the Part 1 oracle on every parameter, from the mixed model; `oracle_c`: only
   c (from 0), from the joint model. For c, the grid also has lr 1e-2, 3e-2 and 1e-1, because
   the spec rates move c by at most 0.9 in 300 steps (`oracle_c_ext`; it changed nothing: the
   learned c_k have norms of 0.05-0.33).

The MLPs train on the CPU, rotating each batch as offline.py does. The CNN trains on the GPU
(MPS) from the training set rotated once under every context, and its 8 per-context models are
trained at once (vmap); same hyperparameters. The MLPs took about 15 minutes each and the CNN about 2 hours per budget, all in parallel.

## Results

`mlp1` is the current 784 -> h -> 10 (its per-context value is the 93.26 / 87.31% reference;
its oracle rows are Part 1). Joint columns: S alone (c = 0) / S + c_k. c-only oracle: the best r
per N, on the extended learning-rate grid (it equals the spec grid within 0.05 pp everywhere).

### Budget 8 (accuracy %, gains in pp over the architecture's mixed model)

| arch | params (r = 0 / 8) | mixed | per-context | gap | S / S+c_k r=1 | S / S+c_k r=2 | S / S+c_k r=4 | S / S+c_k r=8 |
|---|---|---|---|---|---|---|---|---|
| mlp1 | 6370 /  | 74.05 | 93.26 | +19.21 |  |  |  |  |
| mlp2 | 5937 / 6433 | 74.58 | 92.57 | +17.99 | 74.07 / 76.01 | 74.03 / 76.20 | 73.92 / 77.05 | 73.69 / 77.36 |
| mlp3 | 5877 / 6373 | 78.55 | 92.65 | +14.10 | 78.50 / 80.12 | 78.49 / 80.32 | 78.52 / 80.89 | 78.37 / 81.19 |
| cnn | 5720 / 6008 | 95.66 | 98.54 | +2.88 | 95.88 / 96.09 | 95.84 / 96.38 | 95.86 / 96.54 | 95.86 / 96.59 |

Few-shot oracle, gain over mixed (pp); c-only: best r (lr up to 1e-1)

| arch | oracle | N=10 | N=20 | N=50 | N=100 | N=200 | N=1000 |
|---|---|---|---|---|---|---|---|
| mlp1 | all | -0.24 | -0.08 | +0.20 | +2.63 | +6.00 | +11.65 |
| mlp1 | first layer | -0.14 | -0.05 | +0.44 | +3.25 | +5.92 | +10.95 |
| mlp2 | all | -0.04 | +0.05 | +0.15 | +1.44 | +4.43 | +10.10 |
| mlp2 | c only | -0.02 (r=1) | +0.96 (r=1) | +1.57 (r=8) | +2.04 (r=8) | +2.49 (r=8) | +3.03 (r=8) |
| mlp3 | all | -0.11 | -0.05 | +0.14 | +1.02 | +3.02 | +7.71 |
| mlp3 | c only | +0.67 (r=1) | +1.06 (r=1) | +1.26 (r=1) | +1.82 (r=8) | +2.23 (r=8) | +2.67 (r=8) |
| cnn | all | +0.05 | +0.07 | +0.14 | +0.53 | +0.91 | +1.89 |
| cnn | c only | +0.35 (r=8) | +0.46 (r=8) | +0.57 (r=8) | +0.63 (r=8) | +0.71 (r=8) | +0.93 (r=8) |

### Budget 4 (accuracy %, gains in pp over the architecture's mixed model)

| arch | params (r = 0 / 8) | mixed | per-context | gap | S / S+c_k r=1 | S / S+c_k r=2 | S / S+c_k r=4 | S / S+c_k r=8 |
|---|---|---|---|---|---|---|---|---|
| mlp1 | 3190 /  | 64.78 | 87.31 | +22.53 |  |  |  |  |
| mlp2 | 2645 / 3013 | 63.62 | 82.76 | +19.14 | 63.53 / 64.57 | 63.65 / 64.80 | 63.54 / 65.05 | 63.66 / 65.51 |
| mlp3 | 2689 / 3121 | 63.42 | 81.68 | +18.26 | 63.29 / 64.03 | 63.30 / 64.37 | 63.33 / 64.59 | 63.42 / 64.79 |
| cnn | 2795 / 2971 | 92.50 | 98.15 | +5.65 | 92.88 / 93.19 | 92.80 / 93.43 | 92.81 / 93.50 | 92.82 / 93.67 |

Few-shot oracle, gain over mixed (pp); c-only: best r (lr up to 1e-1)

| arch | oracle | N=10 | N=20 | N=50 | N=100 | N=200 | N=1000 |
|---|---|---|---|---|---|---|---|
| mlp1 | all | +0.00 | +0.08 | +0.71 | +4.51 | +8.51 | +15.51 |
| mlp1 | first layer | +0.05 | +0.11 | +1.11 | +3.70 | +6.63 | +11.77 |
| mlp2 | all | -0.17 | -0.04 | +0.11 | +1.71 | +4.92 | +11.82 |
| mlp2 | c only | +0.34 (r=4) | +0.60 (r=2) | +1.04 (r=8) | +1.36 (r=8) | +1.81 (r=8) | +2.20 (r=8) |
| mlp3 | all | -0.08 | -0.01 | +0.07 | +1.27 | +4.95 | +11.41 |
| mlp3 | c only | +0.36 (r=2) | +0.46 (r=2) | +0.71 (r=4) | +0.85 (r=4) | +1.07 (r=4) | +1.50 (r=8) |
| cnn | all | -0.08 | +0.08 | +0.14 | +0.83 | +1.79 | +3.70 |
| cnn | c only | +0.43 (r=8) | +0.47 (r=2) | +0.70 (r=2) | +0.81 (r=2) | +0.90 (r=8) | +1.18 (r=8) |

### Decision

| budget | arch | gap (pp) | c-only gain at N = 50 (pp) | passes? |
|---|---|---|---|---|
| 8 | `mlp2` | 17.99 | +1.57 | no |
| 8 | `mlp3` | 14.10 | +1.26 | no |
| 8 | `cnn` | 2.88 | +0.57 | no |
| 4 | `mlp2` | 19.14 | +1.04 | no |
| 4 | `mlp3` | 18.26 | +0.71 | no |
| 4 | `cnn` | 5.65 | +0.70 | no |

**No architecture passes, so no online runs. Conclusion: rotation is the wrong kind of locality
at this budget; switch.**

## Findings

1. **Depth plus a context vector does not make rotation low-dimensional in an MLP.** The deeper
   MLPs keep a large gap (14-19 pp), but even c_k fitted on the whole training set (`S + c_k`)
   recovers only 1.4-3.7 pp of it, growing slowly with r. Fitted on 50 images, c gains 0.7-1.6
   pp. The c-only oracle does beat fine-tuning every parameter at N <= 50 (+0.1 to +0.2 pp), so a
   few context numbers are the better thing to adapt early, but the ceiling is far too low.
   The learned c_k stay small (norm 0.05-0.33) and larger learning rates for c change nothing.
2. **The CNN mostly removes the problem instead.** At the same budget, one frozen CNN scores
   95.66% (budget 8) and 92.50% (budget 4) on every rotation, above the current MLP's
   per-context ceiling (93.26 / 87.31%). What is left to gain from knowing the rotation is
   2.9 / 5.7 pp, and c recovers about a sixth of it (+0.6 / +0.7 pp at N = 50, at most +0.9 /
   +1.2 at r = 8 on the full set). Fine-tuning every CNN parameter on 1000 images gains only
   +1.9 / +3.7 pp.
3. So the reading "rotation is high-dimensional only because the shallow MLP can store it only in
   its first layer" is half right. Depth alone doesn't fix it (`mlp2`, `mlp3`), but an
   architecture with spatial weight sharing absorbs most of the rotation into one model, which
   leaves too little context-specific signal for online adaptation to be worth it. Either way,
   rotation at this budget does not give a context that is both large and low-dimensional.

Caveats: one seed for every training run; FiLM on every hidden layer with one initialization
(N(0, 1/r)) only; joint pretraining kept 5c's 10 epochs.

## Files and commands (from experiments_MNIST2)

```
../venv/bin/python experiment6_fewshot_ceiling/part1_mlp.py       # about a minute
../venv/bin/python experiment6_fewshot_ceiling/compare_online.py  # part 1 comparisons, no training
THREADS=2 DEVICE=cpu ../venv/bin/python experiment6_fewshot_ceiling/part2.py 8 mlp2   # also mlp3; budget 4
THREADS=1 DEVICE=mps ../venv/bin/python experiment6_fewshot_ceiling/part2.py 8 cnn
../venv/bin/python experiment6_fewshot_ceiling/tables.py          # part 2 tables and the decision
```

`results.csv` (long format): `part`, `budget` (h for part 1), `arch`, `r`, `params`, `measure`
(`mixed`, `per_context`, `joint_S`, `joint_Sc`, `oracle_all`, `oracle_first`, `oracle_c`,
`oracle_c_ext`), `N`, `acc` (%), `gain` (pp over the architecture's mixed model), and the best
cell's `lr` and `steps`. `grids/` holds every grid cell, `weights/` the trained models (reused
on a rerun), `logs/` the run logs.
