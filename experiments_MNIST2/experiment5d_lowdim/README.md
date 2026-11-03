# Experiment 5d (low-dim): is 5c's context part low-dimensional, and is that worth its parameters?

**Question.** In experiment 5c each context's first layer is S1 + D_k, and the D_k are large
(56-62% of S1's norm). If the 8 D_k lie close to a few shared directions B_1..B_r
(D_k ~ sum_i c_ki B_i), online learning would only need to find r numbers per context instead of a
whole first layer. Is that so? And is a model with r directions better than a plain mixed model
with the same number of parameters? This is an offline diagnostic, with no online runs.

**Part 1** (`svd.py`). For each h, each D_k (0.weight and 0.bias) is flattened into one row of
785 h numbers. The 8 rows are stacked and decomposed by SVD, without centering. For r in
{1, 2, 4, 8}, B = the top r right singular vectors, fixed. For each context k, c_k starts at the
projection of D_k onto B. Only c_k (r numbers) is fine-tuned, on the training set under context k,
with first layer S1 + sum_i c_ki B_i; S1, B and the head stay fixed (Adam lr 1e-2, batch 128,
2 epochs, seed 0). The result is scored on the test set under context k and averaged over the 8
contexts. r = 0 is S1 alone. As a check, at r = 8 the projection alone reproduces 5c's ceiling
exactly (89.86 / 84.27%); fine-tuning adds 0.2 pp.

**Part 2** (`matched.py`). A model with r directions stores S1, the head and B:
(795 h + 10) + r * 785 h parameters. For each base h and r in {1, 2, 4}, H is the hidden size
whose plain MLP (795 H + 10 parameters) is closest to that total. A mixed model of width H is
trained exactly as `offline.py` does (its `train`, 50 epochs, seed 0) and scored frozen on the
test set, averaged over the contexts. `weights/mixed_h8.pt` is reused for H = 8 (h = 4, r = 1),
and r = 0 is the mixed model of width h itself. The new models are in `weights/mixed_h<H>.pt`.
r = 8 has no matched model: that was not asked for, and r = 4 already settles the question.

The rank-r ceiling knows the context and has a per-context c_k trained on the full training set.
The matched model is one frozen model for every context. So the margin is the most that an online
method using this low-dimensional context could gain under a fair parameter budget.

## Results

Base h = 8:

| r | explained | rank-r ceiling (%) | total params | matched H | matched mixed (%) | margin (pp) |
|---|---|---|---|---|---|---|
| 0 | 0 | 73.70 | 6370 | 8 | 74.05 | -0.35 |
| 1 | 0.27 | 78.85 | 12650 | 16 | 83.80 | -4.95 |
| 2 | 0.47 | 82.05 | 18930 | 24 | 89.19 | -7.14 |
| 4 | 0.73 | 87.78 | 31490 | 40 | 92.82 | -5.04 |
| 8 | 1.00 | 90.04 | 56610 | | | |

Base h = 4:

| r | explained | rank-r ceiling (%) | total params | matched H | matched mixed (%) | margin (pp) |
|---|---|---|---|---|---|---|
| 0 | 0 | 64.83 | 3190 | 4 | 64.78 | +0.05 |
| 1 | 0.28 | 69.76 | 6330 | 8 | 74.05 | -4.29 |
| 2 | 0.45 | 72.73 | 9470 | 12 | 82.72 | -9.99 |
| 4 | 0.72 | 81.75 | 15750 | 20 | 87.46 | -5.71 |
| 8 | 1.00 | 84.46 | 28310 | | | |

Singular values: 8.41 7.11 6.13 5.64 4.98 4.28 3.87 3.33 (h = 8) and 6.93 5.55 5.07 4.65 4.07 3.46
3.35 2.95 (h = 4).

## Findings

1. **The D_k are not low-dimensional.** The top direction explains 27-28% of the squared norm
   (1/8 = 12.5% for unrelated directions), 2 explain 45-47%, and 4 explain 72-73%. The singular
   values decay slowly. The ceiling follows: r = 4 recovers 85% (h = 8) and 87% (h = 4) of the gap
   from S1 alone to r = 8, while r = 1 and 2 recover only 31-51%.
2. **No r beats the matched-size mixed model; every r loses by 4-10 pp.** Spending the same
   parameters on a wider plain mixed model is clearly better. That model is frozen and gets no
   context, yet H = 40 (92.82%) beats even the full 8-direction ceiling at h = 8 (90.04%), with
   about half the parameters. Under a fair parameter budget, a method using this low-dimensional
   context has no headroom to gain.
3. **Recommendation: do not build a low-dimensional 5d on these directions.** If the parameter
   budget can grow, a wider mixed model is the better use of it. The low-dimensional idea is only
   worth revisiting if B is learned jointly during pretraining.

Caveat: the directions come from 5c's D_k after the fact. A 5d that learns B during pretraining
could do better, so these ceilings are a lower bound. One seed; matched models trained once each.
