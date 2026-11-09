# E2: which weights hold the memory? (claim C5)

**Question:** when only part of Pythia-14M is updated online, how much of the long-range gain is
kept, and which layers' MLPs can hold it?

## Setup

- Main set, K = 1 (as E1). Plain SGD, one step per chunk, on one parameter subset: all
  parameters; output embedding only; all but the output embedding; the MLPs of all six layers;
  the attention of all six layers; one layer's MLP (layers 1-6, 131,712 parameters each).
- Each subset at its own val-pick learning rate (E0). Gain over frozen 14M per distance bin (E1's
  bins), SE over books; "- full" = paired difference with the all-parameter run.
  `results.csv` (all bins).
- **MLP only against all parameters** (`mlp_vs_full.csv`): both runs also on the shuffled stream,
  both with snapshots every 40 chunks (as E5). Locality = in order - shuffled, paired per token;
  forgetting = snapshot - pretrained loss on detokenized WikiText (76 chunks) and the 6 held-out
  books, last-half mean and final snapshot, SE over WikiText chunks or held-out books; exchange
  rate = stream gain / last-half WikiText forgetting. "MLP - all" is paired (same tokens, or same
  held-out units at the same snapshot positions).

## Results (gain over frozen, ± SE over books)

| subset | lr | all tokens | earlier chunk < 1k | 1k-10k, same book | never seen |
|---|---|---|---|---|---|
| all parameters | 1e-4 | +0.100 ± 0.006 | +0.307 ± 0.011 | +0.154 ± 0.008 | -0.059 ± 0.013 |
| output only | 3e-3 | +0.072 ± 0.006 | +0.252 ± 0.017 | +0.072 ± 0.006 | -0.143 ± 0.012 |
| all but output | 1e-4 | +0.094 ± 0.005 | +0.294 ± 0.011 | +0.151 ± 0.009 | -0.058 ± 0.013 |
| MLP only | 3e-4 | +0.107 ± 0.007 | +0.415 ± 0.016 | +0.175 ± 0.011 | -0.077 ± 0.014 |
| attention only | 3e-4 | +0.066 ± 0.005 | +0.259 ± 0.011 | +0.110 ± 0.009 | -0.074 ± 0.014 |
| MLP of layer 1 | 1e-3 | +0.053 ± 0.004 | +0.236 ± 0.012 | +0.113 ± 0.007 | -0.055 ± 0.010 |
| MLP of layer 2 | 1e-3 | +0.069 ± 0.004 | +0.242 ± 0.011 | +0.131 ± 0.008 | -0.051 ± 0.010 |
| MLP of layer 3 | 3e-4 | +0.055 ± 0.004 | +0.204 ± 0.009 | +0.093 ± 0.007 | -0.056 ± 0.011 |
| MLP of layer 4 | 1e-3 | +0.063 ± 0.004 | +0.187 ± 0.010 | +0.101 ± 0.007 | -0.063 ± 0.010 |
| MLP of layer 5 | 3e-3 | +0.073 ± 0.006 | +0.285 ± 0.014 | +0.121 ± 0.009 | -0.090 ± 0.014 |
| MLP of layer 6 | 1e-3 | +0.048 ± 0.004 | +0.111 ± 0.008 | +0.054 ± 0.007 | -0.084 ± 0.012 |

Paired differences with the all-parameter run (`minus_full`): MLP only +0.007 ± 0.002 overall,
+0.108 ± 0.008 within 1k, +0.021 ± 0.004 at 1k-10k same book; all but output -0.006 ± 0.001,
-0.013 ± 0.002, -0.004 ± 0.001; output only -0.028 ± 0.003, -0.055 ± 0.018, -0.082 ± 0.010.

### MLP only (lr 3e-4) against all parameters (lr 1e-4)

| quantity | bin / stream | MLP only | all parameters | MLP only - all |
|---|---|---|---|---|
| gain in order | all tokens | +0.107 ± 0.007 | +0.100 ± 0.006 | +0.007 ± 0.002 |
|  | earlier, < 1k | +0.415 ± 0.016 | +0.307 ± 0.011 | +0.108 ± 0.008 |
|  | 1k-10k, same book | +0.175 ± 0.011 | +0.154 ± 0.008 | +0.021 ± 0.004 |
|  | never seen | -0.077 ± 0.014 | -0.059 ± 0.013 | -0.018 ± 0.004 |
| gain shuffled | all tokens | +0.059 ± 0.003 | +0.062 ± 0.003 | -0.003 ± 0.001 |
|  | earlier, < 1k | +0.081 ± 0.011 | +0.062 ± 0.007 | +0.019 ± 0.005 |
|  | 1k-10k, same book | +0.030 ± 0.006 | +0.034 ± 0.005 | -0.004 ± 0.002* |
|  | never seen | -0.071 ± 0.010 | -0.055 ± 0.009 | -0.016 ± 0.003 |
| locality | all tokens | +0.048 ± 0.004 | +0.038 ± 0.004 | +0.010 ± 0.001 |
|  | earlier, < 1k | +0.334 ± 0.011 | +0.245 ± 0.009 | +0.088 ± 0.005 |
|  | 1k-10k, same book | +0.145 ± 0.009 | +0.120 ± 0.007 | +0.025 ± 0.003 |
|  | never seen | -0.006 ± 0.011* | -0.004 ± 0.010* | -0.002 ± 0.004* |
| WikiText forgetting, last half | K = 1 | +0.235 ± 0.007 | +0.196 ± 0.006 | +0.039 ± 0.003 |
|  | shuffled | +0.144 ± 0.005 | +0.112 ± 0.004 | +0.031 ± 0.002 |
| WikiText forgetting, final | K = 1 | +0.306 ± 0.008 | +0.252 ± 0.007 | +0.054 ± 0.003 |
|  | shuffled | +0.134 ± 0.006 | +0.102 ± 0.005 | +0.032 ± 0.002 |
| held-out books forgetting, last half | K = 1 | +0.016 ± 0.009* | -0.006 ± 0.010* | +0.022 ± 0.002 |
|  | shuffled | -0.034 ± 0.007 | -0.045 ± 0.008 | +0.011 ± 0.001 |
| held-out books forgetting, final | K = 1 | +0.020 ± 0.009 | -0.002 ± 0.009* | +0.022 ± 0.002 |
|  | shuffled | -0.042 ± 0.008 | -0.052 ± 0.008 | +0.010 ± 0.001 |
| exchange rate | K = 1 | 0.45 | 0.51 | -0.06 |
|  | shuffled | 0.41 | 0.55 | -0.14 |

\* under 2 SE (the 1k-10k shuffled difference is -0.0041 ± 0.0021, just under).

![gain vs layer](plots/gain_vs_layer.png)

## Findings

- **The memory is in the MLPs below the output layer.** Updating only the six MLPs (0.79M
  parameters) at their own rate gives more than the full learner at every distance up to 10k
  (+0.415 against +0.307 within 1k, +0.175 against +0.154 at 1k-10k same book, +0.107 against
  +0.100 overall, all beyond 2 SE paired). Freezing the output embedding costs little (-0.004 at
  1k-10k same book); the output embedding alone is a short-range cache (+0.252 within 1k, +0.072 at
  1k-10k same book) that loses the most on never-seen tokens (-0.143).
- **MLP only has more locality:** higher than
  the full learner's at every distance up to 10k (+0.334 against +0.245 within 1k, paired +0.088 ±
  0.005; +0.145 against +0.120 at 1k-10k same book, +0.025 ± 0.003), while shuffled it gains slightly
  less overall (-0.003 ± 0.001).
- **It costs more general knowledge:** last-half WikiText forgetting +0.235 against +0.196 (paired
  +0.039 ± 0.003) on K = 1, and +0.031 ± 0.002 more on shuffled. On the held-out books it loses
  +0.022 ± 0.002 more than the full learner (paired), though on K = 1 neither run's own last-half value is
  beyond 2 SE (+0.016 ± 0.009 and -0.006 ± 0.010); at the final snapshot MLP only is at +0.020 ±
  0.009. Its exchange rate is lower: 0.45 against 0.51 (K = 1),
  0.41 against 0.55 (shuffled). It also loses more on never-seen tokens (-0.018 ± 0.004).
- Attention only keeps less: +0.259 within 1k and +0.110 at 1k-10k same book.
- **Single layers:** each of the six MLPs alone keeps 36-93% of the full learner's within-1k gain
  and 35-85% at 1k-10k same book, and every one is below the full learner at both. Layers 2 and 5
  are the strongest (+0.131 and +0.121 at 1k-10k same book); the last layer is clearly the weakest
  (+0.111 within 1k, +0.054 at 1k-10k same book).

## Caveats

- MLP only and all parameters use different learning rates, each its own val-pick (3e-4 and 1e-4;
  val-pick loss 4.2506 and 4.2565). A possible reason the full learner cannot use the larger step,
  not tested: with all parameters, 3e-4 is already worse on val-pick (4.2732), and attention takes a
  large share of the plain-SGD update at 14M (E7 diagnostics, `diagnostics_summary.csv`: 43% of the
  squared update norm for attention against 50% for the MLPs, gradient norms 26.9 and 28.6), which
  may limit the usable step. Per weight, though, attention moves far less (relative update 4.1e-6
  against 1.2e-4 for the MLPs).
- One model, one seed. Rates are picked on overall val-pick loss, not on long-range gain; the
  single-layer gains within 1k depend strongly on the rate (old test 6's picks on the check books,
  3e-3 for layers 2 and 4, gave +0.395 and +0.357 within 1k, with lower gains on never-seen tokens).
- The "1k-10k" column is same-book tokens only (the bulk of that bin; all tokens in `results.csv`).

Runtime: about 2 min per subset run without snapshots; MLP only with snapshots 18 min in order and
18 min shuffled (`runtime.txt`).
