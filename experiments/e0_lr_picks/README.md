# E0: learning-rate picks

**Question:** which learning rate does each run of the registry use?

## Setup

- **Pick stream:** val-pick, the first 10 PG-19 validation books with at least 20k tokens, 78
  chunks each, read book after book (K = 1, 780 chunks). No test book is used for tuning.
- **Rule:** lowest mean online loss on a 1-3-10 ladder. Start grids: plain SGD 14M {3e-5, 1e-4,
  3e-4}, 31M {1e-4, 3e-4, 1e-3}, 70M {3e-6, 1e-5, 3e-5}; Adam {1e-6 ... 1e-4}; normalized SGD
  {1e-5 ... 1e-3}; parameter subsets {1e-4, 3e-4, 1e-3}. The grid is extended one step at a time
  until both ladder neighbours of the best value have been tried. Every pick is interior.
- Output: `picks.csv` (every rate tried, `picked` column), read by `runs.py`; `rows_<group>.csv`
  per process.

## Picks (val-pick mean loss at the pick)

| size | rule | parameters | pick | rates tried |
|---|---|---|---|---|
| 14M | sgd | all | 1e-4 | 3e-5 ... 3e-4 |
| 14M | adam | all | 3e-6 | 1e-6 ... 1e-4 |
| 14M | nsgd | all | 1e-4 | 1e-5 ... 1e-3 |
| 31M | sgd | all | 3e-4 | 1e-4 ... 1e-3 |
| 31M | adam | all | 3e-6 | 1e-6 ... 1e-4 |
| 31M | nsgd | all | 1e-4 | 1e-5 ... 1e-3 |
| 70M | sgd | all | 1e-5 | 3e-6 ... 3e-5 |
| 70M | adam | all | 3e-7 | 1e-7 ... 1e-4 |
| 70M | nsgd | all | 3e-6 | 1e-6 ... 1e-3 |
| 14M | sgd | output only | 3e-3 | 1e-4 ... 1e-2 |
| 14M | sgd | all but output | 1e-4 | 3e-5 ... 1e-3 |
| 14M | sgd | MLP only | 3e-4 | 1e-4 ... 1e-3 |
| 14M | sgd | attention only | 3e-4 | 1e-4 ... 1e-3 |
| 14M | sgd | MLP of layer 1 / 2 / 3 / 4 / 5 / 6 | 1e-3 / 1e-3 / 3e-4 / 1e-3 / 3e-3 / 1e-3 | 1e-4 ... 3e-3 (layer 5: 1e-2) |

## Findings

- The 9 (size, rule) picks equal old test 13's, with the same val-pick losses to 4 decimals. The
  two old edge picks are now confirmed interior: 70M Adam 3e-7 (1e-7: 3.7379 > 3.7281) and 70M
  normalized SGD 3e-6 (1e-6: 3.7222 > 3.7056).
- Subset picks moved from the old check-book picks: MLP only and attention only 1e-4 -> 3e-4,
  layer-2 and layer-4 MLP 3e-3 -> 1e-3. Output only stays at 3e-3 (now interior: 1e-2 is much
  worse), layer 5 stays 3e-3.

## Caveats

- One pick stream (10 books) and one seed; neighbouring rates are often within 0.01-0.02 nats.

Runtime: 137 min of process time in four groups, run as 3 parallel processes (`rows_*.csv`, `seconds`).
