Saved output of a one-off comparison against the deleted experiments_LLM2_cache/ (in git history before commit 1ba0c5e); the script was removed.

# Regression: old (experiments_LLM2_cache/) against new (experiments/)

Full table also in `regression.csv`. Kinds: **exact** = same run as before, must match to float16 precision (|diff| < 1e-3; exchange rates < 0.01; half-lives < 1 token); **expected shift** = changed chunking / order (E1, E2 against tests 1-6) or changed learning-rate pick. Shifts are flagged when |diff| > 2 SE (SE of the new value, over books) or the sign changes; **grid change** = the same run summarized on the new snapshot grid (every 40 chunks instead of test 13's every 80), for information.

**Summary:** 293 numbers compared, 211 exact (0 mismatches), 82 expected shifts (10 flagged). Per-token identity: 49 arrays compared, 0 not bit-identical in float16.

## Wording changes

All flagged shifts are in C5 and come from the new subset learning-rate picks (val-pick instead of the old check books; MLP only and attention only 1e-4 -> 3e-4, layer-2 and layer-4 MLP 3e-3 -> 1e-3).

- **C5, MLP only:** was "keeps most of the long-range gain (+0.216 within 1k, +0.123 at 1k-10k)"; now it **beats** the full learner: +0.415 against +0.308 within 1k, +0.158 against +0.144 at 1k-10k, +0.107 against +0.100 overall (paired differences beyond 2 SE).
- **C5, attention only:** +0.142 / +0.092 -> +0.259 / +0.102 (still below MLP only and the full learner).
- **C5, single layers:** was "the MLP of layer 2 or 4 alone gains +0.395 / +0.357 within 1k, more than the full learner" (at lr 3e-3, grid edge); now +0.242 / +0.187, and no single layer beats the full learner (best: layer 5, +0.285). "Every MLP but the last holds a large part; the last is the weakest" still holds. "Several picks at the edge of the grid" no longer applies (all interior). Layer 2's overall gain and never-seen loss: +0.055 / -0.129 -> +0.069 / -0.051. "Layer 4 level with the full learner on recent same-book tokens" no longer holds (-0.053 ± 0.004 at 1k-10k same book).
- **C2, numbers only:** share of the gain surviving shuffling 18% -> 20% within 1k, 27% -> 24% at 1k-10k, 61% -> 62% overall (each shift under 2 SE).
- **C10, numbers only:** exchange rates on the new snapshot grid (every 40 chunks) move by 0.01, and 70M sgd 2.68 -> 2.57; the conclusions are unchanged.
- C1, C3, C4, C6, C7, C8, C9: no wording change (C6-C9 are the same runs, exact).

## Per-token identity of the exact runs

| arrays | identical share | max abs diff |
|---|---|---|
| frozen 14m main vs test 7 | 1.0000 | 0.00e+00 |
| frozen 31m main vs test 7 | 1.0000 | 0.00e+00 |
| frozen 70m main vs test 7 | 1.0000 | 0.00e+00 |
| frozen 14m long vs test 7 | 1.0000 | 0.00e+00 |
| 14m sgd main_K1 vs test 7 full_1e-4_K1 | 1.0000 | 0.00e+00 |
| 14m sgd main_shuffled vs test 7 full_1e-4_shuffled | 1.0000 | 0.00e+00 |
| 14m sgd long vs test 7 full_1e-4_long | 1.0000 | 0.00e+00 |
| E4 probe losses vs test 11 full_1e-4 | 1.0000 | 0.00e+00 |
| 14m sgd main_K1 snapshots vs test 12 (WikiText detok + books) | 1.0000 | 0.00e+00 |
| 14m sgd main_shuffled snapshots vs test 12 (WikiText detok + books) | 1.0000 | 0.00e+00 |
| 14m sgd verylong snapshots vs test 12 (WikiText detok + books) | 1.0000 | 0.00e+00 |
| 14m sgd verylong stream vs test 12 | 1.0000 | 0.00e+00 |
| verylong frozen14m vs test 10C frozen14m | 1.0000 | 0.00e+00 |
| verylong frozen31m vs test 10C frozen31m | 1.0000 | 0.00e+00 |
| verylong frozen70m vs test 10C frozen70m | 1.0000 | 0.00e+00 |
| verylong online14m vs test 10C 14m_full_1e-4 | 1.0000 | 0.00e+00 |
| verylong online31m vs test 10C 31m_online_lr3e-4 | 1.0000 | 0.00e+00 |
| 14m sgd K1 stream vs old | 1.0000 | 0.00e+00 |
| 14m sgd shuffled stream vs old | 1.0000 | 0.00e+00 |
| 14m adam K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 14m adam K1 stream vs old | 1.0000 | 0.00e+00 |
| 14m adam shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 14m adam shuffled stream vs old | 1.0000 | 0.00e+00 |
| 14m nsgd K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 14m nsgd K1 stream vs old | 1.0000 | 0.00e+00 |
| 14m nsgd shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 14m nsgd shuffled stream vs old | 1.0000 | 0.00e+00 |
| 31m sgd K1 stream vs old | 1.0000 | 0.00e+00 |
| 31m sgd shuffled stream vs old | 1.0000 | 0.00e+00 |
| 31m adam K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 31m adam K1 stream vs old | 1.0000 | 0.00e+00 |
| 31m adam shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 31m adam shuffled stream vs old | 1.0000 | 0.00e+00 |
| 31m nsgd K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 31m nsgd K1 stream vs old | 1.0000 | 0.00e+00 |
| 31m nsgd shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 31m nsgd shuffled stream vs old | 1.0000 | 0.00e+00 |
| 70m sgd K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 70m sgd K1 stream vs old | 1.0000 | 0.00e+00 |
| 70m sgd shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 70m sgd shuffled stream vs old | 1.0000 | 0.00e+00 |
| 70m adam K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 70m adam K1 stream vs old | 1.0000 | 0.00e+00 |
| 70m adam shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 70m adam shuffled stream vs old | 1.0000 | 0.00e+00 |
| 70m nsgd K1 snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 70m nsgd K1 stream vs old | 1.0000 | 0.00e+00 |
| 70m nsgd shuffled snapshots vs test 13 (every 80) | 1.0000 | 0.00e+00 |
| 70m nsgd shuffled stream vs old | 1.0000 | 0.00e+00 |

## C1

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| gain in order, all tokens | 0.0993 | 0.0997 | +0.0004 | 0.0057 | expected shift | ok | test 1 |
| gain in order, same chunk | -0.1075 | -0.1105 | -0.0030 | 0.0142 | expected shift | ok | test 1 |
| gain in order, earlier chunk, < 1k | 0.3009 | 0.3075 | +0.0066 | 0.0113 | expected shift | ok | test 1 |
| gain in order, earlier chunk, 1k-10k | 0.1433 | 0.1440 | +0.0007 | 0.0080 | expected shift | ok | test 1 |
| gain in order, earlier chunk, 10k-100k | 0.0097 | 0.0090 | -0.0007 | 0.0061 | expected shift | ok | test 1 |
| gain in order, earlier chunk, >= 100k | -0.0346 | -0.0392 | -0.0046 | 0.0092 | expected shift | ok | test 1 |
| gain in order, never seen | -0.0617 | -0.0586 | +0.0031 | 0.0130 | expected shift | ok | test 1 |

## C2

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| gain shuffled, all tokens | 0.0608 | 0.0619 | +0.0011 | 0.0025 | expected shift | ok | test 4 |
| locality, all tokens | 0.0385 | 0.0378 | -0.0007 | 0.0042 | expected shift | ok | test 4 |
| gain shuffled, earlier chunk, < 1k | 0.0541 | 0.0621 | +0.0080 | 0.0071 | expected shift | ok | test 4 |
| locality, earlier chunk, < 1k | 0.2468 | 0.2453 | -0.0015 | 0.0093 | expected shift | ok | test 4 |
| gain shuffled, earlier chunk, 1k-10k | 0.0382 | 0.0346 | -0.0036 | 0.0049 | expected shift | ok | test 4 |
| locality, earlier chunk, 1k-10k | 0.1051 | 0.1094 | +0.0043 | 0.0065 | expected shift | ok | test 4 |
| gain shuffled, earlier chunk, 10k-100k | 0.0234 | 0.0262 | +0.0028 | 0.0047 | expected shift | ok | test 4 |
| locality, earlier chunk, 10k-100k | -0.0136 | -0.0172 | -0.0036 | 0.0059 | expected shift | ok | test 4 |

## C3

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| locality, earlier chunk, < 1k, same book | 0.2476 | 0.2459 | -0.0017 | 0.0093 | expected shift | ok | test 5 |
| gain in order, earlier chunk, < 1k, same book | 0.3017 | 0.3079 | +0.0062 | 0.0113 | expected shift | ok | test 5 |
| locality, earlier chunk, 1k-10k, same book | 0.1169 | 0.1204 | +0.0035 | 0.0071 | expected shift | ok | test 5 |
| gain in order, earlier chunk, 1k-10k, same book | 0.1548 | 0.1542 | -0.0006 | 0.0084 | expected shift | ok | test 5 |
| locality, earlier chunk, 1k-10k, earlier book | 0.0253 | 0.0353 | +0.0100 | 0.0095 | expected shift | ok | test 5 |
| gain in order, earlier chunk, 1k-10k, earlier book | 0.0653 | 0.0757 | +0.0104 | 0.0116 | expected shift | ok | test 5 |
| locality, earlier chunk, 10k-100k, same book | 0.0380 | 0.0364 | -0.0016 | 0.0096 | expected shift | ok | test 5 |
| gain in order, earlier chunk, 10k-100k, same book | 0.0649 | 0.0603 | -0.0046 | 0.0113 | expected shift | ok | test 5 |
| locality, earlier chunk, 10k-100k, earlier book | -0.0199 | -0.0238 | -0.0039 | 0.0058 | expected shift | ok | test 5 |
| gain in order, earlier chunk, 10k-100k, earlier book | 0.0030 | 0.0027 | -0.0003 | 0.0062 | expected shift | ok | test 5 |

## C4

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| unigram cache gain, all tokens | 0.0173 | 0.0176 | +0.0003 | 0.0035 | expected shift | ok | test 3 |
| online - unigram, all tokens | 0.0820 | 0.0821 | +0.0001 | 0.0041 | expected shift | ok | test 3 |
| unigram cache gain, same chunk | 0.0886 | 0.0956 | +0.0070 | 0.0174 | expected shift | ok | test 3 |
| online - unigram, same chunk | -0.1961 | -0.2061 | -0.0100 | 0.0221 | expected shift | ok | test 3 |
| unigram cache gain, earlier chunk, < 1k | 0.4624 | 0.4614 | -0.0010 | 0.0240 | expected shift | ok | test 3 |
| online - unigram, earlier chunk, < 1k | -0.1615 | -0.1539 | +0.0076 | 0.0257 | expected shift | ok | test 3 |
| unigram cache gain, earlier chunk, 1k-10k | 0.0583 | 0.0589 | +0.0006 | 0.0075 | expected shift | ok | test 3 |
| online - unigram, earlier chunk, 1k-10k | 0.0850 | 0.0852 | +0.0002 | 0.0076 | expected shift | ok | test 3 |
| unigram cache gain, earlier chunk, 10k-100k | -0.0511 | -0.0511 | +0.0000 | 0.0000 | expected shift | ok | test 3 |
| online - unigram, earlier chunk, 10k-100k | 0.0608 | 0.0601 | -0.0007 | 0.0061 | expected shift | ok | test 3 |

## C5

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| output, all tokens | 0.0719 | 0.0723 | +0.0004 | 0.0055 | expected shift | ok | test 4, old lr 0.003, new lr 0.003 |
| output, earlier chunk, < 1k | 0.2511 | 0.2520 | +0.0009 | 0.0171 | expected shift | ok | test 4, old lr 0.003, new lr 0.003 |
| output, earlier chunk, 1k-10k | 0.0655 | 0.0649 | -0.0006 | 0.0051 | expected shift | ok | test 4, old lr 0.003, new lr 0.003 |
| output, never seen | -0.1435 | -0.1433 | +0.0002 | 0.0124 | expected shift | ok | test 4, old lr 0.003, new lr 0.003 |
| no_output, all tokens | 0.0939 | 0.0941 | +0.0002 | 0.0054 | expected shift | ok | test 3, old lr 0.0001, new lr 0.0001 |
| no_output, earlier chunk, < 1k | 0.2896 | 0.2941 | +0.0045 | 0.0113 | expected shift | ok | test 3, old lr 0.0001, new lr 0.0001 |
| no_output, earlier chunk, 1k-10k | 0.1387 | 0.1405 | +0.0018 | 0.0085 | expected shift | ok | test 3, old lr 0.0001, new lr 0.0001 |
| no_output, never seen | -0.0579 | -0.0581 | -0.0002 | 0.0129 | expected shift | ok | test 3, old lr 0.0001, new lr 0.0001 |
| mlp, all tokens | 0.0846 | 0.1067 | +0.0221 | 0.0066 | expected shift | > 2 SE | test 4, old lr 0.0001, new lr 0.0003 |
| mlp, earlier chunk, < 1k | 0.2161 | 0.4152 | +0.1991 | 0.0164 | expected shift | > 2 SE | test 4, old lr 0.0001, new lr 0.0003 |
| mlp, earlier chunk, 1k-10k | 0.1232 | 0.1581 | +0.0349 | 0.0098 | expected shift | > 2 SE | test 4, old lr 0.0001, new lr 0.0003 |
| mlp, never seen | -0.0553 | -0.0769 | -0.0216 | 0.0140 | expected shift | ok | test 4, old lr 0.0001, new lr 0.0003 |
| attention, all tokens | 0.0607 | 0.0663 | +0.0056 | 0.0053 | expected shift | ok | test 4, old lr 0.0001, new lr 0.0003 |
| attention, earlier chunk, < 1k | 0.1420 | 0.2593 | +0.1173 | 0.0108 | expected shift | > 2 SE | test 4, old lr 0.0001, new lr 0.0003 |
| attention, earlier chunk, 1k-10k | 0.0919 | 0.1018 | +0.0099 | 0.0082 | expected shift | ok | test 4, old lr 0.0001, new lr 0.0003 |
| attention, never seen | -0.0546 | -0.0738 | -0.0192 | 0.0138 | expected shift | ok | test 4, old lr 0.0001, new lr 0.0003 |
| mlp@0, all tokens | 0.0518 | 0.0534 | +0.0016 | 0.0042 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@0, earlier chunk, < 1k | 0.2293 | 0.2359 | +0.0066 | 0.0123 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@0, earlier chunk, 1k-10k | 0.1027 | 0.1026 | -0.0001 | 0.0069 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@0, never seen | -0.0536 | -0.0546 | -0.0010 | 0.0103 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@1, all tokens | 0.0551 | 0.0693 | +0.0142 | 0.0044 | expected shift | > 2 SE | test 6, old lr 0.003, new lr 0.001 |
| mlp@1, earlier chunk, < 1k | 0.3948 | 0.2418 | -0.1530 | 0.0109 | expected shift | > 2 SE | test 6, old lr 0.003, new lr 0.001 |
| mlp@1, earlier chunk, 1k-10k | 0.1114 | 0.1210 | +0.0096 | 0.0075 | expected shift | ok | test 6, old lr 0.003, new lr 0.001 |
| mlp@1, never seen | -0.1288 | -0.0507 | +0.0781 | 0.0102 | expected shift | > 2 SE | test 6, old lr 0.003, new lr 0.001 |
| mlp@2, all tokens | 0.0549 | 0.0551 | +0.0002 | 0.0039 | expected shift | ok | test 6, old lr 0.0003, new lr 0.0003 |
| mlp@2, earlier chunk, < 1k | 0.2020 | 0.2041 | +0.0021 | 0.0087 | expected shift | ok | test 6, old lr 0.0003, new lr 0.0003 |
| mlp@2, earlier chunk, 1k-10k | 0.0851 | 0.0860 | +0.0009 | 0.0068 | expected shift | ok | test 6, old lr 0.0003, new lr 0.0003 |
| mlp@2, never seen | -0.0582 | -0.0559 | +0.0023 | 0.0108 | expected shift | ok | test 6, old lr 0.0003, new lr 0.0003 |
| mlp@3, all tokens | 0.0674 | 0.0631 | -0.0043 | 0.0042 | expected shift | ok | test 6, old lr 0.003, new lr 0.001 |
| mlp@3, earlier chunk, < 1k | 0.3567 | 0.1867 | -0.1700 | 0.0097 | expected shift | > 2 SE | test 6, old lr 0.003, new lr 0.001 |
| mlp@3, earlier chunk, 1k-10k | 0.1171 | 0.0952 | -0.0219 | 0.0062 | expected shift | > 2 SE | test 6, old lr 0.003, new lr 0.001 |
| mlp@3, never seen | -0.1164 | -0.0632 | +0.0532 | 0.0103 | expected shift | > 2 SE | test 6, old lr 0.003, new lr 0.001 |
| mlp@4, all tokens | 0.0722 | 0.0732 | +0.0010 | 0.0060 | expected shift | ok | test 6, old lr 0.003, new lr 0.003 |
| mlp@4, earlier chunk, < 1k | 0.2852 | 0.2848 | -0.0004 | 0.0141 | expected shift | ok | test 6, old lr 0.003, new lr 0.003 |
| mlp@4, earlier chunk, 1k-10k | 0.1090 | 0.1111 | +0.0021 | 0.0085 | expected shift | ok | test 6, old lr 0.003, new lr 0.003 |
| mlp@4, never seen | -0.0905 | -0.0898 | +0.0007 | 0.0144 | expected shift | ok | test 6, old lr 0.003, new lr 0.003 |
| mlp@5, all tokens | 0.0472 | 0.0478 | +0.0006 | 0.0042 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@5, earlier chunk, < 1k | 0.1076 | 0.1106 | +0.0030 | 0.0079 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@5, earlier chunk, 1k-10k | 0.0509 | 0.0528 | +0.0019 | 0.0061 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |
| mlp@5, never seen | -0.0843 | -0.0843 | +0.0000 | 0.0118 | expected shift | ok | test 6, old lr 0.001, new lr 0.001 |

## C6

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| gain, long | 0.1760 | 0.1760 | +0.0000 | 0.0228 | exact | ok | test 7 |
| gain, K1 | 0.0997 | 0.0997 | +0.0000 | 0.0057 | exact | ok | test 7 |
| gain, K5 | 0.0740 | 0.0740 | +0.0000 | 0.0040 | exact | ok | test 7 |
| gain, K25 | 0.0646 | 0.0646 | +0.0000 | 0.0026 | exact | ok | test 7 |
| gain, shuffled | 0.0619 | 0.0619 | +0.0000 | 0.0025 | exact | ok | test 7 |

## C7

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| memory_A, I = 0 | 0.0969 | 0.0969 | +0.0000 | 0.0128 | exact | ok | test 11 |
| gain_A2, I = 0 | 0.0673 | 0.0673 | +0.0000 | 0.0073 | exact | ok | test 11 |
| gain_C2, I = 0 | -0.0296 | -0.0296 | +0.0000 | 0.0074 | exact | ok | test 11 |
| memory_rare_in_A1, I = 0 | 0.2229 | 0.2229 | +0.0000 | 0.0288 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 0 | 0.0564 | 0.0564 | +0.0000 | 0.0111 | exact | ok | test 11 |
| memory_A, I = 1024 | 0.0725 | 0.0725 | +0.0000 | 0.0099 | exact | ok | test 11 |
| gain_A2, I = 1024 | 0.0615 | 0.0615 | +0.0000 | 0.0065 | exact | ok | test 11 |
| gain_C2, I = 1024 | -0.0110 | -0.0110 | +0.0000 | 0.0044 | exact | ok | test 11 |
| memory_rare_in_A1, I = 1024 | 0.1640 | 0.1640 | +0.0000 | 0.0215 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 1024 | 0.0377 | 0.0377 | +0.0000 | 0.0098 | exact | ok | test 11 |
| memory_A, I = 3072 | 0.0547 | 0.0547 | +0.0000 | 0.0083 | exact | ok | test 11 |
| gain_A2, I = 3072 | 0.0425 | 0.0425 | +0.0000 | 0.0057 | exact | ok | test 11 |
| gain_C2, I = 3072 | -0.0121 | -0.0121 | +0.0000 | 0.0047 | exact | ok | test 11 |
| memory_rare_in_A1, I = 3072 | 0.1094 | 0.1094 | +0.0000 | 0.0216 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 3072 | 0.0148 | 0.0148 | +0.0000 | 0.0107 | exact | ok | test 11 |
| memory_A, I = 9984 | 0.0440 | 0.0440 | +0.0000 | 0.0069 | exact | ok | test 11 |
| gain_A2, I = 9984 | 0.0234 | 0.0234 | +0.0000 | 0.0054 | exact | ok | test 11 |
| gain_C2, I = 9984 | -0.0206 | -0.0206 | +0.0000 | 0.0039 | exact | ok | test 11 |
| memory_rare_in_A1, I = 9984 | 0.0800 | 0.0800 | +0.0000 | 0.0219 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 9984 | 0.0243 | 0.0243 | +0.0000 | 0.0093 | exact | ok | test 11 |
| memory_A, I = 29952 | 0.0424 | 0.0424 | +0.0000 | 0.0060 | exact | ok | test 11 |
| gain_A2, I = 29952 | 0.0101 | 0.0101 | +0.0000 | 0.0053 | exact | ok | test 11 |
| gain_C2, I = 29952 | -0.0324 | -0.0324 | +0.0000 | 0.0057 | exact | ok | test 11 |
| memory_rare_in_A1, I = 29952 | 0.0698 | 0.0698 | +0.0000 | 0.0210 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 29952 | 0.0238 | 0.0238 | +0.0000 | 0.0125 | exact | ok | test 11 |
| memory_A, I = 100096 | 0.0312 | 0.0312 | +0.0000 | 0.0063 | exact | ok | test 11 |
| gain_A2, I = 100096 | 0.0026 | 0.0026 | +0.0000 | 0.0049 | exact | ok | test 11 |
| gain_C2, I = 100096 | -0.0286 | -0.0286 | +0.0000 | 0.0060 | exact | ok | test 11 |
| memory_rare_in_A1, I = 100096 | 0.0398 | 0.0398 | +0.0000 | 0.0240 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 100096 | 0.0087 | 0.0087 | +0.0000 | 0.0167 | exact | ok | test 11 |
| memory_A, I = 300032 | 0.0164 | 0.0164 | +0.0000 | 0.0078 | exact | ok | test 11 |
| gain_A2, I = 300032 | -0.0004 | -0.0004 | +0.0000 | 0.0064 | exact | ok | test 11 |
| gain_C2, I = 300032 | -0.0168 | -0.0168 | +0.0000 | 0.0070 | exact | ok | test 11 |
| memory_rare_in_A1, I = 300032 | 0.0272 | 0.0272 | +0.0000 | 0.0265 | exact | ok | test 11 |
| memory_rare_not_in_A1, I = 300032 | -0.0050 | -0.0050 | +0.0000 | 0.0182 | exact | ok | test 11 |
| memory_A halves at (tokens) | 6115 | 6115 | +0.0000 |  | exact | ok | test 11; |diff| < 1 token required |
| memory_rare_in_A1 halves at (tokens) | 2948 | 2948 | +0.0000 |  | exact | ok | test 11; |diff| < 1 token required |
| gain_A2 halves at (tokens) | 5304 | 5304 | +0.0000 |  | exact | ok | test 11; |diff| < 1 token required |

## C8

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| main_K1 stream gain | 0.0997 | 0.0997 | +0.0000 | 0.0057 | exact | ok | test 12 |
| main_K1 WikiText last half | 0.1958 | 0.1958 | +0.0000 | 0.0060 | exact | ok | test 12 |
| main_K1 WikiText final | 0.2518 | 0.2518 | +0.0000 | 0.0070 | exact | ok | test 12 |
| main_K1 books last half | -0.0056 | -0.0056 | +0.0000 | 0.0101 | exact | ok | test 12 |
| main_K1 books final | -0.0021 | -0.0021 | +0.0000 | 0.0092 | exact | ok | test 12 |
| main_K1 exchange rate | 0.5100 | 0.5100 | +0.0000 |  | exact | ok | test 12; 2 decimals, |diff| < 0.01 required |
| main_shuffled stream gain | 0.0619 | 0.0619 | +0.0000 | 0.0025 | exact | ok | test 12 |
| main_shuffled WikiText last half | 0.1123 | 0.1123 | +0.0000 | 0.0044 | exact | ok | test 12 |
| main_shuffled WikiText final | 0.1018 | 0.1018 | +0.0000 | 0.0051 | exact | ok | test 12 |
| main_shuffled books last half | -0.0451 | -0.0451 | +0.0000 | 0.0081 | exact | ok | test 12 |
| main_shuffled books final | -0.0522 | -0.0522 | +0.0000 | 0.0083 | exact | ok | test 12 |
| main_shuffled exchange rate | 0.5500 | 0.5500 | +0.0000 |  | exact | ok | test 12; 2 decimals, |diff| < 0.01 required |
| verylong stream gain | 0.1702 | 0.1702 | +0.0000 | 0.0110 | exact | ok | test 12 |
| verylong WikiText last half | 0.2681 | 0.2681 | +0.0000 | 0.0077 | exact | ok | test 12 |
| verylong WikiText final | 0.3229 | 0.3229 | +0.0000 | 0.0094 | exact | ok | test 12 |
| verylong books last half | 0.0521 | 0.0521 | +0.0000 | 0.0128 | exact | ok | test 12 |
| verylong books final | 0.0501 | 0.0501 | +0.0000 | 0.0223 | exact | ok | test 12 |
| verylong exchange rate | 0.6300 | 0.6300 | +0.0000 |  | exact | ok | test 12; 2 decimals, |diff| < 0.01 required |
| shuffled - K1, WikiText last half | -0.0836 | -0.0836 | -0.0000 | 0.0028 | exact | ok | test 12 detok files |
| shuffled - K1, books last half | -0.0395 | -0.0395 | +0.0000 | 0.0024 | exact | ok | test 12 |

## C9

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| frozen14m loss | 4.4192 | 4.4192 | +0.0000 | 0.0000 | exact | ok | test 10C |
| frozen14m share of 14M-31M gap closed | -0.0000 | 0.0000 | +0.0000 |  | exact | ok | test 10C |
| frozen31m loss | 4.0686 | 4.0686 | +0.0000 | 0.0102 | exact | ok | test 10C |
| frozen31m share of 14M-31M gap closed | 1.0000 | 1.0000 | +0.0000 |  | exact | ok | test 10C |
| frozen70m loss | 3.8465 | 3.8465 | +0.0000 | 0.0128 | exact | ok | test 10C |
| frozen70m share of 14M-31M gap closed | 1.6336 | 1.6336 | +0.0000 |  | exact | ok | test 10C |
| online14m loss | 4.2490 | 4.2490 | +0.0000 | 0.0110 | exact | ok | test 10C |
| online14m share of 14M-31M gap closed | 0.4854 | 0.4854 | +0.0000 |  | exact | ok | test 10C |
| online31m loss | 3.9008 | 3.9008 | +0.0000 | 0.0126 | exact | ok | test 10C |
| online31m share of 14M-31M gap closed | 1.4787 | 1.4787 | +0.0000 |  | exact | ok | test 10C |
| 14M online - frozen 14M | 0.1702 | 0.1702 | +0.0000 | 0.0110 | exact | ok | test 10C pairs |
| 31M online - frozen 31M | 0.1678 | 0.1678 | +0.0000 | 0.0125 | exact | ok | test 10C pairs |
| 31M online - 14M online | 0.3482 | 0.3482 | +0.0000 | 0.0130 | exact | ok | test 10C pairs |
| frozen 31M - 14M online | 0.1804 | 0.1804 | +0.0000 | 0.0192 | exact | ok | test 10C pairs |
| 31M online - frozen 70M | -0.0543 | -0.0543 | +0.0000 | 0.0103 | exact | ok | test 10C pairs |
| gap closed, 0k-10k | 0.0684 | 0.0680 | -0.0004 |  | exact | ok | test 10C positions (old: ratio of rounded book means) |
| gap closed, 50k-60k | 0.4832 | 0.4830 | -0.0002 |  | exact | ok | test 10C positions (old: ratio of rounded book means) |
| gap closed, 100k-110k | 0.5067 | 0.5070 | +0.0003 |  | exact | ok | test 10C positions (old: ratio of rounded book means) |
| gap closed, 150k-160k | 0.7412 | 0.7410 | -0.0002 |  | exact | ok | test 10C positions (old: ratio of rounded book means) |
| gap closed, 190k-200k | 0.5643 | 0.5640 | -0.0003 |  | exact | ok | test 10C positions (old: ratio of rounded book means) |

## C10

| quantity | old | new | diff | SE | kind | flag | note |
|---|---|---|---|---|---|---|---|
| 14m sgd gain K1, all tokens | 0.0997 | 0.0997 | +0.0000 | 0.0057 | exact | ok | test 13, lr 0.0001 |
| 14m sgd locality, all tokens | 0.0378 | 0.0378 | +0.0000 | 0.0042 | exact | ok | test 13, lr 0.0001 |
| 14m sgd gain K1, earlier chunk < 1k | 0.3075 | 0.3075 | +0.0000 | 0.0113 | exact | ok | test 13, lr 0.0001 |
| 14m sgd locality, earlier chunk < 1k | 0.2453 | 0.2453 | +0.0000 | 0.0093 | exact | ok | test 13, lr 0.0001 |
| 14m sgd gain K1, 1k-10k same book | 0.1542 | 0.1542 | +0.0000 | 0.0084 | exact | ok | test 13, lr 0.0001 |
| 14m sgd locality, 1k-10k same book | 0.1204 | 0.1204 | +0.0000 | 0.0071 | exact | ok | test 13, lr 0.0001 |
| 14m sgd gain K1, never seen | -0.0586 | -0.0586 | +0.0000 | 0.0130 | exact | ok | test 13, lr 0.0001 |
| 14m sgd locality, never seen | -0.0038 | -0.0038 | +0.0000 | 0.0102 | exact | ok | test 13, lr 0.0001 |
| 14m sgd WikiText forgetting K1 (old snapshot grid) | 0.1958 | 0.1958 | +0.0000 | 0.0060 | exact | ok | test 13, lr 0.0001 |
| 14m sgd books forgetting K1 (old snapshot grid) | -0.0056 | -0.0056 | -0.0000 | 0.0101 | exact | ok | test 13, lr 0.0001 |
| 14m sgd WikiText forgetting shuffled (old snapshot grid) | 0.1123 | 0.1123 | -0.0000 | 0.0044 | exact | ok | test 13, lr 0.0001 |
| 14m sgd books forgetting shuffled (old snapshot grid) | -0.0451 | -0.0451 | +0.0000 | 0.0081 | exact | ok | test 13, lr 0.0001 |
| 14m sgd exchange rate K1 (old snapshot grid) | 0.5100 | 0.5100 | +0.0000 |  | exact | ok | test 13, lr 0.0001 |
| 14m sgd exchange rate K1 (new snapshot grid, every 40) | 0.5100 | 0.5100 | +0.0000 |  | exact | ok | test 13, lr 0.0001; old grid every 80 unless reused |
| 14m adam gain K1, all tokens | 0.0865 | 0.0865 | +0.0000 | 0.0049 | exact | ok | test 13, lr 3e-06 |
| 14m adam locality, all tokens | -0.0041 | -0.0041 | +0.0000 | 0.0037 | exact | ok | test 13, lr 3e-06 |
| 14m adam gain K1, earlier chunk < 1k | 0.1420 | 0.1420 | +0.0000 | 0.0108 | exact | ok | test 13, lr 3e-06 |
| 14m adam locality, earlier chunk < 1k | -0.0517 | -0.0517 | +0.0000 | 0.0089 | exact | ok | test 13, lr 3e-06 |
| 14m adam gain K1, 1k-10k same book | 0.1500 | 0.1500 | +0.0000 | 0.0073 | exact | ok | test 13, lr 3e-06 |
| 14m adam locality, 1k-10k same book | 0.0375 | 0.0375 | +0.0000 | 0.0060 | exact | ok | test 13, lr 3e-06 |
| 14m adam gain K1, never seen | -0.2620 | -0.2620 | +0.0000 | 0.0454 | exact | ok | test 13, lr 3e-06 |
| 14m adam locality, never seen | -0.1120 | -0.1120 | +0.0000 | 0.0255 | exact | ok | test 13, lr 3e-06 |
| 14m adam WikiText forgetting K1 (old snapshot grid) | 0.1404 | 0.1404 | +0.0000 | 0.0049 | exact | ok | test 13, lr 3e-06 |
| 14m adam books forgetting K1 (old snapshot grid) | -0.0378 | -0.0378 | -0.0000 | 0.0112 | exact | ok | test 13, lr 3e-06 |
| 14m adam WikiText forgetting shuffled (old snapshot grid) | 0.1006 | 0.1006 | +0.0000 | 0.0045 | exact | ok | test 13, lr 3e-06 |
| 14m adam books forgetting shuffled (old snapshot grid) | -0.0615 | -0.0615 | +0.0000 | 0.0093 | exact | ok | test 13, lr 3e-06 |
| 14m adam exchange rate K1 (old snapshot grid) | 0.6200 | 0.6200 | +0.0000 |  | exact | ok | test 13, lr 3e-06 |
| 14m adam exchange rate K1 (new snapshot grid, every 40) | 0.6200 | 0.6100 | -0.0100 |  | grid change | ok | test 13, lr 3e-06; old grid every 80 unless reused |
| 14m nsgd gain K1, all tokens | 0.1128 | 0.1128 | +0.0000 | 0.0083 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd locality, all tokens | 0.0555 | 0.0555 | +0.0000 | 0.0056 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd gain K1, earlier chunk < 1k | 0.5103 | 0.5103 | +0.0000 | 0.0185 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd locality, earlier chunk < 1k | 0.3725 | 0.3725 | +0.0000 | 0.0115 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd gain K1, 1k-10k same book | 0.1919 | 0.1919 | +0.0000 | 0.0109 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd locality, 1k-10k same book | 0.1567 | 0.1567 | +0.0000 | 0.0081 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd gain K1, never seen | -0.1228 | -0.1228 | +0.0000 | 0.0164 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd locality, never seen | -0.0285 | -0.0285 | +0.0000 | 0.0128 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd WikiText forgetting K1 (old snapshot grid) | 0.3200 | 0.3200 | -0.0000 | 0.0083 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd books forgetting K1 (old snapshot grid) | 0.0632 | 0.0632 | +0.0000 | 0.0180 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd WikiText forgetting shuffled (old snapshot grid) | 0.1969 | 0.1969 | +0.0000 | 0.0058 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd books forgetting shuffled (old snapshot grid) | -0.0123 | -0.0123 | -0.0000 | 0.0123 | exact | ok | test 13, lr 0.0001 |
| 14m nsgd exchange rate K1 (old snapshot grid) | 0.3500 | 0.3500 | +0.0000 |  | exact | ok | test 13, lr 0.0001 |
| 14m nsgd exchange rate K1 (new snapshot grid, every 40) | 0.3500 | 0.3600 | +0.0100 |  | grid change | ok | test 13, lr 0.0001; old grid every 80 unless reused |
| 31m sgd gain K1, all tokens | 0.0849 | 0.0849 | +0.0000 | 0.0055 | exact | ok | test 13, lr 0.0003 |
| 31m sgd locality, all tokens | 0.0395 | 0.0395 | +0.0000 | 0.0041 | exact | ok | test 13, lr 0.0003 |
| 31m sgd gain K1, earlier chunk < 1k | 0.3682 | 0.3682 | +0.0000 | 0.0156 | exact | ok | test 13, lr 0.0003 |
| 31m sgd locality, earlier chunk < 1k | 0.2622 | 0.2622 | +0.0000 | 0.0098 | exact | ok | test 13, lr 0.0003 |
| 31m sgd gain K1, 1k-10k same book | 0.1705 | 0.1705 | +0.0000 | 0.0090 | exact | ok | test 13, lr 0.0003 |
| 31m sgd locality, 1k-10k same book | 0.1349 | 0.1349 | +0.0000 | 0.0064 | exact | ok | test 13, lr 0.0003 |
| 31m sgd gain K1, never seen | -0.0367 | -0.0367 | +0.0000 | 0.0115 | exact | ok | test 13, lr 0.0003 |
| 31m sgd locality, never seen | -0.0226 | -0.0226 | +0.0000 | 0.0102 | exact | ok | test 13, lr 0.0003 |
| 31m sgd WikiText forgetting K1 (old snapshot grid) | 0.2283 | 0.2283 | +0.0000 | 0.0047 | exact | ok | test 13, lr 0.0003 |
| 31m sgd books forgetting K1 (old snapshot grid) | 0.0313 | 0.0313 | +0.0000 | 0.0120 | exact | ok | test 13, lr 0.0003 |
| 31m sgd WikiText forgetting shuffled (old snapshot grid) | 0.1453 | 0.1453 | +0.0000 | 0.0036 | exact | ok | test 13, lr 0.0003 |
| 31m sgd books forgetting shuffled (old snapshot grid) | -0.0197 | -0.0197 | +0.0000 | 0.0071 | exact | ok | test 13, lr 0.0003 |
| 31m sgd exchange rate K1 (old snapshot grid) | 0.3700 | 0.3700 | +0.0000 |  | exact | ok | test 13, lr 0.0003 |
| 31m sgd exchange rate K1 (new snapshot grid, every 40) | 0.3700 | 0.3700 | +0.0000 |  | exact | ok | test 13, lr 0.0003; old grid every 80 unless reused |
| 31m adam gain K1, all tokens | 0.0704 | 0.0704 | +0.0000 | 0.0040 | exact | ok | test 13, lr 3e-06 |
| 31m adam locality, all tokens | -0.0070 | -0.0070 | +0.0000 | 0.0029 | exact | ok | test 13, lr 3e-06 |
| 31m adam gain K1, earlier chunk < 1k | 0.1772 | 0.1772 | +0.0000 | 0.0120 | exact | ok | test 13, lr 3e-06 |
| 31m adam locality, earlier chunk < 1k | -0.0705 | -0.0705 | +0.0000 | 0.0107 | exact | ok | test 13, lr 3e-06 |
| 31m adam gain K1, 1k-10k same book | 0.1512 | 0.1512 | +0.0000 | 0.0074 | exact | ok | test 13, lr 3e-06 |
| 31m adam locality, 1k-10k same book | 0.0273 | 0.0273 | +0.0000 | 0.0061 | exact | ok | test 13, lr 3e-06 |
| 31m adam gain K1, never seen | -0.2468 | -0.2468 | +0.0000 | 0.0476 | exact | ok | test 13, lr 3e-06 |
| 31m adam locality, never seen | -0.1292 | -0.1292 | +0.0000 | 0.0313 | exact | ok | test 13, lr 3e-06 |
| 31m adam WikiText forgetting K1 (old snapshot grid) | 0.1361 | 0.1361 | -0.0000 | 0.0039 | exact | ok | test 13, lr 3e-06 |
| 31m adam books forgetting K1 (old snapshot grid) | -0.0121 | -0.0121 | +0.0000 | 0.0081 | exact | ok | test 13, lr 3e-06 |
| 31m adam WikiText forgetting shuffled (old snapshot grid) | 0.1039 | 0.1039 | +0.0000 | 0.0037 | exact | ok | test 13, lr 3e-06 |
| 31m adam books forgetting shuffled (old snapshot grid) | -0.0387 | -0.0387 | +0.0000 | 0.0065 | exact | ok | test 13, lr 3e-06 |
| 31m adam exchange rate K1 (old snapshot grid) | 0.5200 | 0.5200 | +0.0000 |  | exact | ok | test 13, lr 3e-06 |
| 31m adam exchange rate K1 (new snapshot grid, every 40) | 0.5200 | 0.5100 | -0.0100 |  | grid change | ok | test 13, lr 3e-06; old grid every 80 unless reused |
| 31m nsgd gain K1, all tokens | 0.0917 | 0.0917 | +0.0000 | 0.0075 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd locality, all tokens | 0.0598 | 0.0598 | +0.0000 | 0.0048 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd gain K1, earlier chunk < 1k | 0.5988 | 0.5988 | +0.0000 | 0.0255 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd locality, earlier chunk < 1k | 0.4181 | 0.4181 | +0.0000 | 0.0128 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd gain K1, 1k-10k same book | 0.2237 | 0.2237 | +0.0000 | 0.0121 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd locality, 1k-10k same book | 0.1882 | 0.1882 | +0.0000 | 0.0076 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd gain K1, never seen | -0.1080 | -0.1080 | +0.0000 | 0.0165 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd locality, never seen | -0.0470 | -0.0470 | +0.0000 | 0.0146 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd WikiText forgetting K1 (old snapshot grid) | 0.3438 | 0.3438 | +0.0000 | 0.0064 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd books forgetting K1 (old snapshot grid) | 0.1082 | 0.1082 | +0.0000 | 0.0264 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd WikiText forgetting shuffled (old snapshot grid) | 0.2391 | 0.2391 | +0.0000 | 0.0050 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd books forgetting shuffled (old snapshot grid) | 0.0227 | 0.0227 | -0.0000 | 0.0126 | exact | ok | test 13, lr 0.0001 |
| 31m nsgd exchange rate K1 (old snapshot grid) | 0.2700 | 0.2700 | +0.0000 |  | exact | ok | test 13, lr 0.0001 |
| 31m nsgd exchange rate K1 (new snapshot grid, every 40) | 0.2700 | 0.2600 | -0.0100 |  | grid change | ok | test 13, lr 0.0001; old grid every 80 unless reused |
| 70m sgd gain K1, all tokens | 0.0247 | 0.0247 | +0.0000 | 0.0020 | exact | ok | test 13, lr 1e-05 |
| 70m sgd locality, all tokens | 0.0042 | 0.0042 | +0.0000 | 0.0017 | exact | ok | test 13, lr 1e-05 |
| 70m sgd gain K1, earlier chunk < 1k | 0.0512 | 0.0512 | +0.0000 | 0.0042 | exact | ok | test 13, lr 1e-05 |
| 70m sgd locality, earlier chunk < 1k | 0.0162 | 0.0162 | +0.0000 | 0.0032 | exact | ok | test 13, lr 1e-05 |
| 70m sgd gain K1, 1k-10k same book | 0.0325 | 0.0325 | +0.0000 | 0.0038 | exact | ok | test 13, lr 1e-05 |
| 70m sgd locality, 1k-10k same book | 0.0084 | 0.0084 | +0.0000 | 0.0033 | exact | ok | test 13, lr 1e-05 |
| 70m sgd gain K1, never seen | -0.0379 | -0.0379 | +0.0000 | 0.0078 | exact | ok | test 13, lr 1e-05 |
| 70m sgd locality, never seen | -0.0145 | -0.0145 | +0.0000 | 0.0044 | exact | ok | test 13, lr 1e-05 |
| 70m sgd WikiText forgetting K1 (old snapshot grid) | 0.0092 | 0.0092 | +0.0000 | 0.0015 | exact | ok | test 13, lr 1e-05 |
| 70m sgd books forgetting K1 (old snapshot grid) | -0.0192 | -0.0192 | +0.0000 | 0.0018 | exact | ok | test 13, lr 1e-05 |
| 70m sgd WikiText forgetting shuffled (old snapshot grid) | 0.0079 | 0.0079 | -0.0000 | 0.0016 | exact | ok | test 13, lr 1e-05 |
| 70m sgd books forgetting shuffled (old snapshot grid) | -0.0218 | -0.0218 | +0.0000 | 0.0020 | exact | ok | test 13, lr 1e-05 |
| 70m sgd exchange rate K1 (old snapshot grid) | 2.6800 | 2.6800 | +0.0000 |  | exact | ok | test 13, lr 1e-05 |
| 70m sgd exchange rate K1 (new snapshot grid, every 40) | 2.6800 | 2.5700 | -0.1100 |  | grid change | ok | test 13, lr 1e-05; old grid every 80 unless reused |
| 70m adam gain K1, all tokens | 0.0286 | 0.0286 | +0.0000 | 0.0020 | exact | ok | test 13, lr 3e-07 |
| 70m adam locality, all tokens | -0.0150 | -0.0150 | +0.0000 | 0.0019 | exact | ok | test 13, lr 3e-07 |
| 70m adam gain K1, earlier chunk < 1k | -0.0029 | -0.0029 | +0.0000 | 0.0095 | exact | ok | test 13, lr 3e-07 |
| 70m adam locality, earlier chunk < 1k | -0.1400 | -0.1400 | +0.0000 | 0.0102 | exact | ok | test 13, lr 3e-07 |
| 70m adam gain K1, 1k-10k same book | 0.0338 | 0.0338 | +0.0000 | 0.0057 | exact | ok | test 13, lr 3e-07 |
| 70m adam locality, 1k-10k same book | -0.0435 | -0.0435 | +0.0000 | 0.0045 | exact | ok | test 13, lr 3e-07 |
| 70m adam gain K1, never seen | -0.3861 | -0.3861 | +0.0000 | 0.0624 | exact | ok | test 13, lr 3e-07 |
| 70m adam locality, never seen | -0.1331 | -0.1331 | +0.0000 | 0.0381 | exact | ok | test 13, lr 3e-07 |
| 70m adam WikiText forgetting K1 (old snapshot grid) | 0.0619 | 0.0619 | -0.0000 | 0.0032 | exact | ok | test 13, lr 3e-07 |
| 70m adam books forgetting K1 (old snapshot grid) | -0.0203 | -0.0203 | -0.0000 | 0.0051 | exact | ok | test 13, lr 3e-07 |
| 70m adam WikiText forgetting shuffled (old snapshot grid) | 0.0660 | 0.0660 | +0.0000 | 0.0037 | exact | ok | test 13, lr 3e-07 |
| 70m adam books forgetting shuffled (old snapshot grid) | -0.0231 | -0.0231 | +0.0000 | 0.0039 | exact | ok | test 13, lr 3e-07 |
| 70m adam exchange rate K1 (old snapshot grid) | 0.4600 | 0.4600 | +0.0000 |  | exact | ok | test 13, lr 3e-07 |
| 70m adam exchange rate K1 (new snapshot grid, every 40) | 0.4600 | 0.4700 | +0.0100 |  | grid change | ok | test 13, lr 3e-07; old grid every 80 unless reused |
| 70m nsgd gain K1, all tokens | 0.0573 | 0.0573 | +0.0000 | 0.0035 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd locality, all tokens | 0.0185 | 0.0185 | +0.0000 | 0.0028 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd gain K1, earlier chunk < 1k | 0.2035 | 0.2035 | +0.0000 | 0.0110 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd locality, earlier chunk < 1k | 0.0841 | 0.0841 | +0.0000 | 0.0058 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd gain K1, 1k-10k same book | 0.1061 | 0.1061 | +0.0000 | 0.0051 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd locality, 1k-10k same book | 0.0496 | 0.0496 | +0.0000 | 0.0031 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd gain K1, never seen | -0.0727 | -0.0727 | +0.0000 | 0.0100 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd locality, never seen | -0.0360 | -0.0360 | +0.0000 | 0.0058 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd WikiText forgetting K1 (old snapshot grid) | 0.0388 | 0.0388 | -0.0000 | 0.0025 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd books forgetting K1 (old snapshot grid) | -0.0104 | -0.0104 | -0.0000 | 0.0044 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd WikiText forgetting shuffled (old snapshot grid) | 0.0257 | 0.0257 | +0.0000 | 0.0024 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd books forgetting shuffled (old snapshot grid) | -0.0262 | -0.0262 | +0.0000 | 0.0026 | exact | ok | test 13, lr 3e-06 |
| 70m nsgd exchange rate K1 (old snapshot grid) | 1.4800 | 1.4800 | +0.0000 |  | exact | ok | test 13, lr 3e-06 |
| 70m nsgd exchange rate K1 (new snapshot grid, every 40) | 1.4800 | 1.4800 | +0.0000 |  | grid change | ok | test 13, lr 3e-06; old grid every 80 unless reused |
| diag 14m attention relative update | 4.11e-06 | 4.11e-06 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 14m mlp relative update | 0.000123 | 0.000123 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 14m output embedding relative update | 7.18e-06 | 7.18e-06 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 31m attention relative update | 8.16e-06 | 8.16e-06 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 31m mlp relative update | 9.91e-05 | 9.91e-05 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 31m output embedding relative update | 1.77e-05 | 1.77e-05 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 70m attention relative update | 3.79e-07 | 3.79e-07 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 70m mlp relative update | 1.12e-06 | 1.12e-06 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
| diag 70m output embedding relative update | 4.23e-07 | 4.23e-07 | +0 |  | exact | ok | test 13 step 1; relative |diff| < 1% required |
