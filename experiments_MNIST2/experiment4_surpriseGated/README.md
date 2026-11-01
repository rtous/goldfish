# Experiment 4: surprise-gated forgetting, and what any gate could add

**Question.** Meta-SGD (experiment 3) decays F at a constant rate, so it must choose between
forgetting fast (it plateaus within a burst) and forgetting slowly (it pays at every switch).
Does a meta-learned gate that wipes F when the loss jumps help? And what is the most any gate
could add: (A) if the model knew exactly where the switches are, and (B) can a switch be detected
from the loss at all, and after how many images?

**Method.**
- Gate (`metasgd_gated`): a scalar z per image, meta-learned with S, alpha and lambda. With m the
  running mean of past losses: z = sigmoid(b * (loss_t - m - c)),
  F <- lambda * (1 - z) * F - alpha * (1 + k * z) * g, m <- beta * m + (1 - beta) * loss_t.
- Oracle reset (`"oracle_reset": true`): F <- 0 just before the first image of each burst; S is
  never reset. The shuffled control is reset at the same positions. The model is told where the
  switches are, so this bounds what a perfect gate could do.
- Detectability (`switch_detection.py`): on the losses of the unreset Meta-SGD on the short
  stream, score_t = mean loss over the last w images - mean loss over the 50 before. Positives:
  position == w (the window covers exactly the first w images of a new context); negatives:
  position > 30. AUC by the rank formula, and TPR / FPR at the threshold maximizing TPR - FPR.

| config | method | settings |
|---|---|---|
| `metasgd_gated` | metasgd + gate | meta-learned as in experiment 3, init lambda 0.999, b 2, c 1, k 0.1, beta 0.9 |
| `dual_f1e-2_d0.999_reset` | dual + oracle reset | slow lr 1e-4, fast_lr 1e-2, decay 0.999, hidden |
| `dual_slow0_f1e-2_d0.999_reset` | dual + oracle reset | slow lr 0, fast_lr 1e-2, decay 0.999, hidden |
| `metasgd_reset` | metasgd + oracle reset | experiment 3's `metasgd` checkpoints |
| `metasgd` | metasgd | experiment 3's `metasgd` checkpoints (reproduces its row exactly) |

Meta settings of `metasgd_gated`: as in experiment 3 (episodes of 3 bursts of 20-60, Adam 1e-3,
meta-batch 32, clip 1, init alpha 1e-3, held-out accuracy every 50 steps, patience 30, at most
15 min per h); both runs reached the cap (4350 steps at h = 8, 5050 at h = 4). The other configs
need no meta-training.

## Results

Gain over frozen / locality effect (bursts - shuffled), in pp:

| | short, h = 8 | short, h = 4 | L1000, h = 8 | L1000, h = 4 |
|---|---|---|---|---|
| frozen mixed (%) | 73.25 | 64.41 | 74.05 | 64.78 |
| `metasgd_gated` | +0.98 / +0.60 | +1.32 / +1.36 | +0.92 / +0.48 | +1.76 / +1.54 |
| `dual_f1e-2_d0.999_reset` | -0.82 / +3.71 | -0.58 / +4.07 | +5.18 / +9.37 | +7.20 / +11.39 |
| `dual_slow0_f1e-2_d0.999_reset` | -0.70 / +3.47 | -1.07 / +4.13 | +3.78 / +9.35 | +3.93 / +10.28 |
| `metasgd_reset` | +2.91 / +3.60 | +2.69 / +4.49 | +3.55 / +4.01 | +3.79 / +4.43 |
| `metasgd` | +2.70 / +3.56 | +2.36 / +4.05 | +3.51 / +3.99 | +3.77 / +4.41 |
| `dual_f1e-2_d0.999` (experiment 2) | -3.47 / +2.09 | -3.07 / +2.47 | +4.11 / +8.47 | +6.11 / +10.22 |
| `dual_slow0_f1e-2_d0.999` (experiment 2) | -3.66 / +2.04 | -3.77 / +3.06 | +2.87 / +8.81 | +3.02 / +9.57 |

Switch detection from Meta-SGD's loss on the short stream (`switch_detection.csv`; 87 switches,
7364 negatives), AUC and best TPR / FPR:

| w | h = 8 | h = 4 |
|---|---|---|
| 1 | 0.583 (0.87 / 0.71) | 0.554 (0.56 / 0.44) |
| 3 | 0.558 (0.64 / 0.51) | 0.549 (0.58 / 0.46) |
| 5 | 0.569 (0.59 / 0.44) | 0.574 (0.46 / 0.29) |
| 10 | 0.580 (0.62 / 0.47) | 0.580 (0.35 / 0.20) |

Plots: `plots/summary.png`, `plots/burst_curve_short.png`, `plots/burst_curve_L1000.png`.

## Findings

- The learned gate does not help (+0.9 to +1.8 pp, below Meta-SGD everywhere), and it does not
  fire at switches: its mean z at position 1 is no higher than at positions 11+ (short, h = 8:
  0.13 vs 0.14). It fires on single hard images instead (z > 0.5 on 8% / 2% of images).
- Aggressive dual learning with an oracle reset still does not beat Meta-SGD on short bursts
  (-0.82 / -0.58 pp against +2.70 / +2.36). On L1000 it does (+5.18 / +7.20 against
  +3.51 / +3.77), about +1.1 pp more than without the reset.
- The reset lowers the cost at positions 1-10 by 5 to 6 pp on short (h = 8: 67.3% -> 72.6%) and
  10 to 12 pp on L1000. But positions 11-50 still lag: 71.0% (h = 8) and 61.3% (h = 4), below the
  frozen model and below positions 1-10. Large steps from F = 0 are noisy, and adaptation needs
  about 100 images either way.
- A reset barely helps Meta-SGD: +0.2 / +0.3 pp on short and +0.0 on L1000. Its gain at positions
  1-10 (+2.2 / +1.4 pp) is small because its fast part already forgets quickly.
- Switches cannot be detected from Meta-SGD's loss within 10 images: AUC 0.55 to 0.58 for every
  w, against bursts of 20 to 200. Its loss right after a switch is only slightly higher (h = 8:
  0.85 at positions 1-10 against 0.75 at 31+), far below the per-image spread (SD 1.14).
- Verdict: no, a better gate is not worth building. On Meta-SGD a perfect gate would add at most
  +0.3 pp, and the switches it would need to find are not visible in the loss. The gating
  direction is dropped.

Caveats: one seed; the settings were compared on the evaluation streams; the oracle uses
information the model never has; the gate's meta-training was time-capped at 15 min.

Next step: make adaptation itself faster and less noisy within a context (the 11-50 lag), for
example meta-training Meta-SGD on episodes as long as the bursts, rather than detecting switches.
