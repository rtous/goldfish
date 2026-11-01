# Experiment 3: Meta-SGD feasibility test (rotated, short bursts)

**Question.** On experiment1's rotated stream (bursts of 20-200 images), hand-tuned online
learning gains nothing (≤ +0.4 pp), against a ceiling of about +20 pp. Can meta-learning the
starting weights and per-parameter fast learning rates and decays (Meta-SGD, Li et al. 2017) make
adaptation fast enough to pay within such bursts?

**Answer.** A little. Meta-SGD gains +2.4 pp (h = 4) and +3.5 pp (h = 8) over the frozen mixed
model, with a positive locality effect (+4.1 and +3.7 pp), against a ceiling of +23 and +20 pp.
It is the first method to beat the frozen model on this stream, but it reaches only 10-17% of
the headroom.

## Method

The network predicts with S + F. S (slow) is meta-learned offline and frozen online. F starts at
0, and after each image, with g the gradient of that image's loss at S + F:

    F <- lambda * F - alpha * g,   alpha = exp(log_alpha) > 0,   lambda = sigmoid(logit_lambda)

S, alpha and lambda are all meta-learned, with one alpha and one lambda per parameter.

Meta-training (`meta_train.py`, MNIST training images only, 10000 held out for early stopping):
- an episode is 3 consecutive bursts with different rotation contexts, of 20-60 random images each;
- the inner loop is the online protocol above, starting from F = 0;
- the meta-loss is the mean prediction loss over the episode, backpropagated through the whole
  unroll (exact second order: the MLP's per-image gradient is written in closed form);
- Adam lr 1e-3, 32 episodes per step, gradient norm clipped to 1;
- initialization: S = experiment1's mixed model of the same h, alpha = 1e-3, lambda = 0.99;
- stopping: keep the best online accuracy on 256 held-out episodes, evaluated every 50 steps.

Evaluation: experiment1's `sequence_rotated.csv` and its shuffled control, image by image as in
`online_mlp.py` (checked: same losses as an autograd loop; alpha = 0 reproduces S frozen exactly).

## Results (rotated stream, accuracy %)

| | h = 4 | h = 8 |
|---|---|---|
| frozen mixed model (experiment1) | 64.41 | 73.25 |
| meta S frozen (no online updates) | 64.15 | **74.20** |
| meta-SGD online, bursts | 66.77 | 76.75 |
| meta-SGD online, shuffled control | 62.72 | 73.10 |
| gain over frozen mixed | **+2.36** | **+3.50** |
| gain over meta S frozen | +2.62 | +2.55 |
| locality effect (bursts − shuffled) | +4.05 | +3.65 |
| best hand-tuned online, experiment1 (`fasthidden`) | 64.57 | 73.65 |
| per-context ceiling (oracle on the stream) | 87.51 | 93.25 |
| meta-training | 34 min, 5750 steps | 42 min, 12900 steps (60 min cap) |

Accuracy by position within the burst:

| | 1-5 | 6-20 | 21-60 | 61+ |
|---|---|---|---|---|
| h = 4, frozen mixed | 63.18 | 63.79 | 63.59 | 65.20 |
| h = 4, meta S frozen | 64.09 | 63.48 | 63.46 | 64.76 |
| h = 4, meta-SGD online | 62.27 | 64.85 | 66.10 | 68.08 |
| h = 8, frozen mixed | 73.64 | 73.11 | 73.64 | 73.01 |
| h = 8, meta S frozen | 75.00 | 73.71 | 74.80 | 73.88 |
| h = 8, meta-SGD online | 75.68 | 75.15 | 77.19 | 76.98 |

Median meta-learned alpha / lambda per tensor (`0` = first layer, `2` = output layer):

| tensor | h = 4 alpha | h = 4 lambda | h = 8 alpha | h = 8 lambda |
|---|---|---|---|---|
| 0.weight | 1.2e-4 | 0.954 | 3.8e-5 | 0.911 |
| 0.bias | 6.0e-5 | 0.736 | 1.3e-5 | 0.445 |
| 2.weight | 1.6e-4 | 0.919 | 4.3e-5 | 0.737 |
| 2.bias | 1.3e-4 | 0.879 | 2.7e-5 | 0.550 |

Plot: `plots/meta_training.png` (held-out online accuracy and S frozen during meta-training).

## Verdict

Worth a fuller look, but not as is. Meta-SGD turns a no-gain setting into a small, real gain:
- the locality effect is positive;
- the online model loses on the shuffled stream, so the gain comes from the bursts;
- accuracy rises within a burst (h = 4: from 62% at images 1-5 to 68% at 61+).

But the gain is only 10-17% of the headroom. At h = 8, about +1 pp of the +3.5 comes from the
meta-learned S itself (74.20 vs 73.25 frozen; 75.47 vs 74.05 on the test set averaged over
contexts): that part is extra offline training on rotated data, not adaptation. Adaptation alone
is about +2.6 pp at both sizes. The medians of alpha all shrank from the initial 1e-3, and the
decays of the biases and output layer shrank too, so the learned fast part is small and short-lived.

Caveats: one seed, one meta-training run per size; medians hide the spread of alpha and lambda
(some weights may be much faster than the median); meta-training episodes (3 bursts of 20-60)
are shorter than the stream's bursts (20-200); h = 8 hit the 60 min cap while still improving
slightly (an earlier run with less patience stopped at 11 min with +0.5 pp).

**Next step.** Longer meta-training and episodes matching the stream's burst lengths. A
fast part that can do more per image, e.g. meta-learning a small per-context input transform
or a larger alpha on a few chosen parameters rather than on all of them.
