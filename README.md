# goldfish: exploiting temporal locality at inference time

Many ML inputs have strong temporal locality. A model that sees such inputs does not need to
remember everything all the time: it can remember some basics and adapt fast to the current
context. Under tight capacity limits, online training could then beat offline training, and
the recency bias of neural networks, usually treated as a problem, could become a benefit.

The experiments test this on MNIST. Small networks are trained offline, then updated online
(predict, then one training step per image) on a stream of 10000 test images arranged in bursts
of 3 digits. We assume that real input streams have temporal locality like this one, not that
they are random. A shuffled copy of the same stream is only a control: it holds the same images,
so it tells whether the online gains come from temporal locality and not merely from training on
more data. The key metric is the **locality effect**: online accuracy on bursts minus online
accuracy on the shuffled stream.

## Folders

| folder | contents |
|---|---|
| [`experiments_MNIST/`](experiments_MNIST/README.md) | code, data and one subfolder per experiment |
| [`experiments_MNIST2/`](experiments_MNIST2/README.md) | the same with context streams: locality in the inputs (rotated or permuted MNIST), not in the labels |
| [`paper/`](paper/) | LaTeX draft of the paper (`main.tex`, built as `main.pdf`) |

| experiment | question | answer so far |
|---|---|---|
| [1](experiments_MNIST/experiment1/README.md) | Do small MLPs exploit temporal locality with plain online SGD, and at which learning rate? | yes, for small models: +6.4 pp over the frozen model at h = 2 (lr 0.001); the locality effect grows as the model shrinks and the learning rate rises, but at lr 0.01 every model ends up below the frozen one |
| [2](experiments_MNIST/experiment2/README.md) | Do two time scales (slow + fast weights) exploit it better? | a fast output bias nearly doubles plain SGD's gain at h = 2 (+12.7 vs +6.4 pp) and a label prior that learns no weights gains most (+13.4 pp), while fast parts on the output layer, the first layer or every layer gain much less (+6.5 to +8.3 pp) |
| [MNIST2 / 1](experiments_MNIST2/experiment1/README.md) | With the locality in the inputs (bursts of one rotation, all 10 digits), does plain online SGD beat the frozen mixed model, on short bursts (20-200) or bursts of 1000? | barely: on short bursts at most +0.25 pp, and larger learning rates lose 2 to 7 pp; on bursts of 1000 at most +0.7 pp (h = 8) and +1.6 pp (h = 4), because what larger learning rates gain late in a burst they lose after each switch |
| [MNIST2 / 2](experiments_MNIST2/experiment2/README.md) | Do slow + fast weights (a decaying fast part on the hidden layer) beat it? | on bursts of 1000 yes: +4.1 pp (h = 8) and +6.1 pp (h = 4), with a locality effect of +8 to +10 pp; on short bursts no (at most +0.3 pp), since fast settings lose 3 to 4 pp to the switching cost |
| [MNIST2 / 3](experiments_MNIST2/experiment3_metaSGD/README.md) | Does Meta-SGD (meta-learned slow weights plus per-parameter fast learning rates and decays) do better? | yes, and it is the only method that gains on short bursts: +2.7 pp (h = 8) and +2.4 pp (h = 4), and +3.5 / +3.8 pp on bursts of 1000 (below the best dual setting there); with the slow weights fixed the gain roughly halves |
