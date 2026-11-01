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
| [MNIST2 / 1](experiments_MNIST2/experiment1/README.md) | With the locality in the inputs (bursts of one rotation or pixel permutation, all 10 digits), does online learning (SGD or slow + fast weights) beat the frozen model trained on all contexts? | no: a per-context model would gain 19 to 23 pp (rotated) and 3 to 7 pp (permuted) at h ≤ 8, but every online variant stays within +0.4 pp of the frozen model; larger learning rates only hurt, and the label prior loses as it should (no label locality) |
| [MNIST2 / 2](experiments_MNIST2/experiment2/README.md) | Same, with longer contexts: does online learning start to pay as the burst length L grows (100 to 5000 images)? | on rotated yes: a fast hidden layer gains +2 to +6 pp from L = 500-1000 and plain SGD up to +8.7 pp at L = 5000 (h = 4; 26-39% of the ceiling), but each context switch costs about 20 pp early in the burst and returning contexts are not re-adapted faster; h = 2 and permuted MNIST barely gain |
| [MNIST2 / 3](experiments_MNIST2/experiment3_metaSGD/README.md) | Can Meta-SGD (meta-learned slow weights plus per-parameter fast learning rates and decays) make adaptation pay on experiment 1's short bursts? | a little: +2.4 pp (h = 4) and +3.5 pp (h = 8) over the frozen mixed model with a positive locality effect, the first gain on this stream, but only 10-17% of the ceiling, and about 1 pp of the h = 8 gain comes from the meta-trained slow weights themselves |
