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
| [`experiments_all/`](experiments_all/README.md) | code, data and one subfolder per experiment |
| [`paper/`](paper/) | LaTeX draft of the paper (`main.tex`, built as `main.pdf`) |

| experiment | question | answer so far |
|---|---|---|
| [1](experiments_all/experiment1/README.md) | Do small MLPs exploit temporal locality with plain online SGD, and at which learning rate? | yes, for small models: +6.4 pp over the frozen model at h = 2 (lr 0.001); the locality effect grows as the model shrinks and the learning rate rises, but at lr 0.01 every model ends up below the frozen one |
| [2](experiments_all/experiment2/README.md) | Do two time scales (slow + fast weights) exploit it better? | a fast output bias nearly doubles plain SGD's gain at h = 2 (+12.7 vs +6.4 pp), but a label prior that learns no weights gains most (+13.4 pp); making the whole network fast barely beats plain SGD and doubles the model |
