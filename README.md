# goldfish: exploiting temporal locality at inference time

Many ML inputs have strong temporal locality. A model that sees such inputs does not need to
remember everything all the time: it can remember some basics and adapt fast to the current
context. Under tight capacity limits, online training could then beat offline training, and
the recency bias of neural networks, usually treated as a problem, could become a benefit.

The experiments test this on MNIST. Small networks are trained offline, then updated online
(predict, then one training step per image) on a stream of 10000 test images arranged in bursts
of 3 digits, with 10% intruders (digits from outside the current burst). A shuffled copy of the same stream is the no-locality control. The key metric is
the **locality effect**: online accuracy on bursts minus online accuracy on the shuffled stream.

## Folders

| folder | contents |
|---|---|
| [`experiments_all/`](experiments_all/README.md) | code, data and one subfolder per experiment |
| [`paper/`](paper/) | LaTeX draft of the paper (`main.tex`, built as `main.pdf`) |

| experiment | question | answer so far |
|---|---|---|
| [1](experiments_all/experiment1/README.md) | Do small MLPs exploit temporal locality with plain online SGD, and at which learning rate? | yes, for small models: +5 pp over the frozen model at h = 2 (lr 0.001); higher rates lose more on intruders |
| [2](experiments_all/experiment2/README.md) | Do two time scales (slow + fast weights) exploit it better? | no variant gains on the burst images without losing on intruders; a label prior with no learning gains most overall, a fast output bias is second; making the whole network fast is worse than plain SGD |
