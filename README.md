# goldfish: exploiting temporal locality at inference time

Many ML inputs have strong temporal locality. A model that sees such inputs does not need to
remember everything all the time: it can remember some basics and adapt fast to the current
context. Under tight capacity limits, online training could then beat offline training, and
the recency bias of neural networks, usually treated as a problem, could become a benefit.

The experiments test this on MNIST. Small networks are trained offline, then updated online
(predict, then one training step per image) on a stream of 10000 test images arranged in bursts
of 3 digits. A shuffled copy of the same stream is the no-locality control. The key metric is
the **locality effect**: online accuracy on bursts minus online accuracy on the shuffled stream.

## Folders

| folder | contents |
|---|---|
| [`experiments_all/`](experiments_all/README.md) | code, data and one subfolder per experiment |
| [`paper/`](paper/) | LaTeX draft of the paper (`main.tex`, built as `main.pdf`) |

| experiment | question | answer so far |
|---|---|---|
| [1](experiments_all/experiment1/README.md) | Do small MLPs naturally exploit temporal locality when trained online? | slightly: a small positive locality effect |
| [2](experiments_all/experiment2/README.md) | Do higher online learning rates exploit it better? | yes, for small models: up to +12 pp locality effect |
| [3](experiments_all/experiment3/README.md) | Does it also hold for a small CNN? | partly: locality effect, but no gain over the frozen model |
