"""Online learning over a token stream, scored chunk by chunk, plus the learning-rate check.

The stream is cut into non-overlapping chunks of L = 256 tokens, each scored on its own (no KV
cache or context from earlier chunks): every token after a chunk's first is
predicted from the tokens before it in the chunk. Online: for each chunk, compute the loss (that is
what is scored), then take one plain SGD step on that same loss (mean over the chunk's scored
tokens), all parameters, no momentum, no weight decay. Frozen: the same stream with no updates.
Anything that helps beyond 256 tokens can only come through the weights.

Learning-rate check: online learning with lr in {1e-4, 3e-4, 1e-3} on the 2 extra books
(books 25 and 26 of data.py, not in the tests' stream); the lowest mean loss wins. Writes
lr_check.csv, read by the tests through chosen_lr().

Usage:
    python online.py        # learning-rate check (about 1 min)
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import HERE, books
from models import DEFAULT, DEFAULT_REVISION, load, seed_everything

L = 256
LRS = (1e-4, 3e-4, 1e-3)
LR_BOOKS = (25, 26)
LR_CSV = os.path.join(HERE, "lr_check.csv")


def trained_params(model, params):
    """Parameters updated online. params: "all"; "output" (only the output embedding / unembedding,
    `lm_head.weight`, called embed_out in the original GPT-NeoX code; not tied to the input
    embedding in Pythia); "no_output" (every parameter except that one); "mlp" / "attention" (the
MLP / attention weights and biases of every layer, layer norms excluded); "mlp@4,5" (the MLP of
the listed layers only, 0-based)."""
    out = model.get_output_embeddings().weight
    assert out.data_ptr() != model.get_input_embeddings().weight.data_ptr(), "tied embeddings"
    if params == "all":
        return list(model.parameters())
    if params == "output":
        return [out]
    if params == "no_output":
        return [p for p in model.parameters() if p is not out]
    if params in ("mlp", "attention"):
        return [p for k, p in model.named_parameters() if ".%s." % params in k]
    if params.startswith("mlp@"):
        layers = params[4:].split(",")
        return [p for k, p in model.named_parameters() if any(".layers.%s.mlp." % i in k for i in layers)]
    raise ValueError(params)


def stream_losses(model, init, ids, lr, batch=16, params="all", order=None, decay=None):
    """Per-token loss (nats, float32) for the stream ids [n]; NaN for each chunk's first token (not
    scored). lr = 0: frozen (chunks scored in batches). params: which parameters the SGD step
    updates (trained_params). order: optional permutation of the chunks, the order in which the
    online learner sees them (losses are still returned in stream order). decay: if set, the
    trained parameters are slow weights W (their init values, never changed) plus a fast part F
    that starts at 0: the model predicts with W + F, and after each chunk F <- decay * F - lr * g,
    with g the gradient at W + F (decay = 1 would be plain SGD). The model is reset to the state
    dict init first, and left in its final state."""
    model.load_state_dict(init)
    dev = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(ids, dtype=np.int64))
    out = np.full(len(x), np.nan, dtype=np.float32)
    starts = list(range(0, len(x), L))
    if lr == 0:
        with torch.no_grad():
            for i in range(0, len(starts), batch):
                group = [s for s in starts[i:i + batch] if min(s + L, len(x)) - s >= 2]
                full = [s for s in group if s + L <= len(x)]
                rest = [s for s in group if s + L > len(x)]
                if full:
                    xb = torch.stack([x[s:s + L] for s in full]).to(dev)
                    nll = F.cross_entropy(model(input_ids=xb).logits[:, :-1].transpose(1, 2), xb[:, 1:],
                                          reduction="none").cpu().numpy()
                    for r, s in enumerate(full):
                        out[s + 1:s + L] = nll[r]
                for s in rest:
                    xb = x[s:].unsqueeze(0).to(dev)
                    out[s + 1:] = F.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:],
                                                  reduction="none").cpu().numpy()
        return out
    train = trained_params(model, params)
    ids_train = {id(p) for p in train}
    for p in model.parameters():  # frozen parameters get no gradient (and cost no backward pass)
        p.requires_grad_(id(p) in ids_train)
    opt = torch.optim.SGD(train, lr=lr, momentum=0, weight_decay=0)
    if decay is not None:
        names = {id(p): k for k, p in model.named_parameters()}
        slow = [init[names[id(p)]].to(p.device) for p in train]
    if order is not None:
        starts = [starts[i] for i in order]
    for s in starts:
        e = min(s + L, len(x))
        if e - s < 2:
            continue
        xb = x[s:e].unsqueeze(0).to(dev)
        nll = F.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
        out[s + 1:e] = nll.detach().cpu().numpy()
        opt.zero_grad(set_to_none=True)
        nll.mean().backward()
        if decay is None:
            opt.step()
        else:
            with torch.no_grad():
                for p, w in zip(train, slow):  # W + F with F <- decay * F - lr * g
                    p.copy_(w + decay * (p - w) - lr * p.grad)
    for p in model.parameters():
        p.requires_grad_(True)
    return out


def setup(name=DEFAULT):
    """(model, init state dict, info, books dict); seed 0. Books are always tokenized with the
    default model's tokenizer (Pythia models share it; checked when another model is loaded)."""
    seed_everything(0)
    model, tokenizer, info = load(name, DEFAULT_REVISION)
    if name != DEFAULT:
        from transformers import AutoTokenizer
        from data import DATA
        ref = AutoTokenizer.from_pretrained(DEFAULT, cache_dir=os.path.join(DATA, "hf"))
        assert ref.get_vocab() == tokenizer.get_vocab(), "different tokenizer"
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, init, info, books(DEFAULT)


def chosen_lr():
    if not os.path.exists(LR_CSV):
        raise SystemExit("No lr_check.csv: run `python online.py` first.")
    with open(LR_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["lr"] != "0"]
    return float(min(rows, key=lambda r: float(r["loss"]))["lr"])


def main():
    model, init, info, d = setup()
    ids = np.concatenate([d["ids"][b] for b in LR_BOOKS])
    rows = []
    for lr in (0,) + LRS:
        t = time.time()
        loss = stream_losses(model, init, ids, lr)
        rows.append({"lr": lr, "loss": "%.4f" % np.nanmean(loss), "tokens": int(np.isfinite(loss).sum()),
                     "seconds": "%.0f" % (time.time() - t)})
        print(rows[-1], flush=True)
    with open(LR_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("books %s, %s; device %s; chosen lr %g" % (", ".join(d["titles"][b] for b in LR_BOOKS), info["commit"][:10],
                                                  info["device"], chosen_lr()))


if __name__ == "__main__":
    main()
