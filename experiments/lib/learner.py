"""Frozen scoring and online learners.

Protocol: a stream is cut into non-overlapping chunks of L = 256 tokens, each scored on its own (no
KV cache or context from earlier chunks): every token after a chunk's first is predicted from the
tokens before it in the chunk. Online (prequential): for each chunk, compute the loss (that is what
is scored), then take one update on that same loss (mean over the chunk's scored tokens). Frozen:
the same chunks with no updates. Anything that helps beyond 256 tokens can only come through the
weights.

Update rules (Learner):
    sgd   plain SGD (torch.optim.SGD, no momentum, no weight decay) on a parameter subset
          (trained_params). Stateless, so a run read in segments equals one call over the stream.
    adam  torch.optim.Adam (betas 0.9, 0.999, eps 1e-8, no weight decay), all parameters; a fresh
          optimizer per Learner (per stream), its state carried across segments.
    nsgd  normalized SGD per tensor: W <- W - lr * n(W) * g / (||g|| + 1e-12), n(W) = ||W|| for
          matrices and ||W|| + 1e-3 for 1-D tensors (biases, layer norms); every tensor moves by about
          the same relative amount lr per step. All parameters.
A Learner reads the stream in consecutive segments (multiples of L) and keeps its state between
them; Learner.score() evaluates other text with the current weights (a snapshot), no updates.

check() verifies on one book that segmented runs (with snapshot scoring in between) equal single
runs, for each rule. Usage: python lib/learner.py   (about 1 min)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import DATA, DEFAULT, L
from models import DEFAULT_REVISION, SIZES, load, seed_everything

RULES = ("sgd", "adam", "nsgd")
SUBSETS = ["all", "output", "no_output", "mlp", "attention"] + ["mlp@%d" % i for i in range(6)]


def trained_params(model, params):
    """Parameters updated online. params: "all"; "output" (only the output embedding / unembedding,
    `lm_head.weight`; not tied to the input embedding in Pythia); "no_output" (every parameter except
    that one); "mlp" / "attention" (the MLP / attention weights and biases of every layer, layer norms
    excluded); "mlp@3" (the MLP of one layer, 0-based: mlp@3 is layer 4)."""
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


def stream_losses(model, init, ids, lr, batch=16, params="all"):
    """Per-token loss (nats, float32) for the stream ids [n]; NaN for each chunk's first token (not
    scored). lr = 0: frozen (chunks scored in batches). Otherwise plain SGD on the params subset. The
    model is reset to the state dict init first, and left in its final state."""
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
    for s in starts:
        e = min(s + L, len(x))
        if e - s < 2:
            continue
        xb = x[s:e].unsqueeze(0).to(dev)
        nll = F.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
        out[s + 1:e] = nll.detach().cpu().numpy()
        opt.zero_grad(set_to_none=True)
        nll.mean().backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return out


def setup(size="14m"):
    """(model, init state dict, info); seed 0. All sets are tokenized with the 14M tokenizer (Pythia
    models share it; checked when another model is loaded)."""
    seed_everything(0)
    name = SIZES[size]
    model, tokenizer, info = load(name, DEFAULT_REVISION)
    if name != DEFAULT:
        from transformers import AutoTokenizer
        ref = AutoTokenizer.from_pretrained(DEFAULT, cache_dir=os.path.join(DATA, "hf"))
        assert ref.get_vocab() == tokenizer.get_vocab(), "different tokenizer"
    return model, copy_state(model), info


def copy_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def score(model, state, ids):
    """Frozen per-token losses of ids with the weights state (batched)."""
    return stream_losses(model, state, ids, 0)


class Learner:
    def __init__(self, model, init, rule="sgd", lr=1e-4, params="all"):
        if rule not in RULES:
            raise ValueError(rule)
        if rule != "sgd" and params != "all":
            raise ValueError("%s updates all parameters" % rule)
        self.model, self.rule, self.lr, self.params = model, rule, lr, params
        self.state = {k: v.clone() for k, v in init.items()}
        if rule == "adam":
            model.load_state_dict(self.state)
            self.opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    def read(self, ids):
        """Prequential losses for the next segment of the stream; the state moves on."""
        assert len(ids) % L == 0
        if self.rule == "sgd":
            out = stream_losses(self.model, self.state, ids, self.lr, params=self.params)
        else:
            out = self._read_normalized(ids)
        self.state = copy_state(self.model)
        return out

    def _read_normalized(self, ids):
        """Adam or normalized SGD, all parameters (load_state_dict copies in place, so Adam's state stays
        attached to the same tensors across segments)."""
        model = self.model
        model.load_state_dict(self.state)
        params = list(model.parameters())
        for p in params:
            p.requires_grad_(True)
        x = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        out = np.full(len(x), np.nan, dtype=np.float32)
        for s in range(0, len(x), L):
            xb = x[s:s + L].unsqueeze(0)
            nll = F.cross_entropy(model(input_ids=xb).logits[0, :-1], xb[0, 1:], reduction="none")
            out[s + 1:s + L] = nll.detach().numpy()
            for p in params:
                p.grad = None
            nll.mean().backward()
            if self.rule == "adam":
                self.opt.step()
            else:
                with torch.no_grad():
                    for p in params:
                        scale = p.norm() + (1e-3 if p.dim() == 1 else 0.0)
                        p.sub_(p.grad, alpha=float(self.lr * scale / (p.grad.norm() + 1e-12)))
        for p in params:
            p.grad = None
        return out

    def score(self, ids):
        """Frozen losses of ids with the current weights; the learner's state is untouched."""
        return score(self.model, self.state, ids)


def check():
    """Segmented (40 chunks, a snapshot score, then the rest) vs single run, main book 0, each rule."""
    from data import chunk_set
    model, init, info = setup()
    ids = chunk_set("main")[0].reshape(-1)
    for rule, lr, params in (("sgd", 1e-4, "all"), ("sgd", 1e-3, "mlp@3"), ("adam", 3e-6, "all"), ("nsgd", 1e-4, "all")):
        single = Learner(model, init, rule, lr, params).read(ids)
        lrn = Learner(model, init, rule, lr, params)
        first = lrn.read(ids[:40 * L])
        _ = lrn.score(ids[:2 * L])  # scoring in between must not disturb the stream
        seg = np.concatenate([first, lrn.read(ids[40 * L:41 * L]), lrn.read(ids[41 * L:])])
        print("%s %s lr %g: segmented vs single, max abs diff %.2e" % (rule, params, lr, np.nanmax(np.abs(seg - single))))


if __name__ == "__main__":
    check()
