"""Load a causal language model and its tokenizer by name through `transformers`.

The model name and revision are configuration fields, so another model (e.g.
HuggingFaceTB/SmolLM2-135M) can be swapped in without code changes. Files are downloaded to
data/hf/ (gitignored). Models are put in eval mode, in float32, on CUDA if available, else CPU
(Apple's MPS is not used: CPU keeps runs deterministic).
"""
import os
import random

import numpy as np
import torch

from data import DATA

DEFAULT, DEFAULT_REVISION = "EleutherAI/pythia-14m", "main"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load(name=DEFAULT, revision=DEFAULT_REVISION):
    """(model, tokenizer, info) with info = {model, revision, commit, device, params}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.path.join(DATA, "hf")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, cache_dir=cache, dtype=torch.float32)
    dev = device()
    model.to(dev).eval()
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(name, "config.json", cache_dir=cache, revision=revision)
    commit = os.path.basename(os.path.dirname(path)) if isinstance(path, str) else ""  # .../snapshots/<sha>/
    info = {"model": name, "revision": revision, "commit": commit,
            "device": str(dev) + ("" if dev.type == "cpu" else ":" + torch.cuda.get_device_name(dev)),
            "params": sum(p.numel() for p in model.parameters())}
    return model, tokenizer, info
