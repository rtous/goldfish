"""Shared helpers: MNIST files, streams, the mixed model's weights, experiment folders."""
import csv
import gzip
import json
import os
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def load_idx(name):
    """One MNIST file ("train-images", "t10k-labels", ...), downloaded to data/ if needed."""
    path = os.path.join(HERE, "data", name + "-idx%d-ubyte.gz" % (3 if "images" in name else 1))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(MIRROR + os.path.basename(path), path)
    with gzip.open(path, "rb") as f:
        data = f.read()
    if "images" in name:
        return np.frombuffer(data, np.uint8, offset=16).reshape(-1, 28 * 28)
    return np.frombuffer(data, np.uint8, offset=8)


def load_split(split):
    """Raw images (float in [0, 1], [N, 784]) and labels of "train" or "test"."""
    prefix = "train" if split == "train" else "t10k"
    x = torch.from_numpy(load_idx(prefix + "-images").astype(np.float32) / 255.0)
    y = torch.from_numpy(load_idx(prefix + "-labels").astype(np.int64))
    return x, y


def stream_path(name, shuffled=False):
    return os.path.join(HERE, "sequences", name + ("_random" if shuffled else "") + ".csv")


def load_stream(name, shuffled=False):
    """Test-image index, burst (-1 when shuffled) and context of each position."""
    with open(stream_path(name, shuffled)) as f:
        rows = list(csv.DictReader(f))
    return tuple(np.array([int(r[k]) for r in rows]) for k in ("image_index", "burst", "context"))


def position_in_burst(bursts):
    """1-based position of each image within its burst."""
    starts = np.r_[True, bursts[1:] != bursts[:-1]]
    start_at = np.maximum.accumulate(np.where(starts, np.arange(len(bursts)), 0))
    return np.arange(len(bursts)) - start_at + 1


def mixed_path(h):
    return os.path.join(HERE, "weights", "mixed_h%d.pt" % h)


def load_mixed(h):
    """State dict of the shared mixed model of hidden size h (offline.py)."""
    if not os.path.exists(mixed_path(h)):
        raise SystemExit("No %s: run offline.py first." % mixed_path(h))
    return torch.load(mixed_path(h))["state_dict"]


def load_init(init, h):
    """Starting weights: the mixed model if init is None, else the state dict saved at init (a path
    relative to experiments_MNIST2, with <h> replaced by the size)."""
    if init is None:
        return load_mixed(h)
    path = os.path.join(HERE, init.replace("<h>", str(h)))
    if not os.path.exists(path):
        raise SystemExit("No %s." % path)
    saved = torch.load(path)
    return saved.get("state_dict", saved)


def experiments():
    return sorted(d for d in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, d, "configurations.json")))


def experiment_dir(name):
    """Folder of an experiment given as a folder name, a path or a number ("3")."""
    for d in [name, os.path.join(HERE, name)] + [os.path.join(HERE, e) for e in experiments()
                                                 if e == "experiment" + name or e.startswith("experiment%s_" % name)]:
        if os.path.isfile(os.path.join(d, "configurations.json")):
            return os.path.abspath(d)
    raise SystemExit("Unknown experiment '%s'. Available: %s" % (name, ", ".join(experiments())))


def load_config(exp_dir):
    with open(os.path.join(exp_dir, "configurations.json")) as f:
        return json.load(f)
