"""Omniglot (background + evaluation merged: 50 alphabets, 1623 characters, 20 drawers each),
downscaled to 28x28, plus shared helpers: streams, weights, experiment folders.

Images: the two official zips (github.com/brendenlake/omniglot, python/) are downloaded to data/
if missing. Each 105x105 drawing is downscaled to 28x28 with area averaging, inverted (ink = 1,
background = 0), then normalized with the mean and std of the fit split. The processed arrays are
cached in data/omniglot28.npz.

Classes are numbered 0..1622 in (set, alphabet, character) order, background set first; ALPHABET
gives each class's alphabet (0..49) and ALPHABET_NAMES the names.

Split by drawer (the _NN suffix of each file):
    fit    drawers 1-10   the offline models (mixed, per-alphabet) train on these
    val    drawers 11-12  tuning streams and meta-training episodes (training drawers the
                          offline models never saw, so tuning isn't done on memorized images)
    test   drawers 13-20  the evaluation stream only
"fit" + "val" are the brief's training drawers (1-12).
"""
import csv
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
URL = "https://github.com/brendenlake/omniglot/raw/master/python/%s.zip"
SETS = ("images_background", "images_evaluation")
SPLITS = {"fit": range(1, 11), "val": range(11, 13), "test": range(13, 21)}
N_CLASSES = 1623


def _build():
    """Read both zips: images [1623, 20, 28, 28] uint8-like floats in [0, 1] (ink = 1), alphabet
    of each class, alphabet names."""
    images, alphabet, names = [], [], []
    for s in SETS:
        path = os.path.join(DATA, s + ".zip")
        if not os.path.exists(path):
            os.makedirs(DATA, exist_ok=True)
            print("Downloading %s ..." % (URL % s), flush=True)
            urllib.request.urlretrieve(URL % s, path)
        with zipfile.ZipFile(path) as z:
            files = sorted(n for n in z.namelist() if n.endswith(".png"))
            chars = {}
            for n in files:
                _, alpha, char, fname = n.split("/")
                drawer = int(fname[:-4].split("_")[1])
                chars.setdefault((alpha, char), {})[drawer] = n
            for (alpha, char) in sorted(chars):
                if alpha not in names:
                    names.append(alpha)
                draw = chars[alpha, char]
                assert sorted(draw) == list(range(1, 21)), (alpha, char)
                imgs = []
                for d in range(1, 21):
                    im = Image.open(io.BytesIO(z.read(draw[d]))).convert("L").resize((28, 28), Image.BOX)
                    imgs.append(1.0 - np.asarray(im, dtype=np.float32) / 255.0)
                images.append(np.stack(imgs))
                alphabet.append(names.index(alpha))
    return np.stack(images).astype(np.float32), np.array(alphabet), names


def load_all():
    """images [1623, 20, 784] in [0, 1] (ink = 1), alphabet [1623], alphabet names."""
    path = os.path.join(DATA, "omniglot28.npz")
    if not os.path.exists(path):
        images, alphabet, names = _build()
        np.savez_compressed(path, images=images, alphabet=alphabet, names=np.array(names))
    d = np.load(path)
    return d["images"].reshape(N_CLASSES, 20, 784), d["alphabet"], [str(n) for n in d["names"]]


_IMAGES, ALPHABET, ALPHABET_NAMES = load_all()
N_ALPHABETS = len(ALPHABET_NAMES)
_fit = _IMAGES[:, [d - 1 for d in SPLITS["fit"]]]
MEAN, STD = float(_fit.mean()), float(_fit.std())


def load_split(split):
    """Normalized images [N, 784] and labels of a split, ordered by class then drawer
    (index = class * drawers + drawer offset)."""
    drawers = [d - 1 for d in SPLITS[split]]
    x = (_IMAGES[:, drawers] - MEAN) / STD
    y = np.repeat(np.arange(N_CLASSES), len(drawers))
    return torch.from_numpy(x.reshape(-1, 784).copy()), torch.from_numpy(y)


def alphabet_classes(a):
    return np.nonzero(ALPHABET == a)[0]


def stream_path(name, shuffled=False):
    return os.path.join(HERE, "sequences", name + ("_random" if shuffled else "") + ".csv")


def load_stream(name, shuffled=False):
    """Columns image_index, burst, alphabet, occurrence of a stream, as arrays."""
    with open(stream_path(name, shuffled)) as f:
        rows = list(csv.DictReader(f))
    return tuple(np.array([int(r[k]) for r in rows]) for k in ("image_index", "burst", "alphabet", "occurrence"))


def position_in_burst(bursts):
    """1-based position of each image within its burst."""
    starts = np.r_[True, bursts[1:] != bursts[:-1]]
    start_at = np.maximum.accumulate(np.where(starts, np.arange(len(bursts)), 0))
    return np.arange(len(bursts)) - start_at + 1


def mixed_path(arch):
    return os.path.join(HERE, "weights", "mixed_%s.pt" % arch)


def load_mixed(arch):
    """State dict of the mixed model of an architecture (offline.py)."""
    if not os.path.exists(mixed_path(arch)):
        raise SystemExit("No %s: run offline.py first." % mixed_path(arch))
    return torch.load(mixed_path(arch))["state_dict"]


def experiments():
    return sorted(d for d in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, d, "configurations.json")))


def experiment_dir(name):
    """Folder of an experiment given as a folder name, a path or a number ("1")."""
    for d in [name, os.path.join(HERE, name)] + [os.path.join(HERE, e) for e in experiments()
                                                 if e == "experiment" + name or e.startswith("experiment%s_" % name)]:
        if os.path.isfile(os.path.join(d, "configurations.json")):
            return os.path.abspath(d)
    raise SystemExit("Unknown experiment '%s'. Available: %s" % (name, ", ".join(experiments())))


def load_config(exp_dir):
    with open(os.path.join(exp_dir, "configurations.json")) as f:
        return json.load(f)
