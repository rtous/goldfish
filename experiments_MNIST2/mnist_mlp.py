"""MNIST classifier trained offline on the mix of all contexts of one context family, for every
configuration in configurations.json (on the CUDA GPU, or the CPU if there is none).

Each configuration has a "stream" field, a context family of contexts.py ("rotated" or
"permuted"). It selects the contexts used for offline training and the sequences used online
(sequence_<stream>.csv and sequence_<stream>_random.csv, next to this script). An optional
"sequence" field (e.g. "L1000") uses the experiment's own sequences instead:
<experiment>/sequences/sequence_<sequence>.csv, _random, _tune and _tune_random (make_sequence.py
--burst-lengths), with the contexts of "stream". Offline training is "mixed": each time
a training image is sampled, it is shown under a context drawn uniformly at random. The trained
model is the frozen baseline of the online phase.

The trained weights are saved to weights/<configuration>.pt, where online_mlp.py picks them up.
Configurations with the same model, stream, seed and training_params get the same weights, so
the model is trained once per run and the weights are saved under each of their names.

The frozen model is scored three ways (the offline columns of results.csv):
    offline_accuracy    on the bursts sequence: each test image under the context of its position
                        (sequence_<stream>_random.csv holds the same images and contexts shuffled,
                        so the accuracy is the same); comparable to online_mlp.py
    test_set_accuracy   on all 10000 test images under every context, averaged over the contexts
    oracle_accuracy     the per-context ceiling of headroom.py (one model per context, trained only
                        on it, same size) on the bursts sequence, each position scored by the model
                        of its context; blank if headroom.py has not saved those weights
Retraining clears the configuration's online results, since they belong to the old weights.

training_params (configurations.json):
    learning_rate, epochs, batch_size    required
    optimizer        "adam" (default) or "adamw"
    weight_decay     default 0
    lr_schedule      "none" (default) or "cosine" (cosine decay to 0 over all steps, per batch)
    label_smoothing  default 0
A configuration's top-level "seed" (default 0) sets torch.manual_seed; "n_contexts" (default 8)
the number of contexts of its stream.

--score-only skips training: it scores the saved weights and rewrites the offline columns only,
keeping the online results (e.g. after headroom.py saved the per-context models).

The first argument is the experiment: a subfolder (experiment1, ...; or just its number) holding
configurations.json. Results and weights are written to that subfolder.

Usage:
    python3 mnist_mlp.py experiment1                  # train all configurations
    python3 mnist_mlp.py 1 'rotated_2_*'              # only these (shell-style patterns work)
    python3 mnist_mlp.py 1 --score-only               # rescore the saved weights, no training
"""
import csv
import datetime
import fcntl
import fnmatch
import gzip
import importlib
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from contexts import DEFAULT_CONTEXTS, Family, raw_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
# Sequence kinds: the evaluation stream and its shuffled control, and the tuning stream (another
# seed, used only to tune online hyperparameters) and its shuffled control.
SEQUENCE_KINDS = ("bursts", "random", "tune", "tune_random")
SEQUENCE_SUFFIX = {"bursts": "", "random": "_random", "tune": "_tune", "tune_random": "_tune_random"}
# The sequences online_mlp.py scores by default; results.csv has a <sequence>_accuracy for each.
SEQUENCES = ("bursts", "random")
# Each experiment is a subfolder of HERE with its own configurations.json, results.csv, weights/
# and plots/. set_experiment() points these paths at one of them.
EXPERIMENT_DIR = CONFIG_FILE = RESULTS_FILE = WEIGHTS_DIR = None
# Extra metrics of the online run on the bursts sequence (online_mlp.py), see its docstring.
BURSTS_METRICS = [
    "bursts_pos1_10", "bursts_pos11_50", "bursts_pos51plus",
    "bursts_seen_first20", "bursts_new_first20",
    "bursts_retention", "bursts_retention_slow",
]
RESULT_COLUMNS = [
    "configuration", "stream", "model", "model_params", "training_params", "params", "size_bytes",
    "offline_accuracy", "test_set_accuracy", "oracle_accuracy",
    "online_params", "bursts_accuracy", "random_accuracy",
] + BURSTS_METRICS + ["updated"]
# Columns written only for configurations with a "sequence" field (experiment-local sequences with a
# first_seen column, online_mlp.py): a results.csv without them keeps the columns above exactly.
EXTRA_COLUMNS = [
    "sequence",
    "bursts_bin1_100", "bursts_bin101_500", "bursts_bin501_1000", "bursts_bin1001_2500",
    "bursts_bin2501plus",
    "bursts_first_seen", "bursts_repeated", "bursts_first_seen_frozen", "bursts_repeated_frozen",
    "bursts_first100_firstburst", "bursts_first100_returning",
]
EVAL_CHUNK = 1000  # images per forward pass when scoring many images
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def load_idx(filename):
    """Download (if needed) and parse one MNIST file in IDX format."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Downloading", filename)
        urllib.request.urlretrieve(MIRROR + filename, path)
    with gzip.open(path, "rb") as f:
        data = f.read()
    # Images have a 16-byte header, labels an 8-byte header.
    if "images" in filename:
        return np.frombuffer(data, np.uint8, offset=16).reshape(-1, 28 * 28)
    return np.frombuffer(data, np.uint8, offset=8)


def load_split(split, device):
    """Raw images (float in [0, 1], not normalized: contexts.py transforms then normalizes) and
    labels of the "train" or "test" split."""
    x = raw_tensor(load_idx(FILES[split + "_images"]), device)
    y = torch.from_numpy(load_idx(FILES[split + "_labels"]).astype(np.int64)).to(device)
    return x, y


def sequence_path(stream, kind="bursts", sequence=None, folder=None):
    """The sequence file of a stream (next to this script), or with sequence (e.g. "L1000") the
    file sequence_<sequence>... in folder (default: the current experiment's sequences/)."""
    if sequence is None:
        return os.path.join(HERE, "sequence_%s%s.csv" % (stream, SEQUENCE_SUFFIX[kind]))
    folder = folder or os.path.join(EXPERIMENT_DIR, "sequences")
    return os.path.join(folder, "sequence_%s%s.csv" % (sequence, SEQUENCE_SUFFIX[kind]))


def load_sequence(stream, kind="bursts", sequence=None, first_seen=False):
    """Read a sequence file: test-image indices, the burst (-1 if shuffled) and the context of
    each position; with first_seen=True also its first_seen column (None if it has none)."""
    path = sequence_path(stream, kind, sequence)
    if not os.path.exists(path):
        raise SystemExit("No %s: run make_sequence.py first." % path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    indices = np.array([int(r["image_index"]) for r in rows])
    bursts = np.array([int(r["burst"]) for r in rows])
    contexts = np.array([int(r["context"]) for r in rows])
    if not first_seen:
        return indices, bursts, contexts
    seen = np.array([int(r["first_seen"]) for r in rows], dtype=bool) if "first_seen" in rows[0] else None
    return indices, bursts, contexts, seen


def list_experiments():
    """Subfolders of HERE that hold a configurations.json, in name order."""
    return sorted(d for d in os.listdir(HERE)
                  if os.path.isfile(os.path.join(HERE, d, "configurations.json")))


def experiment_dir(name):
    """Folder of an experiment given as "experiment1", "1" or a path; stops if there is none."""
    for candidate in (name, os.path.join(HERE, name), os.path.join(HERE, "experiment" + name)):
        if os.path.isfile(os.path.join(candidate, "configurations.json")):
            return os.path.abspath(candidate)
    raise SystemExit("Unknown experiment '%s'. Available: %s" % (name, ", ".join(list_experiments())))


def set_experiment(name):
    """Use the given experiment's configurations.json, results.csv and weights/ from now on."""
    global EXPERIMENT_DIR, CONFIG_FILE, RESULTS_FILE, WEIGHTS_DIR
    EXPERIMENT_DIR = experiment_dir(name)
    CONFIG_FILE = os.path.join(EXPERIMENT_DIR, "configurations.json")
    RESULTS_FILE = os.path.join(EXPERIMENT_DIR, "results.csv")
    WEIGHTS_DIR = os.path.join(EXPERIMENT_DIR, "weights")
    print("Experiment:", os.path.basename(EXPERIMENT_DIR))
    return EXPERIMENT_DIR


def load_configurations(names=None):
    """Read configurations.json; if names are given, keep only those (in that order).

    A name may be a shell-style pattern such as 'rotated_*' (matches in file order)."""
    with open(CONFIG_FILE) as f:
        configs = json.load(f)
    for c in configs:
        if c.get("stream") not in ("rotated", "permuted"):
            raise SystemExit("%s: \"stream\" must be rotated or permuted." % c["name"])
    if not names:
        return configs
    selected, missing = [], []
    for n in names:
        matches = [c for c in configs if fnmatch.fnmatchcase(c["name"], n)]
        if not matches:
            missing.append(n)
        selected += [c for c in matches if c not in selected]
    if missing:
        raise SystemExit("Unknown configuration(s): %s. Available: %s"
                         % (", ".join(missing), ", ".join(c["name"] for c in configs)))
    return selected


def config_seed(config):
    return config.get("seed", 0)


def config_family(config):
    return Family(config["stream"], config.get("n_contexts", DEFAULT_CONTEXTS))


def build_model(config):
    """Import models/<config['model']>.py and call its build_model(**model_params)."""
    module = importlib.import_module("models." + config["model"])
    return module.build_model(**config.get("model_params", {}))


def describe(config):
    return "%s  (stream: %s, model: %s %s, training: %s, online: %s)" % (
        config["name"], config["stream"], config["model"], config.get("model_params", {}),
        config.get("training_params", {}), config.get("online_params", {}))


def weights_file(config):
    return os.path.join(WEIGHTS_DIR, config["name"] + ".pt")


def percontext_file(stream, hidden_size, k, n_contexts=DEFAULT_CONTEXTS):
    """Weights of headroom.py's per-context model (the ceiling) for context k of n_contexts."""
    return os.path.join(WEIGHTS_DIR, "percontext", "%s_%d_c%dof%d.pt" % (stream, hidden_size, k, n_contexts))


def save_weights(model, config, path=None):
    """Save the weights together with the model, model_params and stream that produced them."""
    path = path or weights_file(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": config["model"], "model_params": config.get("model_params", {}),
                "stream": config["stream"], "state_dict": model.state_dict()}, path)
    print("Saved weights to", path)


def load_weights(model, config, device, path=None, quiet=False):
    """Load saved weights into model; stop if missing or from another architecture or stream."""
    path = path or weights_file(config)
    if not os.path.exists(path):
        raise SystemExit("No weights %s: run python3 mnist_mlp.py (or headroom.py) first." % path)
    saved = torch.load(path, map_location=device)
    if (saved["model"] != config["model"] or saved["model_params"] != config.get("model_params", {})
            or saved.get("stream") != config["stream"]):
        raise SystemExit("%s was saved for %s %s on %s, but the configuration says %s %s on %s. "
                         "Rerun python3 mnist_mlp.py %s." % (
                             path, saved["model"], saved["model_params"], saved.get("stream"),
                             config["model"], config.get("model_params", {}), config["stream"],
                             config["name"]))
    model.load_state_dict(saved["state_dict"])
    if not quiet:
        print("Loaded weights from", path)


def get_device():
    """The CUDA GPU if there is one, else the CPU."""
    if not torch.cuda.is_available():
        print("No CUDA GPU: using the CPU")
        return torch.device("cpu")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return torch.device("cuda")


def synchronize(device):
    """Wait for queued GPU work (for timing); nothing to do on the CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def model_size_bytes(model):
    """Bytes needed to store all the model's weights and biases (and any buffers)."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)


def fast_layers(names):
    """Short name of a list of fast parameters: a layer with both its weight and bias listed
    becomes its index, e.g. ["2.weight", "2.bias"] -> "layer 2", ["0.weight", "0.bias",
    "2.weight", "2.bias"] -> "layers 0+2", ["0.weight", "0.bias", "2.bias"] -> "0+2.bias"."""
    parts = []
    for n in names:
        layer, _, kind = n.rpartition(".")
        whole = kind in ("weight", "bias") and {layer + ".weight", layer + ".bias"} <= set(names)
        part = layer if whole else n
        if part not in parts:
            parts.append(part)
    if all("." not in part for part in parts):
        return ("layer " if len(parts) == 1 else "layers ") + "+".join(parts)
    return "+".join(parts)


def params_json(config, key):
    return json.dumps(config.get(key, {}))


def read_results():
    """results.csv as {configuration: row}."""
    if not os.path.exists(RESULTS_FILE):
        return {}
    with open(RESULTS_FILE) as f:
        return {r["configuration"]: r for r in csv.DictReader(f)}


def save_result(config, values):
    """Create or update the configuration's row in results.csv with the given columns.

    Rows of configurations no longer in configurations.json are dropped; rows follow the file order."""
    names = [c["name"] for c in load_configurations()]
    with open(RESULTS_FILE + ".lock", "w") as lock:  # online runs may write in parallel
        fcntl.flock(lock, fcntl.LOCK_EX)
        _save_rows(config, values, names)


def _save_rows(config, values, names):
    rows = read_results()
    row = rows.setdefault(config["name"], {})
    row.update(values)
    row.update({"configuration": config["name"],
                "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    fields = RESULT_COLUMNS + [c for c in EXTRA_COLUMNS if any(c in r for r in rows.values())]
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[n] for n in names if n in rows)
    print("Saved results for %s to %s" % (config["name"], RESULTS_FILE))


def predict(model, x):
    """Predicted labels for many images, in chunks of EVAL_CHUNK (call under no_grad, eval mode)."""
    return torch.cat([model(x[i:i + EVAL_CHUNK]).argmax(dim=1) for i in range(0, len(x), EVAL_CHUNK)])


def context_accuracies(model, test_cache, y_test):
    """Accuracy (%) of the frozen model on the whole test set under each context."""
    model.eval()
    with torch.no_grad():
        return [(predict(model, xk) == y_test).float().mean().item() * 100 for xk in test_cache]


def make_optimizer(train, params):
    name = train.get("optimizer", "adam")
    kwargs = {"lr": train["learning_rate"], "weight_decay": train.get("weight_decay", 0)}
    if name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    raise SystemExit("Unknown offline optimizer '%s' (use adam or adamw)." % name)


def train_model(config, family, x_train, y_train, device, context=None, after_epoch=None):
    """Train a fresh model offline. context=None: mixed (each sampled image under a context drawn
    uniformly at random); context=k: every image under context k only (headroom.py's ceiling).
    after_epoch(epoch, model, mean_loss) is called after each epoch (e.g. to record curves).
    Returns the model."""
    torch.manual_seed(config_seed(config))
    train = config["training_params"]
    epochs, batch_size = train["epochs"], train["batch_size"]
    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=train.get("label_smoothing", 0.0))
    optimizer = make_optimizer(train, model.parameters())
    schedule = train.get("lr_schedule", "none")
    steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    elif schedule == "none":
        scheduler = None
    else:
        raise SystemExit("Unknown lr_schedule '%s' (use none or cosine)." % schedule)

    # A single context: transform the training set once.
    fixed = None if context is None else family.under(x_train, context)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss = torch.zeros((), device=device)
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            if fixed is None:
                ctx = torch.randint(family.n_contexts, (len(idx),), device=device)
                x = family.transform(x_train[idx], ctx)
            else:
                x = fixed[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(x), y_train[idx])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.detach() * len(idx)
        mean_loss = total_loss.item() / len(x_train)
        if after_epoch is not None:
            after_epoch(epoch, model, mean_loss)
        else:
            print("Epoch %2d/%d  loss: %.4f" % (epoch, epochs, mean_loss))
    return model


def oracle_accuracy(config, device, x_seq_by_ctx, y_seq, ctx_seq):
    """Accuracy (%) of the per-context models (headroom.py) on the sequence, each position scored
    by the model of its context; None if those weights are missing."""
    family = config_family(config)
    h = config.get("model_params", {}).get("hidden_size")
    correct = torch.zeros(len(y_seq), dtype=torch.bool, device=device)
    for k in range(family.n_contexts):
        path = percontext_file(config["stream"], h, k, family.n_contexts)
        if not os.path.exists(path):
            return None
        model = build_model(config).to(device)
        load_weights(model, config, device, path=path, quiet=True)
        model.eval()
        at = (ctx_seq == k).nonzero().flatten()
        with torch.no_grad():
            correct[at] = predict(model, x_seq_by_ctx[at]) == y_seq[at]
    return correct.float().mean().item() * 100


class StreamData:
    """Test images under every context of a stream, and the bursts sequence's inputs."""

    def __init__(self, config, device):
        self.family = config_family(config)
        x_test, self.y_test = load_split("test", device)
        self.cache = self.family.test_cache(x_test)  # [n_contexts, 10000, 784]
        indices, _, contexts = load_sequence(config["stream"], "bursts", config.get("sequence"))
        if contexts.max() >= self.family.n_contexts:
            raise SystemExit("The bursts sequence of %s has %d contexts, the configuration %d."
                             % (config["name"], contexts.max() + 1, self.family.n_contexts))
        self.idx = torch.from_numpy(indices).to(device)
        self.ctx = torch.from_numpy(contexts).to(device)
        self.x_seq = self.cache[self.ctx, self.idx]
        self.y_seq = self.y_test[self.idx]


def score(model, config, data, device):
    """Score the frozen model; return the offline columns."""
    model.eval()
    with torch.no_grad():
        accuracy = (predict(model, data.x_seq) == data.y_seq).float().mean().item() * 100
    per_context = context_accuracies(model, data.cache, data.y_test)
    set_accuracy = float(np.mean(per_context))
    oracle = oracle_accuracy(config, device, data.x_seq, data.y_seq, data.ctx)
    print("Test set, mean over contexts: %.2f%%  (per context: %s)"
          % (set_accuracy, " ".join("%.1f" % a for a in per_context)))
    print("Sequence accuracy:            %.2f%%  (comparable to online_mlp.py)" % accuracy)
    print("Per-context oracle:           %s" % ("-" if oracle is None else "%.2f%%" % oracle))
    n_params = sum(p.numel() for p in model.parameters())
    extra = {"sequence": config["sequence"]} if "sequence" in config else {}
    return dict(extra, **{
        "stream": config["stream"],
        "model": config["model"],
        "model_params": params_json(config, "model_params"),
        "training_params": params_json(config, "training_params"),
        "params": n_params,
        "size_bytes": model_size_bytes(model),
        "offline_accuracy": "%.2f" % accuracy,
        "test_set_accuracy": "%.2f" % set_accuracy,
        "oracle_accuracy": "" if oracle is None else "%.2f" % oracle,
    })


def training_key(config):
    """Configurations with the same key get the same offline weights."""
    return json.dumps([config["model"], config.get("model_params", {}), config["stream"],
                       config.get("n_contexts", DEFAULT_CONTEXTS), config_seed(config),
                       config["training_params"]], sort_keys=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__.strip() + "\n\nExperiments: " + ", ".join(list_experiments()))
    args = [a for a in sys.argv[2:] if a != "--score-only"]
    only_score = len(args) < len(sys.argv) - 2
    set_experiment(sys.argv[1])
    configs = load_configurations(args)
    device = get_device()
    x_train, y_train = (None, None) if only_score else load_split("train", device)

    data, trained = {}, {}  # StreamData per stream; trained model per training_key
    for config in configs:
        print("\n=== %s ===" % describe(config))
        stream = (config["stream"], config.get("n_contexts", DEFAULT_CONTEXTS), config.get("sequence"))
        if stream not in data:
            data[stream] = StreamData(config, device)
        model = build_model(config).to(device)
        if only_score:
            load_weights(model, config, device)
        else:
            key = training_key(config)
            if key not in trained:
                start = time.time()
                trained[key] = train_model(config, data[stream].family, x_train, y_train, device)
                synchronize(device)
                print("Training time: %.1f s" % (time.time() - start))
            else:
                print("Same model, stream, seed and training as an earlier configuration: reusing its weights")
            model.load_state_dict(trained[key].state_dict())
            save_weights(model, config)
        values = score(model, config, data[stream], device)
        if not only_score:
            # New weights: the online results of the old ones no longer apply.
            values.update({c: "" for c in RESULT_COLUMNS + (EXTRA_COLUMNS if "sequence" in config else [])
                           if c == "online_params" or c.startswith(("bursts_", "random_"))})
        save_result(config, values)


if __name__ == "__main__":
    main()
