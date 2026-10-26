"""MNIST classifier trained on the GPU, for every configuration in configurations.json.

After training on the training set, the model is scored on the burst sequence (sequence.csv,
built from test-set images) so its accuracy is directly comparable to online_mlp.py. The
accuracy on the plain 10000-image test set is also reported.

The trained weights are saved to weights/<configuration>.pt, where online_mlp.py picks them up.

The frozen model is scored on the sequence images (sequence.csv; sequence_random.csv holds the same
images shuffled, so the accuracy is the same) and on the plain test set. results.csv has one row per
configuration; retraining clears that row's online results, since they belong to the old weights.

training_params (configurations.json); the optional ones default to the old MLP behaviour:
    learning_rate, epochs, batch_size    required
    optimizer        "adam" (default) or "adamw"
    weight_decay     default 0
    lr_schedule      "none" (default) or "cosine" (cosine decay to 0 over all steps, per batch)
    label_smoothing  default 0
    augment_shift    default 0; N = random shift of up to N pixels in x and y, training only
A configuration's top-level "seed" (default 0) sets torch.manual_seed.

Usage:
    python3 mnist_mlp.py                 # run all configurations
    python3 mnist_mlp.py mlp_128 other   # run only the named configurations
    python3 mnist_mlp.py 'cnn_*'         # shell-style patterns work too
"""
import csv
import datetime
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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CONFIG_FILE = os.path.join(HERE, "configurations.json")
RESULTS_FILE = os.path.join(HERE, "results.csv")
SEQUENCES = {"bursts": os.path.join(HERE, "sequence.csv"),
             "random": os.path.join(HERE, "sequence_random.csv")}
WEIGHTS_DIR = os.path.join(HERE, "weights")
# One row per configuration. mnist_mlp.py fills the offline columns, online_mlp.py online_params and
# <sequence>_accuracy ("diverged" if the run diverged). offline_accuracy is the frozen model on the
# sequence images; test_set_accuracy on the plain test set.
RESULT_COLUMNS = [
    "configuration", "model", "model_params", "training_params", "params", "size_bytes",
    "offline_accuracy", "test_set_accuracy", "online_params", "bursts_accuracy", "random_accuracy",
    "updated",
]
BACKGROUND = (0 - 0.1307) / 0.3081  # value of a black pixel after normalization
EVAL_CHUNK = 1000  # images per forward pass when scoring many images (keeps CNN activations small)
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


def to_tensors(images, labels, device):
    x = torch.from_numpy(images.astype(np.float32) / 255.0)
    x = (x - 0.1307) / 0.3081  # standard MNIST mean/std normalization
    y = torch.from_numpy(labels.astype(np.int64))
    return x.to(device), y.to(device)


def load_sequence(name="bursts"):
    """Read a sequence file: test-set image indices and the burst each position belongs to."""
    if name not in SEQUENCES:
        raise SystemExit("Unknown sequence '%s'. Available: %s" % (name, ", ".join(SEQUENCES)))
    path = SEQUENCES[name]
    if not os.path.exists(path):
        raise SystemExit("No %s: run make_sequence.py first." % os.path.basename(path))
    with open(path) as f:
        rows = list(csv.DictReader(f))
    indices = np.array([int(r["image_index"]) for r in rows])
    bursts = np.array([int(r["burst"]) for r in rows])
    return indices, bursts


def load_configurations(names=None):
    """Read configurations.json; if names are given, keep only those (in that order).

    A name may be a shell-style pattern such as 'cnn_*' (matches in file order)."""
    with open(CONFIG_FILE) as f:
        configs = json.load(f)
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


def build_model(config):
    """Import models/<config['model']>.py and call its build_model(**model_params)."""
    module = importlib.import_module("models." + config["model"])
    return module.build_model(**config.get("model_params", {}))


def describe(config):
    return "%s  (model: %s %s, training: %s, online: %s)" % (
        config["name"], config["model"], config.get("model_params", {}),
        config.get("training_params", {}), config.get("online_params", {}))


def weights_file(config):
    return os.path.join(WEIGHTS_DIR, config["name"] + ".pt")


def save_weights(model, config):
    """Save the trained weights together with the model and model_params that produced them."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = weights_file(config)
    torch.save({"model": config["model"], "model_params": config.get("model_params", {}),
                "state_dict": model.state_dict()}, path)
    print("Saved weights to", path)


def load_weights(model, config, device):
    """Load weights saved by mnist_mlp.py into model; stop if missing or from another architecture."""
    path = weights_file(config)
    if not os.path.exists(path):
        raise SystemExit("No weights for %s: run python3 mnist_mlp.py %s first."
                         % (config["name"], config["name"]))
    saved = torch.load(path, map_location=device)
    if saved["model"] != config["model"] or saved["model_params"] != config.get("model_params", {}):
        raise SystemExit("%s was saved for model %s %s, but the configuration now says %s %s. "
                         "Rerun python3 mnist_mlp.py %s." % (
                             path, saved["model"], saved["model_params"], config["model"],
                             config.get("model_params", {}), config["name"]))
    model.load_state_dict(saved["state_dict"])
    print("Loaded weights from", path)


def get_device():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available - check your PyTorch install.")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return torch.device("cuda")


def model_size_bytes(model):
    """Bytes needed to store all the model's weights and biases (and any buffers)."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)


def size_report(model, accuracy_pct):
    """Print model size and accuracy-size efficiency (accuracy points per KB); return them."""
    n_params = sum(p.numel() for p in model.parameters())
    size = model_size_bytes(model)
    efficiency = accuracy_pct / (size / 1024)
    print("\nModel parameters:                   %d" % n_params)
    print("Model size:                         %d bytes (%.1f KB)" % (size, size / 1024))
    print("Accuracy-size efficiency:           %.4f %% accuracy per KB" % efficiency)
    return {"accuracy": accuracy_pct, "params": n_params, "size": size, "efficiency": efficiency}


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
    rows = read_results()
    row = rows.setdefault(config["name"], {})
    row.update(values)
    row.update({"configuration": config["name"],
                "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[n] for n in names if n in rows)
    print("Saved results for %s to %s" % (config["name"], RESULTS_FILE))


def print_summary(results):
    print("\n=== Summary ===")
    print("%-20s %9s %10s %12s %12s" % ("configuration", "accuracy", "params", "size (B)", "acc %/KB"))
    for name, r in results:
        print("%-20s %8.2f%% %10d %12d %12.4f"
              % (name, r["accuracy"], r["params"], r["size"], r["efficiency"]))


def predict(model, x):
    """Predicted labels for many images, in chunks of EVAL_CHUNK (call under no_grad, eval mode)."""
    return torch.cat([model(x[i:i + EVAL_CHUNK]).argmax(dim=1) for i in range(0, len(x), EVAL_CHUNK)])


def random_shift(x, max_shift):
    """Shift each flattened 28x28 image by its own random offset of up to max_shift pixels in x
    and y, filling the uncovered border with background (black)."""
    n = len(x)
    padded = nn.functional.pad(x.view(n, 28, 28), (max_shift,) * 4, value=BACKGROUND)
    dy = torch.randint(0, 2 * max_shift + 1, (n, 1, 1), device=x.device)
    dx = torch.randint(0, 2 * max_shift + 1, (n, 1, 1), device=x.device)
    ar = torch.arange(28, device=x.device)
    rows = (ar.view(1, 28, 1) + dy).expand(n, 28, 28)
    cols = (ar.view(1, 1, 28) + dx).expand(n, 28, 28)
    return padded[torch.arange(n, device=x.device).view(n, 1, 1), rows, cols].reshape(n, 28 * 28)


def make_optimizer(train, params):
    name = train.get("optimizer", "adam")
    kwargs = {"lr": train["learning_rate"], "weight_decay": train.get("weight_decay", 0)}
    if name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    raise SystemExit("Unknown offline optimizer '%s' (use adam or adamw)." % name)


def run(config, device, x_train, y_train, x_test, y_test, seq):
    torch.manual_seed(config_seed(config))
    train = config["training_params"]
    epochs, batch_size = train["epochs"], train["batch_size"]
    shift = train.get("augment_shift", 0)

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

    # Training
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss = torch.zeros((), device=device)  # summed on the GPU: no CPU sync per batch
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            x = random_shift(x_train[idx], shift) if shift else x_train[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(x), y_train[idx])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.detach() * len(idx)
        print("Epoch %2d/%d  loss: %.4f" % (epoch, epochs, total_loss.item() / len(x_train)))
    torch.cuda.synchronize()
    train_time = time.time() - start
    print("Training time: %.1f s" % train_time)
    save_weights(model, config)

    # Testing: on the plain test set, and on each sequence (same images and order as online_mlp.py)
    model.eval()
    with torch.no_grad():
        correct = predict(model, x_test) == y_test
    set_accuracy = correct.float().mean().item() * 100
    print("Test-set accuracy:  %.2f%%  (%d images)" % (set_accuracy, len(x_test)))
    accuracy = correct[seq].float().mean().item() * 100
    print("Sequence accuracy:  %.2f%%  (%d images, comparable to online_mlp.py)" % (accuracy, len(seq)))
    report = size_report(model, accuracy)
    save_result(config, {
        "model": config["model"],
        "model_params": params_json(config, "model_params"),
        "training_params": params_json(config, "training_params"),
        "params": report["params"],
        "size_bytes": report["size"],
        "offline_accuracy": "%.2f" % accuracy,
        "test_set_accuracy": "%.2f" % set_accuracy,
        # New weights: the online results of the old ones no longer apply.
        "online_params": "", "bursts_accuracy": "", "random_accuracy": "",
    })
    return report


def main():
    configs = load_configurations(sys.argv[1:])
    device = get_device()

    # The whole dataset is small (~200 MB as float32), so keep it all on the GPU.
    x_train, y_train = to_tensors(load_idx(FILES["train_images"]), load_idx(FILES["train_labels"]), device)
    x_test, y_test = to_tensors(load_idx(FILES["test_images"]), load_idx(FILES["test_labels"]), device)
    seq = torch.from_numpy(load_sequence("bursts")[0]).to(device)
    print("Train samples:", len(x_train), " Test samples:", len(x_test), " Sequence:", len(seq))

    results = []
    for config in configs:
        print("\n=== %s ===" % describe(config))
        results.append((config["name"], run(config, device, x_train, y_train, x_test, y_test, seq)))
    print_summary(results)


if __name__ == "__main__":
    main()
