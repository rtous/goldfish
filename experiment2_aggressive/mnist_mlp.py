"""MNIST classifier trained on the GPU, for every configuration in configurations.json.

After training on the training set, the model is scored on the burst sequence (sequence.csv,
built from test-set images) so its accuracy is directly comparable to online_mlp.py. The
accuracy on the plain 10000-image test set is also reported.

The trained weights are saved to weights/<configuration>.pt, where online_mlp.py picks them up.

The frozen model is scored on every sequence file that exists (sequence.csv -> "bursts",
sequence_random.csv -> "random"), one results.csv row per (configuration, sequence).

Usage:
    python3 mnist_mlp.py                 # run all configurations
    python3 mnist_mlp.py mlp_128 other   # run only the named configurations
"""
import csv
import datetime
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
# One row per (configuration, sequence). mnist_mlp.py fills the test_* columns, online_mlp.py the online_* ones.
# test_accuracy/test_efficiency are measured on the sequence; test_set_accuracy on the plain test set.
RESULT_COLUMNS = [
    "configuration", "sequence", "model", "model_params", "training_params", "params", "size_bytes",
    "test_accuracy", "test_efficiency", "test_set_accuracy", "test_time_s",
    "online_accuracy", "online_first_in_burst_accuracy", "online_efficiency", "online_time_s",
    "updated",
]
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


def parse_args(argv):
    """Split the command line into configuration names and an optional --sequence NAME.

    The sequence is None when --sequence is not given."""
    names, sequence = [], None
    it = iter(argv)
    for arg in it:
        if arg == "--sequence":
            sequence = next(it, None)
            if not sequence:
                raise SystemExit("--sequence needs a name: %s" % ", ".join(SEQUENCES))
        elif arg.startswith("--sequence="):
            sequence = arg.split("=", 1)[1]
        else:
            names.append(arg)
    return names, sequence


def load_configurations(names=None):
    """Read configurations.json; if names are given, keep only those (in that order)."""
    with open(CONFIG_FILE) as f:
        configs = json.load(f)
    if not names:
        return configs
    by_name = {c["name"]: c for c in configs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit("Unknown configuration(s): %s. Available: %s"
                         % (", ".join(missing), ", ".join(by_name)))
    return [by_name[n] for n in names]


def build_model(config):
    """Import models/<config['model']>.py and call its build_model(**model_params)."""
    module = importlib.import_module("models." + config["model"])
    return module.build_model(**config.get("model_params", {}))


def describe(config):
    return "%s  (model: %s %s, training: %s)" % (
        config["name"], config["model"], config.get("model_params", {}),
        config.get("training_params", {}))


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


def save_result(config, values, sequence="bursts"):
    """Create or update the (configuration, sequence) row in results.csv with the given columns."""
    rows = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["configuration"] == config["name"]
                and (r.get("sequence") or "bursts") == sequence), None)
    if row is None:
        row = {}
        rows.append(row)
    row.update(values)
    row.update({
        "configuration": config["name"],
        "sequence": sequence,
        "model": config["model"],
        "model_params": json.dumps(config.get("model_params", {})),
        "training_params": json.dumps(config.get("training_params", {})),
        "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results for %s (%s) to %s" % (config["name"], sequence, RESULTS_FILE))


def print_summary(results):
    print("\n=== Summary ===")
    print("%-20s %9s %10s %12s %12s" % ("configuration", "accuracy", "params", "size (B)", "acc %/KB"))
    for name, r in results:
        print("%-20s %8.2f%% %10d %12d %12.4f"
              % (name, r["accuracy"], r["params"], r["size"], r["efficiency"]))


def run(config, device, x_train, y_train, x_test, y_test, seqs):
    torch.manual_seed(0)
    train = config["training_params"]
    epochs, batch_size = train["epochs"], train["batch_size"]

    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])

    # Training
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print("Epoch %2d/%d  loss: %.4f" % (epoch, epochs, total_loss / len(x_train)))
    torch.cuda.synchronize()
    train_time = time.time() - start
    print("Training time: %.1f s" % train_time)
    save_weights(model, config)

    # Testing: on the plain test set, and on each sequence (same images and order as online_mlp.py)
    model.eval()
    with torch.no_grad():
        correct = model(x_test).argmax(dim=1) == y_test
    set_accuracy = correct.float().mean().item() * 100
    print("Test-set accuracy:  %.2f%%  (%d images)" % (set_accuracy, len(x_test)))
    report = None
    for name, seq in seqs.items():
        accuracy = correct[seq].float().mean().item() * 100
        print("Sequence '%s' accuracy:  %.2f%%  (%d images, comparable to online_mlp.py)"
              % (name, accuracy, len(seq)))
        r = size_report(model, accuracy)
        report = report or r
        save_result(config, {
            "params": r["params"],
            "size_bytes": r["size"],
            "test_accuracy": "%.2f" % accuracy,
            "test_efficiency": "%.4f" % r["efficiency"],
            "test_set_accuracy": "%.2f" % set_accuracy,
            "test_time_s": "%.1f" % train_time,
        }, sequence=name)
    return report


def main():
    names, _ = parse_args(sys.argv[1:])
    configs = load_configurations(names)
    device = get_device()

    # The whole dataset is small (~200 MB as float32), so keep it all on the GPU.
    x_train, y_train = to_tensors(load_idx(FILES["train_images"]), load_idx(FILES["train_labels"]), device)
    x_test, y_test = to_tensors(load_idx(FILES["test_images"]), load_idx(FILES["test_labels"]), device)
    seqs = {name: torch.from_numpy(load_sequence(name)[0]).to(device)
            for name, path in SEQUENCES.items() if os.path.exists(path)}
    if not seqs:
        raise SystemExit("No sequence files: run make_sequence.py first.")
    print("Train samples:", len(x_train), " Test samples:", len(x_test),
          " Sequences:", ", ".join("%s (%d)" % (n, len(s)) for n, s in seqs.items()))

    results = []
    for config in configs:
        print("\n=== %s ===" % describe(config))
        results.append((config["name"], run(config, device, x_train, y_train, x_test, y_test, seqs)))
    print_summary(results)


if __name__ == "__main__":
    main()
