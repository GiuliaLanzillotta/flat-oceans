"""Tier 1 model production: train real-data ReLU networks and save checkpoints.

Implements experiments/TIER1_PIPELINE.md.  Training only: no ocean measurement
is computed here.  Two model families share one pipeline:

    --model mlp   PCA-256 input, ReLU MLP with biases (MNIST canonical)
    --model cnn   raw 3x32x32 input, ReLU CNN with biases and average pooling
                  (CIFAR-10 canonical)

Usage
    python tier1_train.py --dataset mnist   --model mlp --seed 0
    python tier1_train.py --dataset cifar10 --model cnn --seed 0
    python tier1_train.py ... --epochs 2 --tag smoke        # smoke test
    python tier1_train.py ... --resume                      # continue a run
    python tier1_train.py ... --verify                      # reload checks
    python tier1_train.py ... --checksum                    # sha256 manifest

A run directory results/tier1_models/<dataset>_<model>[_<tag>]/seed_<seed>/
is refused when it already holds a config unless --resume or --overwrite is
given.  Resume checks the saved configuration and data hashes first.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- paths and constants
from flatocean.paths import dataset, DATA, RESULTS as RESULTS_ROOT, recorded_result_path
DATA_ROOTS = {
    "mnist": str(dataset("MNIST") / "raw"),
    "cifar10": str(dataset("cifar-10-batches-py")),
    "svhn": str(dataset("SVHN")),
}
# Feature datasets (P3, pretrained heads): rows are precomputed FROZEN backbone features (float32), not raw pixels.
# load_raw returns the feature matrix; the MLP head trains on them via --input features (no backbone is in the graph).
# The feature file is produced once, forward-only, by scripts/analysis/extract_pretrained_features.py.
FEATURE_DATASETS = {
    "mnist_vitb16": {
        "path": str(DATA / "pretrained_features" / "mnist_vitb16.npz"),
        "base_dataset": "mnist",
        "backbone": "torchvision vit_b_16, weights ViT_B_16_Weights.IMAGENET1K_V1 (vit_b_16-c867db91.pth)",
        "feature": "encoder class-token pre-logits (heads=Identity), 768-d; MNIST->3ch, ImageClassification transform (resize 256, center-crop 224, ImageNet normalize)",
    },
    # P3 matched-backbone extension (2026-10-03): ResNet-50 on MNIST, and both backbones on NWPU-RESISC45.
    "mnist_resnet50": {
        "path": str(DATA / "pretrained_features" / "mnist_resnet50.npz"),
        "base_dataset": "mnist",
        "backbone": "torchvision resnet50, weights ResNet50_Weights.IMAGENET1K_V2 (resnet50-11ad3fa6.pth)",
        "feature": "global-average-pooled pre-logits (fc=Identity), 2048-d; MNIST->3ch, ImageClassification transform (resize 232, center-crop 224, ImageNet normalize)",
    },
    "resisc45_vitb16": {
        "path": str(DATA / "pretrained_features" / "resisc45_vitb16.npz"),
        "base_dataset": "resisc45",
        "backbone": "torchvision vit_b_16, weights ViT_B_16_Weights.IMAGENET1K_V1 (vit_b_16-c867db91.pth)",
        "feature": "encoder class-token pre-logits (heads=Identity), 768-d; timm/resisc45 256x256 RGB JPEG, ImageClassification transform (resize 256, center-crop 224, ImageNet normalize)",
    },
    "resisc45_resnet50": {
        "path": str(DATA / "pretrained_features" / "resisc45_resnet50.npz"),
        "base_dataset": "resisc45",
        "backbone": "torchvision resnet50, weights ResNet50_Weights.IMAGENET1K_V2 (resnet50-11ad3fa6.pth)",
        "feature": "global-average-pooled pre-logits (fc=Identity), 2048-d; timm/resisc45 256x256 RGB JPEG, ImageClassification transform (resize 232, center-crop 224, ImageNet normalize)",
    },
}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = str(RESULTS_ROOT / "tier1_models")

SCHEDULE = sorted({0, 1, 2, 5, 10, 20} | set(range(30, 501, 10)))
RECOVERY_EVERY = 50
INTERPOLATION_MSE = 1e-4
MARGIN_QUANTILES = [1e-4, 1e-3, 1e-2, 0.1, 0.5]
NEAR_ZERO_MARGIN = 1e-8
LOG_BINS = torch.linspace(-15.0, 1.0, 161)          # 0.1-decade histogram of log10(margin)
EXACT_QUANTILE_MAX = 20_000_000                     # exact quantiles when at most this many gates
CNN_ARCH = [("conv", 3, 32), ("conv", 32, 32), ("pool",),
            ("conv", 32, 64), ("conv", 64, 64), ("pool",),
            ("conv", 64, 128), ("pool",)]
MNIST_CNN_ARCH = [("conv", 1, 16), ("conv", 16, 16), ("pool",),      # the CIFAR-10 CNN at half width,
                  ("conv", 16, 32), ("conv", 32, 32), ("pool",),     # one input channel; MNIST is zero-padded
                  ("conv", 32, 64), ("pool",)]                       # from 28 x 28 to 32 x 32
HIGH_ACCURACY_ERROR = 1e-3                          # milestone: first epoch with training error <= 0.1 percent


def scale_cnn_arch(arch, mult):
    """Scale conv OUTPUT-channel counts by mult (the first conv's input channels are fixed); pools unchanged."""
    if float(mult) == 1.0:
        return [tuple(a) for a in arch]
    out = []; prev = None
    for layer in arch:
        if layer[0] == "conv":
            _, in_ch, out_ch = layer
            new_in = in_ch if prev is None else prev
            new_out = max(1, int(round(out_ch * mult)))
            out.append(("conv", new_in, new_out)); prev = new_out
        else:
            out.append(tuple(layer))
    return out
VOLATILE_CONFIG_KEYS = {"device", "gpu", "torch", "numpy", "python", "cuda", "epochs"}


# ----------------------------------------------------------------------------- raw data
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        ndim = magic & 0xFF
        shape = [n] + [struct.unpack(">I", f.read(4))[0] for _ in range(ndim - 1)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_raw(dataset):
    """Returns (X_train uint8 [s, D], y_train, X_test, y_test, file hashes).

    MNIST rows are 784 pixels; CIFAR-10 and SVHN rows are 3072 values in
    channel-major order (R, G, B planes of 32 x 32), so reshape(-1, 3, 32, 32)
    is correct.  A feature dataset returns precomputed float32 backbone features
    (no pixels); its hash is the feature-file sha256."""
    if dataset in FEATURE_DATASETS:
        path = FEATURE_DATASETS[dataset]["path"]
        d = np.load(path)
        Xtr = np.ascontiguousarray(d["Xtr"], dtype=np.float32); ytr = d["ytr"].astype(np.int64)
        Xte = np.ascontiguousarray(d["Xte"], dtype=np.float32); yte = d["yte"].astype(np.int64)
        return Xtr, ytr, Xte, yte, {"features_file": sha256(path)}
    root = DATA_ROOTS[dataset]
    if dataset == "mnist":
        files = {k: os.path.join(root, v) for k, v in {
            "train_x": "train-images-idx3-ubyte", "train_y": "train-labels-idx1-ubyte",
            "test_x": "t10k-images-idx3-ubyte", "test_y": "t10k-labels-idx1-ubyte"}.items()}
        for k, v in files.items():
            if not os.path.exists(v):
                files[k] = v + ".gz"
        Xtr = _read_idx(files["train_x"]).reshape(-1, 784)
        ytr = _read_idx(files["train_y"]).astype(np.int64)
        Xte = _read_idx(files["test_x"]).reshape(-1, 784)
        yte = _read_idx(files["test_y"]).astype(np.int64)
    elif dataset == "cifar10":
        files = {f"data_batch_{i}": os.path.join(root, f"data_batch_{i}") for i in range(1, 6)}
        files["test_batch"] = os.path.join(root, "test_batch")
        xs, ys = [], []
        for i in range(1, 6):
            with open(files[f"data_batch_{i}"], "rb") as f:
                d = pickle.load(f, encoding="bytes")
            xs.append(d[b"data"]); ys.append(np.array(d[b"labels"], dtype=np.int64))
        Xtr, ytr = np.concatenate(xs), np.concatenate(ys)
        with open(files["test_batch"], "rb") as f:
            d = pickle.load(f, encoding="bytes")
        Xte, yte = d[b"data"], np.array(d[b"labels"], dtype=np.int64)
    elif dataset == "svhn":
        # SVHN "cropped digits" (train/test; the 'extra' split is not used).  The .mat 'X' is (H, W, C, N) in
        # [0, 255]; transpose to (N, C, H, W) and flatten to channel-major rows so reshape(-1, 3, 32, 32) matches
        # CIFAR-10.  Labels are {1..10} with 10 meaning digit 0; (y % 10) maps 10 -> 0 and leaves 1..9 unchanged.
        from scipy.io import loadmat
        files = {"train": os.path.join(root, "train_32x32.mat"), "test": os.path.join(root, "test_32x32.mat")}
        def _load(p):
            m = loadmat(p)
            X = np.transpose(m["X"], (3, 2, 0, 1)).reshape(-1, 3 * 32 * 32).astype(np.uint8)
            y = (m["y"].astype(np.int64).reshape(-1)) % 10
            return X, y
        Xtr, ytr = _load(files["train"]); Xte, yte = _load(files["test"])
    else:
        raise ValueError(dataset)
    hashes = {k: sha256(v) for k, v in files.items()}
    return Xtr, ytr, Xte, yte, hashes


# ----------------------------------------------------------------------------- preprocessing
def _images(X, pre):
    """Rows to images.  CIFAR-10: 3 x 32 x 32.  MNIST: 1 x 28 x 28, zero-padded
    (raw background value) to 1 x 32 x 32 before standardization."""
    c, h, w = pre.get("shape", (3, 32, 32))
    X = X.reshape(-1, c, h, w)
    pad = pre.get("pad", 0)
    return F.pad(X, (pad, pad, pad, pad)) if pad else X


def fit_preprocessing(Xtr_uint8, model, q, input_mode="pca"):
    """MLP: center, PCA to q components (or raw 784 pixels, or precomputed backbone features), one global scale.
    CNN: per-channel standardization of the raw image (after zero-padding for MNIST).  Fitted in float64."""
    if input_mode == "features":
        # Xtr is a float feature matrix, not pixels: center and one global RMS scale, no division by 255.
        Xf = torch.from_numpy(np.ascontiguousarray(Xtr_uint8).astype(np.float64))
        mean = Xf.mean(0); scale = torch.sqrt(((Xf - mean) ** 2).mean())
        return {"type": "features", "mean": mean, "scale": scale, "dim": int(Xf.shape[1])}
    X = torch.from_numpy(Xtr_uint8.astype(np.float64) / 255.0)
    if model == "mlp":
        mean = X.mean(0)
        Xc = X - mean
        if input_mode == "raw":
            scale = torch.sqrt((Xc ** 2).mean())               # one global scale (RMS over all pixels)
            return {"type": "raw", "mean": mean, "scale": scale, "dim": int(X.shape[1])}
        cov = Xc.T @ Xc / X.shape[0]
        evals, evecs = torch.linalg.eigh(cov)              # ascending
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        components = evecs[:, :q].T.contiguous()           # q x D
        explained = (evals[:q] / evals.sum()).clone()
        Z = Xc @ components.T
        scale = torch.sqrt((Z ** 2).mean())                # RMS per retained coordinate
        return {"type": "pca", "mean": mean, "components": components,
                "explained_variance_ratio": explained, "eigenvalues": evals.clone(),
                "scale": scale, "q": q}
    pre = {"type": "channel"}
    if X.shape[1] == 784:
        pre.update({"shape": (1, 28, 28), "pad": 2})
    X = _images(X, pre)
    pre.update({"mean": X.mean(dim=(0, 2, 3)), "std": X.std(dim=(0, 2, 3))})
    return pre


def apply_preprocessing(X_uint8, pre, dtype):
    if pre.get("type") == "features":
        X = torch.from_numpy(np.ascontiguousarray(X_uint8).astype(np.float64))   # float features, no /255
        return ((X - pre["mean"]) / pre["scale"]).to(dtype)
    X = torch.from_numpy(X_uint8.astype(np.float64) / 255.0)
    if pre.get("type", "pca") == "pca":
        Z = ((X - pre["mean"]) @ pre["components"].T) / pre["scale"]
    elif pre.get("type") == "raw":
        Z = (X - pre["mean"]) / pre["scale"]
    else:
        X = _images(X, pre)
        Z = (X - pre["mean"][None, :, None, None]) / pre["std"][None, :, None, None]
    return Z.to(dtype)


def preprocessing_summary(pre):
    if pre.get("type", "pca") == "pca":
        return {"type": "pca", "q": pre["q"], "scale": pre["scale"].item(),
                "explained_variance_retained": pre["explained_variance_ratio"].sum().item()}
    if pre.get("type") == "raw":
        return {"type": "raw", "dim": pre["dim"], "scale": pre["scale"].item()}
    if pre.get("type") == "features":
        return {"type": "features", "dim": pre["dim"], "scale": pre["scale"].item()}
    return {"type": "channel", "mean": pre["mean"].tolist(), "std": pre["std"].tolist(),
            "shape": list(pre.get("shape", (3, 32, 32))), "pad": pre.get("pad", 0)}


def stratified_indices(y, per_class, seed, exclude=None):
    g = torch.Generator().manual_seed(seed)
    taken = set(exclude.tolist()) if exclude is not None else set()
    out = []
    for c in range(int(y.max()) + 1):
        idx = np.where(y == c)[0]
        idx = np.array([i for i in idx if i not in taken])
        perm = torch.randperm(len(idx), generator=g).numpy()
        out.append(idx[perm[:per_class]])
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------------------- models
# Common interface: forward(x, return_states=True) -> (out, Z, H) with Z the list
# of gated preactivations and H[l] the input of gated layer l; margin_scale(l, h)
# returns ||[w_unit; b]|| * ||[input; 1]|| broadcastable to Z[l]; param_groups()
# lists (name, params) per layer slot; gated_layers is the list of gated modules.

ACTIVATIONS = {
    # name: (function, derivative).  Every entry is asymptotically linear: sigma(z) -> 0 as z -> -inf and
    # sigma(z) - z -> 0 as z -> +inf (ReLU exactly, GELU and SiLU up to exponentially small terms).
    "relu": (torch.relu, lambda z: (z > 0).to(z.dtype)),
    "gelu": (lambda z: torch.nn.functional.gelu(z),
             lambda z: 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)),
    "silu": (torch.nn.functional.silu, lambda z: torch.sigmoid(z) * (1.0 + z * (1.0 - torch.sigmoid(z)))),
    # leaky ReLU with slope 0.1: affine on both sides, derivative in {0.1, 1}, never zero, so no unit can be dead
    "leaky": (lambda z: torch.nn.functional.leaky_relu(z, 0.1),
              lambda z: torch.where(z > 0, torch.ones((), dtype=z.dtype, device=z.device), torch.full((), 0.1, dtype=z.dtype, device=z.device))),
    # identity: the linear control (plan Section 0.6); derivative one, so the downstream map is sample-independent
    "identity": (lambda z: z, lambda z: torch.ones_like(z)),
}


class MLP(torch.nn.Module):
    """h_0 = x, z_l = W_l h_{l-1} + b_l, h_l = sigma(z_l), f = V h_L + b_V, with sigma one of ACTIVATIONS (ReLU by default)."""

    def __init__(self, d_in, widths, d_out, init_seed, dtype, activation="relu", batchnorm=False,
                 readout_init_scale=1.0, readout_bias=None):
        super().__init__()
        self.activation = activation; self.act, self.act_prime = ACTIVATIONS[activation]
        g = torch.Generator().manual_seed(init_seed)
        dims = [d_in] + list(widths)
        with torch.random.fork_rng(devices=[]):         # module constructors leave the global RNG untouched
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(dims[i], dims[i + 1], bias=True, dtype=dtype) for i in range(len(widths))])
            self.readout = torch.nn.Linear(dims[-1], d_out, bias=True, dtype=dtype)
            self.batchnorm = bool(batchnorm)
            self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(dims[i + 1], dtype=dtype) for i in range(len(widths))]) if batchnorm else None
        with torch.no_grad():
            for i, lin in enumerate(self.layers):
                lin.weight.copy_((torch.randn(dims[i + 1], dims[i], generator=g, dtype=torch.float64)
                                  * math.sqrt(2.0 / dims[i])).to(dtype))
                lin.bias.zero_()
            # readout_init_scale defaults to 1.0 (byte-identical: the extra * 1.0 does not touch the generator draw);
            # a smaller nonzero scale lowers the initial output magnitude (P3 RESISC45 last-hidden-layer suppression fix)
            self.readout.weight.copy_((torch.randn(d_out, dims[-1], generator=g, dtype=torch.float64)
                                       * math.sqrt(1.0 / dims[-1]) * float(readout_init_scale)).to(dtype))
            if readout_bias is None:
                self.readout.bias.zero_()
            else:                                           # e.g. the fit-set class frequencies (MSE-optimal constant baseline)
                self.readout.bias.copy_(torch.as_tensor(readout_bias, dtype=torch.float64).to(dtype))
        self.gated_layers = list(self.layers)

    def forward(self, x, return_states=False):
        Z, H = [], []
        h = x
        for l, lin in enumerate(self.layers):
            H.append(h); z = lin(h)
            if self.batchnorm: z = self.bn[l](z)          # gate acts on the BN output
            Z.append(z); h = self.act(z)
        out = self.readout(h)
        return (out, Z, H) if return_states else out

    def margin_scale(self, l, h):
        if getattr(self, "batchnorm", False):
            # BN normalizes the preactivation itself, so there is no plain W/h scale to divide by: the margin
            # diagnostic z / margin_scale then reports the RAW post-BN preactivation |z| (already ~unit-scale via BN),
            # NOT the plain model's ||W||*||h||-normalized margin.  Do not compare the two under one normalized label.
            return torch.ones(h.shape[0], self.layers[l].out_features, dtype=h.dtype, device=h.device)
        lin = self.layers[l]
        wnorm = torch.sqrt((lin.weight ** 2).sum(1) + lin.bias ** 2)      # n_l
        hnorm = torch.sqrt((h ** 2).sum(1) + 1.0)                          # batch
        return hnorm[:, None] * wnorm[None, :]

    def param_groups(self):
        groups = [(f"W{i+1}", list(lin.parameters()) + (list(self.bn[i].parameters()) if getattr(self, "batchnorm", False) else [])) for i, lin in enumerate(self.layers)]
        groups.append(("V", list(self.readout.parameters())))
        return groups


class CNN(torch.nn.Module):
    """ReLU CNN with biases and 2 x 2 average pooling, no normalization, no dropout.

    conv 3->32, conv 32->32, pool, conv 32->64, conv 64->64, pool, conv 64->128,
    pool, flatten (128 x 4 x 4), affine readout to d_out.  All convolutions are
    3 x 3, stride 1, padding 1 (PyTorch cross-correlation), one shared kernel
    and one shared bias per output channel."""

    def __init__(self, arch, d_out, init_seed, dtype, in_hw=32):
        super().__init__()
        g = torch.Generator().manual_seed(init_seed)
        self.arch = list(arch)
        with torch.random.fork_rng(devices=[]):
            self.convs = torch.nn.ModuleList(
                [torch.nn.Conv2d(a[1], a[2], 3, padding=1, bias=True, dtype=dtype) for a in arch if a[0] == "conv"])
            hw = in_hw
            for a in arch:
                if a[0] == "pool":
                    hw //= 2
            c_last = [a for a in arch if a[0] == "conv"][-1][2]
            self.flat_dim = c_last * hw * hw
            self.readout = torch.nn.Linear(self.flat_dim, d_out, bias=True, dtype=dtype)
        with torch.no_grad():
            for conv in self.convs:
                fan_in = conv.in_channels * 9
                conv.weight.copy_((torch.randn(conv.weight.shape, generator=g, dtype=torch.float64)
                                   * math.sqrt(2.0 / fan_in)).to(dtype))
                conv.bias.zero_()
            self.readout.weight.copy_((torch.randn(d_out, self.flat_dim, generator=g, dtype=torch.float64)
                                       * math.sqrt(1.0 / self.flat_dim)).to(dtype))
            self.readout.bias.zero_()
        self.gated_layers = list(self.convs)

    def forward(self, x, return_states=False):
        Z, H = [], []
        h = x
        ci = 0
        for a in self.arch:
            if a[0] == "conv":
                H.append(h); z = self.convs[ci](h); Z.append(z); h = torch.relu(z); ci += 1
            else:
                h = F.avg_pool2d(h, 2)
        out = self.readout(h.flatten(1))
        return (out, Z, H) if return_states else out

    def margin_scale(self, l, h):
        conv = self.convs[l]
        wnorm = torch.sqrt((conv.weight ** 2).sum(dim=(1, 2, 3)) + conv.bias ** 2)    # c_out
        ones = torch.ones(1, conv.in_channels, 3, 3, dtype=h.dtype, device=h.device)
        patch_sq = F.conv2d(h ** 2, ones, padding=1)                                   # batch x 1 x H x W
        return torch.sqrt(patch_sq + 1.0) * wnorm[None, :, None, None]

    def param_groups(self):
        groups = [(f"conv{i+1}", list(c.parameters())) for i, c in enumerate(self.convs)]
        groups.append(("V", list(self.readout.parameters())))
        return groups


def build_model(config, dtype):
    if config.get("model", "mlp") == "mlp":
        return MLP(config["q"], config["widths"], config["d_out"], config["init_seed"], dtype, config.get("activation", "relu"), batchnorm=config.get("batchnorm", False),
                   readout_init_scale=config.get("readout_init_scale", 1.0), readout_bias=config.get("readout_bias_values", None))
    return CNN([tuple(a) for a in config["arch"]], config["d_out"], config["init_seed"], dtype)


# ----------------------------------------------------------------------------- loss, evaluation, diagnostics
def mse_loss(out, y_onehot):
    return ((out - y_onehot) ** 2).mean()


@torch.no_grad()
def evaluate(model, X, Y_onehot, y, bs=2048):
    """Deterministic full pass.  Returns (mse per scalar output, accuracy, error count)."""
    tot, correct = 0.0, 0
    for i in range(0, X.shape[0], bs):
        out = model(X[i:i + bs])
        tot += ((out - Y_onehot[i:i + bs]) ** 2).sum().item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    n = X.shape[0]
    return tot / (n * Y_onehot.shape[1]), correct / n, n - correct


def _hist_log10(m, edges):
    """Histogram of log10(m) with the fixed edges.  torch.histc has no
    deterministic CUDA kernel, but integer counts accumulated in float64 are
    exact whatever the order, so determinism is suspended for this call only."""
    x = torch.log10(m.clamp_min(1e-15)).double()
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        return torch.histc(x, bins=len(edges) - 1, min=edges[0].item(), max=edges[-1].item())
    finally:
        torch.use_deterministic_algorithms(was)


def _quantiles_from_hist(hist, edges, qs):
    hist, edges = hist.cpu(), edges.cpu()              # cumsum has no deterministic CUDA kernel
    cum = torch.cumsum(hist, 0)
    n = cum[-1].item()
    out = []
    for q in qs:
        k = int((cum >= q * n).nonzero()[0])
        out.append(10.0 ** edges[k + 1].item())          # upper edge of the bin: conservative
    return out


@torch.no_grad()
def gate_diagnostics(model, X, prev_gates, block=1000):
    """Per-layer gate statistics on the analysis set (Section 4 of the pipeline).

    Records, per gated layer: fraction of active gates (over all units and
    positions), dead units (never active on the set), normalized-margin
    minimum and quantiles, fraction of margins below NEAR_ZERO_MARGIN, and
    the fraction of gates whose state differs from prev_gates (epoch-end gate
    disagreement).  Quantiles are exact when the layer has at most
    EXACT_QUANTILE_MAX gates and otherwise interpolated from a 0.1-decade
    histogram (conservative: upper bin edge)."""
    L = len(model.gated_layers)
    dev = X.device
    edges = LOG_BINS.to(dev)
    n_gates = [0] * L; n_active = [0] * L; n_flip = [0] * L; n_near0 = [0] * L
    mmin = [float("inf")] * L
    hist = [torch.zeros(len(edges) - 1, dtype=torch.float64, device=dev) for _ in range(L)]
    unit_active = [None] * L
    gates_out = [[] for _ in range(L)]
    margins = [[] for _ in range(L)]
    keep_exact = [True] * L
    for i in range(0, X.shape[0], block):
        _, Z, H = model.forward(X[i:i + block], return_states=True)
        for l in range(L):
            z = Z[l]; gate = z > 0
            m = (z.abs() / model.margin_scale(l, H[l])).reshape(-1)
            n_gates[l] += m.numel(); n_active[l] += int(gate.sum())
            n_near0[l] += int((m < NEAR_ZERO_MARGIN).sum())
            mmin[l] = min(mmin[l], m.min().item())
            hist[l] += _hist_log10(m, edges)
            ua = gate.reshape(gate.shape[0], gate.shape[1], -1).any(dim=(0, 2))
            unit_active[l] = ua if unit_active[l] is None else (unit_active[l] | ua)
            if prev_gates is not None:
                n_flip[l] += int((gate != prev_gates[l][i:i + block]).sum())
            gates_out[l].append(gate)
            if keep_exact[l]:
                margins[l].append(m)
                if sum(x.numel() for x in margins[l]) > EXACT_QUANTILE_MAX:
                    keep_exact[l] = False; margins[l] = []
    stats, gates = [], []
    for l in range(L):
        if keep_exact[l]:
            flat = torch.cat(margins[l])
            qs = torch.quantile(flat, torch.tensor(MARGIN_QUANTILES, dtype=flat.dtype, device=dev)).tolist()
            method = "exact"
        else:
            qs = _quantiles_from_hist(hist[l], edges, MARGIN_QUANTILES); method = "histogram"
        stats.append({
            "n_gates": n_gates[l],
            "active_fraction": n_active[l] / n_gates[l],
            "dead_units": int((~unit_active[l]).sum()),
            "dead_units_meaning": "units with no positive preactivation on the analysis set: a zero feature for ReLU only; "
                                  "for a smooth activation the unit's output and derivative are nonzero there",
            "n_units": int(unit_active[l].numel()),
            "margin_min": mmin[l],
            "margin_quantiles": dict(zip([str(q) for q in MARGIN_QUANTILES], qs)),
            "quantile_method": method,
            "margin_log10_hist": hist[l].tolist(),
            "near_zero_fraction": n_near0[l] / n_gates[l],
            "flip_fraction": (None if prev_gates is None else n_flip[l] / n_gates[l]),
        })
        gates.append(torch.cat(gates_out[l]))
    return stats, gates


# ----------------------------------------------------------------------------- augmentation (off by default)
def augment_batch(x, gen):
    """Random horizontal flip and random translation by up to 4 pixels (zero padding)."""
    n = x.shape[0]
    flip = torch.rand(n, generator=gen) < 0.5
    dx = torch.randint(-4, 5, (n,), generator=gen); dy = torch.randint(-4, 5, (n,), generator=gen)
    out = torch.empty_like(x)
    xp = F.pad(x, (4, 4, 4, 4))
    for j in range(n):
        img = xp[j, :, 4 + dy[j]:36 + dy[j], 4 + dx[j]:36 + dx[j]]
        out[j] = img.flip(-1) if flip[j] else img
    return out


# ----------------------------------------------------------------------------- checkpoints and run directory
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _sha256_file(path, chunk=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def epoch_permutation(order_seed, epoch, n):
    g = torch.Generator().manual_seed((order_seed * 1_000_003 + epoch) % (2 ** 63 - 1))
    return torch.randperm(n, generator=g)


def run_dir(args):
    name = f"{args.dataset}_{args.model}" + (f"_{args.tag}" if args.tag else "")
    return os.path.join(args.out, name, f"seed_{args.seed}")


def make_config(args, d_out, n_params, device, mlp_in=None, train_subset_n=None):
    init_seed, order_seed = 1000 + args.seed, 2000 + args.seed
    cfg = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "init_seed": init_seed, "order_seed": order_seed, "d_out": d_out, "biases": True,
        "n_params": n_params, "loss": "mse_onehot_mean", "optimizer": "adam",
        "lr": args.lr, "batch_size": args.batch_size, "weight_decay": float(args.weight_decay), "schedule": args.schedule,
        "weight_decay_blocks": list(args.weight_decay_blocks) if args.weight_decay_blocks else [], "margin_penalty": float(args.margin_penalty), "margin_tau": args.margin_tau,
        "augment": bool(args.augment), "epochs": args.epochs, "dtype": args.dtype, "tf32": False,
        "deterministic_kernels": True,
        "analysis_seed": args.analysis_seed, "analysis_per_class": args.analysis_per_class,
        "checkpoint_schedule": SCHEDULE, "recovery_every": args.recovery_every,
        "interpolation_mse": INTERPOLATION_MSE,
        "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
    }
    if args.high_accuracy_milestone:
        cfg["high_accuracy_error"] = HIGH_ACCURACY_ERROR
    if train_subset_n is not None:                       # P1 overparameterization: deterministic fixed training subset
        cfg["train_subset"] = int(train_subset_n); cfg["train_subset_seed"] = int(args.train_subset_seed)
    cfg["val_per_class"] = int(args.val_per_class); cfg["val_seed"] = int(args.val_seed)
    cfg["readout_init_scale"] = float(args.readout_init_scale); cfg["readout_bias_init"] = args.readout_bias_init
    cfg["early_stop_patience"] = int(args.early_stop_patience)
    if getattr(args, "spawn_state", None):
        st = torch.load(args.spawn_state, map_location="cpu", weights_only=False)
        cfg.update({"init_seed": st["config"]["init_seed"], "order_seed": int(args.order_seed),
                    "spawn": {"parent_run": st["parent_run"], "parent_config_seed": st["config"]["seed"], "epoch": int(st["epoch"]),
                              "parent_order_seed": st["config"]["order_seed"], "adam_policy": st["adam_policy"], "adam_policy_kind": st.get("adam_policy_kind"),
                              "state_file": os.path.abspath(args.spawn_state), "state_sha256": _sha256_file(args.spawn_state),
                              "verification": st["verification"]}})
    if args.model == "mlp":
        if args.input == "features":
            _in = int(mlp_in); _instr = f"features-{_in}, one global scale"
        elif args.input == "raw":
            _in = 784 if mlp_in is None else int(mlp_in); _instr = "raw-784, one global scale"
        else:
            _in = args.q; _instr = "pca-256, one global scale"
        cfg.update({"q": _in, "widths": [args.width] * args.depth, "input": _instr, "input_mode": args.input, "activation": args.activation, "batchnorm": bool(args.batchnorm)})
    else:
        _arch = scale_cnn_arch(MNIST_CNN_ARCH if args.dataset == "mnist" else CNN_ARCH, args.width_mult)
        if args.dataset == "mnist":
            cfg.update({"arch": [list(a) for a in _arch], "width_mult": float(args.width_mult),
                        "input": "raw 1x28x28 zero-padded to 1x32x32, per-channel standardized"})
        else:
            cfg.update({"arch": [list(a) for a in _arch], "width_mult": float(args.width_mult),
                        "input": "raw 3x32x32, per-channel standardized"})
    return cfg


def _policy_kind_of(spawn_cfg):
    """The Adam policy kind of a spawn configuration: the explicit field, or, for configurations written before it existed
    (2026-09-26), derived from the saved verification: 'reset' when it says so, 'inherited' when the replay passed, else 'failed'."""
    if spawn_cfg.get("adam_policy_kind"):
        return spawn_cfg["adam_policy_kind"]
    ver = spawn_cfg.get("verification") or {}
    if ver.get("policy") == "reset":
        return "reset"
    return "inherited" if ver.get("passed") else "failed"


def _spawn_resume_compatible(saved_spawn, new_spawn, state):
    """A child may resume from a spawn-state file that was rewritten after the child's launch (relabelled provenance)
    when the file records the child's launch-time hash in its `previous_sha256` chain and the parent, epoch, order and
    policy kind are unchanged; the child keeps its historical configuration."""
    if not saved_spawn or not new_spawn:
        return False
    keys = ("parent_run", "epoch", "parent_order_seed")
    if any(saved_spawn.get(k) != new_spawn.get(k) for k in keys):
        return False
    if _policy_kind_of(saved_spawn) != _policy_kind_of(new_spawn):                # the policy may never change across a resume
        return False
    return saved_spawn.get("state_sha256") == new_spawn.get("state_sha256") or saved_spawn.get("state_sha256") in (state.get("previous_sha256") or [])


# additive config fields introduced 2026-09-30; legacy runs predate them, so a missing value means the legacy default.
_LEGACY_CONFIG_DEFAULTS = {"input_mode": "pca", "width_mult": 1.0, "activation": "relu", "margin_penalty": 0.0, "margin_tau": None, "weight_decay_blocks": [], "batchnorm": False, "train_subset": None, "train_subset_seed": None, "val_per_class": 0, "val_seed": 54321, "early_stop_patience": 0, "readout_init_scale": 1.0, "readout_bias_init": "zero", "readout_bias_values": None, "adamw_decoupled_wd": None}

def config_mismatch(saved, new):
    keys = (set(saved) | set(new)) - VOLATILE_CONFIG_KEYS
    def _val(cfg, k):                                   # normalize missing legacy fields; preserves stored provenance
        return cfg.get(k, _LEGACY_CONFIG_DEFAULTS[k]) if k in _LEGACY_CONFIG_DEFAULTS else cfg.get(k)
    return {k: (saved.get(k), new.get(k)) for k in sorted(keys) if _val(saved, k) != _val(new, k)}


# ----------------------------------------------------------------------------- training
def set_deterministic():
    """Deterministic convolution and matmul kernels (cuDNN otherwise picks
    nondeterministic algorithms, and two float32 CNN runs with the same seed
    diverge at the first epoch)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Genuine IEEE float32: cuDNN enables TF32 (10-bit mantissa) convolutions by
    # default on Ampere GPUs, which makes float32 and float64 gates disagree at
    # the 1e-4 level instead of 1e-7.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def train(args):
    set_deterministic()
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    out = run_dir(args)
    ck = os.path.join(out, "checkpoints")
    cfg_path, hist_path = os.path.join(out, "config.json"), os.path.join(out, "history.json")
    pre_path, man_path = os.path.join(out, "preprocessing.pt"), os.path.join(out, "data_manifest.json")

    if os.path.exists(cfg_path) and not args.resume:
        if not args.overwrite:
            sys.exit(f"refusing to overwrite the existing run at {out}; pass --resume or --overwrite")
        shutil.rmtree(out)
    if args.resume and not os.path.exists(cfg_path):
        sys.exit(f"nothing to resume at {out}")
    os.makedirs(ck, exist_ok=True)

    # ---- data
    Xtr_raw, ytr_np, Xte_raw, yte_np, hashes = load_raw(args.dataset)
    spawn = torch.load(args.spawn_state, map_location="cpu", weights_only=False) if getattr(args, "spawn_state", None) else None
    val_idx = None; fit_rows = None
    if args.resume:
        pre = torch.load(pre_path, weights_only=False)
        manifest = json.load(open(man_path))
        if manifest["raw_file_sha256"] != hashes:
            sys.exit("raw data hashes differ from the manifest; refusing to resume")
        analysis_idx = np.array(manifest["analysis_indices"])
        train_subset_idx = np.array(manifest["train_subset_indices"]) if "train_subset_indices" in manifest else None
        val_idx = np.array(manifest["validation_indices"]) if "validation_indices" in manifest else None
        fit_rows = np.setdiff1d(np.arange(len(ytr_np)), val_idx) if val_idx is not None else None
    elif spawn is not None:
        # a child of a spawn: the parent's stored preprocessing and sample sets, never refitted
        pre = torch.load(os.path.join(recorded_result_path(spawn["parent_run"]), "preprocessing.pt"), weights_only=False)
        manifest = json.load(open(os.path.join(recorded_result_path(spawn["parent_run"]), "data_manifest.json")))
        if manifest["raw_file_sha256"] != hashes:
            sys.exit("raw data hashes differ from the parent's manifest; refusing to spawn")
        manifest = dict(manifest, spawned_from={"parent_run": spawn["parent_run"], "epoch": int(spawn["epoch"])})
        analysis_idx = np.array(manifest["analysis_indices"])
        train_subset_idx = np.array(manifest["train_subset_indices"]) if "train_subset_indices" in manifest else None
    else:
        # fixed validation split carved from the training pool (held out of the PCA fit AND the training set); model
        # selection / early stopping use it, never training.  Default (val_per_class 0) leaves the full pool and
        # full-pool PCA fit -> byte-identical to before.
        val_idx = stratified_indices(ytr_np, args.val_per_class, args.val_seed) if args.val_per_class else None
        fit_rows = np.setdiff1d(np.arange(len(ytr_np)), val_idx) if val_idx is not None else None
        pre = fit_preprocessing(Xtr_raw[fit_rows] if fit_rows is not None else Xtr_raw, args.model, args.q, args.input)
        analysis_idx = stratified_indices(ytr_np, args.analysis_per_class, args.analysis_seed)
        comparison_idx = stratified_indices(ytr_np, args.analysis_per_class, args.analysis_seed + 1,
                                            exclude=analysis_idx)
        # P1: a deterministic balanced subset of the TRAINING set (N // 10 per class, --train-subset-seed), fixed
        # across model seeds; the analysis and comparison sets are drawn from the full train set and are unchanged.
        train_subset_idx = stratified_indices(ytr_np, args.train_subset // 10, args.train_subset_seed) if args.train_subset else None
        manifest = {
            "dataset": args.dataset, "raw_file_sha256": hashes,
            "n_train": int(Xtr_raw.shape[0]), "n_test": int(Xte_raw.shape[0]),
            "analysis_seed": args.analysis_seed, "analysis_per_class": args.analysis_per_class,
            "analysis_indices": analysis_idx.tolist(), "comparison_indices": comparison_idx.tolist(),
            "preprocessing": preprocessing_summary(pre),
        }
        if train_subset_idx is not None:
            manifest["train_subset"] = {"n": int(len(train_subset_idx)), "requested_n": int(args.train_subset),
                                        "per_class": int(args.train_subset // 10), "seed": int(args.train_subset_seed),
                                        "scheme": "balanced per-class stratified_indices, fixed across model seeds"}
            manifest["train_subset_indices"] = train_subset_idx.tolist()
        if val_idx is not None:
            manifest["validation"] = {"n": int(len(val_idx)), "per_class": int(args.val_per_class), "seed": int(args.val_seed),
                                      "n_fit": int(len(fit_rows)),
                                      "scheme": "fixed balanced per-class stratified_indices held out of the PCA fit and the training set; model selection only"}
            manifest["validation_indices"] = val_idx.tolist()

    Xtr = apply_preprocessing(Xtr_raw, pre, dtype).to(device)
    Xte = apply_preprocessing(Xte_raw, pre, dtype).to(device)
    ytr = torch.from_numpy(ytr_np).to(device); yte = torch.from_numpy(yte_np).to(device)
    d_out = int(ytr.max().item()) + 1
    Ytr = F.one_hot(ytr, d_out).to(dtype); Yte = F.one_hot(yte, d_out).to(dtype)
    Xan = Xtr[torch.from_numpy(analysis_idx).to(device)]
    # P1: the model is fit on the subset only (analysis/eval sets unchanged); the default is the full set (byte-identical).
    # The fixed validation split (when --val-per-class > 0) is removed from whichever fit set is used.
    if train_subset_idx is not None:
        _fit_idx = np.setdiff1d(train_subset_idx, val_idx) if val_idx is not None else train_subset_idx
        _sel = torch.from_numpy(_fit_idx).to(device)
        Xtr_fit, Ytr_fit, ytr_fit = Xtr[_sel], Ytr[_sel], ytr[_sel]
    elif val_idx is not None:
        _sel = torch.from_numpy(fit_rows).to(device)
        Xtr_fit, Ytr_fit, ytr_fit = Xtr[_sel], Ytr[_sel], ytr[_sel]
    else:
        Xtr_fit, Ytr_fit, ytr_fit = Xtr, Ytr, ytr
    s = Xtr_fit.shape[0]
    # held-out validation tensors (empty when no split): for model selection / early stopping only, never training
    if val_idx is not None:
        _vsel = torch.from_numpy(val_idx).to(device)
        Xval, Yval, yval = Xtr[_vsel], Ytr[_vsel], ytr[_vsel]
    else:
        Xval = Yval = yval = None

    # ---- model, optimizer, config
    _mlp_in = int(Xtr.shape[1]) if args.input in ("raw", "features") else args.q
    _arch = scale_cnn_arch(MNIST_CNN_ARCH if args.dataset == "mnist" else CNN_ARCH, args.width_mult)
    probe = {"model": args.model, "q": _mlp_in, "widths": [args.width] * args.depth, "activation": args.activation,
             "arch": [list(a) for a in _arch], "d_out": d_out, "init_seed": 1000 + args.seed, "batchnorm": bool(args.batchnorm)}
    if spawn is not None:
        pc = spawn["config"]
        if pc.get("model", "mlp") != args.model or pc.get("dataset", args.dataset) != args.dataset or pc["d_out"] != d_out:
            sys.exit(f"spawn: the child's model kind, dataset and output size must equal the parent's ({pc.get('model')}, {pc.get('dataset')}, {pc['d_out']})")
        if args.model == "cnn":
            if [list(a) for a in pc["arch"]] != probe["arch"]:
                sys.exit(f"spawn: the child's convolutional architecture must equal the parent's ({pc['arch']})")
        elif (pc["widths"], pc.get("activation", "relu"), pc["q"], pc["d_out"]) != (probe["widths"], probe["activation"], probe["q"], d_out):
            sys.exit(f"spawn: the child's architecture must equal the parent's ({pc['widths']}, {pc.get('activation', 'relu')}, q {pc['q']})")
        if (pc["lr"], pc["batch_size"], pc["dtype"], pc["schedule"]) != (args.lr, args.batch_size, args.dtype, args.schedule):
            sys.exit("spawn: the child's optimization settings must equal the parent's")
        probe["init_seed"] = pc["init_seed"]
    # readout init (P3 fix): small nonzero weight scale + bias = fit-set class frequencies, so the net need not
    # suppress the last hidden layer to reduce one-hot MSE. Defaults (scale 1.0 / "zero") are byte-identical.
    _ro_bias = None
    if args.model == "mlp":
        if spawn is not None:                                # a spawned child inherits the parent's readout init
            probe["readout_init_scale"] = spawn["config"].get("readout_init_scale", 1.0)
            _ro_bias = spawn["config"].get("readout_bias_values", None)
        else:
            probe["readout_init_scale"] = float(args.readout_init_scale)
            if args.readout_bias_init == "class-freq":
                _cf = torch.bincount(ytr_fit, minlength=d_out).to(torch.float64) / float(ytr_fit.numel())
                _ro_bias = _cf.tolist()
        probe["readout_bias_values"] = _ro_bias
    model = build_model(probe, dtype).to(device)
    _wd_groups = {}
    if args.weight_decay_blocks:
        _valid = [name for name, _ in model.param_groups()]
        _unknown = [b for b in args.weight_decay_blocks if b not in _valid]
        if _unknown:
            sys.exit(f"--weight-decay-blocks: unknown block(s) {_unknown}; valid blocks for this model are {_valid}")
        _wd = set(args.weight_decay_blocks)
        opt = torch.optim.Adam([{"params": ps, "weight_decay": (args.weight_decay if name in _wd else 0.0)}
                                for name, ps in model.param_groups()], lr=args.lr)
        _wd_groups = {name: (args.weight_decay if name in _wd else 0.0) for name, _ in model.param_groups()}
    elif args.adamw_decoupled_wd is not None:
        # decoupled AdamW, weight matrices only (ndim >= 2); biases (and any 1-D params) excluded
        _decay = [p for _, p in model.named_parameters() if p.ndim >= 2]
        _nodecay = [p for _, p in model.named_parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([{"params": _decay, "weight_decay": float(args.adamw_decoupled_wd)},
                                 {"params": _nodecay, "weight_decay": 0.0}], lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    groups = model.param_groups()
    n_params = sum(p.numel() for p in model.parameters())
    config = make_config(args, d_out, n_params, device, mlp_in=_mlp_in,
                         train_subset_n=(int(len(train_subset_idx)) if train_subset_idx is not None else None))
    config["readout_bias_values"] = probe.get("readout_bias_values", None)   # recorded so a fresh build reproduces the init
    if spawn is not None and args.model == "mlp":
        config["readout_init_scale"] = probe.get("readout_init_scale", 1.0)
    if _wd_groups:                                    # gain arm: record the actual per-block decay (Adam coupled L2, biases included)
        config["optimizer_groups"] = _wd_groups
        config["weight_decay_note"] = "Adam coupled L2 (not decoupled AdamW); each selected block's decay applies to its weights AND biases"
    if args.adamw_decoupled_wd is not None:           # P3 matched regularization: decoupled AdamW, weights only
        config["optimizer"] = "adamw_decoupled"; config["weight_decay"] = float(args.adamw_decoupled_wd)
        config["weight_decay_scope"] = "weight matrices only (ndim>=2); biases excluded"
    config["adamw_decoupled_wd"] = args.adamw_decoupled_wd
    if args.resume:
        saved = json.load(open(cfg_path))
        diff = config_mismatch(saved, config)
        if "spawn" in diff and spawn is not None and _spawn_resume_compatible(saved.get("spawn"), config.get("spawn"), spawn):
            del diff["spawn"]                                       # the state file was rewritten after the launch and records its earlier hashes
        if diff:
            sys.exit(f"configuration differs from the saved run, refusing to resume: {diff}")
        if args.epochs < saved["epochs"]:
            sys.exit("cannot resume with a shorter horizon than the saved run")
        if args.epochs != saved["epochs"] and saved["schedule"] != "none":
            sys.exit("cannot change the horizon of a run with a learning-rate schedule")
        config = dict(saved, epochs=args.epochs, device=str(device))
        if args.epochs != saved["epochs"]:
            atomic_json(config, cfg_path)
    else:
        atomic_save(pre, pre_path); atomic_json(manifest, man_path); atomic_json(config, cfg_path)
    order_seed = config["order_seed"]
    print(f"{args.dataset} {args.model} seed {args.seed}: {n_params} parameters, {args.dtype}, {device}", flush=True)

    history, start_epoch = [], 0
    path_len = {g: 0.0 for g, _ in groups}; path_len["total"] = 0.0
    milestones = {"first_zero_error_epoch": None, "interpolation_epoch": None}
    if Xval is not None:
        milestones["best_val_epoch"] = None; milestones["best_val_acc"] = -1.0
    prev_gates = None
    prev_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()

    if spawn is not None and not args.resume:
        # the recovered state at the spawn epoch: weights, Adam moments and step counters, the parent's history so far.
        # A state whose verification failed is refused, and the state must still be bound to the parent's checkpoint on disk.
        ver = spawn.get("verification", {})
        if spawn.get("adam_policy_kind", "failed" if not ver.get("passed") else "inherited") == "failed" or not ver.get("usable", ver.get("passed", False)):
            sys.exit(f"spawn state {args.spawn_state}: verification failed ({spawn.get('adam_policy')}); refusing to train a child from it")
        pck = os.path.join(recorded_result_path(spawn["parent_run"]), "checkpoints", f"epoch_{int(spawn['epoch']):04d}.pt")
        if not os.path.exists(pck) or _sha256_file(pck) != spawn.get("parent_checkpoint_sha256"):
            sys.exit(f"spawn state {args.spawn_state}: the parent checkpoint {pck} is missing or differs from the one the state was built from")
        model.load_state_dict({k: v.to(device) for k, v in spawn["model"].items()})
        opt.load_state_dict(spawn["optimizer"])
        for st_ in opt.state.values():
            for k, v in st_.items():
                if torch.is_tensor(v):
                    st_[k] = v.to(device)
        start_epoch = int(spawn["epoch"]) + 1
        path_len = dict(spawn["path_length"]); milestones = dict(spawn["milestones"])
        milestones.setdefault("high_accuracy_epoch", None)
        history = [dict(e, inherited_from_parent=True) for e in spawn["parent_history"][: spawn["epoch"] + 1]]
        prev_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
        _, prev_gates = gate_diagnostics(model, Xan, None)
        tr_mse0, tr_acc0, tr_err0 = evaluate(model, Xtr_fit, Ytr_fit, ytr_fit)
        obj = {"epoch": int(spawn["epoch"]), "model": model.state_dict(), "train_mse": tr_mse0, "train_acc": tr_acc0, "train_errors": tr_err0,
               "milestones": dict(milestones), "path_length": dict(path_len), "config": config, "spawn_state": True}
        atomic_save(obj, os.path.join(ck, f"epoch_{spawn['epoch']:04d}.pt"))
        print(f"spawned from {spawn['parent_run']} at epoch {spawn['epoch']} (adam: {spawn['adam_policy']}), order seed {config['order_seed']}", flush=True)
    if args.resume:
        recs = sorted(f for f in os.listdir(ck) if f.startswith("recovery_epoch_"))
        if not recs:
            sys.exit("no recovery checkpoint found")
        r = torch.load(os.path.join(ck, recs[-1]), map_location=device, weights_only=False)
        hist_all = json.load(open(hist_path))["epochs"]
        if len(hist_all) < r["epoch"] + 1 or hist_all[r["epoch"]]["epoch"] != r["epoch"]:
            sys.exit(f"history.json is behind the recovery checkpoint {recs[-1]}; refusing to resume")
        model.load_state_dict(r["model"]); opt.load_state_dict(r["optimizer"])
        torch.set_rng_state(r["rng_cpu"].cpu())
        if device.type == "cuda" and r.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(r["rng_cuda"].cpu(), device)
        start_epoch = r["epoch"] + 1
        path_len = r["path_length"]; milestones = r["milestones"]
        history = hist_all[: r["epoch"] + 1]
        prev_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
        _, prev_gates = gate_diagnostics(model, Xan, None)
        print(f"resumed from {recs[-1]} at epoch {start_epoch}")

    def lr_at(epoch):
        if args.schedule == "cosine":
            return args.lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / args.epochs))
        return args.lr

    def record_epoch(epoch, t_train):
        nonlocal prev_gates, prev_flat
        t0 = time.time()
        tr_mse, tr_acc, tr_err = evaluate(model, Xtr_fit, Ytr_fit, ytr_fit)
        te_mse, te_acc, te_err = evaluate(model, Xte, Yte, yte)
        va_mse = va_acc = va_err = None
        if Xval is not None:
            va_mse, va_acc, va_err = evaluate(model, Xval, Yval, yval)
        gstats, prev_gates = gate_diagnostics(model, Xan, prev_gates)
        flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        disp = {"total": (flat - prev_flat).norm().item()}
        norms = {"total": flat.norm().item()}
        i0 = 0
        for g, ps in groups:
            n = sum(p.numel() for p in ps)
            norms[g] = flat[i0:i0 + n].norm().item()
            disp[g] = (flat[i0:i0 + n] - prev_flat[i0:i0 + n]).norm().item(); i0 += n
        prev_flat = flat.clone()
        if milestones["first_zero_error_epoch"] is None and tr_err == 0:
            milestones["first_zero_error_epoch"] = epoch
        if args.high_accuracy_milestone and milestones.get("high_accuracy_epoch") is None \
                and tr_err <= HIGH_ACCURACY_ERROR * s:
            milestones["high_accuracy_epoch"] = epoch
        if milestones["interpolation_epoch"] is None and tr_mse < INTERPOLATION_MSE:
            milestones["interpolation_epoch"] = epoch
        rec = {"epoch": epoch, "train_mse": tr_mse, "train_acc": tr_acc, "train_errors": tr_err,
               "test_mse": te_mse, "test_acc": te_acc, "test_errors": te_err,
               **({"val_mse": va_mse, "val_acc": va_acc, "val_errors": va_err} if Xval is not None else {}),
               "lr": (lr_at(epoch) if epoch > 0 else args.lr),
               "param_norm": norms, "displacement": disp, "path_length": dict(path_len),
               "gates": gstats, "wall_time_train": t_train, "wall_time_eval": time.time() - t0}
        history.append(rec)
        return rec, tr_mse, tr_acc, tr_err

    def save_checkpoint(epoch, tr_mse, tr_acc, tr_err, recovery):
        obj = {"epoch": epoch, "model": model.state_dict(), "train_mse": tr_mse, "train_acc": tr_acc,
               "train_errors": tr_err, "milestones": dict(milestones), "path_length": dict(path_len),
               "config": config}
        atomic_save(obj, os.path.join(ck, f"epoch_{epoch:04d}.pt"))
        if recovery:
            obj = dict(obj, optimizer=opt.state_dict(), rng_cpu=torch.get_rng_state(),
                       rng_cuda=(torch.cuda.get_rng_state(device) if device.type == "cuda" else None))
            atomic_save(obj, os.path.join(ck, f"recovery_epoch_{epoch:04d}.pt"))

    def maybe_update_best_val(rec):
        """If a validation split is active and this epoch's validation accuracy is the best so far, record it and save
        checkpoints/best_val.pt (model selection / early stopping only; never influences training).  Returns True on
        an improvement."""
        if Xval is None:
            return False
        va = rec.get("val_acc")
        if va is not None and va > milestones["best_val_acc"]:
            milestones["best_val_epoch"] = rec["epoch"]; milestones["best_val_acc"] = va
            atomic_save({"epoch": rec["epoch"], "model": model.state_dict(), "val_acc": va, "val_mse": rec.get("val_mse"),
                         "val_errors": rec.get("val_errors"), "test_acc": rec.get("test_acc"), "test_mse": rec.get("test_mse"),
                         "train_acc": rec.get("train_acc"), "milestones": dict(milestones), "config": config},
                        os.path.join(ck, "best_val.pt"))
            return True
        return False

    def flush_history():
        atomic_json({"config": config, "milestones": milestones, "epochs": history}, hist_path)

    def log(rec):
        g1 = rec["gates"][0]
        print(f"ep {rec['epoch']:4d}  mse {rec['train_mse']:.4e}  err {rec['train_errors']:6d}  "
              + (f"val {rec['val_acc']:.4f}  " if "val_acc" in rec else "")
              + f"test {rec['test_acc']:.4f}  flip[l1] {(g1['flip_fraction'] or 0):.2e}  "
              f"margin_min {g1['margin_min']:.1e}  path {path_len['total']:.2f}  "
              f"train {rec['wall_time_train']:.2f}s eval {rec['wall_time_eval']:.2f}s", flush=True)

    # ---- epoch 0
    no_improve = 0
    if start_epoch == 0:
        rec, tr_mse, tr_acc, tr_err = record_epoch(0, 0.0)
        maybe_update_best_val(rec)
        flush_history()
        save_checkpoint(0, tr_mse, tr_acc, tr_err, recovery=False)
        log(rec)
        start_epoch = 1

    # ---- training
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        for pg in opt.param_groups:
            pg["lr"] = lr_at(epoch)
        perm = epoch_permutation(order_seed, epoch, s).to(device)
        aug_gen = torch.Generator().manual_seed((order_seed * 7_919 + epoch) % (2 ** 63 - 1)) if args.augment else None
        for i in range(0, s, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = Xtr_fit[idx]
            if aug_gen is not None:
                xb = augment_batch(xb, aug_gen)
            before = [torch.cat([p.detach().reshape(-1) for p in ps]).clone() for _, ps in groups]
            if args.margin_penalty > 0:
                out, Zpre, _ = model(xb, return_states=True)
                loss = mse_loss(out, Ytr_fit[idx])
                mpen = sum(F.softplus(args.margin_tau - z.abs()).mean() for z in Zpre) / len(Zpre)
                loss = loss + args.margin_penalty * mpen
            else:
                loss = mse_loss(model(xb), Ytr_fit[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sq = 0.0
            for (g, ps), b in zip(groups, before):
                dn = (torch.cat([p.detach().reshape(-1) for p in ps]) - b).norm().item()
                path_len[g] += dn; sq += dn * dn
            path_len["total"] += math.sqrt(sq)
        model.eval()
        rec, tr_mse, tr_acc, tr_err = record_epoch(epoch, time.time() - t0)
        improved = maybe_update_best_val(rec)             # model selection by validation accuracy (never affects training)
        no_improve = 0 if improved else no_improve + 1
        milestone_now = epoch in (milestones["first_zero_error_epoch"], milestones["interpolation_epoch"],
                                  milestones.get("high_accuracy_epoch"))
        last_epoch = epoch == args.epochs or (args.early_stop_patience and Xval is not None and no_improve >= args.early_stop_patience)
        flush_history()                                   # history first, then the recovery state
        if epoch in SCHEDULE or milestone_now or last_epoch:
            save_checkpoint(epoch, tr_mse, tr_acc, tr_err,
                            recovery=(epoch % args.recovery_every == 0 or last_epoch))
        if epoch % args.log_every == 0 or last_epoch:
            log(rec)
        if last_epoch and epoch != args.epochs:
            print(f"early stop at epoch {epoch}: validation accuracy did not improve for {args.early_stop_patience} epochs "
                  f"(best {milestones['best_val_acc']:.4f} @ {milestones['best_val_epoch']})", flush=True)
            break
    print("milestones", milestones)


# ----------------------------------------------------------------------------- verification and checksums
def verify(args):
    """Reload checks (Section 6 of the pipeline).  Asserts: every scheduled and
    milestone checkpoint exists, all weights are finite, every checkpoint
    reproduces its recorded accuracy and error count exactly and its MSE within
    tolerance, raw-data hashes match; then the float32/float64 casting audit.
    Writes validation.json."""
    set_deterministic()
    out = run_dir(args); ck = os.path.join(out, "checkpoints")
    config = json.load(open(os.path.join(out, "config.json")))
    dtype = {"float64": torch.float64, "float32": torch.float32}[config["dtype"]]
    tol = 1e-10 if config["dtype"] == "float64" else 1e-4
    device = torch.device(args.device)
    pre = torch.load(os.path.join(out, "preprocessing.pt"), weights_only=False)
    manifest = json.load(open(os.path.join(out, "data_manifest.json")))
    hist = json.load(open(os.path.join(out, "history.json")))
    Xtr_raw, ytr_np, _, _, hashes = load_raw(args.dataset)
    assert hashes == manifest["raw_file_sha256"], "raw data hash mismatch"
    Xtr = apply_preprocessing(Xtr_raw, pre, dtype).to(device)
    ytr = torch.from_numpy(ytr_np).to(device)
    Ytr = F.one_hot(ytr, config["d_out"]).to(dtype)
    Xan = Xtr[torch.from_numpy(np.array(manifest["analysis_indices"])).to(device)]
    if "train_subset_indices" in manifest:              # P1: recorded train metrics are on the fixed subset the model was fit on
        _sel = torch.from_numpy(np.array(manifest["train_subset_indices"])).to(device)
        Xtr, Ytr, ytr = Xtr[_sel], Ytr[_sel], ytr[_sel]

    last = hist["epochs"][-1]["epoch"]
    expected = {e for e in config["checkpoint_schedule"] if e <= last} | {last}
    expected |= {e for e in hist["milestones"].values() if e is not None}
    if "spawn" in config:                                             # a spawned child starts at its spawn epoch
        e0 = int(config["spawn"]["epoch"]); expected = {e for e in expected if e >= e0} | {e0}
    files = sorted(f for f in os.listdir(ck) if f.startswith("epoch_"))
    present = {int(f[6:10]) for f in files}
    assert present == expected, f"checkpoint inventory mismatch: missing {sorted(expected - present)}, extra {sorted(present - expected)}"

    result = {"n_checkpoints": len(files), "mse_tolerance": tol, "per_checkpoint": []}
    worst = 0.0
    for f in files:
        c = torch.load(os.path.join(ck, f), map_location=device, weights_only=False)
        assert all(torch.isfinite(v).all() for v in c["model"].values()), f"{f}: non-finite weights"
        model = build_model(config, dtype).to(device)
        model.load_state_dict(c["model"])
        mse, acc, err = evaluate(model, Xtr, Ytr, ytr)
        rel = abs(mse - c["train_mse"]) / max(c["train_mse"], 1e-300)
        worst = max(worst, rel)
        assert rel <= tol, f"{f}: relative MSE discrepancy {rel:.2e} exceeds {tol}"
        assert acc == c["train_acc"], f"{f}: accuracy {acc} != {c['train_acc']}"
        if "train_errors" in c:
            assert err == c["train_errors"], f"{f}: errors {err} != {c['train_errors']}"
        result["per_checkpoint"].append({"file": f, "rel_mse_discrepancy": rel, "train_errors": err})
    print(f"{len(files)} checkpoints reload (inventory correct); worst relative MSE discrepancy {worst:.2e}")
    # casting audit on the last checkpoint.  Training representation: the
    # float32 network on float32-preprocessed inputs.  Analysis representation:
    # the float64 network (same weights cast) on inputs preprocessed in float64
    # from the raw integers, which is what every ocean measurement uses.
    c = torch.load(os.path.join(ck, files[-1]), map_location=device, weights_only=False)
    m64 = build_model(config, torch.float64).to(device)
    m64.load_state_dict({k: v.double() for k, v in c["model"].items()})
    m32 = build_model(config, torch.float32).to(device)
    m32.load_state_dict({k: v.float() for k, v in c["model"].items()})
    aidx = np.array(manifest["analysis_indices"])
    Xan64 = apply_preprocessing(Xtr_raw[aidx], pre, torch.float64).to(device)
    Xan32 = apply_preprocessing(Xtr_raw[aidx], pre, torch.float32).to(device)
    L = len(m64.gated_layers)
    n_diff, n_tot, zeros64, zeros32 = [0] * L, [0] * L, [0] * L, [0] * L
    mrange = [[float("inf"), 0.0] for _ in range(L)]
    with torch.no_grad():
        for i in range(0, Xan64.shape[0], 1000):
            _, Z64, H64 = m64.forward(Xan64[i:i + 1000], return_states=True)
            _, Z32, _ = m32.forward(Xan32[i:i + 1000], return_states=True)
            for l, (a, b) in enumerate(zip(Z64, Z32)):
                diff = (a > 0) != (b > 0)
                n_diff[l] += int(diff.sum()); n_tot[l] += a.numel()
                zeros64[l] += int((a == 0).sum()); zeros32[l] += int((b == 0).sum())
                if diff.any():
                    m = (a.abs() / m64.margin_scale(l, H64[l]))[diff]
                    mrange[l] = [min(mrange[l][0], m.min().item()), max(mrange[l][1], m.max().item())]
    result["casting_audit"] = [{"layer": l + 1, "gates": n_tot[l], "differing": n_diff[l],
                                "fraction": n_diff[l] / n_tot[l],
                                "margin_range_of_differing_gates_float64": (mrange[l] if n_diff[l] else None),
                                "exact_zero_preactivations_float64": zeros64[l],
                                "exact_zero_preactivations_float32": zeros32[l]} for l in range(L)]
    for r in result["casting_audit"]:
        print(f"casting audit layer {r['layer']}: {r['differing']} of {r['gates']} analysis-set gates differ "
              f"({r['fraction']:.2e}); float64 margins of those {r['margin_range_of_differing_gates_float64']}; "
              f"exact zeros float64 {r['exact_zero_preactivations_float64']}, float32 {r['exact_zero_preactivations_float32']}")
    result["worst_rel_mse_discrepancy"] = worst
    atomic_json(result, os.path.join(out, "validation.json"))


def checksum(args):
    out = run_dir(args); ck = os.path.join(out, "checkpoints")
    sums = {"preprocessing.pt": sha256(os.path.join(out, "preprocessing.pt"))}
    for f in sorted(os.listdir(ck)):
        if f.endswith(".pt"):
            sums[f"checkpoints/{f}"] = sha256(os.path.join(ck, f))
    atomic_json(sums, os.path.join(out, "checksums.json"))
    print(f"{len(sums)} files hashed into {out}/checksums.json")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATA_ROOTS) + list(FEATURE_DATASETS))
    ap.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--recovery-every", type=int, default=RECOVERY_EVERY, help="epochs between recovery checkpoints with the optimizer state (default 50; tests use small values)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--q", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--schedule", default="none", choices=["none", "cosine"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--activation", default="relu", choices=list(ACTIVATIONS), help="MLP activation (ablation; the CNN is ReLU)")
    ap.add_argument("--input", default="pca", choices=["pca", "raw", "features"], help="MLP input: pca (q components + one global scale), raw (pixels centered + one global scale), or features (precomputed frozen-backbone features centered + one global scale; requires a feature dataset, P3 pretrained heads)")
    ap.add_argument("--batchnorm", action="store_true", help="MLP: insert BatchNorm1d after each hidden Linear (Linear->BN->ReLU); the gate acts on the BN output")
    ap.add_argument("--width-mult", type=float, default=1.0, help="CNN channel multiplier (e.g. 0.5, 2); scales conv output channels, input channels fixed")
    ap.add_argument("--margin-penalty", type=float, default=0.0, help="controlled-radius margin arm: weight lambda of +lambda*mean_l mean_{i,units} softplus(tau-|z|) over gated hidden preactivations")
    ap.add_argument("--margin-tau", type=float, default=None, help="margin arm threshold tau (raw preactivation units); fixed from the baseline |z| distribution before training")
    ap.add_argument("--weight-decay-blocks", nargs="+", default=None, help="controlled-radius gain arm: block names to weight-decay (e.g. W2 W3 V); strength from --weight-decay")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="weight-decay (L2) strength applied to --weight-decay-blocks")
    ap.add_argument("--adamw-decoupled-wd", type=float, default=None, help="use torch.optim.AdamW with DECOUPLED weight decay of this strength on WEIGHT MATRICES ONLY (params with ndim>=2); biases are excluded (wd 0). Default None = the existing Adam path (byte-identical). Distinct from --weight-decay/--weight-decay-blocks (coupled L2 in the gradient, biases included). P3 matched-comparison regularization.")
    ap.add_argument("--high-accuracy-milestone", action="store_true",
                    help="also save the first epoch with training error <= 0.1 percent (runs from 2026-09-17 on)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--readout-init-scale", type=float, default=1.0, help="multiplier on the readout weight init std (default 1.0 = sqrt(1/width), byte-identical). A smaller value (<1) shrinks the initial output magnitude; P3 RESISC45 last-hidden-layer suppression fix.")
    ap.add_argument("--readout-bias-init", default="zero", choices=["zero", "class-freq"], help="readout bias init: zero (default, byte-identical) or class-freq (the fit-set class frequencies = the MSE-optimal constant baseline, so the net need not suppress the last hidden layer to reduce one-hot MSE).")
    ap.add_argument("--analysis-per-class", type=int, default=1000)
    ap.add_argument("--analysis-seed", type=int, default=12345)
    ap.add_argument("--train-subset", type=int, default=None, help="P1 overparameterization: fit only a deterministic balanced subset of N training examples (N // 10 per class); the analysis and test sets are unchanged. Default None = full training set (byte-identical to before)")
    ap.add_argument("--train-subset-seed", type=int, default=30000, help="seed of the balanced training subset (fixed across model seeds, so --seed varies only initialization and minibatch order)")
    ap.add_argument("--val-per-class", type=int, default=0, help="carve a FIXED seeded stratified validation split of N images per class out of the TRAINING pool, held out of fitting (and out of the PCA fit); used for model selection / early stopping only, NEVER for training. Default 0 = no split (byte-identical to before). The best-validation-accuracy checkpoint is saved as checkpoints/best_val.pt and best_val_epoch/best_val_acc recorded in milestones.")
    ap.add_argument("--val-seed", type=int, default=54321, help="seed of the fixed validation split (fixed across model seeds, so the held-out set is identical for every seed)")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="stop training early if validation accuracy has not improved for this many epochs (0 = never stop early; always complete --epochs). Requires --val-per-class > 0. The best-val checkpoint is kept regardless.")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spawn-state", default=None, help="child of a spawn: the recovered parent state (python -m flatocean lmc spawn-state); requires --order-seed")
    ap.add_argument("--order-seed", type=int, default=None, help="minibatch-order seed of a spawned child (must differ from the parent's)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--checksum", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.spawn_state and args.order_seed is None:
        ap.error("--spawn-state requires --order-seed")
    if args.margin_penalty > 0 and args.margin_tau is None:
        ap.error("--margin-penalty requires --margin-tau")
    if args.weight_decay > 0 and not args.weight_decay_blocks:
        ap.error("--weight-decay is set but --weight-decay-blocks is empty; the decay strength would be ignored")
    _is_feature_ds = args.dataset in FEATURE_DATASETS
    if _is_feature_ds and args.model != "mlp":
        ap.error(f"feature dataset {args.dataset} is an MLP-head dataset; use --model mlp")
    if _is_feature_ds and args.input != "features":
        ap.error(f"feature dataset {args.dataset} requires --input features")
    if args.input == "features" and not _is_feature_ds:
        ap.error("--input features requires a feature dataset (e.g. --dataset mnist_vitb16)")
    if args.train_subset is not None:
        if args.train_subset < 10 or args.train_subset % 10 != 0:
            ap.error("--train-subset must be a positive multiple of 10 (N // 10 examples per class)")
        if args.spawn_state:
            ap.error("--train-subset cannot be combined with --spawn-state (a spawned child inherits the parent's data)")
    if args.early_stop_patience and not args.val_per_class:
        ap.error("--early-stop-patience requires --val-per-class > 0")
    if args.val_per_class and args.spawn_state:
        ap.error("--val-per-class cannot be combined with --spawn-state (a spawned child inherits the parent's data split)")
    if args.verify:
        verify(args)
    elif args.checksum:
        checksum(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
