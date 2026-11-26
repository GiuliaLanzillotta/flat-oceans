"""Run directories, checkpoint loading and training stages of the Tier 1 models.

A run lives at results/tier1_models/<dataset>_<model>[_<tag>]/seed_<s>/ with config.json, data_manifest.json,
preprocessing.pt, history.json, checksums.json and checkpoints/epoch_XXXX.pt.  `load_run` returns the float64 model at
one epoch and the float64 analysis inputs (the 10k analysis set; optionally the comparison, train and test sets).
"""
import json
import os

import numpy as np
import torch

from flatocean import tier1_train as T
RESULTS = T.RESULTS                                      # trained models (tier1_train)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("FLATOCEAN_RESULTS_ROOT", os.path.join(HERE, "results"))   # every analysis output (records, spectra, bases)
INPUT_ROOT = os.environ.get("FLATOCEAN_INPUT_ROOT", RESULTS_ROOT)                        # stored factorizations consumed as inputs
# Trained models stay at tier1_train.RESULTS.  A dry run sets FLATOCEAN_RESULTS_ROOT to a scratch directory (and, to read the
# production factors, FLATOCEAN_INPUT_ROOT to the production results): nothing under the production results is then written.


def out_dir(*parts):
    """Output directory under the results root, created on demand."""
    d = os.path.join(RESULTS_ROOT, *parts); os.makedirs(d, exist_ok=True); return d
EXTRA_EPOCHS = (50, 100, 150, 200, 250)   # descriptive intermediate points (also the CNN spawn epochs), no stage label
EARLY_ERR, HIGH_ERR, INTERP_MSE = 0.10, 0.001, 1e-4
STAGE_ORDER = ["initialization", "early_fitting", "high_accuracy", "mse_interpolation", "epoch_100", "epoch_250", "late_endpoint"]
MATCHED_LABELS = {"high_accuracy": "ReLU-matched high-accuracy epoch", "mse_interpolation": "ReLU-matched interpolation epoch"}


def run_name(dataset, model, seed, tag=""):
    return f"{dataset}_{model}{'_' + tag if tag else ''}_seed{seed}"


def run_dir(dataset, model, seed, tag=""):
    name = f"{dataset}_{model}" + (f"_{tag}" if tag else "")
    return os.path.join(RESULTS, name, f"seed_{seed}")


def load_run(dataset, model, seed, tag="", epoch=500, device="cuda", sets=("analysis",), indices=None):
    """Float64 model at the given epoch plus float64 inputs; sets: any of "analysis", "comparison", "train", "test".

    `indices` (ADDITIVE, default None = unchanged): an explicit index array (or {"source": "train"|"test", "idx": [...]})
    selecting a CUSTOM sample set, added to data under the key "indices".  Default source is the training set, so passing
    indices = manifest["analysis_indices"] reproduces the "analysis" set byte-for-byte (validated in tests)."""
    out = run_dir(dataset, model, seed, tag)
    config = json.load(open(os.path.join(out, "config.json")))
    manifest = json.load(open(os.path.join(out, "data_manifest.json")))
    pre = torch.load(os.path.join(out, "preprocessing.pt"), weights_only=False)
    ck = torch.load(os.path.join(out, "checkpoints", f"epoch_{epoch:04d}.pt"), map_location="cpu", weights_only=False)
    net = T.build_model(config, torch.float64)
    net.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    net = net.to(device).eval()
    Xtr, ytr, Xte, yte, hashes = T.load_raw(dataset)
    assert hashes == manifest["raw_file_sha256"]
    data = {}
    if "analysis" in sets:
        idx = np.array(manifest["analysis_indices"])
        data["analysis"] = (T.apply_preprocessing(Xtr[idx], pre, torch.float64).to(device), torch.from_numpy(ytr[idx]).to(device), idx)
    if "comparison" in sets:
        idx = np.array(manifest["comparison_indices"])
        data["comparison"] = (T.apply_preprocessing(Xtr[idx], pre, torch.float64).to(device), torch.from_numpy(ytr[idx]).to(device), idx)
    if "train" in sets:
        data["train"] = (T.apply_preprocessing(Xtr, pre, torch.float64).to(device), torch.from_numpy(ytr).to(device), np.arange(len(ytr)))
    if "test" in sets:
        data["test"] = (T.apply_preprocessing(Xte, pre, torch.float64).to(device), torch.from_numpy(yte).to(device), np.arange(len(yte)))
    if indices is not None:
        src = "train"
        if isinstance(indices, dict):
            src = indices.get("source", "train"); idx = np.asarray(indices["idx"])
        else:
            idx = np.asarray(indices)
        Xsrc, ysrc = (Xtr, ytr) if src == "train" else (Xte, yte)
        data["indices"] = (T.apply_preprocessing(Xsrc[idx], pre, torch.float64).to(device), torch.from_numpy(ysrc[idx]).to(device), idx)
    return dict(net=net, config=config, manifest=manifest, checkpoint=ck, epoch=epoch, data=data, dir=out)


def saved_checkpoints(rd):
    return sorted(int(f[6:10]) for f in os.listdir(os.path.join(rd, "checkpoints")) if f.startswith("epoch_"))


def nearest_checkpoint(rd, epoch):
    """The saved checkpoint closest to `epoch` (earlier one on ties)."""
    return min(saved_checkpoints(rd), key=lambda e: (abs(e - epoch), e))


def stage_epochs(rd):
    """Declared stages -> milestone epoch and first saved checkpoint at or after it, plus the fit at that checkpoint."""
    h = json.load(open(os.path.join(rd, "history.json")))
    E = h["epochs"]; cfg = h["config"]
    # The training-error denominator is the number of examples the model was fit on: a fixed subset (P1) when set,
    # otherwise the dataset's training-set size (MNIST/feature-MNIST 60000, CIFAR-10 50000, SVHN 73257).
    ntrain = cfg.get("train_subset") or {"mnist": 60000, "mnist_vitb16": 60000, "cifar10": 50000, "svhn": 73257}.get(cfg["dataset"], 60000)
    saved = saved_checkpoints(rd); last = E[-1]["epoch"]
    def first_saved(ep):
        return None if ep is None else min((e for e in saved if ep <= e <= last), default=None)
    def first_epoch(pred):
        return next((e["epoch"] for e in E if pred(e)), None)
    err = lambda e: (e["train_errors"] if "train_errors" in e else round((1 - e["train_acc"]) * ntrain)) / ntrain
    ms = {"initialization": 0,
          "early_fitting": first_epoch(lambda e: err(e) <= EARLY_ERR),
          "high_accuracy": first_epoch(lambda e: err(e) <= HIGH_ERR),
          "mse_interpolation": first_epoch(lambda e: e["train_mse"] < INTERP_MSE),
          "late_endpoint": E[-1]["epoch"] if E[-1]["epoch"] >= h["config"]["epochs"] else None}
    stages = {k: {"milestone_epoch": v, "checkpoint_epoch": first_saved(v)} for k, v in ms.items()}
    for e in EXTRA_EPOCHS:
        stages[f"epoch_{e}"] = {"milestone_epoch": e, "checkpoint_epoch": e if (e <= last and e in saved) else None}
    bv = (h.get("milestones") or {}).get("best_val_epoch")              # validation-selected checkpoint (additive, 2026-10-03): added only when the run carries a val split, so runs without one are byte-identical
    if bv is not None:
        stages["best_val"] = {"milestone_epoch": bv, "checkpoint_epoch": bv if (bv <= last and bv in saved) else None}
    for k, v in stages.items():
        if v["checkpoint_epoch"] is not None:
            rec = E[v["checkpoint_epoch"]]
            v.update(train_mse=rec["train_mse"], train_errors=rec.get("train_errors", round((1 - rec["train_acc"]) * ntrain)),
                     test_acc=rec["test_acc"], test_mse=rec["test_mse"], param_norm=rec["param_norm"]["total"],
                     same_as=[kk for kk, vv in stages.items() if kk != k and vv["checkpoint_epoch"] == v["checkpoint_epoch"]])
    return stages


def matched_stages(rd, source_rd):
    """Stages of `source_rd` transferred to the run `rd` (the identity control at the ReLU-matched checkpoints): for
    every stage of the source with a checkpoint, the nearest saved checkpoint of `rd`, the requested epoch, the offset
    and the label."""
    out = {}
    for st, v in stage_epochs(source_rd).items():
        ep_req = v["checkpoint_epoch"]
        if ep_req is None:
            continue
        ep = nearest_checkpoint(rd, ep_req)
        out[st] = {"checkpoint_epoch": ep, "requested_epoch": ep_req, "epoch_offset": ep - ep_req, "label": MATCHED_LABELS.get(st, st)}
    return out
