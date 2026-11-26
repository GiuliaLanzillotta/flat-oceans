"""Stored spectral factors of Jacobian blocks (results/persistence/<run>/epochXXXX_<block>.npz).

A factor file holds the singular values, the float32 right singular vectors V, the float64 kernel columns V_kernel64
(from `kernel_start`, the declared rank), the block and epoch, and an identity manifest.  Loading is authenticated: the
manifest must match the current run, checkpoint, preprocessing and ordered samples field by field (`factor_status`).
Files written before manifests existed are accepted only through a passing audit entry or an explicit legacy flag.
"""
import json
import os
import re

import numpy as np
import torch

from flatocean import provenance as P
from flatocean import runs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTOR_DIR = os.path.join(runs.INPUT_ROOT, "persistence")           # stored factorizations (inputs of most analyses)
PROTOCOL = "memos/2026-09-10_protocol_stages_atomic_transition.md, Section 5 (amendment 2026-09-10); identifier kernel-persistence-v1"
EPS_GRID = tuple(10.0 ** -k for k in range(1, 11))
DIRECTION_SEEDS = tuple(range(100, 132))
RANK_TOL = 1e-9
FACTOR_VERSION = 2                 # 2: float64 kernel columns, identity manifest (2026-09-11)


def factor_path(name, ep, blk):
    return os.path.join(FACTOR_DIR, name, f"epoch{ep:04d}_{blk}.npz")


def factor_config(blk):
    return {"factor_version": FACTOR_VERSION, "block": str(blk), "rank_tolerance": RANK_TOL, "eps_grid": list(EPS_GRID),
            "direction_seeds": list(DIRECTION_SEEDS), "coordinates": "column-major [W b] (ocean_lib / cnn_jacobian)", "protocol": PROTOCOL,
            "storage": "float32 basis, float64 kernel columns, float64 singular values"}


def factor_identity(run, rd, ep, blk):
    return P.analysis_identity(run, rd, ep, factor_config(blk))


def audit_entry_for(path):
    run_dir, fname = os.path.split(path); name = os.path.basename(run_dir)
    ap = os.path.join(FACTOR_DIR, f"{name}_audit.json")
    if not os.path.exists(ap):
        return None
    m = re.match(r"epoch(\d+)_(.+)\.npz", fname); e, blk = str(int(m.group(1))), m.group(2)
    return json.load(open(ap)).get("per_stage", {}).get(blk, {}).get(e)


def factor_status(path, run, rd, blk, allow_legacy=False, z=None, ep=None):
    """Provenance status of a factor file against the current run; raises ProvenanceError on a mismatch.
    The expected epoch is `ep` (or the epoch in the file name): a file whose stored epoch differs is refused, so a factor
    placed at the wrong path can never authenticate."""
    z = np.load(path) if z is None else z
    if ep is None:
        m = re.search(r"epoch(\d+)_", os.path.basename(path)); ep = int(m.group(1)) if m else int(z["epoch"])
    if int(z["epoch"]) != int(ep):
        raise P.ProvenanceError(f"factor {path}: stored epoch {int(z['epoch'])} differs from the requested epoch {int(ep)}")
    cur = factor_identity(run, rd, int(ep), blk)
    if "identity" in z.files:
        return P.require_match(json.loads(str(z["identity"])), cur, f"factor {path}"), cur
    return P.legacy_factor_status(path, audit_entry_for(path), cur, allow_legacy), cur


def factor_spectrum(name, ep, blk, run, rd):
    """Authenticated singular values (padded to p) without the basis: (sv, p, s, sha256, status), or None if no file."""
    path = factor_path(name, ep, blk)
    if not os.path.exists(path):
        return None
    z = np.load(path); status, _ = factor_status(path, run, rd, blk, False, z, ep=ep)
    sv = np.asarray(z["singular_values"], dtype=float); p = int(z["p"]); sv = np.concatenate([sv, np.zeros(max(0, p - sv.size))])
    return sv, p, int(z["s"]), P.sha256_file(path), status


def kernel_basis(name, ep, blk, run, rd, device="cpu"):
    """Float64 kernel columns (p x dim K) of an authenticated factor, or None."""
    path = factor_path(name, ep, blk)
    if not os.path.exists(path):
        return None
    z = np.load(path); factor_status(path, run, rd, blk, False, z, ep=ep)
    if "V_kernel64" not in z.files:
        return None
    return torch.from_numpy(z["V_kernel64"]).to(device)


def load_factor(name, ep, blk, run, rd, allow_legacy=False, return_status=False):
    """Full factor (sv, V float64 with float64 kernel columns, s, p) after the provenance check."""
    path = factor_path(name, ep, blk); z = np.load(path)
    status, _ = factor_status(path, run, rd, blk, allow_legacy, z, ep=ep)
    V = torch.from_numpy(z["V"]).double()
    if "V_kernel64" in z.files:
        r = int(z["kernel_start"]); V[:, r:] = torch.from_numpy(z["V_kernel64"])
    out = (z["singular_values"], V, int(z["s"]), int(z["p"]))
    return out + (status,) if return_status else out
