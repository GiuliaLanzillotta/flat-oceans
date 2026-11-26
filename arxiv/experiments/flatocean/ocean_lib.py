"""Shared analysis library for the Tier 1 models (exact ocean quantities).

Everything here works in float64 on the analysis representation of a run:
saved weights cast to float64, inputs preprocessed in float64 from the raw
integer images.  Conventions:

* A layer with weight W (n x m) and bias b (n) is one parameter block
  A = [W b] of shape n x (m+1), vectorized column-major: entry (j, k) sits at
  index k*n + j.  The augmented input is [h; 1].
* The layer Jacobian block J_l has s*d rows (sample i, output a) and n(m+1)
  columns, row (i, a) = [h_i;1]^T kron (D_i^T e_a), so that
  J_l vec(A) = stack_i(D_i A [h_i;1])   (theory file (3.6), (3.10)).
* D_i = D_{l,i} is the downstream map from the layer-l preactivation to the
  output, including the gate S_{l,i} (theory file (3.5)).
* Ranks are read from the singular values of the triangular factor R of a
  blocked QR over row blocks; never from J^T J.

Validation (run `python ocean_lib.py --test`): Jacobian rows against
per-sample autograd; blocked-QR singular values against a dense SVD.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from flatocean import tier1_train as T # Compatibility layer (2026-09-24): the implementations live in the flatocean package; this module keeps the old names.
# Threads are no longer set on import: drivers call flatocean.cli.configure, and jobs.sh sets OMP_NUM_THREADS.
from flatocean.runs import RESULTS, run_dir, load_run                                  # noqa: F401
from flatocean.linalg import (SVD_DEVICE, TOL_LADDER, REL_GRID, ABS_GRID, svdvals, spectral_gap, profiles, numrank_matrix,    # noqa: F401
                              assembled_R as full_R_cpu, blocked_R)
from flatocean.blocks import mlp_structure, mlp_rows, readout_rows, unit_states, bn_slopes  # noqa: F401


def rank_ladder(sv, tols=TOL_LADDER):
    """Own-reference ladder counts with the historical key format (kept for the stored records)."""
    sv = np.asarray(sv, dtype=np.float64)
    if sv.size == 0 or sv[0] == 0:
        return {str(t): 0 for t in tols}
    return {str(t): int((sv > t * sv[0]).sum()) for t in tols}


# ----------------------------------------------------------------------------- atomic decomposition of one MLP layer
@torch.no_grad()
def atomic_layer(st, l, s, device, block=1000, verbose=False):
    """Exact atomic quantities of hidden layer l (0-based) on the first s samples of st."""
    H = st["H"][l][:s]; Dl = st["D"][l][:s]
    m1 = H.shape[1]; d, n = Dl.shape[1], Dl.shape[2]
    p = n * m1
    rh_l, sv_h = numrank_matrix(H)
    rd_l, sv_d = numrank_matrix(Dl.reshape(s * d, n))
    R = full_R_cpu(lambda idx: mlp_rows(st, l, idx), s, p, device, verbose=verbose)
    sv = svdvals(R); del R
    if sv.size < p:
        sv = np.concatenate([sv, np.zeros(p - sv.size)])
    rJ_l = rank_ladder(sv)
    krel, kabs = profiles(sv, s)
    rec = {"layer": l + 1, "n_in_aug": m1, "n_out": n, "p": p, "rows": s * d,
           "counting_nullity": max(0, p - s * d),
           "r_h": rh_l, "r_d": rd_l, "rank_J": rJ_l,
           "gap_J": {t: spectral_gap(sv, rJ_l[t]) for t in rJ_l},
           "gap_h": {t: spectral_gap(sv_h, rh_l[t]) for t in rh_l},
           "gap_d": {t: spectral_gap(sv_d, rd_l[t]) for t in rd_l},
           "k_rel": krel, "k_abs": kabs,
           **unit_states(st, l, s)}
    for t in rJ_l:
        rh, rd, rJ = rh_l[t], rd_l[t], rJ_l[t]
        rec[f"components@{t}"] = {"dim_K": p - rJ, "feature_null": n * (m1 - rh),
                                  "downstream_null": rh * (n - rd), "deficit": rh * rd - rJ}
    return rec, sv, sv_h, sv_d


@torch.no_grad()
def atomic_readout(st, s, device, block=1000):
    H = st["H"][st["L"]][:s]; d = st["d"]; m1 = H.shape[1]; p = d * m1
    rh_l, sv_h = numrank_matrix(H)
    R = full_R_cpu(lambda idx: readout_rows(st, idx), s, p, device)
    sv = svdvals(R); del R
    if sv.size < p:
        sv = np.concatenate([sv, np.zeros(p - sv.size)])
    rJ_l = rank_ladder(sv); krel, kabs = profiles(sv, s)
    rec = {"layer": "readout", "n_in_aug": m1, "n_out": d, "p": p, "rows": s * d,
           "counting_nullity": max(0, p - s * d), "r_h": rh_l, "r_d": {t: d for t in rh_l}, "rank_J": rJ_l,
           "gap_J": {t: spectral_gap(sv, rJ_l[t]) for t in rJ_l}, "gap_h": {t: spectral_gap(sv_h, rh_l[t]) for t in rh_l},
           "k_rel": krel, "k_abs": kabs, **unit_states(st, st["L"] - 1, s)}
    for t in rJ_l:
        rh, rJ = rh_l[t], rJ_l[t]
        rec[f"components@{t}"] = {"dim_K": p - rJ, "feature_null": d * (m1 - rh), "downstream_null": 0, "deficit": rh * d - rJ}
    return rec, sv, sv_h


# ----------------------------------------------------------------------------- tests
def _test_rows_vs_autograd(device, tag=""):
    run = load_run("mnist", "mlp", 0, tag=tag, device=device)
    net = run["net"]; X = run["data"]["analysis"][0][:3]; act = getattr(net, "act", torch.relu)
    st = mlp_structure(net, X)
    worst = 0.0
    for l in range(st["L"]):
        lin = net.layers[l]
        for i in range(3):
            def f(W, b):
                h = X[i:i + 1]
                for k, lk in enumerate(net.layers):
                    h = act(torch.nn.functional.linear(h, W if k == l else lk.weight, b if k == l else lk.bias))
                return net.readout(h)[0]
            JW, Jb = torch.func.jacrev(f, argnums=(0, 1))(lin.weight, lin.bias)   # d x n x m, d x n
            Aaug = torch.cat([JW, Jb[..., None]], -1)                              # d x n x (m+1)
            auto = Aaug.permute(0, 2, 1).reshape(st["d"], -1)                     # index k*n + j
            rows = mlp_rows(st, l, torch.tensor([i], device=device))
            worst = max(worst, ((auto - rows).abs().max() / auto.abs().max()).item())
    # readout
    for i in range(3):
        def g(V, b):
            h = X[i:i + 1]
            for lk in net.layers:
                h = act(lk(h))
            return torch.nn.functional.linear(h, V, b)[0]
        JV, Jb = torch.func.jacrev(g, argnums=(0, 1))(net.readout.weight, net.readout.bias)
        auto = torch.cat([JV, Jb[..., None]], -1).permute(0, 2, 1).reshape(st["d"], -1)
        rows = readout_rows(st, torch.tensor([i], device=device))
        worst = max(worst, ((auto - rows).abs().max() / auto.abs().max()).item())
    print(f"rows vs autograd ({getattr(net, 'activation', 'relu')}), all layers and readout, 3 samples: worst relative error {worst:.2e}")
    assert worst < 1e-12


def _test_blocked_vs_dense(device):
    run = load_run("mnist", "mlp", 0, device=device)
    X = run["data"]["analysis"][0][:800]
    st = mlp_structure(run["net"], X)
    l = 1
    J = torch.cat([mlp_rows(st, l, torch.arange(i, i + 200, device=device)) for i in range(0, 800, 200)], 0)
    sv_dense = svdvals(J); del J
    R = blocked_R(lambda idx: mlp_rows(st, l, idx), 800, 200, st["D"][l].shape[2] * st["H"][l].shape[1], device)
    sv_blk = svdvals(R)
    k = min(sv_dense.size, sv_blk.size)
    rel = np.abs(sv_dense[:k] - sv_blk[:k]) / sv_dense[0]
    r_dense, r_blk = rank_ladder(sv_dense), rank_ladder(sv_blk)
    print(f"blocked QR vs dense SVD (layer 2, 800 samples, 8000 x {R.shape[1]}): max |dsigma|/sigma_1 = {rel.max():.2e}; ranks dense {r_dense} blocked {r_blk}")
    assert rel.max() < 1e-10 and r_dense == r_blk


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.test:
        for tag in ("", "gelu", "leaky"):                       # every trained activation whose run exists
            if os.path.exists(os.path.join(run_dir("mnist", "mlp", 0, tag), "checkpoints", "epoch_0500.pt")):
                _test_rows_vs_autograd(a.device, tag)
        _test_blocked_vs_dense(a.device); print("all tests passed")
