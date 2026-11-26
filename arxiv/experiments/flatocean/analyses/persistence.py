"""Kernel persistence, rotation, and positive-tail conversion across training stages.

Protocol: memos/2026-09-10_protocol_stages_atomic_transition.md, Section 5
(amendment).  Identifier: kernel-persistence-v1.

Phase "factor": for each stage of a run and each requested block, assemble
the layer Jacobian J_l (s d x p_l) on the CPU, take its triangular factor R
by one QR, and the full SVD of R with right singular vectors; store the
singular values (float64) and the right singular basis V (float32) under
results/persistence/<run>/epoch<e>_<block>.npz (large; not committed).  The
right singular vectors of R are those of J_l, in the fixed Euclidean
parameter coordinates of the block (column-major [W b], see ocean_lib).

Phase "overlap": for stage pairs (t, t') (successive distinct stages and
each earlier stage to the late endpoint) and each block, with
  K_t         = right singular subspace with sigma <= 1e-9 sigma_1 (declared rank rule),
  E_t(eps)    = right singular subspace of J_l/sqrt(s) with sigma <= eps (includes K_t),
  E_t^+(eps)  = E_t(eps) minus K_t (the positive tail),
report dim, Q(U,V) = tr(P_U P_V)/dim U, its reverse, projector distance
d = ||P_U - P_V||_F/sqrt(2), principal-angle summaries, the exposure at both
stages of unit directions sampled from K_t and E_t^+(eps) by projecting the
32 ambient probes (seeds 100-131), and the overlap of K_t' with the
dead-unit coordinate subspaces of the block.

    python -m flatocean kernel-persistence factor  --seeds 0 --blocks 2 3 readout
    python -m flatocean kernel-persistence overlap --seeds 0 --blocks 2 3 readout
    python -m flatocean kernel-persistence factor  --dataset cifar10 --model cnn --blocks conv3 conv4 readout
    python -m flatocean kernel-persistence test

CNN blocks (conv1..conv5, readout) use the tied Jacobian of cnn_jacobian.py (validated against
per-sample autodiff, the adjoint identity, and the tied/untied relation) in the same streamed
QR; for them the factor file also records the feature and downstream ranks r_h, r_d of the
three-component identity.
"""
# Ported unchanged from experiments/exp_kernel_persistence.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean kernel-persistence).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os
import re
import subprocess
import time

import numpy as np
from flatocean import records, runs
import torch

from flatocean import cnn_jacobian as CJ
from flatocean import frozen_lib as FL
from flatocean import provenance as P
from flatocean import ocean_lib as O
from flatocean import tier1_train as T
from flatocean.runs import stage_epochs

from flatocean.factors import (FACTOR_DIR, PROTOCOL, EPS_GRID, DIRECTION_SEEDS, RANK_TOL, FACTOR_VERSION,      # noqa: F401
                               factor_path, factor_config, factor_identity, audit_entry_for, factor_status, factor_spectrum, kernel_basis, load_factor)
from flatocean.runs import STAGE_ORDER, out_dir                                                                          # noqa: F401
OUT = out_dir("persistence")            # written here: factors (factor phase), overlap and audit records; read through FACTOR_DIR
EPS_SAMPLED = (1e-3, 1e-4, 1e-6)
LP64_LIMIT = 26000                 # largest square factor MKL gesdd with vectors handles under 32-bit workspace indexing


# ----------------------------------------------------------------------------- subspace statistics
def overlap_stats(VU, VV, angles=True):
    """VU: p x kU, VV: p x kV orthonormal bases (torch, float64).  Returns Q(U,V), Q(V,U), d, and (optionally) principal angles."""
    kU, kV = VU.shape[1], VV.shape[1]
    if kU == 0 or kV == 0:
        # tr(P_U P_V) = 0 when either projector is zero; only the statistic whose own space is empty is undefined
        return {"dim_U": kU, "dim_V": kV, "Q_U_V": (0.0 if kU > 0 else None), "Q_V_U": (0.0 if kV > 0 else None),
                "projector_distance": (float(np.sqrt((kU + kV) / 2)) if (kU + kV) else 0.0), "angles_deg": None}
    M = VU.T @ VV
    fro2 = (M * M).sum().item()
    if not angles:
        return {"dim_U": kU, "dim_V": kV, "Q_U_V": fro2 / kU, "Q_V_U": fro2 / kV,
                "projector_distance": float(np.sqrt(max(kU + kV - 2 * fro2, 0.0) / 2)), "angles_deg": None}
    cos = torch.linalg.svdvals(M).clamp(0, 1)
    ang = torch.rad2deg(torch.arccos(cos)).cpu().numpy()
    return {"dim_U": kU, "dim_V": kV, "Q_U_V": fro2 / kU, "Q_V_U": fro2 / kV,
            "projector_distance": float(np.sqrt(max(kU + kV - 2 * fro2, 0.0) / 2)),
            "angles_deg": {"n": int(ang.size), "min": float(ang.min()), "median": float(np.median(ang)), "max": float(ang.max()),
                           "count_below_1deg": int((ang < 1).sum()), "count_below_10deg": int((ang < 10).sum()),
                           "count_above_80deg": int((ang > 80).sum())}}


def coordinate_overlap(VK, coords):
    """tr(P_K P_C)/dim K and /dim C for a coordinate subspace C (index list) and basis VK (p x k)."""
    k = VK.shape[1]
    if k == 0 or len(coords) == 0:
        return {"dim_K": k, "dim_C": len(coords), "Q_K_C": (0.0 if k > 0 else None), "Q_C_K": (0.0 if len(coords) > 0 else None)}
    tr = (VK[coords] ** 2).sum().item()
    return {"dim_K": k, "dim_C": len(coords), "Q_K_C": tr / k, "Q_C_K": tr / len(coords)}


def _tests():
    torch.manual_seed(0); p = 50
    Qm, _ = torch.linalg.qr(torch.randn(p, p, dtype=torch.float64))
    A = Qm[:, :10]; B = Qm[:, :10]; C = Qm[:, 10:20]; D = Qm[:, :5]; E = Qm[:, :20]
    r = overlap_stats(A, B); assert abs(r["Q_U_V"] - 1) < 1e-12 and r["projector_distance"] < 1e-6 and r["angles_deg"]["max"] < 1e-5, ("identical", r)
    r = overlap_stats(D, E); assert abs(r["Q_U_V"] - 1) < 1e-12 and abs(r["Q_V_U"] - 0.25) < 1e-12 and abs(r["projector_distance"] - np.sqrt(15 / 2)) < 1e-9, ("nested", r)
    r = overlap_stats(A, C); assert abs(r["Q_U_V"]) < 1e-12 and abs(r["projector_distance"] - np.sqrt(10)) < 1e-9 and r["angles_deg"]["min"] > 89.99, ("orthogonal", r)
    r = overlap_stats(A[:, :0], C); assert r["Q_U_V"] is None and r["Q_V_U"] == 0.0 and r["dim_U"] == 0 and abs(r["projector_distance"] - np.sqrt(5)) < 1e-12, ("empty source", r)
    r = overlap_stats(C, A[:, :0]); assert r["Q_U_V"] == 0.0 and r["Q_V_U"] is None and r["angles_deg"] is None, ("empty target", r)
    r = overlap_stats(A[:, :0], C[:, :0]); assert r["Q_U_V"] is None and r["Q_V_U"] is None and r["projector_distance"] == 0.0, ("both empty", r)
    r = coordinate_overlap(A, []); assert r["Q_K_C"] == 0.0 and r["Q_C_K"] is None, ("empty coordinate set", r)
    r = coordinate_overlap(A[:, :0], [0, 1]); assert r["Q_K_C"] is None and r["Q_C_K"] == 0.0, ("empty kernel", r)
    r = overlap_stats(D, C); assert abs(r["Q_U_V"]) < 1e-12 and abs(r["Q_V_U"]) < 1e-12, ("unequal orthogonal", r)
    # coordinate overlap: K spanned by e_0..e_9, C = {0..4} -> Q_K_C = 0.5, Q_C_K = 1
    VK = torch.eye(p, dtype=torch.float64)[:, :10]
    r = coordinate_overlap(VK, list(range(5))); assert abs(r["Q_K_C"] - 0.5) < 1e-12 and abs(r["Q_C_K"] - 1) < 1e-12, r
    P._tests()
    print("exp_kernel_persistence: projector tests passed")


# ----------------------------------------------------------------------------- Jacobian action (for exposures)
@torch.no_grad()
def apply_J(st, block, v, s):
    """J_l v / sqrt(s) for a flat direction v in the block's column-major coordinates; returns the (s, d) output change."""
    block = block if block == "readout" else int(block)
    if block == "readout":
        H = st["H"][st["L"]][:s]; d = st["d"]; m1 = H.shape[1]
        A = v.reshape(m1, d).T                                # index k*d + c  ->  A[c, k]
        return (H @ A.T) / np.sqrt(s)
    l = block - 1
    H = st["H"][l][:s]; D = st["D"][l][:s]; n = D.shape[2]; m1 = H.shape[1]
    A = v.reshape(m1, n).T                                    # index k*n + j  ->  A[j, k]
    return torch.einsum("iaj,ij->ia", D, H @ A.T) / np.sqrt(s)


def dead_units(S, tau=0.0):
    """Indices of the units whose derivative values S (s x n) satisfy |S_ij| <= tau on every sample: the dead units of a
    ReLU layer for tau = 0 (the gate is never open) and the tau-dead units of a smooth activation (GELU line item (b),
    2026-09-26; the derivative of GELU is never exactly zero on a finite sample, so a threshold is part of the definition)."""
    return ((S.abs() > tau).sum(0) == 0).nonzero().flatten().tolist()


def dead_coordinates(st, block, s, tau=0.0):
    """Coordinate index sets of the block: columns of dead input units and rows of dead output units (tau-dead for tau > 0)."""
    block = block if block == "readout" else int(block)
    if block == "readout":
        H = st["H"][st["L"]][:s]; d = st["d"]; m1 = H.shape[1]
        dead_in = dead_units(st["S"][st["L"] - 1][:s], tau)
        cols = [k * d + c for k in dead_in for c in range(d)]
        return {"dead_input_columns": cols, "dead_output_rows": []}
    l = block - 1
    n = st["D"][l].shape[2]; m1 = st["H"][l].shape[1]
    dead_out = dead_units(st["S"][l][:s], tau)
    dead_in = dead_units(st["S"][l - 1][:s], tau) if l >= 1 else []
    cols = [k * n + j for k in dead_in for j in range(n)]
    rows = [k * n + j for k in range(m1) for j in dead_out]
    return {"dead_input_columns": cols, "dead_output_rows": rows}


# ----------------------------------------------------------------------------- block contexts (MLP and CNN)
class BlockContext:
    """Row generator, size, exposure and dead-unit coordinates of one parameter block at one checkpoint.
    MLP blocks: "2", "3", ..., "readout" (ocean_lib coordinates).  CNN blocks: "conv1", ..., "readout"
    (cnn_jacobian coordinates).  Coordinates are the same at every checkpoint of a trajectory."""

    def __init__(self, run, blk, device):
        self.run = run; self.net = run["net"]; self.X = run["data"]["analysis"][0]; self.s = self.X.shape[0]; self.device = device
        self.blk = str(blk); self.is_cnn = not isinstance(self.net, T.MLP)
        if self.is_cnn:
            self.masks = FL.base_masks(self.net, self.X)
            self.p = CJ.block_size(self.net, self.blk)
            self.row_fn = lambda idx: CJ.cnn_rows(self.net, self.masks, self.X, idx, self.blk)
        else:
            self.st = O.mlp_structure(self.net, self.X)
            if self.blk == "readout":
                self.p = self.st["d"] * self.st["H"][self.st["L"]].shape[1]
                self.row_fn = lambda idx: O.readout_rows(self.st, idx)
            else:
                l = int(self.blk) - 1; self.p = self.st["D"][l].shape[2] * self.st["H"][l].shape[1]
                self.row_fn = lambda idx: O.mlp_rows(self.st, l, idx)

    @torch.no_grad()
    def exposure(self, V, chunk=250):
        """||J v||_2 / sqrt(s) for every column v of V (p x m), by streamed row products."""
        V = V.to(self.device); acc = torch.zeros(V.shape[1], dtype=torch.float64, device=self.device)
        for i in range(0, self.s, chunk):
            idx = torch.arange(i, min(i + chunk, self.s), device=self.device)
            acc += (self.row_fn(idx) @ V).pow(2).sum(0)
        return (acc / self.s).sqrt().cpu().numpy()

    def dead_coordinates(self, tau=0.0):
        """Coordinate index sets: columns of always-inactive input units (exact kernel directions) and
        rows of always-inactive output units; tau-dead units for tau > 0 (MLP only; the CNN uses exact gates)."""
        if not self.is_cnn:
            return dead_coordinates(self.st, self.blk, self.s, tau)
        if tau:
            raise ValueError("tau-dead coordinates are defined for the MLP's derivative values only")
        d = self.net.readout.out_features
        if self.blk == "readout":
            m = self.net.readout.in_features
            # input features are pooled channel-positions; a feature is dead if the last conv channel-position
            # feeding it is always inactive (average pooling of always-inactive positions)
            S = self.masks[-1]; pooled = torch.nn.functional.avg_pool2d(S.double(), 2).flatten(1)      # s x m
            dead_in = (pooled.sum(0) == 0).nonzero().flatten().tolist()
            return {"dead_input_columns": [k * d + c for k in dead_in for c in range(d)], "dead_output_rows": []}
        l = int(self.blk[4:]) - 1; conv = self.net.convs[l]; C, cin = conv.out_channels, conv.in_channels; K = 9 * cin
        dead_out = (self.masks[l].reshape(self.s, C, -1).any(dim=(0, 2)) == 0).nonzero().flatten().tolist()
        if l == 0:
            dead_in = []
        else:
            # input channels come from the previous conv (possibly pooled): always-inactive channel
            dead_in = (self.masks[l - 1].reshape(self.s, cin, -1).any(dim=(0, 2)) == 0).nonzero().flatten().tolist()
        cols = [k * C + c for ci in dead_in for k in range(ci * 9, ci * 9 + 9) for c in range(C)]
        rows = [k * C + j for k in range(K + 1) for j in dead_out]
        return {"dead_input_columns": cols, "dead_output_rows": rows}

    @torch.no_grad()
    def cnn_component_ranks(self):
        """CNN blocks: rank of the patch matrix (feature space, all samples and positions) and of the
        stacked position-wise downstream covectors (downstream space), at the declared tolerance."""
        if not self.is_cnn or self.blk == "readout":
            return None
        l = int(self.blk[4:]) - 1; net = self.net; masks = self.masks
        def patch_rows(idx):
            _, Z, H, _ = CJ._forward_states(net, self.X[idx])
            P = torch.nn.functional.unfold(H[l], 3, padding=1)
            P = torch.cat([P, torch.ones(P.shape[0], 1, P.shape[2], dtype=P.dtype, device=P.device)], 1)
            return P.permute(0, 2, 1).reshape(-1, P.shape[1])
        def down_rows(idx):
            _, Z, H, _ = CJ._forward_states(net, self.X[idx])
            D = CJ.downstream(net, masks, idx, l, Z[l].shape, self.device)                    # b x d x C x H x W
            return D.reshape(D.shape[0], D.shape[1], D.shape[2], -1).permute(0, 1, 3, 2).reshape(-1, D.shape[2])
        Rh = O.blocked_R(patch_rows, self.s, 250, patch_rows(torch.arange(0, 1, device=self.device)).shape[1], self.device)
        Rd = O.blocked_R(down_rows, self.s, 100, net.convs[l].out_channels, self.device)
        svh = O.svdvals(Rh); svd_ = O.svdvals(Rd)
        return {"r_h": int((svh > RANK_TOL * svh[0]).sum()), "r_d": int((svd_ > RANK_TOL * svd_[0]).sum()),
                "m_plus_1": int(Rh.shape[1]), "n": int(Rd.shape[1]), "sv_h": svh.tolist(), "sv_d": svd_.tolist()}


# ----------------------------------------------------------------------------- phases
def factor(a):
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"; os.makedirs(os.path.join(OUT, name), exist_ok=True)
        epochs = sorted({v["checkpoint_epoch"] for v in stages.values() if v["checkpoint_epoch"] is not None})
        if a.epochs:
            epochs = [e for e in epochs if e in a.epochs]
        for ep in epochs:
            run = None
            for blk in a.blocks:
                if run is None:
                    run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device)
                factor_block(run, rd, name, ep, blk, a.device, chunk=a.chunk, no_vectors=a.no_vectors, allow_legacy=a.allow_legacy)
            del run


def factor_block(run, rd, name, ep, blk, device, chunk=500, no_vectors=False, allow_legacy=False):
    """One block's stored factor at one checkpoint: the canonical builder body of `factor`, extracted verbatim
    (2026-11-22) so that the MLP parent fork-epoch builds (scripts/analysis/parent_factor_build.py) reuse the
    identical protocol identifier, identity manifest and rank rule for saved checkpoints that are not declared
    stage epochs.  An existing factor is authenticated (factor_status; raises on mismatch) and never rebuilt.
    Returns the factor file path."""
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    path = os.path.join(OUT, name, f"epoch{ep:04d}_{blk}.npz")
    if os.path.exists(path) and (no_vectors or "V" in np.load(path).files):
        st, _ = factor_status(path, run, rd, blk, allow_legacy)              # raises on mismatch or unaudited legacy
        print(f"== {name} epoch {ep} {blk}: exists ({st})", flush=True); return path
    ctx = BlockContext(run, blk, device); s, p = ctx.s, ctx.p; t0 = time.time()
    comp = ctx.cnn_component_ranks()
    R = O.full_R_cpu(ctx.row_fn, s, p, device, chunk=chunk, verbose=True)
    extra = {"s": s, "p": p, "block": str(blk), "epoch": ep, "protocol": PROTOCOL, "code_revision": git_rev(),
             "identity": json.dumps(factor_identity(run, rd, ep, blk), default=P._jsonable)}
    if comp is not None:
        extra.update(r_h=comp["r_h"], r_d=comp["r_d"], sv_h=np.array(comp["sv_h"]), sv_d=np.array(comp["sv_d"]))
    if no_vectors:
        sv = torch.linalg.svdvals(R).numpy(); del R
        np.savez(path, singular_values=sv, **extra)
    elif p > LP64_LIMIT:
        # MKL's divide-and-conquer SVD needs a workspace above 2^31 entries beyond this size (32-bit LAPACK
        # integers in torch's MKL); numpy's OpenBLAS build uses 64-bit integers, same gesdd algorithm, ~25% slower
        U, sv, Vh = np.linalg.svd(R.numpy(), full_matrices=False); del U, R
        r9 = int((sv > RANK_TOL * sv[0]).sum()); V64 = np.ascontiguousarray(Vh[r9:].T)
        V = np.ascontiguousarray(Vh.T, dtype=np.float32); del Vh
        np.savez(path, singular_values=sv, V=V, V_kernel64=V64, kernel_start=r9, svd_driver="gesdd (numpy/openblas64)", **extra); del V, V64
    else:
        U, sv, Vh = torch.linalg.svd(R, full_matrices=True); del U, R
        sv = sv.numpy(); V = Vh.T.contiguous(); del Vh
        r9 = int((sv > RANK_TOL * sv[0]).sum())
        np.savez(path, singular_values=sv, V=V.to(torch.float32).numpy(), V_kernel64=V[:, r9:].numpy(), kernel_start=r9,
                 svd_driver="gesdd", **extra); del V
    r9 = int((sv > RANK_TOL * sv[0]).sum())
    msg = f"== {name} epoch {ep} {blk}: p={p} rank@1e-9={r9} dimK={p - r9}"
    if comp is not None:
        fn = comp["n"] * (comp["m_plus_1"] - comp["r_h"]); dn = comp["r_h"] * (comp["n"] - comp["r_d"])
        msg += f" = fn {fn} + dn {dn} + def {p - r9 - fn - dn} (r_h={comp['r_h']}/{comp['m_plus_1']}, r_d={comp['r_d']}/{comp['n']})"
    print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
    del sv, ctx
    return path


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return None


def subspaces(sv, V, s):
    """Column index sets: K (declared rule), E(eps) on J/sqrt(s), E+(eps) = E(eps) \\ K."""
    p = sv.size; r = int((sv > RANK_TOL * sv[0]).sum())
    K = list(range(r, p))
    svh = sv / np.sqrt(s)
    E = {str(e): list(range(int((svh > e).sum()), p)) for e in EPS_GRID}
    Eplus = {e: [j for j in cols if j < r] for e, cols in E.items()}
    return r, K, E, Eplus


def overlap(a):
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        # distinct checkpoint epochs in chronological order (stage names are labels only; the descriptive
        # epochs 100 and 250 can fall between or before the declared stages)
        seen = sorted({v["checkpoint_epoch"] for v in stages.values() if v["checkpoint_epoch"] is not None})
        if a.epochs:
            seen = [e for e in seen if e in a.epochs]
        pairs = [(seen[i], seen[i + 1]) for i in range(len(seen) - 1)] + [(e, seen[-1]) for e in seen[:-2]]
        out_path = os.path.join(OUT, f"{name}_overlaps.json")
        res = json.load(open(out_path)) if (os.path.exists(out_path) and a.resume) else {
            "protocol": PROTOCOL, "code_revision": git_rev(), "run": rd, "stages": stages, "rank_tolerance": RANK_TOL,
            "eps_grid": EPS_GRID, "direction_seeds": DIRECTION_SEEDS, "pairs": [], "per_stage": {}}
        ctxs = {}
        def ctx(e, blk):
            if (e, blk) not in ctxs:
                run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=e, device=a.device)
                ctxs[(e, blk)] = BlockContext(run, blk, a.device)
            return ctxs[(e, blk)]
        for blk in a.blocks:
            for (e1, e2) in pairs:
                done_pr = next((pr for pr in res["pairs"] if pr["block"] == str(blk) and pr["epochs"] == [e1, e2]), None)
                if done_pr is not None:
                    # completed pair: its own identity digests and source file hashes must equal the current ones;
                    # legacy pairs (no digests recorded) are reused only with --allow-legacy and carry that status
                    cur = [factor_status(os.path.join(FACTOR_DIR, name, f"epoch{e:04d}_{blk}.npz"), ctx(e, blk).run, rd, blk, a.allow_legacy) for e in (e1, e2)]
                    cur_dig = [c[1]["digest"] for c in cur]; cur_sha = [P.sha256_file(os.path.join(FACTOR_DIR, name, f"epoch{e:04d}_{blk}.npz")) for e in (e1, e2)]
                    stale = ("identity_digests" not in done_pr) or done_pr["identity_digests"] != cur_dig or done_pr["source_factor_sha256"] != cur_sha
                    if not stale:
                        done_pr["provenance_status"] = "identity verified"; continue
                    if a.recompute_stale:
                        print(f"== {name} {blk} {e1}->{e2}: stored pair does not match the current factors/identity; recomputing", flush=True)
                        res["pairs"].remove(done_pr)
                    elif "identity_digests" in done_pr:
                        raise P.ProvenanceError(f"{name} {blk} {e1}->{e2}: completed pair belongs to different factors or a different analysis identity (pass --recompute-stale)")
                    elif a.allow_legacy:
                        done_pr["provenance_status"] = "legacy pair, UNVERIFIED: sources validated now (" + "; ".join(c[0] for c in cur) + "), pair identity not recorded at compute time"; continue
                    else:
                        raise P.ProvenanceError(f"{name} {blk} {e1}->{e2}: completed pair without a recorded identity; pass --allow-legacy or --recompute-stale")
                sv1, V1, s, p, st1 = load_factor(name, e1, blk, ctx(e1, blk).run, rd, a.allow_legacy, True)
                sv2, V2, _, _, st2 = load_factor(name, e2, blk, ctx(e2, blk).run, rd, a.allow_legacy, True)
                r1, K1, E1, E1p = subspaces(sv1, V1, s); r2, K2, E2, E2p = subspaces(sv2, V2, s)
                rec = {"block": str(blk), "epochs": [e1, e2], "p": p, "rank": [r1, r2], "dim_K": [len(K1), len(K2)],
                       "source_status": [st1, st2], "provenance_status": "identity verified", "code_revision": git_rev(),
                       "identity_digests": [factor_identity(ctx(e, blk).run, rd, e, blk)["digest"] for e in (e1, e2)],
                       "source_factor_sha256": [P.sha256_file(os.path.join(FACTOR_DIR, name, f"epoch{e:04d}_{blk}.npz")) for e in (e1, e2)],
                       "K_K": overlap_stats(V1[:, K1], V2[:, K2]),
                       "E_K": {e: {"dim_E": len(E1[e]), **overlap_stats(V1[:, E1[e]], V2[:, K2], angles=False)} for e in E1},
                       "Eplus_K": {e: {"dim_Eplus": len(E1p[e]), **overlap_stats(V1[:, E1p[e]], V2[:, K2], angles=False)} for e in E1p}}
                for key, cols in [("K", K1)] + [(f"Eplus_{e}", E1p[str(e)]) for e in EPS_SAMPLED]:
                    if len(cols) == 0:
                        rec[f"sampled_{key}"] = {"dim": 0, "note": "empty space, unavailable"}; continue
                    B = V1[:, cols]; dirs, resid, rejected = [], [], 0
                    for ds in DIRECTION_SEEDS:
                        g = torch.Generator().manual_seed(ds)
                        u = torch.randn(p, generator=g, dtype=torch.float64)
                        v = B @ (B.T @ u); nv = v.norm().item()
                        if nv < 1e-8 * u.norm().item():
                            rejected += 1; continue
                        dirs.append(v / nv); resid.append(((u - v).norm() / u.norm()).item())
                    Vd = torch.stack(dirs, 1)
                    exp1 = ctx(e1, blk).exposure(Vd); exp2 = ctx(e2, blk).exposure(Vd)
                    rec[f"sampled_{key}"] = {"dim": len(cols), "n_directions": len(dirs), "rejected": rejected,
                                             "degenerate": len(cols) < len(DIRECTION_SEEDS),
                                             "exposure_earlier": exp1.tolist(), "exposure_later": exp2.tolist(),
                                             "projection_residual_median": float(np.median(resid)) if resid else None}
                res["pairs"].append(rec)
                print(f"== {name} {blk} {e1}->{e2}: dimK {len(K1)}->{len(K2)} Q(K1,K2)={rec['K_K']['Q_U_V']} Q(K2,K1)={rec['K_K']['Q_V_U']} "
                      f"d={rec['K_K']['projector_distance']:.1f}; Eplus(1e-4)->K2 Q={rec['Eplus_K']['0.0001']['Q_U_V']}", flush=True)
                del V1, V2
                records.dump_json(out_path, res)
                for k in [k for k in ctxs if k[1] == blk and k[0] not in (e1, e2)]:    # keep GPU memory bounded on shared cards
                    del ctxs[k]
                torch.cuda.empty_cache()
            for e in seen:
                ps = res["per_stage"].get(str(blk), {}).get(str(e))
                if ps is not None:
                    fp = os.path.join(FACTOR_DIR, name, f"epoch{e:04d}_{blk}.npz")
                    if ps.get("source_factor_sha256") == P.sha256_file(fp):
                        continue
                    if not a.recompute_stale:
                        if a.allow_legacy:
                            ps["provenance_status"] = "legacy per-stage record, UNVERIFIED (factor hash not recorded)"; continue
                        raise P.ProvenanceError(f"{name} {blk} epoch {e}: per-stage record without a matching factor hash (pass --allow-legacy or --recompute-stale)")
                sv, V, s, p, st_ = load_factor(name, e, blk, ctx(e, blk).run, rd, a.allow_legacy, True); r, K, E, Ep = subspaces(sv, V, s)
                dc = ctx(e, blk).dead_coordinates()
                allc = sorted(set(dc["dead_input_columns"]) | set(dc["dead_output_rows"]))
                res["per_stage"].setdefault(str(blk), {})[str(e)] = {
                    "source_status": st_, "provenance_status": "identity verified", "source_factor_sha256": P.sha256_file(os.path.join(FACTOR_DIR, name, f"epoch{e:04d}_{blk}.npz")),
                    "identity_digest": factor_identity(ctx(e, blk).run, rd, e, blk)["digest"], "code_revision": git_rev(),
                    "rank": r, "dim_K": len(K), "dim_E": {k: len(v) for k, v in E.items()},
                    "dead_input_columns": len(dc["dead_input_columns"]), "dead_output_rows": len(dc["dead_output_rows"]), "dead_union": len(allc),
                    "overlap_K_dead_union": coordinate_overlap(V[:, K], allc),
                    "overlap_K_dead_input": coordinate_overlap(V[:, K], dc["dead_input_columns"]),
                    "overlap_K_dead_output": coordinate_overlap(V[:, K], dc["dead_output_rows"])}
                del V
                records.dump_json(out_path, res)
            # release the contexts of this block
            for k in [k for k in ctxs if k[1] == blk]:
                del ctxs[k]
            torch.cuda.empty_cache()
        records.dump_json(out_path, res)


def audit(a):
    """Stored-basis audit: after float32 storage and float64 reload, the orthonormality error of the kernel basis,
    its idempotence error, the exposure ||J v||/sqrt(s) of every kernel column against the rank scale
    eta = 1e-9 sigma_1 / sqrt(s), the same for the 1e-6 positive tail, and the sensitivity of the kernel-to-kernel
    overlap of successive stages to the 1e-9 / 1e-12 rank rule."""
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        seen = sorted({v["checkpoint_epoch"] for v in stages.values() if v["checkpoint_epoch"] is not None})
        if a.epochs:
            seen = [e for e in seen if e in a.epochs]
        ap = os.path.join(OUT, f"{name}_audit.json")
        out = json.load(open(ap)) if os.path.exists(ap) else {"protocol": PROTOCOL, "run": rd, "per_stage": {}, "ladder_pairs": {}}
        out["code_revision"] = git_rev(); out.setdefault("audit_runs", []).append({"blocks": list(a.blocks), "epochs": seen, "code_revision": git_rev()})
        prev = {}
        for e in seen:
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=e, device=a.device)
            for blk in a.blocks:
                path = os.path.join(FACTOR_DIR, name, f"epoch{e:04d}_{blk}.npz"); z = np.load(path)
                prov, cur_ident = factor_status(path, run, rd, blk, allow_legacy=True, z=z)   # the audit is what qualifies legacy files
                sv, V, s, p = z["singular_values"], torch.from_numpy(z["V"]).double(), int(z["s"]), int(z["p"])
                if "V_kernel64" in z.files:
                    V[:, int(z["kernel_start"]):] = torch.from_numpy(z["V_kernel64"])
                r9, K9, E9, Ep9 = subspaces(sv, V, s); r12 = int((sv > 1e-12 * sv[0]).sum()); K12 = list(range(r12, p))
                r6 = int((sv > 1e-6 * sv[0]).sum()); K6 = list(range(r6, p))
                eta = 1e-9 * sv[0] / np.sqrt(s); ctx = BlockContext(run, blk, a.device)
                rec = {"provenance_before_audit": prov, "audit_version": P.AUDIT_VERSION, "audit_pass_criteria": P.AUDIT_PASS,
                       "factor_sha256": P.sha256_file(path), "identity_digest": cur_ident["digest"], "identity": cur_ident,
                       "storage": ("float64 kernel columns" if "V_kernel64" in z.files else "float32 basis"),
                       "rank_1e-6": r6, "rank_1e-9": r9, "rank_1e-12": r12, "dim_K_1e-6": len(K6), "dim_K_1e-9": len(K9), "dim_K_1e-12": len(K12),
                       "eta_rank_scale_on_Jhat": float(eta), "sigma1": float(sv[0])}
                if K9:
                    B = V[:, K9]; G = B.T @ B
                    rec["kernel_basis_orthonormality_max_abs"] = float((G - torch.eye(len(K9), dtype=G.dtype)).abs().max())
                    g = torch.Generator().manual_seed(7); Xr = torch.randn(p, 64, generator=g, dtype=torch.float64)
                    PX = B @ (B.T @ Xr); PPX = B @ (B.T @ PX)
                    rec["projector_idempotence_rel"] = float((PPX - PX).norm() / PX.norm())        # ||P^2 x - P x|| / ||P x|| on random x
                    ex = ctx.exposure(B, chunk=min(a.chunk, 250))
                    rec["kernel_columns_exposure"] = {"max": float(ex.max()), "median": float(np.median(ex)), "max_over_eta": float(ex.max() / eta),
                                                      "fraction_above_eta": float((ex > eta).mean())}
                # spectral consistency against the CURRENT Jacobian: stored (sigma_j, v_j) with sigma_j >= 1e-3 sigma_1 must
                # reproduce ||J v_j|| / sqrt(s) (float32 storage limits this to ~1e-4); this binds the stored spectrum to
                # the current checkpoint, preprocessing and samples independently of the kernel (an empty kernel proves nothing)
                big = [j for j in range(p) if sv[j] >= 1e-3 * sv[0]]
                pick = sorted(set(np.linspace(0, len(big) - 1, 24, dtype=int).tolist() + list(range(max(0, len(big) - 8), len(big)))))
                cols_sc = [big[j] for j in pick]
                ex_sc = ctx.exposure(V[:, cols_sc], chunk=min(a.chunk, 250)); pred = sv[cols_sc] / np.sqrt(s)
                rec["spectral_consistency"] = {"n_columns": len(cols_sc), "max_rel_deviation": float(np.max(np.abs(ex_sc - pred) / pred)),
                                               "smallest_checked_sigma_over_sigma1": float(sv[cols_sc[-1]] / sv[0])}
                cols = Ep9["1e-06"]
                if cols:
                    ex = ctx.exposure(V[:, cols], chunk=min(a.chunk, 250)); rec["tail_1e-6_columns_exposure"] = {"dim": len(cols), "max": float(ex.max()), "min": float(ex.min()), "min_over_eta": float(ex.min() / eta)}
                rec["pass"], rec["pass_reason"] = P.audit_passes(rec)
                out["per_stage"].setdefault(str(blk), {})[str(e)] = rec
                # ladder sensitivity of the successive kernel overlap
                if (blk in prev) and len(K9) and len(prev[blk]["K9"]):
                    Vp = prev[blk]["V"]; pv = prev[blk]
                    q6 = overlap_stats(Vp[:, pv["K6"]], V[:, K6], angles=False); q9 = overlap_stats(Vp[:, pv["K9"]], V[:, K9], angles=False)
                    q12 = overlap_stats(Vp[:, pv["K12"]], V[:, K12], angles=False)
                    # positive-tail overlap sensitivity: E+_t(1e-4) (defined against K_t at each rule) into K_t' at each rule
                    def tail_cols(svp, rp):
                        return [j for j in range(int((svp / np.sqrt(s) > 1e-4).sum()), rp)]
                    t9 = overlap_stats(Vp[:, tail_cols(pv["sv"], pv["r9"])], V[:, K9], angles=False)
                    t12 = overlap_stats(Vp[:, tail_cols(pv["sv"], pv["r12"])], V[:, K12], angles=False)
                    out["ladder_pairs"].setdefault(str(blk), {})[f"{pv['epoch']}->{e}"] = {
                        "dims_1e-6": [len(pv["K6"]), len(K6)], "dims_1e-9": [len(pv["K9"]), len(K9)], "dims_1e-12": [len(pv["K12"]), len(K12)],
                        "Q_fwd_1e-6": q6["Q_U_V"], "Q_fwd_1e-9": q9["Q_U_V"], "Q_fwd_1e-12": q12["Q_U_V"],
                        "Q_bwd_1e-6": q6["Q_V_U"], "Q_bwd_1e-9": q9["Q_V_U"], "Q_bwd_1e-12": q12["Q_V_U"],
                        "Q_tail_1e-4_fwd_1e-9": t9["Q_U_V"], "Q_tail_1e-4_fwd_1e-12": t12["Q_U_V"], "tail_dims": [t9["dim_U"], t12["dim_U"]]}
                prev[blk] = {"V": V, "K6": K6, "K9": K9, "K12": K12, "epoch": e, "sv": sv, "r9": r9, "r12": r12}
                print(f"== audit {name} {blk} epoch {e}: {json.dumps({k: v for k, v in rec.items() if k != 'provenance'})}", flush=True)
                del ctx; torch.cuda.empty_cache()
            del run; torch.cuda.empty_cache()
            records.dump_json(ap, out)                       # after every stage: a crash keeps completed entries
        records.dump_json(ap, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["factor", "overlap", "audit", "test"])
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--blocks", nargs="+", default=["2", "3", "readout"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--tag", default="", help="run tag (e.g. gelu, w64)")
    ap.add_argument("--epochs", type=int, nargs="*", default=None, help="restrict to these checkpoint epochs")
    ap.add_argument("--no-vectors", action="store_true", help="factor: singular values only (no basis stored)")
    ap.add_argument("--resume", action="store_true", help="overlap: keep pairs already in the JSON")
    ap.add_argument("--chunk", type=int, default=500, help="factor: samples per row chunk (GPU memory; CNN blocks need ~25 MB per sample at 500)")
    ap.add_argument("--allow-legacy", action="store_true", help="reuse factors, pairs or records without a bound identity or passing audit (status propagated as UNVERIFIED)")
    ap.add_argument("--recompute-stale", action="store_true", help="overlap: drop and recompute stored pairs and per-stage records whose factors or identity changed")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.phase == "test":
        _tests()
    elif a.phase == "factor":
        factor(a)
    elif a.phase == "audit":
        audit(a)
    else:
        overlap(a)


