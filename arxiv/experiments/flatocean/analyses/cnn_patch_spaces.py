"""
Tied selective patch spaces of the MNIST CNN, the direct sums (selective spaces and full atomic kernels) as
candidate multilayer branches, the full-network tangent kernel, and rays (EXPERIMENTS_PLAN.md Section 0.4, item 9).

For conv block l and output channel c let A_c be the set of (sample, position) pairs where the channel is active
and P_{A_c} the matrix of their augmented input patches ([9 c_in patch values; 1]).  The selective patch space of
the channel is the orthogonal complement of the row space of P_{A_c} in R^{9 c_in + 1}: a kernel edit there leaves
every active preactivation of the channel unchanged, so nothing propagates (the tied analogue of the selective-unit
space of the MLP).  The readout has no gate: its space is the feature-null space of the pooled features.  Ranks
are reported on the ladder 1e-6 / 1e-9 / 1e-12 with the gap at the declared rule 1e-9.

Phase `spaces`: per block dim G_l, its projected mass inside the stored atomic kernel K_l and the share of K_l it
explains; the direct sum of the selective spaces AND the direct sum of the full atomic kernels (stored factors),
each validated by the frozen-polynomial residual on 32 random combinations (finite, tolerance 1e-9, pass flag;
parameters restored in `finally`).  Bases are saved with the analysis identity and their file hashes are recorded.
If the atomic sum fails the test, the certified block-coordinate constructions would be needed; that path is
reported, not run here.

Phase `tangent`: the full-network Jacobian on the analysis set (all blocks, tied coordinates), its rank ladder,
and an orthonormal basis of ker J_X (float64), for the tangent-flat control.

Phase `rays`: matched ambient probes (seeds 100-227) projected into the branch (selective sum), the atomic sum,
ker J_X minus the branch (tangent-flat control) or used directly (generic); scaled block by block for the
block-balanced relative grid (layerwise sets, generic) or as a whole by the weighted relative norm (tangent,
labelled), and by the Euclidean norm for the absolute grid; complete transition decomposition per ray.  Every set
and grid is checked for completeness on resume after the identity check.

    python -m flatocean cnn-patch-spaces spaces  --seeds 0 1 2 --stages high_accuracy mse_interpolation epoch_250 late_endpoint
    python -m flatocean cnn-patch-spaces tangent --seeds 0 --stages late_endpoint
    python -m flatocean cnn-patch-spaces rays    --seeds 0 --stages late_endpoint
"""
# Ported unchanged from experiments/exp_cnn_patch_spaces.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean cnn-patch-spaces).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os
import time

import numpy as np
from flatocean import records, runs
import torch
import torch.nn.functional as F

from flatocean import cnn_jacobian as CJ
from flatocean.analyses import persistence as KP
from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.analyses import transitions as TR
from flatocean.runs import stage_epochs

OUT = runs.out_dir("branches")
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.4 item 9 (2026-09-22); identifier cnn-patch-spaces-v2"
RTOL = 1e-9; LADDER = (1e-6, 1e-9, 1e-12); VALID_TOL = 1e-9
ALPHAS = TR.ALPHAS; RADII = TR.RADII_V3
SETS = ("branch", "constructed", "atomic_sum", "tangent_flat", "generic")


def ladder_counts(sv):
    sv = np.asarray(sv); return {f"{t:g}": int((sv > t * sv[0]).sum()) if sv.size and sv[0] > 0 else 0 for t in LADDER}


def gap_at(sv, r):
    return float(sv[r - 1] / max(float(sv[r]), 1e-300)) if 0 < r < len(sv) else None


@torch.no_grad()
def patch_spaces(net, X, device, chunk=200):
    """Per conv block: per-channel (active count, rank ladder, gap) and the orthonormal basis of the selective patch
    space in the block's column-major coordinates (index k * C + c); readout: feature-null space of the pooled features."""
    spaces = {}; records = {}; L = len(net.convs)
    for l in range(L):
        conv = net.convs[l]; C, cin = conv.out_channels, conv.in_channels; K = 9 * cin + 1
        R = [None] * C; counts = [0] * C
        for i in range(0, X.shape[0], chunk):
            _, Z, H, _ = CJ._forward_states(net, X[i:i + chunk])
            Pm = F.unfold(H[l], 3, padding=1)
            Pm = torch.cat([Pm, torch.ones(Pm.shape[0], 1, Pm.shape[2], dtype=Pm.dtype, device=device)], 1)
            act = (Z[l] > 0).reshape(Z[l].shape[0], C, -1)
            for c in range(C):
                rows = Pm.permute(0, 2, 1)[act[:, c]]; counts[c] += int(rows.shape[0])
                if rows.shape[0] == 0:
                    continue
                M = rows if R[c] is None else torch.cat([R[c], rows], 0); R[c] = torch.linalg.qr(M, mode="r").R[:K]
        bases = []; rec = []
        for c in range(C):
            if R[c] is None:
                Nc = torch.eye(K, dtype=X.dtype, device=device); r = 0; lad = {f"{t:g}": 0 for t in LADDER}; gap = None
            else:
                _, sv, Vh = torch.linalg.svd(R[c], full_matrices=True); svn = sv.cpu().numpy(); lad = ladder_counts(svn); r = lad["1e-09"]; gap = gap_at(svn, r); Nc = Vh[r:].T
            rec.append({"channel": c, "active_positions": counts[c], "rank": r, "rank_ladder": lad, "free": K - r, "gap": gap})
            if Nc.shape[1]:
                Bc = torch.zeros(K * C, Nc.shape[1], dtype=X.dtype, device=device); Bc[torch.arange(K, device=device) * C + c] = Nc; bases.append(Bc)
        spaces[f"conv{l + 1}"] = torch.cat(bases, 1) if bases else torch.zeros(K * C, 0, dtype=X.dtype, device=device)
        records[f"conv{l + 1}"] = {"p": K * C, "dim": int(spaces[f"conv{l + 1}"].shape[1]), "dim_ladder": {t: sum(K - q["rank_ladder"][t] for q in rec) for t in rec[0]["rank_ladder"]},
                                   "channels": rec, "channels_selective": sum(1 for q in rec if q["free"] > 0), "min_gap": min([q["gap"] for q in rec if q["gap"]], default=None)}
    Rr = None
    for i in range(0, X.shape[0], chunk):
        _, _, _, hf = CJ._forward_states(net, X[i:i + chunk]); Hf = torch.cat([hf, torch.ones(hf.shape[0], 1, dtype=hf.dtype, device=device)], 1)
        Rr = torch.linalg.qr(Hf if Rr is None else torch.cat([Rr, Hf], 0), mode="r").R[:Hf.shape[1]]
    _, sv, Vh = torch.linalg.svd(Rr, full_matrices=True); svn = sv.cpu().numpy(); lad = ladder_counts(svn); r = lad["1e-09"]; comp = Vh[r:].T; m1 = Rr.shape[1]; d = net.readout.out_features
    Bv = torch.zeros(m1 * d, comp.shape[1] * d, dtype=X.dtype, device=device)
    for a in range(d):
        Bv[torch.arange(m1, device=device) * d + a, a * comp.shape[1]:(a + 1) * comp.shape[1]] = comp
    spaces["readout"] = Bv
    records["readout"] = {"p": m1 * d, "dim": int(Bv.shape[1]), "dim_ladder": {t: d * (m1 - lad[t]) for t in lad}, "feature_rank": r, "feature_rank_ladder": lad, "gap": gap_at(svn, r)}
    return spaces, records


def embed(net, spaces, coeffs):
    w = torch.zeros(sum(q.numel() for q in net.parameters()), dtype=torch.float64, device=next(net.parameters()).device)
    for blk, B in spaces.items():
        if B.shape[1] == 0:
            continue
        w[torch.as_tensor(CJ.block_index_map(net, blk), device=w.device)] += B @ coeffs[blk]
    return w


def _nanmax(a, b):
    return float("nan") if not (np.isfinite(a) and np.isfinite(b)) else max(a, b)


@torch.no_grad()
def polynomial_validation(net, X, spaces, theta0, masks, device, n_dirs=32):
    """Frozen-polynomial residual of random combinations of the direct sum at weighted relative norms 0.1, 1, 10
    (six points per line); finite residuals at most VALID_TOL are required for `passed`."""
    idx = torch.arange(X.shape[0], device=device); sl = FL.block_slices(net)
    F0 = torch.cat([FL.frozen_forward(net, masks, X, idx[i:i + 500]) for i in range(0, X.shape[0], 500)]); scale = FL.x_norm(F0)
    worst = {str(r_): 0.0 for r_ in (0.1, 1.0, 10.0)}
    if not (np.isfinite(scale) and scale > 0):
        return {"tolerance": VALID_TOL, "residual_rel": {k: float("nan") for k in worst}, "passed": False}
    gen = torch.Generator().manual_seed(1234)
    try:
        for _ in range(n_dirs):
            w = embed(net, spaces, {blk: torch.randn(B.shape[1], generator=gen, dtype=torch.float64).to(device) for blk, B in spaces.items()})
            if not w.any():
                break
            w, _ = FL.scale_direction(w, theta0, sl, "weighted")
            for r_ in (0.1, 1.0, 10.0):
                for t in (0.37, 0.61, 1.0, -0.83, -0.45, 1.7):
                    FL.set_flat(net, theta0 + r_ * t * w)
                    Fw = torch.cat([FL.frozen_forward(net, masks, X, idx[i:i + 500]) for i in range(0, X.shape[0], 500)])
                    worst[str(r_)] = _nanmax(worst[str(r_)], FL.x_norm(Fw - F0) / scale)
    finally:
        FL.set_flat(net, theta0)
    return {"tolerance": VALID_TOL, "residual_rel": worst, "passed": bool(all(np.isfinite(v) and v <= VALID_TOL for v in worst.values())),
            "scope": "32 random combinations, weighted relative norms 0.1, 1, 10, six points per line"}


def stage_identity(run, rd, ep):
    return P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "rtol": RTOL, "ladder": list(LADDER), "valid_tol": VALID_TOL})


def basis_path(name, ep, kind, blk):
    return os.path.join(OUT, f"{name}_epoch{ep:04d}_{kind}_{blk}.pt")


def save_basis(path, B, digest):
    torch.save({"identity_digest": digest, "basis": B.cpu()}, path); return P.sha256_file(path)


def load_bases(name, ep, kind, blocks, srec, device):
    out = {}
    for blk in blocks:
        path = basis_path(name, ep, kind, blk); obj = torch.load(path, weights_only=False)
        if obj["identity_digest"] != srec["identity"]["digest"] or P.sha256_file(path) != srec["basis_sha256"][kind][blk]:
            raise P.ProvenanceError(f"{path}: basis does not match the stage record")
        out[blk] = obj["basis"].to(device)
    return out


def stage_incomplete(old, ident, name, ep, run, rd):
    """None when a stored patch-space stage can be reused: same identity, complete, every saved basis present with its
    recorded hash, and every consumed factor still the same bytes and authenticating against the current run (checked
    before any cache decision, 2026-09-25); otherwise the reason."""
    from flatocean import factors as FA
    if (old.get("identity") or {}).get("digest") != ident["digest"]:
        return "stored record under another identity"
    if not old.get("complete"):
        return "stored record incomplete"
    for kind, blocks in (old.get("basis_sha256") or {}).items():
        for blk, h in blocks.items():
            path = basis_path(name, ep, kind, blk)
            if not os.path.exists(path) or P.sha256_file(path) != h:
                return f"saved {kind} basis of {blk} missing or changed"
    for blk, b in (old.get("blocks") or {}).items():
        ak = b.get("atomic_kernel")
        fpath = os.path.join(KP.FACTOR_DIR, name, f"epoch{ep:04d}_{blk}.npz")
        if ak is None:
            if os.path.exists(fpath):
                return f"a factor for {blk} exists now (the record was made without it)"
            continue
        if not os.path.exists(fpath) or P.sha256_file(fpath) != ak.get("factor_sha256"):
            return f"consumed factor of {blk} missing or changed"
        try:
            FA.factor_status(fpath, run, rd, blk, ep=ep)
        except P.ProvenanceError as e:
            return f"consumed factor of {blk} does not authenticate: {e}"
    return None


def spaces_phase(a):
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, tag=a.tag); stages = stage_epochs(rd); name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        out_path = os.path.join(OUT, f"{name}_patch_spaces.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "stages": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device); net = run["net"]; X = run["data"]["analysis"][0]
            ident = stage_identity(run, rd, ep)
            old = res["stages"].get(stage)
            if old and not a.overwrite:
                why = stage_incomplete(old, ident, name, ep, run, rd)
                if why is None:
                    print(f"== {name} {stage}: complete under the current identity (bases and factors verified)", flush=True); continue
                print(f"== {name} {stage}: {why}; recomputing", flush=True)
            t0 = time.time(); spaces, brec = patch_spaces(net, X, a.device)
            theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X)
            kernels = {}
            for blk in brec:
                fpath = os.path.join(KP.FACTOR_DIR, name, f"epoch{ep:04d}_{blk}.npz")
                if not os.path.exists(fpath):
                    brec[blk]["atomic_kernel"] = None; continue
                sv, V, s_, p_ = KP.load_factor(name, ep, blk, run, rd); lad = ladder_counts(sv); r = lad["1e-09"]; BK = V[:, r:].to(a.device); del V
                kernels[blk] = BK
                G = spaces[blk]; Q = float((BK.T @ G).pow(2).sum() / G.shape[1]) if (BK.shape[1] and G.shape[1]) else None
                brec[blk]["atomic_kernel"] = {"dim_K": int(BK.shape[1]), "dim_K_ladder": {t: p_ - lad[t] for t in lad}, "gap": gap_at(sv, r), "factor_sha256": P.sha256_file(fpath),
                                                 "Q_space_in_K": Q, "share_of_K_explained": (G.shape[1] * Q / BK.shape[1]) if (Q is not None and BK.shape[1]) else None}
            val_sel = polynomial_validation(net, X, spaces, theta0, masks, a.device)
            missing = [blk for blk in brec if blk not in kernels]
            val_atomic = polynomial_validation(net, X, kernels, theta0, masks, a.device) if not missing else {"passed": None, "note": f"factors missing for {missing}"}
            digest = ident["digest"]; sha = {"selective": {}, "atomic": {}}
            for blk, B in spaces.items():
                sha["selective"][blk] = save_basis(basis_path(name, ep, "selective", blk), B, digest)
            for blk, B in kernels.items():
                sha["atomic"][blk] = save_basis(basis_path(name, ep, "atomic", blk), B, digest)
            rec = {"epoch": ep, "identity": ident, "blocks": brec, "selective_sum_dim": int(sum(B.shape[1] for B in spaces.values())), "selective_sum_validation": val_sel,
                   "atomic_sum_dim": int(sum(B.shape[1] for B in kernels.values())), "atomic_sum_validation": val_atomic,
                   "atomic_sum_is_branch": val_atomic.get("passed"), "certified_constructions_needed": (val_atomic.get("passed") is False),
                   "basis_sha256": sha, "seconds": time.time() - t0, "complete": True}
            res["stages"][stage] = rec
            print(f"== {name} {stage} (epoch {ep}): selective dims " + ", ".join(f"{b} {brec[b]['dim']}" + (f" (K {brec[b]['atomic_kernel']['dim_K']}, share {brec[b]['atomic_kernel']['share_of_K_explained']:.3f})" if brec[b].get('atomic_kernel') and brec[b]['atomic_kernel']['share_of_K_explained'] is not None else "") for b in brec)
                  + f"; selective sum {rec['selective_sum_dim']} passed {val_sel['passed']} ({max(val_sel['residual_rel'].values()):.1e}); atomic sum {rec['atomic_sum_dim']} passed {val_atomic.get('passed')}"
                  + (f" ({max(val_atomic['residual_rel'].values()):.1e})" if val_atomic.get('residual_rel') else "") + f" ({time.time() - t0:.0f}s)", flush=True)
            records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def tangent_phase(a):
    """Full-network Jacobian on the analysis set (rows of every block side by side, tied coordinates), rank ladder,
    ker J_X basis.  p = 45,114 columns exceeds the LP64 limit, so the SVD with vectors uses numpy's 64-bit LAPACK."""
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, tag=a.tag); stages = stage_epochs(rd); name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        out_path = os.path.join(OUT, f"{name}_tangent.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device); net = run["net"]; X = run["data"]["analysis"][0]
            ident = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "phase": "tangent", "rtol": RTOL, "ladder": list(LADDER)})
            if stage in res and res[stage].get("identity", {}).get("digest") == ident["digest"] and not a.overwrite:
                continue
            masks = FL.base_masks(net, X); blocks = CJ.conv_blocks(net); t0 = time.time()
            maps = {blk: torch.as_tensor(CJ.block_index_map(net, blk)) for blk in blocks}; p = sum(q.numel() for q in net.parameters())
            def row_fn(idx):
                rows = torch.zeros(len(idx) * net.readout.out_features, p, dtype=torch.float64)
                for blk in blocks:
                    rows[:, maps[blk]] = CJ.cnn_rows(net, masks, X, idx, blk).cpu()
                return rows
            R = O.full_R_cpu(row_fn, X.shape[0], p, a.device, chunk=100, verbose=True)
            U, sv, Vh = np.linalg.svd(R.numpy(), full_matrices=False); del U, R
            lad = ladder_counts(sv); r = lad["1e-09"]; Kx = np.ascontiguousarray(Vh[r:].T); del Vh
            kp = os.path.join(OUT, f"{name}_epoch{ep:04d}_kerJX.pt"); torch.save({"identity_digest": ident["digest"], "basis": torch.from_numpy(Kx)}, kp)
            np.save(os.path.join(OUT, f"{name}_epoch{ep:04d}_JX_singular_values.npy"), sv)
            res[stage] = {"epoch": ep, "identity": ident, "p": p, "rows": int(X.shape[0] * net.readout.out_features), "counting_nullity": max(0, p - X.shape[0] * net.readout.out_features),
                          "rank_ladder": lad, "dim_ker_JX": p - r, "gap": gap_at(sv, r), "kernel_basis_sha256": P.sha256_file(kp), "seconds": time.time() - t0}
            print(f"== {name} {stage}: p {p} rank ladder {lad} dim ker J_X {p - r} gap {res[stage]['gap']} ({time.time() - t0:.0f}s)", flush=True)
            records.dump_json(out_path, res)


def rays_phase(a):
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, tag=a.tag); stages = stage_epochs(rd); name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        srec_all = json.load(open(os.path.join(OUT, f"{name}_patch_spaces.json")))["stages"]
        out_path = os.path.join(OUT, f"{name}_patch_rays.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "results": {}}
        seeds_used = list(TR.DIRECTION_SEEDS[: a.ndirs])
        if a.ndirs < 128 and not a.exploratory:
            raise SystemExit("structured rays use 128 matched directions; pass --exploratory for smaller counts")
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device); net = run["net"]; X = run["data"]["analysis"][0]
            sident = stage_identity(run, rd, ep); srec = srec_all.get(stage)
            if not srec or srec["identity"]["digest"] != sident["digest"] or not srec.get("complete"):
                raise P.ProvenanceError(f"{name} {stage}: patch-space record missing, incomplete or under another identity")
            if not srec["selective_sum_validation"]["passed"]:
                raise RuntimeError(f"{name} {stage}: the selective sum did not pass validation; not a branch")
            blocks = CJ.conv_blocks(net); sel = load_bases(name, ep, "selective", blocks, srec, a.device)
            atomic = load_bases(name, ep, "atomic", [b for b in blocks if b in srec["basis_sha256"]["atomic"]], srec, a.device) if srec.get("atomic_sum_is_branch") else None
            # the certified construction of exp_cnn_branch (bottom-up order), authenticated by its record: identity digest, basis hash, certificate
            constructed = None; constructed_sha = None; bp = os.path.join(OUT, f"{name}_branch.json")
            if os.path.exists(bp):
                brec = json.load(open(bp))["results"].get(stage)
                if brec and brec.get("epoch") == ep and "bottom_up" in brec.get("constructions", {}) and brec["constructions"]["bottom_up"].get("certified_branch"):
                    bpath = os.path.join(OUT, f"{name}_epoch{ep:04d}_branch_bottom_up.pt"); bobj = torch.load(bpath, weights_only=False)
                    bident = P.analysis_identity(run, rd, ep, brec["identity"]["config"])
                    if bident["digest"] != brec["identity"]["digest"] or bobj["identity_digest"] != brec["identity"]["digest"] or P.sha256_file(bpath) != brec["constructions"]["bottom_up"]["basis_sha256"]:
                        raise P.ProvenanceError(f"{bpath}: constructed branch does not match its record or the loaded inputs")
                    constructed = {blk: bobj["bases"][blk].to(a.device) for blk in blocks if bobj["bases"][blk].shape[1] > 0}; constructed_sha = brec["constructions"]["bottom_up"]["basis_sha256"]
            tp = os.path.join(OUT, f"{name}_tangent.json"); Kx = None
            if os.path.exists(tp) and stage in json.load(open(tp)):
                trec = json.load(open(tp))[stage]; tident = P.analysis_identity(run, rd, ep, trec["identity"]["config"])
                kp = os.path.join(OUT, f"{name}_epoch{ep:04d}_kerJX.pt"); kobj = torch.load(kp, weights_only=False)
                if tident["digest"] != trec["identity"]["digest"] or kobj["identity_digest"] != trec["identity"]["digest"] or P.sha256_file(kp) != trec["kernel_basis_sha256"]:
                    raise P.ProvenanceError(f"{kp}: tangent basis does not match the loaded inputs")
                Kx = kobj["basis"].to(a.device); tangent_sha = trec["kernel_basis_sha256"]
            theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X, block=a.block); sl = FL.block_slices(net)
            with torch.no_grad():
                F0 = torch.cat([net(X[i:i + a.block]) for i in range(0, X.shape[0], a.block)])
            _, _, _, bins, units = TR.base_diagnostics(net, X, masks, block=a.block); unit_state = [u[1] for u in units]; pos_state = [u[2] for u in units]
            ident = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "alphas": list(ALPHAS), "radii": list(RADII), "direction_seeds": seeds_used, "basis_sha256": srec["basis_sha256"],
                                                      "tangent_basis_sha256": tangent_sha if Kx is not None else None,
                                                      "scaling": "block-balanced (layerwise sets, generic), weighted (tangent)"})
            rec = res["results"].get(stage)
            if rec and rec.get("identity", {}).get("digest") != ident["digest"]:
                if not a.overwrite:
                    raise P.ProvenanceError(f"{name} {stage}: stored rays under another identity; use --overwrite")
                rec = None
            if rec is None or a.overwrite:
                rec = {"epoch": ep, "identity": ident, "sets": {}}
            res["results"][stage] = rec
            # full branch matrices in flat coordinates (columns), built once
            def full_matrix(bases):
                cols = []
                for blk, B in bases.items():
                    if B.shape[1] == 0:
                        continue
                    M = torch.zeros(theta0.numel(), B.shape[1], dtype=torch.float64, device=a.device); M[torch.as_tensor(CJ.block_index_map(net, blk), device=a.device)] = B; cols.append(M)
                return torch.cat(cols, 1) if cols else torch.zeros(theta0.numel(), 0, dtype=torch.float64, device=a.device)
            Bsel = full_matrix(sel); Bat = full_matrix(atomic) if atomic is not None else None; Bcon = full_matrix(constructed) if constructed is not None else None
            # complement of the largest certified branch inside ker J_X: numerical rank of the projected tangent basis (nesting audit)
            comp_rank = None; nesting = None
            if Kx is not None:
                Bref = Bcon if Bcon is not None else Bsel
                proj = Kx - Bref @ (Bref.T @ Kx); svp = torch.linalg.svdvals(proj).cpu().numpy(); comp_rank = int((svp > RTOL * max(svp[0], 1e-300)).sum())
                nesting = float((Bref - Kx @ (Kx.T @ Bref)).norm() / Bref.norm())            # how far the branch lies outside ker J_X
            for sname in SETS:
                if sname == "atomic_sum" and Bat is None:
                    rec["sets"][sname] = {"note": "atomic sum not validated as a branch; skipped"}; continue
                if sname == "constructed" and Bcon is None:
                    rec["sets"][sname] = {"note": "no certified construction record (exp_cnn_branch); skipped"}; continue
                if sname == "tangent_flat" and Kx is None:
                    rec["sets"][sname] = {"note": "tangent kernel not available; run the tangent phase"}; continue
                done = rec["sets"].get(sname, {})
                stale = (sname == "constructed" and done.get("constructed_basis_sha256") != constructed_sha) or \
                        (sname == "tangent_flat" and (done.get("complement_of") != ("constructed" if Bcon is not None else "branch") or done.get("complement_basis_sha256") != (constructed_sha if Bcon is not None else srec["basis_sha256"]["selective"].get("conv5"))))
                if not stale and all(len(done.get(gn, {}).get(str(g), [])) >= len(seeds_used) for gn, grid in (("relative", ALPHAS), ("absolute", RADII)) for g in grid):
                    continue
                out = {"relative": {str(g): [] for g in ALPHAS}, "absolute": {str(g): [] for g in RADII}}; t0 = time.time()
                proto = "weighted" if sname == "tangent_flat" else "block-balanced"; out["relative_protocol"] = proto
                out["dim"] = {"branch": int(Bsel.shape[1]), "constructed": int(Bcon.shape[1]) if Bcon is not None else None, "atomic_sum": int(Bat.shape[1]) if Bat is not None else None,
                              "tangent_flat": comp_rank, "generic": None}[sname]
                if sname == "tangent_flat":
                    out.update({"full_kernel_dim": int(Kx.shape[1]), "complement_rank": comp_rank, "complement_of": "constructed" if Bcon is not None else "branch", "branch_outside_kernel_rel": nesting,
                                "complement_basis_sha256": constructed_sha if Bcon is not None else srec["basis_sha256"]["selective"].get("conv5")})
                if sname == "constructed":
                    out["constructed_basis_sha256"] = constructed_sha
                for ds in seeds_used:
                    g_ = torch.Generator().manual_seed(ds); probe = torch.randn(theta0.numel(), generator=g_, dtype=torch.float64).to(a.device)
                    if sname == "branch":
                        v = Bsel @ (Bsel.T @ probe)
                    elif sname == "atomic_sum":
                        v = Bat @ (Bat.T @ probe)
                    elif sname == "constructed":
                        v = Bcon @ (Bcon.T @ probe)
                    elif sname == "tangent_flat":
                        Bref = Bcon if Bcon is not None else Bsel
                        v = Kx @ (Kx.T @ probe); v = v - Bref @ (Bref.T @ v); v = Kx @ (Kx.T @ v); v = v - Bref @ (Bref.T @ v)
                    else:
                        v = probe
                    for gn, grid, mode in (("relative", ALPHAS, proto), ("absolute", RADII, "absolute")):
                        w, info = FL.scale_direction(v, theta0, sl, mode)
                        for g in grid:
                            m, _, _, _ = TR.ray(net, theta0, masks, X, w, g, F0, bins, unit_state, block=a.block, pos_state=pos_state)
                            m.update({"direction_seed": ds, "euclidean_norm": float(g * w.norm()), "block_fractions": {k: g * v_ for k, v_ in info["unit_step_block_fractions"].items()},
                                      "cancellation_ratio": m["propagated_sum_of_norms"] / m["transition_propagated"] if m["transition_propagated"] > 0 else None, "base_output_norm_X": FL.x_norm(F0)})
                            out[gn][str(g)].append(m)
                    if (seeds_used.index(ds) + 1) % 16 == 0:
                        rec["sets"][sname] = out; records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
                out["seconds"] = time.time() - t0; rec["sets"][sname] = out
                med = lambda gn, g, k: float(np.median([q[k] for q in out[gn][str(g)]]))
                print(f"== {name} {stage} {sname} (dim {out['dim']}, {proto}): alpha=0.01 true {med('relative', 0.01, 'true'):.2e} frozen {med('relative', 0.01, 'frozen'):.2e}; r=0.1 true {med('absolute', 0.1, 'true'):.2e}; r=1 true {med('absolute', 1.0, 'true'):.2e} ({time.time() - t0:.0f}s)", flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def _tests():
    """NaN guard of the validation and per-set resume on a tiny tied CNN."""
    from flatocean import tier1_train as T; import unittest.mock as mock
    torch.manual_seed(0); net = T.CNN([("conv", 1, 2), ("pool",), ("conv", 2, 3), ("pool",)], 2, init_seed=1, dtype=torch.float64, in_hw=8)
    X = torch.randn(6, 1, 8, 8, dtype=torch.float64); spaces, records = patch_spaces(net, X, "cpu")
    theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X)
    v = polynomial_validation(net, X, spaces, theta0, masks, "cpu", n_dirs=4); assert v["passed"], v
    real = FL.frozen_forward; calls = {"n": 0}
    def nan_after_first(*args, **kw):
        calls["n"] += 1; out = real(*args, **kw); return out if calls["n"] == 1 else out * float("nan")
    with mock.patch.object(FL, "frozen_forward", side_effect=nan_after_first):
        v2 = polynomial_validation(net, X, spaces, theta0, masks, "cpu", n_dirs=4)
    assert not v2["passed"] and all(np.isnan(x) for x in v2["residual_rel"].values()), v2
    assert torch.equal(FL.flat_params(net), theta0)
    print("cnn-patch-space tests passed: validation", v["residual_rel"], "NaN rejected, parameters restored")
    _test_spaces_phase_end_to_end(net, X)


def _test_spaces_phase_end_to_end(net, X):
    """The spaces phase on a tiny CNN through saving and reuse: the record is written (regression for the local name that
    shadowed the records module, 2026-09-25), a second run reuses it, a removed basis or a factor appearing later forces
    a recomputation."""
    import tempfile, unittest.mock as mock, argparse
    global OUT
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(KP, "FACTOR_DIR", os.path.join(tmp, "factors")):
        out_before = OUT; OUT = os.path.join(tmp, "branches"); os.makedirs(OUT)
        try:
            ident = {"identity_version": 1, "checkpoint_sha256": "ck", "config": {"protocol": PROTOCOL}}; ident["digest"] = P.identity_digest(ident)
            run = {"net": net, "data": {"analysis": (X,)}, "dir": tmp}
            with mock.patch.object(O, "run_dir", lambda *a_, **k: tmp), mock.patch("flatocean.analyses.cnn_patch_spaces.stage_epochs", lambda rd: {"high_accuracy": {"checkpoint_epoch": 1}}), \
                 mock.patch.object(O, "load_run", lambda *a_, **k: run), mock.patch("flatocean.analyses.cnn_patch_spaces.stage_identity", lambda r, rd, ep: ident):
                a = argparse.Namespace(seeds=[0], stages=["high_accuracy"], device="cpu", threads=1, overwrite=False, dataset="mnist", model="cnn", tag="")
                spaces_phase(a)
                rec = json.load(open(os.path.join(OUT, "mnist_cnn_seed0_patch_spaces.json")))["stages"]["high_accuracy"]
                assert rec["complete"] and rec["identity"]["digest"] == ident["digest"] and rec["selective_sum_validation"]["passed"], rec.keys()
                assert all(os.path.exists(basis_path("mnist_cnn_seed0", 1, "selective", b)) for b in rec["blocks"]), "bases not saved"
                assert stage_incomplete(rec, ident, "mnist_cnn_seed0", 1, run, tmp) is None
                calls = []; real_ps = patch_spaces
                with mock.patch("flatocean.analyses.cnn_patch_spaces.patch_spaces", side_effect=lambda *a_, **k: calls.append(1) or real_ps(*a_, **k)):
                    spaces_phase(a); assert not calls, "complete stage recomputed"
                    os.remove(basis_path("mnist_cnn_seed0", 1, "selective", next(iter(rec["blocks"]))))
                    spaces_phase(a); assert len(calls) == 1, "removed basis not detected"
                    os.makedirs(os.path.join(KP.FACTOR_DIR, "mnist_cnn_seed0")); open(os.path.join(KP.FACTOR_DIR, "mnist_cnn_seed0", "epoch0001_conv1.npz"), "wb").write(b"")
                    rec2 = json.load(open(os.path.join(OUT, "mnist_cnn_seed0_patch_spaces.json")))["stages"]["high_accuracy"]
                    assert stage_incomplete(rec2, ident, "mnist_cnn_seed0", 1, run, tmp) is not None, "a factor appearing later not detected"
        finally:
            OUT = out_before
    print("cnn-patch-space spaces phase: saved, reused, invalidated on a removed basis and on a new factor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["spaces", "tangent", "rays", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--stages", nargs="+", default=["high_accuracy", "mse_interpolation", "epoch_250", "late_endpoint"])
    ap.add_argument("--dataset", default="mnist", help="ADDITIVE (default mnist = byte-identical): CNN run family, e.g. svhn for the P2 native-color CNN.")
    ap.add_argument("--model", default="cnn", help="ADDITIVE (default cnn = byte-identical).")
    ap.add_argument("--tag", default="", help="run tag (e.g. a spawn child lmc_cnn_s0_e250_c0); records are named <dataset>_<model>_<tag>_seed<seed> so parent and child cannot collide")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--ndirs", type=int, default=128); ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args()
    {"spaces": spaces_phase, "tangent": tangent_phase, "rays": rays_phase, "test": lambda a: _tests()}[a.phase](a)


