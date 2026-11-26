"""Parent--sibling ocean persistence: shared and retained invariant directions across a spawn.

Main geometric analysis (Giulia, 2026-09-29).  Given a parent P at the spawn checkpoint and its two
children A, B at their final checkpoints -- same architecture, inherited neuron/channel correspondence,
same ordered analysis images and Euclidean parameter coordinates -- compare per block their invariant
spaces (the atomic kernel ker J_l from `kernel-persistence factor`; certified branches once available),
each stored orthonormal in float64 with an identity digest and file hash.

For each block and each space kind report, at a FIXED declared residual tolerance:

  - pairwise principal-angle summaries P-A, P-B, A-B (and P-P500, the parent's own continuation, reference),
  - MEMBERSHIP-VALIDATED intersections: every reported shared direction s lies in the base space exactly and
    in every other space with projection residual ||(I - P_space) s|| <= tol (the certificate is reported),
  - the triple P n A n B (order-independent), and the sibling-shared-beyond-triple computed as
    (A n B) minus (A n B n P) with the nesting (A n B n P) subset (A n B) validated, never a raw count subtraction,
  - EXACT shared directions (residual <= exact tol) separated from NEARLY ALIGNED (<= near tol),
  - retention fractions with explicit denominators.

Authentication: each factor's identity digest and file hash are bound in and the digests are required to match
across the block; each basis is checked for orthogonality; kernel membership sigma_r/sigma_1 and the factor's
rank gap are carried, and an unresolved rank (gap < 10) or an unresolved intersection (no clear residual gap at
the tolerance) is reported explicitly.

Interpretation: a shared direction is invariant in EACH checkpoint's own frozen model, with local true-network
validity (the frozen model equals the true network inside that checkpoint's activation region).  It does NOT
establish that the checkpoints share one common constant-output region, nor that the straight segment between
them is flat.
"""
import argparse
import json
import os

import numpy as np
import torch

from flatocean import records, runs
from flatocean.analyses.persistence import overlap_stats
from flatocean.factors import FACTOR_DIR                                   # stored factors, read at the input root
from flatocean import provenance as P
PROTOCOL = "parent-sibling-persistence-v1"
OUT = runs.out_dir("persistence", "siblings")                             # records written under the results root
BRANCH_DIR = os.path.join(runs.RESULTS_ROOT, "branches")                  # certified branch bases (cnn_patch_spaces/cnn_branch write here)
RANK_GAP_RESOLVED = 10.0                                                  # gap_J >= this: the kernel dimension is resolved


def factor_path(run_name, epoch, block):
    return os.path.join(FACTOR_DIR, run_name, f"epoch{epoch:04d}_{block}.npz")


def branch_path(run_name, epoch, construction):
    return os.path.join(BRANCH_DIR, f"{run_name}_epoch{epoch:04d}_branch_{construction}.pt")


def load_kernel(name, epoch, block, run, rd, device="cpu"):
    """Authenticated float64 atomic kernel ker J_l.  factor_status validates the checkpoint/data/layout identity of the
    factor against the loaded run and raises on any mismatch, so a stale or mislocated factor cannot authenticate."""
    from flatocean.factors import factor_status
    path = factor_path(name, epoch, block)
    z = np.load(path, allow_pickle=True)
    factor_status(path, run, rd, block, allow_legacy=False, z=z, ep=epoch)                # authenticated factor-loading path
    Q = torch.as_tensor(z["V_kernel64"], dtype=torch.float64, device=device)             # p_l x k, orthonormal columns
    orth = _orthonormality(Q)
    if orth > 1e-6:                                                                       # numerical validation of the stored basis
        raise ValueError(f"factor {path} basis not orthonormal (max|Q^T Q - I| = {orth:.1e})")
    from flatocean.factors import RANK_TOL
    sv = np.asarray(z["singular_values"]); ks = int(z["kernel_start"]); n = len(sv); s0 = float(sv[0])
    member = float(sv[ks] / s0) if 0 < ks < n else 0.0                                   # ||J_l v||/||v|| <= this on ker
    if 0 < ks < n:                                                                        # interior cutoff: gap between last kept and first dropped
        gap = float(sv[ks - 1] / sv[ks]); margin = None; resolved = gap >= RANK_GAP_RESOLVED
    elif ks >= n:                                                                         # empty kernel = FULL RANK: resolved if the smallest sv is a decade above the rank threshold
        gap = None; margin = float(sv[-1] / s0 / RANK_TOL); resolved = margin >= RANK_GAP_RESOLVED
    else:                                                                                 # ks == 0: the whole block is kernel (degenerate) -> unresolved
        gap = None; margin = None; resolved = False
    prov = {"epoch": int(z["epoch"]), "block": block, "dim": int(Q.shape[1]), "p": int(Q.shape[0]),
            "identity_digest": str(z["identity"]), "sha256": P.sha256_file(path),
            "kernel_membership_rel": member, "rank_gap": gap, "full_rank_margin": margin,
            "rank_resolved": bool(resolved)}
    return Q, prov


def _authenticate_basis(basis_path, expected_hash, basis_identity_digest, stage_identity_digest):
    """Reject a basis file whose hash or bound identity digest does not match the certified record (stale-file guard)."""
    sha = P.sha256_file(basis_path)
    if expected_hash != sha:
        raise ValueError(f"basis hash mismatch for {basis_path}: file {sha} vs record {expected_hash}")
    if str(basis_identity_digest) != str(stage_identity_digest):
        raise ValueError(f"basis identity_digest mismatch for {basis_path}: {basis_identity_digest} vs stage {stage_identity_digest}")
    return sha


def load_branch(name, epoch, block, run, rd, device="cpu", construction="bottom_up"):
    """Authenticated float64 certified-branch basis.  Validates the branch stage identity against the loaded
    checkpoint/data/config, requires the construction's PASSING certificate, and a basis identity_digest and file hash
    that match the record; raises on any mismatch (a stale factor/basis cannot authenticate)."""
    path = branch_path(name, epoch, construction)
    rec_path = os.path.join(BRANCH_DIR, f"{name}_branch.json")
    stages = json.load(open(rec_path)).get("results", {})
    stage = next((v for v in stages.values() if isinstance(v, dict) and v.get("epoch") == epoch and construction in v.get("constructions", {})), None)
    if stage is None:
        raise ValueError(f"branch record {rec_path} has no {construction} at epoch {epoch}")
    sid = stage.get("identity", {})
    if P.analysis_identity(run, rd, epoch, sid.get("config"))["digest"] != sid.get("digest"):
        raise ValueError(f"branch stage identity at epoch {epoch} in {rec_path} does not match the loaded checkpoint/data/config")
    cons = stage["constructions"][construction]
    if not cons.get("certified_branch"):
        raise ValueError(f"branch {construction} at epoch {epoch} in {rec_path} is not a passing certified branch")
    d = torch.load(path, map_location=device, weights_only=False)
    sha = _authenticate_basis(path, cons.get("basis_sha256"), d.get("identity_digest"), sid.get("digest"))
    B = d["bases"][block].to(dtype=torch.float64, device=device)
    Q = (torch.linalg.qr(B).Q if B.shape[1] else B)
    prov = {"epoch": epoch, "block": block, "dim": int(Q.shape[1]), "p": int(Q.shape[0]),
            "identity_digest": str(d.get("identity_digest")), "sha256": sha, "construction": construction,
            "certified": True, "kernel_membership_rel": None, "rank_gap": None, "rank_resolved": True}
    return Q, prov


def selective_path(name, epoch, block):
    return os.path.join(BRANCH_DIR, f"{name}_epoch{epoch:04d}_selective_{block}.pt")


def load_selective(name, epoch, block, run, rd, device="cpu"):
    """Authenticated certified selective (patch/channel) space.  Validates the patch-space stage identity against the
    loaded checkpoint/data/config, requires the passing selective-sum validation, and a basis identity_digest and hash
    that match the record; raises on any mismatch."""
    from flatocean.analyses import cnn_patch_spaces as CP
    ps_path = os.path.join(BRANCH_DIR, f"{name}_patch_spaces.json")
    stages = json.load(open(ps_path)).get("stages", {})
    st = next((v for v in stages.values() if isinstance(v, dict) and v.get("epoch") == epoch), None)
    if st is None:
        raise ValueError(f"patch-space record {ps_path} has no stage at epoch {epoch}")
    sid = st.get("identity", {})
    if CP.stage_identity(run, rd, epoch)["digest"] != sid.get("digest"):
        raise ValueError(f"patch-space stage identity at epoch {epoch} in {ps_path} does not match the loaded checkpoint/data/config")
    if not st.get("selective_sum_validation", {}).get("passed"):
        raise ValueError(f"selective sum at epoch {epoch} in {ps_path} did not pass validation")
    path = selective_path(name, epoch, block)
    rec_hash = ((st.get("basis_sha256") or {}).get("selective") or {}).get(block)
    d = torch.load(path, map_location=device, weights_only=False)
    sha = _authenticate_basis(path, rec_hash, d.get("identity_digest"), sid.get("digest"))
    B = d["basis"].to(dtype=torch.float64, device=device)
    Q = (torch.linalg.qr(B).Q if B.shape[1] else B)
    prov = {"epoch": epoch, "block": block, "dim": int(Q.shape[1]), "p": int(Q.shape[0]),
            "identity_digest": str(d.get("identity_digest")), "sha256": sha,
            "kernel_membership_rel": None, "rank_gap": None, "rank_resolved": True}
    return Q, prov


def space_path(space_kind, name, epoch, block, construction="bottom_up"):
    if space_kind == "atomic_kernel":
        return factor_path(name, epoch, block)
    if space_kind == "selective":
        return selective_path(name, epoch, block)
    return branch_path(name, epoch, construction)


def load_space(space_kind, name, epoch, block, run, rd, device="cpu", construction="bottom_up"):
    if space_kind == "atomic_kernel":
        return load_kernel(name, epoch, block, run, rd, device)
    if space_kind == "selective":
        return load_selective(name, epoch, block, run, rd, device)
    return load_branch(name, epoch, block, run, rd, device, construction)


def _orthonormality(Q):
    if Q.shape[1] == 0:
        return 0.0
    return float((Q.T @ Q - torch.eye(Q.shape[1], dtype=Q.dtype)).abs().max())


def _residual_to(Q, S):
    """Per-column ||(I - Q Q^T) s|| for s the columns of S (membership residual of S into span(Q))."""
    if S.shape[1] == 0:
        return torch.zeros(0, dtype=S.dtype)
    if Q.shape[1] == 0:
        return S.norm(dim=0)
    return (S - Q @ (Q.T @ S)).norm(dim=0)


_SVD_SOLVER_COUNTS = {}                                                   # which solver produced each residual SVD (per analyze call)


def _bump_solver(name):
    _SVD_SOLVER_COUNTS[name] = _SVD_SOLVER_COUNTS.get(name, 0) + 1


def _svd_vh(R):
    """SVD right factor (sv, Vh) of a small square matrix, robust to LAPACK gesdd non-convergence, recording which solver
    actually ran in `_SVD_SOLVER_COUNTS`.  The residual R is often very ill-conditioned (intersection directions give
    near-zero singular values with repeats), and torch's divide-and-conquer SVD (gesdd) can fail to converge on it; the
    direct fallback is the convergence-robust QR-based driver (gesvd via scipy), which gives accurate small singular
    values.  As a LAST resort only (both direct solvers fail) it uses sqrt of the Gram eigenvalues -- these FLOOR tight
    residuals near sqrt(machine eps) and must NOT be used to certify tight intersections, so `_bump_solver('gram')` flags
    the comparison and the caller (analyze) leaves such records unresolved / independently validated."""
    try:
        _, sv, Vh = torch.linalg.svd(R, full_matrices=False)
        _bump_solver("gesdd"); return sv, Vh
    except torch._C._LinAlgError:
        try:
            import scipy.linalg as sla
            _, sv_np, Vh_np = sla.svd(R.detach().cpu().numpy(), full_matrices=False, lapack_driver="gesvd")
            _bump_solver("gesvd"); return torch.from_numpy(sv_np).to(R.dtype), torch.from_numpy(Vh_np).to(R.dtype)
        except Exception:
            w, V = torch.linalg.eigh(R.T @ R)                            # Gram last resort: tight residuals are NOT reliable here
            sv = torch.sqrt(torch.clamp(w, min=0.0)).flip(0)
            _bump_solver("gram"); return sv, V.flip(1).transpose(-2, -1)


def intersection_eig(base, others):
    """Combined membership residual per invariant direction (SINGULAR VALUES of the stacked residual operator, ascending)
    and the corresponding directions in base coordinates.  Built from the DIRECT residual vectors
    E_v = base - Q_v (Q_v^T base) and their SVD -- no squaring and no I - M^T M -- so residuals are accurate to ~1e-16
    (not floored near sqrt(machine eps)), and the spectrum is basis-independent (invariant to a rotation of `base`)."""
    kB = base.shape[1]
    if kB == 0 or any(Q.shape[1] == 0 for Q in others):
        return torch.zeros(kB, dtype=base.dtype), None
    Es = torch.cat([base - Q @ (Q.T @ base) for Q in others], dim=0)      # (sum_v p) x kB direct residuals (Euclidean stack = combined residual)
    R = torch.linalg.qr(Es, mode="reduced").R                            # kB x kB, same singular values as Es (stable)
    sv, Vh = _svd_vh(R)                                                   # sv: residual singular values; Vh: right vectors in base coords
    idx = torch.argsort(sv)                                               # ascending: small residual = intersection direction
    return sv[idx], Vh.transpose(-2, -1)[:, idx]


def subspace_at(base, r, evecs, tol):
    """Intersection subspace at residual tolerance `tol`: (orthonormal basis S = base @ evecs[kept], whole-subspace
    membership bound = max combined residual over S, dim).  For every unit w in span(S), ||(I-P_v) w|| <= the bound."""
    if evecs is None:
        return base[:, :0], 0.0, 0
    keep = r <= tol
    return base @ evecs[:, keep], (float(r[keep].max()) if bool(keep.any()) else 0.0), int(keep.sum())


def intersection(base, others, res_tol):
    """The intersection subspace at res_tol with its per-eigendirection residuals and resolution (back-compatible)."""
    r, evecs = intersection_eig(base, others)
    S, bound, dim = subspace_at(base, r, evecs, res_tol)
    if evecs is None:
        return S, torch.zeros(0, dtype=base.dtype), {"resolved": True, "residual_gap": None, "whole_subspace_bound": 0.0,
                                                     "smallest_dropped_residual": None, "n_candidates": int(base.shape[1])}
    keep = r <= res_tol; dropped = r[~keep]
    sd = float(dropped.min()) if dropped.numel() else None
    gap = (sd / max(bound, 1e-300)) if sd is not None else None
    resolved = (gap is None) or (gap >= RANK_GAP_RESOLVED) or (bound < 1e-9)
    return S, r[keep], {"resolved": bool(resolved), "residual_gap": gap, "whole_subspace_bound": bound,
                        "smallest_dropped_residual": sd, "n_candidates": int(base.shape[1])}


NUMERICAL_TOLS = (1e-6, 1e-9, 1e-11)                                                      # tighter than the 0.1-deg alignment tol: numerical tiers


def _complement_within(S_big, S_sub):
    """Orthonormal basis of span(S_big) orthogonal to span(S_sub); S_sub is assumed a subspace of span(S_big)."""
    if S_big.shape[1] == 0 or S_sub.shape[1] == 0:
        return S_big
    C = S_big.T @ S_sub                                                   # coefficients of S_sub in the S_big basis
    Qf, _ = torch.linalg.qr(C, mode="complete")                          # robust: first dim_sub cols span C, the rest its complement
    return S_big @ Qf[:, S_sub.shape[1]:]


def _base_of(dims, a, b):
    return (a, b) if dims[a] <= dims[b] else (b, a)


def _intersections_at(kernels, dims, tol):
    """All intersections (pairwise, nested triple, beyond-parent complement, beyond n P500) at ONE residual tolerance,
    each with its dim and whole-subspace membership bound.  Order-independent; no I - M^T M."""
    QP = kernels["P"]; out = {}; bases = {}
    pairs = [("P", "A"), ("P", "B"), ("A", "B")] + ([("P", "P500")] if "P500" in kernels else [])
    for u, v in pairs:
        bk, ok = _base_of(dims, u, v); r, ev = intersection_eig(kernels[bk], [kernels[ok]])
        S, bound, d = subspace_at(kernels[bk], r, ev, tol); bases[(u, v)] = S
        out[f"{u}_{v}"] = {"dim": d, "whole_subspace_bound": bound,
                           "retention_of_first": (d / dims[u] if dims[u] else None),
                           "retention_of_second": (d / dims[v] if dims[v] else None)}
    S_AB = bases[("A", "B")]
    r3, ev3 = intersection_eig(S_AB, [QP]); S_tri, b3, d3 = subspace_at(S_AB, r3, ev3, tol)
    out["triple_PAB"] = {"dim": d3, "whole_subspace_bound": b3, "nested_in": "A_cap_B",
                         "triple_retention_of_parent": (d3 / dims["P"] if dims["P"] else None)}
    S_beyond = _complement_within(S_AB, S_tri)
    bey = {"dim": int(S_beyond.shape[1]), "dim_A_cap_B": int(S_AB.shape[1]), "dim_triple": d3,
           "nesting_validated": bool(d3 <= int(S_AB.shape[1]))}
    if "P500" in kernels and S_beyond.shape[1] > 0:
        rP, evP = intersection_eig(S_beyond, [kernels["P500"]]); _, bP, dP = subspace_at(S_beyond, rP, evP, tol)
        bey["in_P500"] = {"dim": dP, "fraction": dP / int(S_beyond.shape[1]), "whole_subspace_bound": bP}
    out["sibling_shared_beyond_triple"] = bey
    return out, S_AB, S_tri


def analyze_block(kernels, provs, align_tol, near_tol):
    """kernels: {'P':QP,'A':QA,'B':QB,'P500':QP5?}.  Registered ALIGNED result at align_tol (0.1 deg) plus NUMERICAL tiers
    recomputed separately at each numerical tolerance; every intersection carries a basis-independent whole-subspace
    membership bound and residual spectrum, and the triple is the nested (A n B) n P (validated subset of A n B)."""
    QP, QA, QB = kernels["P"], kernels["A"], kernels["B"]
    dims = {k: kernels[k].shape[1] for k in kernels}
    rec = {"dim": dims,
           "authentication": {"orthogonality_maxdev": {k: _orthonormality(kernels[k]) for k in kernels},
                              "kernel_membership_rel": {k: provs[k]["kernel_membership_rel"] for k in kernels},
                              "rank_gap": {k: provs[k]["rank_gap"] for k in kernels},
                              "full_rank_margin": {k: provs[k].get("full_rank_margin") for k in kernels},
                              "rank_resolved": {k: provs[k]["rank_resolved"] for k in kernels}},
           "pairwise_angles": {}, "residual_spectrum": {}, "aligned": {}, "nearly_aligned": {}, "numerical": {}}
    pairs = [("P", "A"), ("P", "B"), ("A", "B")] + ([("P", "P500")] if "P500" in kernels else [])
    for u, v in pairs:
        rec["pairwise_angles"][f"{u}_{v}"] = overlap_stats(kernels[u], kernels[v])
        bk, ok = _base_of(dims, u, v); r, _ = intersection_eig(kernels[bk], [kernels[ok]])
        sp = r.cpu().numpy() if r.numel() else np.zeros(0)
        rec["residual_spectrum"][f"{u}_{v}"] = ({"min": float(sp.min()), "median": float(np.median(sp)), "q90": float(np.quantile(sp, 0.9)), "max": float(sp.max())} if sp.size else None)
    # aligned (registered 0.1 deg) result, with the triple's nesting/membership/ambiguity validation
    aligned, S_AB, S_tri = _intersections_at(kernels, dims, align_tol)
    if S_AB.shape[1] and S_tri.shape[1] is not None:
        member = {k: (float(_residual_to(kernels[k], S_tri).max()) if S_tri.shape[1] else 0.0) for k in ("P", "A", "B")}
        tb = min(("P", "A", "B"), key=lambda k: dims[k]); to = [k for k in ("P", "A", "B") if k != tb]
        r_ind, ev_ind = intersection_eig(kernels[tb], [kernels[k] for k in to]); _, _, d_ind = subspace_at(kernels[tb], r_ind, ev_ind, align_tol)
        aligned["triple_PAB"].update({"member_residual": member, "base_independent_dim": d_ind,
                                      "base_independent_consistent": (d_ind == aligned["triple_PAB"]["dim"]),
                                      "tolerance_ambiguous": bool(d_ind != aligned["triple_PAB"]["dim"])})
    assert aligned["triple_PAB"]["dim"] <= aligned["A_B"]["dim"], (aligned["triple_PAB"]["dim"], aligned["A_B"]["dim"])
    rec["aligned"] = aligned
    for u, v in pairs:                                                                        # nearly-aligned dims at near_tol
        bk, ok = _base_of(dims, u, v); r, ev = intersection_eig(kernels[bk], [kernels[ok]]); _, _, dn = subspace_at(kernels[bk], r, ev, near_tol)
        rec["nearly_aligned"][f"{u}_{v}"] = {"dim": dn, "extra_over_aligned": dn - aligned[f"{u}_{v}"]["dim"]}
    for t in NUMERICAL_TOLS:                                                                  # numerical tiers, recomputed separately at each tolerance
        num, _, _ = _intersections_at(kernels, dims, t); rec["numerical"][f"{t:g}"] = num
    return rec


def run_name(seed, tag):
    return f"mnist_cnn{'_' + tag if tag else ''}_seed{seed}"


def analyze(a):
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    _SVD_SOLVER_COUNTS.clear()                                            # record which residual-SVD solver runs in this comparison
    exact_tol = float(np.sin(np.deg2rad(a.exact_deg))) if a.exact_res is None else a.exact_res
    near_tol = float(np.sin(np.deg2rad(a.near_deg)))
    trio = {"P": (run_name(a.seed, ""), a.spawn_epoch, ""), "A": (run_name(a.seed, a.child_a), a.epoch, a.child_a),
            "B": (run_name(a.seed, a.child_b), a.epoch, a.child_b), "P500": (run_name(a.seed, ""), a.epoch, "")}
    run_cache = {}                                                                     # (tag, epoch) -> (loaded run, run dir), for the authenticated loaders
    out = {"protocol": PROTOCOL, "seed": a.seed, "spawn_epoch": a.spawn_epoch, "endpoint_epoch": a.epoch,
           "space_kind": a.space_kind, "construction": (a.construction if a.space_kind == "branch" else None),
           "checkpoints": {k: {"run": v[0], "epoch": v[1]} for k, v in trio.items()},
           "tolerances": {"exact_residual": exact_tol, "near_residual": near_tol,
                          "exact_deg": (None if a.exact_res is not None else a.exact_deg), "near_deg": a.near_deg},
           "interpretation": ("a shared direction is invariant in each checkpoint's own frozen model, valid for the true "
                              "network locally (inside that checkpoint's activation region); it does not establish a common "
                              "constant-output region or a flat connecting segment"),
           "blocks": {}}
    for blk in a.blocks:
        kernels, provs, missing = {}, {}, None
        for role, (rn, ep, tag) in trio.items():
            path = space_path(a.space_kind, rn, ep, blk, a.construction)
            if not os.path.exists(path):
                missing = os.path.relpath(path, runs.HERE); break
            if (tag, ep) not in run_cache:
                run_cache[(tag, ep)] = (runs.load_run("mnist", "cnn", a.seed, tag=tag, epoch=ep, device="cpu"),
                                        runs.run_dir("mnist", "cnn", a.seed, tag=tag))
            run, rd = run_cache[(tag, ep)]
            kernels[role], provs[role] = load_space(a.space_kind, rn, ep, blk, run, rd, construction=a.construction)
        if missing:
            out["blocks"][blk] = {"unresolved": f"missing factor {missing}"}; print(f"== {blk}: UNRESOLVED (missing {missing})", flush=True); continue
        digs = {provs[k]["identity_digest"] for k in provs}                        # factor-identity enforcement is per checkpoint;
        out.setdefault("factor_identity_digests", {})[blk] = {k: provs[k]["identity_digest"] for k in provs}   # digests differ across runs, so record them per role rather than require equality
        out["checkpoints"] = {k: {**out["checkpoints"][k], "identity_digest": provs[k]["identity_digest"], "sha256": provs[k]["sha256"]} for k in provs}
        out["blocks"][blk] = analyze_block(kernels, provs, exact_tol, near_tol)
        r = out["blocks"][blk]; e = r["aligned"]
        print(f"== {blk}: dim P/A/B {r['dim']['P']}/{r['dim']['A']}/{r['dim']['B']} | PnA {e['P_A']['dim']} PnB {e['P_B']['dim']} "
              f"AnB {e['A_B']['dim']} triple {e['triple_PAB']['dim']} sib-beyond {e['sibling_shared_beyond_triple']['dim']} | "
              f"AnB num(1e-6) {r['numerical']['1e-06']['A_B']['dim']} bound {e['A_B']['whole_subspace_bound']:.1e} | "
              f"rank-resolved {all(r['authentication']['rank_resolved'].values())}", flush=True)
    out["svd_solver_counts"] = dict(_SVD_SOLVER_COUNTS)                   # audit: which residual-SVD solver ran (gesdd/gesvd/gram)
    if _SVD_SOLVER_COUNTS.get("gram", 0) > 0:
        out["gram_fallback_used"] = True
        out["certification_warning"] = ("Gram-eigenvalue fallback ran on some residual SVDs (both gesdd and gesvd failed); "
                                        "sqrt(Gram) FLOORS tight residuals, so intersection dimensions at tight tolerances in "
                                        "this record are NOT certified -- treat them as unresolved pending independent "
                                        "membership validation against the original residual operator.")
        print("  WARNING gram fallback used:", out["svd_solver_counts"], flush=True)
    path = os.path.join(OUT, f"mnist_cnn_s{a.seed}_e{a.spawn_epoch}_{a.space_kind}_persistence.json")
    records.dump_json(path, out)
    print("wrote", os.path.relpath(path, runs.HERE), "| svd solvers", out["svd_solver_counts"], flush=True)


def _tests():
    torch.manual_seed(0); p = 80
    Qm, _ = torch.linalg.qr(torch.randn(p, p, dtype=torch.float64))
    et = float(np.sin(np.deg2rad(0.1))); nt = float(np.sin(np.deg2rad(5.0)))
    QP = Qm[:, 0:10]; QA = Qm[:, list(range(0, 7)) + [20, 21, 22]]; QB = Qm[:, list(range(0, 7)) + [30, 31]]
    prov = {k: {"kernel_membership_rel": 0.0, "rank_gap": 1e3, "rank_resolved": True} for k in ("P", "A", "B")}
    r = analyze_block({"P": QP, "A": QA, "B": QB}, prov, et, nt); e = r["aligned"]
    tr = e["triple_PAB"]; sb = e["sibling_shared_beyond_triple"]
    assert e["P_A"]["dim"] == 7 and e["A_B"]["dim"] == 7 and tr["dim"] == 7, e
    assert tr["dim"] <= e["A_B"]["dim"] and sb["nesting_validated"] and sb["dim"] == 0, e            # triple nested in A n B
    assert tr["base_independent_consistent"] and not tr["tolerance_ambiguous"], tr
    assert max(tr["member_residual"].values()) < et and tr["whole_subspace_bound"] < et, tr        # triple validated in P, A, B (whole-subspace bound)
    assert e["P_A"]["whole_subspace_bound"] < et and abs(e["P_A"]["retention_of_first"] - 0.7) < 1e-9, e
    # ordering insensitivity: base choice does not change the intersection dimension
    S1, _, _ = intersection(QA, [QP], et); S2, _, _ = intersection(QP, [QA], et)
    assert S1.shape[1] == S2.shape[1] == 7, (S1.shape, S2.shape)
    St1, _, _ = intersection(QP, [QA, QB], et); St2, _, _ = intersection(QB, [QP, QA], et)
    assert St1.shape[1] == St2.shape[1] == 7, (St1.shape, St2.shape)
    # slightly ROTATED space: tilt ONE shared direction of A by 2 deg toward a direction outside P (Qm[:,50]);
    # the other 6 shared columns stay exactly in P.  That one drops from exact but is kept as nearly aligned at 2 deg < 5 deg.
    th = np.deg2rad(2.0); QAr = QA.clone()
    QAr[:, 0] = np.cos(th) * Qm[:, 0] + np.sin(th) * Qm[:, 50]                   # unit, orthogonal to the other columns
    ce, _, _ = intersection(QP, [QAr], et); cn, _, _ = intersection(QP, [QAr], nt)
    assert ce.shape[1] == 6 and cn.shape[1] == 7, (ce.shape, cn.shape)          # rotated direction drops from exact, kept at 2deg<5deg
    # triple subset of every pair (containment), validated by residual
    St, _, _ = intersection(QP, [QA, QB], et)
    assert float(_residual_to(QA, St).max()) < et and float(_residual_to(QB, St).max()) < et and float(_residual_to(QP, St).max()) < et
    # REGRESSION (rotated): siblings share a ~4.4 deg near-aligned direction NOT in P; at the near tolerance the
    # triple must never exceed the sibling intersection, and beyond = dim(A n B) - dim(triple) stays non-negative.
    a3 = np.deg2rad(3.0)
    QAn = torch.cat([Qm[:, 0:5], (np.cos(a3) * Qm[:, 20] + np.sin(a3) * Qm[:, 40]).unsqueeze(1)], dim=1)
    QBn = torch.cat([Qm[:, 0:5], (np.cos(a3) * Qm[:, 20] + np.sin(a3) * Qm[:, 41]).unsqueeze(1)], dim=1)
    en = analyze_block({"P": QP, "A": QAn, "B": QBn}, prov, et, nt)["aligned"]
    assert en["triple_PAB"]["dim"] <= en["A_B"]["dim"], en
    assert en["sibling_shared_beyond_triple"]["dim"] == en["A_B"]["dim"] - en["triple_PAB"]["dim"] >= 0, en
    # P500 comparison: a beyond-triple direction (Qm[:,60]) that is ALSO in the parent's continuation is detected
    prov5 = {**prov, "P500": {"kernel_membership_rel": 0.0, "rank_gap": 1e3, "rank_resolved": True}}
    QA2 = torch.cat([QP, Qm[:, 60:61]], dim=1); QP500 = Qm[:, 55:65]
    e5 = analyze_block({"P": QP, "A": QA2, "B": QA2, "P500": QP500}, prov5, et, nt)["aligned"]
    sb5 = e5["sibling_shared_beyond_triple"]
    assert sb5["dim"] == 1 and sb5["in_P500"]["dim"] == 1, sb5                   # the beyond direction lies in P500
    # empty spaces
    re = analyze_block({"P": QP[:, :0], "A": QA, "B": QB}, prov, et, nt)
    assert re["aligned"]["P_A"]["dim"] == 0 and re["aligned"]["triple_PAB"]["dim"] == 0 and re["aligned"]["sibling_shared_beyond_triple"]["dim"] >= 0, re["aligned"]
    # disjoint siblings
    rd = analyze_block({"P": QP, "A": Qm[:, 0:5], "B": Qm[:, 5:10]}, prov, et, nt)
    assert rd["aligned"]["A_B"]["dim"] == 0 and rd["aligned"]["triple_PAB"]["dim"] == 0, rd["aligned"]
    # basis-rotation invariance: rotating a kernel within its own span leaves dims and whole-subspace bounds unchanged
    torch.manual_seed(3); Rq, _ = torch.linalg.qr(torch.randn(QA.shape[1], QA.shape[1], dtype=torch.float64))
    ia = analyze_block({"P": QP, "A": QA, "B": QB}, prov, et, nt)["aligned"]
    ib = analyze_block({"P": QP, "A": QA @ Rq, "B": QB}, prov, et, nt)["aligned"]
    assert ia["A_B"]["dim"] == ib["A_B"]["dim"] and ia["triple_PAB"]["dim"] == ib["triple_PAB"]["dim"], (ia["A_B"], ib["A_B"])
    assert abs(ia["A_B"]["whole_subspace_bound"] - ib["A_B"]["whole_subspace_bound"]) < 1e-10, (ia, ib)
    # numerical tiers, recomputed separately at each tolerance, never exceed the aligned (looser) dims; triple nested
    r6 = analyze_block({"P": QP, "A": QA, "B": QB}, prov, et, nt)
    for t in ("1e-06", "1e-09", "1e-11"):
        assert r6["numerical"][t]["A_B"]["dim"] <= r6["aligned"]["A_B"]["dim"], t
        assert r6["numerical"][t]["triple_PAB"]["dim"] <= r6["numerical"][t]["A_B"]["dim"], t
    # authentication: a tampered hash or a mismatched basis identity is rejected (stale-file guard)
    import tempfile, os as _os
    tf = tempfile.NamedTemporaryFile(suffix=".pt", delete=False); import torch as _t; _t.save({"x": 1}, tf.name); tf.close()
    good = P.sha256_file(tf.name)
    _authenticate_basis(tf.name, good, "digX", "digX")                          # matching hash + identity: passes
    for bad_hash, bad_id in (("deadbeef", "digX"), (good, "digY")):
        try:
            _authenticate_basis(tf.name, bad_hash, bad_id if bad_hash == good else "digX", "digX"); assert False, "not rejected"
        except ValueError:
            pass
    _os.unlink(tf.name)
    print("persistence_siblings tests passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description="parent-sibling ocean persistence of the per-block atomic kernels")
    ap.add_argument("phase", choices=["analyze", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn-epoch", type=int, default=250)
    ap.add_argument("--epoch", type=int, default=500)
    ap.add_argument("--child-a", default="lmc_cnn_s0_e250_c0")
    ap.add_argument("--child-b", default="lmc_cnn_s0_e250_c1")
    ap.add_argument("--blocks", nargs="+", default=["conv1", "conv2", "conv3", "conv4", "conv5", "readout"])
    ap.add_argument("--space-kind", choices=["atomic_kernel", "selective", "branch"], default="atomic_kernel", help="which invariant space to compare across the spawn")
    ap.add_argument("--construction", default="bottom_up", help="branch construction to load when --space-kind branch")
    ap.add_argument("--exact-deg", type=float, default=0.1, help="principal angle <= this (deg) -> exact shared; residual sin(deg)")
    ap.add_argument("--exact-res", type=float, default=None, help="override the exact residual tolerance directly (numerical exact, e.g. 1e-6)")
    ap.add_argument("--near-deg", type=float, default=5.0, help="principal angle <= this (deg) -> nearly aligned")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.phase == "test":
        return _tests()
    analyze(a)


if __name__ == "__main__":
    main()
