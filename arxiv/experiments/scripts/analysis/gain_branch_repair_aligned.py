#!/usr/bin/env python
"""ALIGNED-criterion gain-branch tightening (Task E repair v2, 2026-10-03).

The earlier pilot's ~1e-8 full-operator residual is NOT an irreducible floor: the CONSTRUCTION thresholded the
constraint singular values at RTOL*sigma_ref (sigma_ref ~= 84 for W3 => an effective ~8e-8 acceptance), while the
VALIDATION normalizes per image by ||D_i||_F -- a criterion/normalization mismatch that re-expresses the loose
construction cutoff as ~1e-8.  This script aligns them: it TIGHTENS the construction with the ABSOLUTE aligned cutoff
`accept_abs=1e-9` on the unit-covector forward criterion |z^T A u| (branch.build(accept_abs=...)), builds against the
6-sketch consensus UNION of received/covector spaces (so omitted incoming directions are captured, not just matched
thresholds), keeps the final 1e-9 acceptance FIXED, and VALIDATES on FRESH sketches disjoint from the construction ones.
Reports, for BOTH bottom_up and top_down: RETAINED dims, dims SACRIFICED (vs the atomic sum and vs the loose consensus of
the earlier pilot), and the final fresh-sketch residuals -- not a floor.

Usage: OMP_NUM_THREADS=16 python scripts/analysis/gain_branch_repair_aligned.py [--accept-abs 1e-9]
Output: results/gain_maskedbound/mnist_mlp_gain_d1_seed1_branchrepair_aligned.json
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import torch

from flatocean import ocean_lib as O
from flatocean import frozen_lib as FL
import flatocean.analyses.branch as B

OUT = os.path.join(os.path.dirname(B.OUT), "gain_maskedbound")
STAGE = "late_endpoint"


def forward_criterion_full_fromU(nt, M, U, chunk=32):
    """branch.forward_criterion_full against a GIVEN received space U (full operator, all basis elements & received
    vectors, all samples).  U from the FRESH-sketch union is a superset of every fresh single-sketch received space."""
    worst = {}
    for l, blk in enumerate(nt.blocks):
        N = M[blk]
        if N.shape[1] == 0 or U[l].shape[2] == 0:
            worst[blk] = 0.0; continue
        Nt = N.T.reshape(-1, *nt.shape[blk]); w = 0.0
        for i in range(0, nt.s, chunk):
            Au = torch.einsum("knm,smu->sknu", Nt, U[l][i:i + chunk])
            if blk == "V":
                val = Au.norm(dim=2)
            else:
                Dn = nt.D[l][i:i + chunk]; scale = Dn.reshape(Dn.shape[0], -1).norm(dim=1).clamp_min(1e-300)
                val = torch.einsum("sdn,sknu->skdu", Dn, Au).norm(dim=2) / scale.view(-1, 1, 1)
            w = B._nanmax(w, float(val.max()))
        worst[blk] = w
    return worst


def sampled_fresh(nt, M, fresh):
    per = {}
    for s in fresh:
        fc = B.forward_criterion(nt, M, torch.Generator().manual_seed(int(s)))
        per[int(s)] = {k: float(v) for k, v in fc.items()}
    return per, float(max(max(per[s].values()) for s in per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gain_d1"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--build-sketches", type=int, nargs="+", default=[1234, 1235, 1236, 1237, 1238, 1239])
    ap.add_argument("--fresh-sketches", type=int, nargs="+", default=[2234, 2235, 2236, 2237, 2238, 2239])
    ap.add_argument("--accept-abs", type=float, default=1e-9)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_all = time.time()
    name = f"mnist_mlp_{a.tag}_seed{a.seed}"
    build_rec = json.load(open(os.path.join(B.OUT, f"{name}.json")))["results"][STAGE]; ep = build_rec["epoch"]
    run = O.load_run("mnist", "mlp", a.seed, tag=a.tag, epoch=ep, device="cpu")
    X = run["data"]["analysis"][0][: build_rec["identity"]["config"]["samples"]]; net = run["net"]; nt = B.Net(net, X)
    assert build_rec["blocks"]["W1"]["dim_atomic_kernel"] == 0
    B.SVD_SOLVER_COUNTS.clear()
    t0 = time.time(); K, krec = B.atomic_kernels(nt, known_full_rank=("W1",))
    atomic_svd = dict(B.SVD_SOLVER_COUNTS); atomic_gram = atomic_svd.get("gram", 0) > 0
    atomic_dims = {b: int(K[b].shape[1]) for b in nt.blocks}; atomic_sum = int(sum(atomic_dims.values()))
    sref = {b: krec[b].get("sigma1") for b in nt.blocks}
    print(f"[aligned] atomic {atomic_dims} sum {atomic_sum} ({time.time()-t0:.0f}s) svd {atomic_svd} gram {atomic_gram}", flush=True)
    # prior loose-consensus pilot for the comparison (dims only)
    prior = {}
    pp = os.path.join(OUT, f"{name}_branchrepair.json")
    if os.path.exists(pp):
        pj = json.load(open(pp))["orders"]
        prior = {o: {"fragile_total": pj[o]["fragile_single_sketch"]["total"], "loose_total": pj[o]["robust_consensus"]["total"],
                     "loose_dims": pj[o]["robust_consensus"]["dims"], "loose_full_vs_union": pj[o]["robust_consensus"]["full_criterion_max"]} for o in pj}

    out = {"protocol": "Task E repair v2 (2026-10-03): aligned absolute cutoff + consensus, fresh-sketch validation, fixed 1e-9",
           "name": name, "epoch": ep, "accept_abs": a.accept_abs, "valid_tol": B.VALID_TOL,
           "build_sketches": a.build_sketches, "fresh_sketches": a.fresh_sketches,
           "atomic_dims": atomic_dims, "atomic_sum": atomic_sum, "atomic_svd_solver_counts": atomic_svd, "atomic_gram_fallback_used": bool(atomic_gram),
           "orders": {}}
    for order_name in ("bottom_up", "top_down"):
        order = B.orders(nt, 0, a.seed)[order_name]
        _svd0 = dict(B.SVD_SOLVER_COUNTS)
        t0 = time.time()
        M, steps = B.build(nt, order, K, torch.Generator().manual_seed(a.build_sketches[0]), verbose=True,
                            sigma_ref=sref, consensus_seeds=a.build_sketches, accept_abs=a.accept_abs)
        dims = {b: int(M[b].shape[1]) for b in nt.blocks}; total = int(sum(dims.values())); t_build = time.time() - t0
        print(f"[aligned] {order_name} built {dims} total {total} ({t_build:.0f}s)", flush=True)
        # FRESH-sketch validation (disjoint from construction): union received space + full operator + sampled scan + poly
        Uf, _ = B.forward_spaces_consensus(nt, M, a.fresh_sketches)
        full = {k: float(v) for k, v in forward_criterion_full_fromU(nt, M, Uf).items()}; full_max = float(max(full.values()))
        samp_per, samp_worst = sampled_fresh(nt, M, a.fresh_sketches)
        samp_pass = int(sum(max(samp_per[s].values()) <= B.VALID_TOL for s in samp_per))
        poly = B.polynomial_residual(net, nt, X, M, torch.Generator().manual_seed(a.build_sketches[0])); poly_max = float(max(poly.values()))
        svd = {k: B.SVD_SOLVER_COUNTS.get(k, 0) - _svd0.get(k, 0) for k in set(B.SVD_SOLVER_COUNTS) | set(_svd0)}
        gram = bool(svd.get("gram", 0) > 0 or atomic_gram)
        certifies = bool(full_max <= B.VALID_TOL and poly_max <= B.VALID_TOL and not gram)
        w3step = next((s for s in steps if s.get("block") == "W3"), {})
        od = {"retained_dims": dims, "retained_total": total,
              "sacrificed_vs_atomic_sum": atomic_sum - total,
              "sacrificed_vs_fragile": (prior.get(order_name, {}).get("fragile_total") - total) if prior.get(order_name, {}).get("fragile_total") else None,
              "sacrificed_vs_loose_consensus": (prior.get(order_name, {}).get("loose_total") - total) if prior.get(order_name, {}).get("loose_total") else None,
              "prior_loose_consensus": prior.get(order_name),
              "fresh_full_criterion_per_block": full, "fresh_full_criterion_max": full_max,
              "fresh_sampled_per_sketch": samp_per, "fresh_sampled_worst": samp_worst, "fresh_sampled_passing_1e-9": samp_pass,
              "polynomial_residual_max": poly_max, "svd_solver_counts": svd, "gram_fallback_used": gram,
              "certifies_at_fixed_1e-9_fresh": certifies, "build_seconds": round(t_build, 1),
              "W3_build_step": {k: w3step.get(k) for k in ("dim", "rank_within", "gap", "cutoff", "cutoff_mode", "largest_rejected_over_cut", "largest_rejected_over_ref", "sigma_ref")}}
        out["orders"][order_name] = od
        print(f"[aligned] {order_name}: retained {total} (sacrificed vs atomic {od['sacrificed_vs_atomic_sum']}, vs loose {od['sacrificed_vs_loose_consensus']}) | "
              f"fresh full {full_max:.2e} (vs prior loose {prior.get(order_name,{}).get('loose_full_vs_union')}) sampled {samp_pass}/{len(a.fresh_sketches)} poly {poly_max:.2e} gram {gram} CERTIFIES {certifies}", flush=True)

    out["seconds"] = round(time.time() - t_all, 1)
    os.makedirs(OUT, exist_ok=True)
    _sfx = "" if a.accept_abs == 1e-9 else f"_{a.accept_abs:.0e}"           # keep the 1e-9 record's legacy name; suffix others
    op = os.path.join(OUT, f"{name}_branchrepair_aligned{_sfx}.json")
    with open(op + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(op + ".tmp", op)
    print(f"[aligned] saved {op} ({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
