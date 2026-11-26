"""Canonical single-sketch MLP certified-branch build by AUTHENTICATED REUSE of the persistence atomic bases
(2026-10-04).  sigma_ref is computed EXACTLY (not approximately) by matrix-free power iteration on the SAME unrestricted
atomic constraint operator kernel_within uses (rows vec(z u^T), U0 = H/||H||, Z0 = batched_span(D^T)); it reproduces the
canonical krec[b]['sigma1'] to ~1e-13 (verified on e350_c0: W2 34.9638518823, W3 31.8940606286).  Generator seed 1234,
block order = nt.blocks, single sketch (consensus_seeds=None) -- the canonical procedure.  FAIL-CLOSED authentication
(checkpoint / preprocessing / analysis-index / p / analysis_size), orthonormality and kernel-exposure residual, carried
with the build+validate SVD-solver provenance (gesdd / no-Gram -> 1e-9 certifiable).  If a canonical branch record exists
for the child, verifies the dims match.  Usage: python lmc_mlp_branch_build_reuse.py <seed> <spawn> <child 0|1>
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time
import numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); os.chdir(HERE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
from flatocean import runs
from flatocean.analyses import branch as B
from flatocean import provenance as P
BLK2FACTOR = {"W2": "2", "W3": "3", "V": "readout"}
# FAIL-CLOSED authentication tolerances on the REUSED atomic-kernel bases (observed: ortho ~1e-14, exposure ~1e-14).
# A reused basis that violates either is rejected with a non-zero exit (ProvenanceError) BEFORE any reconstruction.
ORTHO_TOL = 1e-9     # basis orthonormality maxdev, to the 1e-9 certification standard (RTOL/VALID_TOL)
EXPO_TOL = 1e-6      # input kernel exposure ||Jv||_X pre-check; the authoritative 1e-9 check is validate() on M


def col_to_row(Vcol, n, m1):
    k = Vcol.shape[1]
    return Vcol.reshape(m1, n, k).transpose(0, 1).reshape(n * m1, k).contiguous()


def exposure(nt, blk, Vrow):
    n, m1 = nt.shape[blk]; mx = 0.0
    for i in range(Vrow.shape[1]):
        A = Vrow[:, i].reshape(n, m1)
        if blk == "V":
            out = torch.einsum("dm,im->id", A, nt.H[nt.L])
        else:
            l = int(blk[1:]) - 1
            out = torch.einsum("idn,in->id", nt.D[l], torch.einsum("nm,im->in", A, nt.H[l]))
        mx = max(mx, float((out.pow(2).sum(1).mean()).sqrt()))
    return mx


def sigma_ref_hidden(nt, l, iters=400, tol=1e-13):
    """EXACT canonical sigma_ref: largest singular value of A -> (Z0[i,:,p]^T A U0[i,:,0]); matrix-free power iteration."""
    U0s = nt.H[l] / nt.H[l].norm(dim=1, keepdim=True)
    Z0, _ = B.batched_span(nt.D[l].transpose(1, 2), ref=float(nt.D[l].norm(dim=(1, 2)).max()))
    n, m1 = nt.shape[nt.blocks[l]]
    g = torch.Generator().manual_seed(0); A = torch.randn(n, m1, generator=g, dtype=U0s.dtype); A /= A.norm(); lam = 0.0
    for it in range(iters):
        Au = torch.einsum("jk,ik->ij", A, U0s); TA = torch.einsum("ij,ijp->ip", Au, Z0)
        Zr = torch.einsum("ip,ijp->ij", TA, Z0); TTA = torch.einsum("ij,ik->jk", Zr, U0s)
        lam_new = float((A * TTA).sum()); A = TTA / TTA.norm()
        if abs(lam_new - lam) <= tol * max(lam_new, 1.0):
            lam = lam_new; break
        lam = lam_new
    return lam ** 0.5, it + 1


def main(seed, spawn, child):
    tag = f"lmc_s{seed}_e{spawn}_c{child}"; name = f"mnist_mlp_{tag}_seed{seed}"
    run = runs.load_run("mnist", "mlp", seed, tag=tag, epoch=500, device="cpu", sets=("analysis",))
    rd = runs.run_dir("mnist", "mlp", seed, tag); X = run["data"]["analysis"][0]
    net = run["net"]; nt = B.Net(net, X)
    cur_ck = P.sha256_file(os.path.join(rd, "checkpoints", "epoch_0500.pt"))
    cur_pre = P.sha256_file(os.path.join(run["dir"], "preprocessing.pt"))
    cur_idx = P.sha256_index(run["data"]["analysis"][2])
    t0 = time.time(); K = {}; auth = {}; sref = {}; sref_iters = {}
    for blk in nt.blocks:
        n, m1 = nt.shape[blk]
        if blk == "W1":
            K[blk] = torch.zeros(n * m1, 0, dtype=torch.float64); auth[blk] = {"status": "excluded_frozen"}; continue
        z = np.load(os.path.join(HERE, "results", "persistence", name, f"epoch0500_{BLK2FACTOR[blk]}.npz"), allow_pickle=True)
        ident = json.loads(str(z["identity"]))
        for fld, cur in [("checkpoint_sha256", cur_ck), ("preprocessing_sha256", cur_pre), ("analysis_index_sha256", cur_idx)]:
            if ident.get(fld) != cur:
                raise P.ProvenanceError(f"{name} {blk}: persistence {fld} != run (FAIL CLOSED)")
        if int(z["p"]) != n * m1 or int(ident["analysis_size"]) != nt.s:
            raise P.ProvenanceError(f"{name} {blk}: p/analysis_size mismatch (FAIL CLOSED)")
        Vrow = col_to_row(torch.from_numpy(np.asarray(z["V_kernel64"], dtype=np.float64)), n, m1)
        ortho = float((Vrow.T @ Vrow - torch.eye(Vrow.shape[1], dtype=torch.float64)).abs().max()) if Vrow.shape[1] else 0.0
        if Vrow.shape[1] and ortho > ORTHO_TOL:
            raise P.ProvenanceError(f"{name} {blk}: reused basis orthonormality maxdev {ortho:.3e} > {ORTHO_TOL:.0e} (FAIL CLOSED)")
        expo = exposure(nt, blk, Vrow) if Vrow.shape[1] else 0.0
        if Vrow.shape[1] and expo > EXPO_TOL:
            raise P.ProvenanceError(f"{name} {blk}: reused basis kernel exposure ||Jv||_X {expo:.3e} > {EXPO_TOL:.0e} (FAIL CLOSED)")
        K[blk] = Vrow
        if blk != "V":
            sr, nit = sigma_ref_hidden(nt, int(blk[1:]) - 1); sref[blk] = sr; sref_iters[blk] = nit
        auth[blk] = {"dim": int(Vrow.shape[1]), "checkpoint_match": True, "preprocessing_match": True, "analysis_index_match": True,
                     "p_match": True, "orthonormality_maxdev": ortho, "kernel_exposure_Jv_X_max": expo,
                     "sigma_ref_power_iter": sref.get(blk), "sigma_ref_iters": sref_iters.get(blk),
                     "factor_sha256": P.sha256_file(os.path.join(HERE, "results", "persistence", name, f"epoch0500_{BLK2FACTOR[blk]}.npz"))}
        print(f"  {blk}: dim {Vrow.shape[1]} ortho {ortho:.1e} exposure {expo:.2e} sigma_ref {sref.get(blk)} auth PASS", flush=True)
    t_load = time.time() - t0
    gen = torch.Generator().manual_seed(1234); svd0 = dict(B.SVD_SOLVER_COUNTS)
    M = {blk: K[blk][:, :0] for blk in nt.blocks}; steps = []; t_span = 0.0; t_solve = 0.0
    for blk in list(nt.blocks):
        tb = time.time(); U, udims = B.forward_spaces(nt, M, gen)
        if blk == "V":
            ts = time.time(); t_span += ts - tb; M[blk], rec = B.solve_readout(nt, U[nt.L]); t_solve += time.time() - ts
        else:
            l = nt.blocks.index(blk); Zhat = B.backward_spaces(nt, M, gen)
            ts = time.time(); t_span += ts - tb
            M[blk], rec = B.solve_hidden(nt, l, U[l], Zhat[l], K[blk], sigma_ref=sref.get(blk)); t_solve += time.time() - ts
            rec["mean_dim_Zhat"] = float((Zhat[l].abs().amax(1) > 0).sum(1).double().mean())
        li = nt.L if blk == "V" else nt.blocks.index(blk)
        rec.update({"block": blk, "mean_dim_U": float(udims[li].double().mean())}); steps.append(rec)
        print(f"     {blk}: dim {rec['dim']} of atomic {K[blk].shape[1]} (sigma_ref {sref.get(blk)}, gap {rec.get('gap')})", flush=True)
    tv = time.time(); val = B.validate(net, nt, X, M, gen); t_val = time.time() - tv
    svd_counts = {k: B.SVD_SOLVER_COUNTS.get(k, 0) - svd0.get(k, 0) for k in set(B.SVD_SOLVER_COUNTS) | set(svd0)}
    gram_used = bool(svd_counts.get("gram", 0) > 0)
    dims = {b: int(M[b].shape[1]) for b in nt.blocks}; total = int(sum(dims.values()))
    cf = os.path.join(B.OUT, f"{name}.json"); matches = None
    if os.path.exists(cf):
        try:
            cbu = json.load(open(cf))["results"]["late_endpoint"]["constructions"]["bottom_up"]; matches = bool(dims == cbu["dims"])
        except Exception:
            matches = None
    rec = {"child": name, "seed": seed, "spawn": spawn, "method": "authenticated_reuse_canonical_singlesketch (power-iter sigma_ref)",
           "authentication": auth, "sigma_ref": sref, "branch_dims": dims, "branch_total": total, "matches_canonical": matches,
           "certified_branch": bool(val["passed"] and not gram_used), "validation": val, "svd_solver_counts": svd_counts,
           "gram_fallback_used": gram_used, "steps": [{k: v for k, v in s.items() if k != "prepass"} for s in steps],
           "timings_sec": {"load_convert_authenticate_sigmaref": t_load, "span_reconstruction": t_span, "shared_block_solve": t_solve, "validation": t_val}}
    poly_max = max(val["polynomial_residual_rel"].values())
    jpath = os.path.join(B.OUT, f"{name}_epoch0500_reuse_bottom_up.json")
    ptpath = os.path.join(B.OUT, f"{name}_epoch0500_reuse_bottom_up.pt")
    if not rec["certified_branch"]:
        # FAIL CLOSED: validation residuals exceeded VALID_TOL or a Gram fallback was used.  Do NOT write the production
        # branch artifacts (so a rerun retries and the master's .pt-exists skip never marks a failed build as done);
        # write a diagnostic record and exit non-zero so `wait_pidfiles ... || exit 1` halts all dependent stages.
        fpath = os.path.join(B.OUT, f"{name}_epoch0500_reuse_bottom_up.FAILED.json")
        json.dump(rec, open(fpath, "w"), indent=2, default=P._jsonable)
        print(f"== {name}: CERTIFICATION FAILED (validation passed {val['passed']} gram {gram_used} poly {poly_max:.1e}); "
              f"wrote diagnostic {fpath}; NO production artifacts written; exiting 1 (FAIL CLOSED)", flush=True)
        sys.exit(1)
    json.dump(rec, open(jpath, "w"), indent=2, default=P._jsonable)
    torch.save({"bases": {b: M[b].contiguous() for b in nt.blocks}, "name": name, "method": "reuse_canonical"}, ptpath)
    print(f"== {name}: dims {dims} total {total} certified {rec['certified_branch']} matches_canonical {matches} gram {gram_used} "
          f"poly {poly_max:.1e}  timings load {t_load:.0f} span {t_span:.0f} solve {t_solve:.0f} val {t_val:.0f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
