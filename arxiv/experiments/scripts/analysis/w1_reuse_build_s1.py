"""W1-INCLUSIVE bottom-up branch construction, seed-1 endpoint (approved 2026-11-11, conditional on w1_admission_s1).

usage: python3 w1_reuse_build_s1.py <child 0|1>

NEW construction (no canonical W1-inclusive reference exists): atomic kernels are REUSED from the authenticated
persistence factors (fail-closed identity checks, as lmc_mlp_branch_reuse_controlled.py); the W1 component is the
ADMITTED robust candidate set from results/branches/w1_admission_s1.json (numerically validated kernel candidates;
the boundary direction is EXCLUDED from this primary construction). The build runs the canonical builder
B.build(order=nt.blocks, gen seed 1234) with the CANONICAL sigma_ref convention (2026-10-04 root-cause fix; the
first run of this script passed sigma_ref=None, which makes kernel_within use sv[0] of the constraint operator
RESTRICTED to the near-kernel candidates (~1e-11..1e-13) instead of the global operator scale (~33), driving the
acceptance cutoff ~12 orders below numerical noise and rejecting even exact kernel candidates — that conservative
record is preserved under SUPERSEDED_fresh_sigma_ref/): every hidden block's sigma_ref is computed by
sigma_ref_hidden — EXACT matrix-free power iteration of the UNRESTRICTED step operator at THIS endpoint, copied
verbatim from lmc_mlp_branch_build_reuse.py (the pipeline that built the existing seed-1 records); this convention
is defined for W1 (l=0) identically, so no new specification is introduced. Validation: B.validate (polynomial
residual + forward criteria; Gram fallback disqualifies certification). POST-BUILD EXPOSURE GATE (new, reviewer
correction 4): per-block per-column analytic exposure of the FINAL components, gate max <= 1e-12, plus an autodiff
JVP spot-check on 8 final-W1 columns; the first-order exposure gate alone does NOT certify the multilayer branch --
certification rests on B.validate. Gate/validation failure -> record saved, exit 1.
Writes results/branches/{name}_epoch0500_w1_inclusive_bottom_up.{json,pt} (new output names; nothing overwritten)."""

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
from torch.func import functional_call, jvp as fjvp

SEED, SPAWN = 1, 350; CHILD = int(sys.argv[1]); GATE = 1e-12
BLK2FACTOR = {"W1": "1", "W2": "2", "W3": "3", "V": "readout"}
tag = f"lmc_s{SEED}_e{SPAWN}_c{CHILD}"; name = f"mnist_mlp_{tag}_seed{SEED}"
failures = []

adm = json.load(open(os.path.join(B.OUT, "w1_admission_s1.json")))
if not adm.get("passed"):
    print("ADMISSION RECORD NOT PASSED -- refusing to build (FAIL CLOSED)", flush=True); sys.exit(1)
admitted = adm["children"][f"c{CHILD}"]["admitted_robust_indices"]

run = runs.load_run("mnist", "mlp", SEED, tag=tag, epoch=500, device="cpu", sets=("analysis",))
rd = runs.run_dir("mnist", "mlp", SEED, tag)
net = run["net"]; X = run["data"]["analysis"][0]; nt = B.Net(net, X)
cur_ck = P.sha256_file(os.path.join(rd, "checkpoints", "epoch_0500.pt"))
cur_pre = P.sha256_file(os.path.join(run["dir"], "preprocessing.pt"))
cur_idx = P.sha256_index(run["data"]["analysis"][2])


def col_to_row(Vcol, n, m1):
    k = Vcol.shape[1]
    return Vcol.reshape(m1, n, k).transpose(0, 1).reshape(n * m1, k).contiguous()


def sigma_ref_hidden(l, iters=400, tol=1e-13):
    """EXACT canonical sigma_ref (copied from lmc_mlp_branch_build_reuse.py): largest singular value of
    A -> (Z0[i,:,p]^T A U0[i,:,0]) on the UNRESTRICTED step operator; matrix-free power iteration."""
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


t0 = time.time(); K = {}; auth = {}
for blk in nt.blocks:
    n, m1 = nt.shape[blk]
    fp = os.path.join(HERE, "results", "persistence", name, f"epoch0500_{BLK2FACTOR[blk]}.npz")
    z = np.load(fp, allow_pickle=True); ident = json.loads(str(z["identity"]))
    for fld, cur in [("checkpoint_sha256", cur_ck), ("preprocessing_sha256", cur_pre), ("analysis_index_sha256", cur_idx)]:
        if ident.get(fld) != cur:
            raise P.ProvenanceError(f"{name} {blk}: persistence {fld} != run (FAIL CLOSED)")
    if int(z["p"]) != n * m1 or int(ident["analysis_size"]) != nt.s:
        raise P.ProvenanceError(f"{name} {blk}: p/analysis_size mismatch (FAIL CLOSED)")
    Vcol = torch.from_numpy(np.asarray(z["V_kernel64"], dtype=np.float64))
    if blk == "W1":
        Vcol = Vcol[:, admitted]                       # admitted robust candidates only (boundary excluded)
    Vrow = col_to_row(Vcol, n, m1)
    ortho = float((Vrow.T @ Vrow - torch.eye(Vrow.shape[1], dtype=torch.float64)).abs().max()) if Vrow.shape[1] else 0.0
    K[blk] = Vrow
    auth[blk] = {"dim": int(Vrow.shape[1]), "orthonormality_maxdev": ortho, "factor_sha256": P.sha256_file(fp),
                 "auth_ck_pre_idx": "PASS", "w1_admitted_from": "w1_admission_s1.json" if blk == "W1" else None}
    print(f"  {blk}: atomic dim {Vrow.shape[1]} ortho {ortho:.1e} auth PASS", flush=True)
t_load = time.time() - t0

sref_iters = {}
sigma_ref = {"V": None}
for blk in ("W1", "W2", "W3"):
    sigma_ref[blk], sref_iters[blk] = sigma_ref_hidden(int(blk[1:]) - 1)
print(f"sigma_ref (power-iter canonical convention): {sigma_ref} (iters {sref_iters})", flush=True)
gen = torch.Generator().manual_seed(1234)
svd0 = dict(B.SVD_SOLVER_COUNTS)
tb = time.time()
M, steps = B.build(nt, list(nt.blocks), K, gen, verbose=True, sigma_ref=sigma_ref)
t_build = time.time() - tb
tv = time.time(); val = B.validate(net, nt, X, M, gen); t_val = time.time() - tv
svd_counts = {k: B.SVD_SOLVER_COUNTS.get(k, 0) - svd0.get(k, 0) for k in set(B.SVD_SOLVER_COUNTS) | set(svd0)}
gram_used = bool(svd_counts.get("gram", 0) > 0)
if not val["passed"]:
    failures.append("B.validate failed")
if gram_used:
    failures.append("gram fallback used on build/validation path (cannot certify)")

# ---- post-build exposure gate on the FINAL components (analytic; autodiff spot-check on W1) ----
def expo_analytic(blk, v):
    n, m1 = nt.shape[blk]; A = v.reshape(n, m1)
    if blk == "V":
        out = torch.einsum("dm,im->id", A, nt.H[nt.L])
    else:
        l = int(blk[1:]) - 1
        dz = torch.einsum("nm,im->in", A, nt.H[l])
        out = torch.einsum("idn,in->id", nt.D[l], dz)
    return float((out.pow(2).sum(1).mean()).sqrt())

post = {}
for blk in nt.blocks:
    Mb = M[blk]
    if Mb.shape[1] == 0:
        post[blk] = {"dim": 0}; continue
    ex = np.array([expo_analytic(blk, Mb[:, i]) for i in range(Mb.shape[1])])
    post[blk] = {"dim": int(Mb.shape[1]), "exposure_max": float(ex.max()), "count_gt_gate": int((ex > GATE).sum())}
    if ex.max() > GATE:
        failures.append(f"post-build exposure gate: {blk} max {ex.max():.2e} > {GATE:.0e}")

# autodiff spot-check: 8 final-W1 columns against the original network
n1, m11 = nt.shape["W1"]
lin = net.layers[0]
wname = next(nm for nm, pm in net.named_parameters() if pm is lin.weight)
bname = next(nm for nm, pm in net.named_parameters() if pm is lin.bias)
netd = net.double(); Xd = X.double()
base = {nm: pm.detach() for nm, pm in netd.named_parameters()}
def expo_autodiff_w1(v):
    A = v.reshape(n1, m11); dW = A[:, :-1].contiguous().double(); db = A[:, -1].contiguous().double()
    def Gf(w, b):
        pr = dict(base); pr[wname] = w; pr[bname] = b
        return functional_call(netd, pr, (Xd,))
    _, tang = fjvp(Gf, (base[wname], base[bname]), (dW, db))
    return float((tang.pow(2).sum(1).mean()).sqrt())
spot = []
if M["W1"].shape[1]:
    g2 = torch.Generator().manual_seed(11)
    cols = sorted(set(torch.randint(0, M["W1"].shape[1], (8,), generator=g2).tolist()))
    for i in cols:
        ad = expo_autodiff_w1(M["W1"][:, i].double())
        spot.append({"col": int(i), "autodiff": ad})
        if ad > GATE:
            failures.append(f"W1 final col {i}: autodiff exposure {ad:.2e} > gate")

dims = {b: int(M[b].shape[1]) for b in nt.blocks}; total = int(sum(dims.values()))
rec = {"child": name, "method": "w1_inclusive_bottom_up (admitted numerically validated W1 kernel candidates; "
                                "atomic factor reuse; CANONICAL sigma_ref for W2/W3; W1 sigma_ref = factor operator scale)",
       "sigma_ref_used": {blk: {"value": sigma_ref[blk], "power_iters": sref_iters[blk],
                                "source": "sigma_ref_hidden power iteration on the unrestricted step operator "
                                          "at this endpoint (canonical convention of lmc_mlp_branch_build_reuse.py)"}
                          for blk in ("W1", "W2", "W3")},
       "label": "W1 candidates are numerically validated, not mathematically established exact kernels; "
                "the post-build exposure gate is first-order only and does not certify the multilayer branch -- "
                "certification rests on the validation record",
       "authentication": auth, "branch_dims": dims, "branch_total": total,
       "w1_admission": {"n_admitted": len(admitted), "boundary_excluded": True},
       "certified_branch": bool(val["passed"] and not gram_used and not failures),
       "validation": val, "post_build_exposure": post, "post_build_w1_autodiff_spot": spot,
       "svd_solver_counts_build_validate": svd_counts, "gram_fallback_used": gram_used,
       "failures": failures,
       "steps": [{k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "prepass"})
                  for k, v in s.items() if k != "prepass"} for s in steps],
       "timings_sec": {"load_authenticate": t_load, "construction": t_build, "validation": t_val}}
outp = os.path.join(B.OUT, f"{name}_epoch0500_w1_inclusive_bottom_up.json")
json.dump(rec, open(outp, "w"), indent=2, default=P._jsonable)
torch.save({"bases": {b: M[b].contiguous() for b in nt.blocks}, "name": name, "method": "w1_inclusive_bottom_up"},
           os.path.join(B.OUT, f"{name}_epoch0500_w1_inclusive_bottom_up.pt"))
print(f"== W1-INCLUSIVE dims {dims} total {total} certified {rec['certified_branch']} "
      f"poly {max(val['polynomial_residual_rel'].values()):.1e} gram {gram_used}", flush=True)
print(("BUILD PASS" if not failures else "BUILD FAIL:\n  " + "\n  ".join(failures)), flush=True)
print("WROTE", outp, flush=True)
sys.exit(0 if not failures else 1)
