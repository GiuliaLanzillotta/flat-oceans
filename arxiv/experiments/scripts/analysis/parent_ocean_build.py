"""PARENT ocean (certified branch) at the MLP fork checkpoint (two-day completion plan 2026-11-22).

usage: python3 parent_ocean_build.py <seed> <epoch>     (parents: (0,350),(1,350),(2,350),(0,450))   HEAVY

Stage 1 -- W1 ADMISSION CHECK FIRST, identical convention to the children's records (w1_admission_s1.py /
w1_zero_admission.py): robust candidate = autodiff torch.func.jvp exposure <= 1e-12 against the ORIGINAL network.
If the authenticated W1 parent factor has kernel columns, EVERY column is validated (analytic vs autodiff
agreement asserted) and the robust set is admitted (boundary columns reported separately, excluded).  If the
numerical kernel is EMPTY, the measured ZERO is recorded exactly as w1_zero_admission.py does (8 smallest-exposure
stored columns cross-checked by JVP; fp32 agreement floor 3e-4): W1 is then included in the canonical construction
with a measured zero-dimensional atomic input -- NEVER silently frozen or omitted.  Writes
results/branches/w1_admission_parent_s<seed>_e<epoch>.json; any admission failure -> exit 1 before any build.

Stage 2 -- canonical certified branch construction by AUTHENTICATED REUSE of the parent factors (fail-closed
checkpoint/preprocessing/analysis-index/p/analysis-size identity, orthonormality and kernel-exposure prechecks,
as lmc_mlp_branch_build_reuse.py), W1-INCLUSIVE as w1_reuse_build_s1.py: B.build(order = nt.blocks, generator
seed 1234, single sketch) with the CANONICAL sigma_ref convention (sigma_ref_hidden: exact matrix-free power
iteration of the UNRESTRICTED step operator at THIS checkpoint, copied verbatim; defined identically for W1,
l=0).  B.validate certifies (polynomial residual + forward criteria; any Gram fallback disqualifies); post-build
per-column analytic exposure gate 1e-12 on the FINAL components plus an 8-column autodiff spot-check on final W1
(first-order only; certification rests on B.validate).  Certified -> writes
results/branches/mnist_mlp_seed<seed>_epoch<epoch>_parent_bottom_up.{json,pt} (new names, nothing overwritten).
NOT certified -> the full diagnostic record is PRESERVED at ..._parent_bottom_up.FAILED.json, no production
artifacts, exit 1 (fail closed)."""

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

PARENTS = {(0, 350), (1, 350), (2, 350), (0, 450)}
SEED, EP = int(sys.argv[1]), int(sys.argv[2])
assert (SEED, EP) in PARENTS, f"(seed, epoch) must be one of {sorted(PARENTS)}"
GATE = 1e-12                  # admission + post-build exposure gate (children's convention)
ORTHO_TOL = 1e-9              # reused-basis orthonormality (lmc_mlp_branch_build_reuse.py)
EXPO_TOL = 1e-6               # reused-basis kernel-exposure precheck (ditto; authoritative check is B.validate)
BLK2FACTOR = {"W1": "1", "W2": "2", "W3": "3", "V": "readout"}
name = f"mnist_mlp_seed{SEED}"
out_base = os.path.join(B.OUT, f"{name}_epoch{EP:04d}_parent_bottom_up")
adm_path = os.path.join(B.OUT, f"w1_admission_parent_s{SEED}_e{EP}.json")

_missing = [b for b in ("1", "2", "3", "readout")
            if not os.path.exists(os.path.join(HERE, "results", "persistence", name, f"epoch{EP:04d}_{b}.npz"))]
if _missing:
    print(f"FAIL: parent factors missing for blocks {_missing} (run parent_factor_build.py {SEED} {EP} first)", flush=True)
    sys.exit(1)

run = runs.load_run("mnist", "mlp", SEED, tag="", epoch=EP, device="cpu", sets=("analysis",))
rd = runs.run_dir("mnist", "mlp", SEED, "")
net = run["net"].eval().double(); X = run["data"]["analysis"][0].double(); nt = B.Net(net, X)
cur_ck = P.sha256_file(os.path.join(rd, "checkpoints", f"epoch_{EP:04d}.pt"))
cur_pre = P.sha256_file(os.path.join(run["dir"], "preprocessing.pt"))
cur_idx = P.sha256_index(run["data"]["analysis"][2])


def authenticate(z, blk, n, m1, label):
    ident = json.loads(str(z["identity"]))
    for fld, cur in [("checkpoint_sha256", cur_ck), ("preprocessing_sha256", cur_pre), ("analysis_index_sha256", cur_idx)]:
        if ident.get(fld) != cur:
            raise P.ProvenanceError(f"{label}: factor {fld} != run (FAIL CLOSED)")
    if int(z["p"]) != n * m1 or int(ident["analysis_size"]) != nt.s:
        raise P.ProvenanceError(f"{label}: p/analysis_size mismatch (FAIL CLOSED)")
    if int(z["epoch"]) != EP:
        raise P.ProvenanceError(f"{label}: stored epoch {int(z['epoch'])} != requested {EP} (FAIL CLOSED)")


def col_to_row(Vcol, n, m1):
    k = Vcol.shape[1]
    return Vcol.reshape(m1, n, k).transpose(0, 1).reshape(n * m1, k).contiguous()


def sigma_ref_hidden(l, iters=400, tol=1e-13):
    """EXACT canonical sigma_ref (copied verbatim from lmc_mlp_branch_build_reuse.py): largest singular value of
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


# W1 autodiff JVP exposure against the ORIGINAL network (children's admission convention)
n1, m11 = nt.shape["W1"]
lin0 = net.layers[0]
wname = next(nm for nm, pm in net.named_parameters() if pm is lin0.weight)
bname = next(nm for nm, pm in net.named_parameters() if pm is lin0.bias)
base = {nm: pm.detach() for nm, pm in net.named_parameters()}


def expo_autodiff_w1(v):
    A = v.reshape(n1, m11); dW = A[:, :-1].contiguous().double(); db = A[:, -1].contiguous().double()
    def G(w, b):
        pr = dict(base); pr[wname] = w; pr[bname] = b
        return functional_call(net, pr, (X,))
    _, tang = fjvp(G, (base[wname], base[bname]), (dW, db))
    return float((tang.pow(2).sum(1).mean()).sqrt())


def expo_analytic(blk, v):
    n, m1 = nt.shape[blk]; A = v.reshape(n, m1)
    if blk == "V":
        out = torch.einsum("dm,im->id", A, nt.H[nt.L])
    else:
        l = int(blk[1:]) - 1
        dz = torch.einsum("nm,im->in", A, nt.H[l])
        out = torch.einsum("idn,in->id", nt.D[l], dz)
    return float((out.pow(2).sum(1).mean()).sqrt())


# ------------------------------------------------------------------ stage 1: W1 admission (fail closed, runs first)
def w1_admission():
    t0 = time.time(); fails = []
    fp = os.path.join(HERE, "results", "persistence", name, f"epoch{EP:04d}_1.npz")
    z = np.load(fp, allow_pickle=True)
    authenticate(z, "W1", n1, m11, f"{name} W1 admission")
    p = int(z["p"]); ks = int(z["kernel_start"]); kdim = p - ks
    rec = {"seed": SEED, "fork_epoch": EP, "parent": name, "gate_autodiff_exposure": GATE,
           "convention": "identical to the children's admission records (w1_admission_s1.py / w1_zero_admission.py): "
                         "robust candidate = autodiff JVP exposure <= 1e-12 against the original network",
           "factor": fp, "factor_sha256": P.sha256_file(fp), "p": p, "declared_rank": ks, "declared_kernel_dims": kdim}
    sv = np.asarray(z["singular_values"], dtype=float)
    expo = np.concatenate([sv, np.zeros(max(0, p - sv.size))]) / np.sqrt(int(z["s"]))
    if kdim == 0:
        # measured-zero protocol (w1_zero_admission.py): the 8 smallest-exposure stored columns bound all others
        order = np.argsort(expo); cols = np.sort(order[:8])
        Vst = np.asarray(z["V"])                         # read the stored fp32 basis once (one 4.3 GB decompression)
        got = []
        for j in cols:
            vcol = torch.from_numpy(np.ascontiguousarray(Vst[:, int(j):int(j) + 1])).double()[:, 0]
            got.append(expo_autodiff_w1(col_to_row(vcol[:, None], n1, m11)[:, 0]))
        del Vst
        got = np.asarray(got); stored = expo[cols]
        agree = float(np.max(np.abs(got - stored)) / max(float(expo[0]), 1e-300))
        if agree > 3e-4:
            fails.append(f"JVP-vs-stored disagreement {agree:.2e} (fp32 floor 3e-4)")
        n_adm = int((got <= GATE).sum())
        if n_adm != 0:
            fails.append(f"declared-empty kernel but {n_adm} columns under the gate (inconsistent factor)")
        rec.update(n_admitted=0 if not fails else None, admitted_robust_indices=[],
                   min_stored_exposure=float(expo.min()), smallest8_stored_exposure=stored.tolist(),
                   smallest8_jvp_exposure=got.tolist(), jvp_agreement_over_sigma0rate=agree,
                   min_exposure_over_gate=float(expo.min() / GATE),
                   label="MEASURED ZERO: the W1 numerical kernel is empty at this parent checkpoint (zero admitted "
                         "candidates; a real measured zero, not missing/uncomputed); W1 is included in the canonical "
                         "construction with its zero-dimensional atomic input, never silently frozen")
    else:
        Vk = torch.from_numpy(np.asarray(z["V_kernel64"], dtype=np.float64))
        Vrow = col_to_row(Vk, n1, m11)
        ortho = float((Vrow.T @ Vrow - torch.eye(kdim, dtype=torch.float64)).abs().max())
        if ortho > ORTHO_TOL:
            fails.append(f"kernel orthonormality {ortho:.2e} > {ORTHO_TOL:.0e}")
        Hl = nt.H[0]
        const_cols = [j for j in range(Hl.shape[1]) if torch.allclose(Hl[:, j], torch.ones_like(Hl[:, j]))]
        if const_cols != [m11 - 1]:
            fails.append(f"bias-column convention violated (constant cols {const_cols})")
        e_ad = np.empty(kdim); e_an = np.empty(kdim); agree_fail = 0
        for i in range(kdim):
            v = Vrow[:, i]
            e_an[i] = expo_analytic("W1", v); e_ad[i] = expo_autodiff_w1(v)
            if abs(e_an[i] - e_ad[i]) > 1e-12 + 1e-6 * e_ad[i]:
                agree_fail += 1
        if agree_fail:
            fails.append(f"analytic-vs-autodiff disagreement on {agree_fail}/{kdim} columns")
        robust = np.where(e_ad <= GATE)[0]; boundary = np.where(e_ad > GATE)[0]
        rec.update(orthonormality_maxdev=ortho, bias_column_ok=const_cols == [m11 - 1],
                   n_admitted=int(len(robust)), admitted_robust_indices=robust.tolist(),
                   autodiff_exposure_max_robust=float(e_ad[robust].max()) if len(robust) else None,
                   boundary_reported_separately=[{"col": int(i), "autodiff": float(e_ad[i]), "analytic": float(e_an[i])}
                                                 for i in boundary.tolist()],
                   agreement_failures=agree_fail,
                   label="numerically validated kernel candidates (W1); NOT mathematically established exact kernels; "
                         "boundary columns excluded from the primary construction and reported separately")
    rec["passed"] = not fails; rec["failures"] = fails; rec["runtime_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(adm_path, "w"), indent=1)
    print(f"W1 admission: kernel dims {kdim} admitted {rec.get('n_admitted')} "
          f"[{rec['runtime_s']}s] -> {adm_path}", flush=True)
    if fails:
        print("ADMISSION FAIL:\n  " + "\n  ".join(fails), flush=True); sys.exit(1)
    return rec


adm = w1_admission()
admitted = adm["admitted_robust_indices"]

# ------------------------------------------------------------------ stage 2: canonical W1-inclusive build + validation
failures = []
t0 = time.time(); K = {}; auth = {}
for blk in nt.blocks:
    n, m1 = nt.shape[blk]
    fp = os.path.join(HERE, "results", "persistence", name, f"epoch{EP:04d}_{BLK2FACTOR[blk]}.npz")
    z = np.load(fp, allow_pickle=True)
    authenticate(z, blk, n, m1, f"{name} {blk}")
    Vcol = torch.from_numpy(np.asarray(z["V_kernel64"], dtype=np.float64))
    if blk == "W1":
        Vcol = Vcol[:, admitted] if Vcol.shape[1] else Vcol[:, :0]       # admitted robust candidates only
    Vrow = col_to_row(Vcol, n, m1)
    ortho = float((Vrow.T @ Vrow - torch.eye(Vrow.shape[1], dtype=torch.float64)).abs().max()) if Vrow.shape[1] else 0.0
    if Vrow.shape[1] and ortho > ORTHO_TOL:
        raise P.ProvenanceError(f"{name} {blk}: reused basis orthonormality maxdev {ortho:.3e} > {ORTHO_TOL:.0e} (FAIL CLOSED)")
    expo = 0.0
    if blk != "W1" and Vrow.shape[1]:                                    # W1 columns already JVP-validated by admission
        expo = max(expo_analytic(blk, Vrow[:, i]) for i in range(Vrow.shape[1]))
        if expo > EXPO_TOL:
            raise P.ProvenanceError(f"{name} {blk}: reused basis kernel exposure ||Jv||_X {expo:.3e} > {EXPO_TOL:.0e} (FAIL CLOSED)")
    K[blk] = Vrow
    auth[blk] = {"dim": int(Vrow.shape[1]), "auth_ck_pre_idx": "PASS", "orthonormality_maxdev": ortho,
                 "kernel_exposure_Jv_X_max": expo if blk != "W1" else None,
                 "w1_admitted_from": os.path.basename(adm_path) if blk == "W1" else None,
                 "factor_sha256": P.sha256_file(fp)}
    print(f"  {blk}: atomic dim {Vrow.shape[1]} ortho {ortho:.1e} auth PASS", flush=True)
t_load = time.time() - t0

sigma_ref = {"V": None}; sref_iters = {}
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

# post-build exposure gate on the FINAL components (first-order only; certification rests on B.validate)
post = {}
for blk in nt.blocks:
    Mb = M[blk]
    if Mb.shape[1] == 0:
        post[blk] = {"dim": 0}; continue
    ex = np.array([expo_analytic(blk, Mb[:, i]) for i in range(Mb.shape[1])])
    post[blk] = {"dim": int(Mb.shape[1]), "exposure_max": float(ex.max()), "count_gt_gate": int((ex > GATE).sum())}
    if ex.max() > GATE:
        failures.append(f"post-build exposure gate: {blk} max {ex.max():.2e} > {GATE:.0e}")
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
rec = {"parent": name, "seed": SEED, "fork_epoch": EP,
       "method": "parent_bottom_up: canonical W1-INCLUSIVE single-sketch bottom-up construction at the parent fork "
                 "checkpoint, by authenticated reuse of the parent persistence factors (conventions of "
                 "lmc_mlp_branch_build_reuse.py + w1_reuse_build_s1.py; canonical power-iteration sigma_ref)",
       "sigma_ref_used": {blk: {"value": sigma_ref[blk], "power_iters": sref_iters[blk],
                                "source": "sigma_ref_hidden power iteration on the unrestricted step operator at this "
                                          "checkpoint (canonical convention of lmc_mlp_branch_build_reuse.py)"}
                          for blk in ("W1", "W2", "W3")},
       "label": ("W1 candidates are numerically validated, not mathematically established exact kernels; "
                 if len(admitted) else
                 "the W1 atomic input is a MEASURED ZERO (admission record), included, not frozen; ")
                + "the post-build exposure gate is first-order only and does not certify the multilayer branch -- "
                  "certification rests on the validation record",
       "w1_admission": {"record": os.path.basename(adm_path), "n_admitted": len(admitted),
                        "declared_kernel_dims": adm["declared_kernel_dims"], "passed": adm["passed"]},
       "authentication": auth, "branch_dims": dims, "branch_total": total,
       "certified_branch": bool(val["passed"] and not gram_used and not failures),
       "validation": val, "post_build_exposure": post, "post_build_w1_autodiff_spot": spot,
       "svd_solver_counts_build_validate": svd_counts, "gram_fallback_used": gram_used, "failures": failures,
       "steps": [{k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "prepass"})
                  for k, v in s.items() if k != "prepass"} for s in steps],
       "timings_sec": {"load_authenticate": t_load, "construction": t_build, "validation": t_val}}
poly_max = max(val["polynomial_residual_rel"].values())
if not rec["certified_branch"]:
    fpath = out_base + ".FAILED.json"
    json.dump(rec, open(fpath, "w"), indent=2, default=P._jsonable)
    print(f"== {name} e{EP}: CERTIFICATION FAILED (validation passed {val['passed']} gram {gram_used} "
          f"poly {poly_max:.1e}); diagnostic PRESERVED at {fpath}; NO production artifacts; exit 1 (FAIL CLOSED)", flush=True)
    sys.exit(1)
json.dump(rec, open(out_base + ".json", "w"), indent=2, default=P._jsonable)
torch.save({"bases": {b: M[b].contiguous() for b in nt.blocks}, "name": name, "fork_epoch": EP,
            "method": "parent_bottom_up"}, out_base + ".pt")
print(f"== PARENT {name} e{EP}: dims {dims} total {total} certified {rec['certified_branch']} "
      f"poly {poly_max:.1e} gram {gram_used}  timings load {t_load:.0f} build {t_build:.0f} val {t_val:.0f}", flush=True)
print("WROTE", out_base + ".json", "+ .pt", flush=True)
