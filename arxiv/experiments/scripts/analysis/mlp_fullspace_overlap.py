"""MLP seeds 0/2: continuous-overlap analysis of the certified branch bases (directive A, 2026-11-16).

Identical convention and output format as the seed-1 record (lmc_w1branch_overlap_s1.py ->
w1branch_overlap_s1_e350.json): COMPLETE endpoint invariant-subspace bases Q_A, Q_B per block, NO angular
filtering; the four displacement overlaps ||P_A d||, ||P_B d||, ||P_B P_A d||, ||P_A P_B d|| (norm + squared
fractions, labelled) per block and full branch; 256 isotropic + 256 layer-length-matched controls; principal
angles per block via the reviewed direct-residual formulation with solver provenance.

Seeds 0/2 use the certified authenticated-reuse branches (reuse_bottom_up). Their W1 component is GENUINELY
ZERO-DIMENSIONAL: the admission record (w1_admission_s<seed>.json, identical convention to seed 1's) finds zero
robust W1 kernel candidates (min exposure ~1e-7, >= 1e5 x the 1e-12 gate), so W1's overlaps are EXACT ZEROS
(real zeros, not missing/uncomputed) and W1 principal angles do not exist. Certification gate: the branch JSONs'
certified_branch must be True and dims must match the saved bases. No factor rebuild, no basis recompute.
Writes results/lmc/decomp/w1branch_overlap_s<seed>_e350.json + _arrays.pt (aligned with the s1 record).
usage: mlp_fullspace_overlap.py <seed> [fork_epoch; default 350] with (seed, fork) in {(0,350), (2,350), (0,450)}
HEAVY (W3 cross-SVD ~15k).  Fork 450 (seed 0 only) reuses the identical conventions on the e450 reuse_bottom_up
branches and requires the e450 admission record (w1_admission_s0_e450.json, from w1_zero_admission.py 0 450)."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time
import numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
os.chdir(HERE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
from flatocean import frozen_lib as FL
from flatocean import provenance as P
from flatocean import runs
from flatocean.analyses import persistence_siblings as PS

SD = int(sys.argv[1]); E = int(sys.argv[2]) if len(sys.argv) > 2 else 350
assert (SD, E) in ((0, 350), (2, 350), (0, 450)), \
    f"(seed, fork) = ({SD}, {E}) not admitted; seed 1 fork 350 already has w1branch_overlap_s1_e350.json (w1-inclusive)"
ALIGN_TOL = float(np.sin(np.deg2rad(0.1))); NUM_TOLS = (1e-6, 1e-9, 1e-11)
BR = os.path.join(HERE, "results", "branches"); OUT = os.path.join(HERE, "results", "lmc", "decomp")
dtype = torch.float64; BLOCKS = ["W1", "W2", "W3", "V"]

ADM_NAME = f"w1_admission_s{SD}.json" if E == 350 else f"w1_admission_s{SD}_e{E}.json"
adm = json.load(open(os.path.join(BR, ADM_NAME)))
assert adm["seed"] == SD and adm["spawn"] == E, "admission record does not belong to this (seed, fork) cell"
assert adm.get("passed") and all(adm["children"][c]["n_admitted"] == 0 for c in ("c0", "c1")), \
    "zero-admission record must pass before the W1 component may be reported as a genuine zero"

tagA, tagB = f"lmc_s{SD}_e{E}_c0", f"lmc_s{SD}_e{E}_c1"
rA = runs.load_run("mnist", "mlp", SD, tag=tagA, epoch=500, device="cpu", sets=("analysis",))
rB = runs.load_run("mnist", "mlp", SD, tag=tagB, epoch=500, device="cpu", sets=("analysis",))
netA = rA["net"].eval().double(); netB = rB["net"].eval().double()
thA = FL.flat_params(netA).detach().double(); thB = FL.flat_params(netB).detach().double()
p = thA.numel(); d = thB - thA; dn = float(d.norm())
bsl = {name: (a, b) for name, a, b in FL.block_slices(netA)}

prov = {}
sv_ = {}
for nm, tg in (("A", tagA), ("B", tagB)):
    base = os.path.join(BR, f"mnist_mlp_{tg}_seed{SD}_epoch0500_reuse_bottom_up")
    bj = json.load(open(base + ".json"))
    assert bj.get("certified_branch") is True, f"{nm}: reuse branch not certified"
    assert bj["authentication"]["W1"].get("status") == "excluded_frozen"
    sv_[nm] = torch.load(base + ".pt", map_location="cpu", weights_only=False)
    for blk in BLOCKS:
        assert sv_[nm]["bases"][blk].shape[1] == bj["branch_dims"][blk], (nm, blk)
    prov[nm] = {"branch_json": os.path.basename(base + ".json"), "branch_dims": bj["branch_dims"],
                "certified_branch": True, "branch_pt_sha256": P.sha256_file(base + ".pt")}
svA, svB = sv_["A"], sv_["B"]


def shape_of(blk):
    if blk == "V":
        n, m = netA.readout.weight.shape
    else:
        n, m = netA.layers[int(blk[1:]) - 1].weight.shape
    return n, m + 1


def row_to_flatblock(M, blk):
    n, m1 = shape_of(blk); k = M.shape[1]
    A3 = M.reshape(n, m1, k)
    return torch.cat([A3[:, :-1, :].reshape(n * (m1 - 1), k), A3[:, -1, :]], 0)


QA = {}; QB = {}
for blk in BLOCKS:
    QA[blk] = row_to_flatblock(svA["bases"][blk].to(dtype), blk)
    QB[blk] = row_to_flatblock(svB["bases"][blk].to(dtype), blk)
    for nm, Q in (("A", QA[blk]), ("B", QB[blk])):
        o = float((Q.T @ Q - torch.eye(Q.shape[1], dtype=dtype)).abs().max()) if Q.shape[1] else 0.0
        assert o <= 1e-8, f"{blk}/{nm}: embed orthonormality {o:.1e}"

t0 = time.time()


def frac(vec_part, vec_full):
    nf = float(torch.as_tensor(vec_full).norm()) if not isinstance(vec_full, float) else vec_full
    r = float(torch.as_tensor(vec_part).norm()) / max(nf, 1e-300)
    return {"norm_frac": r, "sq_mass_frac": r * r}


def overlap_quants(v):
    out = {"per_block": {}, "full_branch": {}}
    accA = torch.zeros(p, dtype=dtype); accB = torch.zeros(p, dtype=dtype)
    accBA = torch.zeros(p, dtype=dtype); accAB = torch.zeros(p, dtype=dtype)
    nv = float(v.norm())
    for blk in BLOCKS:
        a, b = bsl[blk]; vb = v[a:b]; Qa, Qb = QA[blk], QB[blk]
        Pa = Qa @ (Qa.T @ vb) if Qa.shape[1] else torch.zeros_like(vb)
        Pb = Qb @ (Qb.T @ vb) if Qb.shape[1] else torch.zeros_like(vb)
        Pba = Qb @ (Qb.T @ Pa) if Qb.shape[1] else torch.zeros_like(vb)
        Pab = Qa @ (Qa.T @ Pb) if Qa.shape[1] else torch.zeros_like(vb)
        nb = float(vb.norm())
        out["per_block"][blk] = {"PA_d": frac(Pa, nb), "PB_d": frac(Pb, nb),
                                 "PB_PA_d": frac(Pba, nb), "PA_PB_d": frac(Pab, nb),
                                 "block_norm_frac_of_v": nb / max(nv, 1e-300)}
        if blk == "W1":
            out["per_block"][blk]["exact_zero"] = True
        accA[a:b] = Pa; accB[a:b] = Pb; accBA[a:b] = Pba; accAB[a:b] = Pab
    out["full_branch"] = {"PA_d": frac(accA, nv), "PB_d": frac(accB, nv),
                          "PB_PA_d": frac(accBA, nv), "PA_PB_d": frac(accAB, nv)}
    return out


res = {"seed": SD, "spawn": E, "p": p, "disp_norm": dn,
       "scope": "continuous overlap DIAGNOSTICS of the certified branch bases; NOT a new shared subspace, NOT a "
                "finite-distance certificate; same convention and format as w1branch_overlap_s1_e350.json",
       "labels": "norm_frac = ||proj||/||ref||; sq_mass_frac = its square; both reported explicitly",
       "w1_component_note": ("W1 branch component GENUINELY ZERO-DIMENSIONAL at both endpoints (zero admitted "
                             f"candidates, {ADM_NAME}: min exposure "
                             f"{min(adm['children'][c]['min_stored_exposure'] for c in ('c0', 'c1')):.1e} "
                             f"vs gate {adm['gate_autodiff_exposure']:g}); its overlap "
                             "entries are exact zeros (real zeros, not missing/uncomputed); W1 principal angles do "
                             "not exist"),
       "branch_provenance": prov,
       "displacement": overlap_quants(d)}

g0 = torch.Generator().manual_seed(909 + 10 * SD + E)
d_bn = {blk: float(d[bsl[blk][0]:bsl[blk][1]].norm()) for blk in BLOCKS}
def matched():
    full = torch.zeros(p, dtype=dtype)
    for blk in BLOCKS:
        a, b = bsl[blk]; w = torch.randn(b - a, generator=g0, dtype=dtype); full[a:b] = w / w.norm() * d_bn[blk]
    return full
ctrl_arrays = {}
for mode, sampler in (("isotropic", lambda: torch.randn(p, generator=g0, dtype=dtype)), ("layer_length_matched", matched)):
    fb = {q: [] for q in ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d")}
    pbl = {blk: {q: [] for q in ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d")} for blk in BLOCKS}
    for _ in range(256):
        oq = overlap_quants(sampler())
        for q in fb:
            fb[q].append(oq["full_branch"][q]["norm_frac"])
            for blk in BLOCKS:
                pbl[blk][q].append(oq["per_block"][blk][q]["norm_frac"])
    def stats(a):
        a = np.asarray(a, float)
        return {"mean_norm_frac": float(a.mean()), "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "mean_sq_mass": float((a ** 2).mean())}
    res[f"controls_{mode}"] = {"full_branch": {q: stats(v) for q, v in fb.items()},
                               "per_block": {blk: {q: stats(v) for q, v in pbl[blk].items()} for blk in BLOCKS}}
    ctrl_arrays[mode] = {"full_branch": {q: np.asarray(v) for q, v in fb.items()},
                         "per_block": {blk: {q: np.asarray(v) for q, v in pbl[blk].items()} for blk in BLOCKS}}
# exceedance counts (statistical wording convention: counts among 256, plus-one MC estimate 1/257)
for mode in ("isotropic", "layer_length_matched"):
    exc = {}
    for q in ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d"):
        dv = res["displacement"]["full_branch"][q]["norm_frac"]
        exc[q] = int((ctrl_arrays[mode]["full_branch"][q] >= dv).sum())
    res[f"controls_{mode}"]["full_branch_exceedances_of_256"] = exc
print(f"displacement + controls done [{time.time()-t0:.0f}s]", flush=True)

spectra = {}; angles = {}; Sbases = {}
for blk in BLOCKS:
    A, B = QA[blk], QB[blk]
    if min(A.shape[1], B.shape[1]) == 0:
        angles[blk] = {"dim_A": int(A.shape[1]), "dim_B": int(B.shape[1]),
                       "note": "no principal angles: component genuinely zero-dimensional"}
        spectra[blk] = {"sin_principal_angles_ascending": torch.zeros(0)}
        continue
    base, other, base_nm = (A, B, "A") if A.shape[1] <= B.shape[1] else (B, A, "B")
    snap = dict(PS._SVD_SOLVER_COUNTS)
    r, ev = PS.intersection_eig(base, [other])
    solvers = {k: PS._SVD_SOLVER_COUNTS.get(k, 0) - snap.get(k, 0) for k in set(PS._SVD_SOLVER_COUNTS) | set(snap)}
    sin_t = r.cpu().numpy(); cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
    angles[blk] = {"dim_A": int(A.shape[1]), "dim_B": int(B.shape[1]), "base": base_nm,
                   "solver_counts_this_block": {k: v for k, v in solvers.items() if v},
                   "smallest_angle_deg": float(np.degrees(np.arcsin(sin_t[0]))) if sin_t.size else None,
                   "median_angle_deg": float(np.degrees(np.arcsin(np.median(sin_t)))) if sin_t.size else None,
                   "cos_quantiles": {q: float(np.percentile(cos_t, qq)) for q, qq in
                                      (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))},
                   "mean_sq_overlap": float((cos_t ** 2).mean()),
                   "count_cos_gt_0.9999985": int((sin_t <= ALIGN_TOL).sum()),
                   "count_cos_gt_0.99": int((cos_t > 0.99).sum()), "count_cos_gt_0.9": int((cos_t > 0.9).sum())}
    spectra[blk] = {"sin_principal_angles_ascending": torch.from_numpy(sin_t)}
    tiers = {}
    for t in (ALIGN_TOL,) + NUM_TOLS:
        S, bound, dim = PS.subspace_at(base, r, ev, t)
        tiers[f"{t:g}"] = {"dim": dim, "bound": bound}
        if t == ALIGN_TOL:
            Sbases[blk] = S
    angles[blk]["intersection_tiers"] = tiers
    print(f"{blk}: dims {A.shape[1]}/{B.shape[1]} theta_min {angles[blk]['smallest_angle_deg']:.3f} deg "
          f"median {angles[blk]['median_angle_deg']:.2f} deg mean cos^2 {angles[blk]['mean_sq_overlap']:.4f} "
          f"solvers {angles[blk]['solver_counts_this_block']} [{time.time()-t0:.0f}s]", flush=True)

res["principal_angles"] = angles
res["runtime_s"] = round(time.time() - t0, 1)
outp = os.path.join(OUT, f"w1branch_overlap_s{SD}_e{E}.json")
json.dump(res, open(outp, "w"), indent=1, default=str)
torch.save({"spectra_sin_principal_angles": {b: spectra[b]["sin_principal_angles_ascending"] for b in BLOCKS},
            "aligned_intersection_bases_flatblock": {b: Sbases[b] for b in Sbases},
            "control_norm_frac_arrays": ctrl_arrays, "displacement_record": res["displacement"],
            "note": "persisted reusable: principal-angle spectra, aligned (0.1deg) intersection bases in flat "
                    "block coords, per-direction control arrays"},
           os.path.join(OUT, f"w1branch_overlap_s{SD}_e{E}_arrays.pt"))
fb = res["displacement"]["full_branch"]
print(f"== d overlaps (norm_frac): PA {fb['PA_d']['norm_frac']:.4f} PB {fb['PB_d']['norm_frac']:.4f} "
      f"PB_PA {fb['PB_PA_d']['norm_frac']:.4f} PA_PB {fb['PA_PB_d']['norm_frac']:.4f} | {res['runtime_s']}s", flush=True)
print("WROTE", outp, "+ _arrays.pt", flush=True)
