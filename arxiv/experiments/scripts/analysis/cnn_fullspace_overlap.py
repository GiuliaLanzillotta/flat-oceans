"""CNN sibling pairs at fork 200: FULL-SPACE continuous-overlap analysis (PI decision 2, 2026-11-16).

PRIMARY convention from this decision on: the COMPLETE certified endpoint invariant subspaces Q_A, Q_B are used
as in the MLP continuous-overlap analysis (lmc_w1branch_overlap_s1.py). NO angular filtering: endpoint bases are
not restricted to directions within 0.1 degrees, and that cutoff is not replaced by another one. The earlier
aligned-intersection containment (cnn_pair_displacement_s0_e200.json) is HISTORICAL SUPPLEMENTARY material.

Reported for the sibling displacement d (P_A = Q_A Q_A^T, P_B = Q_B Q_B^T, blockwise direct sums):
  ||P_A d||/||d||, ||P_B d||/||d||, ||P_B P_A d||/||d||, ||P_A P_B d||/||d||
with squared-mass fractions alongside, per block AND full network; same 256 isotropic + 256 layer-length-matched
controls as the MLP analysis (control draws were not persisted by the earlier record, so fresh draws with the
MLP generator convention, per-direction arrays persisted here). Principal-angle distributions per block from the
provenance-tracked direct-residual formulation (min/median angle, cos quantiles, mean cos^2; descriptive aligned
counts only -- angles are NOT used to discard directions). Sequential projections measure continuous overlap;
they are NOT intersection projectors. Endpoint construction and compatibility tolerances unchanged; certification
is checked, not redone. Writes results/lmc/decomp/cnn_fullspace_overlap_s<seed>_e200.{json,_arrays.pt}.

usage: cnn_fullspace_overlap.py <seed> [fork_epoch in {50,100,150,200,250}; default 200]"""

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
from flatocean import cnn_jacobian as CJ
from flatocean import runs
from flatocean.analyses import persistence_siblings as PS

SD = int(sys.argv[1]); E = int(sys.argv[2]) if len(sys.argv) > 2 else 200
assert E in (50, 100, 150, 200, 250), f"fork epoch {E} not in the sweep grid"
BLOCKS = ["conv1", "conv2", "conv3", "conv4", "conv5", "readout"]
ALIGN_SIN = float(np.sin(np.deg2rad(0.1)))                       # descriptive count only; NOT a filter
BR = os.path.join(HERE, "results", "branches"); OUT = os.path.join(HERE, "results", "lmc", "decomp")
dtype = torch.float64

tags = {"A": f"lmc_cnn_s{SD}_e{E}_c0", "B": f"lmc_cnn_s{SD}_e{E}_c1"}
rA = runs.load_run("mnist", "cnn", SD, tag=tags["A"], epoch=500, device="cpu", sets=("analysis",))
rB = runs.load_run("mnist", "cnn", SD, tag=tags["B"], epoch=500, device="cpu", sets=("analysis",))
thA = FL.flat_params(rA["net"]).detach().double(); thB = FL.flat_params(rB["net"]).detach().double()
p = thA.numel(); d = thB - thA; dn = float(d.norm())
IDX = {b: torch.as_tensor(CJ.block_index_map(rA["net"], b)) for b in BLOCKS}
allidx = torch.cat([IDX[b] for b in BLOCKS])
assert allidx.numel() == p and torch.unique(allidx).numel() == p, "blocks must partition parameter space"

# certification gate (checked, not redone) + provenance
prov = {"endpoints": {}}
Q = {"A": {}, "B": {}}
for nm, ep in (("A", "c0"), ("B", "c1")):
    bj = json.load(open(os.path.join(BR, f"mnist_cnn_lmc_cnn_s{SD}_e{E}_{ep}_seed{SD}_branch.json")))
    bu = bj["results"]["late_endpoint"]["constructions"]["bottom_up"]
    assert bu["certified_branch"] is True, f"{ep}: bottom_up branch not certified"
    bp = torch.load(os.path.join(BR, f"mnist_cnn_lmc_cnn_s{SD}_e{E}_{ep}_seed{SD}_epoch0500_branch_bottom_up.pt"),
                    map_location="cpu", weights_only=False)
    for b in BLOCKS:
        Qb = bp["bases"][b].to(dtype)
        if Qb.shape[1]:
            o = float((Qb.T @ Qb - torch.eye(Qb.shape[1], dtype=dtype)).abs().max())
            assert o <= 1e-8, f"{ep}/{b}: orthonormality {o:.1e}"
        Q[nm][b] = Qb
    prov["endpoints"][ep] = {"tag": tags[nm], "certified_branch": True,
                             "basis_sha256": bu["basis_sha256"],
                             "identity_digest": bj["results"]["late_endpoint"]["identity"]["digest"],
                             "branch_pt_identity_digest": bp.get("identity_digest"),
                             "dims": {b: int(Q[nm][b].shape[1]) for b in BLOCKS}}
prov["analysis_script"] = "scripts/analysis/cnn_fullspace_overlap.py"
print(f"s{SD}: dims A {prov['endpoints']['c0']['dims']} | B {prov['endpoints']['c1']['dims']}", flush=True)

t0 = time.time()


def frac(npart, nref):
    r = npart / max(nref, 1e-300)
    return {"norm_frac": r, "sq_mass_frac": r * r}


def overlap_quants(v):
    """||P_A v||, ||P_B v||, ||P_B P_A v||, ||P_A P_B v|| per block and full network (blockwise projectors)."""
    out = {"per_block": {}, "full_network": {}}
    nv = float(v.norm()); acc = {k: 0.0 for k in ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d")}
    for b in BLOCKS:
        vb = v[IDX[b]]; Qa, Qb = Q["A"][b], Q["B"][b]
        Pa = Qa @ (Qa.T @ vb) if Qa.shape[1] else torch.zeros_like(vb)
        Pb = Qb @ (Qb.T @ vb) if Qb.shape[1] else torch.zeros_like(vb)
        Pba = Qb @ (Qb.T @ Pa) if Qb.shape[1] else torch.zeros_like(Pa)
        Pab = Qa @ (Qa.T @ Pb) if Qa.shape[1] else torch.zeros_like(Pb)
        nb = float(vb.norm())
        ns = {"PA_d": float(Pa.norm()), "PB_d": float(Pb.norm()),
              "PB_PA_d": float(Pba.norm()), "PA_PB_d": float(Pab.norm())}
        out["per_block"][b] = {q: frac(n_, nb) for q, n_ in ns.items()}
        out["per_block"][b]["block_norm_frac_of_v"] = nb / max(nv, 1e-300)
        for q in acc:
            acc[q] += ns[q] ** 2
    out["full_network"] = {q: frac(float(np.sqrt(s)), nv) for q, s in acc.items()}
    return out


res = {"seed": SD, "spawn": E, "p": p, "disp_norm": dn, "provenance": prov,
       "scope": ("PRIMARY analysis (PI decision 2026-11-16): continuous overlap of the COMPLETE certified "
                 "endpoint invariant subspaces Q_A, Q_B -- NO angular filtering, no replacement cutoff. "
                 "Sequential projections P_B P_A, P_A P_B measure continuous overlap; they are NOT intersection "
                 "projectors. The aligned-intersection record (cnn_pair_displacement_s0_e200.json) is historical "
                 "supplementary material. Endpoint construction and compatibility tolerances unchanged."),
       "labels": "norm_frac = ||proj||/||ref||; sq_mass_frac = its square; both reported explicitly",
       "displacement": overlap_quants(d)}

# ---- controls: 256 isotropic + 256 layer-length-matched (MLP-analysis construction; fresh fixed-seed draws,
#      earlier record persisted only scores) ----
g0 = torch.Generator().manual_seed(909 + 10 * SD + E)
d_bn = {b: float(d[IDX[b]].norm()) for b in BLOCKS}
def matched():
    full = torch.zeros(p, dtype=dtype)
    for b in BLOCKS:
        w = torch.randn(IDX[b].numel(), generator=g0, dtype=dtype)
        full[IDX[b]] = w / w.norm() * d_bn[b]
    return full
ctrl_arrays = {}
for mode, sampler in (("isotropic", lambda: torch.randn(p, generator=g0, dtype=dtype)), ("layer_length_matched", matched)):
    fb = {q: [] for q in ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d")}
    pbl = {b: {q: [] for q in fb} for b in BLOCKS}
    for _ in range(256):
        oq = overlap_quants(sampler())
        for q in fb:
            fb[q].append(oq["full_network"][q]["norm_frac"])
            for b in BLOCKS:
                pbl[b][q].append(oq["per_block"][b][q]["norm_frac"])
    def stats(a):
        a = np.asarray(a, float)
        return {"mean_norm_frac": float(a.mean()), "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max()), "mean_sq_mass": float((a ** 2).mean())}
    res[f"controls_{mode}"] = {"generator_seed": 909 + 10 * SD + E, "n": 256,
                              "full_network": {q: stats(v) for q, v in fb.items()},
                              "per_block": {b: {q: stats(v) for q, v in pbl[b].items()} for b in BLOCKS}}
    ctrl_arrays[mode] = {"full_network": {q: np.asarray(v) for q, v in fb.items()},
                         "per_block": {b: {q: np.asarray(v) for q, v in pbl[b].items()} for b in BLOCKS}}
print(f"displacement + controls done [{time.time()-t0:.0f}s]", flush=True)

# ---- principal angles per block: FULL spectra from Q_A^T Q_B (direct-residual, provenance-tracked); angles are
#      reported, never used to discard directions ----
angles = {}; spectra = {}
for b in BLOCKS:
    A, B = Q["A"][b], Q["B"][b]
    if min(A.shape[1], B.shape[1]) == 0:
        angles[b] = {"dim_A": int(A.shape[1]), "dim_B": int(B.shape[1]),
                     "note": "empty branch block at one or both endpoints; no angles defined"}
        continue
    base, other, base_nm = (A, B, "A") if A.shape[1] <= B.shape[1] else (B, A, "B")
    snap = dict(PS._SVD_SOLVER_COUNTS)
    r, ev = PS.intersection_eig(base, [other])     # residual sv = sin principal angles, ascending
    solvers = {k: PS._SVD_SOLVER_COUNTS.get(k, 0) - snap.get(k, 0) for k in set(PS._SVD_SOLVER_COUNTS) | set(snap)}
    sin_t = np.sort(np.asarray(r.cpu(), dtype=np.float64)); cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
    angles[b] = {"dim_A": int(A.shape[1]), "dim_B": int(B.shape[1]), "base": base_nm,
                 "solver_counts_this_block": {k: v for k, v in solvers.items() if v},
                 "smallest_angle_deg": float(np.degrees(np.arcsin(sin_t[0]))),
                 "median_angle_deg": float(np.degrees(np.arcsin(np.median(sin_t)))),
                 "cos_quantiles": {q: float(np.percentile(cos_t, qq)) for q, qq in
                                   (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))},
                 "mean_sq_overlap": float((cos_t ** 2).mean()),
                 "descriptive_counts": {"cos_gt_0.9999985_(0.1deg)": int((sin_t <= ALIGN_SIN).sum()),
                                        "cos_gt_0.99": int((cos_t > 0.99).sum()),
                                        "cos_gt_0.9": int((cos_t > 0.9).sum()),
                                        "note": "descriptive only; NOT used to discard directions"}}
    spectra[b] = torch.from_numpy(sin_t)
    print(f"{b}: dims {A.shape[1]}/{B.shape[1]} theta_min {angles[b]['smallest_angle_deg']:.4f} deg "
          f"median {angles[b]['median_angle_deg']:.2f} deg mean cos^2 {angles[b]['mean_sq_overlap']:.4f} "
          f"solvers {angles[b]['solver_counts_this_block']} [{time.time()-t0:.0f}s]", flush=True)
res["principal_angles"] = angles
res["runtime_s"] = round(time.time() - t0, 1)

outp = os.path.join(OUT, f"cnn_fullspace_overlap_s{SD}_e{E}.json")
json.dump(res, open(outp, "w"), indent=1, default=str)
torch.save({"spectra_sin_principal_angles": spectra, "control_norm_frac_arrays": ctrl_arrays,
            "displacement_record": res["displacement"],
            "note": "persisted reusable: full principal-angle spectra, per-direction control arrays (fresh "
                    "fixed-seed draws, generator 909+10*seed+fork_epoch), displacement record; endpoint bases live in "
                    "results/branches/*.pt (sha-recorded in the JSON provenance)"},
           outp.replace(".json", "_arrays.pt"))
fn = res["displacement"]["full_network"]
print(f"== s{SD} d overlaps (norm_frac): PA {fn['PA_d']['norm_frac']:.4f} PB {fn['PB_d']['norm_frac']:.4f} "
      f"PB_PA {fn['PB_PA_d']['norm_frac']:.4f} PA_PB {fn['PA_PB_d']['norm_frac']:.4f} | {res['runtime_s']}s", flush=True)
print("WROTE", outp, "+ _arrays.pt", flush=True)
