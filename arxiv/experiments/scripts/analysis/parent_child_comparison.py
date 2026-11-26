"""PARENT/CHILD comparison at one MLP fork (two-day completion plan 2026-11-22, "MLP parents and fork time
completion").   usage: python3 parent_child_comparison.py <seed> <epoch>      HEAVY (W2/W3 cross-SVDs ~16.5k).

Spaces (same dataset -- analysis-index identity asserted across all three runs -- same [W b] coordinates, same
canonical bottom-up construction definitions):
  P = the parent's certified branch at the fork checkpoint (mnist_mlp_seed<s>_epoch<e>_parent_bottom_up, W1-inclusive);
  A, B = the children's existing certified branches at epoch 500 (s1_e350: w1_inclusive_bottom_up; s0/s2_e350 and
  s0_e450: reuse_bottom_up, whose W1 component is excluded_frozen -- for e350 backed by the genuine-zero admission
  records w1_admission_s<seed>.json; for e450 no admission record exists and that is REPORTED as an asymmetry).
Construction/support asymmetries between the parent and child spaces are REPORTED, never relabelled.

Primary measurements, per block AND full network, NO angular filtering (mlp_fullspace_overlap.py conventions):
  - sibling displacement d = theta_B - theta_A: ||P_A d||, ||P_B d||, ||P_B P_A d||, ||P_A P_B d|| (norm +
    squared-mass fractions, labelled), plus the ADDITIONAL parent projection ||P_P d||;
  - parent-to-child projection masses ||P_child q||^2 for every parent basis column q, and ||P_parent q||^2 for
    every child basis column, with dims, totals (Frobenius^2), fractions and the dimension-aware expectation k/n;
  - principal-angle distributions per block via the provenance-tracked PS.intersection_eig direct-residual route
    (full-network angle multisets are the per-block unions: the spaces are exactly block-diagonal);
  - numerical intersection and TRIPLE-intersection dimensions separately, under the existing rank-ladder
    convention (lmc_intersection_dims.py: ladder {1e-12,1e-9,1e-7,1e-6,1e-4}, primary sin <= 1e-9, resolved iff
    >= 10x spectral gap at the cut);
  - 256 isotropic + 256 layer-length-matched controls (existing generator convention, seed 909 + 10*seed + epoch);
  - the historical 0.1-degree alignment counts are reproduced but labelled historical, NEVER primary.
Writes results/lmc/decomp/parent_inheritance_s<seed>_e<epoch>.json + _arrays.pt."""

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

PARENTS = {(0, 350), (1, 350), (2, 350), (0, 450)}
SD, E = int(sys.argv[1]), int(sys.argv[2])
assert (SD, E) in PARENTS, f"(seed, epoch) must be one of {sorted(PARENTS)}"
ALIGN_TOL = float(np.sin(np.deg2rad(0.1)))                   # historical 0.1-degree tolerance: labelled, never primary
LADDER = (1e-12, 1e-9, 1e-7, 1e-6, 1e-4); PRIM = 1e-9; GAP = 10.0
BR = os.path.join(HERE, "results", "branches"); OUT = os.path.join(HERE, "results", "lmc", "decomp")
dtype = torch.float64; BLOCKS = ["W1", "W2", "W3", "V"]
CHILD_CONSTRUCTION = "w1_inclusive_bottom_up" if (SD, E) == (1, 350) else "reuse_bottom_up"

# ---------------------------------------------------------------- load + authenticate the three certified branches
prov = {"spaces": {}}
pb = os.path.join(BR, f"mnist_mlp_seed{SD}_epoch{E:04d}_parent_bottom_up")
_need = [pb + ".json", pb + ".pt"] + [os.path.join(BR, f"mnist_mlp_lmc_s{SD}_e{E}_c{c}_seed{SD}_epoch0500_{CHILD_CONSTRUCTION}{x}")
                                      for c in (0, 1) for x in (".json", ".pt")]
_missing = [f for f in _need if not os.path.exists(f)]
if _missing:
    print("FAIL: required certified branch records missing:\n  " + "\n  ".join(_missing), flush=True)
    sys.exit(1)
pj = json.load(open(pb + ".json"))
assert pj.get("certified_branch") is True, "parent branch not certified (comparison must not run on a failed ocean)"
svP = torch.load(pb + ".pt", map_location="cpu", weights_only=False)
for blk in BLOCKS:
    assert svP["bases"][blk].shape[1] == pj["branch_dims"][blk], ("P", blk)
prov["spaces"]["P"] = {"branch_json": os.path.basename(pb + ".json"), "construction": "parent_bottom_up (W1-inclusive)",
                       "fork_epoch": E, "branch_dims": pj["branch_dims"], "certified_branch": True,
                       "w1_admission": pj.get("w1_admission"), "branch_pt_sha256": P.sha256_file(pb + ".pt")}

child_w1_support = {}
sv_ = {}
tagA, tagB = f"lmc_s{SD}_e{E}_c0", f"lmc_s{SD}_e{E}_c1"
for nm, tg in (("A", tagA), ("B", tagB)):
    base = os.path.join(BR, f"mnist_mlp_{tg}_seed{SD}_epoch0500_{CHILD_CONSTRUCTION}")
    bj = json.load(open(base + ".json"))
    assert bj.get("certified_branch") is True, f"{nm}: child branch not certified"
    sv_[nm] = torch.load(base + ".pt", map_location="cpu", weights_only=False)
    for blk in BLOCKS:
        assert sv_[nm]["bases"][blk].shape[1] == bj["branch_dims"][blk], (nm, blk)
    if CHILD_CONSTRUCTION == "reuse_bottom_up":
        assert bj["authentication"]["W1"].get("status") == "excluded_frozen", (nm, "W1 status")
    prov["spaces"][nm] = {"branch_json": os.path.basename(base + ".json"), "construction": CHILD_CONSTRUCTION,
                          "endpoint_epoch": 500, "branch_dims": bj["branch_dims"], "certified_branch": True,
                          "branch_pt_sha256": P.sha256_file(base + ".pt")}
adm_file = os.path.join(BR, f"w1_admission_s{SD}.json")
if CHILD_CONSTRUCTION == "reuse_bottom_up":
    if E == 350 and os.path.exists(adm_file):
        adm = json.load(open(adm_file))
        assert adm.get("passed") and all(adm["children"][c].get("n_admitted", adm["children"][c].get("n_robust")) == 0
                                         for c in ("c0", "c1")), "children's zero-admission record must pass"
        child_w1_support = {"status": "genuine_zero", "record": os.path.basename(adm_file),
                            "note": "children's W1 component GENUINELY ZERO-DIMENSIONAL (zero admitted candidates)"}
    else:
        child_w1_support = {"status": "excluded_frozen_unverified", "record": None,
                            "note": f"no W1 admission record exists for the fork-{E} children: their reuse branches "
                                    "exclude W1 as excluded_frozen WITHOUT a genuine-zero admission measurement; "
                                    "reported as an asymmetry, not relabelled"}
else:
    adm = json.load(open(adm_file))
    assert adm.get("passed"), "s1 admission record must pass"
    child_w1_support = {"status": "admitted_candidates", "record": os.path.basename(adm_file),
                        "n_robust": {c: adm["children"][c]["n_robust"] for c in ("c0", "c1")}}

rA = runs.load_run("mnist", "mlp", SD, tag=tagA, epoch=500, device="cpu", sets=("analysis",))
rB = runs.load_run("mnist", "mlp", SD, tag=tagB, epoch=500, device="cpu", sets=("analysis",))
netA = rA["net"].eval().double(); netB = rB["net"].eval().double()
thA = FL.flat_params(netA).detach().double(); thB = FL.flat_params(netB).detach().double()
p = thA.numel(); d = thB - thA; dn = float(d.norm())
bsl = {name: (a, b) for name, a, b in FL.block_slices(netA)}

# dataset identity: the parent run and both children must analyze the SAME ordered sample set
par_man = json.load(open(os.path.join(runs.run_dir("mnist", "mlp", SD, ""), "data_manifest.json")))
idx_par = P.sha256_index(np.array(par_man["analysis_indices"]))
for nm, rr in (("A", rA), ("B", rB)):
    assert P.sha256_index(rr["data"]["analysis"][2]) == idx_par, f"analysis-index mismatch parent vs child {nm} (FAIL CLOSED)"


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


Q = {"P": {}, "A": {}, "B": {}}
for blk in BLOCKS:
    for nm, sv in (("P", svP), ("A", sv_["A"]), ("B", sv_["B"])):
        Qb = row_to_flatblock(sv["bases"][blk].to(dtype), blk)
        o = float((Qb.T @ Qb - torch.eye(Qb.shape[1], dtype=dtype)).abs().max()) if Qb.shape[1] else 0.0
        assert o <= 1e-8, f"{blk}/{nm}: embed orthonormality {o:.1e}"
        assert Qb.shape[0] == bsl[blk][1] - bsl[blk][0], f"{blk}/{nm}: coordinate size mismatch"
        Q[nm][blk] = Qb

t0 = time.time()


def frac(vec_part, nf):
    r = float(torch.as_tensor(vec_part).norm()) / max(nf, 1e-300)
    return {"norm_frac": r, "sq_mass_frac": r * r}


def overlap_quants(v):
    """The four child-space displacement overlaps (primary, mlp_fullspace_overlap convention) plus the ADDITIONAL
    parent projection PP_d, per block and full network."""
    out = {"per_block": {}, "full_branch": {}}
    acc = {q: torch.zeros(p, dtype=dtype) for q in ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d", "PP_d")}
    nv = float(v.norm())
    for blk in BLOCKS:
        a, b = bsl[blk]; vb = v[a:b]
        Qa, Qb, Qp = Q["A"][blk], Q["B"][blk], Q["P"][blk]
        Pa = Qa @ (Qa.T @ vb) if Qa.shape[1] else torch.zeros_like(vb)
        Pb = Qb @ (Qb.T @ vb) if Qb.shape[1] else torch.zeros_like(vb)
        Pba = Qb @ (Qb.T @ Pa) if Qb.shape[1] else torch.zeros_like(vb)
        Pab = Qa @ (Qa.T @ Pb) if Qa.shape[1] else torch.zeros_like(vb)
        Pp = Qp @ (Qp.T @ vb) if Qp.shape[1] else torch.zeros_like(vb)
        nb = float(vb.norm())
        out["per_block"][blk] = {"PA_d": frac(Pa, nb), "PB_d": frac(Pb, nb), "PB_PA_d": frac(Pba, nb),
                                 "PA_PB_d": frac(Pab, nb), "PP_d": frac(Pp, nb),
                                 "block_norm_frac_of_v": nb / max(nv, 1e-300)}
        for qn, vec in (("PA_d", Pa), ("PB_d", Pb), ("PB_PA_d", Pba), ("PA_PB_d", Pab), ("PP_d", Pp)):
            acc[qn][a:b] = vec
    out["full_branch"] = {qn: frac(acc[qn], nv) for qn in acc}
    return out


res = {"seed": SD, "fork_epoch": E, "p": p, "disp_norm": dn,
       "scope": "parent-inheritance DIAGNOSTICS of certified branch bases (parent at the fork checkpoint vs the two "
                "children at epoch 500); projections use the COMPLETE spaces, NO angular filtering; NOT a new shared "
                "subspace, NOT a finite-distance certificate; fork time preserved per pair (MLP fork {} is never "
                "pooled with CNN fork 200 as a matched architecture condition)".format(E),
       "labels": "norm_frac = ||proj||/||ref||; sq_mass_frac = its square; both reported explicitly; PP_d is the "
                 "ADDITIONAL parent projection of the sibling displacement, not one of the four primary child overlaps",
       "asymmetries": {
           "constructions": {"P": "parent_bottom_up at the FORK checkpoint (W1-inclusive; W1 atomic input = admitted "
                                  "candidates or a measured zero)",
                             "A/B": CHILD_CONSTRUCTION + " at epoch 500"},
           "child_w1_support": child_w1_support,
           "parent_w1": pj.get("w1_admission"),
           "order": "all three constructions use the canonical bottom-up block order (W1, W2, W3, V), generator seed 1234",
           "note": "support differences (W1 components) and the parent/child checkpoint difference are intrinsic to the "
                   "question and REPORTED here, not relabelled as identical constructions"},
       "branch_provenance": prov,
       "displacement": overlap_quants(d)}

# ---------------------------------------------------------------- controls (existing generator convention)
g0 = torch.Generator().manual_seed(909 + 10 * SD + E)
d_bn = {blk: float(d[bsl[blk][0]:bsl[blk][1]].norm()) for blk in BLOCKS}
def matched():
    full = torch.zeros(p, dtype=dtype)
    for blk in BLOCKS:
        a, b = bsl[blk]; w = torch.randn(b - a, generator=g0, dtype=dtype); full[a:b] = w / w.norm() * d_bn[blk]
    return full
QUANTS = ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d", "PP_d")
ctrl_arrays = {}
for mode, sampler in (("isotropic", lambda: torch.randn(p, generator=g0, dtype=dtype)), ("layer_length_matched", matched)):
    fb = {q: [] for q in QUANTS}
    pbl = {blk: {q: [] for q in QUANTS} for blk in BLOCKS}
    for _ in range(256):
        oq = overlap_quants(sampler())
        for q in QUANTS:
            fb[q].append(oq["full_branch"][q]["norm_frac"])
            for blk in BLOCKS:
                pbl[blk][q].append(oq["per_block"][blk][q]["norm_frac"])
    def stats(a):
        a = np.asarray(a, float)
        return {"n": int(a.size), "mean_norm_frac": float(a.mean()), "p5": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean_sq_mass": float((a ** 2).mean())}
    res[f"controls_{mode}"] = {"full_branch": {q: stats(v) for q, v in fb.items()},
                               "per_block": {blk: {q: stats(v) for q, v in pbl[blk].items()} for blk in BLOCKS}}
    ctrl_arrays[mode] = {"full_branch": {q: np.asarray(v) for q, v in fb.items()},
                         "per_block": {blk: {q: np.asarray(v) for q, v in pbl[blk].items()} for blk in BLOCKS}}
for mode in ("isotropic", "layer_length_matched"):
    exc = {}
    for q in QUANTS:
        dv = res["displacement"]["full_branch"][q]["norm_frac"]
        exc[q] = int((ctrl_arrays[mode]["full_branch"][q] >= dv).sum())
    res[f"controls_{mode}"]["full_branch_exceedances_of_256"] = exc
print(f"displacement + controls done [{time.time()-t0:.0f}s]", flush=True)

# ---------------------------------------------------------------- parent-to-child projection masses (per basis column)
mass_arrays = {}
proj = {"per_block": {}, "full_network": {}}
tot = {key: 0.0 for key in ("F2_PA", "F2_PB", "F2_AP", "F2_BP")}
dims_tot = {nm: 0 for nm in ("P", "A", "B")}; n_tot = 0
for blk in BLOCKS:
    a, b = bsl[blk]; n_blk = b - a
    Qp, Qa, Qb = Q["P"][blk], Q["A"][blk], Q["B"][blk]
    kP, kA, kB = Qp.shape[1], Qa.shape[1], Qb.shape[1]
    ent = {"n_block": n_blk, "dims": {"P": kP, "A": kA, "B": kB}}
    def col_masses(Qc, Qt):
        """||P_{span Qt} q||^2 for every column q of Qc (both orthonormal)."""
        if Qc.shape[1] == 0 or Qt.shape[1] == 0:
            return np.zeros(Qc.shape[1])
        return (Qt.T @ Qc).pow(2).sum(0).cpu().numpy()
    mPA, mPB = col_masses(Qp, Qa), col_masses(Qp, Qb)           # parent columns into child spaces
    mAP, mBP = col_masses(Qa, Qp), col_masses(Qb, Qp)           # child columns into the parent space
    def summ(m, k_self, k_target):
        if m.size == 0:
            return {"n_cols": 0, "total_sq_mass": 0.0, "mean": None, "expectation_k_over_n": k_target / n_blk}
        return {"n_cols": int(m.size), "total_sq_mass": float(m.sum()),
                "mean": float(m.mean()), "min": float(m.min()), "max": float(m.max()),
                "p5": float(np.percentile(m, 5)), "p50": float(np.percentile(m, 50)), "p95": float(np.percentile(m, 95)),
                "mean_over_expectation": float(m.mean() / (k_target / n_blk)) if k_target else None,
                "expectation_k_over_n": k_target / n_blk}
    ent["parent_cols_into_A"] = summ(mPA, kP, kA); ent["parent_cols_into_B"] = summ(mPB, kP, kB)
    ent["A_cols_into_P"] = summ(mAP, kA, kP); ent["B_cols_into_P"] = summ(mBP, kB, kP)
    if min(kP, kA, kB) == 0 and blk == "W1":
        ent["note"] = "zero-dimensional component(s): masses are exact zeros (see asymmetries.child_w1_support)"
    proj["per_block"][blk] = ent
    mass_arrays[blk] = {"parent_into_A": mPA, "parent_into_B": mPB, "A_into_P": mAP, "B_into_P": mBP}
    tot["F2_PA"] += float(mPA.sum()); tot["F2_PB"] += float(mPB.sum())
    tot["F2_AP"] += float(mAP.sum()); tot["F2_BP"] += float(mBP.sum())
    for nm, k in (("P", kP), ("A", kA), ("B", kB)):
        dims_tot[nm] += k
    n_tot += n_blk
proj["full_network"] = {
    "dims": dims_tot, "p": p, "p_blocks_total": n_tot,
    "parent_into_A": {"frobenius_sq": tot["F2_PA"], "frac_of_dim_P": tot["F2_PA"] / max(dims_tot["P"], 1),
                      "expectation_sum_kPkA_over_n": sum(Q["P"][b].shape[1] * Q["A"][b].shape[1] / (bsl[b][1] - bsl[b][0]) for b in BLOCKS)},
    "parent_into_B": {"frobenius_sq": tot["F2_PB"], "frac_of_dim_P": tot["F2_PB"] / max(dims_tot["P"], 1),
                      "expectation_sum_kPkB_over_n": sum(Q["P"][b].shape[1] * Q["B"][b].shape[1] / (bsl[b][1] - bsl[b][0]) for b in BLOCKS)},
    "A_into_P": {"frobenius_sq": tot["F2_AP"], "frac_of_dim_A": tot["F2_AP"] / max(dims_tot["A"], 1)},
    "B_into_P": {"frobenius_sq": tot["F2_BP"], "frac_of_dim_B": tot["F2_BP"] / max(dims_tot["B"], 1)},
    "note": "spaces are exactly block-diagonal in the [W b] coordinates, so full-network Frobenius masses are the "
            "sums of the per-block masses"}
res["projection_masses"] = proj
print(f"projection masses done [{time.time()-t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- principal angles + intersections (rank ladder)
def count_ladder(sin_sorted, zeros_exact=0):
    """lmc_intersection_dims.py rank-ladder convention, copied: primary cut sin <= 1e-9, resolved iff >= 10x gap."""
    ent = {"zeros_exact": int(zeros_exact), "counts_at_tol": {}}
    for t in LADDER:
        ent["counts_at_tol"][f"{t:g}"] = int(zeros_exact + (sin_sorted <= t).sum())
    k = int((sin_sorted <= PRIM).sum())
    below = sin_sorted[k - 1] if k > 0 else None
    above = sin_sorted[k] if k < sin_sorted.size else None
    gap = (float(above / below) if (below is not None and above is not None and below > 0) else None)
    ent["primary_dim"] = int(zeros_exact + k)
    ent["gap_at_primary_cut"] = gap
    ent["resolved"] = bool(gap is None or gap >= GAP or k == 0 or k == sin_sorted.size)
    if k == 0 and above is not None:
        ent["resolved"] = bool(above > PRIM * GAP)
    ent["smallest_nonzero_sin"] = float(sin_sorted[sin_sorted > 0][0]) if (sin_sorted > 0).any() else None
    ent["range_1e-9_to_1e-4"] = [ent["counts_at_tol"]["1e-09"], ent["counts_at_tol"]["0.0001"]]
    return ent


PAIRS = (("P", "A"), ("P", "B"), ("A", "B"))
angles = {}; inter = {"convention": "rank cut sin<=1e-9 primary; resolved iff >=10x gap; ladder "
                                    "{1e-12,1e-9,1e-7,1e-6,1e-4}; NOT the 0.1-deg definition; projection analyses unfiltered",
                      "per_block": {}, "full_network": {}}
spectra = {}; pair_eigs = {}
for blk in BLOCKS:
    angles[blk] = {}; inter["per_block"][blk] = {}; n_blk = bsl[blk][1] - bsl[blk][0]
    for x, y in PAIRS:
        key = f"{x}{y}"
        Qx, Qy = Q[x][blk], Q[y][blk]
        kx, ky = Qx.shape[1], Qy.shape[1]
        if min(kx, ky) == 0:
            angles[blk][key] = {"dim_" + x: kx, "dim_" + y: ky,
                                "note": "no principal angles: a component is zero-dimensional"}
            inter["per_block"][blk][key] = {"dims": [kx, ky], "intersection": 0, "genuine_zero": True}
            spectra[f"{blk}_{key}"] = torch.zeros(0)
            continue
        base, other, base_nm = (Qx, Qy, x) if kx <= ky else (Qy, Qx, y)
        snap = dict(PS._SVD_SOLVER_COUNTS)
        r, ev = PS.intersection_eig(base, [other])
        solvers = {k: PS._SVD_SOLVER_COUNTS.get(k, 0) - snap.get(k, 0) for k in set(PS._SVD_SOLVER_COUNTS) | set(snap)}
        pair_eigs[(blk, key)] = (base, r, ev)
        sin_t = np.sort(np.asarray(r.cpu().numpy(), float)); cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
        angles[blk][key] = {"dim_" + x: kx, "dim_" + y: ky, "base": base_nm,
                            "solver_counts_this_block": {k: v for k, v in solvers.items() if v},
                            "smallest_angle_deg": float(np.degrees(np.arcsin(min(sin_t[0], 1.0)))) if sin_t.size else None,
                            "median_angle_deg": float(np.degrees(np.arcsin(min(float(np.median(sin_t)), 1.0)))) if sin_t.size else None,
                            "cos_quantiles": {q: float(np.percentile(cos_t, qq)) for q, qq in
                                              (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))},
                            "mean_sq_overlap": float((cos_t ** 2).mean()),
                            "expected_mean_sq_overlap_random": max(kx, ky) / n_blk,
                            "count_cos_gt_0.99": int((cos_t > 0.99).sum()), "count_cos_gt_0.9": int((cos_t > 0.9).sum()),
                            "historical_0.1deg": {"count_cos_gt_0.9999985": int((sin_t <= ALIGN_TOL).sum()),
                                                  "label": "historical 0.1-degree alignment reproduction; NOT primary"}}
        inter["per_block"][blk][key] = {"dims": [kx, ky], "intersection_ladder": count_ladder(sin_t),
                                        "intersection": count_ladder(sin_t)["primary_dim"]}
        spectra[f"{blk}_{key}"] = torch.from_numpy(sin_t)
        print(f"{blk} {key}: dims {kx}/{ky} theta_min {angles[blk][key]['smallest_angle_deg']:.4f} deg "
              f"inter@1e-9 {inter['per_block'][blk][key]['intersection']} [{time.time()-t0:.0f}s]", flush=True)
    # triple intersection P cap A cap B: intersect the (P,A) intersection subspace with B at each ladder tolerance
    kP, kA, kB = (Q[nm][blk].shape[1] for nm in ("P", "A", "B"))
    if min(kP, kA, kB) == 0:
        inter["per_block"][blk]["PAB_triple"] = {"dims": [kP, kA, kB], "intersection": 0, "genuine_zero": True}
    else:
        base, r, ev = pair_eigs[(blk, "PA")]
        trip = {"dims": [kP, kA, kB], "counts_at_tol": {}, "route": "subspace_at((P,A), tol) intersected with B at tol "
                                                                    "(provenance-tracked PS.intersection_eig route)"}
        prim_entry = None
        for tol in LADDER:
            S, bound, dim = PS.subspace_at(base, r, ev, tol)
            if dim == 0:
                trip["counts_at_tol"][f"{tol:g}"] = 0
                if tol == PRIM:
                    prim_entry = {"primary_dim": 0, "resolved": True, "whole_subspace_bound_PA": bound}
                continue
            r2, _ = PS.intersection_eig(S, [Q["B"][blk]])
            sin2 = np.sort(np.asarray(r2.cpu().numpy(), float))
            trip["counts_at_tol"][f"{tol:g}"] = int((sin2 <= tol).sum())
            if tol == PRIM:
                lad2 = count_ladder(sin2)
                prim_entry = {"primary_dim": lad2["primary_dim"], "resolved": lad2["resolved"],
                              "gap_at_primary_cut": lad2["gap_at_primary_cut"], "whole_subspace_bound_PA": bound}
        trip["primary"] = prim_entry
        trip["intersection"] = prim_entry["primary_dim"] if prim_entry else 0
        inter["per_block"][blk]["PAB_triple"] = trip
        print(f"{blk} PAB triple: {trip['counts_at_tol']} [{time.time()-t0:.0f}s]", flush=True)

fn = {}
for x, y in PAIRS:
    key = f"{x}{y}"
    sins = np.sort(np.concatenate([np.asarray(spectra[f"{blk}_{key}"], float) for blk in BLOCKS]))
    dx = sum(Q[x][b].shape[1] for b in BLOCKS); dy = sum(Q[y][b].shape[1] for b in BLOCKS)
    lad = count_ladder(sins)
    cos2 = float(np.clip(1 - sins ** 2, 0, 1).sum())
    fn[key] = {"dims": [dx, dy], "intersection_ladder": lad, "intersection_primary": lad["primary_dim"],
               "intersection_frac_of_p": lad["primary_dim"] / p, "frobenius_sq": cos2,
               "frobenius_sq_over_min_dim": cos2 / max(min(dx, dy), 1),
               "note": "full-network angle multiset = union of per-block multisets (block-diagonal spaces)"}
fn["PAB_triple"] = {"intersection_primary": sum(inter["per_block"][b]["PAB_triple"].get("intersection", 0) for b in BLOCKS),
                    "counts_at_tol": {f"{t:g}": sum(inter["per_block"][b]["PAB_triple"].get("counts_at_tol", {}).get(f"{t:g}", 0)
                                                    for b in BLOCKS) for t in LADDER}}
inter["full_network"] = fn
res["principal_angles"] = angles
res["intersections"] = inter
res["runtime_s"] = round(time.time() - t0, 1)

outp = os.path.join(OUT, f"parent_inheritance_s{SD}_e{E}.json")
json.dump(res, open(outp, "w"), indent=1, default=str)
torch.save({"spectra_sin_principal_angles": spectra,
            "projection_mass_arrays": mass_arrays,
            "control_norm_frac_arrays": ctrl_arrays,
            "displacement_record": res["displacement"],
            "note": "persisted reusable: per-pair principal-angle sine spectra (per block), per-column projection-mass "
                    "arrays, per-direction control arrays"},
           os.path.join(OUT, f"parent_inheritance_s{SD}_e{E}_arrays.pt"))
fb = res["displacement"]["full_branch"]
print(f"== d overlaps (norm_frac): PA {fb['PA_d']['norm_frac']:.4f} PB {fb['PB_d']['norm_frac']:.4f} "
      f"PB_PA {fb['PB_PA_d']['norm_frac']:.4f} PA_PB {fb['PA_PB_d']['norm_frac']:.4f} PP {fb['PP_d']['norm_frac']:.4f} "
      f"| {res['runtime_s']}s", flush=True)
print("WROTE", outp, "+ _arrays.pt", flush=True)
