"""Dimensionality tradeoff and the neuron-wise / cross-neuron decomposition of an MLP layer's atomic kernel
(plan Section 0.6 items 1, 2 and the decisions of 2026-09-24; 0.7 procedure).

Quantities of hidden layer l (augmented input h_i in R^m, n units, downstream maps D_i (d x n), block J_l = [J_l1 ... J_ln]):
    r_h, r_d, rank J_l, delta_joint = r_h r_d - rank J_l;
    raw decomposition of the whole kernel   dim K_l = sum_j (m - rank J_lj)  +  (sum_j rank J_lj - rank J_l);
    projected decomposition of the deficit   A -> P_C A P_H removes the marginally null parts; projected ranks of the full
        kernel (must equal delta_joint), of the neuron-wise kernel U_l and, for ReLU, of the selective-unit space give the
        selective, other neuron-wise and genuine cross-neuron contributions.
References: rank J_l and the per-unit ranks relative to sigma_1(J_l) (Reference.BLOCK_SIGMA1); r_h, r_d relative to their own
sigma_1; projected stacks with absolute rungs.  rank J_l and sigma_1 come from the authenticated stored factor when one
exists, otherwise the block is assembled and factored here.
"""
import json
import os
import time

import numpy as np
import torch

from flatocean import provenance as P
from flatocean import blocks, cli, factors, linalg, records, runs
from flatocean.linalg import Reference, key

HERE = runs.HERE
OUT = lambda: runs.out_dir("neuronwise")          # noqa: E731  (resolved at call time so a test can redirect the root)
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.6 items 1-2 (2026-09-24); identifier neuronwise-v2"
LAD = linalg.TOL_LADDER


def sigma1_lanczos(row_fn, s, p, device, chunk=1000, tol=1e-8):
    """Largest singular value of a block by Lanczos (scipy svds) on a matrix-free operator."""
    from scipy.sparse.linalg import LinearOperator, svds
    first = row_fn(torch.arange(0, min(chunk, s), device=device)); d = first.shape[0] // min(chunk, s); del first
    def matvec(v):
        v = torch.as_tensor(np.asarray(v).reshape(-1), dtype=torch.float64, device=device)
        return torch.cat([(row_fn(torch.arange(i, min(i + chunk, s), device=device)) @ v).cpu() for i in range(0, s, chunk)]).numpy()
    def rmatvec(u):
        u = torch.as_tensor(np.asarray(u).reshape(-1), dtype=torch.float64, device=device); acc = torch.zeros(p, dtype=torch.float64, device=device)
        for i in range(0, s, chunk):
            rows = row_fn(torch.arange(i, min(i + chunk, s), device=device)); acc += rows.T @ u[i * d: i * d + rows.shape[0]]
        return acc.cpu().numpy()
    return float(svds(LinearOperator((s * d, p), matvec=matvec, rmatvec=rmatvec, dtype=np.float64), k=1, tol=tol, return_singular_vectors=False)[0])


def project_visible(B, PH, PC, m1, n):
    """A_vis = P_C A P_H for every column of B (p x K, index k*n + j <-> A[j, k]); returns the projected stack (p x K)."""
    return torch.einsum("ak,kjc,bj->abc", PH, B.reshape(m1, n, -1), PC).reshape(m1 * n, -1)


def embed_rows(cols, m1, n, device, dtype):
    """Stack of block directions supported on single rows: cols[j] = basis (m1 x k_j) of row j -> p x sum_j k_j."""
    parts = []
    for j, Nj in enumerate(cols):
        if Nj is not None and Nj.shape[1]:
            Bj = torch.zeros(m1, n, Nj.shape[1], dtype=dtype, device=device); Bj[:, j, :] = Nj; parts.append(Bj.reshape(m1 * n, -1))
    return torch.cat(parts, 1) if parts else torch.zeros(m1 * n, 0, dtype=dtype, device=device)


@torch.no_grad()
def kronecker_spectrum(H, D0):
    """Singular values of J = H (x) D0 (rows h_i (x) D0, a constant downstream map): the sorted products sigma_a(H) sigma_b(D0),
    so the declared ladder (relative to sigma_1 = sigma_1(H) sigma_1(D0)) is applied to the same numbers a dense SVD would give."""
    svH = linalg.svdvals(H); svD = linalg.svdvals(D0)
    sv = np.sort(np.outer(svH, svD).ravel())[::-1]
    return np.concatenate([sv, np.zeros(max(0, H.shape[1] * D0.shape[1] - sv.size))]), svH, svD


def constant_downstream(D, rtol=1e-12):
    """True when D_i is the same matrix for every sample (identity activation: D_i = V W_3 W_2 ...)."""
    return bool((D - D[0]).abs().max() <= rtol * D.abs().max())


def atomic_spectrum(name, stage, ep, rd, layer_index):
    """Historical layer spectrum from the atomic decomposition (results/atomic): accepted only when the record's epoch is the
    requested one and the checkpoint file still hashes to the training manifest (checksums.json); the spectrum file's
    SHA-256 is recorded.  Returns (sv padded to p, provenance dict) or None."""
    ap = os.path.join(runs.RESULTS_ROOT, "atomic", f"{name}.json")
    if not os.path.exists(ap):
        return None
    v = json.load(open(ap))["results"].get(stage)
    if not v or v.get("epoch") != ep or "layers" not in v:
        return None
    npz = os.path.join(runs.RESULTS_ROOT, "atomic", f"{name}_epoch{ep:04d}_spectra.npz")
    sums_path = os.path.join(rd, "checksums.json"); ck = f"checkpoints/epoch_{ep:04d}.pt"
    if not (os.path.exists(npz) and os.path.exists(sums_path)):
        return None
    if json.load(open(sums_path)).get(ck) != P.sha256_file(os.path.join(rd, ck)):
        raise P.ProvenanceError(f"{name} epoch {ep}: checkpoint hash differs from checksums.json; the historical atomic spectrum cannot be used")
    with np.load(npz) as z:
        sv = np.asarray(z[f"J{layer_index + 1}"], dtype=float)
    L = v["layers"][layer_index]; p = int(L["p"]); sv = np.concatenate([sv, np.zeros(max(0, p - sv.size))])
    reread = linalg.counts_only(sv, Reference.OWN)
    if {k: int(r) for k, r in L["rank_J"].items()} != reread:                       # the spectrum must reproduce the record at every rung
        raise P.ProvenanceError(f"{name} {stage} epoch {ep}: the atomic record's rank ladder {L['rank_J']} differs from the re-read spectrum {reread}")
    prov = {"atomic_record": os.path.relpath(ap, runs.HERE), "spectrum_file": os.path.relpath(npz, runs.HERE), "spectrum_sha256": P.sha256_file(npz),
            "checkpoint_hash_matches_checksums": True, "recorded_rank_ladder": L["rank_J"], "record_has_identity_manifest": "identity" in v}
    return sv, prov


def block_spectrum(blk, name, ep, run, rd, device, vectors=True):
    """rank ladder, gap, sigma_1 and the kernel basis of the block: from the authenticated stored factor when one exists;
    for a constant downstream map from the Kronecker structure (no assembly); otherwise computed here (assembled QR, SVD
    with the 64-bit fallback).  `vectors=False` computes singular values only (a values-only pass: rank and deficit
    without a kernel basis); the kernel is then None."""
    fs = factors.factor_spectrum(name, ep, blk.name, run, rd) if blk.s == 10000 else None
    if fs is not None:
        sv, p, _, sha, status = fs; lad = linalg.rank_ladder(sv, Reference.OWN)
        kernel = None
        if vectors:
            kernel = factors.kernel_basis(name, ep, blk.name, run, rd, device) if lad["counts"][linalg.DECLARED] < p else torch.zeros(p, 0, dtype=torch.float64, device=device)
        return dict(rank_J=lad["counts"], gap_J=gaps(sv, lad), sigma1_J=float(sv[0]), rank_source="stored factor (identity manifest checked)", factor_sha256=sha, factor_status=status, rank_verified=True), kernel
    if hasattr(blk, "D") and constant_downstream(blk.D):
        sv, svH, svD = kronecker_spectrum(blk.H, blk.D[0]); lad = linalg.rank_ladder(sv, Reference.OWN)
        return dict(rank_J=lad["counts"], gap_J=gaps(sv, lad), sigma1_J=float(sv[0]), rank_source="Kronecker structure (constant downstream map): products of the spectra of H and D",
                    rank_verified=True, kronecker={"sv_H": svH, "sv_D": svD}), None
    R = linalg.assembled_R(blk.rows, blk.s, blk.p, device, chunk=500 if blk.s <= 10000 else 2000)
    sv, Vh, driver = linalg.svd_factor(R, vectors=vectors); del R
    lad = linalg.rank_ladder(sv, Reference.OWN)
    kernel = Vh[lad["counts"][linalg.DECLARED]:].T.to(device) if vectors else None
    del Vh
    return dict(rank_J=lad["counts"], gap_J=gaps(sv, lad), sigma1_J=float(sv[0]), rank_source="computed here" + ("" if vectors else " (values only)"), svd_driver=driver, rank_verified=True), kernel


def gaps(sv, lad):
    """Spectral gap at every rung (the layout of the atomic records)."""
    return {k: linalg.spectral_gap(sv, r) for k, r in lad["counts"].items()}


@torch.no_grad()
def layer_record(blk, spectrum, kernel_B, activation):
    """The record of one MLP hidden layer: dimensionality quantities, per-unit ranks and spectra, the raw and the projected
    decompositions with their consistency flags.  `blk`: MLPLayerBlock; `spectrum`: from block_spectrum."""
    H, D, Z = blk.H, blk.D, blk.Z; m1, n, d, s, p = blk.m1, blk.n, blk.d, blk.s, blk.p; dev = H.device
    rh, sv_h = linalg.numrank_matrix(H); rd, sv_d = linalg.numrank_matrix(D.reshape(s * d, n))
    rec = {"layer": blk.name, "m_aug": m1, "n": n, "p": p, "samples": s, "r_h": rh, "r_d": rd, "r_h_r_d": {t: rh[t] * rd[t] for t in rh},
           "gap_h": linalg.spectral_gap(sv_h, rh[linalg.DECLARED]), "gap_d": linalg.spectral_gap(sv_d, rd[linalg.DECLARED]), **spectrum}
    sigma1 = spectrum["sigma1_J"]
    # per-unit blocks J_lj: spectra, null spaces (whole-block reference) and the hard-mask selective bases
    Hd = H.repeat_interleave(d, 0); unit_sv = np.zeros((n, m1)); unit_V = []; sel_V = []; active = np.zeros(n, dtype=int)
    sel_rank = {key(t): np.zeros(n, dtype=int) for t in LAD}; annihilated = np.zeros(n, dtype=int)
    for j in range(n):
        # QR on the device, SVD of the small triangular factor on the CPU (cusolver float64 SVD is slow and falls back to gesvd)
        Rj = torch.linalg.qr(blk.unit_rows(j, Hd), mode="r").R; _, svj_t, Vhj = torch.linalg.svd(Rj.cpu(), full_matrices=True)
        svj = svj_t.numpy(); unit_sv[j, :svj.size] = svj; unit_V.append(Vhj.T.to(dev))
        act = Z[:, j] > 0; active[j] = int(act.sum())
        if active[j]:
            Ha = H[act]; _, svh_t, Vhh = torch.linalg.svd((torch.linalg.qr(Ha, mode="r").R if act.sum() > m1 else Ha).cpu(), full_matrices=True); svh = svh_t.numpy(); Vhh = Vhh.to(dev)
            for t in LAD:
                sel_rank[key(t)][j] = int((svh > t * svh[0]).sum())
            sel_V.append(Vhh.T[:, sel_rank[linalg.DECLARED][j]:])
            col = D[act][:, :, j].norm(dim=1); annihilated[j] = int((col <= 1e-9 * col.max()).sum()) if col.max() > 0 else int(act.sum())
        else:
            sel_V.append(torch.eye(m1, dtype=H.dtype, device=dev))
    assert float(unit_sv[:, 0].max()) <= sigma1 * (1 + 1e-6), "unit sigma_1 above the block sigma_1"
    rec["sigma1_reference"] = {"used": sigma1, "lower_bound_max_unit": float(unit_sv[:, 0].max()), "reference": Reference.BLOCK_SIGMA1.value}
    dec = {}
    for t in LAD:
        k = key(t); r_units = (unit_sv > t * sigma1).sum(1); r_own = np.array([(unit_sv[j] > t * unit_sv[j, 0]).sum() if unit_sv[j, 0] > 0 else 0 for j in range(n)])
        rJ = rec["rank_J"][k]; cross = int(r_units.sum() - rJ); sel = np.array([m1 - sel_rank[k][j] if active[j] else m1 for j in range(n)]); extra = (m1 - r_units) - sel
        unit_gaps = [unit_sv[j, r - 1] / max(unit_sv[j, r], 1e-300) for j, r in enumerate(r_units) if 0 < r < m1]
        dec[k] = {"dim_K": p - rJ, "neuronwise": int((m1 - r_units).sum()), "cross_neuron": cross,
                  "consistent": bool(cross >= 0 and ((m1 - r_units) >= 0).all()), "rank_verified": bool(spectrum.get("rank_verified", False)),
                  "neuronwise_own_reference": int((m1 - r_own).sum()), "hard_mask_selective_diagnostic": int(sel.sum()), "selective_is_kernel_mechanism": activation == "relu",
                  "extra_neuronwise_beyond_selective": int(extra.sum()) if activation == "relu" else None,
                  "units_violating_selective_containment": int((extra < 0).sum()) if activation == "relu" else None,
                  "units_with_kernel": int((r_units < m1).sum()), "units_full_rank": int((r_units == m1).sum()), "units_selective": int((sel > 0).sum()),
                  "min_unit_gap_at_cutoff": float(min(unit_gaps)) if unit_gaps else None, "units_without_gap_below_10": int(sum(1 for g in unit_gaps if g < 10)),
                  "closest_unit_singular_value_to_cutoff_log10": float(np.abs(np.log10(unit_sv[unit_sv > 0] / (t * sigma1))).min()) if (unit_sv > 0).any() and sigma1 > 0 else None,
                  "note": "neuronwise + cross_neuron = dim_K by construction; 'consistent' is false when a numerical rank violates the exact-rank inequalities"}
    rec["decomposition"] = dec
    rec["units"] = {"active_samples": active.tolist(), "rank_global_ref@1e-09": (unit_sv > 1e-9 * sigma1).sum(1).tolist(),
                    "selective_dim@1e-09": [int(m1 - sel_rank[linalg.DECLARED][j]) if active[j] else m1 for j in range(n)],
                    "annihilated_active_samples": annihilated.tolist(), "sigma1_unit": unit_sv[:, 0].tolist()}
    # projected decomposition of the co-occurrence deficit
    _, _, VhH = torch.linalg.svd(torch.linalg.qr(H, mode="r").R.cpu(), full_matrices=True); QH = VhH.T[:, :rh[linalg.DECLARED]].to(dev); PH = QH @ QH.T
    _, _, VhD = torch.linalg.svd(torch.linalg.qr(D.reshape(s * d, n), mode="r").R.cpu(), full_matrices=True); QC = VhD.T[:, :rd[linalg.DECLARED]].to(dev); PC = QC @ QC.T
    delta = rh[linalg.DECLARED] * rd[linalg.DECLARED] - rec["rank_J"][linalg.DECLARED]
    r9 = (unit_sv > 1e-9 * sigma1).sum(1)
    U = embed_rows([unit_V[j][:, r9[j]:] for j in range(n)], m1, n, dev, H.dtype); pU = linalg.projected_rank(project_visible(U, PH, PC, m1, n)); del U
    proj = {"delta_joint": int(delta), "reference": "absolute rungs on the projected stack of an orthonormal basis", "neuronwise_projected": pU}
    pS = None
    if activation == "relu":
        Msel = embed_rows(sel_V, m1, n, dev, H.dtype); pS = linalg.projected_rank(project_visible(Msel, PH, PC, m1, n)); del Msel
        proj["selective_projected"] = pS
    else:
        proj["selective_projected"] = None; proj["selective_note"] = "hard-mask candidate is not a subspace of the kernel for this activation; no selective contribution reported"
    if kernel_B is not None:
        pK = linalg.projected_rank(project_visible(kernel_B, PH, PC, m1, n)); proj["kernel_projected"] = pK; proj["kernel_projected_equals_delta"] = bool(pK["counts"][linalg.DECLARED] == delta)
    elif spectrum.get("kronecker") is not None:
        # J = H (x) D: Pi(v_a (x) w_b) vanishes unless a < r_h and b < r_d, so dim Pi(K) counts the visible pairs whose product is below the rung
        # same object as the dense path: kernel (sigma_a sigma_b <= 1e-9 sigma_1) and marginal spaces (r_h, r_d) fixed at the declared
        # rung; the projected vectors v_a (x) w_b are orthonormal, so the absolute ladder on the projected stack gives one count at every rung
        svH, svD = spectrum["kronecker"]["sv_H"], spectrum["kronecker"]["sv_D"]; t9 = float(linalg.DECLARED)
        n9 = int((np.outer(svH[:rh[linalg.DECLARED]], svD[:rd[linalg.DECLARED]]) <= t9 * svH[0] * svD[0]).sum())
        proj["kernel_projected"] = {"counts": {key(t): n9 for t in LAD}, "reference": "Kronecker structure, declared spaces: pairs (a < r_h, b < r_d) with sigma_a(H) sigma_b(D) <= 1e-9 sigma_1(H) sigma_1(D); projected vectors orthonormal"}
        proj["kernel_projected_equals_delta"] = bool(n9 == delta)
    else:
        proj["kernel_projected"] = None; proj["kernel_projected_equals_delta"] = None; proj["kernel_note"] = "kernel basis not computed (values-only pass): dim Pi(K) not checked"
    rU = pU["counts"][linalg.DECLARED]; rS = pS["counts"][linalg.DECLARED] if pS else None
    proj["contributions@1e-09"] = {"selective": rS, "other_neuronwise": (rU - rS) if rS is not None else None, "neuronwise_total": rU, "genuine_cross_neuron": int(delta - rU),
                                   "consistent": bool(delta - rU >= 0 and (rS is None or rU - rS >= 0))}
    # the quantities of the plan: delta = dim Pi(K) = r_h r_d - rank J, delta_neuron = dim Pi(U), delta_cross = delta - delta_neuron
    proj.update({"delta": int(delta), "delta_neuron": int(rU), "delta_cross": int(delta - rU),
                 "identities": {"delta_equals_rhrd_minus_rank": True, "dim_Pi_K_equals_delta": proj["kernel_projected_equals_delta"],
                                "delta_neuron_plus_delta_cross_equals_delta": True, "delta_neuron_within_bounds": bool(0 <= rU <= delta),
                                "tolerance": f"declared rung {linalg.DECLARED} (own sigma_1 for ranks, absolute rungs on projected orthonormal stacks)"}})
    rec["deficit_decomposition"] = proj
    rec["raw_decomposition_note"] = "'decomposition' is the raw kernel split (diagnostic): its cross-neuron term can lie entirely in the downstream-null space and vanish after projection; the deficit decomposition above is the primary quantity"
    return rec, unit_sv


def spectra_path(out, name, samples, ep):
    return os.path.join(out, f"{name}_s{samples}_epoch{ep:04d}_unit_spectra.npz")


def save_spectra(path, new):
    """Merge `new` arrays into the NPZ at `path` (existing arrays of other layers are kept) and write atomically."""
    old = {}
    if os.path.exists(path):
        with np.load(path) as z:
            old = {k: z[k] for k in z.files}
    old.update(new); tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **old); os.replace(tmp, path)


def layer_complete(rec, l, path, name, ep, run, rd, values_only=False, historical=False, stage=None):
    """A layer measurement counts as complete when its record exists, its spectra are on disk, the stored factor it
    consumed (if any) is still the file that exists now, and (for a full pass) a record from a values-only pass with a
    nonzero deficit is not accepted, since it lacks the kernel-projected check."""
    L = (rec.get("layers") or {}).get(str(l))
    if L is None or not os.path.exists(path):
        return False
    if not values_only and L.get("pass") in ("values-only", "historical") and L["deficit_decomposition"]["delta"] != 0:
        return False
    if L.get("pass") == "historical":
        if not historical:                                   # a fresh computation was requested: it replaces the provisional record
            return False
        prov = L.get("historical_provenance") or {}
        try:
            hs = atomic_spectrum(name, stage, ep, rd, l - 1)
        except P.ProvenanceError:
            return False
        if hs is None or hs[1]["spectrum_sha256"] != prov.get("spectrum_sha256") or hs[1]["recorded_rank_ladder"] != prov.get("recorded_rank_ladder"):
            return False
        if L["deficit_decomposition"]["delta"] != 0:
            return False
    with np.load(path) as z:
        if f"unit_sv_layer{l}" not in z.files:
            return False
    if L.get("factor_sha256"):
        # the consumed factor must still be the same bytes AND authenticate against the current run and configuration
        fp = factors.factor_path(name, ep, str(l))
        if not os.path.exists(fp) or P.sha256_file(fp) != L["factor_sha256"]:
            return False
        try:
            factors.factor_status(fp, run, rd, str(l), ep=ep)
        except P.ProvenanceError:
            return False
    return True


def run(a):
    cli.configure(a); out = OUT()
    for seed in a.seeds:
        rd = runs.run_dir(a.dataset, a.model, seed, a.tag); name = runs.run_name(a.dataset, a.model, seed, a.tag)
        store = records.ResultStore(os.path.join(out, f"{name}_s{a.samples}.json"), {"protocol": PROTOCOL})
        for stage, ep, ep_req, label in cli.stages_of(a, seed):
            sets = ("analysis",) if a.samples <= 10000 else ("train",)
            run_ = runs.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device, sets=sets)
            X, y, idx = run_["data"][sets[0]]; X = X[: a.samples]; idx = idx[: a.samples]
            if sets[0] == "train":
                run_["data"]["analysis"] = (X, y[: a.samples], idx)                      # the identity is taken over the ordered training samples
            # the stage identity does not name the layers: every layer is a measurement of its own with its own inputs
            ident = records.identity(run_, rd, ep, {"protocol": PROTOCOL, "samples": int(X.shape[0]), "set": sets[0], "requested_epoch": ep_req,
                                                    "stage_label": label, "ladder": list(LAD)})
            rec = store.stage(stage, ident); npz = spectra_path(out, name, a.samples, ep)
            todo = [l for l in a.layers if a.overwrite or not layer_complete(rec, l, npz, name, ep, run_, rd, a.values_only, a.rank_from_atomic, stage)]
            if not todo:
                print(f"== {name} {stage}: layers {a.layers} complete", flush=True); continue
            rec.update({"epoch": ep, "requested_epoch": ep_req, "epoch_offset": ep - ep_req, "stage_label": label}); rec.setdefault("layers", {})
            st = blocks.mlp_structure(run_["net"], X)
            for l in todo:
                t0 = time.time(); blk = blocks.MLPLayerBlock(st, l - 1, X.shape[0])
                if a.rank_from_atomic:
                    # historical-only mode: audited atomic evidence or nothing; never a fresh factorization, never a nonzero deficit
                    hs = atomic_spectrum(name, stage, ep, rd, l - 1)
                    if hs is None:
                        print(f"== {name} {stage} layer {l}: no audited atomic evidence for epoch {ep}; skipped (needs a fresh computation)", flush=True); continue
                    sv, prov = hs; lad = linalg.rank_ladder(sv, Reference.OWN)
                    spectrum, kernel_B = dict(rank_J=lad["counts"], gap_J=gaps(sv, lad), sigma1_J=float(sv[0]), rank_source="historical atomic spectrum (checkpoint hash verified, spectrum re-read)",
                                              rank_verified=False, historical_provenance=prov), None
                else:
                    spectrum, kernel_B = block_spectrum(blk, name, ep, run_, rd, a.device, vectors=not a.values_only)
                L, unit_sv = layer_record(blk, spectrum, kernel_B, st["activation"]); L["seconds"] = time.time() - t0
                if a.rank_from_atomic:
                    if L["deficit_decomposition"]["delta"] != 0:                   # zero only: a negative value is a numerical inconsistency
                        print(f"== {name} {stage} layer {l}: historical deficit {L['deficit_decomposition']['delta']} != 0; not recorded (needs a fresh factorization)", flush=True); continue
                    L["pass"] = "historical"; L["provisional"] = True; L["provisional_note"] = "zero deficit taken from the audited atomic spectrum; replaced by any fresh computation"
                else:
                    L["pass"] = "values-only" if a.values_only else "full"
                if a.values_only and L["deficit_decomposition"]["delta"] > 0 and L["deficit_decomposition"]["kernel_projected"] is None:
                    print(f"   {name} {stage} layer {l}: delta {L['deficit_decomposition']['delta']} > 0; rerun without --values-only for the kernel-projected check", flush=True)
                save_spectra(npz, {f"unit_sv_layer{l}": unit_sv})                       # artifact first, then the record that points to it
                L.pop("kronecker", None); rec["layers"][str(l)] = L; store.save()
                c = L["decomposition"][linalg.DECLARED]; pd = L["deficit_decomposition"]
                print(f"== {name} {stage} (ep {ep}) layer {l}: r_h {L['r_h'][linalg.DECLARED]} r_d {L['r_d'][linalg.DECLARED]} rank J {L['rank_J'][linalg.DECLARED]} ({L['rank_source'][:22]}) "
                      f"delta {pd['delta_joint']} | dim K {c['dim_K']} = neuronwise {c['neuronwise']} + cross {c['cross_neuron']} (consistent {c['consistent']}) | "
                      f"deficit {pd['contributions@1e-09']} kernel-projected {(pd.get('kernel_projected') or {}).get('counts')} ({L['seconds']:.0f}s)", flush=True)


def main(argv=None):
    """Command line of the analysis (python -m flatocean neuronwise ...; formerly exp_neuronwise.py, removed 2026-09-25).

        --tag "" --seeds 0 1 2 --layers 2 3
        --tag identity --seeds 0 1 2 --layers 2 3 --match-stages-of "" --stages high_accuracy mse_interpolation epoch_100 epoch_250 late_endpoint
        --tag "" --seeds 0 --layers 1 --compute-layer1      # layer-1 factorization computed here (heavy)
        test                                                # the pytest file of this analysis
    """
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["test"]:
        import pytest; return pytest.main(["-q", os.path.join(HERE, "tests", "test_neuronwise.py")])
    ap = cli.parser("neuron-wise / cross-neuron decomposition")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--compute-layer1", action="store_true", help="kept for compatibility: every layer without a stored factor is factored here")
    ap.add_argument("--rank-from-atomic", action="store_true", help="with --values-only: take the layer spectrum from the audited atomic record (checkpoint hash verified) instead of assembling the block; zero-deficit layers are recorded this way")
    ap.add_argument("--values-only", action="store_true", help="singular values only for layers without a stored factor: rank, deficit and the neuron-wise terms, no kernel basis (cheap sweep; a nonzero deficit is flagged for a full pass)")
    run(ap.parse_args(argv))
