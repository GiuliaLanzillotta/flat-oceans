"""
Rays along certified multilayer branches and their controls (plan amendment 2026-09-17, item 5).

Direction sets, 32 directions each, from the same ambient Gaussian probes (seeds 100-131):
  branch:<construction>   the probe projected onto the certified branch M = (+)_l M_l (random linear combination);
  naive_atomic_sum        projected onto the direct sum of the atomic kernels (not a branch: negative control);
  tangent_flat            projected onto ker J_X and then off the largest certified branch: flat to first order,
                          in no certified branch;
  generic                 the probe itself.
Every direction is scaled as a whole to the weighted relative norm ||w||_rel = sqrt(sum_l ||w_l||^2/||theta_l||^2)
= alpha, because rescaling the blocks of a multilayer branch direction independently can leave the branch.
Measured per direction and alpha: true, frozen and transition output change, gates crossed per layer (the same
ray routine as flatocean.analyses.transitions).

Phase `tangent` computes the full-network Jacobian J_X (rows D_i^T e_a (x) [h;1] per block, row-major [W b]
coordinates), its rank on the 10k analysis set and on the 20k union with the comparison set (parameter counting
forces no kernel here: p < 10 s), and an orthonormal basis of ker J_X on the 10k set.

    python -m flatocean branch-rays tangent --tag w64 --seeds 0 --stages high_accuracy
    python -m flatocean branch-rays rays    --tag w64 --seeds 0 --stages high_accuracy
"""
# Ported unchanged from experiments/exp_branch_rays.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean branch-rays).
import argparse
import json
import os
import time

import numpy as np
from flatocean import records, runs
import torch

from flatocean.analyses import branch as B
from flatocean.analyses import transitions as TR
from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs

OUT = B.OUT
ALPHAS = TR.ALPHAS; RADII = TR.RADII_V3
# Scaling protocols (plan Section 0.4 items 5-6).  Layerwise direct-sum branches, the naive atomic sum and the
# feature-null branch are scaled block by block ('block-balanced': every nonzero block moved by the fraction alpha,
# membership preserved); ker J_X is coupled, so tangent-flat directions are scaled as a whole by the weighted relative
# norm ('weighted', labelled); generic directions follow the block-balanced protocol of the generic transition runner.
# Every set is also measured on the absolute grid (whole direction of Euclidean norm r).
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.4 items 5-6 (2026-09-22); identifier branch-rays-v3"


def load_basis(path, build_rec, stage_digest):
    """Branch basis with provenance: the file's bytes must be the ones the build record hashed, and the digest stored
    in the file must be the analysis identity of the build stage.  Files written before 2026-09-18 carry no digest
    and are refused (rebuild the stage)."""
    obj = torch.load(path, weights_only=False)
    if not (isinstance(obj, dict) and "bases" in obj):
        raise P.ProvenanceError(f"{path}: basis file without an identity (written before build version 2); rebuild the stage")
    if obj["identity_digest"] != stage_digest or P.sha256_file(path) != build_rec.get("basis_sha256"):
        raise P.ProvenanceError(f"{path}: basis does not belong to the recorded build (identity digest or file hash differs)")
    return obj["bases"]


def jacobian_rows(nt, idx):
    """Rows of J_X for samples idx in the block coordinates of exp_multilayer_branch (row-major vec of [W b])."""
    cols = []
    for l in range(nt.L):
        cols.append(torch.einsum("san,sm->sanm", nt.D[l][idx], nt.H[l][idx]).reshape(len(idx) * nt.d, -1))
    eye = torch.eye(nt.d, dtype=nt.H[0].dtype)
    cols.append(torch.einsum("ab,sm->sabm", eye, nt.H[nt.L][idx]).reshape(len(idx) * nt.d, -1))
    return torch.cat(cols, 1)


def split_blocks(nt, v):
    """Block-coordinate vector -> dict of block matrices."""
    out = {}; i = 0
    for blk in nt.blocks:
        n = nt.p(blk); out[blk] = v[i:i + n].reshape(nt.shape[blk]); i += n
    return out


def to_block_coords(nt, w_flat):
    """Flat parameter vector (order of net.parameters()) -> block-coordinate vector (row-major [W b] per block)."""
    parts = []; i = 0
    for blk in nt.blocks:
        n, m1 = nt.shape[blk]; W = w_flat[i:i + n * (m1 - 1)].reshape(n, m1 - 1); i += n * (m1 - 1)
        b = w_flat[i:i + n].reshape(n, 1); i += n
        parts.append(torch.cat([W, b], 1).reshape(-1))
    return torch.cat(parts)


def tangent(a):
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}{'_' + a.indices_tag if getattr(a, 'indices_tag', None) else ''}_seed{seed}"
        out_path = os.path.join(OUT, f"{name}_tangent.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue
            sets = ("analysis",) if a.sample_set == "10000" else ("analysis", "comparison")
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=sets)
            cfg = {"protocol": PROTOCOL, "phase": "tangent", "rtol": B.RTOL}
            if a.sample_set != "10000":
                cfg["comparison_index_sha256"] = P.sha256_index(run["data"]["comparison"][2])
            if a.sample_set != "both":
                cfg["sample_set"] = a.sample_set
            ident = P.analysis_identity(run, rd, ep, cfg)
            if stage in res and not a.overwrite:
                if res[stage].get("identity", {}).get("digest") not in (None, ident["digest"]):
                    raise P.ProvenanceError(f"{name} {stage}: stored tangent record has another identity; use --overwrite")
                if "identity" not in res[stage]:
                    print(f"== {name} {stage}: stored tangent record predates identities (UNVERIFIED); use --overwrite to recompute", flush=True)
                continue
            Xa = run["data"]["analysis"][0]; rec = {"epoch": ep, "identity": ident}
            if a.sample_set == "both":
                pairs = (("20000", torch.cat([Xa, run["data"]["comparison"][0]], 0)), ("10000", Xa))
            elif a.sample_set == "10000":
                pairs = (("10000", Xa),)
            else:
                pairs = (("20000", torch.cat([Xa, run["data"]["comparison"][0]], 0)),)
            for label, X in pairs:
                t0 = time.time(); nt = B.Net(run["net"], X); p = sum(nt.p(b) for b in nt.blocks)
                R = O.full_R_cpu(lambda idx: jacobian_rows(nt, idx.cpu()), X.shape[0], p, "cpu", chunk=500)
                if label == "10000":
                    _, sv, Vh = torch.linalg.svd(R, full_matrices=True); sv = sv.numpy()
                else:
                    sv = torch.linalg.svdvals(R).numpy()
                r = int((sv > B.RTOL * sv[0]).sum())
                rec[label] = {"p": p, "rows": int(X.shape[0] * nt.d), "counting_nullity": max(0, p - X.shape[0] * nt.d), "rank": r, "dim_ker_JX": p - r,
                              "gap": float(sv[r - 1] / max(sv[r], 1e-300)) if r < sv.size else None,
                              "rank_ladder": {str(t): int((sv > t * sv[0]).sum()) for t in (1e-6, 1e-9, 1e-12)}, "seconds": time.time() - t0}
                print(f"== {name} {stage} s={label}: p {p} rank {r} dim ker J_X {p - r} gap {rec[label]['gap']} ({rec[label]['seconds']:.0f}s)", flush=True)
                if label == "10000":
                    kp = os.path.join(OUT, f"{name}_epoch{ep:04d}_kerJX.pt")
                    torch.save({"identity_digest": ident["digest"], "basis": Vh[r:].T.contiguous()}, kp); rec["kernel_basis_sha256"] = P.sha256_file(kp)
                    np.save(os.path.join(OUT, f"{name}_epoch{ep:04d}_JX_singular_values.npy"), sv)
            res[stage] = rec
            records.dump_json(out_path, res)


def adopt(a):
    """Bind a tangent-kernel basis written before identities existed (a bare tensor) to the current analysis without
    repeating the two-hour factorization: audit it against the current checkpoint and samples (orthonormality; the
    exposure ||J_X v|| / sqrt(s) of every basis column against the rank scale eta = RTOL sigma_1 / sqrt(s), pass
    within 2 eta; dimension equal to the recorded one), then rewrite it with the identity digest and store the audit
    and the file hash in the tangent record.  A failed audit leaves everything untouched."""
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}{'_' + a.indices_tag if getattr(a, 'indices_tag', None) else ''}_seed{seed}"
        out_path = os.path.join(OUT, f"{name}_tangent.json"); res = json.load(open(out_path))
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]; kp = os.path.join(OUT, f"{name}_epoch{ep:04d}_kerJX.pt")
            obj = torch.load(kp, weights_only=False)
            if isinstance(obj, dict):
                print(f"== {name} {stage}: basis already carries an identity", flush=True); continue
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=("analysis", "comparison"))
            ident = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "phase": "tangent", "rtol": B.RTOL,
                                                      "comparison_index_sha256": P.sha256_index(run["data"]["comparison"][2])})
            X = run["data"]["analysis"][0]; nt = B.Net(run["net"], X); Kx = obj; s_ = X.shape[0]
            sv = np.load(os.path.join(OUT, f"{name}_epoch{ep:04d}_JX_singular_values.npy")); eta = B.RTOL * sv[0] / np.sqrt(s_)
            acc = torch.zeros(Kx.shape[1], dtype=torch.float64); top = 0.0
            g = torch.Generator().manual_seed(0); probe = torch.randn(Kx.shape[0], 8, generator=g, dtype=torch.float64); probe /= probe.norm(dim=0)
            acc_p = torch.zeros(8, dtype=torch.float64)
            for i in range(0, s_, 250):
                rows = jacobian_rows(nt, torch.arange(i, min(i + 250, s_)))
                acc += (rows @ Kx).pow(2).sum(0); acc_p += (rows @ probe).pow(2).sum(0)
            expo = (acc / s_).sqrt(); generic = float((acc_p / s_).sqrt().median())
            audit = {"orthonormality_max_abs": float((Kx.T @ Kx - torch.eye(Kx.shape[1], dtype=torch.float64)).abs().max()),
                     "kernel_columns_max_over_eta": float(expo.max() / eta), "eta": float(eta), "generic_exposure": generic,
                     "dim": int(Kx.shape[1]), "dim_recorded": res[stage]["10000"]["dim_ker_JX"]}
            audit["passed"] = bool(audit["orthonormality_max_abs"] < 1e-10 and audit["kernel_columns_max_over_eta"] <= 2.0 and audit["dim"] == audit["dim_recorded"])
            print(f"== {name} {stage}: adoption audit {audit}", flush=True)
            if not audit["passed"]:
                raise P.ProvenanceError(f"{kp}: adoption audit failed; rerun the tangent phase with --overwrite")
            torch.save({"identity_digest": ident["digest"], "basis": Kx, "adopted": "2026-09-18, audited against the current checkpoint and samples"}, kp)
            res[stage].update({"identity": ident, "kernel_basis_sha256": P.sha256_file(kp), "adoption_audit": audit})
            records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def rays(a):
    torch.set_num_threads(a.threads)
    # ADDITIVE: --radii replaces the absolute grid (default None => the module RADII, byte-identical); --out-suffix
    # writes a separate record; --exposure-indices evaluates the fitted bases on held-out rows.
    radii = tuple(a.radii) if getattr(a, "radii", None) else RADII
    _suffix = getattr(a, "out_suffix", None) or ""
    _expo = getattr(a, "exposure_idx", None)
    # ADDITIVE: --abs-only computes ONLY the absolute grid (default False => both grids, byte-identical).  The tuple of
    # (grid_name, grid, scaling_mode) pairs is built once; the relative/alpha grid is dropped when abs-only is set.
    _abs_only = getattr(a, "abs_only", False)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}{'_' + a.indices_tag if getattr(a, 'indices_tag', None) else ''}_seed{seed}"
        built = json.load(open(os.path.join(OUT, f"{name}.json")))["results"]
        out_path = os.path.join(OUT, f"{name}{_suffix}_rays.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "alphas": list(ALPHAS), "results": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue
            _cidx = getattr(a, "custom_idx", None)
            _eval_idx = _expo if _expo is not None else _cidx
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", indices=_eval_idx)
            bcfg = built[stage]["identity"]["config"]
            expo_sha = None
            if _expo is None:
                # the build's identity, recomputed from the inputs loaded now (checkpoint bytes, preprocessing, ordered
                # samples, sample count and build configuration), must equal the stored one before any basis is used
                cur_build = P.analysis_identity(run, rd, ep, bcfg)
                if cur_build["digest"] != built[stage]["identity"]["digest"]:
                    raise P.ProvenanceError(f"{name} {stage}: the build record does not match the inputs loaded now "
                                            f"(checkpoint, preprocessing, sample order or configuration); rebuild the stage")
                if _cidx is not None:                    # P1: probe on the EXACT fitted subset the branch was built on
                    if P.sha256_index(_cidx) != bcfg.get("custom_index_sha256"):
                        raise P.ProvenanceError(f"{name} {stage}: --indices do not match the build's custom_index_sha256")
                    X = run["data"]["indices"][0]
                else:
                    X = run["data"]["analysis"][0][: bcfg["samples"]]
            else:
                # EXPOSURE (additive): evaluate the fitted-set branch directions on a DIFFERENT (held-out) index set.
                # The bases are the fitted ones (load_basis still authenticates them against the build record).  Only the
                # evaluation INDICES are allowed to differ: the checkpoint and preprocessing are STILL authenticated
                # against the build record (same network, same input pipeline), and the full build-identity / index-match
                # assertions are the only ones relaxed.  The exposure index sha is recorded alongside the authenticated
                # checkpoint/preprocessing shas, so the record never shares an identity with the fitted probe.
                built_id = built[stage]["identity"]
                cur_ck = P.sha256_file(os.path.join(rd, "checkpoints", f"epoch_{ep:04d}.pt"))
                cur_pre = P.sha256_file(os.path.join(run["dir"], "preprocessing.pt"))
                if cur_ck != built_id.get("checkpoint_sha256") or cur_pre != built_id.get("preprocessing_sha256"):
                    raise P.ProvenanceError(f"{name} {stage}: exposure probe -- checkpoint or preprocessing does not match "
                                            f"the build record (checkpoint {cur_ck[:12]} vs {str(built_id.get('checkpoint_sha256'))[:12]}, "
                                            f"preprocessing {cur_pre[:12]} vs {str(built_id.get('preprocessing_sha256'))[:12]})")
                X = run["data"]["indices"][0]; expo_sha = P.sha256_index(_expo)
                print(f"== {name} {stage}: EXPOSURE probe on {X.shape[0]} held-out rows (sha {expo_sha[:12]}); checkpoint+preprocessing "
                      f"AUTHENTICATED ({cur_ck[:12]}/{cur_pre[:12]}), only the index-match assertion relaxed", flush=True)
            net = run["net"]; nt = B.Net(net, X)
            theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X)
            with torch.no_grad():
                F0 = net(X)
            _, _, _, bins, units = TR.base_diagnostics(net, X, masks); unit_state = [u[1] for u in units]
            cons = built[stage]["constructions"]; stage_digest = built[stage]["identity"]["digest"]
            ident_ck = built[stage]["identity"]["checkpoint_sha256"]; tangent_sha = None
            # only constructions whose validation passed count as branches; the direct sum of the atomic kernels is
            # labelled by what it is and carries its own validation outcome
            certified = {c: v["total"] for c, v in cons.items() if v["certified_branch"] and c not in ("feature_null", "naive_atomic_sum")}
            probe_uncert = getattr(a, "probe_uncertified", False)
            if not certified and not probe_uncert:
                raise RuntimeError(f"{name} {stage}: no validated construction")
            best = max(certified, key=certified.get) if certified else None
            distinct = sorted({json.dumps(cons[c]["dims"], sort_keys=True) for c in certified})
            if a.only is not None:
                # --probe-uncertified: probe the named candidates even when the registered certificate failed, clearly
                # labelled; the ray still records true, frozen and transition so the FULL drift decomposition is kept.
                avail = [c for c in a.only if c in cons]
                chosen = avail if probe_uncert else [c for c in avail if cons[c]["certified_branch"]]
                if not chosen:
                    raise RuntimeError(f"{name} {stage}: none of --only {a.only} is "
                                       f"{'a known' if probe_uncert else 'a certified'} construction (have {sorted(cons)})")
            else:
                want = ("bottom_up", "top_down", best, "feature_null")
                chosen = [c for c in want if c in cons and (probe_uncert or cons[c]["certified_branch"])]
                chosen = list(dict.fromkeys(chosen)) + ["naive_atomic_sum"]
            bases = {c: load_basis(os.path.join(OUT, f"{name}_epoch{ep:04d}_{c}.pt"), cons[c], stage_digest) for c in chosen}
            full = lambda M: torch.block_diag(*[M[b] for b in nt.blocks]) if all(M[b].shape[1] for b in nt.blocks) else \
                torch.cat([torch.cat([torch.zeros(sum(nt.p(x) for x in nt.blocks[:j]), M[b].shape[1], dtype=torch.float64), M[b],
                                      torch.zeros(sum(nt.p(x) for x in nt.blocks[j + 1:]), M[b].shape[1], dtype=torch.float64)], 0) for j, b in enumerate(nt.blocks)], 1)
            sets = {f"branch:{c}" if c != "naive_atomic_sum" else c: full(bases[c]) for c in chosen}
            kpath = os.path.join(OUT, f"{name}_epoch{ep:04d}_kerJX.pt")
            if _expo is None and os.path.exists(kpath) and not a.no_tangent and f"branch:{best}" in sets:
                kobj = torch.load(kpath, weights_only=False)
                if not isinstance(kobj, dict):
                    raise P.ProvenanceError(f"{kpath}: tangent basis without an identity; rerun the tangent phase with --overwrite")
                trec = json.load(open(os.path.join(OUT, f"{name}_tangent.json")))[stage]
                if kobj["identity_digest"] != trec.get("identity", {}).get("digest") or P.sha256_file(kpath) != trec.get("kernel_basis_sha256"):
                    raise P.ProvenanceError(f"{kpath}: tangent basis does not match its record")
                run_c = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=("analysis", "comparison"))
                cur_t = P.analysis_identity(run_c, rd, ep, trec["identity"]["config"]); del run_c
                if cur_t["digest"] != trec["identity"]["digest"]:
                    raise P.ProvenanceError(f"{kpath}: the tangent record does not match the inputs loaded now")
                tangent_sha = trec["kernel_basis_sha256"]
                Kx = kobj["basis"]; Bb = sets[f"branch:{best}"]
                sets["tangent_flat"] = ("tangent", Kx, Bb)
            sets["generic"] = None
            seeds_used = list(TR.DIRECTION_SEEDS[: a.nprobes])
            if a.nprobes < 128 and not a.exploratory:
                raise SystemExit("structured rays use 128 matched directions (--nprobes 128); pass --exploratory for smaller counts")
            # the ray identity binds the probes, the build identity, and the bytes of every basis used (a rebuild under
            # the same configuration changes the basis hashes and therefore invalidates stored rays)
            ident = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "alphas": list(ALPHAS), "radii": list(radii), "scaling": "block-balanced (layerwise sets, generic), weighted (tangent)", "direction_seeds": seeds_used,
                                                      "build_identity_digest": stage_digest, "samples": bcfg["samples"],
                                                      "basis_sha256": {c: cons[c]["basis_sha256"] for c in chosen},
                                                      "tangent_basis_sha256": tangent_sha if "tangent_flat" in sets else None,
                                                      **({"exposure": True, "exposure_index_sha256": expo_sha, "exposure_size": int(X.shape[0]),
                                                          "exposure_authenticated": "checkpoint+preprocessing (against the fitted build); only the index-match assertion relaxed",
                                                          "authenticated_checkpoint_sha256": built[stage]["identity"].get("checkpoint_sha256"),
                                                          "authenticated_preprocessing_sha256": built[stage]["identity"].get("preprocessing_sha256")} if _expo is not None else {})})
            srec = res["results"].get(stage)
            if srec and srec.get("identity", {}).get("digest") != ident["digest"]:
                if not a.overwrite:
                    raise P.ProvenanceError(f"{name} {stage}: stored rays have another identity (probes, build, checkpoint); use --overwrite")
                srec = None
            if srec is None or a.overwrite:
                srec = {"epoch": ep, "identity": ident, "best_construction": best, "sets": {},
                        "n_distinct_certified_dimension_vectors": len(distinct),
                        "certified_by_construction": {c: bool(cons[c]["certified_branch"]) for c in chosen},   # label uncertified probes
                        "probe_uncertified": bool(probe_uncert),
                        "radii": list(radii),
                        **({"exposure": True, "exposure_index_sha256": expo_sha, "exposure_size": int(X.shape[0]),
                            "note": "EXPOSURE: fitted-set branch directions evaluated on held-out rows; frozen/transition drift measured on inputs the branch was NOT built on"} if _expo is not None else {}),
                        "selection": "bottom_up, top_down, the largest validated construction, feature_null; other random orders are not evaluated"}
            res["results"][stage] = srec
            for sname, basis in sets.items():
                _grids_meta = ((("relative", ALPHAS),) if not _abs_only else ()) + (("absolute", radii),)
                complete = sname in srec["sets"] and all(len(srec["sets"][sname].get(grid_name, {}).get(str(g), [])) >= len(seeds_used) for grid_name, grid in _grids_meta for g in grid)
                if complete and isinstance(basis, tuple) and "complement_rank" not in srec["sets"][sname]:
                    complete = False                                             # record from before the complement diagnostics: recompute the set
                if complete:
                    continue
                t0 = time.time(); out = {**({"relative": {str(al): [] for al in ALPHAS}} if not _abs_only else {}), "absolute": {str(r_): [] for r_ in radii}}
                proto = "weighted" if isinstance(basis, tuple) else "block-balanced"; out["relative_protocol"] = proto
                if basis is not None and not isinstance(basis, tuple):
                    out["dim"] = int(basis.shape[1])
                    if basis.shape[1] == 0:
                        srec["sets"][sname] = {"dim": 0, "note": "empty"}; continue
                out["validation_passed"] = cons[sname.split(":")[-1]]["certified_branch"] if sname.split(":")[-1] in cons else None
                for ds in seeds_used:
                    g = torch.Generator().manual_seed(ds); probe = torch.randn(theta0.numel(), generator=g, dtype=torch.float64)
                    v = to_block_coords(nt, probe)
                    if isinstance(basis, tuple):
                        _, Kx, Bb = basis; v = Kx @ (Kx.T @ v); v = v - Bb @ (Bb.T @ v); v = Kx @ (Kx.T @ v); v = v - Bb @ (Bb.T @ v)
                        if "complement_rank" not in out:
                            svp = torch.linalg.svdvals(Kx - Bb @ (Bb.T @ Kx)); out["complement_rank"] = int((svp > 1e-9 * max(float(svp[0]), 1e-300)).sum())
                            out["full_kernel_dim"] = int(Kx.shape[1]); out["branch_outside_kernel_rel"] = float((Bb - Kx @ (Kx.T @ Bb)).norm() / Bb.norm())
                        out["dim"] = out["complement_rank"]
                    elif basis is not None:
                        v = basis @ (basis.T @ v)
                    w0 = B.embed(net, nt, split_blocks(nt, v)); sl = FL.block_slices(net)
                    for grid_name, grid, mode in ((("relative", ALPHAS, proto),) if not _abs_only else ()) + (("absolute", radii, "absolute"),):
                        w, info = FL.scale_direction(w0, theta0, sl, mode)
                        for g in grid:
                            m, _, _, _ = TR.ray(net, theta0, masks, X, w, g, F0, bins, unit_state)
                            m.update({"direction_seed": ds, "euclidean_norm": float(g * w.norm()), "block_fractions": {k: g * v_ for k, v_ in info["unit_step_block_fractions"].items()},
                                      "cancellation_ratio": m["propagated_sum_of_norms"] / m["transition_propagated"] if m["transition_propagated"] > 0 else None,
                                      "base_output_norm_X": FL.x_norm(F0)})
                            out[grid_name][str(g)].append(m)
                out["seconds"] = time.time() - t0; srec["sets"][sname] = out
                med = lambda gn, g, k: float(np.median([q[k] for q in out[gn][str(g)]]))
                _rshow = 1.0 if 1.0 in radii else radii[-1]      # custom --radii may omit 1.0: report the largest available radius
                _rel_msg = "" if _abs_only else (f"alpha=0.01 true {med('relative', 0.01, 'true'):.2e} frozen {med('relative', 0.01, 'frozen'):.2e}; "
                                                 f"alpha=0.1 true {med('relative', 0.1, 'true'):.2e}; ")
                print(f"== {name} {stage} {sname} (dim {out.get('dim')}, {proto}): {_rel_msg}"
                      f"r={_rshow:g} true {med('absolute', _rshow, 'true'):.2e} frozen {med('absolute', _rshow, 'frozen'):.2e}", flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tangent", "rays", "adopt"])
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="w64")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--stages", nargs="+", default=["high_accuracy"])
    ap.add_argument("--nprobes", type=int, default=128); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", nargs="+", default=None, help="rays: restrict the compared constructions (e.g. --only bottom_up)")
    ap.add_argument("--no-tangent", action="store_true", help="rays: skip the optional tangent_flat set even if a kerJX basis exists")
    ap.add_argument("--probe-uncertified", action="store_true", help="rays: also probe constructions whose registered certificate FAILED, clearly labelled (certified_by_construction); the ray still records true, frozen and transition so the full drift decomposition ΔF = ΔF_frozen + (F_true - F_frozen) is kept")
    ap.add_argument("--sample-set", choices=["both", "10000", "20000"], default="both",
                    help="tangent: which J_X sample set(s) to factorize; 10000 = the bounded motivation (analysis set only, no 20k union), recorded in provenance")
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--indices", default=None, help="P1 (additive): probe on a CUSTOM fitted subset, a .npy of training-set row indices matching the branch build's custom_index_sha256 (default None = the standard analysis set, unchanged).")
    ap.add_argument("--indices-tag", default=None, help="P1 (additive): the same name suffix passed to the branch build, so rays reads the matching record/bases.")
    ap.add_argument("--radii", type=float, nargs="+", default=None,
                    help="rays (additive): the absolute-grid radii to probe (space-separated). Default None = the module RADII, BYTE-IDENTICAL to the current behaviour; a given list REPLACES the absolute grid (the relative/alpha grid is unchanged) and is recorded in the ray identity.")
    ap.add_argument("--out-suffix", default=None,
                    help="rays (additive): a suffix inserted before '_rays.json' in the OUTPUT filename, so an extended-radii or exposure probe writes to a SEPARATE record and leaves the canonical '<name>_rays.json' untouched. Default None = unchanged.")
    ap.add_argument("--exposure-indices", default=None,
                    help="rays (additive): a .npy of training-set row indices on which to EVALUATE the SAME fitted-set branch directions (held-out / excluded rows). The branch bases are loaded from the fitted build unchanged (still authenticated against the build record); only the evaluation inputs differ, so the build-hash and build-identity assertions are skipped. The record is labelled 'exposure' with the exposure index sha256 in provenance. Default None = the fitted set, unchanged.")
    ap.add_argument("--abs-only", action="store_true",
                    help="rays (additive): compute ONLY the absolute (Euclidean-norm) grid and skip the relative/alpha grid. Default False = both grids, byte-identical to the current behaviour. Used to probe extra absolute radii cheaply without recomputing the unchanged relative grid.")
    a = ap.parse_args()
    import numpy as _np
    a.custom_idx = _np.load(a.indices) if a.indices else None
    a.exposure_idx = _np.load(a.exposure_indices) if a.exposure_indices else None
    {"tangent": tangent, "rays": rays, "adopt": adopt}[a.phase](a)


