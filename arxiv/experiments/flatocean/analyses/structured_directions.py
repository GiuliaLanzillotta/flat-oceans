"""Structured directions through the activation-transition measurement (protocol Section 4,
"Directions, radii, and uncertainty"; identifier transition-residual-v2, structured).

For each stage of a run and each requested block (MLP layer 2, 3, readout), the 32 ambient
probes (seeds 100-131) are projected into a subspace of that block's parameters, with all
other layers fixed, and normalized:

  kernel_<blk>           atomic kernel K_t of the block (declared rank rule, 1e-9), a certified frozen branch
  tail_<blk>_<eps>       positive spectral tail E_t^+(eps) = E_t(eps) \\ K_t on J/sqrt(s)
  dead_out_<blk>         coordinate subspace: outgoing weights of always-inactive input units (exact branch)
  dead_in_<blk>          coordinate subspace: incoming weights and bias of always-inactive output units
                         (frozen drift is zero; true drift measures reactivation)

The subspace bases come from results/persistence/<run>/epoch<e>_<blk>.npz (flatocean.analyses.persistence factor).
Every ray is measured exactly as for the generic probes (exp_transition_residual.ray): true drift,
frozen drift, transition residual (subtraction and propagation), identity residual, flips per layer,
and the conditional flip counts.  The projector, dimension, projection residual per probe and rejected
projections are recorded.  Output: results/transition_v2/<run>_structured.json and _structured_counts.npz.

    python -m flatocean structured-directions --seeds 0 --blocks 2 3 readout
    python -m flatocean structured-directions test
"""
# Ported unchanged from experiments/exp_structured_directions.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean structured-directions).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os
import sys
import time

import numpy as np
from flatocean import records, runs
import torch

from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs
from flatocean.analyses.transitions import RADII, DIRECTION_SEEDS, NBINS, PROTOCOL, base_diagnostics, ray, git_rev
from flatocean.analyses.persistence import load_factor, subspaces, dead_coordinates, apply_J, FACTOR_DIR as PERSIST   # factors are inputs (input root)

OUT = runs.out_dir("transition_v2")
TAIL_EPS = (1e-3, 1e-4, 1e-6)
DIAGNOSTICS_VERSION = 2            # 2: per-layer residual norms and propagated contributions in every ray record (2026-09-11)
REJECT_REL = 1e-8


def block_index_map(net, blk):
    """Flat-parameter index of every block coordinate, in ocean_lib's column-major order
    (layer block: index k*n + j <-> W[j, k] for k < m, bias[j] for k = m;
     readout: index k*d + c <-> V[c, k] for k < n_L, bias[c] for k = n_L)."""
    sl = {name: (s, e) for name, s, e in FL.param_slices(net)}
    if blk == "readout":
        W = net.readout.weight; d, m = W.shape
        ws, _ = sl["readout.weight"]; bs, _ = sl["readout.bias"]
        idx = np.empty((m + 1) * d, dtype=np.int64)
        for k in range(m):
            idx[k * d: (k + 1) * d] = ws + np.arange(d) * m + k
        idx[m * d:] = bs + np.arange(d)
        return idx
    l = int(blk) - 1
    W = net.layers[l].weight; n, m = W.shape
    ws, _ = sl[f"layers.{l}.weight"]; bs, _ = sl[f"layers.{l}.bias"]
    idx = np.empty((m + 1) * n, dtype=np.int64)
    for k in range(m):
        idx[k * n: (k + 1) * n] = ws + np.arange(n) * m + k
    idx[m * n:] = bs + np.arange(n)
    return idx


def embed(idx, v_block, p_total, device):
    u = torch.zeros(p_total, dtype=torch.float64, device=device)
    u[torch.as_tensor(idx, device=device)] = v_block.to(device)
    return u


def structured_config():
    return {"diagnostics_version": DIAGNOSTICS_VERSION, "radii": list(RADII), "direction_seeds": list(DIRECTION_SEEDS), "tail_eps": list(TAIL_EPS),
            "reject_rel": REJECT_REL, "tolerances": FL.TOL, "protocol": PROTOCOL + "; structured directions", "nbins": NBINS}


def reuse_decision(old, factor_sha):
    """Whether a stored category may be reused given the hash of the factor whose basis it was projected on (2026-09-25).
    Returns (reuse, note): a record with the same hash is reused; a record with another hash is recomputed; a record
    without the field (written before the hash was recorded) is reused, since the stage validation has authenticated
    the current factor against the run, and is annotated so that the missing binding stays visible."""
    old_sha = old.get("factor_sha256")
    if old_sha is None:
        return True, "factor hash not recorded when measured (before 2026-09-25); current factor authenticated, binding not verified"
    if old_sha == factor_sha:
        return True, None
    return False, f"factor changed since the measurement ({old_sha[:12]} -> {factor_sha[:12]})"


def validate_stage_record(old_rec, current_identity, allow_legacy):
    """Validate an existing stage record BEFORE any mutation.  Records with a full identity must match it; older
    records (checkpoint hash only) are accepted only with allow_legacy and only if their recorded hash matches.
    Returns the provenance status to attach to preserved categories; raises P.ProvenanceError otherwise."""
    if old_rec is None:
        return None
    if "identity" in old_rec:
        return P.require_match(old_rec["identity"], current_identity, "structured stage record")
    return P.legacy_status({"checkpoint_sha256": old_rec.get("checkpoint_sha256")}, current_identity, "structured stage record", allow_legacy)


def categories(name, ep, blk, st, s, blocks_dir, run=None, rd=None, allow_legacy=False):
    """Orthonormal bases (columns, torch float64 on CPU) of the structured subspaces of one block at one stage.
    The factor's provenance is validated against the current run before use."""
    cats = {}
    sv, V, s_f, p, status = load_factor(name, ep, blk, run, rd, allow_legacy, return_status=True)
    assert s_f == s
    cats["_factor_status"] = status
    cats["_factor_sha256"] = P.sha256_file(os.path.join(blocks_dir, name, f"epoch{ep:04d}_{blk}.npz"))
    r, K, E, Ep = subspaces(sv, V, s)
    cats[f"kernel_{blk}"] = (V[:, K].clone(), {"dim": len(K), "rank": r, "source": "right singular vectors, sigma <= 1e-9 sigma_1"})
    for e in TAIL_EPS:
        cols = Ep[str(e)]
        cats[f"tail_{blk}_{e:g}"] = (V[:, cols].clone(), {"dim": len(cols), "eps": e, "source": "right singular vectors, 1e-9 sigma_1 < sigma <= eps sqrt(s)"})
    dc = dead_coordinates(st, blk, s)
    for key, cols in (("dead_out", dc["dead_input_columns"]), ("dead_in", dc["dead_output_rows"])):
        B = torch.zeros(p, len(cols), dtype=torch.float64)
        for j, c in enumerate(cols):
            B[c, j] = 1.0
        cats[f"{key}_{blk}"] = (B, {"dim": len(cols), "source": "coordinate subspace"})
    del V
    return cats


def _test_stage_validation():
    """Regression: an old stage record with a different checkpoint hash and no per-category hash must be refused
    before any metadata is modified; a matching legacy record is accepted only with allow_legacy and gets an
    UNVERIFIED status; a full-identity record must match field by field."""
    cur = {"identity_version": 1, "checkpoint_sha256": "cur", "preprocessing_sha256": "p", "analysis_index_sha256": "i", "analysis_size": 3,
           "analysis_dtype": "float64", "config": {}}; cur["digest"] = P.identity_digest(cur)
    old = {"epoch": 500, "checkpoint_sha256": "OLD", "categories": {"kernel_2": {"dim": 3, "probes": {}}}}
    snapshot = json.dumps(old, sort_keys=True)
    for flag in (False, True):
        try:
            validate_stage_record(old, cur, flag); raise AssertionError("stale stage accepted")
        except P.ProvenanceError:
            pass
    assert json.dumps(old, sort_keys=True) == snapshot, "record modified during validation"
    ok = {"epoch": 500, "checkpoint_sha256": "cur", "categories": {}}
    try:
        validate_stage_record(ok, cur, False); raise AssertionError("legacy accepted without flag")
    except P.ProvenanceError:
        pass
    assert "UNVERIFIED" in validate_stage_record(ok, cur, True)
    other = dict(cur); other["analysis_index_sha256"] = "J"; other["digest"] = P.identity_digest(other)
    try:
        validate_stage_record({"identity": other}, cur, True); raise AssertionError("different sample order accepted")
    except P.ProvenanceError:
        pass
    assert validate_stage_record({"identity": cur}, cur, False) == "identity verified"
    assert reuse_decision({"factor_sha256": "abc"}, "abc") == (True, None)
    assert reuse_decision({"factor_sha256": "abc"}, "abd")[0] is False
    ok, note = reuse_decision({}, "abc"); assert ok and "not recorded" in note
    print("exp_structured_directions: stage-record validation and reuse-decision tests passed")


def _test(device="cuda"):
    _test_stage_validation()
    """Embedding test: a single-block perturbation of the frozen network is exactly linear, so
    Delta F^frz(h * embed(v)) / h must equal apply_J(v) * sqrt(s) for every block."""
    run = O.load_run("mnist", "mlp", 0, epoch=500, device=device); net = run["net"]
    X = run["data"]["analysis"][0][:2000]; s = X.shape[0]
    st = O.mlp_structure(net, X); masks = FL.base_masks(net, X)
    theta0 = FL.flat_params(net).clone(); p = theta0.numel()
    idx_all = torch.arange(s, device=device)
    F0 = FL.frozen_forward(net, masks, X, idx_all)
    worst = 0.0
    for blk in ("2", "3", "readout"):
        idx = block_index_map(net, blk)
        assert len(set(idx.tolist())) == len(idx)
        g = torch.Generator().manual_seed(7)
        v = torch.randn(len(idx), generator=g, dtype=torch.float64); v /= v.norm()
        u = embed(idx, v, p, device)
        h = 1.0                                   # exact linearity in one block: any step; h = 1 keeps roundoff at 1e-13
        FL.set_flat(net, theta0 + h * u)
        dF = (FL.frozen_forward(net, masks, X, idx_all) - F0) / h
        FL.set_flat(net, theta0)
        Jv = apply_J(st, blk if blk == "readout" else int(blk), v.to(device), s) * np.sqrt(s)
        rel = ((dF - Jv).norm() / Jv.norm()).item(); worst = max(worst, rel)
        print(f"  block {blk}: p={len(idx)} relative disagreement frozen-vs-J {rel:.2e}")
    assert worst < 1e-11, worst
    # ordering check: the last n coordinates of a layer block are the biases
    idx = block_index_map(net, "2"); sl = {n: (a, b) for n, a, b in FL.param_slices(net)}
    assert idx[-1] == sl["layers.1.bias"][1] - 1 and idx[0] == sl["layers.1.weight"][0]
    print("exp_structured_directions: embedding tests passed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _test(); return
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--blocks", nargs="+", default=["2", "3", "readout"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true", help="recompute categories already in the result file")
    ap.add_argument("--allow-legacy", action="store_true", help="accept factor files without a manifest and without a passing audit entry")
    ap.add_argument("--tag", default="", help="run tag (gelu, leaky, ...)")
    a = ap.parse_args()
    torch.set_num_threads(8)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        sums = json.load(open(os.path.join(rd, "checksums.json")))
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        jpath = os.path.join(OUT, f"{name}_structured.json"); npath = os.path.join(OUT, f"{name}_structured_counts.npz")
        res = json.load(open(jpath)) if os.path.exists(jpath) else {
            "protocol": PROTOCOL + "; structured directions", "code_revision": git_rev(), "run": rd, "radii": RADII,
            "direction_seeds": DIRECTION_SEEDS, "tail_eps": TAIL_EPS, "reject_rel": REJECT_REL, "tolerances": FL.TOL, "stages": stages, "results": {}}
        counts = dict(np.load(npath, allow_pickle=True)) if os.path.exists(npath) else {}
        for stage, info in stages.items():
            ep = info["checkpoint_epoch"]
            if ep is None or (a.stages and stage not in a.stages):
                continue
            missing = [b for b in a.blocks if not os.path.exists(os.path.join(PERSIST, name, f"epoch{ep:04d}_{b}.npz"))]
            if missing:
                print(f"== {name} {stage}: basis files missing for blocks {missing}, skipped", flush=True); continue
            done = res["results"].get(stage, {}).get("categories", {})
            t0 = time.time()
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device)
            net = run["net"]; X, y, _ = run["data"]["analysis"]; s = X.shape[0]
            theta0 = FL.flat_params(net).clone(); p = theta0.numel()
            masks = FL.base_masks(net, X); F0 = net(X)
            st = O.mlp_structure(net, X)
            hist, zeros, undefined, bins, units = base_diagnostics(net, X, masks)
            unit_state = [u[1] for u in units]
            ident = P.analysis_identity(run, rd, ep, structured_config())
            if sums.get(f"checkpoints/epoch_{ep:04d}.pt") not in (None, ident["checkpoint_sha256"]):
                raise RuntimeError(f"{name} epoch {ep}: checkpoint file hash differs from checksums.json")
            old_rec = res["results"].get(stage)
            if old_rec is not None and a.overwrite:
                print(f"== {name} {stage}: --overwrite, discarding the existing record (all categories recomputed with fresh metadata)", flush=True)
                old_rec = None
            preserved_status = validate_stage_record(old_rec, ident, a.allow_legacy)   # raises before anything is modified
            if old_rec is None:
                rec = res["results"][stage] = {"epoch": ep, "checkpoint_sha256": ident["checkpoint_sha256"], "identity": ident,
                                               "param_norm": theta0.norm().item(), "categories": {}}
            else:
                rec = old_rec                                                          # old metadata kept; fresh metadata only per recomputed category
                for cat, crec in rec["categories"].items():
                    crec.setdefault("provenance_status", preserved_status)
            for blk in a.blocks:
                idx = block_index_map(net, blk)
                cats = categories(name, ep, blk, st, s, PERSIST, run, rd, a.allow_legacy); factor_status = cats.pop("_factor_status"); factor_sha = cats.pop("_factor_sha256")
                for cat, (B, meta) in cats.items():
                    if cat in done and not a.overwrite:
                        old = done[cat]                                                # validated above with the stage record
                        reuse, note = reuse_decision(old, factor_sha)
                        if reuse:
                            if note:
                                old["factor_sha256_at_reuse"] = note
                            if old.get("diagnostics_version", 1) < DIAGNOSTICS_VERSION:
                                print(f"   {name} {stage} {cat}: kept (diagnostics version {old.get('diagnostics_version', 1)} < {DIAGNOSTICS_VERSION}; rerun with --overwrite to add per-layer residuals)", flush=True)
                            continue
                        print(f"   {name} {stage} {cat}: {note}; recomputed", flush=True)
                    k = B.shape[1]
                    crec = {**meta, "block": blk, "projector": f"P = B B^T, B = stored basis ({meta['source']})", "degenerate": k < len(DIRECTION_SEEDS),
                            "factor_status": factor_status, "factor_sha256": factor_sha, "checkpoint_sha256": ident["checkpoint_sha256"], "identity_digest": ident["digest"],
                            "provenance_status": "identity verified", "diagnostics_version": DIAGNOSTICS_VERSION,
                            "code_revision": git_rev(), "rejected": [], "projection_residual": [], "probes": {str(r): [] for r in RADII}}
                    if k == 0:
                        crec["note"] = "empty space, unavailable"; rec["categories"][cat] = crec; continue
                    bb = np.zeros((len(masks), len(RADII), len(DIRECTION_SEEDS), 2, NBINS), dtype=np.int64)
                    bs = np.zeros((len(masks), len(RADII), len(DIRECTION_SEEDS), 3), dtype=np.int64)
                    for q, ds in enumerate(DIRECTION_SEEDS):
                        g = torch.Generator().manual_seed(ds)
                        u_full = torch.randn(p, generator=g, dtype=torch.float64); u_full /= u_full.norm()
                        ub = u_full[torch.as_tensor(idx)]
                        vb = B @ (B.T @ ub); nv = vb.norm().item()
                        if nv < REJECT_REL * ub.norm().item():
                            crec["rejected"].append(ds); continue
                        crec["projection_residual"].append(((ub - vb).norm() / ub.norm()).item())
                        u = embed(idx, vb / nv, p, a.device)
                        for j, r in enumerate(RADII):
                            m, by_bin, by_state, _ = ray(net, theta0, masks, X, u, r, F0, bins, unit_state)
                            m["direction_seed"] = ds; crec["probes"][str(r)].append(m)
                            for l in range(len(masks)):
                                bb[l, j, q] = by_bin[l]; bs[l, j, q] = by_state[l]
                    rec["categories"][cat] = crec
                    counts[f"{stage}_{cat}_flips_by_sign_bin"] = bb; counts[f"{stage}_{cat}_flips_by_state"] = bs
                    r1 = crec["probes"]["1.0"]
                    med = lambda key: float(np.median([x[key] for x in r1])) if r1 else float("nan")
                    print(f"   {name} {stage} {cat}: dim {k}, at r=1 median true {med('true'):.3e} frozen {med('frozen'):.3e} transition {med('transition'):.3e} "
                          f"flips {np.median([x['flips'] for x in r1], 0).astype(int).tolist() if r1 else None} ({time.time() - t0:.0f}s)", flush=True)
                    np.savez_compressed(npath, **counts); records.dump_json(jpath, res)
            counts[f"{stage}_gates_per_state"] = np.array([units[l][0]["gates_per_state"] for l in range(len(masks))])
            np.savez_compressed(npath, **counts); records.dump_json(jpath, res)
            print(f"== {name} {stage} done in {time.time() - t0:.0f}s", flush=True)
            del net, run, masks, bins, st; torch.cuda.empty_cache()


