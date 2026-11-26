"""Direct activation-transition residual with 32 paired probes (protocol v2).

Protocol: memos/2026-09-10_protocol_stages_atomic_transition.md, Section 4
(amendment).  Identifier: transition-residual-v2.  For every run and stage
(declared stages plus epochs 100 and 250), on the fixed analysis set in the
float64 representation:

  base diagnostics per gated layer
    signed normalized margin z / (||[w;b]|| ||[h;1]||): 0.1-decade magnitude
    histogram split by base sign; exact zeros and undefined denominators
    counted separately; unit states (always inactive, always active, mixed)
    per unit, and for the CNN also per channel-position coordinate;
  per probe (32 Gaussian ambient directions, seeds 100-131, unit norm) and
  radius (the original absolute grid):
    ||dF_true||_X, ||dF_frz||_X, ||A_tr||_X by subtraction and by residual
    propagation, their discrepancy (identity residual), per-layer gate
    disagreement counts, flipped-gate counts by (base sign, magnitude bin)
    and by unit-state category, with the base denominators stored once.

Outputs: results/transition_v2/<dataset>_<model>_seed<s>.json (everything
except the conditional-count arrays) and ..._counts.npz (the arrays).

    python -m flatocean transition-residual --dataset mnist --model mlp --seeds 0 1 2

Block-relative radii (plan amendment 2026-09-17, identifier transition-residual-v3-relative).  With
theta_l = [W_l; b_l] the parameter block of layer slot l, the same 32 ambient probes g are rescaled block by
block, u_l = ||theta_l|| g_l / ||g_l||, and the perturbation is w = alpha u, so ||w_l|| = alpha ||theta_l|| in
every perturbed block.  --perturb all moves every block by the same fraction; --perturb <block> (W1, W2, ...,
V, conv1, ...) moves one block and leaves the others fixed.  Outputs go to results/transition_rel/
<dataset>_<model>_seed<s>_<perturb>.json with the block norms recorded per stage.

    python -m flatocean transition-residual --radius-mode relative --perturb all --seeds 0 1 2 --device cpu
"""
# Ported unchanged from experiments/exp_transition_residual.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean transition-residual).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
from flatocean import records, runs
import torch

from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean import tier1_train as T
from flatocean.runs import stage_epochs

OUT = runs.out_dir("transition_v2")
PROTOCOL = "memos/2026-09-10_protocol_stages_atomic_transition.md, Section 4 (amendment 2026-09-10); identifier transition-residual-v2"
from flatocean.directions import (RADII, ALPHAS, DIRECTION_SEEDS, RADII_V3, NBINS, STATE_NAMES, magnitude_bin, base_diagnostics, ray,     # noqa: F401
                                  relative_direction, absolute_direction, block_slices, OUT_REL, OUT_V3, PROTOCOL_REL, PROTOCOL_V3)


def recorded_epochs(res):
    """Epochs with a computed record (an alias {'same_as_epoch': e} is not a record)."""
    return {v["epoch"] for v in res["results"].values() if "epoch" in v}


CURVATURE_FIELDS = ("curvature", "curvature_defined", "preactivation_displacement")
BASE_FIELDS = ("true", "frozen", "transition", "identity_residual", "flips", "rho_norms", "propagated_norms", "propagated_sum_of_norms", "transition_propagated", "direction_seed")
V3_FIELDS = ("cancellation_ratio", "base_output_norm_X", "radius_over_param_norm", "block_radius_over_block_norm")


def stage_defects(done, radii, seeds, smooth):
    """Why a stored v3 stage record is incomplete: every configured radius must hold exactly one probe per configured direction
    seed, in that order; every probe the base diagnostics, the v3 decomposition fields and, for a smooth activation, the
    curvature fields.  Empty list = complete."""
    probes = done.get("probes")
    if not isinstance(probes, dict):
        return ["no probe collection"]
    out = []
    for r in radii:
        lst = probes.get(str(r))
        if not isinstance(lst, list) or len(lst) != len(seeds):
            out.append(f"radius {r}: {0 if not lst else len(lst)}/{len(seeds)} probes"); continue
        if [q.get("direction_seed") for q in lst] != list(seeds):
            out.append(f"radius {r}: direction seeds differ from the configuration"); continue
        need = BASE_FIELDS + V3_FIELDS + (CURVATURE_FIELDS if smooth else ())
        bad = [f for f in need if any(f not in q for q in lst)]
        if bad:
            out.append(f"radius {r}: probes lack {bad}")
    extra = sorted(set(probes) - {str(r) for r in radii})
    if extra:
        out.append(f"radii not in the configuration: {extra}")
    return out


def radii_without_curvature(done):
    """Kept for the older call sites: radii whose probes lack the curvature validity flag."""
    return [r for r, lst in done.get("probes", {}).items() if any("curvature_defined" not in q for q in lst)]


def drop_stage(res, stage):
    """Remove a stale stage record and return the refreshed set of recorded epochs, so that a stage whose only record was
    the stale one is recomputed instead of being aliased to its own epoch."""
    res["results"].pop(stage, None)
    return recorded_epochs(res)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return None


def generic_config(nprobes, mode="absolute", perturb="all", v3=False):
    if v3:
        return {"diagnostics_version": 3, "radius_mode": "block-relative" if mode == "relative" else "absolute", "perturb": perturb,
                "radii": list(ALPHAS if mode == "relative" else RADII_V3), "direction_seeds": list(DIRECTION_SEEDS[:nprobes]),
                "tolerances": FL.TOL, "protocol": PROTOCOL_V3, "nbins": NBINS}
    if mode == "relative":
        return {"diagnostics_version": 2, "radius_mode": "block-relative", "perturb": perturb, "radii": list(ALPHAS),
                "direction_seeds": list(DIRECTION_SEEDS[:nprobes]), "tolerances": FL.TOL, "protocol": PROTOCOL_REL, "nbins": NBINS}
    return {"diagnostics_version": 2, "radii": list(RADII), "direction_seeds": list(DIRECTION_SEEDS[:nprobes]), "tolerances": FL.TOL,
            "protocol": PROTOCOL, "nbins": NBINS}


def validate_generic_record(done, current_identity, res, allow_legacy):
    """Validate a completed stage record before reuse: full identity if recorded, else the legacy fields
    (checkpoint hash, probe seeds); raises on mismatch, returns the status to propagate."""
    if "identity" in done:
        return P.require_match(done["identity"], current_identity, "generic stage record")
    if done.get("direction_seeds", res.get("direction_seeds")) != current_identity["config"]["direction_seeds"]:
        raise P.ProvenanceError("generic stage record: recorded probe seeds differ from the current configuration")
    return P.legacy_status({"checkpoint_sha256": done.get("checkpoint_sha256")}, current_identity, "generic stage record", allow_legacy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", nargs="*", default=None, help="subset of stage names (default: all)")
    ap.add_argument("--nprobes", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true", help="recompute stages that are already in the result file")
    ap.add_argument("--allow-legacy", action="store_true", help="reuse stage records without a full identity (status propagated as UNVERIFIED)")
    ap.add_argument("--radius-mode", default="absolute", choices=["absolute", "relative"])
    ap.add_argument("--perturb", default="all", help="relative mode: 'all' or one block name (W1, W2, ..., V, conv1, ...)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--v3", action="store_true", help="2026-09-22 protocol: results/transition_v3, 256 matched probes, absolute grid to 100 or the relative grid, "
                                                      "block-specific absolute perturbations, complete stored decomposition")
    ap.add_argument("--tag", default="", help="run tag (gelu, leaky, w64, ...)")
    ap.add_argument("--exploratory", action="store_true", help="allow fewer than 256 probes under --v3 (smoke tests)")
    ap.add_argument("--out-dir", default=None, help="v3 output directory (smoke tests write elsewhere)")
    ap.add_argument("--block", type=int, default=1000, help="samples per forward block (GPU memory; results are block-independent, validated to 1e-13)")
    a = ap.parse_args()
    if a.v3 and a.nprobes < 256 and not a.exploratory:
        raise SystemExit("the v3 protocol uses 256 matched probes (--nprobes 256); smaller counts are exploratory and must be requested with --exploratory")
    rel = a.radius_mode == "relative"
    out_dir, radii, protocol = (OUT_REL, ALPHAS, PROTOCOL_REL) if rel else (OUT, RADII, PROTOCOL)
    if a.v3:
        out_dir, radii, protocol = (a.out_dir or OUT_V3), (ALPHAS if rel else RADII_V3), PROTOCOL_V3
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag)
        stages = stage_epochs(rd)
        sums_path = os.path.join(rd, "checksums.json")                     # written after training; absent while a run is in progress
        sums = json.load(open(sums_path)) if os.path.exists(sums_path) else {}
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}" + (f"_{a.radius_mode}_{a.perturb}" if a.v3 else (f"_{a.perturb}" if rel else ""))
        jpath = os.path.join(out_dir, f"{name}.json"); npath = os.path.join(out_dir, f"{name}_counts.npz")
        res = json.load(open(jpath)) if os.path.exists(jpath) else {
            "protocol": protocol, "code_revision": git_rev(), "run": rd, "dtype": "float64 analysis representation",
            "radius_mode": a.radius_mode, "perturb": a.perturb if rel else "all (unit-norm ambient probe)",
            "radii": radii, "direction_seeds": DIRECTION_SEEDS[:a.nprobes], "tolerances": FL.TOL, "stages": stages, "results": {}}
        counts = dict(np.load(npath, allow_pickle=True)) if os.path.exists(npath) else {}
        done_epochs = recorded_epochs(res)
        for stage, info in stages.items():
            ep = info["checkpoint_epoch"]
            if ep is None or (a.stages and stage not in a.stages):
                continue
            if stage in res["results"] and "epoch" in res["results"][stage] and not a.overwrite:
                done = res["results"][stage]
                run_chk = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device)
                ident = P.analysis_identity(run_chk, rd, ep, generic_config(a.nprobes, a.radius_mode, a.perturb, a.v3)); del run_chk
                status = validate_generic_record(done, ident, res, a.allow_legacy)      # raises on any recorded mismatch
                done["provenance_status"] = status
                v = done.get("diagnostics_version", 1)
                # v3: a record is complete only with every configured radius, one probe per configured seed, every base
                # diagnostic and, for a smooth activation, the curvature fields
                if a.v3:
                    smooth = FL.activation_of(O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu")["net"]) not in ("relu", "leaky")
                    defects = stage_defects(done, radii, DIRECTION_SEEDS[:a.nprobes], smooth)
                    if defects:
                        print(f"== {name} {stage}: stored record incomplete ({'; '.join(defects[:3])}{' ...' if len(defects) > 3 else ''}); recomputing", flush=True)
                        done_epochs = drop_stage(res, stage)          # the stale record must not leave its epoch behind as an alias target
                    else:
                        print(f"== {name} {stage}: already computed ({status}; diagnostics version {v}, all radii, probes and diagnostics present)", flush=True); continue
                else:
                    print(f"== {name} {stage}: already computed ({status}; diagnostics version {v}" + ("" if v >= 2 else ", per-layer residuals absent; use --overwrite to add them") + ")", flush=True); continue
            if ep in done_epochs and not a.overwrite:
                res["results"][stage] = {"same_as_epoch": ep}; continue
            t0 = time.time()
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device)
            net = run["net"]; X, y, _ = run["data"]["analysis"]
            theta0 = FL.flat_params(net).clone(); p = theta0.numel()
            masks = FL.base_masks(net, X, block=a.block)
            with torch.no_grad():                                          # blocked base forward (a full 10k CNN forward needs ~17 GB)
                F0 = torch.cat([net(X[i:i + a.block]) for i in range(0, X.shape[0], a.block)])
            Y = torch.nn.functional.one_hot(y, F0.shape[1]).to(F0.dtype)
            hist, zeros, undefined, bins, units = base_diagnostics(net, X, masks, block=a.block)
            unit_state = [u[1] for u in units]; pos_state = [u[2] for u in units]
            valid = FL._validate(net, X[:500], f"{name} epoch {ep}", a.device) if stage == list(stages)[0] or True else None
            ident = P.analysis_identity(run, rd, ep, generic_config(a.nprobes, a.radius_mode, a.perturb, a.v3))
            if sums.get(f"checkpoints/epoch_{ep:04d}.pt") not in (None, ident["checkpoint_sha256"]):
                raise RuntimeError(f"{name} epoch {ep}: checkpoint file hash differs from checksums.json")
            base = {"epoch": ep, "checkpoint_sha256": ident["checkpoint_sha256"], "identity": ident, "provenance_status": "identity verified",
                    "code_revision": git_rev(), "param_norm": theta0.norm().item(),
                    "F0_norm_X": FL.x_norm(F0), "Y_norm_X": FL.x_norm(Y), "n_gates": [int(m.numel()) for m in masks],
                    "validation": {k: v for k, v in valid.items() if not isinstance(v, dict)} if valid else None,
                    "diagnostics_version": 2,          # per-layer rho norms, propagated contributions, position-state counts (2026-09-11)
                    "layers": [{"exact_zero_preactivations": zeros[l], "undefined_denominators": undefined[l],
                                "active_fraction": masks[l].double().mean().item(), **units[l][0]} for l in range(len(masks))],
                    "probes": {str(r): [] for r in radii}}
            slices = block_slices(net)
            base["block_norms"] = {n_: theta0[i0:i1].norm().item() for n_, i0, i1 in slices}
            base_Z = None
            if FL.activation_of(net) not in ("relu", "leaky"):                  # smooth activation: base preactivations for the displacement and curvature diagnostics
                with torch.no_grad():
                    base_Z = [torch.cat(z) for z in zip(*[FL.true_states(net, X, torch.arange(i, min(i + a.block, X.shape[0]), device=a.device))[1] for i in range(0, X.shape[0], a.block)])]
            bb = np.zeros((len(masks), len(radii), a.nprobes, 2, NBINS), dtype=np.int64)
            bs = np.zeros((len(masks), len(radii), a.nprobes, 3), dtype=np.int64); bp = np.zeros_like(bs)
            for k, ds in enumerate(DIRECTION_SEEDS[:a.nprobes]):
                g = torch.Generator().manual_seed(ds)
                u = torch.randn(p, generator=g, dtype=torch.float64).to(a.device)
                u = relative_direction(u, theta0, slices, a.perturb) if rel else (absolute_direction(u, slices, a.perturb) if a.v3 else u / u.norm())
                for j, r in enumerate(radii):
                    m, by_bin, by_state, by_pos = ray(net, theta0, masks, X, u, r, F0, bins, unit_state, block=a.block, pos_state=pos_state, base_Z=base_Z)
                    m["direction_seed"] = ds
                    if a.v3:                                                       # complete stored decomposition (item 7)
                        m["cancellation_ratio"] = m["propagated_sum_of_norms"] / m["transition_propagated"] if m["transition_propagated"] > 0 else None
                        m["base_output_norm_X"] = base["F0_norm_X"]
                        m["radius_over_param_norm"] = float(r * u.norm() / theta0.norm())
                        m["block_radius_over_block_norm"] = {n_: float(r * u[i0:i1].norm() / theta0[i0:i1].norm()) for n_, i0, i1 in slices if u[i0:i1].any()}
                    base["probes"][str(r)].append(m)
                    for l in range(len(masks)):
                        bb[l, j, k] = by_bin[l]; bs[l, j, k] = by_state[l]; bp[l, j, k] = by_pos[l]
                if k % 8 == 7:
                    print(f"   {name} {stage}: {k + 1}/{a.nprobes} probes, {time.time() - t0:.0f}s", flush=True)
            base["seconds"] = time.time() - t0
            res["results"][stage] = base
            counts[f"{stage}_hist_by_sign"] = np.stack([h.cpu().numpy() for h in hist])
            counts[f"{stage}_flips_by_sign_bin"] = bb; counts[f"{stage}_flips_by_state"] = bs
            counts[f"{stage}_gates_per_state"] = np.array([units[l][0]["gates_per_state"] for l in range(len(masks))])
            if any(u[2] is not None for u in units):
                counts[f"{stage}_flips_by_position_state"] = bp
                counts[f"{stage}_gates_per_position_state"] = np.array([units[l][0]["gates_per_position_state"] for l in range(len(masks))])
            np.savez_compressed(npath, **counts); records.dump_json(jpath, res)
            done_epochs.add(ep)
            r1 = [q for q in base["probes"]["0.1" if rel else "1.0"]]
            med = lambda key: float(np.median([q[key] for q in r1]))
            print(f"== {name} {stage} (epoch {ep}) done in {base['seconds']:.0f}s: at {'alpha=0.1' if rel else 'r=1'} median true {med('true'):.3e} frozen {med('frozen'):.3e} "
                  f"transition {med('transition'):.3e} identity {med('identity_residual'):.1e}; flips {np.median([q['flips'] for q in r1], 0).astype(int).tolist()}", flush=True)
            del net, run, masks, bins; torch.cuda.empty_cache()


