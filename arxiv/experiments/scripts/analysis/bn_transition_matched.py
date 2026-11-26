"""Matched-coordinate activation-transition probes for the BN-vs-plain comparison (request 2026-10-01).

Why a dedicated driver (not `flatocean transition-residual --v3`): the v3 path draws g = randn(p) with p the FULL
parameter count, which differs between mnist_mlp_w64 (p=25418) and mnist_mlp_bn_w64 (p=25802, +384 BN gamma/beta).
The PyTorch normal draw diverges in its tail (first differing index ~25402, inside the readout block), so the same
direction seed gives NON-identical W/b vectors across the two models even after `absolute_direction` masks the BN
entries.  The request requires IDENTICAL W/b vectors across the paired models, not merely identical RNG labels.

Fix (explicit conditional-coordinate mask): the Gaussian is drawn ONCE on the common W/b coordinate space (the union
of the [W_l; b_l] block slices, dimension 25418, identical in both models) and embedded into each model's flat vector
at exactly those positions; gamma/beta positions stay ZERO.  This gives bit-identical W/b directions and zero BN mass.
Everything else reuses the authenticated library: frozen_lib (BN-corrected frozen_forward / frozen_tail_linear carry
the eval-mode slope a_l), directions.base_diagnostics and directions.ray.

BN is analysed in EVALUATION mode with the checkpoint's running statistics and gamma/beta held fixed; only ordinary
W/b coordinates are perturbed.  Geometry/response are reported in the original affine coordinates.  Grids: the absolute
Euclidean radius grid 1e-3..100 (directions.RADII_V3).  Probe seeds 100..355 (256 matched Gaussian directions).

Outputs: results/transition_bn/<dataset>_<model>_<tag>_seed<s>.json (one file per run; stages initialization and
late_endpoint).  Does not touch results/transition_v2 / results/transition_v3.

    python -u scripts/analysis/bn_transition_matched.py --tags w64 bn_w64 --seeds 0 1 2 \
        --stages initialization late_endpoint --device cuda --nprobes 256
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/
import sys
sys.path.insert(0, HERE)

from flatocean import frozen_lib as FL
from flatocean import provenance as P
from flatocean import runs, records
from flatocean.directions import RADII_V3, DIRECTION_SEEDS, base_diagnostics, ray, block_slices
from flatocean.runs import stage_epochs

PROTOCOL = ("memos/2026-10-01_request_bn_mechanism_and_transitions.md Section B; identifier "
            "transition-residual-bn-matched-v1 (common W/b coordinate Gaussian, BN gamma/beta held at 0, eval-mode)")
OUT = runs.out_dir("transition_bn")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return None


def common_coord_descriptor(net_plain, net_bn):
    """Verify the two models share the same ordered W/b block slices, and return the common coordinate layout."""
    sp = block_slices(net_plain); sb = block_slices(net_bn)
    assert [n for n, _, _ in sp] == [n for n, _, _ in sb], (sp, sb)
    assert [(b - a) for _, a, b in sp] == [(b - a) for _, a, b in sb], "block sizes differ between the two models"
    dim = sum(b - a for _, a, b in sp)
    return {"blocks": [{"name": n, "size": b - a, "plain_span": [a, b], "bn_span": [sb[i][1], sb[i][2]]}
                       for i, (n, a, b) in enumerate(sp)],
            "common_dim": dim,
            "note": "Gaussian drawn on this common W/b space (dim %d); embedded at the listed spans; BN gamma/beta held at 0" % dim}


def matched_direction(net, g_common, slices):
    """Embed the common-coordinate Gaussian g_common into net's flat vector at its W/b block positions (in block order),
    leave BN gamma/beta at zero, and return the unit-norm absolute direction u (length = net param count)."""
    p = sum(x.numel() for x in net.parameters())
    u = torch.zeros(p, dtype=torch.float64)
    off = 0
    for _, a, b in slices:
        n = b - a
        u[a:b] = g_common[off:off + n]
        off += n
    assert off == g_common.numel()
    return (u / u.norm())


def bn_slope_summary(net):
    if not getattr(net, "batchnorm", False):
        return None
    out = []
    for l in range(len(net.layers)):
        a, offv = FL.bn_affine(net, l)
        a = a.double()
        out.append({"layer": l, "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                    "n_zero_slope": int((a.abs() <= 1e-12).sum()), "n_units": int(a.numel()),
                    "offset_min": float(offv.double().min()), "offset_max": float(offv.double().max())})
    return out


def run_stage(net, X, device, nprobes, block, slices, common_dim):
    """Compute base diagnostics and the matched-probe transition grid for one checkpoint (net already loaded/edited)."""
    theta0 = FL.flat_params(net).clone()
    masks = FL.base_masks(net, X, block=block)
    with torch.no_grad():
        F0 = torch.cat([net(X[i:i + block]) for i in range(0, X.shape[0], block)])
    F0_norm = FL.x_norm(F0)
    hist, zeros, undefined, bins, units = base_diagnostics(net, X, masks, block=block)
    unit_state = [u[1] for u in units]
    L = len(masks)
    radii = list(RADII_V3)
    seeds = list(DIRECTION_SEEDS[:nprobes])
    # per-radius lists over probes; store compact per-layer arrays
    probes = {}
    t0 = time.time()
    for k, ds in enumerate(seeds):
        g = torch.randn(common_dim, generator=torch.Generator().manual_seed(ds), dtype=torch.float64)
        u = matched_direction(net, g, slices).to(device)
        for r in radii:
            m, by_bin, by_state, by_pos = ray(net, theta0, masks, X, u, r, F0, bins, unit_state,
                                              block=block, pos_state=[None] * L, base_Z=None)
            onoff = [int(by_bin[l][1].sum()) for l in range(L)]     # base ON  -> off  (on->off)
            offon = [int(by_bin[l][0].sum()) for l in range(L)]     # base OFF -> on   (off->on)
            dead_react = [int(by_state[l][0]) for l in range(L)]    # flips of always-inactive (dead) units = reactivations
            rec = {"direction_seed": ds, "true": m["true"], "frozen": m["frozen"], "transition": m["transition"],
                   "transition_propagated": m["transition_propagated"], "identity_residual": m["identity_residual"],
                   "flips": m["flips"], "onoff": onoff, "offon": offon, "dead_reactivations": dead_react,
                   "rho_norms": m["rho_norms"], "propagated_norms": m["propagated_norms"],
                   "true_rel": (m["true"] / F0_norm if F0_norm > 0 else None),
                   "transition_rel": (m["transition"] / F0_norm if F0_norm > 0 else None)}
            probes.setdefault(f"{r:g}", []).append(rec)
        if k % 32 == 31:
            print(f"      {k + 1}/{nprobes} probes, {time.time() - t0:.0f}s", flush=True)
    layers = []
    for l in range(L):
        rec, state, _ = units[l]
        layers.append({"layer": l, "n_units": rec["n_units"], "unit_state_counts": rec["unit_state_counts"],
                       "unit_states": ["always_inactive", "always_active", "mixed"],
                       "active_fraction": float(masks[l].double().mean()),
                       "dead_units_always_inactive": int(rec["unit_state_counts"][0]),
                       "exact_zero_preactivations": int(zeros[l]), "undefined_denominators": int(undefined[l])})
    ident_extra = {"param_norm": float(theta0.norm()), "F0_norm_X": F0_norm, "n_gates": [int(m.numel()) for m in masks],
                   "block_norms": {n: float(theta0[a:b].norm()) for n, a, b in slices},
                   "bn_slopes": bn_slope_summary(net),
                   "margin_type": ("raw_post_bn_preactivation" if getattr(net, "batchnorm", False)
                                   else "normalized_Wh_margin"),
                   "max_identity_residual_over_true": max(
                       (p["identity_residual"] / p["true"]) for lst in probes.values() for p in lst if p["true"] > 0),
                   "layers": layers, "seconds": time.time() - t0}
    return radii, seeds, probes, ident_extra


def main():
    ap = argparse.ArgumentParser(description="matched-coordinate BN-vs-plain activation transitions")
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--tags", nargs="+", default=["w64", "bn_w64"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stages", nargs="+", default=["initialization", "late_endpoint"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nprobes", type=int, default=256)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = a.out_dir or OUT
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    for seed in a.seeds:
        for tag in a.tags:
            rd = runs.run_dir(a.dataset, a.model, seed, tag)
            name = runs.run_name(a.dataset, a.model, seed, tag)
            se = stage_epochs(rd)
            sums_path = os.path.join(rd, "checksums.json")
            sums = json.load(open(sums_path)) if os.path.exists(sums_path) else {}
            jpath = os.path.join(out, f"{name}.json")
            res = json.load(open(jpath)) if os.path.exists(jpath) else {
                "protocol": PROTOCOL, "code_revision": git_rev(), "run": rd, "tag": tag,
                "dtype": "float64 analysis representation", "radius_mode": "absolute (Euclidean radius)",
                "radii": list(RADII_V3), "direction_seeds": list(DIRECTION_SEEDS[:a.nprobes]),
                "direction_space": "common W/b coordinate Gaussian; BN gamma/beta held at 0 (eval mode)",
                "tolerances": FL.TOL, "stages": se, "results": {}}
            for stage in a.stages:
                info = se.get(stage)
                if info is None or info.get("checkpoint_epoch") is None:
                    print(f"== {name} {stage}: no checkpoint; skipped", flush=True); continue
                ep = info["checkpoint_epoch"]
                if stage in res["results"] and not a.overwrite:
                    print(f"== {name} {stage}: already present; skipped (use --overwrite)", flush=True); continue
                run_plain = runs.load_run(a.dataset, a.model, seed, tag=tag, epoch=ep, device=a.device)
                net = run_plain["net"]; X = run_plain["data"]["analysis"][0]
                # a plain-model reference net of the SAME seed/stage to certify the common coordinate layout
                run_ref = runs.load_run(a.dataset, a.model, seed, tag="w64", epoch=ep, device="cpu")
                desc = common_coord_descriptor(run_ref["net"], net)
                slices = block_slices(net); common_dim = desc["common_dim"]
                ident = P.analysis_identity(run_plain, rd, ep, {
                    "protocol": PROTOCOL, "nprobes": a.nprobes, "radii": list(RADII_V3),
                    "direction_seeds": list(DIRECTION_SEEDS[:a.nprobes]), "radius_mode": "absolute",
                    "conditional_mask": desc, "tolerances": FL.TOL})
                if sums.get(f"checkpoints/epoch_{ep:04d}.pt") not in (None, ident["checkpoint_sha256"]):
                    raise RuntimeError(f"{name} epoch {ep}: checkpoint hash differs from checksums.json")
                print(f"== {name} {stage} (epoch {ep}): computing {a.nprobes} matched probes x {len(RADII_V3)} radii", flush=True)
                radii, seeds, probes, extra = run_stage(net, X, a.device, a.nprobes, a.block, slices, common_dim)
                rec = {"epoch": ep, "stage": stage, "identity": ident, "conditional_mask": desc,
                       "train_mse": info.get("train_mse"), "train_errors": info.get("train_errors"),
                       "test_acc": info.get("test_acc"), "test_mse": info.get("test_mse"),
                       "provenance_status": "identity verified", "probes": probes, **extra}
                res["results"][stage] = rec
                records.dump_json(jpath, res)
                r1 = probes["1"]
                med = lambda key: float(np.median([q[key] for q in r1]))
                print(f"== {name} {stage} done ({extra['seconds']:.0f}s): r=1 median true {med('true'):.3e} frozen {med('frozen'):.3e} "
                      f"transition {med('transition'):.3e}; max identity/true {extra['max_identity_residual_over_true']:.1e}; "
                      f"flips {np.median([q['flips'] for q in r1], 0).astype(int).tolist()}", flush=True)
                del run_plain, run_ref, net
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
