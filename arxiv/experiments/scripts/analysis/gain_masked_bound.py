#!/usr/bin/env python
"""POINTWISE masked-gain bound vs the measured drift for the gain-arm bottom_up branches (Task E, 2026-10-02).

For each gain arm (mnist_mlp_gain_d0, gain_d1), seed and probe direction, at a set of absolute radii, compute the
POINTWISE masked-gain bound on the activation-transition (gate-flip) term

    ||A_tr(w)||_X = ||sum_l D^post_l(w) rho_l(w)||_X  <=  sum_l ||D^post_l(w) rho_l(w)||_X  (propagated sum of norms)
                                                       <=  sum_l g_mask_l(w) * omega_l(w)      (masked-gain bound)
                                                       <=  sum_l plain_l(w) * omega_l(w)      (plain weight-norm bound)

with, at the EDITED downstream weights theta0 + r w and the BASE gates (masks):
    g_mask_l(w) = max_i ||D^post_{l,i}(w)||_op      (pointwise masked downstream gain; D folds no BN here, plain MLP),
    plain_l(w)  = ||V(w)||_op * prod_{k>l} ||W_k(w)||_op   (gates dropped: |S|<=1),
    omega_l(w)  = ||rho_l(w)||_X                    (per-layer local activation residual, READ from the rays record).

This is a valid POINTWISE bound (edited weights, base gates), distinct from the theorem's supremum over a parameter
ball.  Predictions stay CONDITIONAL: a tighter bound is NOT a smaller drift.

The measured side (transition, transition_propagated, propagated_sum_of_norms, propagated_norms, rho_norms=omega) is
READ from the already-delivered rays records (results/branches/<name>_rays.json), matched per direction_seed; only the
new g_mask / plain gain factors are computed here.  The probe directions are reconstructed bit-identically to
flatocean.analyses.branch_rays.rays (same seed, to_block_coords, block-diagonal basis projection, embed, absolute
scaling); the reconstruction is asserted against the stored euclidean_norm and block_fractions before any bound.

Usage: OMP_NUM_THREADS=8 python scripts/analysis/gain_masked_bound.py [--tags gain_d0 gain_d1] [--seeds 0 1 2]
                                                                       [--radii 0.1 1.0 3.16] [--selftest]
Output: results/gain_maskedbound/mnist_mlp_<tag>_seed<seed>_maskedbound.json  (one file per arm/seed).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # experiments/ on the path
import numpy as np
import torch

from flatocean import ocean_lib as O
from flatocean import frozen_lib as FL
from flatocean import provenance as P
import flatocean.analyses.branch as B
import flatocean.analyses.branch_rays as BR

OUT_BRANCH = B.OUT                                   # results/branches
OUT = os.path.join(os.path.dirname(OUT_BRANCH), "gain_maskedbound")
STAGE = "late_endpoint"
PROTOCOL = "Task E (2026-10-02): pointwise masked-gain bound (edited downstream weights, base gates) vs measured drift"
ATOL_RECON = 1e-9                                    # reconstruction must match the stored probe to this


def full_blockdiag(nt, bases):
    """Block-diagonal embedding of the per-block branch bases, exactly as branch_rays.rays builds it."""
    if all(bases[b].shape[1] for b in nt.blocks):
        return torch.block_diag(*[bases[b] for b in nt.blocks])
    return torch.cat([torch.cat([torch.zeros(sum(nt.p(x) for x in nt.blocks[:j]), bases[b].shape[1], dtype=torch.float64),
                                 bases[b],
                                 torch.zeros(sum(nt.p(x) for x in nt.blocks[j + 1:]), bases[b].shape[1], dtype=torch.float64)], 0)
                      for j, b in enumerate(nt.blocks)], 1)


@torch.no_grad()
def gain_factors(net, X, masks, theta0, r, w, omega, chunk=5000):
    """At the edited weights theta0 + r w and the base masks: g_mask_l = max_i ||D^post_{l,i}||_op and the plain
    weight-norm factor plain_l = ||V||_op * prod_{k>l} ||W_k||_op, for every gated layer l.  D^post_{l,i} is the frozen
    tail map (branch_rays/frozen_lib convention): v -> readout( prod_{k>l} S_k W_k v ), gates held at the base masks.
    The per-sample spectral norm is taken from the small d x d Gram eigenvalue (identical to svdvals to ~1e-14).
    g_mask_l is computed only where omega_l>0 (elsewhere the bound term g_mask_l*omega_l is exactly 0); the last gated
    layer's tail is V itself, so g_mask_{L-1}=||V||_op with no per-sample loop."""
    FL.set_flat(net, theta0 + r * w)
    try:
        L = len(net.layers); Ws = [lin.weight.detach() for lin in net.layers]; V = net.readout.weight.detach()
        assert getattr(net, "bn", None) is None, "gain arm is a plain MLP; BN tail slope not handled here"
        s = X.shape[0]
        sn = [float(torch.linalg.svdvals(Wk)[0]) for Wk in Ws]; snV = float(torch.linalg.svdvals(V)[0])
        plain = [snV * float(np.prod([sn[k] for k in range(l + 1, L)])) for l in range(L)]
        gm = [0.0] * L
        for l in range(L):
            if omega[l] <= 0:                                                              # term g_mask_l*omega_l is 0: skip the tail
                continue
            if l == L - 1:                                                                 # tail is V: spectral norm is ||V||_op
                gm[l] = snV; continue
            nl = Ws[l].shape[0]; mx = 0.0
            for i in range(0, s, chunk):
                idx = slice(i, min(i + chunk, s)); b = min(chunk, s - i)
                T = torch.eye(nl, dtype=torch.float64).unsqueeze(0).expand(b, -1, -1).contiguous()   # b x nl x nl (post-l perturbation basis)
                for k in range(l + 1, L):
                    T = torch.einsum("ab,sbc->sac", Ws[k], T)                              # W_k
                    T = masks[k][idx][:, :, None] * T                                      # base gate S_k
                D = torch.einsum("ab,sbc->sac", V, T)                                      # readout -> b x d x nl
                ev = torch.linalg.eigvalsh(D @ D.transpose(1, 2))[:, -1].clamp_min(0.0)    # largest singular value^2 (d x d Gram)
                mx = max(mx, float(ev.max()) ** 0.5)
            gm[l] = mx
    finally:
        FL.set_flat(net, theta0)
    return gm, plain


def process(tag, seed, radii, verbose=True):
    name = f"mnist_mlp_{tag}_seed{seed}"
    build = json.load(open(os.path.join(OUT_BRANCH, f"{name}.json")))["results"][STAGE]
    rays = json.load(open(os.path.join(OUT_BRANCH, f"{name}_rays.json")))["results"][STAGE]
    ep = build["epoch"]; bcfg = build["identity"]["config"]; digest = build["identity"]["digest"]
    cons = build["constructions"]; bu = cons["bottom_up"]
    run = O.load_run("mnist", "mlp", seed, tag=tag, epoch=ep, device="cpu")
    X = run["data"]["analysis"][0][: bcfg["samples"]]; net = run["net"]
    nt = B.Net(net, X); theta0 = FL.flat_params(net).clone(); sl = FL.block_slices(net); masks = FL.base_masks(net, X)
    bases = BR.load_basis(os.path.join(OUT_BRANCH, f"{name}_epoch{ep:04d}_bottom_up.pt"), bu, digest)
    basis = full_blockdiag(nt, bases)
    rset = rays["sets"]["branch:bottom_up"]["absolute"]
    by_seed = {}                                                                           # direction_seed -> {radius -> stored probe}
    for rk, lst in rset.items():
        for q in lst:
            by_seed.setdefault(q["direction_seed"], {})[rk] = q
    results = {r: {"transition": [], "propagated_sum_of_norms": [], "masked_bound": [], "plain_bound": [],
                   "transition_propagated": [], "ratio_prop_over_transition": [], "ratio_masked_over_prop": [],
                   "ratio_plain_over_masked": [], "g_mask": [], "omega": [], "plain": []} for r in radii}
    seeds_used = sorted(by_seed)
    t0 = time.time()
    for ds in seeds_used:
        g = torch.Generator().manual_seed(ds); probe = torch.randn(theta0.numel(), generator=g, dtype=torch.float64)
        v = BR.to_block_coords(nt, probe); v = basis @ (basis.T @ v)
        w0 = B.embed(net, nt, BR.split_blocks(nt, v)); w, info = FL.scale_direction(w0, theta0, sl, "absolute")
        for r in radii:
            rk = str(r); q = by_seed[ds].get(rk)
            if q is None:
                raise RuntimeError(f"{name}: radius {rk} missing for direction_seed {ds} in the rays record")
            # consistency: the reconstructed direction must be the one the stored ray used
            assert abs(float(r * w.norm()) - q["euclidean_norm"]) <= ATOL_RECON * max(1.0, q["euclidean_norm"]), \
                (name, ds, r, float(r * w.norm()), q["euclidean_norm"])
            for k, bf in info["unit_step_block_fractions"].items():
                assert abs(float(r * bf) - q["block_fractions"][k]) <= ATOL_RECON * max(1.0, abs(q["block_fractions"][k])), (name, ds, r, k)
            omega = q["rho_norms"]                                                          # per-layer ||rho_l||_X at this radius
            gm, plain = gain_factors(net, X, masks, theta0, r, w, omega)
            # the pointwise masked-gain bound uses the SAME per-layer omega the ray measured
            assert len(omega) == len(gm) == len(plain)
            mb = float(sum(gm[l] * omega[l] for l in range(len(gm))))
            pb = float(sum(plain[l] * omega[l] for l in range(len(plain))))
            tr = q["transition"]; psum = q["propagated_sum_of_norms"]; tp = q["transition_propagated"]
            # the bound is an upper bound on the transition term: assert the chain holds per probe (tolerant to fp noise)
            rel = 1e-6 * max(1.0, mb)
            assert tr <= psum + rel and psum <= mb + rel and mb <= pb + rel, \
                (name, ds, r, "ordering", tr, psum, mb, pb)
            res = results[r]
            res["transition"].append(tr); res["propagated_sum_of_norms"].append(psum); res["transition_propagated"].append(tp)
            res["masked_bound"].append(mb); res["plain_bound"].append(pb)
            res["ratio_prop_over_transition"].append(psum / tr if tr > 0 else None)
            res["ratio_masked_over_prop"].append(mb / psum if psum > 0 else None)
            res["ratio_plain_over_masked"].append(pb / mb if mb > 0 else None)
            res["g_mask"].append(gm); res["omega"].append(omega); res["plain"].append(plain)
        if verbose and ds == seeds_used[0]:
            print(f"   {name}: {len(seeds_used)} probes, radii {radii}; first probe r={radii[-1]} "
                  f"masked_bound {results[radii[-1]]['masked_bound'][0]:.3f} vs transition "
                  f"{results[radii[-1]]['transition'][0]:.3e}", flush=True)

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None

    summary = {}
    for r in radii:
        res = results[r]
        gm_arr = np.array(res["g_mask"]); om_arr = np.array(res["omega"]); pl_arr = np.array(res["plain"])
        summary[str(r)] = {
            "n_probes": len(res["transition"]),
            "measured_transition_median": med(res["transition"]),
            "measured_transition_propagated_median": med(res["transition_propagated"]),
            "measured_propagated_sum_of_norms_median": med(res["propagated_sum_of_norms"]),
            "masked_gain_bound_median": med(res["masked_bound"]),
            "plain_weightnorm_bound_median": med(res["plain_bound"]),
            "ratio_prop_over_transition_median": med(res["ratio_prop_over_transition"]),
            "ratio_masked_bound_over_prop_sum_median": med(res["ratio_masked_over_prop"]),
            "ratio_plain_over_masked_median": med(res["ratio_plain_over_masked"]),
            "masked_bound_over_transition_median": med([m / t for m, t in zip(res["masked_bound"], res["transition"]) if t > 0]),
            "g_mask_per_layer_median": [med(gm_arr[:, l].tolist()) for l in range(gm_arr.shape[1])],
            "omega_per_layer_median": [med(om_arr[:, l].tolist()) for l in range(om_arr.shape[1])],
            "plain_per_layer_median": [med(pl_arr[:, l].tolist()) for l in range(pl_arr.shape[1])],
        }
    out = {"protocol": PROTOCOL, "name": name, "epoch": ep, "radii": list(radii),
           "build_identity_digest": digest, "bottom_up_basis_sha256": bu["basis_sha256"],
           "bottom_up_certified": bool(bu["certified_branch"]), "branch_dim": bu["total"],
           "block_dims": bu["dims"], "n_probes": len(seeds_used), "seconds": time.time() - t0,
           "note": "CONDITIONAL: a tighter bound is not a smaller drift. masked-gain bound uses EDITED downstream weights "
                   "+ BASE gates (pointwise), NOT the parameter-ball supremum. Measured drift/residuals READ from the "
                   "rays record; omega=rho_norms. g_mask_l=max_i ||D^post_{l,i}||_op.",
           "summary": summary,
           "per_probe": {str(r): {"direction_seed": seeds_used,
                                  "transition": results[r]["transition"], "propagated_sum_of_norms": results[r]["propagated_sum_of_norms"],
                                  "masked_bound": results[r]["masked_bound"], "plain_bound": results[r]["plain_bound"],
                                  "g_mask": results[r]["g_mask"], "omega": results[r]["omega"]} for r in radii}}
    os.makedirs(OUT, exist_ok=True)
    op = os.path.join(OUT, f"{name}_maskedbound.json")
    with open(op + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(op + ".tmp", op)
    print(f"saved {op}  ({out['seconds']:.0f}s)", flush=True)
    for r in radii:
        sm = summary[str(r)]
        print(f"   r={r}: transition {sm['measured_transition_median']:.3e}  prop_sum {sm['measured_propagated_sum_of_norms_median']:.3e}  "
              f"masked {sm['masked_gain_bound_median']:.3e}  plain {sm['plain_weightnorm_bound_median']:.3e}  "
              f"(masked/prop {sm['ratio_masked_bound_over_prop_sum_median']:.2f}, plain/masked {sm['ratio_plain_over_masked_median']:.2f})", flush=True)
    return out


def selftest():
    """Reconstruction + bound-ordering on the first two probes of gain_d0 seed 0 at r=1 (cheap)."""
    out = process("gain_d0", 0, (1.0,), verbose=False)
    sm = out["summary"]["1.0"]
    assert sm["measured_transition_median"] <= sm["measured_propagated_sum_of_norms_median"] <= sm["masked_gain_bound_median"] <= sm["plain_weightnorm_bound_median"]
    assert sm["masked_bound_over_transition_median"] > 1.0                                  # a bound, not the drift
    print("selftest passed:", {k: round(v, 4) if isinstance(v, float) else v for k, v in sm.items() if "median" in k and v is not None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["gain_d0", "gain_d1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--radii", type=float, nargs="+", default=[0.1, 1.0, 3.16])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.selftest:
        selftest(); return
    for tag in a.tags:
        for seed in a.seeds:
            process(tag, seed, list(a.radii))


if __name__ == "__main__":
    main()
