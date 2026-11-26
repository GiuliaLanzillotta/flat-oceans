"""
Structural (data-independent) symmetries of the trained networks and their relation to the tangent kernel and
the certified branches (EXPERIMENTS_PLAN.md Section 0.4, items 3 and 4).

Counted here: hidden-unit (channel) permutations and, for positively homogeneous activations (ReLU, leaky ReLU),
the positive rescaling of every hidden unit: incoming row and bias times e^t, outgoing column times e^{-t}.  Its
tangent generator at theta_0 is v_{lj} = (+W_l[j,:], +b_l[j], -W_{l+1}[:,j]) (readout column for the last hidden
layer).  For the CNN the unit is a channel: kernel slab and bias of conv l, the input slab of conv l+1 (average
pooling is linear channel by channel, so it commutes with the scaling); for conv5 the outgoing coordinates are
the readout columns of the channel's 16 pooled positions.  Not counted: anything created by the data (dead or
selective units, feature nullity, gate coincidences).  GELU is not positively homogeneous: permutations only,
and the rescaling generators are evaluated to show that they are NOT symmetries there.

Phase `generators` (every run, every stage): the generator matrix G (p x sum_l n_l), its rank at the relative
threshold 1e-9 (expected sum_l n_l), stabilizers (generators with zero incoming and outgoing blocks), the exact
response ||J_X v|| / ||v|| of each generator by forward-mode autodiff on the analysis set against the response of
a random unit direction, and the finite transformation (t = +-1, i.e. factors e and 1/e) verified on the
analysis set and on 512 random inputs.

Phase `compare` (width-64 seed 0, high accuracy, where ker J_X and the certified branch B are stored): with
T = span G, dim T, ||J_X T||, the principal angles between T and C = ker J_X cap B^perp, dim(B + T) against
dim ker J_X, dim(B cap T), and the residual dim ker J_X - dim(B + T).

Phase `rays`: 128 random combinations of the generators, unit norm; straight rays theta_0 + r v (quadratic leak
expected) against the exact curved trajectory of the group with the same log-scales, scaled so that the parameter
displacement is r (no change expected), on the absolute grid.

    python -m flatocean symmetries generators --tags "" leaky gelu w64 --seeds 0 1 2
    python -m flatocean symmetries generators --model cnn --seeds 0 1 2
    python -m flatocean symmetries compare --tag w64 --seeds 0
    python -m flatocean symmetries rays --tag w64 --seeds 0
"""
# Ported unchanged from experiments/exp_symmetries.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean symmetries).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import math
import os
import time

import numpy as np
from flatocean import records, runs
import torch

from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean import tier1_train as T
from flatocean.runs import stage_epochs

OUT = runs.out_dir("symmetries")
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.4 items 3-4 (2026-09-22); identifier symmetries-v1"
RTOL = 1e-9; LADDER = (1e-6, 1e-9, 1e-12)
RADII = tuple(float(f"{v:.3g}") for v in (10.0 ** (k / 2) for k in range(-6, 5)))


def ladder(sv):
    sv = np.asarray(sv); return {f"{t:g}": int((sv > t * sv[0]).sum()) if sv.size and sv[0] > 0 else 0 for t in LADDER}


def orbit_point(theta0, io, direction, r, tol=1e-10):
    """Point of the rescaling orbit with log-scales t * direction whose Euclidean displacement from theta0 is r,
    by bisection on t (the displacement is increasing in t on the branch containing 0); returns (theta, displacement, t)."""
    disp = lambda t: float((transformed(theta0, io, t * direction) - theta0).norm())
    lo, hi = 0.0, 1.0
    while disp(hi) < r:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if disp(mid) < r:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    t = 0.5 * (lo + hi); return transformed(theta0, io, t * direction), disp(t), t


# ----------------------------------------------------------------------------- generators
def unit_generators(net):
    """Rescaling tangent generators as flat parameter vectors (p x n_hidden), with (layer, unit) labels and, for
    each, the index sets of its incoming and outgoing coordinates (for the finite transformation)."""
    sl = {name: (s, e) for name, s, e in FL.param_slices(net)}
    theta = FL.flat_params(net).detach(); p = theta.numel(); gens = []; labels = []; io = []
    if isinstance(net, T.MLP):
        for l, lin in enumerate(net.layers):
            n, m = lin.weight.shape; ws, _ = sl[f"layers.{l}.weight"]; bs, _ = sl[f"layers.{l}.bias"]
            nxt = net.layers[l + 1].weight if l + 1 < len(net.layers) else net.readout.weight
            ns, _ = sl[f"layers.{l + 1}.weight"] if l + 1 < len(net.layers) else sl["readout.weight"]
            for j in range(n):
                inc = np.concatenate([ws + j * m + np.arange(m), [bs + j]])
                out = ns + np.arange(nxt.shape[0]) * nxt.shape[1] + j
                v = torch.zeros(p, dtype=theta.dtype, device=theta.device); v[inc] = theta[inc]; v[out] = -theta[out]
                gens.append(v); labels.append((l + 1, j)); io.append((inc, out))
    else:
        convs = net.convs; d = net.readout.out_features
        for l, conv in enumerate(convs):
            C, cin = conv.out_channels, conv.in_channels; K = 9 * cin
            ws, _ = sl[f"convs.{l}.weight"]; bs, _ = sl[f"convs.{l}.bias"]
            for c in range(C):
                inc = np.concatenate([ws + c * K + np.arange(K), [bs + c]])
                if l + 1 < len(convs):
                    nxt = convs[l + 1]; Cn, cinn = nxt.out_channels, nxt.in_channels; ns, _ = sl[f"convs.{l + 1}.weight"]
                    out = np.concatenate([ns + co * (9 * cinn) + c * 9 + np.arange(9) for co in range(Cn)])
                else:
                    m = net.readout.in_features; per = m // C; ns, _ = sl["readout.weight"]
                    out = np.concatenate([ns + a * m + c * per + np.arange(per) for a in range(d)])
                v = torch.zeros(p, dtype=theta.dtype, device=theta.device); v[inc] = theta[inc]; v[out] = -theta[out]
                gens.append(v); labels.append((l + 1, c)); io.append((inc, out))
    return torch.stack(gens, 1), labels, io


def transformed(theta, io, t):
    """theta after the exact rescaling with log-scales t (one per hidden unit)."""
    th = theta.clone()
    for (inc, out), tj in zip(io, t.tolist()):
        th[inc] *= math.exp(tj); th[out] *= math.exp(-tj)
    return th


@torch.no_grad()
def response(net, X, V, block=2000):
    """||J_X v||_X for the columns v of V (flat directions), by forward-mode autodiff of the true network."""
    params = dict(net.named_parameters()); names = list(params); shapes = [params[n].shape for n in names]
    def split(v):
        out = {}; i = 0
        for n, sh in zip(names, shapes):
            k = int(np.prod(sh)); out[n] = v[i:i + k].reshape(sh); i += k
        return out
    res = []
    for k in range(V.shape[1]):
        tang = split(V[:, k]); acc = 0.0
        for i in range(0, X.shape[0], block):
            xb = X[i:i + block]
            _, jv = torch.func.jvp(lambda pr: torch.func.functional_call(net, pr, (xb,)), (dict(params),), (tang,))
            acc += float(jv.pow(2).sum())
        res.append(math.sqrt(acc / X.shape[0]))
    return np.array(res)


def generators(a):
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for tag in a.tags:
        for seed in a.seeds:
            rd = O.run_dir(a.dataset, a.model, seed, tag); stages = stage_epochs(rd)
            name = f"{a.dataset}_{a.model}{'_' + tag if tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_generators.json")
            res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "stages": {}}
            for stage, info in stages.items():
                ep = info["checkpoint_epoch"]
                if ep is None:
                    continue
                run = O.load_run(a.dataset, a.model, seed, tag=tag, epoch=ep, device="cpu"); net = run["net"]; X = run["data"]["analysis"][0]
                ident_now = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "rtol": RTOL})
                if stage in res["stages"] and res["stages"][stage].get("identity", {}).get("digest") == ident_now["digest"]:
                    continue
                act = getattr(net, "activation", "relu"); homogeneous = act in ("relu", "leaky")
                theta0 = FL.flat_params(net).detach().clone(); t0 = time.time()
                G, labels, io = unit_generators(net)
                widths = [lin.out_features for lin in net.layers] if isinstance(net, T.MLP) else [c.out_channels for c in net.convs]
                perm_log10 = sum(math.lgamma(n + 1) for n in widths) / math.log(10.0)
                sv = torch.linalg.svdvals(G); lad = ladder(sv.numpy()); rank = lad["1e-09"]; gap = float(sv[rank - 1] / max(float(sv[rank]), 1e-300)) if rank < sv.numel() else None
                norms = G.norm(dim=0); stab = [labels[k] for k in range(G.shape[1]) if norms[k] == 0]
                inc_zero = [labels[k] for k, (inc, out) in enumerate(io) if float(theta0[inc].abs().max()) == 0]
                out_zero = [labels[k] for k, (inc, out) in enumerate(io) if float(theta0[out].abs().max()) == 0]
                # exact response of the generators against random directions
                gen = torch.Generator().manual_seed(100); R = torch.randn(theta0.numel(), 8, generator=gen, dtype=torch.float64); R /= R.norm(dim=0)
                r_gen = response(net, X, G / norms.clamp_min(1e-300)); r_rand = response(net, X, R)
                # finite transformation on X and on random inputs
                gen2 = torch.Generator().manual_seed(7); tt = torch.randn(G.shape[1], generator=gen2, dtype=torch.float64).sign()   # factors e and 1/e
                Xr = torch.randn(512, X.shape[1], generator=gen2, dtype=torch.float64) if isinstance(net, T.MLP) else torch.randn(512, *X.shape[1:], generator=gen2, dtype=torch.float64)
                with torch.no_grad():
                    F0 = net(X); F0r = net(Xr); FL.set_flat(net, transformed(theta0, io, tt)); F1 = net(X); F1r = net(Xr); FL.set_flat(net, theta0)
                rec = {"epoch": ep, "activation": act, "positively_homogeneous": homogeneous, "widths": widths,
                       "permutations_log10_count": perm_log10, "n_generators": int(G.shape[1]), "generator_rank": rank, "generator_rank_ladder": lad, "generator_gap": gap,
                       "stabilizers_zero_generator": stab, "units_with_zero_incoming_block": inc_zero, "units_with_zero_outgoing_block": out_zero,
                       "response_generators_max_over_norm": float(r_gen.max()), "response_generators_median": float(np.median(r_gen)),
                       "response_random_unit_direction_median": float(np.median(r_rand)),
                       "finite_transformation_max_abs_change_X": float((F1 - F0).abs().max()), "finite_transformation_max_abs_change_random": float((F1r - F0r).abs().max()),
                       "output_scale_X": float(F0.abs().max()), "param_norm": float(theta0.norm()), "seconds": time.time() - t0,
                       "identity": ident_now}
                res["stages"][stage] = rec
                print(f"== {name} {stage} (epoch {ep}, {act}): {G.shape[1]} generators, rank {rank} (gap {gap}), stabilizers {len(stab)}; "
                      f"||J v||: generators max {r_gen.max():.1e} vs random {np.median(r_rand):.2e}; finite transform max change X {rec['finite_transformation_max_abs_change_X']:.1e} random {rec['finite_transformation_max_abs_change_random']:.1e}; "
                      f"permutations 10^{perm_log10:.0f}", flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


# ----------------------------------------------------------------------------- comparison with ker J_X and the branch
def compare(a):
    from flatocean.analyses import branch as B
    from flatocean.analyses import branch_rays as BR
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            built = json.load(open(os.path.join(B.OUT, f"{name}.json")))["results"][stage]
            trec = json.load(open(os.path.join(B.OUT, f"{name}_tangent.json")))[stage]
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu"); net = run["net"]; X = run["data"]["analysis"][0][: built["identity"]["config"]["samples"]]
            cur = P.analysis_identity(run, rd, ep, built["identity"]["config"])
            assert cur["digest"] == built["identity"]["digest"], "branch record does not match the loaded inputs"
            nt = B.Net(net, X)
            kp = os.path.join(B.OUT, f"{name}_epoch{ep:04d}_kerJX.pt"); kobj = torch.load(kp, weights_only=False); Kx = kobj["basis"]
            run_c = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=("analysis", "comparison"))
            tident = P.analysis_identity(run_c, rd, ep, trec["identity"]["config"]); del run_c
            if tident["digest"] != trec["identity"]["digest"] or kobj["identity_digest"] != trec["identity"]["digest"] or P.sha256_file(kp) != trec.get("kernel_basis_sha256"):
                raise P.ProvenanceError(f"{kp}: tangent basis or record does not match the loaded inputs")
            sv_jx = np.load(os.path.join(B.OUT, f"{name}_epoch{ep:04d}_JX_singular_values.npy")); kx_ladder = {t: int(sv_jx.size) - c for t, c in ladder(sv_jx).items()}
            best = max((c for c, v in built["constructions"].items() if v["certified_branch"] and c not in ("feature_null", "naive_atomic_sum")), key=lambda c: built["constructions"][c]["total"])
            bases = BR.load_basis(os.path.join(B.OUT, f"{name}_epoch{ep:04d}_{best}.pt"), built["constructions"][best], built["identity"]["digest"])
            Bb = torch.block_diag(*[bases[b] for b in nt.blocks]) if all(bases[b].shape[1] for b in nt.blocks) else BR.rays.__globals__["torch"].cat([
                torch.cat([torch.zeros(sum(nt.p(x) for x in nt.blocks[:j]), bases[b].shape[1], dtype=torch.float64), bases[b],
                           torch.zeros(sum(nt.p(x) for x in nt.blocks[j + 1:]), bases[b].shape[1], dtype=torch.float64)], 0) for j, b in enumerate(nt.blocks)], 1)
            G, labels, io = unit_generators(net)
            Tb = torch.stack([BR.to_block_coords(nt, G[:, k]) for k in range(G.shape[1])], 1)
            Ut, svT, _ = torch.linalg.svd(Tb, full_matrices=False); dimT = int((svT > RTOL * svT[0]).sum()); Tq = Ut[:, :dimT]     # left singular vectors: the span at the rank, also under dependence
            # T inside ker J_X?
            resid_K = float(((Tq - Kx @ (Kx.T @ Tq)).norm(dim=0)).max())
            # complement of B in ker J_X
            Cm = Kx - Bb @ (Bb.T @ Kx); Uc, sc, _ = torch.linalg.svd(Cm, full_matrices=False); dimC = int((sc > 1e-8).sum()); Cq = Uc[:, :dimC]
            cos = torch.linalg.svdvals(Cq.T @ Tq).clamp(max=1.0); angles = torch.rad2deg(torch.acos(cos))
            # B + T and B cap T
            svBT = torch.linalg.svdvals(torch.cat([Bb, Tq], 1)); dimBT = int((svBT > RTOL * svBT[0]).sum())
            rec = {"epoch": ep, "construction": best, "dim_B": int(Bb.shape[1]), "dim_kerJX": int(Kx.shape[1]), "dim_kerJX_ladder": kx_ladder, "dim_complement": dimC, "dim_T_sym": dimT, "T_sym_rank_ladder": ladder(svT.numpy()), "n_generators": int(G.shape[1]),
                   "reduced_counts": {"dim_kerJX_minus_T": int(Kx.shape[1]) - dimT, "dim_B_minus_B_cap_T": int(Bb.shape[1]) - (int(Bb.shape[1]) + dimT - dimBT)},
                   "T_outside_kerJX_max_column_residual": resid_K, "response_JX_T_max": float(response(net, X, G / G.norm(dim=0)).max()),
                   "principal_angles_T_vs_complement_deg": {"min": float(angles.min()), "median": float(angles.median()), "max": float(angles.max()),
                                                            "count_below_1deg": int((angles < 1).sum()), "count_above_80deg": int((angles > 80).sum())},
                   "dim_B_plus_T": dimBT, "dim_B_cap_T": int(Bb.shape[1]) + dimT - dimBT, "residual_tangent_dim": int(Kx.shape[1]) - dimBT,
                   "identity": cur}
            os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"{name}_{stage}_compare.json"); records.dump_json(path, rec)
            print(f"== {name} {stage}: dim B {rec['dim_B']}, dim ker J_X {rec['dim_kerJX']}, complement {dimC}, dim T_sym {dimT}; T outside ker J_X {resid_K:.1e}; "
                  f"angles T vs complement min/median/max {angles.min():.2f}/{angles.median():.2f}/{angles.max():.2f} deg; dim(B+T) {dimBT}, dim(B cap T) {rec['dim_B_cap_T']}, residual {rec['residual_tangent_dim']}", flush=True)


# ----------------------------------------------------------------------------- rays
def rays(a):
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_rays.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "radii": list(RADII), "n_directions": a.ndirs, "stages": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu"); net = run["net"]; X = run["data"]["analysis"][0]
            ident_now = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "radii": list(RADII), "ndirs": a.ndirs, "curved_radius": "bisection to the target Euclidean displacement"})
            if stage in res["stages"] and res["stages"][stage].get("identity", {}).get("digest") == ident_now["digest"] and all(len(res["stages"][stage]["curved"][str(r)]) >= a.ndirs for r in RADII):
                continue
            theta0 = FL.flat_params(net).detach().clone(); masks = FL.base_masks(net, X)
            with torch.no_grad():
                F0 = net(X)
            G, labels, io = unit_generators(net); Gn = G / G.norm(dim=0).clamp_min(1e-300)
            rec = {"epoch": ep, "identity": ident_now, "straight": {str(r): [] for r in RADII}, "curved": {str(r): [] for r in RADII}}
            for k in range(a.ndirs):
                g = torch.Generator().manual_seed(100 + k); c = torch.randn(G.shape[1], generator=g, dtype=torch.float64)
                v = G @ c; v = v / v.norm()                                    # unit tangent direction of the group orbit
                cc = c / (G @ c).norm()                                       # log-scales giving displacement ~ r at first order
                for r in RADII:
                    m, _ = FL.measure_ray(net, theta0, masks, X, v, r, F0)
                    rec["straight"][str(r)].append({"true": m["true"], "frozen": m["frozen"], "transition": m["transition"], "transition_propagated": m["transition_propagated"],
                                                    "identity_residual": m["identity_residual"], "flips": m["flips"], "direction_seed": 100 + k, "euclidean_norm": r})
                    # curved: the point of the orbit with log-scales t * cc whose Euclidean displacement is exactly r (bisection on t)
                    th, disp, t_ = orbit_point(theta0, io, cc, r)
                    with torch.no_grad():
                        FL.set_flat(net, th); F1 = net(X); FL.set_flat(net, theta0)
                    rec["curved"][str(r)].append({"true": FL.x_norm(F1 - F0), "displacement": disp, "target_radius": r, "log_scale_parameter": t_, "direction_seed": 100 + k})
                if k % 32 == 31:
                    print(f"   {name} {stage}: {k + 1}/{a.ndirs} directions", flush=True)
            res["stages"][stage] = rec
            med = lambda kind, r, key: float(np.median([q[key] for q in rec[kind][str(r)]]))
            print(f"== {name} {stage} (epoch {ep}): straight true at r=0.01/0.1/1: {med('straight', 0.01, 'true'):.2e}/{med('straight', 0.1, 'true'):.2e}/{med('straight', 1.0, 'true'):.2e}; "
                  f"curved true: {med('curved', 0.01, 'true'):.1e}/{med('curved', 0.1, 'true'):.1e}/{med('curved', 1.0, 'true'):.1e}; curved displacement at r=100: "
                  f"{min(q['displacement'] for q in rec['curved']['100.0']):.4g} to {max(q['displacement'] for q in rec['curved']['100.0']):.4g}", flush=True)
            records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["generators", "compare", "rays"])
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--tag", default=""); ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--stages", nargs="+", default=["high_accuracy"])
    ap.add_argument("--ndirs", type=int, default=128); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.tags is None:
        a.tags = [a.tag]
    {"generators": generators, "compare": compare, "rays": rays}[a.phase](a)


