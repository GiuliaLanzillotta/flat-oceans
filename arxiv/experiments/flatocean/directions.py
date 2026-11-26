"""Probes, scaling protocols and the ray evaluation of the activation-transition residual.

Probe directions are matched across checkpoints, activations and architectures (Gaussian ambient directions from
DIRECTION_SEEDS).  Scaling protocols (frozen_lib.scale_direction): block-balanced (every nonzero block to alpha ||theta_l||),
weighted (whole direction by the weighted relative norm, for coupled tangent directions) and absolute (Euclidean radius).
`ray` evaluates one probe at one radius: true, frozen and transition displacement, the identity residual, gate flips by margin
bin and unit state, per-layer departures and propagated contributions (and, for smooth activations, preactivation
displacement and curvature with a validity flag).  Grids: RADII (v2 absolute), ALPHAS (block-relative), RADII_V3 (absolute).
"""
import os

import numpy as np
import torch

from flatocean import runs

from flatocean import frozen_lib as FL
from flatocean.frozen_lib import block_slices, scale_direction  # noqa: F401  (re-exported)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_V2 = runs.out_dir("transition_v2")
OUT_REL = runs.out_dir("transition_rel")
OUT_V3 = runs.out_dir("transition_v3")
PROTOCOL_V2 = "memos/2026-09-10_protocol_stages_atomic_transition.md, Section 4 (amendment 2026-09-10); identifier transition-residual-v2"
PROTOCOL_REL = "EXPERIMENTS_PLAN.md Section 0.2 item 2 (amendment 2026-09-17); identifier transition-residual-v3-relative"
PROTOCOL_V3 = "EXPERIMENTS_PLAN.md Section 0.4 items 5 to 7 (2026-09-22); identifier transition-residual-v3"
RADII = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0)
ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0)
DIRECTION_SEEDS = tuple(range(100, 356))          # matched across checkpoints, activations and architectures; 32 or 256 used
RADII_V3 = tuple(float(f"{v:.3g}") for v in (10.0 ** (k / 2) for k in range(-6, 5)))   # 1e-3 ... 100, half-decades
NBINS = 160                       # log10 |margin| in [-15, 1), 0.1 decade each; bin 0 also holds |m| < 1e-15
STATE_NAMES = ("always_inactive", "always_active", "mixed")



def magnitude_bin(m_abs):
    """0.1-decade bin index of |m|; values below 1e-15 (including exact zero) fall in bin 0."""
    lg = torch.log10(m_abs.clamp_min(1e-15))
    return ((lg + 15.0) * 10.0).floor().clamp(0, NBINS - 1).to(torch.int16)


@torch.no_grad()
def base_diagnostics(net, X, masks, block=1000):
    """Per layer: signed-margin histograms, zero/undefined counts, per-gate sign and bin
    (kept on the device for the conditional counts), and unit states."""
    L = len(net.gated_layers); dev = X.device
    hist = [torch.zeros(2, NBINS, dtype=torch.int64, device=dev) for _ in range(L)]
    zeros = [0] * L; undefined = [0] * L
    bins = [[] for _ in range(L)]
    for i in range(0, X.shape[0], block):
        _, Z, H = net.forward(X[i:i + block], return_states=True)
        for l, z in enumerate(Z):
            scale = net.margin_scale(l, H[l]).expand_as(z)
            und = scale == 0
            m = torch.where(und, torch.zeros_like(z), z / torch.where(und, torch.ones_like(scale), scale))
            zeros[l] += int((z == 0).sum()); undefined[l] += int(und.sum())
            b = magnitude_bin(m.abs()); s = (z > 0)
            idx = s.to(torch.int64).reshape(-1) * NBINS + b.to(torch.int64).reshape(-1)
            hist[l] += torch.bincount(idx, minlength=2 * NBINS).reshape(2, NBINS)
            bins[l].append(b)
    bins = [torch.cat(b) for b in bins]
    units = []
    for l in range(L):
        S = masks[l]                                  # s x n (x H x W)
        # unit = channel (dim 1) over all examples and positions
        red = S.reshape(S.shape[0], S.shape[1], -1)
        any_on = red.any(dim=(0, 2)); all_on = red.all(dim=(0, 2))
        state = torch.full((S.shape[1],), 2, dtype=torch.int8, device=dev)   # mixed
        state[~any_on] = 0; state[all_on] = 1
        rec = {"n_units": int(S.shape[1]), "unit_state_counts": [int((state == k).sum()) for k in range(3)],
               "gates_per_state": [int(red[:, state == k, :].numel()) for k in range(3)]}
        pos_state = None
        if S.dim() == 4:                              # CNN: channel-position coordinates over examples
            flat = S.reshape(S.shape[0], -1)
            any_e = flat.any(0); all_e = flat.all(0)
            rec["channel_position"] = {"n": int(flat.shape[1]), "always_inactive": int((~any_e).sum()),
                                       "always_active": int(all_e.sum()), "mixed": int((any_e & ~all_e).sum())}
            pos_state = torch.full((flat.shape[1],), 2, dtype=torch.int8, device=dev)
            pos_state[~any_e] = 0; pos_state[all_e] = 1
            pos_state = pos_state.reshape(S.shape[1:])
            rec["gates_per_position_state"] = [int((pos_state == k).sum()) * int(S.shape[0]) for k in range(3)]
        units.append((rec, state, pos_state))
    return hist, zeros, undefined, bins, units


@torch.no_grad()
def ray(net, theta0, masks, X, u, r, F0, bins, unit_state, block=1000, pos_state=None, base_Z=None):
    """One probe at one radius: drift norms, flips, conditional flip counts, and the per-layer decomposition
    of the propagated transition residual: departure magnitudes ||rho_l||_X (local activation residuals),
    the propagated contributions ||D^post_l rho_l||_X (downstream visibility), and their sum of norms against
    the norm of the sum (cancellation).  For CNNs, flips are also counted by channel-position state."""
    FL.set_flat(net, theta0 + r * u)
    L = len(net.gated_layers); s = X.shape[0]
    dtrue = torch.zeros_like(F0); dfrz = torch.zeros_like(F0); prop = torch.zeros_like(F0)
    contrib = [torch.zeros_like(F0) for _ in range(L)]; rho_sq = [0.0] * L; disp_sq = [0.0] * L
    flips = [0] * L
    by_bin = [torch.zeros(2, NBINS, dtype=torch.int64, device=X.device) for _ in range(L)]
    by_state = [torch.zeros(3, dtype=torch.int64, device=X.device) for _ in range(L)]
    by_pos = [torch.zeros(3, dtype=torch.int64, device=X.device) for _ in range(L)]
    for i in range(0, s, block):
        idx = torch.arange(i, min(i + block, s), device=X.device)
        f_frz = FL.frozen_forward(net, masks, X, idx)
        f_true, Z = FL.true_states(net, X, idx)
        p = torch.zeros_like(f_true)
        if base_Z is not None:                                                  # preactivation displacement per layer (smooth activations)
            for l, z in enumerate(Z):
                disp_sq[l] += float((z - base_Z[l][idx]).double().pow(2).sum())
        for l, z in enumerate(Z):
            S = masks[l][idx]
            fl = (z > 0) != S
            nfl = int(fl.sum()); flips[l] += nfl
            rho = FL.local_residual(net, masks, l, idx, z)
            if float(rho.abs().max()) > 0:
                c = FL.frozen_tail_linear(net, masks, idx, l, rho)
                p = p + c; contrib[l][idx] = c; rho_sq[l] += float(rho.double().pow(2).sum())
                b = bins[l][idx][fl].to(torch.int64); sg = S[fl].to(torch.int64)
                by_bin[l] += torch.bincount(sg * NBINS + b, minlength=2 * NBINS).reshape(2, NBINS)
                st = unit_state[l].view(1, -1, *([1] * (z.dim() - 2))).expand_as(z)[fl].to(torch.int64)
                by_state[l] += torch.bincount(st, minlength=3)
                if pos_state is not None and pos_state[l] is not None:
                    ps = pos_state[l].unsqueeze(0).expand_as(z)[fl].to(torch.int64)
                    by_pos[l] += torch.bincount(ps, minlength=3)
        dtrue[idx] = f_true - F0[idx]; dfrz[idx] = f_frz - F0[idx]; prop[idx] = p
    FL.set_flat(net, theta0)
    atr = dtrue - dfrz
    contrib_norms = [FL.x_norm(c) for c in contrib]
    return {"true": FL.x_norm(dtrue), "frozen": FL.x_norm(dfrz), "transition": FL.x_norm(atr),
            "transition_propagated": FL.x_norm(prop), "identity_residual": FL.x_norm(atr - prop), "flips": flips,
            "rho_norms": [float(np.sqrt(q / s)) for q in rho_sq], "propagated_norms": contrib_norms,
            "propagated_sum_of_norms": float(sum(contrib_norms)),
            **({"preactivation_displacement": [float(np.sqrt(q / s)) for q in disp_sq],
                # curvature is defined only where the layer's preactivations move; an unperturbed upstream layer has zero
                # displacement and its (rounding-level) residual has no curvature meaning: null, with the validity flag
                "curvature": [float(np.sqrt(rq / s) / np.sqrt(dq / s) ** 2) if dq > 0 and np.sqrt(dq / s) > 1e-12 * max(1.0, float(F0.norm())) else None for rq, dq in zip(rho_sq, disp_sq)],
                "curvature_defined": [bool(dq > 0 and np.sqrt(dq / s) > 1e-12 * max(1.0, float(F0.norm()))) for dq in disp_sq]} if base_Z is not None else {})}, \
        [b.cpu().numpy() for b in by_bin], [b.cpu().numpy() for b in by_state], [b.cpu().numpy() for b in by_pos]


def relative_direction(g, theta0, slices, perturb):
    """u with u_l = ||theta_l|| g_l / ||g_l|| on the perturbed blocks and zero elsewhere."""
    u = torch.zeros_like(g)
    for name, a, b in slices:
        if perturb in ("all", name):
            u[a:b] = g[a:b] * (theta0[a:b].norm() / g[a:b].norm())
    if not u.any():
        raise ValueError(f"unknown block {perturb}; blocks are {[n for n, _, _ in slices]}")
    return u


def absolute_direction(g, slices, perturb):
    """Unit-norm direction supported on the perturbed blocks (all, or one block), for the absolute grid ||w|| = r."""
    u = torch.zeros_like(g)
    for name, a, b in slices:
        if perturb in ("all", name):
            u[a:b] = g[a:b]
    if not u.any():
        raise ValueError(f"unknown block {perturb}; blocks are {[n for n, _, _ in slices]}")
    return u / u.norm()



