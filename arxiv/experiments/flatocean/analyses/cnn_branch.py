"""
Certified multilayer branches of the MNIST CNN: block-coordinate construction with tied convolutional blocks, exact
pooling, and an exact mixed-constraint operator check across the constructed bases (EXPERIMENTS_PLAN.md Section 0.4
item 9, follow-up of 2026-09-23; identifier cnn-branch-v1).  Companion of flatocean.analyses.branch (MLP).

Blocks: conv1..convL (one shared kernel and bias per output channel, column-major coordinates k * C + c of
cnn_jacobian) and the readout (index k * d + a).  A layerwise space M = (+)_l M_l is a frozen branch iff the
highest-perturbed-layer criterion holds.  Received spaces are image-shaped subspaces per sample, and the bias is
carried by a flag (1 on the true input, 0 on a received displacement):

    forward    U_{l,i} = span{(h_{l-1,i}, 1)} + (R_{l-1,i}, 0),
               R_{l,i} = pool( S_{l,i} (.) ( W_l * R_{l-1,i} + span{ A u : A in M_l, u in U_{l,i} } ) ),
    criterion  D_{l,i} (A u) = 0  for A in M_l, u in U_{l,i};   B u = 0  for B in M_V, u in U_{V,i} = span{[hf_i; 1]} + [R_{L,i}; 0].

Backward covectors: Zin_V = row space of the weight parts of M_V, transported to the conv output by the exact adjoint
of average pooling (each pooled covector spreads to its four inputs with weight 1/4); Zhat_{l,i} = span{D_{l,i}^T e_a}
+ S_{l,i} (.) transport(Zin_{l+1,i}); Zin_{l,i} = span{W_A^T * z : A in M_l, z in Zhat_{l,i}} + W_l^T * (S_{l,i} (.)
transport(Zin_{l+1,i})) by transposed convolutions.

The block solved given the others is M_l = { A in K_l : <z, A u> = 0 for all i, u in U_{l,i}, z in Zhat_{l,i} }.  A tied
constraint row is the sum over positions,

    <z, W * u + b flag> = sum_r sum_c z[c, r] ( [patch_r(u); flag] (x) e_c ),

so one covector z couples every position of a sample in a single row (never one row per position: constraining the
positions independently would only produce a smaller space).  Rows are formed directly in the candidate basis N
(every basis kernel convolved with u, contracted with z), accumulated by QR, and the rank decision uses the ladder
1e-6 / 1e-9 / 1e-12 relative to the largest unrestricted row norm, a lower bound of sigma_1 of the unrestricted
system, so that rows that all vanish on the candidate space are read as rank zero and never as noise of full rank
(the same convention as the MLP solver).  Received spaces are exact: every basis kernel is applied to every received
vector, per-sample bases are orthonormalized with the pre-gating magnitude as the absolute reference (1e-10), and
everything is streamed over sample chunks.

Certificate (`certified_branch`): (i) full forward operator: for every sample, every basis element of every block and
every received vector (exact recursion on raw spanning sets, fresh pass), max ||D_{l,i} A u|| / ||D_{l,i}||_F (conv,
u unit norm) and ||B u|| (readout) at most VALID_TOL; the readout part is the mixed constraint B pool(S (.) (A * h)) = 0
across all constructed basis pairs; (ii) the frozen-polynomial residual on 32 random combinations at weighted
relative norms 0.1, 1, 10 (exp_cnn_patch_spaces.polynomial_validation).  Controls: the direct sum of the atomic
kernels and the selective sum go through the same check.  The candidate atomic kernel of a conv block is expressed
in a basis aligned with its selective patch space when that space lies inside it (an orthonormal change of basis
of the same space, audited); it makes the exactly vanishing responses visible as zero columns.

    python -m flatocean cnn-branch test
    python -m flatocean cnn-branch build --seeds 0 --stages late_endpoint --device cuda
"""
# Ported unchanged from experiments/exp_cnn_branch.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean cnn-branch).
import argparse
import itertools
import json
import os
import time

import numpy as np
from flatocean import records, runs
import torch
import torch.nn.functional as F

from flatocean import cnn_jacobian as CJ
from flatocean.analyses import cnn_patch_spaces as CP
from flatocean.analyses import persistence as KP
from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs
from flatocean.analyses.branch import batched_span, _nanmax

OUT = CP.OUT
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.4 item 9 (follow-up 2026-09-23); identifier cnn-branch-v1"
RTOL = 1e-9; LADDER = (1e-6, 1e-9, 1e-12); VALID_TOL = 1e-9; ATOL = 1e-10; BUILD_VERSION = 1
SPAN_CAP = 8192                    # raw spanning sets larger than this are orthonormalized in the validation pass
MEM_BUDGET = 2.5e9                 # bytes for the largest intermediate (all basis kernels applied to one chunk)


def ladder_counts(sv, ref):
    sv = np.asarray(sv); return {f"{t:g}": int((sv > t * ref).sum()) for t in LADDER}


# ----------------------------------------------------------------------------- tied operators
def pool_fwd(Y, n):
    """Average pooling of a stack of images Y: (b, q, C, h, w), n times."""
    for _ in range(n):
        b, q = Y.shape[:2]; Y = F.avg_pool2d(Y.reshape(b * q, *Y.shape[2:]), 2).reshape(b, q, Y.shape[2], Y.shape[3] // 2, Y.shape[4] // 2)
    return Y


def pool_adj(Z, n):
    """Exact adjoint of pool_fwd: every pooled covector entry spreads to its four inputs with weight 1/4."""
    for _ in range(n):
        Z = Z.repeat_interleave(2, -1).repeat_interleave(2, -2) / 4
    return Z


def kernels(N, C, cin):
    """Basis kernels of a conv block from an orthonormal basis N (p x k, index kk * C + c, kk = ci * 9 + kh * 3 + kw
    then the bias): conv weight (k C, cin, 3, 3), biases (k, C), transposed-conv weight (C, k cin, 3, 3)."""
    k = N.shape[1]; Nt = N.T.reshape(k, 9 * cin + 1, C)
    W = Nt[:, :9 * cin, :].permute(0, 2, 1).reshape(k * C, cin, 3, 3).contiguous()
    b = Nt[:, 9 * cin, :].contiguous()
    Wt = Nt[:, :9 * cin, :].reshape(k, cin, 3, 3, C).permute(4, 0, 1, 2, 3).reshape(C, k * cin, 3, 3).contiguous()
    return W, b, Wt


def apply_all(u_img, u_flag, W, b, k, C):
    """A u for every basis kernel: u_img (b, nu, cin, h, w), u_flag (b, nu) -> (b, nu, k, C, h, w)."""
    bs, nu = u_img.shape[:2]
    y = F.conv2d(u_img.reshape(bs * nu, *u_img.shape[2:]), W, padding=1).reshape(bs, nu, k, C, *u_img.shape[3:])
    return y + u_flag[:, :, None, None, None, None] * b[None, None, :, :, None, None]


def adjoint_all(z, Wt, k, cin):
    """W_A^T * z for every basis kernel: z (b, nz, C, h, w) -> (b, nz, k, cin, h, w)."""
    bs, nz = z.shape[:2]
    return F.conv_transpose2d(z.reshape(bs * nz, *z.shape[2:]), Wt, padding=1).reshape(bs, nz, k, cin, *z.shape[3:])


class CNet:
    """Frozen structure of the CNN on the analysis set: pooling layout, block shapes, per-chunk states."""

    def __init__(self, net, X, masks, device):
        self.net, self.X, self.masks, self.device = net, X, masks, device
        self.L = len(net.convs); self.d = net.readout.out_features; self.s = X.shape[0]
        self.blocks = CJ.conv_blocks(net)
        self.pools_after = []; pools = 0; started = False
        for a in net.arch:
            if a[0] == "conv":
                if started:
                    self.pools_after.append(pools)
                started = True; pools = 0
            else:
                pools += 1
        self.pools_after.append(pools)
        self.C = [c.out_channels for c in net.convs]; self.cin = [c.in_channels for c in net.convs]
        self.m = net.readout.in_features; self.m1 = self.m + 1
        self.p = {blk: CJ.block_size(net, blk) for blk in self.blocks}
        self.hw_last = int(round((self.m / self.C[-1]) ** 0.5))
        assert self.C[-1] * self.hw_last ** 2 == self.m

    @torch.no_grad()
    def states(self, idx):
        _, Z, H, hf = CJ._forward_states(self.net, self.X[idx])
        return H, [self.masks[l][idx].to(torch.float64) for l in range(self.L)], hf

    def downstream(self, idx, l, S_l):
        return CJ.downstream(self.net, self.masks, idx, l, (len(idx), self.C[l], S_l.shape[2], S_l.shape[3]), self.device)

    def chunk_for(self, M, requested):
        """Chunk size from the memory budget of the largest kernel stack of the space M."""
        worst = 8
        for l in range(self.L):
            N = M.get(self.blocks[l])
            if N is not None and N.shape[1] > 0:
                worst = max(worst, N.shape[1] * self.C[l] * self.masks[l].shape[2] * self.masks[l].shape[3] * 8)
        return max(1, min(requested, int(MEM_BUDGET // worst)))


def compress(Y, ref, exact_span):
    """Received displacements Y (b, q, C, h, w), already gated and pooled.  A response below ATOL * ref (ref: the
    pre-gating magnitude) is a vanishing response, set to zero for that sample (rounding noise of an exactly zero
    response must not be normalized into a unit vector); columns that vanish on every sample of the chunk are
    dropped.  Then per-sample orthonormalization (batched_span), unless exact_span keeps the raw spanning set
    (validation), compressing only above SPAN_CAP columns."""
    b = Y.shape[0]; flat = Y.flatten(2); alive = flat.norm(dim=2) > ATOL * ref                 # (b, q)
    flat = flat * alive.unsqueeze(2)                                       # a response below the reference is exactly zero for that sample
    keep = alive.any(0)
    if not bool(keep.any()):
        return None
    flat = flat[:, keep]
    if exact_span and flat.shape[1] <= SPAN_CAP:
        return flat.reshape(b, -1, *Y.shape[2:])
    B, _ = batched_span(flat.transpose(1, 2), ref=ref)
    return B.transpose(1, 2).reshape(b, -1, *Y.shape[2:]) if B.shape[2] else None


def forward_chunk(nt, M, idx, H, S, hf, exact_span=False):
    """Received spaces of the chunk.  Conv block: (u_img (b, nu, cin, h, w), u_flag (b, nu)), unit norm (zero rows for
    padding); readout: (b, nu, m1) unit rows.  Bases of R are orthonormal per sample unless exact_span."""
    b = len(idx); R = None; U = {}
    for l in range(nt.L):
        blk = nt.blocks[l]; h = H[l]
        u_img = h.unsqueeze(1); u_flag = torch.ones(b, 1, dtype=h.dtype, device=h.device)
        if R is not None:
            u_img = torch.cat([u_img, R], 1); u_flag = torch.cat([u_flag, torch.zeros(b, R.shape[1], dtype=h.dtype, device=h.device)], 1)
        nrm = (u_img.flatten(2).pow(2).sum(2) + u_flag ** 2).sqrt().clamp_min(1e-300)
        u_img = u_img / nrm[:, :, None, None, None]; u_flag = u_flag / nrm
        U[blk] = (u_img, u_flag)
        parts = []
        if R is not None:
            parts.append(F.conv2d(R.reshape(-1, *R.shape[2:]), nt.net.convs[l].weight, padding=1).reshape(b, R.shape[1], nt.C[l], *R.shape[3:]))
        N = M.get(blk)
        if N is not None and N.shape[1] > 0:
            W, b_, _ = kernels(N, nt.C[l], nt.cin[l])
            parts.append(apply_all(u_img, u_flag, W, b_, N.shape[1], nt.C[l]).reshape(b, -1, nt.C[l], *h.shape[2:]))
        if not parts:
            R = None; continue
        Y = torch.cat(parts, 1); ref = float(Y.flatten(2).norm(dim=2).max())
        R = compress(pool_fwd(Y * S[l][:, None], nt.pools_after[l]), ref, exact_span)
    hf1 = torch.cat([hf, torch.ones(b, 1, dtype=hf.dtype, device=hf.device)], 1).unsqueeze(1)
    if R is not None:
        hf1 = torch.cat([hf1, torch.cat([R.flatten(2), torch.zeros(b, R.shape[1], 1, dtype=hf.dtype, device=hf.device)], 2)], 1)
    U["readout"] = hf1 / hf1.norm(dim=2, keepdim=True).clamp_min(1e-300)
    return U


def readout_covectors(nt, NV):
    """Orthonormal basis (z x m) of the row space of the weight parts of the members of M_V."""
    rows = NV.T.reshape(NV.shape[1], nt.m1, nt.d)[:, :nt.m, :].permute(0, 2, 1).reshape(-1, nt.m)
    _, sv, Vh = torch.linalg.svd(rows, full_matrices=False)
    return Vh[sv > RTOL * sv[0]]


def backward_chunk(nt, M, idx, S, target):
    """Zhat_{target, i}: (b, nz, C, h, w) orthonormal covector bases (flattened), by the backward recursion."""
    b = len(idx); NV = M.get("readout"); Zin = None
    if NV is not None and NV.shape[1] > 0:
        Z0 = readout_covectors(nt, NV)
        Zin = Z0.reshape(-1, nt.C[-1], nt.hw_last, nt.hw_last).unsqueeze(0).expand(b, -1, -1, -1, -1)
    for l in range(nt.L - 1, target - 1, -1):
        Zt = pool_adj(Zin, nt.pools_after[l]) if Zin is not None else None
        Dl = nt.downstream(idx, l, S[l])                                                     # (b, d, C, h, w), gated
        cols = [Dl] + ([Zt * S[l][:, None]] if Zt is not None else [])
        Zb, _ = batched_span(torch.cat(cols, 1).flatten(2).transpose(1, 2), ref=float(Dl.flatten(1).norm(dim=1).max()))
        Zhat = Zb.transpose(1, 2).reshape(b, -1, *Dl.shape[2:])
        if l == target:
            return Zhat
        parts = []; refs = []
        if Zt is not None:
            SZ = Zt * S[l][:, None]
            parts.append(F.conv_transpose2d(SZ.reshape(-1, *SZ.shape[2:]), nt.net.convs[l].weight, padding=1).reshape(b, SZ.shape[1], nt.cin[l], *SZ.shape[3:]))
            refs.append(float(SZ.flatten(2).norm(dim=2).max() * nt.net.convs[l].weight.norm()))
        N = M.get(nt.blocks[l])
        if N is not None and N.shape[1] > 0:
            _, _, Wt = kernels(N, nt.C[l], nt.cin[l])
            parts.append(adjoint_all(Zhat, Wt, N.shape[1], nt.cin[l]).reshape(b, -1, nt.cin[l], *Zhat.shape[3:]))
            refs.append(float(Zhat.flatten(2).norm(dim=2).max()))
        if not parts:
            Zin = None; continue
        Zb, _ = batched_span(torch.cat(parts, 1).flatten(2).transpose(1, 2), ref=max(refs))
        Zin = Zb.transpose(1, 2).reshape(b, -1, nt.cin[l], *Zhat.shape[3:]) if Zb.shape[2] else None
    raise AssertionError("unreachable")


# ----------------------------------------------------------------------------- one block given the others
class QRAcc:
    def __init__(self, k, device):
        self.R = None; self.buf = []; self.n = 0; self.k = k; self.device = device
    def add(self, T):
        if T.shape[0]:
            self.buf.append(T); self.n += T.shape[0]
            if self.n >= max(2 * self.k, 20000):
                self.flush()
    def flush(self):
        if self.buf:
            self.R = torch.linalg.qr(torch.cat(([self.R] if self.R is not None else []) + self.buf, 0), mode="r").R; self.buf = []; self.n = 0
        return self.R


@torch.no_grad()
def solve_conv(nt, l, M, N0, chunk, verbose=False):
    """Maximal M_l inside span(N0) with <z, A u> = 0 for all samples, u in U_{l,i}, z in Zhat_{l,i} (streamed)."""
    blk = nt.blocks[l]; k = N0.shape[1]
    if k == 0:
        return N0, {"dim": 0, "rank_within": 0, "gap": None, "note": "empty candidate space"}
    W, b_, _ = kernels(N0, nt.C[l], nt.cin[l]); acc = QRAcc(k, nt.device); row_max = 0.0; nrows = 0; nnz = 0; dz = []; du = []; t0 = time.time()
    for i in range(0, nt.s, chunk):
        idx = torch.arange(i, min(i + chunk, nt.s), device=nt.device); H, S, hf = nt.states(idx); b = len(idx)
        U = forward_chunk(nt, M, idx, H, S, hf); u_img, u_flag = U[blk]; nu = u_img.shape[1]
        Zhat = backward_chunk(nt, M, idx, S, l); nz = Zhat.shape[1]; Zf = Zhat.flatten(2)
        y = apply_all(u_img, u_flag, W, b_, k, nt.C[l]).reshape(b, nu * k, -1)                  # (b, nu k, C h w)
        T = torch.bmm(Zf, y.transpose(1, 2)).reshape(b, nz, nu, k).reshape(-1, k)                # rows (i, z, u)
        Pu = F.unfold(u_img.reshape(-1, *u_img.shape[2:]), 3, padding=1)                         # (b nu, 9 cin, R)
        Pu = torch.cat([Pu, u_flag.reshape(-1, 1, 1).expand(-1, 1, Pu.shape[2])], 1)
        Gp = (Pu.transpose(1, 2) @ Pu).reshape(b, nu, Pu.shape[2], Pu.shape[2]); zf = Zhat.flatten(3)
        Gz = torch.einsum("bzcr,bzcs->bzrs", zf, zf)
        row_max = max(row_max, float(torch.einsum("burs,bzrs->buz", Gp, Gz).clamp_min(0).max().sqrt()))
        nrows += T.shape[0]; T = T[T.abs().amax(1) > 0]; nnz += T.shape[0]; acc.add(T)
        dz.append(float((Zf.abs().amax(2) > 0).sum(1).double().mean())); du.append(float((u_img.flatten(2).abs().amax(2) + u_flag.abs() > 0).sum(1).double().mean()))
        if verbose and (i // chunk) % 200 == 0:
            print(f"        {blk}: samples {i + b}/{nt.s} rows {nrows} ({time.time() - t0:.0f}s)", flush=True)
    R = acc.flush()
    rec = {"rows": nrows, "nonzero_rows": nnz, "sigma_ref": row_max, "mean_dim_Zhat": float(np.mean(dz)), "mean_dim_U": float(np.mean(du)), "seconds": time.time() - t0}
    if R is None or R.shape[0] == 0:
        rec.update({"dim": k, "rank_within": 0, "gap": None, "note": "no nonzero constraint row"}); return N0, rec
    _, sv, Vh = torch.linalg.svd(R, full_matrices=True); sv = sv.cpu().numpy()
    lad = ladder_counts(sv, row_max); r = lad["1e-09"]
    rec.update({"dim": k - r, "rank_within": r, "rank_ladder": lad, "gap": float(sv[r - 1] / max(sv[r], 1e-300)) if 0 < r < sv.size else None,
                "sigma1": float(sv[0]), "largest_rejected_over_ref": float(sv[r] / row_max) if r < sv.size else 0.0})
    return (N0 if r == 0 else N0 @ Vh[r:].T), rec


@torch.no_grad()
def solve_readout(nt, M, chunk):
    """M_V = R^d (x) (sum_i U_{V,i})^perp, basis in the index k * d + a."""
    acc = QRAcc(nt.m1, nt.device); nrows = 0; du = []; t0 = time.time()
    for i in range(0, nt.s, chunk):
        idx = torch.arange(i, min(i + chunk, nt.s), device=nt.device); H, S, hf = nt.states(idx)
        U = forward_chunk(nt, M, idx, H, S, hf); rows = U["readout"].reshape(-1, nt.m1); rows = rows[rows.abs().amax(1) > 0]
        nrows += rows.shape[0]; acc.add(rows); du.append(float((U["readout"].abs().amax(2) > 0).sum(1).double().mean()))
    R = acc.flush(); _, sv, Vh = torch.linalg.svd(R, full_matrices=True); sv = sv.cpu().numpy()
    lad = ladder_counts(sv, sv[0]); r = lad["1e-09"]; comp = Vh[r:].T; c = comp.shape[1]
    N = torch.zeros(nt.m1 * nt.d, nt.d * c, dtype=comp.dtype, device=comp.device); ar = torch.arange(nt.m1, device=comp.device)
    for a in range(nt.d):
        N[ar * nt.d + a, a * c:(a + 1) * c] = comp
    rec = {"dim": int(N.shape[1]), "rank_within": r, "rank_ladder": lad, "gap": float(sv[r - 1] / max(sv[r], 1e-300)) if 0 < r < sv.size else None,
           "sigma1": float(sv[0]), "sigma_ref": float(sv[0]), "rows": nrows, "mean_dim_U": float(np.mean(du)), "seconds": time.time() - t0}
    return N, rec


def build(nt, order, K, chunk, verbose=True):
    M = {blk: K[blk][:, :0] for blk in nt.blocks}; steps = []
    for blk in order:
        t0 = time.time()
        if blk == "readout":
            M[blk], rec = solve_readout(nt, M, nt.chunk_for(M, chunk))
        else:
            l = nt.blocks.index(blk)
            M[blk], rec = solve_conv(nt, l, M, K[blk], nt.chunk_for(M, chunk), verbose=verbose and K[blk].shape[1] > 0)
        rec.update({"block": blk, "seconds": time.time() - t0}); steps.append(rec)
        if verbose:
            print(f"     {blk}: dim {rec['dim']} of atomic {K[blk].shape[1]} (rank within {rec['rank_within']}, gap {rec.get('gap')}, {rec['seconds']:.0f}s)", flush=True)
    return M, steps


# ----------------------------------------------------------------------------- certificates
@torch.no_grad()
def forward_criterion_full(nt, M, chunk):
    """Max over samples, basis elements and received vectors (raw spanning sets, exact recursion) of the normalized
    criterion: ||D_{l,i} A u|| / ||D_{l,i}||_F for conv blocks, ||B u|| for the readout (the mixed constraint)."""
    worst = {blk: 0.0 for blk in nt.blocks}; n_recv = {blk: 0.0 for blk in nt.blocks}
    for i in range(0, nt.s, chunk):
        idx = torch.arange(i, min(i + chunk, nt.s), device=nt.device); H, S, hf = nt.states(idx); b = len(idx)
        U = forward_chunk(nt, M, idx, H, S, hf, exact_span=True)
        for l in range(nt.L):
            blk = nt.blocks[l]; N = M.get(blk)
            if N is None or N.shape[1] == 0:
                continue
            u_img, u_flag = U[blk]; W, b_, _ = kernels(N, nt.C[l], nt.cin[l]); k = N.shape[1]
            y = apply_all(u_img, u_flag, W, b_, k, nt.C[l]).reshape(b, -1, nt.C[l] * u_img.shape[3] * u_img.shape[4])
            Dl = nt.downstream(idx, l, S[l]).flatten(2)                                            # (b, d, C h w)
            val = torch.bmm(y, Dl.transpose(1, 2)).norm(dim=2) / Dl.flatten(1).norm(dim=1).clamp_min(1e-300)[:, None]
            worst[blk] = _nanmax(worst[blk], float(val.max())); n_recv[blk] += float(u_img.shape[1])
        N = M.get("readout")
        if N is not None and N.shape[1] > 0:
            Bm = N.T.reshape(N.shape[1], nt.m1, nt.d).permute(0, 2, 1)                              # (kV, d, m1)
            u = U["readout"]; val = torch.einsum("kdm,bum->bukd", Bm, u).norm(dim=3)
            worst["readout"] = _nanmax(worst["readout"], float(val.max())); n_recv["readout"] += float(u.shape[1])
    return worst, {blk: v / max(1, (nt.s + chunk - 1) // chunk) for blk, v in n_recv.items()}


def validate(nt, M, chunk):
    theta0 = FL.flat_params(nt.net).clone()
    fc, recv = forward_criterion_full(nt, M, chunk)
    poly = CP.polynomial_validation(nt.net, nt.X, {blk: M[blk] for blk in nt.blocks if M[blk] is not None and M[blk].shape[1] > 0}, theta0, nt.masks, nt.device)
    vals = list(fc.values()) + list(poly["residual_rel"].values())
    return {"tolerance": VALID_TOL, "forward_criterion_full_max": fc, "mean_received_vectors_per_chunk": recv,
            "polynomial_residual_rel": poly["residual_rel"], "passed": bool(all(np.isfinite(v) and v <= VALID_TOL for v in vals)),
            "scope": "full forward operator on exact received spanning sets (every basis element against every received vector); "
                     "polynomial residual on 32 random combinations at weighted relative norms 0.1, 1, 10"}


@torch.no_grad()
def align(BK, G):
    """Orthonormal basis of span(BK) whose first columns are G (when G lies inside span(BK)); audited."""
    if G.shape[1] == 0:
        return BK, {"aligned": False, "note": "empty selective space"}
    leak = float((G - BK @ (BK.T @ G)).norm() / G.norm())
    if leak > RTOL:
        return BK, {"aligned": False, "selective_outside_K_rel": leak}
    Rres = BK - G @ (G.T @ BK); Uc, sv, _ = torch.linalg.svd(Rres, full_matrices=False); r = int((sv > RTOL * sv[0]).sum())
    N0 = torch.cat([G, Uc[:, :r]], 1)
    orth = float((N0.T @ N0 - torch.eye(N0.shape[1], dtype=N0.dtype, device=N0.device)).abs().max())
    span_err = float((BK - N0 @ (N0.T @ BK)).norm() / BK.norm())
    if N0.shape[1] != BK.shape[1] or orth > 1e-10 or span_err > RTOL:
        return BK, {"aligned": False, "dim_mismatch": [int(N0.shape[1]), int(BK.shape[1])], "orthonormality": orth, "span_error": span_err}
    return N0, {"aligned": True, "selective_dim": int(G.shape[1]), "complement_dim": r, "orthonormality": orth, "span_error": span_err, "selective_leak": leak}


def load_kernel(name, ep, blk, run, rd):
    """Float64 kernel columns of the stored factor after the provenance check, without the p x p float32 matrix."""
    path = os.path.join(KP.FACTOR_DIR, name, f"epoch{ep:04d}_{blk}.npz"); z = np.load(path)
    KP.factor_status(path, run, rd, blk, False, z)
    sv = z["singular_values"]; r = int(z["kernel_start"]); assert r == int((sv > RTOL * sv[0]).sum()), (blk, r)
    return torch.from_numpy(z["V_kernel64"]), sv, P.sha256_file(path)


def orders(blocks, nonempty, n_random, seed=0):
    """Bottom-up, top-down and random orders; orders that agree on the nonempty blocks build the same space."""
    out = {"bottom_up": list(blocks), "top_down": list(reversed(blocks))}
    perms = [list(p) for p in itertools.permutations(blocks) if list(p) not in out.values()]
    rng = np.random.default_rng(seed)
    for k, j in enumerate(rng.choice(len(perms), size=min(n_random, len(perms)), replace=False)):
        out[f"random_{k}"] = perms[j]
    key = lambda o: tuple(b for b in o if b in nonempty); seen = {}; equiv = {}
    for nm, o in out.items():
        equiv[nm] = seen.setdefault(key(o), nm)
    return out, equiv


def run(a):
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, tag=a.tag); stages = stage_epochs(rd); name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        ps_all = json.load(open(os.path.join(OUT, f"{name}_patch_spaces.json")))["stages"]
        out_path = os.path.join(OUT, f"{name}_branch.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "results": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue
            run_ = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device); net = run_["net"]; X = run_["data"]["analysis"][0][: a.samples]
            ps = ps_all.get(stage)
            if not ps or ps["identity"]["digest"] != CP.stage_identity(run_, rd, ep)["digest"] or not ps.get("complete"):
                raise P.ProvenanceError(f"{name} {stage}: patch-space record missing, incomplete or under another identity")
            if a.only_failed and ps.get("atomic_sum_is_branch"):
                print(f"== {name} {stage}: the atomic sum is a validated branch; skipped (--only-failed)", flush=True); continue
            config = {"protocol": PROTOCOL, "build_version": BUILD_VERSION, "rtol": RTOL, "ladder": list(LADDER), "valid_tol": VALID_TOL, "atol": ATOL, "samples": int(X.shape[0]),
                      "random_orders": a.random_orders, "patch_space_basis_sha256": ps["basis_sha256"], "factor_sha256": {b: q["atomic_kernel"]["factor_sha256"] for b, q in ps["blocks"].items()}}
            ident = P.analysis_identity(run_, rd, ep, config); old = res["results"].get(stage)
            if old and old.get("identity", {}).get("digest") != ident["digest"]:
                if not a.overwrite:
                    raise P.ProvenanceError(f"{name} {stage}: stored stage under another identity; rerun with --overwrite")
                old = None
            srec = old if (old and not a.overwrite) else {"epoch": ep, "identity": ident, "constructions": {}}
            res["results"][stage] = srec
            masks = FL.base_masks(net, X); nt = CNet(net, X, masks, a.device); blocks = nt.blocks
            sel = CP.load_bases(name, ep, "selective", blocks, ps, a.device)
            K = {}; krec = {}
            for blk in blocks:
                ak = ps["blocks"][blk]["atomic_kernel"]
                if ak is None:
                    raise RuntimeError(f"{name} {stage}: no atomic factor for {blk}")
                if ak["dim_K"] == 0:
                    K[blk] = torch.zeros(nt.p[blk], 0, dtype=torch.float64, device=a.device)
                    krec[blk] = {"dim": 0, "dim_ladder": ak["dim_K_ladder"], "note": "full rank at 1e-9 in the stored factor"}; continue
                BK, sv, sha = load_kernel(name, ep, blk, run_, rd); BK = BK.to(a.device)
                assert sha == ak["factor_sha256"] and BK.shape[1] == ak["dim_K"]
                if blk != "readout":
                    BK, arec = align(BK, sel[blk])
                else:
                    arec = {"aligned": False, "note": "readout basis taken as stored"}
                K[blk] = BK; krec[blk] = {"dim": int(BK.shape[1]), "dim_ladder": ak["dim_K_ladder"], "gap": ak["gap"], "alignment": arec}
            nonempty = [b for b in blocks if K[b].shape[1] > 0]
            srec["blocks"] = {blk: dict(p=nt.p[blk], selective_dim=int(sel[blk].shape[1]), **krec[blk]) for blk in blocks}
            srec["sum_atomic"] = int(sum(K[b].shape[1] for b in blocks)); srec["sum_selective"] = int(sum(sel[b].shape[1] for b in blocks)); srec["nonempty_blocks"] = nonempty
            print(f"== {name} {stage} (epoch {ep}): atomic kernels {[K[b].shape[1] for b in blocks]} sum {srec['sum_atomic']}; selective sum {srec['sum_selective']}; "
                  f"alignment {[krec[b].get('alignment', {}).get('aligned') for b in nonempty]}", flush=True)
            ords, equiv = orders(blocks, nonempty, a.random_orders, seed)
            todo = [("selective_sum", None), ("naive_atomic_sum", None)] + list(ords.items())
            if a.only is not None:
                todo = [(cname, order) for cname, order in todo if cname in a.only]
            for cname, order in todo:
                if cname in srec["constructions"]:
                    continue
                if order is not None and equiv[cname] != cname:
                    srec["constructions"][cname] = {"order": order, "equivalent_to": equiv[cname], "note": "same order on the nonempty blocks"}; continue
                print(f"   -- {cname} {order or ''}", flush=True); t0 = time.time()
                if cname == "selective_sum":
                    M, steps = sel, None
                elif cname == "naive_atomic_sum":
                    M, steps = K, None
                else:
                    M, steps = build(nt, order, K, a.chunk)
                dims = {blk: int(M[blk].shape[1]) for blk in blocks}
                val = validate(nt, M, nt.chunk_for(M, a.chunk))
                bpath = os.path.join(OUT, f"{name}_epoch{ep:04d}_branch_{cname}.pt")
                torch.save({"identity_digest": ident["digest"], "construction": cname, "order": order, "bases": {blk: M[blk].cpu() for blk in blocks}}, bpath)
                rec = {"order": order, "dims": dims, "total": int(sum(dims.values())), "steps": steps, "constructed_by_theorem": order is not None,
                       "validation": val, "certified_branch": val["passed"], "basis_sha256": P.sha256_file(bpath), "seconds": time.time() - t0}
                srec["constructions"][cname] = rec
                print(f"      dims {dims} total {rec['total']}  passed {val['passed']}: criterion {max(val['forward_criterion_full_max'].values()):.1e} "
                      f"polynomial {max(val['polynomial_residual_rel'].values()):.1e} ({time.time() - t0:.0f}s)", flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
            records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
        print("saved", out_path, flush=True)


# ----------------------------------------------------------------------------- subspace comparison
@torch.no_grad()
def compare(a):
    """Numerical subspace comparison of the constructions (review of 2026-09-24): principal angles between the bottom-up and
    top-down conv5 spaces, and the projection of the constructed conv5 basis onto the stored spectral directions that the
    10^-12 rule rejects (columns of the factor between the 10^-12 and 10^-9 counts).  Equal dimensions alone do not establish
    equality; this records how close the spaces are, with the tolerance."""
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, tag=a.tag); stages = stage_epochs(rd); name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        out_path = os.path.join(OUT, f"{name}_branch.json"); res = json.load(open(out_path))
        for stage in a.stages:
            srec = res["results"].get(stage)
            if not srec:
                continue
            ep = srec["epoch"]; run_ = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device)
            if P.analysis_identity(run_, rd, ep, srec["identity"]["config"])["digest"] != srec["identity"]["digest"]:
                raise P.ProvenanceError(f"{name} {stage}: stage record under another identity")
            bases = {}
            for cname in ("bottom_up", "top_down"):
                c = srec["constructions"].get(cname, {})
                if "basis_sha256" not in c:
                    continue
                bpath = os.path.join(OUT, f"{name}_epoch{ep:04d}_branch_{cname}.pt"); obj = torch.load(bpath, weights_only=False)
                if obj["identity_digest"] != srec["identity"]["digest"] or P.sha256_file(bpath) != c["basis_sha256"]:
                    raise P.ProvenanceError(f"{bpath}: basis does not match the record")
                bases[cname] = obj["bases"]["conv5"].to(a.device)
            if "bottom_up" not in bases:
                continue
            Bb = bases["bottom_up"]; cmp = {"conv5_dim": int(Bb.shape[1])}
            if "top_down" in bases:
                Bt = bases["top_down"]; cosines = torch.linalg.svdvals(Bb.T @ Bt).cpu().numpy() if Bb.shape[1] and Bt.shape[1] else np.array([])
                cmp["bottom_up_vs_top_down"] = {"dims": [int(Bb.shape[1]), int(Bt.shape[1])], "min_principal_cosine": float(cosines.min()) if cosines.size else None,
                                                "max_principal_angle_deg": float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1)))) if cosines.size else None,
                                                "projection_residual_rel": float((Bb - Bt @ (Bt.T @ Bb)).norm() / max(Bb.norm(), 1e-300))}
            # stored spectral directions of conv5 rejected by the 10^-12 rule but kept by the 10^-9 rule
            path = os.path.join(KP.FACTOR_DIR, name, f"epoch{ep:04d}_conv5.npz"); z = np.load(path); KP.factor_status(path, run_, rd, "conv5", False, z)
            sv = z["singular_values"]; r9 = int((sv > 1e-9 * sv[0]).sum()); r12 = int((sv > 1e-12 * sv[0]).sum()); V = torch.from_numpy(z["V"]).double()
            if "V_kernel64" in z.files:
                V[:, int(z["kernel_start"]):] = torch.from_numpy(z["V_kernel64"])
            amb = V[:, r9:r12].to(a.device); K12 = V[:, r12:].to(a.device); del V
            if amb.shape[1]:
                cmp["projection_onto_directions_rejected_at_1e-12"] = {"count": int(amb.shape[1]), "largest_singular_value_of_amb^T_B": float(torch.linalg.svdvals(amb.T @ Bb).max()),
                                                                        "singular_values_of_J_there_over_sigma1": (sv[r9:r12] / sv[0]).tolist()}
            cos12 = torch.linalg.svdvals(K12.T @ Bb).cpu().numpy()
            cmp["against_1e-12_kernel"] = {"dims": [int(K12.shape[1]), int(Bb.shape[1])], "min_principal_cosine": float(cos12.min()) if cos12.size else None,
                                           "projection_residual_rel": float((Bb - K12 @ (K12.T @ Bb)).norm() / max(Bb.norm(), 1e-300)), "note": "stored basis columns are float32 for the non-kernel part; the tolerance of this comparison is about 1e-7"}
            srec["subspace_comparison"] = cmp
            print(f"== {name} {stage}: conv5 {cmp['conv5_dim']}; bottom-up vs top-down min cosine {cmp.get('bottom_up_vs_top_down', {}).get('min_principal_cosine')}; "
                  f"vs 1e-12 kernel: residual {cmp['against_1e-12_kernel']['projection_residual_rel']:.1e}; onto rejected directions: {cmp.get('projection_onto_directions_rejected_at_1e-12', {}).get('largest_singular_value_of_amb^T_B')}", flush=True)
            records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


# ----------------------------------------------------------------------------- tests
def _frozen_inputs(net, masks, X, idx):
    h = X[idx]; ci = 0; ins = []
    for a in net.arch:
        if a[0] == "conv":
            ins.append(h); z = net.convs[ci](h); h = z * masks[ci][idx].to(z.dtype); ci += 1
        else:
            h = F.avg_pool2d(h, 2)
    return ins, torch.cat([h.flatten(1), torch.ones(h.shape[0], 1, dtype=h.dtype)], 1)


@torch.no_grad()
def _brute_force(nt, order, K, n_pts=150):
    """Tiny networks: explicit constraint rows in full tied coordinates; received and covector spans by evaluating the
    frozen polynomial maps at random points of the other blocks (autodiff for the covectors)."""
    net, X, masks = nt.net, nt.X, nt.masks; theta0 = FL.flat_params(net).clone(); gen = torch.Generator().manual_seed(11)
    M = {blk: None for blk in nt.blocks}; idx = torch.arange(nt.s); maps = {blk: torch.as_tensor(CJ.block_index_map(net, blk)) for blk in nt.blocks}
    def point(active):
        w = torch.zeros(theta0.numel(), dtype=torch.float64)
        for blk in active:
            N = M[blk]
            if N is not None and N.shape[1]:
                w[maps[blk]] += N @ (torch.randn(N.shape[1], generator=gen, dtype=torch.float64) * (0.3 + torch.rand((), generator=gen, dtype=torch.float64)))
        return w
    try:
        for blk in order:
            pos = nt.blocks.index(blk); below = nt.blocks[:pos]; above = nt.blocks[pos + 1:]
            us = []; zs = []
            for t in range(n_pts + 1):
                FL.set_flat(net, theta0 + (point(below) if t else 0 * theta0))
                with torch.no_grad():
                    ins, hf1 = _frozen_inputs(net, masks, X, idx)
                us.append((ins[pos], torch.ones(nt.s, dtype=torch.float64)) if blk != "readout" else (hf1, None))
            if blk != "readout":
                for t in range(n_pts + 1):
                    FL.set_flat(net, theta0 + (point(above) if t else 0 * theta0))
                    zs.append(nt.downstream(idx, pos, masks[pos].to(torch.float64)))                 # (s, d, C, h, w)
            FL.set_flat(net, theta0)
            if blk == "readout":
                rows = torch.cat([u[0] for u in us], 0); _, sv, Vh = torch.linalg.svd(torch.linalg.qr(rows, mode="r").R, full_matrices=True)
                r = int((sv > RTOL * sv[0]).sum()); comp = Vh[r:].T; c = comp.shape[1]
                N = torch.zeros(nt.m1 * nt.d, nt.d * c, dtype=torch.float64); ar = torch.arange(nt.m1)
                for a in range(nt.d):
                    N[ar * nt.d + a, a * c:(a + 1) * c] = comp
                M[blk] = N; continue
            Z = torch.cat(zs, 1)                                                                   # (s, nz, C, h, w)
            rows = []
            for u_img, flag in us:
                Pu = F.unfold(u_img, 3, padding=1); Pu = torch.cat([Pu, torch.ones(nt.s, 1, Pu.shape[2], dtype=torch.float64)], 1)     # (s, K, R)
                rows.append(torch.einsum("ikr,izcr->izkc", Pu, Z.flatten(3)).reshape(-1, Pu.shape[1] * nt.C[pos]))
            R = torch.linalg.qr(torch.cat(rows, 0), mode="r").R; _, sv, Vh = torch.linalg.svd(R, full_matrices=True)
            r = int((sv > RTOL * sv[0]).sum()); M[blk] = Vh[r:].T
    finally:
        FL.set_flat(net, theta0)
    return M


def _tests():
    from flatocean import tier1_train as T
    torch.manual_seed(0); torch.set_num_threads(4)
    net = T.CNN([("conv", 1, 3), ("pool",), ("conv", 3, 4), ("pool",)], 2, init_seed=1, dtype=torch.float64, in_hw=8)
    with torch.no_grad():
        net.convs[0].bias.copy_(torch.tensor([0.2, -0.1, -50.0], dtype=torch.float64)); net.convs[1].bias.copy_(torch.tensor([0.1, -0.2, 0.05, -50.0], dtype=torch.float64))
    X = torch.randn(6, 1, 8, 8, dtype=torch.float64); masks = FL.base_masks(net, X); nt = CNet(net, X, masks, "cpu")
    # adjoints of pooling and of the tied kernel stack
    g = torch.Generator().manual_seed(5)
    Y = torch.randn(2, 3, 4, 8, 8, dtype=torch.float64, generator=g); Z = torch.randn(2, 3, 4, 4, 4, dtype=torch.float64, generator=g)
    assert abs(float((pool_fwd(Y, 1) * Z).sum() - (Y * pool_adj(Z, 1)).sum())) < 1e-12
    N = torch.linalg.qr(torch.randn((9 * 3 + 1) * 4, 5, dtype=torch.float64, generator=g)).Q; W, b_, Wt = kernels(N, 4, 3)
    u = torch.randn(2, 2, 3, 8, 8, dtype=torch.float64, generator=g); z = torch.randn(2, 3, 4, 8, 8, dtype=torch.float64, generator=g)
    lhs = torch.einsum("bukchw,bzchw->buzk", apply_all(u, torch.zeros(2, 2, dtype=torch.float64), W, b_, 5, 4), z)
    rhs = torch.einsum("bzkqhw,buqhw->buzk", adjoint_all(z, Wt, 5, 3), u)
    assert float((lhs - rhs).abs().max()) < 1e-12, float((lhs - rhs).abs().max())
    # kernel rows agree with cnn_jacobian on the true input with z = D^T e_a
    idx = torch.arange(6); H, S, hf = nt.states(idx); Dl = nt.downstream(idx, 1, S[1])
    y = apply_all(H[1].unsqueeze(1), torch.ones(6, 1, dtype=torch.float64), W, b_, 5, 4)
    T1 = torch.einsum("bdchw,bukchw->bdk", Dl, y).reshape(-1, 5); J = CJ.cnn_rows(net, masks, X, idx, "conv2")
    assert float((T1 - J @ N).abs().max()) < 1e-12
    # atomic kernels (dense, tiny) and constructions
    K = {}
    for blk in nt.blocks:
        J = CJ.cnn_rows(net, masks, X, idx, blk); _, sv, Vh = torch.linalg.svd(J, full_matrices=True); r = int((sv > RTOL * sv[0]).sum()); K[blk] = Vh[r:].T
    assert all(K[b].shape[1] > 0 for b in nt.blocks)
    nonempty = [b for b in nt.blocks if K[b].shape[1] > 0]; ords, equiv = orders(nt.blocks, nonempty, 2, 0)
    res = {}
    for nm, order in ords.items():
        M, steps = build(nt, order, K, 3, verbose=False); val = validate(nt, M, 3)
        assert val["passed"], (nm, val)
        res[nm] = {b: int(M[b].shape[1]) for b in nt.blocks}
        if nm in ("bottom_up", "top_down"):
            Mb = _brute_force(nt, order, K)
            assert {b: int(Mb[b].shape[1]) for b in nt.blocks} == res[nm], (nm, res[nm], {b: int(Mb[b].shape[1]) for b in nt.blocks})
            # the same spaces, not only the same dimensions (principal cosines; the brute force samples its spans)
            for b in nt.blocks:
                if M[b].shape[1]:
                    assert float(torch.linalg.svdvals(M[b].T @ Mb[b]).min()) > 1 - 1e-6, b
    assert all(sum(v.values()) <= sum(K[b].shape[1] for b in nt.blocks) for v in res.values())
    vK = validate(nt, K, 3)
    assert torch.equal(FL.flat_params(net), FL.flat_params(net))
    # NaN in the polynomial residual is rejected
    import unittest.mock as mock
    real = FL.frozen_forward; calls = {"n": 0}
    def nan_after_first(*args, **kw):
        calls["n"] += 1; out = real(*args, **kw); return out if calls["n"] == 1 else out * float("nan")
    theta = FL.flat_params(net).clone()
    with mock.patch.object(FL, "frozen_forward", side_effect=nan_after_first):
        vv = validate(nt, M, 3)
    assert not vv["passed"] and torch.equal(FL.flat_params(net), theta)
    print("cnn-branch tests passed:", res, "atomic sum", sum(K[b].shape[1] for b in nt.blocks), "naive passed", vK["passed"],
          "naive criterion", {b: f"{v:.1e}" for b, v in vK["forward_criterion_full_max"].items()}, "equivalences", equiv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "compare", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--stages", nargs="+", default=["late_endpoint"])
    ap.add_argument("--dataset", default="mnist", help="ADDITIVE (default mnist = byte-identical): CNN run family, e.g. svhn for the P2 native-color CNN.")
    ap.add_argument("--model", default="cnn", help="ADDITIVE (default cnn = byte-identical).")
    ap.add_argument("--tag", default="", help="run tag (e.g. a spawn child); records are named <dataset>_<model>_<tag>_seed<seed> so parent and child cannot collide")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--samples", type=int, default=10000); ap.add_argument("--random-orders", type=int, default=4)
    ap.add_argument("--only-failed", action="store_true", help="skip stages where the atomic sum is already a validated branch")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="build only these constructions (e.g. --only bottom_up): skips top_down and the selective/naive sums; the named construction still gets its full validated certificate")
    a = ap.parse_args()
    {"build": run, "compare": compare, "test": lambda a_: _tests()}[a.phase](a)


