"""
Certified multilayer (layerwise) frozen branches on a Tier 1 MLP (Claim E; plan amendment 2026-09-17, item 5).

A layerwise space M = M_1 (+) ... (+) M_L (+) M_V, with M_l a subspace of the block [W_l b_l], is a frozen branch
iff the highest-perturbed-layer criterion holds (theory file, Theorem 4.7, written here with biases through the
augmented input [h; 1]):

    forward    R_0 = 0,   U_l,i = span{[h_{l-1,i}; 1]} + [R_{l-1,i}; 0],
               R_l,i = S_l,i ( W_l R_{l-1,i} + M_l U_l,i ),
    criterion  D_l,i A u = 0  for A in M_l, u in U_l,i  (hidden blocks),   B u = 0  for B in M_V, u in U_{L+1,i}.

Construction by block-coordinate maximization.  Every output monomial contains at most one factor from a given
block, so with the other blocks fixed the criterion is linear in the block being solved.  Collect the covectors
that the blocks above impose on a displacement of the input of layer k (backward recursion):

    Zin_{L+1} = row space of the weight part of M_V,
    Zhat_k,i  = span{D_k,i^T e_a} + S_k,i Zin_{k+1,i},
    Zin_k,i   = span{ A[:, :m]^T z : A in M_k, z in Zhat_k,i } + W_k^T S_k,i Zin_{k+1,i}.

The maximal admissible block j given the others is  M_j = { A : z^T A u = 0 for all i, u in U_j,i, z in Zhat_j,i }.
Solving the blocks in the order (1, ..., L, V) with empty blocks above is the bottom-up procedure of the theory
file; any other order (top-down, random permutations) yields a branch certified by the same theorem.  Every
finished branch is re-checked by (i) the forward criterion, computed independently of the backward recursion, and
(ii) the frozen-polynomial residual on random linear combinations at large relative norm.

Spans of bilinear images (M_l U_l,i and M_k^T Zhat_k,i) are computed per sample from 2n random rank-one samples
A^(r) u^(r); random rank-one tensors span M (x) U, so the sampled range equals the full range with probability one.
The constraint system of a block is reduced by QR accumulation restricted to the block's atomic kernel (which
contains every admissible M_j), first on a subset of samples to shrink the candidate space, then on all samples,
where the rank decision (relative threshold 1e-9, gap recorded) is taken.

Scope of the numerical certificate.  A construction is a branch by the theorem when its blocks solve the systems
above exactly.  Numerically, the received spaces come from a randomized range finder and the rank decisions use a
relative threshold, so every construction (and the direct sum of the atomic kernels) is validated afterwards and
the record stores the outcome: `certified_branch` is true only if (i) the forward criterion holds for EVERY basis
element of every block against EVERY received basis vector of every sample (full operator, max normalized entry
at most VALID_TOL), (ii) the sampled forward criterion and (iii) the frozen-polynomial residual on random
combinations are finite and at most VALID_TOL.  The received spaces used in (i) are recomputed by the forward
recursion with fresh random samples.  `constructed_by_theorem` records how the space was built; it never implies
validation.  Downstream scripts select only validated branches.

Controls: the feature-null branch (A [h;1] = 0, R = 0), and the direct sum of the atomic kernels, which is not a
branch (its frozen residual is reported).

    python -m flatocean multilayer-branch test
    python -m flatocean multilayer-branch build --tag w64 --seeds 0 --stages high_accuracy --device cpu
"""
# Ported unchanged from experiments/exp_multilayer_branch.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean multilayer-branch).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import itertools
import json
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

OUT = runs.out_dir("branches")
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.2 item 5 (amendment 2026-09-17); identifier multilayer-branch-v1"
RTOL = 1e-9
VALID_TOL = 1e-9                   # acceptance threshold of the validation residuals (relative, see validate)
BUILD_VERSION = 2
DIRECTION_SEEDS = tuple(range(100, 132))


# ----------------------------------------------------------------- robust SVD: gesdd non-convergence fallback (2026-10-02)
# torch's batched divide-and-conquer SVD (gesdd) can raise torch._C._LinAlgError on an ill-conditioned / repeated-
# singular-value element (observed in batched_span via backward_spaces on a P1 subsample build).  Same fallback class as
# frozen_lib / persistence_siblings._svd_vh: on non-convergence fall back to the QR-based driver (scipy gesvd), and as a
# LAST resort to the Gram eigendecomposition (sqrt of the eigenvalues FLOORS tight singular values, so it is flagged in
# SVD_SOLVER_COUNTS).  ADDITIVE and default-safe: when the fast path converges these return exactly torch.linalg.svd's
# factors, so every caller is byte-identical on the converging path.
SVD_SOLVER_COUNTS = {}


def _bump_solver(name):
    SVD_SOLVER_COUNTS[name] = SVD_SOLVER_COUNTS.get(name, 0) + 1


def _robust_svd_Usv_2d(Mi, k):
    """U[:, :k], sv[:k] of one 2D matrix, robust to gesdd non-convergence (gesdd -> scipy gesvd -> Gram)."""
    try:
        Ui, si, _ = torch.linalg.svd(Mi, full_matrices=False); _bump_solver("gesdd")
        return Ui[:, :k], si[:k]
    except torch._C._LinAlgError:
        try:
            import scipy.linalg as sla
            Ui_np, si_np, _ = sla.svd(Mi.detach().cpu().numpy(), full_matrices=False, lapack_driver="gesvd"); _bump_solver("gesvd")
            return torch.from_numpy(Ui_np).to(Mi.dtype)[:, :k], torch.from_numpy(si_np).to(Mi.dtype)[:k]
        except Exception:
            w, V = torch.linalg.eigh(Mi.T @ Mi)                         # Gram last resort (floors tiny sv; flagged)
            si = torch.sqrt(torch.clamp(w, min=0.0)).flip(0)[:k]; Vf = V.flip(1)[:, :k]
            Ui = (Mi @ Vf) / si.clamp_min(1e-300).unsqueeze(0); _bump_solver("gram")
            Ui, _ = torch.linalg.qr(Ui)  # M V / sigma drifts from orthonormal (~5e-9) at tiny sigma; span unchanged
            return Ui, si


def _robust_batched_svd_Usv(Mat):
    """U (b x n x k), sv (b x k), k=min(n,c), of a batched stack, robust to gesdd non-convergence on individual batch
    elements.  Fast path: the batched torch.linalg.svd (identical output when it converges).  On failure: per element
    gesdd -> scipy gesvd -> Gram."""
    try:
        U, sv, _ = torch.linalg.svd(Mat, full_matrices=False); _bump_solver("gesdd"); return U, sv
    except torch._C._LinAlgError:
        b, n, c = Mat.shape; k = min(n, c)
        U = Mat.new_zeros(b, n, k); sv = Mat.new_zeros(b, k)
        for i in range(b):
            Ui, si = _robust_svd_Usv_2d(Mat[i], k); U[i] = Ui; sv[i, :si.shape[0]] = si
        return U, sv


def _robust_svd_svVh(X, full_matrices):
    """sv, Vh of one 2D matrix, robust to gesdd non-convergence (gesdd -> scipy gesvd -> Gram); honours full_matrices
    for the complement-basis callers (solve_readout, feature_null, kernel_within)."""
    try:
        _, sv, Vh = torch.linalg.svd(X, full_matrices=full_matrices); _bump_solver("gesdd"); return sv, Vh
    except torch._C._LinAlgError:
        try:
            import scipy.linalg as sla
            _, sv_np, Vh_np = sla.svd(X.detach().cpu().numpy(), full_matrices=full_matrices, lapack_driver="gesvd"); _bump_solver("gesvd")
            return torch.from_numpy(sv_np).to(X.dtype), torch.from_numpy(Vh_np).to(X.dtype)
        except Exception:
            w, V = torch.linalg.eigh(X.T @ X)
            sv = torch.sqrt(torch.clamp(w, min=0.0)).flip(0); _bump_solver("gram"); return sv, V.flip(1).transpose(-2, -1)


# ----------------------------------------------------------------------------- batched subspace helpers
def batched_span(Mat, ref=None, rtol=RTOL, atol=1e-10):
    """Mat: s x n x c.  Orthonormal bases of the column spaces, zero-padded: s x n x min(n, c); dims: s.
    A singular value counts when it exceeds rtol times the sample's largest one and atol times the reference
    magnitude `ref` of the quantities the columns were formed from (before gating), so that columns that vanish
    identically (rounding noise of size 1e-16 ref) span nothing, in one sample or in all of them."""
    if Mat.shape[2] == 0:
        return Mat.new_zeros(Mat.shape[0], Mat.shape[1], 0), torch.zeros(Mat.shape[0], dtype=torch.long)
    U, sv = _robust_batched_svd_Usv(Mat)                                   # robust to gesdd non-convergence (byte-identical when it converges)
    keep = sv > rtol * sv[:, :1].clamp_min(1e-300)
    keep &= sv > atol * (float(sv.max()) if ref is None else ref)
    U = U * keep.unsqueeze(1)
    dims = keep.sum(1)
    return U[:, :, : int(dims.max()) if dims.numel() else 0].contiguous(), dims


def random_members(N, shape, q, gen):
    """q random members of the subspace with orthonormal basis N (p x k), reshaped to q x n x m1."""
    c = torch.randn(N.shape[1], q, generator=gen, dtype=N.dtype)
    return (N @ c).T.reshape(q, *shape)


class Net:
    """Frozen structure of an MLP with biases on the analysis set (ocean_lib conventions for H, S, D)."""

    def __init__(self, net, X):
        st = O.mlp_structure(net, X)
        self.L = st["L"]; self.s = X.shape[0]; self.d = st["d"]
        self.H = st["H"]; self.S = st["S"]; self.D = st["D"]              # H[l]: s x (m+1);  S[l]: s x n;  D[l]: s x d x n
        # D[l] already folds the eval-mode BN slope a_l (blocks.mlp_structure).  The forward/backward space recursions
        # propagate a preactivation perturbation through the SAME BN affine before the gate, so they use the effective
        # gate Sa[l] = S[l] * a_l (a_l = gamma_l / sqrt(var_l + eps)); without BN, a is None and Sa is S unchanged.
        self.a = O.bn_slopes(net)
        self.Sa = self.S if self.a is None else [self.S[l] * self.a[l].view(1, -1) for l in range(self.L)]
        self.W = [lin.weight.detach() for lin in net.layers]
        self.blocks = [f"W{l + 1}" for l in range(self.L)] + ["V"]
        self.shape = {f"W{l + 1}": (self.W[l].shape[0], self.W[l].shape[1] + 1) for l in range(self.L)}
        self.shape["V"] = (self.d, self.W[-1].shape[0] + 1)

    def p(self, blk):
        return self.shape[blk][0] * self.shape[blk][1]


# ----------------------------------------------------------------------------- forward and backward recursions
def forward_spaces(nt, M, gen, q_mult=2):
    """U[l] (l = 0..L, the last for the readout): s x m1 x u bases of the received spaces; dims per sample."""
    U, udims, R = [], [], None
    for l in range(nt.L + 1):
        h = nt.H[l].unsqueeze(2)                                           # s x m1 x 1
        if R is not None and R.shape[2] > 0:
            Rz = torch.cat([R, R.new_zeros(nt.s, 1, R.shape[2])], 1)       # [R; 0]
            Ub, ud = batched_span(torch.cat([h / h.norm(dim=1, keepdim=True), Rz], 2))
        else:
            Ub, ud = h / h.norm(dim=1, keepdim=True), torch.ones(nt.s, dtype=torch.long)
        U.append(Ub); udims.append(ud)
        if l == nt.L:
            break
        parts = []
        if R is not None and R.shape[2] > 0:
            parts.append(torch.einsum("nm,smr->snr", nt.W[l], R))
        N = M[nt.blocks[l]]
        if N.shape[1] > 0:
            n, m1 = nt.shape[nt.blocks[l]]; q = q_mult * n
            A = random_members(N, (n, m1), q, gen)                         # q x n x m1
            c = torch.randn(nt.s, Ub.shape[2], q, generator=gen, dtype=Ub.dtype)
            parts.append(torch.einsum("qnm,smq->snq", A, Ub @ c))
        if parts:
            Y = torch.cat(parts, 2)
            R, _ = batched_span(Y * nt.Sa[l].unsqueeze(2), ref=float(Y.norm(dim=(1, 2)).max()))   # gate incl BN slope a_l
        else:
            R = None
    return U, udims


def backward_spaces(nt, M, gen, q_mult=2):
    """Zhat[l] (l = 0..L-1): s x n_l x z bases of the covectors that block l must annihilate on A u."""
    NV = M["V"]; d, m1 = nt.shape["V"]
    if NV.shape[1] > 0:
        rows = NV.T.reshape(-1, d, m1)[:, :, :-1].reshape(-1, m1 - 1)      # weight parts of the members of M_V
        sv, Vh = _robust_svd_svVh(rows, False)
        Zin = Vh[sv > RTOL * sv[0]].T.unsqueeze(0).expand(nt.s, -1, -1)    # s x n_L x z, the same for every sample
    else:
        Zin = None
    Zhat = [None] * nt.L
    for l in range(nt.L - 1, -1, -1):
        Dl = nt.D[l].transpose(1, 2)                                       # s x n x d, columns D^T e_a (D folds a_l)
        SZ = Zin * nt.Sa[l].unsqueeze(2) if Zin is not None and Zin.shape[2] > 0 else None          # gate incl BN slope a_l
        Zhat[l], _ = batched_span(torch.cat([Dl, SZ], 2) if SZ is not None else Dl, ref=float(Dl.norm(dim=(1, 2)).max()))
        if l == 0:
            break
        parts = []
        if SZ is not None:
            parts.append(torch.einsum("nm,snz->smz", nt.W[l], SZ))
        N = M[nt.blocks[l]]
        if N.shape[1] > 0:
            n, m1 = nt.shape[nt.blocks[l]]; q = q_mult * (m1 - 1)
            A = random_members(N, (n, m1), q, gen)                         # q x n x m1 (weight part used below)
            c = torch.randn(nt.s, Zhat[l].shape[2], q, generator=gen, dtype=Dl.dtype); zc = Zhat[l] @ c
            parts.append(torch.einsum("qnm,snq->smq", A[:, :, :-1], zc))
            ref = float(A.reshape(q, -1).norm(dim=1).max() * zc.norm(dim=1).max())
        else:
            ref = None
        Zin = batched_span(torch.cat(parts, 2), ref=max(ref or 0.0, float(nt.W[l].norm())))[0] if parts else None
    return Zhat


# ----------------------------------------------------------------------- sketch-robust (consensus) received spaces (2026-10-02)
# REPAIR of the sketch-seed-dependent incoming-span reconstruction instability (gain arm, memo 2026-10-01/2026-10-02):
# a single random sketch's batched_span truncation retains a weak incoming direction only sometimes, so the block solve
# built against ONE sketch can miss a constraint another sketch would expose, leaving the candidate uncertified under a
# fresh sketch.  The consensus recursions below run the forward/backward recursion for several sketch seeds and take the
# PER-SAMPLE UNION of the received/covector spaces.  Unioning (not intersecting) is CONSERVATIVE: a direction retained by
# ANY sketch is kept, so the solve enforces every constraint any sketch would, and the branch certifies for all of them
# -- at the cost of the (sacrificed) dimension the extra constraints remove.  These are ADDITIVE: the single-sketch path
# (consensus_seeds=None) is byte-identical to the original forward_spaces/backward_spaces/build.
def _union_batched(spaces, nt):
    """Per-sample union span of several zero-padded batched orthonormal bases (each s x m x u_k): concatenate along the
    basis axis and re-orthonormalize, giving an orthonormal basis of the span of EVERY retained direction (each sketch's
    columns are already orthonormal, so a unique direction has singular value 1 and is never dropped by batched_span)."""
    if all(S.shape[2] == 0 for S in spaces):
        return spaces[0][:, :, :0], torch.zeros(nt.s, dtype=torch.long)
    return batched_span(torch.cat(spaces, 2))


def forward_spaces_consensus(nt, M, seeds, q_mult=2):
    """Sketch-robust received spaces: per-sample union of forward_spaces' U[l] over the sketch seeds `seeds`."""
    per = [forward_spaces(nt, M, torch.Generator().manual_seed(int(s)), q_mult)[0] for s in seeds]
    U, udims = [], []
    for l in range(nt.L + 1):
        Ul, dl = _union_batched([p[l] for p in per], nt)
        U.append(Ul); udims.append(dl)
    return U, udims


def backward_spaces_consensus(nt, M, seeds, q_mult=2):
    """Sketch-robust covector spaces: per-sample union of backward_spaces' Zhat[l] over the sketch seeds `seeds`."""
    per = [backward_spaces(nt, M, torch.Generator().manual_seed(int(s)), q_mult) for s in seeds]
    return [_union_batched([p[l] for p in per], nt)[0] for l in range(nt.L)]


# ----------------------------------------------------------------------------- one block given the others
def kernel_within(N, row_chunks, sigma_ref=None, accept_abs=None):
    """Kernel, inside span(N), of the stacked rows produced by row_chunks(N) (each chunk already multiplied by N:
    rows x k).  QR accumulation, one SVD, threshold RTOL * sigma_ref, where sigma_ref is the largest singular value
    of the block's unrestricted atomic system (rows of unit-norm u and z), so that a restricted system whose rows
    all vanish on span(N) is read as rank zero and not as noise of full rank.  Returns the basis and the record.

    `accept_abs` (2026-10-03, additive, default None = the RTOL*sigma_ref relative cutoff, byte-identical): when set,
    the retained kernel is thresholded at an ABSOLUTE value on the constraint singular values (sv <= accept_abs).  The
    sv are the magnitudes of the UNIT-covector forward criterion |z^T A u| (z, u unit; A orthonormal in span(N)), so an
    absolute accept_abs aligns the CONSTRUCTION acceptance to the VALIDATION tolerance in one identical normalization
    instead of the sigma_ref-scaled relative cutoff (sigma_ref can be ~1e2, inflating the effective acceptance)."""
    k = N.shape[1]; R = None; buf = []; nbuf = 0
    def flush():
        nonlocal R, buf, nbuf
        if buf:
            R = torch.linalg.qr(torch.cat(([R] if R is not None else []) + buf, 0), mode="r").R; buf = []; nbuf = 0
    if k == 0:
        return N, {"dim": 0, "rank_within": 0, "gap": None, "note": "empty candidate space"}
    for T_ in row_chunks(N):
        if T_.shape[0] == 0:                                               # all constraint rows of the chunk vanish
            continue
        buf.append(T_); nbuf += T_.shape[0]
        if nbuf >= max(2 * k, 20000):                                      # few large QRs instead of one per chunk
            flush()
    flush()
    if R is None or R.shape[0] == 0:                                       # empty constraint operator: everything is admissible
        return N, {"dim": k, "rank_within": 0, "gap": None, "note": "no nonzero constraint row"}
    sv, Vh = _robust_svd_svVh(R, True)
    ref = float(sv[0]) if sigma_ref is None else sigma_ref
    cut = RTOL * ref if accept_abs is None else float(accept_abs)
    r = int((sv > cut).sum())
    gap = float(sv[r - 1] / max(sv[r], 1e-300)) if 0 < r < sv.numel() else None
    return N @ Vh[r:].T, {"dim": k - r, "rank_within": r, "gap": gap, "sigma1": float(sv[0]), "sigma_ref": ref,
                          "cutoff": cut, "cutoff_mode": "absolute" if accept_abs is not None else "relative_rtol_sigma_ref",
                          "largest_rejected_over_cut": float(sv[r] / cut) if r < sv.numel() else 0.0,
                          "largest_rejected_over_ref": float(sv[r] / ref) if r < sv.numel() else 0.0}


def solve_hidden(nt, l, U_l, Zhat_l, N0, chunk=64, pre=400, sigma_ref=None, accept_abs=None):
    """Maximal M_l inside span(N0) with z^T A u = 0 for all samples, u in U_l,i, z in Zhat_l,i.  `accept_abs` (additive,
    default None): absolute aligned cutoff threaded to kernel_within (see there)."""
    n, m1 = nt.shape[nt.blocks[l]]
    if N0.shape[1] == 0:                                                   # empty atomic kernel: nothing is admissible
        return N0, {"dim": 0, "rank_within": 0, "gap": None, "prepass": None}
    def chunks_for(idx_all):
        def gen(N):
            Nt = N.T.reshape(-1, n, m1)                                     # k x n x m1
            for i in range(0, len(idx_all), chunk):
                idx = idx_all[i:i + chunk]
                T_ = torch.einsum("snz,knm,smu->szuk", Zhat_l[idx], Nt, U_l[idx]).reshape(-1, Nt.shape[0])
                yield T_[T_.abs().amax(1) > 0]
        return gen
    N, rec_pre = N0, None
    if pre and nt.s > 2 * pre and N0.shape[1] > 50:
        sub = torch.linspace(0, nt.s - 1, pre).long()
        N, rec_pre = kernel_within(N0, chunks_for(sub), sigma_ref, accept_abs)
        if N.shape[1] == 0:                                                # the prepass already removed every candidate
            rec_pre["prepass"] = None
            return N, dict(rec_pre, note="empty after the subset prepass", prepass=dict(rec_pre))
    N, rec = kernel_within(N, chunks_for(torch.arange(nt.s)), sigma_ref, accept_abs)
    rec["prepass"] = rec_pre
    return N, rec


def solve_readout(nt, U_V, accept_abs=None):
    """M_V = R^d (x) (sum_i U_{L+1,i})^perp, as an orthonormal basis of vec(B), B: d x m1 (row-major).  `accept_abs`
    (additive, default None = RTOL*sv[0] relative): absolute aligned cutoff on the received-space singular values."""
    d, m1 = nt.shape["V"]
    rows = U_V.transpose(1, 2).reshape(-1, m1); rows = rows[rows.abs().amax(1) > 0]
    R = torch.linalg.qr(rows, mode="r").R
    sv, Vh = _robust_svd_svVh(R, True)
    cut = RTOL * float(sv[0]) if accept_abs is None else float(accept_abs)
    r = int((sv > cut).sum())
    comp = Vh[r:].T                                                        # m1 x c
    N = torch.einsum("ab,mc->ambc", torch.eye(d, dtype=comp.dtype), comp).reshape(d * m1, d * comp.shape[1])
    gap = float(sv[r - 1] / max(sv[r], 1e-300)) if 0 < r < sv.numel() else None
    return N, {"dim": N.shape[1], "rank_within": r, "gap": gap, "sigma1": float(sv[0]),
               "cutoff": cut, "cutoff_mode": "absolute" if accept_abs is not None else "relative_rtol_sigma1"}


def atomic_kernels(nt, known_full_rank=()):
    """Orthonormal bases (row-major vec of A = [W b]) of the atomic kernels of every block, with the rank record.
    Blocks listed in known_full_rank (full rank in the stored atomic decomposition of the same checkpoint and
    samples) get the empty kernel without forming the p x p identity; this is how the 33k-column input layer of
    the width-128 networks is handled."""
    K, rec = {}, {}
    for l in range(nt.L):
        blk = nt.blocks[l]; n, m1 = nt.shape[blk]
        if blk in known_full_rank:
            K[blk] = torch.zeros(n * m1, 0, dtype=nt.H[0].dtype); rec[blk] = {"dim": 0, "rank_within": n * m1, "gap": None, "note": "full rank in the stored atomic decomposition"}; continue
        U0 = (nt.H[l] / nt.H[l].norm(dim=1, keepdim=True)).unsqueeze(2)
        Z0, _ = batched_span(nt.D[l].transpose(1, 2), ref=float(nt.D[l].norm(dim=(1, 2)).max()))
        K[blk], rec[blk] = solve_hidden(nt, l, U0, Z0, torch.eye(n * m1, dtype=U0.dtype), pre=0)
    K["V"], rec["V"] = solve_readout(nt, (nt.H[nt.L] / nt.H[nt.L].norm(dim=1, keepdim=True)).unsqueeze(2))
    return K, rec


def feature_null(nt):
    """A [h;1] = 0 for every sample: R^n (x) (feature span)^perp per block."""
    Mfn = {}
    for l, blk in enumerate(nt.blocks):
        n, m1 = nt.shape[blk]
        sv, Vh = _robust_svd_svVh(torch.linalg.qr(nt.H[l], mode="r").R, True)
        comp = Vh[int((sv > RTOL * sv[0]).sum()):].T
        Mfn[blk] = torch.einsum("ab,mc->ambc", torch.eye(n, dtype=comp.dtype), comp).reshape(n * m1, n * comp.shape[1])
    return Mfn


def build(nt, order, K, gen, verbose=True, sigma_ref=None, consensus_seeds=None, accept_abs=None):
    """Block-coordinate construction in the given order of blocks; returns the bases and the per-step records.
    consensus_seeds=None (default): single-sketch recursions, byte-identical to the original.  A list of sketch seeds:
    sketch-robust build -- the received/covector spaces are the per-sample UNION over those seeds (see the consensus
    recursions above), giving a branch that certifies across sketches at the cost of sacrificed dimension.
    accept_abs=None (default): the RTOL*sigma_ref relative block cutoff, byte-identical.  A float: the ABSOLUTE aligned
    cutoff on every block solve (see kernel_within) so the construction acceptance matches the validation tolerance in
    one identical normalization."""
    M = {blk: K[blk][:, :0] for blk in nt.blocks}; steps = []
    for blk in order:
        t0 = time.time()
        U, udims = forward_spaces(nt, M, gen) if consensus_seeds is None else forward_spaces_consensus(nt, M, consensus_seeds)
        if blk == "V":
            M[blk], rec = solve_readout(nt, U[nt.L], accept_abs=accept_abs)
        else:
            l = nt.blocks.index(blk)
            Zhat = backward_spaces(nt, M, gen) if consensus_seeds is None else backward_spaces_consensus(nt, M, consensus_seeds)
            M[blk], rec = solve_hidden(nt, l, U[l], Zhat[l], K[blk], sigma_ref=sigma_ref[blk] if sigma_ref else None, accept_abs=accept_abs)
            rec["mean_dim_Zhat"] = float((Zhat[l].abs().amax(1) > 0).sum(1).double().mean())
        li = nt.L if blk == "V" else nt.blocks.index(blk)
        rec.update({"block": blk, "mean_dim_U": float(udims[li].double().mean()), "seconds": time.time() - t0})
        steps.append(rec)
        if verbose:
            print(f"     {blk}: dim {rec['dim']} of atomic {K[blk].shape[1]}  (mean dim U {rec['mean_dim_U']:.1f}, gap {rec['gap']}, {rec['seconds']:.0f}s)", flush=True)
    return M, steps


# ----------------------------------------------------------------------------- certificates
def embed(net, nt, mats):
    """Flat parameter direction (order of net.parameters()) from block matrices A = [W b].

    Explicit conditional-coordinate mapping: each block's [W b] is written at the true flat position of its weight/bias
    (FL.block_slices), and every other coordinate -- BatchNorm gamma/beta in particular -- is left at zero, so the
    branch perturbs only the W/b blocks and holds the BN affine (and running-stat buffers) fixed.  Length matches
    FL.flat_params(net), so theta0 + r*w is well defined for plain and BatchNorm models alike."""
    total = sum(p.numel() for p in net.parameters())
    dtype = mats[nt.blocks[0]].dtype
    w = torch.zeros(total, dtype=dtype)
    slices = {name: (a, b) for name, a, b in FL.block_slices(net)}
    for blk in nt.blocks:
        A = mats[blk]; a, b = slices[blk]
        w[a:b] = torch.cat([A[:, :-1].reshape(-1), A[:, -1].reshape(-1)])
    return w


def random_branch_matrices(nt, M, gen):
    mats = {}
    for blk in nt.blocks:
        N = M[blk]
        mats[blk] = (N @ torch.randn(N.shape[1], generator=gen, dtype=N.dtype)).reshape(nt.shape[blk]) if N.shape[1] else torch.zeros(nt.shape[blk], dtype=N.dtype)
    return mats


def relative_norm(net, nt, w, theta0):
    """||w||_rel = sqrt(sum_l ||w_l||^2 / ||theta_l||^2) over parameter blocks."""
    tot = 0.0; i = 0
    for blk in nt.blocks:
        n = nt.p(blk); tot += float(w[i:i + n].norm() ** 2 / theta0[i:i + n].norm() ** 2); i += n
    return tot ** 0.5


@torch.no_grad()
def polynomial_residual(net, nt, X, M, gen, n_dirs=32, rel_norms=(0.1, 1.0, 10.0)):
    """Frozen output change along random linear combinations of the whole branch, at several weighted relative
    norms; the frozen output is a polynomial of degree at most L+1 along a line, so vanishing at more distinct
    points than its degree on random lines tests every coefficient.  Values are relative to ||F0||_X."""
    theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X); idx = torch.arange(X.shape[0])
    F0 = FL.frozen_forward(net, masks, X, idx); scale = FL.x_norm(F0); worst = {str(r): 0.0 for r in rel_norms}
    if not (np.isfinite(scale) and scale > 0):
        return {str(r): float("nan") for r in rel_norms}                    # no meaningful normalization: fails validation
    pts = [0.37, 0.61, 1.0, -0.83, -0.45, 1.7]                              # six points for a degree <= L+1 = 4 polynomial
    try:
        for _ in range(n_dirs):
            w = embed(net, nt, random_branch_matrices(nt, M, gen))
            if not w.any():
                break
            w = w / relative_norm(net, nt, w, theta0)
            for r in rel_norms:
                for t in pts:
                    FL.set_flat(net, theta0 + r * t * w)
                    worst[str(r)] = _nanmax(worst[str(r)], FL.x_norm(FL.frozen_forward(net, masks, X, idx) - F0) / scale)
    finally:
        FL.set_flat(net, theta0)                                            # the base parameters are restored whatever happens
    return worst


def _nanmax(a, b):
    """max that propagates a nonfinite value instead of discarding it (Python's max(0.0, nan) is 0.0)."""
    return float("nan") if not (np.isfinite(a) and np.isfinite(b)) else max(a, b)


def forward_criterion(nt, M, gen, n_test=16):
    """max over blocks of ||D_l,i A u|| / (||D_l,i||_F ||A||_F) for random A in M_l and random unit u in U_l,i
    (forward recursion only), and ||B u|| / ||B||_F at the readout."""
    U, _ = forward_spaces(nt, M, gen); worst = {}
    for l, blk in enumerate(nt.blocks):
        N = M[blk]
        if N.shape[1] == 0:
            worst[blk] = 0.0; continue
        A = random_members(N, nt.shape[blk], n_test, gen)
        c = torch.randn(nt.s, U[l].shape[2], n_test, generator=gen, dtype=N.dtype)
        u = U[l] @ c; u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-300)
        Au = torch.einsum("qnm,smq->snq", A, u)
        if blk == "V":
            val = Au.norm(dim=1) / A.reshape(n_test, -1).norm(dim=1)
        else:
            val = torch.einsum("sdn,snq->sdq", nt.D[l], Au).norm(dim=1) / (A.reshape(n_test, -1).norm(dim=1) * nt.D[l].reshape(nt.s, -1).norm(dim=1, keepdim=True).clamp_min(1e-300))
        worst[blk] = _nanmax(0.0, float(val.max()))
    return worst


def forward_criterion_full(nt, M, gen, chunk=32):
    """Full-operator forward criterion: max over samples, basis elements A of M_l (orthonormal, ||A||_F = 1) and
    received basis vectors u of U_l,i (unit norm) of ||D_l,i A u|| / ||D_l,i||_F (hidden) or ||B u|| (readout).
    U comes from a fresh forward recursion."""
    U, _ = forward_spaces(nt, M, gen); worst = {}
    for l, blk in enumerate(nt.blocks):
        N = M[blk]
        if N.shape[1] == 0:
            worst[blk] = 0.0; continue
        Nt = N.T.reshape(-1, *nt.shape[blk]); w = 0.0
        for i in range(0, nt.s, chunk):
            Au = torch.einsum("knm,smu->sknu", Nt, U[l][i:i + chunk])                  # s x k x n x u
            if blk == "V":
                val = Au.norm(dim=2)
            else:
                Dn = nt.D[l][i:i + chunk]; scale = Dn.reshape(Dn.shape[0], -1).norm(dim=1).clamp_min(1e-300)
                val = torch.einsum("sdn,sknu->skdu", Dn, Au).norm(dim=2) / scale.view(-1, 1, 1)
            w = _nanmax(w, float(val.max()))
        worst[blk] = w
    return worst


def validate(net, nt, X, M, gen):
    """Validation record of a candidate layerwise space; `passed` requires finite residuals at most VALID_TOL AND that no
    SVD on the validation path fell through to the Gram last resort.  The Gram eigendecomposition FLOORS tight singular
    values near sqrt(machine eps), so a received-space reconstruction that used it cannot certify a 1e-9 criterion; such
    a result is flagged `gram_fallback_used` and NOT certified (2026-10-03)."""
    _snap = dict(SVD_SOLVER_COUNTS)
    rec = {"tolerance": VALID_TOL,
           "polynomial_residual_rel": polynomial_residual(net, nt, X, M, gen),
           "forward_criterion_sampled_max": forward_criterion(nt, M, gen),
           "forward_criterion_full_max": forward_criterion_full(nt, M, gen),
           "scope": "full forward operator on received spaces from a randomized forward recursion; polynomial residual on 32 random "
                    "combinations at relative norms 0.1, 1, 10 (six points per line)"}
    vals = list(rec["polynomial_residual_rel"].values()) + list(rec["forward_criterion_sampled_max"].values()) + list(rec["forward_criterion_full_max"].values())
    rec["svd_solver_counts"] = {k: SVD_SOLVER_COUNTS.get(k, 0) - _snap.get(k, 0) for k in set(SVD_SOLVER_COUNTS) | set(_snap)}
    rec["gram_fallback_used"] = bool(rec["svd_solver_counts"].get("gram", 0) > 0)
    rec["residuals_within_tolerance"] = bool(all(np.isfinite(v) and v <= VALID_TOL for v in vals))
    rec["passed"] = bool(rec["residuals_within_tolerance"] and not rec["gram_fallback_used"])   # Gram path cannot certify 1e-9
    return rec


def build_config(a, n_samples, scope=None):
    cfg = {"protocol": PROTOCOL, "build_version": BUILD_VERSION, "rtol": RTOL, "valid_tol": VALID_TOL, "samples": int(n_samples),
           "random_orders": a.random_orders, "generator_seed": 1234}
    if getattr(a, "_custom_index_sha", None):            # P1 custom fitted-subset build: pin the subset in the identity
        cfg["custom_index_sha256"] = a._custom_index_sha
    if scope is not None and scope.get("ident_fields"):
        # a RESTRICTED build (some blocks taken empty without factorizing them here) records the excluded blocks and the
        # authenticated atomic-record hash in the identity, so it never shares a cache identity with an unrestricted build
        # of the same run.  A full build adds nothing, keeping its legacy digest.
        cfg["block_scope"] = scope["ident_fields"]
    if getattr(a, "consensus_seeds", None):
        # a sketch-robust consensus build is a DIFFERENT construction from the single-sketch one (the received spaces are
        # the union over these sketch seeds), so the sketch set is part of the construction identity -- a single-sketch
        # cache must never be reused for a consensus build or vice versa.  Default (None) adds nothing (legacy digest).
        cfg["consensus_seeds"] = list(a.consensus_seeds)
    if getattr(a, "accept_abs", None) is not None:
        # an aligned absolute-cutoff build is a DIFFERENT construction from the relative-cutoff one; bind it so the two
        # cannot share a cache identity.  Default (None) adds nothing (legacy digest).
        cfg["accept_abs"] = float(a.accept_abs)
    return cfg


def block_scope(a, name, stage, ep, nt, results_root):
    """Which blocks are factorized here vs taken empty, and why, from the AUTHENTICATED stored atomic record.
    Distinguishes 'measured' (kernel computed in this build), 'full_rank_in_atomic' (dim_K == 0 measured there, taken
    empty) and 'excluded_frozen' (excluded from the atomic pass -- a frozen input layer -- never factorized).  The atomic
    record is authenticated by file hash, matching epoch and per-block parameter counts.  Returns the empty-kernel tuple,
    the per-block source, the record metadata and the identity fields (present only when the build is restricted)."""
    kernel_source = {b: "measured" for b in nt.blocks}; excluded = {}; atomic_auth = None
    hidden = [b for b in nt.blocks if b != "V"]
    if a.use_atomic_ranks and (nt.s == 10000 or getattr(a, "custom_idx", None) is not None):
        ap_ = os.path.join(results_root, "atomic", f"{name}.json")
        if os.path.exists(ap_):
            doc = json.load(open(ap_)); arec = doc["results"].get(stage, {})
            if "same_as_epoch" in arec:
                arec = next(v for v in doc["results"].values() if v.get("epoch") == arec["same_as_epoch"])
            if arec.get("epoch") == ep:
                atomic_auth = {"path": os.path.relpath(ap_, results_root), "sha256": P.sha256_file(ap_), "epoch": int(ep),
                               "protocol": doc.get("protocol")}
                bylayer = {f"W{L['layer']}": L for L in arec["layers"] if L["layer"] != "readout"}
                for b, L in bylayer.items():                                    # authenticate against THIS network
                    if b in nt.blocks and "p" in L and int(L["p"]) != int(nt.p(b)):
                        raise P.ProvenanceError(f"{name} {stage}: atomic record block {b} p={L['p']} != network p={nt.p(b)}")
                for b in hidden:
                    L = bylayer.get(b)
                    if L is None:
                        continue
                    if L.get("status") == "excluded":
                        excluded[b] = "excluded_frozen"; kernel_source[b] = "excluded_frozen"
                    else:
                        comp = L.get("components@1e-09")
                        if isinstance(comp, dict) and comp.get("dim_K") == 0:
                            excluded[b] = "full_rank_in_atomic"; kernel_source[b] = "full_rank_in_atomic"
            else:
                print(f"   [scope] atomic record for {name} {stage} not at epoch {ep}; no exclusions applied", flush=True)
    record = {"restricted": bool(excluded), "allowed_blocks": [b for b in nt.blocks if b not in excluded],
              "excluded_blocks": excluded, "atomic_record": atomic_auth}
    ident_fields = None
    if excluded:
        ident_fields = {"excluded_blocks": {b: excluded[b] for b in sorted(excluded)},
                        "atomic_sha256": atomic_auth["sha256"] if atomic_auth else None}
    return {"empty": tuple(excluded), "kernel_source": kernel_source, "record": record, "ident_fields": ident_fields}


# ----------------------------------------------------------------------------- orders
def orders(nt, n_random, seed=0):
    blocks = nt.blocks
    out = {"bottom_up": list(blocks), "top_down": list(reversed(blocks))}
    perms = [list(p) for p in itertools.permutations(blocks) if list(p) not in out.values()]
    rng = np.random.default_rng(seed)
    for k, j in enumerate(rng.choice(len(perms), size=min(n_random, len(perms)), replace=False)):
        out[f"random_{k}"] = perms[j]
    return out


def run(a):
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}{'_' + a.indices_tag if getattr(a, 'indices_tag', None) else ''}_seed{seed}"
        out_path = os.path.join(OUT, f"{name}.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "run": rd, "stages": stages, "rank_tolerance": RTOL, "results": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                print(f"== {name} {stage}: not reached", flush=True); res["results"][stage] = {"note": "stage not reached"}; continue
            _cidx = getattr(a, "custom_idx", None)
            run_ = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", indices=_cidx)
            if _cidx is not None:                        # P1: build on the EXACT fitted subset (default path unchanged)
                X = run_["data"]["indices"][0]; a._custom_index_sha = P.sha256_index(_cidx)
            else:
                X = run_["data"]["analysis"][0][: a.samples]
            net = run_["net"]
            nt = Net(net, X); gen = torch.Generator().manual_seed(1234)
            scope = block_scope(a, name, stage, ep, nt, os.path.dirname(OUT))   # exclusions (authenticated) BEFORE the identity
            full = scope["empty"]
            ident = P.analysis_identity(run_, rd, ep, build_config(a, X.shape[0], scope))
            old = res["results"].get(stage)
            if old and "constructions" not in old:                          # an "unavailable stage" note is not a record
                old = None
            if old:
                # validate before reuse: a stage computed under another identity is never relabelled
                if old.get("identity", {}).get("digest") != ident["digest"]:
                    if not a.overwrite:
                        raise P.ProvenanceError(f"{name} {stage}: stored stage has another analysis identity (samples, checkpoint, "
                                                f"configuration or build version); rerun with --overwrite to rebuild the whole stage")
                    print(f"== {name} {stage}: identity differs, rebuilding the whole stage", flush=True); old = None
            srec = old if old else {"epoch": ep, "identity": ident, "constructions": {}}
            if a.overwrite:
                srec = {"epoch": ep, "identity": ident, "constructions": {}}
            res["results"][stage] = srec
            srec["block_scope"] = scope["record"]                              # allowed / excluded (with reason) / authenticated atomic record
            print(f"== {name} {stage} (epoch {ep}): atomic kernels", flush=True)
            if full:
                print(f"   blocks taken as empty (not factorized here): {dict(scope['record']['excluded_blocks'])}", flush=True)
            _svd_atomic0 = dict(SVD_SOLVER_COUNTS)                             # count the atomic-kernel SVDs that FEED every construction
            K, krec = atomic_kernels(nt, full)
            srec["atomic_svd_solver_counts"] = {k: SVD_SOLVER_COUNTS.get(k, 0) - _svd_atomic0.get(k, 0) for k in set(SVD_SOLVER_COUNTS) | set(_svd_atomic0)}
            srec["atomic_gram_fallback_used"] = bool(srec["atomic_svd_solver_counts"].get("gram", 0) > 0)
            _atomic_gram = srec["atomic_gram_fallback_used"]
            # kernel_source distinguishes a MEASURED zero from an EXCLUDED (frozen) block that was never factorized
            srec["blocks"] = {blk: {"p": nt.p(blk), "dim_atomic_kernel": int(K[blk].shape[1]), "gap": krec[blk]["gap"],
                                    "kernel_source": scope["kernel_source"][blk]} for blk in nt.blocks}
            srec["sum_atomic"] = int(sum(K[b].shape[1] for b in nt.blocks)); srec["n_params"] = int(sum(nt.p(b) for b in nt.blocks))
            print(f"   atomic kernels {[K[b].shape[1] for b in nt.blocks]} sum {srec['sum_atomic']}", flush=True)
            cons = dict(orders(nt, a.random_orders, seed))
            todo = [("feature_null", None), ("naive_atomic_sum", None)] + list(cons.items())
            if a.only is not None:
                todo = [(c, o) for c, o in todo if c in a.only]
            for cname, order in todo:
                if cname in srec["constructions"]:
                    continue
                print(f"   -- {cname} {order or ''}", flush=True); t0 = time.time()
                _svd0 = dict(SVD_SOLVER_COUNTS)                                 # solver usage for THIS construction (build + validate)
                if cname == "feature_null":
                    M, steps = feature_null(nt), None
                elif cname == "naive_atomic_sum":
                    M, steps = K, None
                else:
                    M, steps = build(nt, order, K, gen, sigma_ref={b: krec[b].get("sigma1") for b in nt.blocks}, consensus_seeds=getattr(a, "consensus_seeds", None), accept_abs=getattr(a, "accept_abs", None))
                dims = {blk: int(M[blk].shape[1]) for blk in nt.blocks}
                val = validate(net, nt, X, M, gen)
                svd_counts = {k: SVD_SOLVER_COUNTS.get(k, 0) - _svd0.get(k, 0) for k in set(SVD_SOLVER_COUNTS) | set(_svd0)}
                gram_used = bool(svd_counts.get("gram", 0) > 0 or _atomic_gram)  # Gram anywhere in atomic+build+validate -> not 1e-9-certifiable
                bpath = os.path.join(OUT, f"{name}_epoch{ep:04d}_{cname}.pt")
                torch.save({"identity_digest": ident["digest"], "construction": cname, "order": order, "bases": {blk: M[blk] for blk in nt.blocks}}, bpath)
                rec = {"order": order, "dims": dims, "total": int(sum(dims.values())), "steps": steps,
                       "constructed_by_theorem": cname != "naive_atomic_sum",      # how the space was built; not a validation
                       "validation": val, "svd_solver_counts": svd_counts, "gram_fallback_used": gram_used,
                       "atomic_gram_fallback_used": bool(_atomic_gram),           # atomic-kernel SVDs feed every construction
                       "certified_branch": bool(val["passed"] and not gram_used), # certified only if validation passed AND no Gram fallback (atomic incl.)
                       "consensus_seeds": list(a.consensus_seeds) if getattr(a, "consensus_seeds", None) else None,
                       "accept_abs": getattr(a, "accept_abs", None),
                       "basis_sha256": P.sha256_file(bpath), "seconds": time.time() - t0}
                srec["constructions"][cname] = rec
                srec["svd_solver_counts_cumulative"] = dict(SVD_SOLVER_COUNTS)  # auditable solver path (gesdd/gesvd/gram) for the stage
                print(f"      dims {dims} total {rec['total']}  validation passed {val['passed']}: polynomial {max(val['polynomial_residual_rel'].values()):.1e} "
                      f"criterion full {max(val['forward_criterion_full_max'].values()):.1e} sampled {max(val['forward_criterion_sampled_max'].values()):.1e}", flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
        records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
        print("saved", out_path, flush=True)
    if SVD_SOLVER_COUNTS:                                                  # audit: which SVD solver ran (gesdd fast path, else gesvd/gram fallback)
        print("svd solver counts:", dict(SVD_SOLVER_COUNTS), flush=True)


# ----------------------------------------------------------------------------- tests
def _brute_force_bottom_up(nt, K):
    """Bottom-up construction with exact (non-randomized) spans, tiny networks only."""
    M = {}; R = [torch.zeros(nt.shape[nt.blocks[0]][1] - 1, 0, dtype=torch.float64) for _ in range(nt.s)]
    for l, blk in enumerate(nt.blocks):
        n, m1 = nt.shape[blk]; rows = []; Us = []
        for i in range(nt.s):
            Ui = torch.cat([nt.H[l][i].unsqueeze(1), torch.cat([R[i], torch.zeros(1, R[i].shape[1], dtype=torch.float64)], 0)], 1)
            Ui = torch.linalg.svd(Ui, full_matrices=False); Ui = Ui.U[:, Ui.S > 1e-10 * Ui.S[0]]; Us.append(Ui)
            Di = torch.eye(n, dtype=torch.float64) if blk == "V" else nt.D[l][i]
            for u in Ui.T:
                for drow in Di:
                    rows.append(torch.outer(drow, u).reshape(-1))
        Cm = torch.stack(rows); _, sv, Vh = torch.linalg.svd(Cm, full_matrices=True)
        r = int((sv > RTOL * sv[0]).sum()); M[blk] = Vh[r:].T
        if blk == "V":
            break
        newR = []
        for i in range(nt.s):
            cols = [nt.W[l] @ R[i]] if R[i].shape[1] else []
            if M[blk].shape[1]:
                Ab = M[blk].T.reshape(-1, n, m1)
                cols.append(torch.einsum("knm,mu->nku", Ab, Us[i]).reshape(n, -1))
            if cols:
                Y = torch.cat(cols, 1) * nt.Sa[l][i].unsqueeze(1); sv_ = torch.linalg.svd(Y, full_matrices=False)   # gate incl a_l
                newR.append(sv_.U[:, sv_.S > 1e-10 * max(sv_.S[0], 1e-300)] if sv_.S.numel() and sv_.S[0] > 1e-12 else torch.zeros(n, 0, dtype=torch.float64))
            else:
                newR.append(torch.zeros(n, 0, dtype=torch.float64))
        R = newR
    return M


def _tests():
    torch.manual_seed(0); torch.set_num_threads(4)
    net = T.MLP(6, [7, 7, 6], 3, init_seed=3, dtype=torch.float64)
    with torch.no_grad():                                                   # kill a few units so that kernels are not trivial
        net.layers[1].bias[:2] = -50.0; net.layers[2].bias[:1] = -50.0; net.layers[0].bias[:1] = -50.0
    X = torch.randn(12, 6, dtype=torch.float64) @ torch.diag(torch.tensor([1.0, 1, 1, 1, 0, 0], dtype=torch.float64))
    nt = Net(net, X); gen = torch.Generator().manual_seed(7)
    K, krec = atomic_kernels(nt); sref = {b: krec[b].get("sigma1") for b in nt.blocks}
    assert sum(K[b].shape[1] for b in nt.blocks) > 0
    res = {}
    for name, order in orders(nt, 6, 0).items():
        M, _ = build(nt, order, K, gen, verbose=False, sigma_ref=sref)
        pr = polynomial_residual(net, nt, X, M, gen, n_dirs=8); fc = forward_criterion(nt, M, gen)
        assert max(pr.values()) < 1e-10 and max(fc.values()) < 1e-10, (name, pr, fc)
        res[name] = sum(M[b].shape[1] for b in nt.blocks)
        if name == "bottom_up":
            Mb = _brute_force_bottom_up(nt, K)
            assert {b: Mb[b].shape[1] for b in nt.blocks} == {b: M[b].shape[1] for b in nt.blocks}, ({b: Mb[b].shape[1] for b in nt.blocks}, {b: M[b].shape[1] for b in nt.blocks})
    # consensus (sketch-robust) build (2026-10-02, additive): a valid branch, never larger than the single-sketch build
    Mc, _ = build(nt, orders(nt, 6, 0)["bottom_up"], K, gen, verbose=False, sigma_ref=sref, consensus_seeds=[7, 8, 9])
    prc = polynomial_residual(net, nt, X, Mc, gen, n_dirs=8); fcc = forward_criterion(nt, Mc, gen)
    assert max(prc.values()) < 1e-10 and max(fcc.values()) < 1e-10, ("consensus build not flat", prc, fcc)
    assert sum(Mc[b].shape[1] for b in nt.blocks) <= res["bottom_up"], ("consensus dim exceeds single-sketch", {b: Mc[b].shape[1] for b in nt.blocks})
    # aligned ABSOLUTE cutoff (2026-10-03): a valid branch, dim <= the default relative-cutoff build (tighter acceptance)
    Ma, _ = build(nt, orders(nt, 6, 0)["bottom_up"], K, gen, verbose=False, sigma_ref=sref, accept_abs=1e-12)
    pra = polynomial_residual(net, nt, X, Ma, gen, n_dirs=8); fca = forward_criterion(nt, Ma, gen)
    assert max(pra.values()) < 1e-10 and max(fca.values()) < 1e-10, ("accept_abs build not flat", pra, fca)
    assert sum(Ma[b].shape[1] for b in nt.blocks) <= res["bottom_up"], ("accept_abs dim exceeds relative", {b: Ma[b].shape[1] for b in nt.blocks})
    pr_naive = polynomial_residual(net, nt, X, K, gen, n_dirs=8)
    assert max(pr_naive.values()) > 1e-6, pr_naive                          # the direct sum of atomic kernels is not a branch
    Mfn = feature_null(nt); theta_before = FL.flat_params(net).clone()
    assert max(polynomial_residual(net, nt, X, Mfn, gen, n_dirs=8).values()) < 1e-10
    assert all(v <= sum(K[b].shape[1] for b in nt.blocks) for v in res.values())
    # acceptance: the certified flag follows the validation, not the name
    v_ok = validate(net, nt, X, M, gen); v_bad = validate(net, nt, X, K, gen)
    assert v_ok["passed"] and not v_bad["passed"], (v_ok, v_bad)
    import unittest.mock as mock
    real_ff = FL.frozen_forward
    calls = {"n": 0}
    def nan_after_first(*args, **kw):
        calls["n"] += 1; out = real_ff(*args, **kw)
        return out if calls["n"] == 1 else out * float("nan")               # base output finite, every perturbed output NaN
    with mock.patch.object(FL, "frozen_forward", side_effect=nan_after_first):
        vv = validate(net, nt, X, Mfn, gen)
    assert not vv["passed"] and all(np.isnan(x) for x in vv["polynomial_residual_rel"].values()), vv
    assert torch.equal(FL.flat_params(net), theta_before)                   # parameters restored after the failure
    with mock.patch(__name__ + ".polynomial_residual", return_value={"1.0": 1.0}):
        assert not validate(net, nt, X, M, gen)["passed"]
    with mock.patch(__name__ + ".polynomial_residual", return_value={"1.0": float("nan")}):
        assert not validate(net, nt, X, M, gen)["passed"]
    # empty constraint operator: fully dead hidden layers have unrestricted atomic kernels
    dead = T.MLP(2, [3, 3], 1, init_seed=1, dtype=torch.float64)
    with torch.no_grad():
        for lin in dead.layers:
            lin.weight.zero_(); lin.bias.fill_(-1.0)
    ntd = Net(dead, torch.randn(5, 2, dtype=torch.float64)); Kd, _ = atomic_kernels(ntd)
    assert Kd["W1"].shape[1] == 9 and Kd["W2"].shape[1] == 12, {b: Kd[b].shape[1] for b in ntd.blocks}
    # the subset prepass removes every candidate
    class _F:                                                              # minimal stand-in for Net
        s = 1000; blocks = ["W1"]; shape = {"W1": (8, 8)}
    I8 = torch.eye(8, dtype=torch.float64).unsqueeze(0).expand(1000, -1, -1)
    Ne, rece = solve_hidden(_F, 0, I8, I8, torch.eye(64, dtype=torch.float64), pre=2)
    assert Ne.shape[1] == 0, rece
    # BatchNorm: the eval-mode affine a_l = gamma/sqrt(var+eps) must be carried through D, the forward/backward space
    # recursions and the frozen forward, and embed must write only the [W b] blocks (gamma/beta held) at their true
    # flat positions.  Verify the actual build -> certificate -> ray path is flat and in original coordinates.
    bnn = T.MLP(6, [7, 7, 6], 3, init_seed=1, dtype=torch.float64, batchnorm=True)
    with torch.no_grad():
        gb = torch.Generator().manual_seed(101)
        for l in range(3):
            bnn.bn[l].weight.copy_(1.0 + 0.4 * torch.randn(bnn.bn[l].weight.shape, generator=gb, dtype=torch.float64))
            bnn.bn[l].bias.copy_(0.2 * torch.randn(bnn.bn[l].bias.shape, generator=gb, dtype=torch.float64))
            bnn.bn[l].running_mean.copy_(0.3 * torch.randn(bnn.bn[l].running_mean.shape, generator=gb, dtype=torch.float64))
            bnn.bn[l].running_var.copy_(0.5 + torch.rand(bnn.bn[l].running_var.shape, generator=gb, dtype=torch.float64))
        bnn.bn[2].bias[:1] = -40.0        # kill one top-layer unit; live coupling remains so bottom_up strictly restricts the atomic sum
    bnn.eval()
    a_sl = O.bn_slopes(bnn)
    assert a_sl is not None and max(float((a - 1).abs().max()) for a in a_sl) > 0.1, "BN slope must be nontrivial"
    Xb = torch.randn(12, 6, dtype=torch.float64) @ torch.diag(torch.tensor([1.0, 1, 1, 1, 0, 0], dtype=torch.float64))
    ntb = Net(bnn, Xb); gbn = torch.Generator().manual_seed(11)
    Kb, krb = atomic_kernels(ntb); srb = {b: krb[b].get("sigma1") for b in ntb.blocks}
    assert sum(Kb[b].shape[1] for b in ntb.blocks) > 0, "BN atomic kernels vanished"
    Mbn, _ = build(ntb, orders(ntb, 6, 0)["bottom_up"], Kb, gbn, verbose=False, sigma_ref=srb)
    prb = polynomial_residual(bnn, ntb, Xb, Mbn, gbn, n_dirs=8); fcb = forward_criterion(ntb, Mbn, gbn)
    assert max(prb.values()) < 1e-10 and max(fcb.values()) < 1e-10, ("BN certificate not flat", prb, fcb)
    prb_naive = polynomial_residual(bnn, ntb, Xb, Kb, gbn, n_dirs=8)
    assert max(prb_naive.values()) > 1e-6, ("BN naive atomic sum should not be a branch", prb_naive)
    Bbn = _brute_force_bottom_up(ntb, Kb)
    assert {b: Bbn[b].shape[1] for b in ntb.blocks} == {b: Mbn[b].shape[1] for b in ntb.blocks}, \
        ("BN brute vs randomized dims", {b: Bbn[b].shape[1] for b in ntb.blocks}, {b: Mbn[b].shape[1] for b in ntb.blocks})
    assert sum(Mbn[b].shape[1] for b in ntb.blocks) < sum(Kb[b].shape[1] for b in ntb.blocks), \
        "BN test must exercise the recursions: bottom_up should strictly restrict the atomic sum"
    # coordinate mapping + explicit frozen ray: full length, BN coords zero, gamma/beta held, frozen response flat
    theta_b = FL.flat_params(bnn).clone(); wb_end = max(e for _, _, e in FL.block_slices(bnn))
    wdir = embed(bnn, ntb, random_branch_matrices(ntb, Mbn, gbn))
    assert wdir.numel() == theta_b.numel() and float(wdir[wb_end:].abs().max()) == 0.0, "embed: length / BN coords"
    wdir = wdir / relative_norm(bnn, ntb, wdir, theta_b)
    masks_b = FL.base_masks(bnn, Xb); idxb = torch.arange(Xb.shape[0])
    F0b = FL.frozen_forward(bnn, masks_b, Xb, idxb); scb = FL.x_norm(F0b); g_before = [bnn.bn[l].weight.clone() for l in range(3)]
    worst_ray = max(FL.x_norm(FL.frozen_forward(bnn, masks_b, Xb, idxb) - F0b) / scb
                    for r in (0.1, 1.0) for _ in [FL.set_flat(bnn, theta_b + r * wdir)])
    FL.set_flat(bnn, theta_b)
    assert worst_ray < 1e-10, ("BN frozen ray not flat", worst_ray)
    assert all(torch.equal(bnn.bn[l].weight, g_before[l]) for l in range(3)) and torch.equal(FL.flat_params(bnn), theta_b)
    # resume of a stage that was unavailable when the record was written
    import tempfile, types
    with tempfile.TemporaryDirectory() as d:
        fake = {"protocol": PROTOCOL, "results": {"high_accuracy": {"note": "stage not reached"}}}
        json.dump(fake, open(os.path.join(d, "x.json"), "w"))
        old = json.load(open(os.path.join(d, "x.json")))["results"].get("high_accuracy")
        if old and "constructions" not in old:
            old = None
        assert old is None
    print("multilayer-branch tests passed:", res, "atomic sum", sum(K[b].shape[1] for b in nt.blocks),
          "feature-null", sum(Mfn[b].shape[1] for b in nt.blocks), "naive residual", pr_naive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "test"])
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="w64")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=["high_accuracy"])
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--random-orders", type=int, default=6)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--use-atomic-ranks", action="store_true", help="take blocks that are full rank in results/atomic as empty kernels (width-128 input layer)")
    ap.add_argument("--only", nargs="+", default=None, help="build only these constructions (e.g. --only bottom_up): skips top_down, feature_null and naive-atomic-sum")
    ap.add_argument("--consensus-seeds", type=int, nargs="+", default=None, help="sketch-robust build (additive, 2026-10-02): build the ordered constructions against the per-sample UNION of the received/covector spaces over these sketch seeds, repairing the single-sketch span-reconstruction instability (default None = single-sketch, byte-identical)")
    ap.add_argument("--accept-abs", type=float, default=None, help="aligned ABSOLUTE block acceptance cutoff (additive, 2026-10-03): threshold the construction rank decisions at this absolute value on the unit-covector forward criterion |z^T A u| instead of RTOL*sigma_ref, aligning the construction acceptance to the validation tolerance (default None = relative RTOL*sigma_ref, byte-identical)")
    ap.add_argument("--indices", default=None, help="P1 (additive): build on a CUSTOM fitted subset, a .npy of training-set row indices (default None = the standard analysis set, unchanged). The subset is pinned in the build identity (custom_index_sha256).")
    ap.add_argument("--indices-tag", default=None, help="P1 (additive): a short suffix appended to the record/basis NAME for a custom-subset build, so e.g. a condition-(a) subsample build on the full-data run (tag '') does not collide with the full-data record. Must match between build and branch-rays.")
    a = ap.parse_args()
    a.custom_idx = np.load(a.indices) if a.indices else None
    _tests() if a.phase == "test" else run(a)


