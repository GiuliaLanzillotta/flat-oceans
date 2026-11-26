"""Numerical linear algebra with the rank conventions of the paper.

Every rank decision states its reference scale (`Reference`):
    BLOCK_SIGMA1  thresholds relative to sigma_1 of a block Jacobian (the declared rule of the atomic decomposition);
    OWN           thresholds relative to the largest singular value of the matrix being ranked (features, downstream
                  stacks, stacked orthonormal bases);
    ABSOLUTE      absolute thresholds, for the projection of an orthonormal basis (columns of norm at most one).
The ladder is (1e-6, 1e-9, 1e-9 declared, 1e-12); the gap sigma_r / sigma_{r+1} at the declared rank says whether the
decision is stable.  Tensor shapes are written in the docstrings.
"""
import enum
import time

import numpy as np
import torch

TOL_LADDER = (1e-6, 1e-9, 1e-12)
DECLARED = "1e-09"
REL_GRID = tuple(10.0 ** -k for k in range(2, 13))   # k_rel(tau) profile
ABS_GRID = tuple(10.0 ** -k for k in range(1, 11))   # k(eps) profile on J / sqrt(s)
LP64_LIMIT = 26000       # largest square factor MKL gesdd with vectors handles under 32-bit workspace indexing
SVD_DEVICE = "cpu"


class Reference(str, enum.Enum):
    BLOCK_SIGMA1 = "sigma_1 of the block Jacobian"
    OWN = "own largest singular value"
    ABSOLUTE = "absolute"


def svdvals(M):
    """Singular values of a dense matrix (numpy, descending), computed on SVD_DEVICE."""
    return torch.linalg.svdvals(torch.as_tensor(M).to(SVD_DEVICE)).cpu().numpy()


def key(t):
    return f"{t:g}"


def rank_ladder(sv, reference, ref_value=None, tols=TOL_LADDER):
    """Counts of singular values above t * scale for every rung t, with `scale` = ref_value (BLOCK_SIGMA1), sv[0] (OWN) or 1
    (ABSOLUTE).  Returns {"counts": {rung: count}, "gap": gap at the declared rung, "reference": ..., "scale": scale}."""
    sv = np.asarray(sv, dtype=np.float64); reference = Reference(reference)
    if reference == Reference.OWN:
        scale = float(sv[0]) if sv.size and sv[0] > 0 else 0.0
    elif reference == Reference.ABSOLUTE:
        scale = 1.0
    else:
        if ref_value is None:
            raise ValueError("BLOCK_SIGMA1 needs ref_value = sigma_1 of the block")
        scale = float(ref_value)
    counts = {key(t): (int((sv > t * scale).sum()) if scale > 0 else 0) for t in tols}
    r = counts[DECLARED]
    return {"counts": counts, "gap": spectral_gap(sv, r), "reference": reference.value, "scale": scale, "sigma1": float(sv[0]) if sv.size else 0.0}


def counts_only(sv, reference, ref_value=None, tols=TOL_LADDER):
    return rank_ladder(sv, reference, ref_value, tols)["counts"]


def spectral_gap(sv, r):
    """sigma_r / sigma_{r+1}; None when r is 0 or the last index (full rank, no gap defined)."""
    sv = np.asarray(sv, dtype=np.float64)
    if r == 0 or r >= sv.size:
        return None
    return float(sv[r - 1] / sv[r]) if sv[r] > 0 else float("inf")


def profiles(sv, s):
    """k_rel(tau) (relative to sigma_1) and k(eps) on J / sqrt(s): numbers of singular values at or below the cut."""
    sv = np.asarray(sv, dtype=np.float64); n = sv.size; svh = sv / np.sqrt(s)
    return ({str(t): int(n - (sv > t * sv[0]).sum()) for t in REL_GRID}, {str(e): int(n - (svh > e).sum()) for e in ABS_GRID})


@torch.no_grad()
def numrank_matrix(M, tols=TOL_LADDER):
    """Own-reference rank ladder counts and spectrum of a moderate dense matrix (features s x m, downstream stacks)."""
    sv = svdvals(M)
    return counts_only(sv, Reference.OWN, tols=tols), sv


def span_basis(M, rtol=1e-9):
    """Orthonormal basis (n x r) of the column space of M (n x c), rank at rtol relative to the own sigma_1."""
    if M.shape[1] == 0:
        return M[:, :0]
    U, sv, _ = torch.linalg.svd(M, full_matrices=False)
    r = int((sv > rtol * sv[0]).sum()) if sv.numel() and sv[0] > 0 else 0
    return U[:, :r]


@torch.no_grad()
def assembled_R(row_fn, s, p, device, chunk=500, verbose=False):
    """Triangular factor R (min(s d, p) x p) of the stacked rows: the block is assembled on the CPU from row chunks generated
    on `device` (row_fn(idx) -> (b d) x p) and factored by one LAPACK QR.  100k x 33k float64 is 26 GB."""
    t0 = time.time()
    first = row_fn(torch.arange(0, min(chunk, s), device=device)); d = first.shape[0] // min(chunk, s)
    J = torch.empty(s * d, p, dtype=torch.float64); J[: first.shape[0]] = first.cpu(); del first
    for i in range(chunk, s, chunk):
        B = row_fn(torch.arange(i, min(i + chunk, s), device=device)); J[i * d: i * d + B.shape[0]] = B.cpu(); del B
    if verbose:
        print(f"    assembled {tuple(J.shape)} on CPU in {time.time() - t0:.0f}s", flush=True)
    R = torch.linalg.qr(J, mode="r").R; del J
    if verbose:
        print(f"    CPU QR done in {time.time() - t0:.0f}s", flush=True)
    return R


@torch.no_grad()
def blocked_R(row_fn, s, block, p, device, verbose=False):
    """R factor by successive QR updates over sample blocks (memory-bounded alternative to assembled_R)."""
    if p > 20000:
        block = min(block, 500)
    R = None; t0 = time.time()
    for i in range(0, s, block):
        B = row_fn(torch.arange(i, min(i + block, s), device=device))
        M = B if R is None else torch.cat([R, B], 0); del B
        R = torch.linalg.qr(M, mode="r").R; del M
        if device != "cpu":
            torch.cuda.empty_cache()
        if verbose:
            print(f"    block {i // block + 1}/{(s + block - 1) // block}: R {tuple(R.shape)}  {time.time() - t0:.0f}s", flush=True)
    return R


def svd_factor(R, vectors=True):
    """Singular values (numpy, padded to p) and right singular vectors Vh (torch, p x p) of a triangular factor R (r x p).
    Above LP64_LIMIT columns MKL's 32-bit workspace overflows, so numpy's 64-bit LAPACK is used."""
    p = R.shape[1]
    if not vectors:
        sv = svdvals(R); return np.concatenate([sv, np.zeros(max(0, p - sv.size))]), None, "gesdd (values only)"
    if p > LP64_LIMIT:
        _, sv, Vh = np.linalg.svd(np.asarray(R.cpu().numpy(), dtype=np.float64), full_matrices=True); Vh = torch.from_numpy(np.ascontiguousarray(Vh)); driver = "gesdd (numpy/openblas64)"
    else:
        _, sv_t, Vh = torch.linalg.svd(R.cpu(), full_matrices=True); sv = sv_t.numpy(); driver = "gesdd (torch)"
    return np.concatenate([sv, np.zeros(max(0, p - sv.size))]), Vh, driver


def projected_rank(Bp, tols=TOL_LADDER):
    """Absolute-threshold ladder of a projected stack Bp (p x K) of an orthonormal basis: a projected direction of norm
    below the rung is marginally null.  Returns the rank_ladder dict."""
    if Bp.shape[1] == 0:
        return {"counts": {key(t): 0 for t in tols}, "gap": None, "reference": Reference.ABSOLUTE.value, "scale": 1.0, "sigma1": 0.0}
    R = torch.linalg.qr(Bp, mode="r").R if Bp.shape[0] > Bp.shape[1] else Bp
    return rank_ladder(svdvals(R), Reference.ABSOLUTE, tols=tols)      # singular values on the CPU: GPU float64 SVD is ~30x slower


def principal_cosines(A, B):
    """Cosines of the principal angles between the column spaces of orthonormal A (n x a) and B (n x b)."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.array([])
    return torch.linalg.svdvals(A.T @ B).cpu().numpy()
