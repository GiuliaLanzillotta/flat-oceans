#!/usr/bin/env python
"""Matrix-free randomized numerical-rank estimator (Halko-Martinsson-Tropp range finder) and its
self-validation on matrices of KNOWN numerical rank.

Priority P5 ("width-512 randomized rank validation"): the ESTIMATOR and its validation only.  This file
is deliberately self-contained and additive -- it imports no shared project module, trains nothing, and is
NOT pointed at any real network operator.  The last section of this docstring names the operator access
that a later step would hand it.

What it does
------------
Given a linear operator A: R^n -> R^m available ONLY as closures

    matvec(V)  = A @ V         (V is n x b,  returns m x b),
    rmatvec(U) = A.T @ U       (U is m x b,  returns n x b),   optional for a symmetric operator,

and its shape (m, n), it estimates the NUMERICAL RANK of A at a relative tolerance rel_tol:

    rank(rel_tol) = #{ i : sigma_i(A) > rel_tol * sigma_1(A) },

together with the recovered leading singular values and a resolution witness, WITHOUT ever forming the
dense matrix and WITHOUT being given the true singular values.  The stored/true singular values are used
only by the self-test below as the REFERENCE ANSWER to check the estimator; they are never its input.

Method (randomized subspace iteration, HMT 2011)
------------------------------------------------
Draw a Gaussian test matrix Omega (n x k), k = target_rank + oversampling.  Form the sketch Y = A Omega
(m x k) and orthonormalize Q = orth(Y).  Refine with q subspace ("power") iterations in the A.T A form,
re-orthonormalizing by QR between every application for numerical stability:

    for j = 1..q:   Q <- orth( A ( orth( A.T Q ) ) ).

Then the small matrix B = Q.T A (k x n) has, to the accuracy of the captured subspace, the top-k singular
values of A.  We compute them as the singular values of B.T = A.T Q (formed by one rmatvec) and count
those above rel_tol * sigma_max.  With q power iterations the error decays geometrically in the spectral
gap, so q = 2 already recovers well-separated spectra to full float64 precision.

The A.T A form (alternating matvec / rmatvec with a QR between them) is exactly power iteration with the
Gram operator A.T A (n-side) or A A.T (m-side); for a symmetric operator supplied with matvec only it
reduces to power iteration with A^2, so rmatvec may be omitted when the operator is square and symmetric.

Reference: N. Halko, P. G. Martinsson, J. A. Tropp, "Finding Structure with Randomness: Probabilistic
Algorithms for Constructing Approximate Matrix Decompositions", SIAM Review 53(2), 2011 (Algorithms 4.3
and 4.4, randomized range finder with power iteration; Section 5, spectral-norm error bounds).

Adaptive variant
-----------------
`adaptive_randomized_rank` grows k (geometrically) until the sketch window strictly brackets the tolerance
with a margin -- i.e. at least `reject_margin` recovered singular values fall BELOW rel_tol * sigma_max --
so the caller need not know the rank in advance.  The tail ratio (largest rejected singular value divided
by sigma_max) is then guaranteed below rel_tol and is returned as the resolution witness; if k reaches
min(m, n) the sketch is a full factorization and the count is exact.

Everything is float64 and CPU-only.  Run single-threaded-ish:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/analysis/randomized_rank.py

It finishes in well under a minute and uses trivial memory (the largest self-test matrix is 20000 x 256).

Pointing it at the real width-512 block operator (NEXT STEP -- not done here)
----------------------------------------------------------------------------
A width-512 MLP hidden-layer Jacobian block J_l is already available matrix-free in the project through
`flatocean.blocks`.  Build the per-sample structure and the block, then wrap its row generator exactly as
`flatocean.analyses.neuronwise.sigma1_lanczos` does:

    from flatocean import blocks
    st  = blocks.mlp_structure(net, X)            # net: the trained width-512 MLP, X: analysis samples
    blk = blocks.MLPLayerBlock(st, l, s)          # hidden layer l (0-based); rows (s*d) x p, p = n*(m+1)
    # J_l @ v  and  J_l.T @ u  from blk.rows(idx) (see sigma1_lanczos for the chunked matvec/rmatvec):
    op = LinearOp((blk.s * blk.d, blk.p), matvec=..., rmatvec=...)
    est = adaptive_randomized_rank(op, rel_tol=1e-9)

The readout block is `blocks.ReadoutBlock`, a CNN block is `blocks.CNNBlock`; all expose the same
`rows(idx)` interface, so the same wrapper applies.  The declared project tolerance is rel_tol = 1e-9
relative to the block's own sigma_1 (flatocean.linalg.DECLARED).  This estimator has been validated here
ONLY on synthetic matrices of known rank; it has NOT been run on any network operator.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import os

# Keep BLAS single-threaded-ish even if the caller forgot the env (must precede the numpy import to take
# effect; setdefault never overrides a value the caller already set).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from dataclasses import dataclass, field  # noqa: E402
from typing import Callable, Optional     # noqa: E402

import numpy as np  # noqa: E402

__all__ = ["LinearOp", "RankEstimate", "randomized_rank", "adaptive_randomized_rank"]


# ----------------------------------------------------------------------------- matrix-free operator
def _apply(fn, X, out_dim):
    """Apply a matvec/rmatvec closure `fn` to a column block X (in_dim x b) -> (out_dim x b), float64.

    Tries a single batched call fn(X); if the closure only accepts a 1-D vector (e.g. a scipy
    LinearOperator-style matvec) or returns the wrong shape, falls back to a per-column loop.  This lets
    the estimator accept both batched closures (V -> A @ V) and vector-only closures unchanged."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return np.asarray(fn(X), dtype=np.float64).reshape(-1)
    try:
        Y = np.asarray(fn(X), dtype=np.float64)
        if Y.shape == (out_dim, X.shape[1]):
            return Y
    except Exception:
        pass
    return np.stack([np.asarray(fn(X[:, j]), dtype=np.float64).reshape(-1) for j in range(X.shape[1])], axis=1)


class LinearOp:
    """A linear operator A (m x n) given only by its action, never as a dense matrix.

    matvec(V):  V (n x b) -> A @ V  (m x b).
    rmatvec(U): U (m x b) -> A.T @ U (n x b).  May be None only when the operator is square and symmetric
                (assume_symmetric=True), in which case A.T @ U is computed with matvec.
    Closures may be batched (accept a 2-D column block) or vector-only; both are handled by _apply."""

    def __init__(self, shape, matvec: Callable, rmatvec: Optional[Callable] = None,
                 assume_symmetric: bool = False, name: Optional[str] = None):
        self.shape = (int(shape[0]), int(shape[1]))
        self._matvec = matvec
        self._rmatvec = rmatvec
        self.assume_symmetric = bool(assume_symmetric)
        self.name = name
        if rmatvec is None and not (assume_symmetric and self.shape[0] == self.shape[1]):
            raise ValueError("rmatvec is required unless the operator is square and assume_symmetric=True")

    def matvec(self, V):
        return _apply(self._matvec, V, self.shape[0])

    def rmatvec(self, U):
        fn = self._rmatvec if self._rmatvec is not None else self._matvec   # symmetric fallback A.T = A
        return _apply(fn, U, self.shape[1])


# ----------------------------------------------------------------------------- the estimate record
@dataclass
class RankEstimate:
    """Result of a randomized numerical-rank estimate.

    rank             estimated numerical rank at rel_tol = #{recovered sigma > rel_tol * sigma_max}.
    singular_values  recovered leading singular values of A (descending), length = sketch size k.
    tail_ratio       largest REJECTED singular value / sigma_max (the resolution witness); 0.0 when no
                     recovered value falls below the cut (then the window did not bracket the tolerance and
                     the rank may be an under-estimate -- see `resolved`).
    gap              sigma_r / sigma_{r+1} at the rank cut (spectral-gap stability), None/inf at extremes.
    resolved         True when the sketch window strictly brackets the tolerance (rank < k), so the tail is
                     genuinely below rel_tol rather than truncated.
    settings         dict with k, q, seed, rel_tol, oversampling (and, for the adaptive call, the k history).
    """
    rank: int
    singular_values: np.ndarray
    tail_ratio: float
    gap: Optional[float]
    resolved: bool
    settings: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- core sketch + rank read-off
def _sketch(op: LinearOp, k: int, q: int, rng: np.random.Generator):
    """Randomized subspace iteration: returns (recovered singular values descending, range basis Q, effective k).
    q power iterations sharpen the captured subspace but SUPPRESS directions at relative magnitude (sigma/sigma_max)
    by (sigma/sigma_max)^(2q+1), so a near-cut cluster far below sigma_max is strongly attenuated in the sketch at the
    sketch sizes actually tried; this is why resolution is certified by the independent omitted-range residual
    (_omitted_range_residual) on Q, not by the recovered spectrum alone, and why the exact dimension-cap sketch uses q=0.
    (No claim is made that such a cluster is lost for ALL k; the certificate simply declines to resolve until the omitted
    range is provably below the cut.)"""
    m, n = op.shape
    k = int(min(k, m, n))
    if k <= 0:
        return np.zeros(0), np.zeros((m, 0)), 0
    Omega = rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(op.matvec(Omega))                    # m x k, orth(A Omega)
    for _ in range(q):                                       # A.T A power iteration, QR between applications
        Q2, _ = np.linalg.qr(op.rmatvec(Q))                  # n x k, orth(A.T Q)
        Q, _ = np.linalg.qr(op.matvec(Q2))                   # m x k, orth(A Q2)
    Bt = op.rmatvec(Q)                                       # n x k == (Q.T A).T
    sv = np.linalg.svd(Bt, compute_uv=False)                 # top-k singular values of A (descending)
    return sv, Q, k


def _omitted_range_residual(op: LinearOp, Q: np.ndarray, rng: np.random.Generator, n_probes: int = 6):
    """HMT a-posteriori estimate of the OMITTED range norm ||(I - Q Q^T) A|| by Gaussian probing.  With n_probes
    standard-Gaussian columns omega_i, E = sqrt(2/pi) * max_i ||(I - Q Q^T) A omega_i|| estimates that operator norm,
    and 10*E bounds it with probability >= 1 - 10^(-n_probes) (Halko-Martinsson-Tropp 2011, eq. 4.3).  All above-cut
    singular directions are captured by Q iff this residual is below rel_tol * sigma_max; this is the independent
    certificate of resolution (it does not rely on the recovered spectrum, which power iteration can under-resolve)."""
    m, n = op.shape
    Om = rng.standard_normal((n, n_probes))
    Y = op.matvec(Om)                                        # m x n_probes = A Omega
    if Q.shape[1]:
        Y = Y - Q @ (Q.T @ Y)                                # project out the captured range
    e = float(np.linalg.norm(Y, axis=0).max()) if Y.size else 0.0
    est = float(np.sqrt(2.0 / np.pi) * e)
    return est, 10.0 * est                                   # (point estimate, high-probability upper bound)


def _read_rank(sv: np.ndarray, rel_tol: float):
    """Numerical rank, tail ratio and spectral gap from a recovered spectrum."""
    if sv.size == 0 or sv[0] <= 0:
        return 0, 0.0, None
    smax = float(sv[0])
    r = int((sv > rel_tol * smax).sum())
    tail_ratio = float(sv[r] / smax) if r < sv.size else 0.0
    if 0 < r < sv.size:
        gap = float(sv[r - 1] / sv[r]) if sv[r] > 0 else float("inf")
    else:
        gap = None
    return r, tail_ratio, gap


def _certify_rank(sv: np.ndarray, rel_tol: float, delta: float, num_eps: float = 1e-14):
    """Decide the numerical rank at threshold tau=rel_tol ACCOUNTING FOR the residual bound `delta` (a high-probability
    upper bound on the omitted-range operator norm ||(I-QQ^T)A||, which upper-bounds the error in the recovered sigma_i
    AND the largest un-captured singular value).  With recovered sigma_i = sv_i, t1 = sv_0 and an allowance
    d = delta + num_eps*t1 for floating error, the sufficient conditions for a CERTIFIED rank r are
        sv_{r-1} > tau*(t1 + d)   (the r-th value is above the cut even if t1 is under-recovered by d), and
        sv_r + d <= tau*t1        (the (r+1)-th value, inflated by the residual, is below the cut).
    When both hold, rank r is certified.  Otherwise return the rank INTERVAL [r_lo, r_hi]:
        r_lo = #{sv_i > tau*(t1 + d)}   (certainly above the cut),
        r_hi = #{sv_i + d  > tau*t1}    (possibly above the cut),
    and resolved=False.  Returns (rank, resolved, r_lo, r_hi, margin_above, margin_below)."""
    if sv.size == 0 or sv[0] <= 0:
        return 0, True, 0, 0, None, None
    t1 = float(sv[0]); d = float(delta) + num_eps * t1; cut = rel_tol * t1
    r = int((sv > cut).sum())
    r_lo = int((sv > rel_tol * (t1 + d)).sum())               # certainly above the cut (true sigma >= recovered)
    # The largest above-cut singular value may lie OUTSIDE the captured range: `delta` bounds every un-captured (beyond-k)
    # true singular value.  If delta exceeds the cut, above-cut mass may be omitted -> the rank is unbounded above at this
    # k (not resolvable; grow k).  Only when delta <= cut are the beyond-k values all below the cut, so the rank is then
    # bounded by the recovered values possibly-above-cut within the residual allowance.
    if d > cut:
        resolved = False; r_hi = -1                           # sentinel: unbounded above (omitted range exceeds the cut)
    else:
        r_hi = int((sv + d > cut).sum())
        resolved = bool(r_lo == r_hi)
    tail = max(float(sv[r]) if r < sv.size else 0.0, d)       # largest value not counted in rank r (recovered tail or omitted)
    margin_above = float(sv[r - 1] - rel_tol * (t1 + d)) if r > 0 else None
    margin_below = float(cut - tail)                          # >=0 when everything beyond rank r is below the cut
    rank = r if resolved else r_lo
    return rank, resolved, r_lo, r_hi, margin_above, margin_below


def randomized_rank(op: LinearOp, rel_tol: float = 1e-9, target_rank: Optional[int] = None,
                    k: Optional[int] = None, oversampling: int = 10, q: int = 2,
                    seed: int = 0) -> RankEstimate:
    """Fixed-budget randomized numerical-rank estimate at relative tolerance rel_tol.

    The sketch size is k if given, else target_rank + oversampling, else min(m, n) (a full factorization).
    target_rank is only an upper-bound BUDGET (any comfortable over-estimate works); the returned rank is
    the measured quantity.  Use `adaptive_randomized_rank` when no budget is known."""
    m, n = op.shape
    if k is None:
        base = target_rank if target_rank is not None else min(m, n)
        k = base + oversampling
    k = int(min(k, m, n))
    rng = np.random.default_rng(seed)
    sv, Q, k = _sketch(op, k, q, rng)
    _, tail_ratio, gap = _read_rank(sv, rel_tol)
    smax = float(sv[0]) if sv.size else 0.0
    n_probes = 6
    omitted, omitted_bound = _omitted_range_residual(op, Q, np.random.default_rng((seed, 7919)), n_probes)
    rank, resolved, r_lo, r_hi, m_above, m_below = _certify_rank(sv, rel_tol, omitted_bound)  # certify with the HP BOUND
    return RankEstimate(rank=rank, singular_values=sv, tail_ratio=tail_ratio, gap=gap, resolved=resolved,
                        settings=dict(k=int(k), q=int(q), seed=int(seed), rel_tol=float(rel_tol),
                                      oversampling=int(oversampling), n_probes=int(n_probes),
                                      omitted_range_residual=omitted, omitted_range_bound=omitted_bound,
                                      omitted_range_rel=(omitted_bound / smax if smax > 0 else None),
                                      rank_interval=[r_lo, r_hi], margin_above=m_above, margin_below=m_below,
                                      probe_confidence=float(1.0 - 10.0 ** (-n_probes))))


def adaptive_randomized_rank(op: LinearOp, rel_tol: float = 1e-9, oversampling: int = 10, q: int = 2,
                             seed: int = 0, k0: int = 32, growth: float = 2.0,
                             reject_margin: Optional[int] = None, max_k: Optional[int] = None,
                             n_probes: int = 6) -> RankEstimate:
    """Randomized numerical-rank estimate that grows the sketch until the tolerance is CERTIFIED, or k reaches
    min(m, n) (exact), or a supplied max_k cap is hit (then the result is an unresolved lower bound).

    Certification is by the INDEPENDENT omitted-range residual, not by the recovered spectrum: via _certify_rank, the
    rank is certified only when the HIGH-PROBABILITY BOUND on ||(I - Q Q^T) A|| (Gaussian-probe estimate,
    _omitted_range_residual) is below rel_tol * sigma_max (no above-cut singular direction lies outside Q) AND the
    recovered spectrum's sufficient conditions hold within that residual allowance; otherwise a rank INTERVAL /
    unresolved status is returned.  This replaces stability-only certification, which was INSUFFICIENT: a near-cut
    cluster far below sigma_max (e.g. the diagonal spectrum [1, cluster near 1e-9]) can stay below the cut in the
    recovered spectrum at the tested k, so a stability rule reads it as rank 1 (no claim that the cluster is lost for all
    k).  The final dimension-cap sketch uses q=0 so such a cluster is recovered when the full range is available.  The
    probe confidence 1 - n_tests * 10^(-n_probes) accounts for the repeated adaptive tests; records the residual bound,
    the rank-decision margins and the interval."""
    m, n = op.shape
    full = min(m, n)
    kcap = full if max_k is None else int(min(max_k, full))
    if reject_margin is None:
        reject_margin = max(2, oversampling // 2)
    k = int(min(max(k0, oversampling + 1, 1), kcap))
    k_history, sv, kk, tail_ratio, gap = [], np.zeros(0), 0, 0.0, None
    resolved, attempt, omitted, omitted_bound = False, 0, None, None
    rank, r_lo, r_hi, m_above, m_below = 0, 0, 0, None, None
    while True:
        at_full = (k >= full)                                 # the whole range fits: use q=0 so near-cut dirs are not crushed
        rng = np.random.default_rng((seed, attempt))
        sv, Q, kk = _sketch(op, k, 0 if at_full else q, rng)
        _, tail_ratio, gap = _read_rank(sv, rel_tol)
        omitted, omitted_bound = _omitted_range_residual(op, Q, np.random.default_rng((seed, attempt, 7919)), n_probes)
        k_history.append(int(kk))
        # certify with the HIGH-PROBABILITY BOUND on ||(I-QQ^T)A|| (the quantity the probe confidence covers), accounting
        # for the approximation uncertainty in the rank decision (_certify_rank's sufficient conditions).
        rank, resolved, r_lo, r_hi, m_above, m_below = _certify_rank(sv, rel_tol, omitted_bound)
        if at_full:                                            # exact factorization (q=0, full range): trust the decision as-is
            break
        if resolved:                                           # the uncertainty does not straddle the cut
            break
        if kk >= kcap:                                         # hit the user cap below the full dimension: unresolved lower bound
            break
        k = int(min(np.ceil(k * growth), kcap))
        attempt += 1
    n_tests = len(k_history)                                   # union bound over the repeated adaptive tests
    confidence = float(1.0 - n_tests * 10.0 ** (-n_probes))
    return RankEstimate(rank=int(rank), singular_values=sv, tail_ratio=tail_ratio, gap=gap, resolved=bool(resolved),
                        settings=dict(k=int(kk), q=int(q), seed=int(seed), rel_tol=float(rel_tol),
                                      oversampling=int(oversampling), adaptive=True, k_history=k_history,
                                      reject_margin=int(reject_margin), n_probes=int(n_probes), n_tests=int(n_tests),
                                      probe_confidence=confidence, rank_interval=[int(r_lo), int(r_hi)],
                                      margin_above=m_above, margin_below=m_below, omitted_range_residual=omitted,
                                      omitted_range_bound=omitted_bound,
                                      omitted_range_rel=(omitted_bound / (float(sv[0]) if sv.size else 1.0))))


# ============================================================================= self-validation
def _matrix_with_spectrum(m, n, sv, seed):
    """Dense m x n matrix with prescribed singular values `sv` (len r <= min(m,n)); exact zero tail beyond r.
    Random orthonormal left/right factors.  Returned only so the self-test can form a DENSE reference SVD
    and matvec/rmatvec closures; the estimator itself receives only the closures."""
    rng = np.random.default_rng(seed)
    r = len(sv)
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]          # m x r, orthonormal columns
    V = np.linalg.qr(rng.standard_normal((n, r)))[0]          # n x r, orthonormal columns
    return (U * np.asarray(sv, dtype=np.float64)) @ V.T


def _closures(A):
    """matvec / rmatvec closures over a dense A -- the ONLY thing the estimator is given (never A itself)."""
    return (lambda V: A @ V), (lambda U: A.T @ U)


def _case(name, A, rel_tol, estimate_fn, sv_relerr_tol=1e-6):
    """Run one validation case: dense reference rank + spectrum, then the matrix-free estimate; report."""
    dense_sv = np.linalg.svd(A, compute_uv=False)                        # REFERENCE ANSWER (never given to the estimator)
    smax = float(dense_sv[0])
    known_rank = int((dense_sv > rel_tol * smax).sum())
    mv, rmv = _closures(A)
    op = LinearOp(A.shape, matvec=mv, rmatvec=rmv, name=name)
    est = estimate_fn(op)
    # precision of the recovered retained block against the dense reference (relative to sigma_max)
    r = est.rank
    if r > 0 and r <= dense_sv.size and r <= est.singular_values.size:
        sv_relerr = float(np.max(np.abs(est.singular_values[:r] - dense_sv[:r])) / smax)
    else:
        sv_relerr = float("nan")
    rank_ok = (est.rank == known_rank)
    sv_ok = np.isfinite(sv_relerr) and (sv_relerr <= sv_relerr_tol)
    tail_ok = (est.tail_ratio <= rel_tol) if est.resolved else True      # resolved => tail genuinely below cut
    passed = bool(rank_ok and sv_ok and tail_ok and est.resolved)
    khist = est.settings.get("k_history")
    extra = f" k_hist={khist}" if khist is not None else f" k={est.settings['k']}"
    gap = est.gap
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<34} shape={A.shape!s:<13} rel_tol={rel_tol:.0e}  "
          f"rank est/known={est.rank}/{known_rank}  sv_relerr={sv_relerr:.2e}  "
          f"tail_ratio={est.tail_ratio:.2e}  gap={('%.2e' % gap) if gap is not None else 'n/a':<8}"
          f"  resolved={est.resolved}{extra}")
    return passed


def _self_test():
    print("=" * 118)
    print("Randomized numerical-rank estimator -- validation on KNOWN matrices (matrix-free; estimator sees only matvec/rmatvec)")
    print("q=2 power iterations, oversampling=10, float64; dense SVD is the reference answer, never an input to the estimator")
    print("=" * 118)
    results = []

    # (a) exact low-rank, zero tail -- fixed-budget estimator with a rough over-estimate of the rank as budget
    A = _matrix_with_spectrum(300, 200, np.logspace(0, -2, 20), seed=1)            # true rank 20 at 1e-9
    results.append(_case("(a) exact low-rank r=20", A, 1e-9,
                         lambda op: randomized_rank(op, rel_tol=1e-9, target_rank=40, q=2, seed=1)))

    # (a2) tall exact low-rank (rows >> cols), fixed budget
    A = _matrix_with_spectrum(4000, 128, np.logspace(0, -1.5, 15), seed=2)         # true rank 15
    results.append(_case("(a2) tall exact r=15 (4000x128)", A, 1e-9,
                         lambda op: randomized_rank(op, rel_tol=1e-9, target_rank=30, q=2, seed=2)))

    # (b) low-rank + noise floor below the tolerance -- ~512 columns; adaptive (no rank given)
    sig = np.logspace(0, -3, 40)                                                   # 40 signal values, 1 .. 1e-3
    rng = np.random.default_rng(3)
    noise = 1e-7 * (0.5 + rng.random(min(800, 512) - 40))                          # flat noise floor ~1e-7
    A = _matrix_with_spectrum(800, 512, np.concatenate([sig, noise]), seed=3)      # true rank 40 at 1e-5
    results.append(_case("(b) low-rank+noise r=40 (800x512)", A, 1e-5,
                         lambda op: adaptive_randomized_rank(op, rel_tol=1e-5, q=2, seed=3)))

    # (c) smooth exponential-decay spectrum -- ~512 columns; numerical rank well defined at the tolerance; adaptive.
    # sigma_i = 10^(-0.3 i); the tolerance is placed at the geometric midpoint of the gap between sigma_19 and
    # sigma_20 (10^(-0.3*19.5) = 1.41e-6) so the rank cut is unambiguous -- a tolerance landing ON a singular value
    # (e.g. exactly 1e-6 = sigma_20) is ill-posed and would flip on sub-epsilon rounding of sigma_max.
    decay = np.power(10.0, -0.3 * np.arange(min(700, 512)))                        # sigma_i = 10^(-0.3 i)
    A = _matrix_with_spectrum(700, 512, decay, seed=4)                             # rank at 1.41e-6 = 20 (checked vs dense)
    results.append(_case("(c) exponential decay (700x512)", A, float(10.0 ** (-0.3 * 19.5)),
                         lambda op: adaptive_randomized_rank(op, rel_tol=float(10.0 ** (-0.3 * 19.5)), q=2, seed=4)))

    # (d) true rank EXCEEDS the initial sketch size -- adaptive growth must find it (k0=20, true rank 60)
    A = _matrix_with_spectrum(1000, 400, np.logspace(0, -2, 60), seed=5)           # true rank 60 at 1e-9
    results.append(_case("(d) adaptive growth r=60, k0=20", A, 1e-9,
                         lambda op: adaptive_randomized_rank(op, rel_tol=1e-9, q=2, seed=5, k0=20)))

    # (e) tall Jacobian-like block (rows >> cols) with a noise floor; adaptive
    sig = np.logspace(0, -2, 30)
    rng = np.random.default_rng(6)
    noise = 1e-8 * (0.5 + rng.random(256 - 30))
    A = _matrix_with_spectrum(20000, 256, np.concatenate([sig, noise]), seed=6)    # true rank 30 at 1e-6
    results.append(_case("(e) tall block r=30 (20000x256)", A, 1e-6,
                         lambda op: adaptive_randomized_rank(op, rel_tol=1e-6, q=2, seed=6)))

    # (f) symmetric PSD operator supplied with matvec ONLY (rmatvec omitted) -- exercises the symmetric path
    B = _matrix_with_spectrum(300, 300, np.logspace(0, -1, 25), seed=7)            # 25 nonzero singular values
    S = B @ B.T                                                                    # PSD, rank 25, eigvals = sigma(B)^2
    dense_sv = np.linalg.svd(S, compute_uv=False)
    known_rank = int((dense_sv > 1e-9 * dense_sv[0]).sum())
    op = LinearOp(S.shape, matvec=(lambda V: S @ V), rmatvec=None, assume_symmetric=True, name="(f) symmetric")
    est = adaptive_randomized_rank(op, rel_tol=1e-9, q=2, seed=7)
    sv_relerr = float(np.max(np.abs(est.singular_values[:est.rank] - dense_sv[:est.rank]) / dense_sv[0])) if est.rank else float("nan")
    passed_f = bool(est.rank == known_rank and sv_relerr <= 1e-6 and est.resolved and est.tail_ratio <= 1e-9)
    print(f"[{'PASS' if passed_f else 'FAIL'}] {'(f) symmetric PSD, matvec-only':<34} shape={S.shape!s:<13} rel_tol={1e-9:.0e}  "
          f"rank est/known={est.rank}/{known_rank}  sv_relerr={sv_relerr:.2e}  tail_ratio={est.tail_ratio:.2e}  "
          f"gap={('%.2e' % est.gap) if est.gap is not None else 'n/a':<8}  resolved={est.resolved} k_hist={est.settings.get('k_history')}")
    results.append(passed_f)

    # (g) REGRESSION: a near-cut cluster far below sigma_max.  Spectrum [1, 10 x 1.001e-9, 245 x 0.999e-9]; at rel_tol
    # 1e-9 the true rank is 11 (the "1" plus the ten 1.001e-9 values).  q=2 power iteration strongly attenuates the
    # ~1e-9 cluster at the sketch sizes tried, so a stability-only rule FALSELY resolves rank 1; the independent
    # omitted-range residual stays above the cut and forces growth to the full dimension, where the q=0 exact sketch
    # recovers rank 11.  Certification is by the residual bound (no claim the cluster is lost for all k).
    sv_h = np.concatenate([[1.0], np.full(10, 1.001e-9), np.full(245, 0.999e-9)])
    A = _matrix_with_spectrum(256, 256, sv_h, seed=11)               # true rank 11 at 1e-9
    results.append(_case("(g) near-cut cluster (rank 11 not 1)", A, 1e-9,
                         lambda op: adaptive_randomized_rank(op, rel_tol=1e-9, q=2, seed=0)))

    print("-" * 118)
    n_pass = sum(bool(x) for x in results)
    verdict = "ALL PASS" if n_pass == len(results) else "SOME FAILED"
    print(f"SUMMARY: {n_pass}/{len(results)} cases passed -- {verdict}")
    print("=" * 118)
    return n_pass == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
