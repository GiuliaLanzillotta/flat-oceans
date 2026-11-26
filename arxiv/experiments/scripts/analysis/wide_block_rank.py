#!/usr/bin/env python
"""P5 step 2 -- matrix-free numerical rank of a WIDTH-512 MLP hidden-block Jacobian operator.

This file is ADDITIVE: it imports the already-verified estimator `randomized_rank.py` UNCHANGED and the
shared project modules (`ocean_lib`, `flatocean.blocks`, `flatocean.linalg`) read-only.  It trains nothing
and modifies no shared module.

Why matrix-free.  The width-512 hidden block J_l has p = 512*513 = 262,656 columns and s*d = 10000*10 =
100,000 rows, so a dense SVD (or the blocked-QR R factor, p x p = 552 GB) is out of reach.  The explicit
row generator `blocks.MLPLayerBlock.rows(idx)` materialises a (len(idx)*d) x p dense matrix (~21 GB per
1000 samples at p = 262,656), so it cannot be used at full width either.

The operator (genuinely matrix-free, never materialises a row block)
----------------------------------------------------------------------
Row (i, a) of J_l is  h_i^T kron (D_i[:, a])  with the (k, j) column at index k*n + j (blocks.py convention,
the same reshape as blocks.mlp_rows: einsum("ik,iaj->iakj", H, D)).  Hence for a parameter vector v reshaped
to V[k, j] (shape m1 x n):

    (J_l v)[i, a] = sum_{k, j} H[i, k] D[i, a, j] V[k, j]
                  = sum_j D[i, a, j] (H @ V)[i, j]          -> matvec: G = H @ V,  out = einsum("iaj,ijq->iaq", D, G)
    (J_l^T u)[k, j] = sum_{i, a} H[i, k] D[i, a, j] U[i, a] -> rmatvec: P = einsum("iaj,iaq->ijq", D, U), acc = H^T @ P

Only H (s x m1) and D (s x d x n) from `blocks.mlp_structure` are touched; no (s*d) x p matrix is ever formed.
The per-call working set is bounded by chunking over samples; the unavoidable arrays are the operand V (p x b),
the result (s*d x b) and the accumulator (m1 x n x b) == (p x b).  Everything is float64 / CPU.

Modes
-----
  --validate [--tag w64 --layer 2]   STEP B: on a SMALL block whose dense SVD IS feasible (default width-64,
                                      hidden block W3, p = 4160) check (1) matvec/rmatvec vs the explicit
                                      blocks.rows(idx) matrix and the adjoint identity <A v, u> = <v, A^T u>
                                      to ~1e-10, and (2) adaptive_randomized_rank recovers the SAME numerical
                                      rank as a dense SVD of the block.  Exits non-zero unless ALL checks pass.
  --run --seed S --blocks 2 3 readout  STEP C: adaptive_randomized_rank(op, rel_tol, q=2) on the width-512
                                      block(s); writes one JSON per (seed, block) to results/analysis/.

The estimator sees ONLY matvec / rmatvec; stored singular values are never its input.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import os

# BLAS threads: honour the launcher's env (jobs.sh exports OMP/MKL/OPENBLAS); default modest for a direct run.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse  # noqa: E402
import json      # noqa: E402
import resource  # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(os.path.dirname(_HERE))          # .../experiments
sys.path.insert(0, _EXP)                                # ocean_lib, flatocean, provenance
sys.path.insert(0, _HERE)                               # randomized_rank (sibling, unmodified)

from flatocean import ocean_lib as O # noqa: E402
from flatocean import provenance as P # noqa: E402
from flatocean import blocks, linalg                    # noqa: E402
from randomized_rank import LinearOp, adaptive_randomized_rank  # noqa: E402

torch.set_num_threads(int(os.environ.get("MKL_NUM_THREADS", "8")))
OUT = os.path.join(_EXP, "results", "analysis")


# ----------------------------------------------------------------------------- build block + matrix-free operator
def build_block(seed, tag, which, epoch=500, dataset="mnist", model="mlp"):
    run = O.load_run(dataset, model, seed, tag=tag, epoch=epoch, device="cpu")
    net = run["net"]; X = run["data"]["analysis"][0][:10000]
    st = blocks.mlp_structure(net, X)
    if str(which) == "readout":
        blk = blocks.MLPReadoutBlock(st)
    else:
        blk = blocks.MLPLayerBlock(st, int(which))
    return run, st, blk


def make_op(blk, name=None, chunk_bytes=2.0e9):
    """LinearOp (s*d, p) with matvec = J @ V and rmatvec = J^T @ U, matrix-free.

    Hidden block (MLPLayerBlock, has .D): row (i,a) col (k,j)=k*n+j is H[i,k]*D[i,a,j], so
        matvec:  G = H @ V (per sample);  out[i,a] = sum_j D[i,a,j] G[i,j]
        rmatvec: P[i,j] = sum_a D[i,a,j] U[i,a];  acc = H^T @ P
    Readout block (MLPReadoutBlock, no .D): row (i,a) col (k,c)=k*d+c is H[i,k]*delta_{a,c}, the identity
    downstream, so J acts as H independently per output channel:
        matvec:  out[i,a] = sum_k H[i,k] V[k,a] = (H @ V)[i,a]
        rmatvec: acc[k,c] = sum_i H[i,k] U[i,c] = (H^T @ U)[k,c]
    The per-chunk intermediate has shape (chunk, n, b); `chunk` is sized so chunk*n*b*8 <= chunk_bytes."""
    H = blk.H.detach().double().contiguous()            # (s, m1)
    s, m1 = H.shape; d = blk.d; p = blk.p; rows = s * d
    is_readout = not hasattr(blk, "D")
    n = d if is_readout else blk.n
    assert p == m1 * n, (p, m1, n)
    D = None if is_readout else blk.D.detach().double().contiguous()   # (s, d, n)

    def _chunks(b):
        c = max(1, int(chunk_bytes // (n * max(b, 1) * 8)))
        return [(i, min(i + c, s)) for i in range(0, s, c)]

    def matvec(V):
        V = np.asarray(V, dtype=np.float64)
        b = V.reshape(p, -1).shape[1]
        Vt = torch.from_numpy(np.ascontiguousarray(V.reshape(p, -1))).view(m1, n, b)
        out = torch.empty((s, d, b), dtype=torch.float64)
        for i0, i1 in _chunks(b):
            if is_readout:
                out[i0:i1] = torch.einsum("ik,kaq->iaq", H[i0:i1], Vt)   # V indexed (k, a)
            else:
                G = torch.einsum("ik,kjq->ijq", H[i0:i1], Vt)           # (c, n, b) == H_chunk @ V
                out[i0:i1] = torch.einsum("iaj,ijq->iaq", D[i0:i1], G)
        return out.reshape(rows, b).numpy()

    def rmatvec(U):
        U = np.asarray(U, dtype=np.float64)
        b = U.reshape(rows, -1).shape[1]
        Ut = torch.from_numpy(np.ascontiguousarray(U.reshape(rows, -1))).view(s, d, b)
        acc = torch.zeros((m1, n, b), dtype=torch.float64)
        for i0, i1 in _chunks(b):
            if is_readout:
                acc += torch.einsum("ik,iaq->kaq", H[i0:i1], Ut[i0:i1])  # (m1, d, b)
            else:
                P = torch.einsum("iaj,iaq->ijq", D[i0:i1], Ut[i0:i1])    # (c, n, b)
                acc += torch.einsum("ik,ijq->kjq", H[i0:i1], P)          # (m1, n, b)
        return acc.reshape(p, b).numpy()

    return LinearOp((rows, p), matvec=matvec, rmatvec=rmatvec, name=name or blk.name)


def _peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0   # ru_maxrss is KiB on Linux


# ----------------------------------------------------------------------------- STEP B: validate on a small block
def validate(seed=0, tag="w64", which=2, rel_tol=1e-9, dataset="mnist", model="mlp"):
    run, st, blk = build_block(seed, tag, which, dataset=dataset, model=model)
    op = make_op(blk, name=f"{tag}-{which}")
    s, d, p = blk.s, blk.d, blk.p; rows = s * d
    label = "readout block" if str(which) == "readout" else f"hidden block W{int(which)+1} (l={which})"
    nshow = getattr(blk, "n", d)
    print(f"STEP B validation: {tag} seed {seed}, {label}; "
          f"shape (s*d, p) = ({rows}, {p}); m1={blk.m1} n={nshow} d={d}", flush=True)
    rng = np.random.default_rng(0)

    def dense_matvec(v):                                 # explicit blk.rows(idx) matrix, chunked
        out = np.empty(rows); vt = torch.from_numpy(np.ascontiguousarray(v)).double()
        for i0 in range(0, s, 1000):
            idx = torch.arange(i0, min(i0 + 1000, s)); R = blk.rows(idx).double()
            out[i0 * d: i0 * d + R.shape[0]] = (R @ vt).numpy()
        return out

    def dense_rmatvec(u):
        acc = torch.zeros(p, dtype=torch.float64); ut = torch.from_numpy(np.ascontiguousarray(u)).double()
        for i0 in range(0, s, 1000):
            idx = torch.arange(i0, min(i0 + 1000, s)); R = blk.rows(idx).double()
            acc += R.T @ ut[i0 * d: i0 * d + R.shape[0]]
        return acc.numpy()

    v = rng.standard_normal(p); u = rng.standard_normal(rows)
    mv_op, mv_dn = op.matvec(v), dense_matvec(v)
    rmv_op, rmv_dn = op.rmatvec(u), dense_rmatvec(u)
    rel_mv = float(np.linalg.norm(mv_op - mv_dn) / np.linalg.norm(mv_dn))
    rel_rmv = float(np.linalg.norm(rmv_op - rmv_dn) / np.linalg.norm(rmv_dn))
    lhs, rhs = float(mv_op @ u), float(v @ rmv_op)
    adj = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    Vb = rng.standard_normal((p, 5))                     # batched matvec vs column-by-column dense
    mvb = op.matvec(Vb); mvb_dn = np.stack([dense_matvec(Vb[:, j]) for j in range(5)], 1)
    rel_mvb = float(np.linalg.norm(mvb - mvb_dn) / np.linalg.norm(mvb_dn))

    R = linalg.assembled_R(lambda idx: blk.rows(idx), s, p, "cpu")   # dense SVD reference (R has J's singular values)
    sv = linalg.svdvals(R); rank_dense = int((sv > rel_tol * sv[0]).sum())

    est = adaptive_randomized_rank(op, rel_tol=rel_tol, q=2)
    r = est.rank
    sv_relerr = float(np.max(np.abs(est.singular_values[:r] - sv[:r]) / sv[0])) if r > 0 else float("nan")

    c_mv = rel_mv < 1e-10; c_rmv = rel_rmv < 1e-10; c_adj = adj < 1e-10; c_mvb = rel_mvb < 1e-10
    c_rank = (est.rank == rank_dense); c_sv = np.isfinite(sv_relerr) and sv_relerr <= 1e-6; c_res = bool(est.resolved)
    print(f"  (1) matvec   vs explicit rows : rel err {rel_mv:.2e}   [{'PASS' if c_mv else 'FAIL'}] (<1e-10)", flush=True)
    print(f"      rmatvec  vs explicit rows : rel err {rel_rmv:.2e}   [{'PASS' if c_rmv else 'FAIL'}] (<1e-10)")
    print(f"      batched matvec (5 cols)   : rel err {rel_mvb:.2e}   [{'PASS' if c_mvb else 'FAIL'}] (<1e-10)")
    print(f"      adjoint <Av,u>=<v,A^Tu>   : rel err {adj:.2e}   [{'PASS' if c_adj else 'FAIL'}] (<1e-10)")
    print(f"  (2) numerical rank  est/dense : {est.rank}/{rank_dense}   [{'PASS' if c_rank else 'FAIL'}]")
    print(f"      recovered sv vs dense     : rel err {sv_relerr:.2e}   [{'PASS' if c_sv else 'FAIL'}] (<1e-6)")
    print(f"      estimator resolved        : {est.resolved}   [{'PASS' if c_res else 'FAIL'}]")
    print(f"      tail_ratio={est.tail_ratio:.2e}  gap={est.gap}  k_history={est.settings.get('k_history')}")
    ok = all([c_mv, c_rmv, c_adj, c_mvb, c_rank, c_sv, c_res])
    print(f"  VALIDATION: {'ALL PASS' if ok else 'FAILED'}", flush=True)
    return ok


# ----------------------------------------------------------------------------- STEP C: run on width-512
def _trim_sv(sv, r):
    sv = np.asarray(sv, dtype=float)
    lead = sv[:min(100, sv.size)]
    lo, hi = max(0, r - 5), min(sv.size, r + 15)
    return {"leading_100": [float(x) for x in lead],
            "around_cut": {"from_index": int(lo), "values": [float(x) for x in sv[lo:hi]]},
            "sigma_max": float(sv[0]) if sv.size else 0.0,
            "sigma_at_rank": float(sv[r - 1]) if 0 < r <= sv.size else None,
            "sigma_at_rank_plus_1": float(sv[r]) if 0 <= r < sv.size else None}


def run_block(seed, tag, which, rel_tol=1e-9, q=2, max_k=None, dataset="mnist", model="mlp"):
    run, st, blk = build_block(seed, tag, which, dataset=dataset, model=model)
    op = make_op(blk, name=f"{tag}-s{seed}-{which}")
    rows, p = op.shape
    print(f"STEP C: {tag} seed {seed}, block {which} (name {blk.name}); shape ({rows}, {p}); "
          f"rel_tol={rel_tol} q={q} max_k={max_k}", flush=True)
    t0 = time.time()
    est = adaptive_randomized_rank(op, rel_tol=rel_tol, q=q, max_k=max_k)
    wall = time.time() - t0
    kcap_eff = min(max_k, rows, p) if max_k is not None else min(rows, p)
    k_final = int(est.settings.get("k")); margin = k_final - int(est.rank)
    reject_margin = int(est.settings.get("reject_margin") or 5)
    # The cap only LIMITS the result if the sketch saturated at it (rank fills it, tail not resolved below the cap).
    # A cap that coincides with a genuinely resolved spectrum (a big margin of below-cut values and tail_ratio < rel_tol)
    # does NOT limit: the block spectra have a huge gap at the numerical rank, so any k above the true rank exposes it.
    cap_reached = bool(max_k is not None and est.settings.get("k_history") and est.settings["k_history"][-1] >= kcap_eff)
    cap_limited = bool(cap_reached and not est.resolved)
    resolved_trustworthy = bool(est.resolved and not cap_limited)
    S = est.settings
    # Frobenius norm ||J||_F (cheap, from H and D) upper-bounds ||J||_2 = sigma_max; it substantiates a RIGOROUS lower
    # bound on a cap-limited block: the randomized sketch UNDER-estimates true singular values, so if the smallest
    # recovered sigma exceeds rel_tol * ||J||_F (>= rel_tol * sigma_max) then at least k_final true values clear the cut.
    import torch as _t
    fro = None; lower_bound = None
    if str(which) != "readout":
        Hb, Db = blk.H, blk.D                                            # s x m1 ; s x d x n
        fro = float(_t.sqrt(((Hb ** 2).sum(1) * (Db ** 2).sum((1, 2))).sum()))
        sig_min_recovered = float(est.singular_values[min(est.rank, est.singular_values.size) - 1]) if est.singular_values.size else 0.0
        if cap_limited and fro > 0:
            lower_bound = {"rank_ge": int(est.rank), "smallest_recovered_sigma": sig_min_recovered,
                           "frobenius_sigma_max_upper_bound": fro,
                           "rel_threshold_vs_upper_bound": float(rel_tol * fro),
                           "rigorous": bool(sig_min_recovered > rel_tol * fro)}
    ident = P.analysis_identity(run, O.run_dir(dataset, model, seed, tag=tag), 500,
                                {"analysis": "wide_block_randomized_rank", "block": str(which), "rel_tol": rel_tol,
                                 "q": q, "max_k": max_k, "estimator": "randomized_rank.adaptive (omitted-range certified)"})
    rec = {"protocol": "P5 step 2: matrix-free randomized numerical rank (randomized_rank.adaptive, HMT) on the "
                       "width-512 MLP hidden-block Jacobian; operator validated by wide_block_rank.py --validate on width-64.",
           "dataset": dataset, "model": model, "tag": tag, "seed": int(seed), "epoch": 500, "stage": "late_endpoint",
           "block": str(which), "block_name": blk.name, "shape_rows": int(rows), "p": int(p),
           "n_samples": int(blk.s), "d": int(blk.d), "m1": int(blk.m1), "n": int(getattr(blk, "n", blk.d)),
           "rel_tol": float(rel_tol), "q": int(q), "max_k": (int(max_k) if max_k is not None else None),
           "rank": int(est.rank), "resolved": bool(est.resolved), "tail_ratio": float(est.tail_ratio),
           "gap": (float(est.gap) if est.gap is not None else None),
           "k_history": est.settings.get("k_history"), "k_final": k_final,
           "reject_margin": reject_margin, "below_cut_margin": int(margin),
           "cap_reached": cap_reached, "cap_limited": cap_limited,
           "resolved_trustworthy": resolved_trustworthy,
           # estimator certification fields (omitted-range bound + uncertainty-aware rank decision)
           "rank_interval": S.get("rank_interval"), "margin_above": S.get("margin_above"),
           "margin_below": S.get("margin_below"), "omitted_range_residual": S.get("omitted_range_residual"),
           "omitted_range_bound": S.get("omitted_range_bound"), "omitted_range_rel": S.get("omitted_range_rel"),
           "n_probes": S.get("n_probes"), "n_tests": S.get("n_tests"), "probe_confidence": S.get("probe_confidence"),
           "frobenius_sigma_max_upper_bound": fro, "lower_bound_substantiation": lower_bound,
           "identity": ident,
           "singular_values": _trim_sv(est.singular_values, est.rank),
           "wall_s": round(wall, 1), "peak_rss_gb": round(_peak_gb(), 2)}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"wide_rank_{tag}_s{seed}_{which}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  rank={est.rank}  resolved={est.resolved}  trustworthy={resolved_trustworthy}  cap_limited={cap_limited}  "
          f"below_cut_margin={margin}  tail_ratio={est.tail_ratio:.2e}  gap={est.gap}  "
          f"k_history={est.settings.get('k_history')}", flush=True)
    print(f"  sigma_max={rec['singular_values']['sigma_max']:.6e}  sigma_rank={rec['singular_values']['sigma_at_rank']}  "
          f"sigma_rank+1={rec['singular_values']['sigma_at_rank_plus_1']}")
    print(f"  wall={wall:.0f}s  peak_rss={_peak_gb():.1f} GB  -> {path}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="P5 step 2: matrix-free randomized rank of a wide MLP block.")
    ap.add_argument("--validate", action="store_true", help="STEP B: validate the operator on a small block.")
    ap.add_argument("--run", action="store_true", help="STEP C: run adaptive randomized rank on the block(s).")
    ap.add_argument("--dataset", default="mnist", help="ADDITIVE (default mnist = byte-identical): run family, e.g. mnist_vitb16 for the P3 ViT-B/16 head.")
    ap.add_argument("--model", default="mlp", help="ADDITIVE (default mlp = byte-identical).")
    ap.add_argument("--tag", default="w512")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", default="2", help="validate: 0-based hidden layer (2 = W3) or 'readout'.")
    ap.add_argument("--blocks", nargs="+", default=["readout", "2", "1"],
                    help="run: block ids, 0-based hidden layer (1=W2, 2=W3) or 'readout'.")
    ap.add_argument("--rel-tol", type=float, default=1e-9)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--max-k", type=int, default=None, help="run: memory-safety ceiling on the sketch size k.")
    a = ap.parse_args()
    if a.validate:
        which = a.layer if a.layer == "readout" else int(a.layer)
        ok = validate(seed=a.seed, tag=("w64" if a.tag == "w512" else a.tag), which=which, rel_tol=a.rel_tol,
                      dataset=a.dataset, model=a.model)
        sys.exit(0 if ok else 1)
    if a.run:
        for which in a.blocks:
            run_block(a.seed, a.tag, which, rel_tol=a.rel_tol, q=a.q, max_k=a.max_k,
                      dataset=a.dataset, model=a.model)
        return
    ap.error("choose --validate or --run")


if __name__ == "__main__":
    main()
