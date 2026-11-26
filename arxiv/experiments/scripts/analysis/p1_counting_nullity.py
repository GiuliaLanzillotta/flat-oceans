"""P1 counting-nullity geometry (bounded, verified-method).

Measures the atomic decomposition of hidden blocks 2, 3 and the readout of the WIDTH-128 MNIST MLP as a function of the
number of FITTED samples N, in two conditions, on the FITTED subset and on a DISJOINT HELD-OUT exposure set.

Conditions:
  (a) subsample  -- the full-data model `mnist_mlp` (trained on all 60k), geometry RESTRICTED to the N-subset (no retrain).
  (b) retrain    -- `mnist_mlp_p1_sub<N>` (trained only on the N-subset).
Sample sets (both size N, balanced):
  fit  -- the fitted training subset: stratified_indices(y_train, N//10, 30000)  [nested across N; condition (b)'s own fit set].
  exp  -- disjoint held-out: stratified_indices(y_test, N//10, 40000)             [MNIST test, never trained on].

Per (condition, N, seed, set, block) it records the verified ocean_lib atomic decomposition at the declared tolerance
1e-9: counting_nullity = max(0, p - s*d), r_h, r_d, dim_K, feature_null, downstream_null, deficit, dead units.  Layer 1
(256->128, p=32896) is frozen-cheap with no kernel and is excluded.  Everything reuses ocean_lib.atomic_layer /
atomic_readout (validated: rows vs autograd, blocked-QR vs dense SVD), so no new numerical program is introduced; the
R factor is p x p (<=2.2 GB for the p=16512 hidden blocks), independent of s, so the sweep is bounded.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # experiments/
_AN = os.path.dirname(os.path.abspath(__file__))                                      # scripts/analysis (randomized_rank, wide_block_rank)
sys.path.insert(0, HERE); sys.path.insert(0, _AN)
from flatocean import tier1_train as T
from flatocean import ocean_lib as O
from flatocean.runs import run_dir
from flatocean import blocks as FB
from flatocean.linalg import numrank_matrix
import randomized_rank as RR
import wide_block_rank as WBR

TOL = "1e-09"                       # DECLARED tolerance key (flatocean.linalg ladder)
STAGE_EPOCH = 500                   # late endpoint
SUBSET_SEED = 30000                 # must match --train-subset-seed of the retrain cohort
EXPOSURE_SEED = 40000


def build_net(dataset, tag, seed, device):
    rd = run_dir(dataset, "mlp", seed, tag)
    config = json.load(open(os.path.join(rd, "config.json")))
    pre = torch.load(os.path.join(rd, "preprocessing.pt"), weights_only=False)
    ck = torch.load(os.path.join(rd, "checkpoints", f"epoch_{STAGE_EPOCH:04d}.pt"), map_location="cpu", weights_only=False)
    net = T.build_model(config, torch.float64)
    net.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    return net.to(device).eval(), pre, config


def w1_decomp(st, s, max_k, rel_tol=1e-9):
    """W1 (input layer, p=32896) counting-kernel decomposition, MATRIX-FREE (no dense ~26 GB SVD).

    r_h = rank([x;1]) and r_d are cheap dense SVDs of small matrices; rank_J is the verified matrix-free randomized
    estimator (wide_block_rank operator + randomized_rank).  dim_K = p - rank_J; structured = feature_null +
    downstream_null (affine / frozen-gate-flat building blocks), deficit = r_h*r_d - rank_J (counting / tangent null)."""
    blk = FB.MLPLayerBlock(st, 0, s)
    m1, n, p, d = blk.m1, blk.n, blk.p, blk.d
    rh = numrank_matrix(blk.H)[0][TOL]
    rd = numrank_matrix(blk.D.reshape(s * d, n))[0][TOL]
    est = RR.adaptive_randomized_rank(WBR.make_op(blk, name="W1"), rel_tol=rel_tol, q=1, max_k=max_k)
    rJ = int(est.rank)
    return {"p": p, "rows": s * d, "counting_nullity": max(0, p - s * d), "r_h": rh, "r_d": rd,
            "dim_K": p - rJ, "feature_null": n * (m1 - rh), "downstream_null": rh * (n - rd),
            "deficit": rh * rd - rJ, "rank_J": rJ, "rank_resolved": bool(est.resolved), "matrix_free": True}


def atomic_on(net, Xraw, idx, pre, device, w1=True, w1_max_k=16512):
    """Atomic decomposition of blocks 1(matrix-free),2,3,readout on the sample set Xraw[idx] (s = len(idx))."""
    X = T.apply_preprocessing(Xraw[idx], pre, torch.float64).to(device)
    st = O.mlp_structure(net, X)
    s = X.shape[0]
    out = {}
    if w1:
        out["layer1"] = w1_decomp(st, s, max_k=min(w1_max_k, s * st["d"] + 2000))
    for l in (1, 2):                                   # 0-based layers 2 and 3
        rec, _, _, _ = O.atomic_layer(st, l, s, device)
        c = rec[f"components@{TOL}"]
        out[f"layer{l+1}"] = {"p": rec["p"], "rows": rec["rows"], "counting_nullity": rec["counting_nullity"],
                              "r_h": rec["r_h"][TOL], "r_d": rec["r_d"][TOL], "dim_K": c["dim_K"],
                              "feature_null": c["feature_null"], "downstream_null": c["downstream_null"],
                              "deficit": c["deficit"], "dead_units": rec.get("dead_units")}
    rec, _, _ = O.atomic_readout(st, s, device)
    c = rec[f"components@{TOL}"]
    out["readout"] = {"p": rec["p"], "rows": rec["rows"], "counting_nullity": rec["counting_nullity"],
                      "r_h": rec["r_h"][TOL], "dim_K": c["dim_K"], "feature_null": c["feature_null"],
                      "deficit": c["deficit"], "dead_units": rec.get("dead_units")}
    del st, X
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[1000, 2000, 5000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reference", action="store_true", help="also the full-data model on the 10k analysis set (large-N, ~0 counting nullity)")
    ap.add_argument("--quick", action="store_true", help="one combo (N=1000 seed 0, both conditions) for validation")
    ap.add_argument("--no-w1", action="store_true", help="skip the matrix-free W1 (input-layer) counting kernel")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "analysis", "p1_counting_nullity.json"))
    a = ap.parse_args()
    device = torch.device(a.device); W1 = not a.no_w1
    Ns = [1000] if a.quick else a.Ns
    seeds = [0] if a.quick else a.seeds

    Xtr, ytr, Xte, yte, _ = T.load_raw("mnist")
    fit_idx = {N: T.stratified_indices(ytr, N // 10, SUBSET_SEED) for N in Ns}
    exp_idx = {N: T.stratified_indices(yte, N // 10, EXPOSURE_SEED) for N in Ns}
    # nesting sanity
    for i in range(len(Ns) - 1):
        assert set(fit_idx[Ns[i]].tolist()) <= set(fit_idx[Ns[i + 1]].tolist()), "fit subsets not nested"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = {"meta": {"model": "mnist_mlp width128", "epoch": STAGE_EPOCH, "tol": TOL,
                        "subset_seed": SUBSET_SEED, "exposure_seed": EXPOSURE_SEED,
                        "seed_loop_outer": True, "N_ascending": True,
                        "conditions": {"subsample": "mnist_mlp (full data) restricted to subset",
                                       "retrain": "mnist_mlp_p1_sub<N> (trained on subset)"}},
               "subsample": {}, "retrain": {}}
    def flush():
        json.dump(results, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    flush()
    t0 = time.time()
    # Seed-outer, N-ascending: a COMPLETE single-seed result lands first; cheap high-counting-nullity combos first.
    for seed in seeds:
        neta, prea, _ = build_net("mnist", "", seed, device)          # condition (a): full-data model, reused across N
        for N in Ns:
            key = f"N{N}_seed{seed}"
            results["subsample"][key] = {"fit": atomic_on(neta, Xtr, fit_idx[N], prea, device, w1=W1),
                                         "exp": atomic_on(neta, Xte, exp_idx[N], prea, device, w1=W1)}
            flush(); print(f"[{time.time()-t0:6.0f}s] subsample {key} done", flush=True)
        del neta
        for N in Ns:                                                  # condition (b): subset-trained model
            key = f"N{N}_seed{seed}"
            try:
                netb, preb, _ = build_net("mnist", f"p1_sub{N}", seed, device)
            except FileNotFoundError:
                print(f"  retrain {key}: checkpoint missing (cohort still training?) -- skipped", flush=True); continue
            results["retrain"][key] = {"fit": atomic_on(netb, Xtr, fit_idx[N], preb, device, w1=W1),
                                       "exp": atomic_on(netb, Xte, exp_idx[N], preb, device, w1=W1)}
            flush(); print(f"[{time.time()-t0:6.0f}s] retrain {key} done", flush=True)
            del netb
        if a.reference and not a.quick:
            man = json.load(open(os.path.join(run_dir("mnist", "mlp", seed, ""), "data_manifest.json")))
            neta, prea, _ = build_net("mnist", "", seed, device)
            results.setdefault("subsample_reference_10k", {})[f"seed{seed}"] = atomic_on(neta, Xtr, np.array(man["analysis_indices"]), prea, device, w1=False)
            flush(); print(f"[{time.time()-t0:6.0f}s] reference 10k seed{seed} done", flush=True); del neta
    print("wrote", a.out, f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
