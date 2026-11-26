"""P1 rank reconciliation with P5-style uncertainty/provenance (plan 2026-10-02 item 6).

READER-WRAPPER: reads a SNAPSHOT of the running results/analysis/p1_counting_nullity.json (held by another process,
PID 2815361) and writes a SEPARATE reconciled record results/analysis/p1_rank_reconciled.json.  It NEVER writes the
source file.

For every rank in every (condition, N, seed, set, block) it applies the P5 treatment (randomized_rank._certify_rank
conventions): a rank INTERVAL, the resolved flag, the method, an omitted-range note, and whether dim_K is exact or a
bound.  Two cases:

  layers 2, 3, readout     computed by ocean_lib.atomic_layer / atomic_readout as an EXACT dense SVD at tol 1e-9, so the
                           rank is resolved, rank_interval = [r, r], omitted_range_bound = 0, dim_K exact.
  layer 1 (W1)             the matrix-free randomized estimator (randomized_rank) with a max_k cap.  rank_J and its
                           resolved flag are serialized by the source script.  When resolved, dim_K is exact; when the
                           cap is hit (resolved False, e.g. N>=2000 W1), rank_J is a LOWER BOUND on the rank, so dim_K
                           = p - rank_J is an UPPER BOUND on the kernel dimension and the rank_interval is open above
                           ([rank_J, min(p, rows)]).  The source script stores only (rank_J, resolved), NOT the full
                           RankEstimate, so the omitted-range residual bound is not available here and is marked
                           'not_serialized_in_source' (re-run p1_counting_nullity W1 with the full estimate to certify).

Provenance identity per (condition, N, seed): the checkpoint sha256, the fitted-subset and (test) exposure index
sha256 reproduced from the declared stratified seeds, the epoch and the tolerance.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from flatocean import tier1_train as T
from flatocean import provenance as P
from flatocean.runs import run_dir

EXACT_BLOCKS = {"layer2", "layer3", "readout"}


def reconcile_block(blk_name, bd):
    p = bd["p"]; rows = bd["rows"]; dimK = bd["dim_K"]
    if blk_name == "layer1":
        rank = bd["rank_J"]; resolved = bool(bd["rank_resolved"]); method = "matrix_free_randomized"
        if resolved:
            interval = [rank, rank]; dimK_status = "exact"; rank_status = "exact"
        else:   # randomized max_k cap hit below the full dimension
            interval = [rank, int(min(p, rows))]; dimK_status = "upper_bound"; rank_status = "lower_bound"
        omitted = "not_serialized_in_source"   # source stores only (rank_J, resolved); full RankEstimate bound unavailable
    else:
        rank = p - dimK; resolved = True; method = "exact_dense_svd"
        interval = [rank, rank]; dimK_status = "exact"; rank_status = "exact"; omitted = 0.0
    return {"p": p, "rows": rows, "counting_nullity": bd.get("counting_nullity"),
            "rank": rank, "rank_status": rank_status, "rank_interval": interval, "resolved": resolved,
            "dim_K": dimK, "dim_K_status": dimK_status, "method": method, "omitted_range_bound": omitted,
            **{k: bd[k] for k in ("r_h", "r_d", "feature_null", "downstream_null", "deficit", "dead_units") if k in bd}}


def provenance(cond, N, seed, ytr, yte, epoch, tol):
    tag = "" if cond == "subsample" else f"p1_sub{N}"
    rd = run_dir("mnist", "mlp", seed, tag)
    ckpt = os.path.join(rd, "checkpoints", f"epoch_{epoch:04d}.pt")
    fit = T.stratified_indices(ytr, N // 10, 30000)
    exp = T.stratified_indices(yte, N // 10, 40000)   # source script's exposure set is the TEST set
    return {"run_dir": rd, "checkpoint": f"checkpoints/epoch_{epoch:04d}.pt",
            "checkpoint_sha256": P.sha256_file(ckpt) if os.path.exists(ckpt) else None,
            "fit_index_sha256": P.sha256_index(fit), "exp_index_sha256_testset": P.sha256_index(exp),
            "epoch": epoch, "tol": tol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "results", "analysis", "p1_counting_nullity.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "analysis", "p1_rank_reconciled.json"))
    a = ap.parse_args()
    src = json.load(open(a.src))
    meta = src["meta"]; epoch = meta["epoch"]; tol = meta["tol"]
    _, ytr, _, yte, _ = T.load_raw("mnist")
    out = {"meta": {**meta, "reconciled_from": os.path.relpath(a.src, HERE),
                    "source_sha256": P.sha256_file(a.src), "treatment": "P5 rank interval + resolved + method + bound status + provenance",
                    "note": "reader-wrapper; the source p1_counting_nullity.json is NEVER modified"},
           "subsample": {}, "retrain": {}}
    unresolved = []
    for cond in ("subsample", "retrain"):
        for key in sorted(src.get(cond, {})):
            N = int(key.split("_")[0][1:]); seed = int(key.split("seed")[1])
            rec = {"provenance": provenance(cond, N, seed, ytr, yte, epoch, tol)}
            for sset in ("fit", "exp"):
                if sset not in src[cond][key]:
                    continue
                rec[sset] = {}
                for blk, bd in src[cond][key][sset].items():
                    rb = reconcile_block(blk, bd); rec[sset][blk] = rb
                    if not rb["resolved"]:
                        unresolved.append(f"{cond} {key} {sset} {blk}: rank>={rb['rank']} (dim_K<={rb['dim_K']}, cap-limited)")
            out[cond][key] = rec
    out["meta"]["unresolved_ranks"] = unresolved
    json.dump(out, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}")
    print(f"combos: subsample {len(out['subsample'])}, retrain {len(out['retrain'])}; unresolved (lower-bound) ranks: {len(unresolved)}")
    for u in unresolved:
        print("  LOWER BOUND:", u)


if __name__ == "__main__":
    main()
