"""Prepare the P1 affine-vs-tangent branch certificate on the EXACT fitted subsets.

For a given subset size N it writes:
  - results/analysis/p1_subset_idx_N<N>.npy : the fitted-subset TRAINING indices (stratified_indices(y, N//10, 30000)),
    i.e. exactly the set --train-subset trained on (and, for condition (a), the set the full-data model is restricted to).
  - results/atomic/<run_name>.json : a minimal authenticated atomic record that marks the INPUT layer W1 as
    status "excluded" (FIRST-LAYER-FROZEN certificate -- on a small subset W1, p=32896, has a large counting kernel, so
    it is held fixed rather than factorized densely), and lists W2, W3, readout with their true column counts p so the
    branch build authenticates against this network and factorizes those blocks (bounded on a small subset).

This lets `multilayer-branch build --indices ... --use-atomic-ranks --only bottom_up` run the hidden+readout branch on
the fitted subset without any dense W1 factorization, and `branch-rays rays --indices ...` then probes at matched
absolute radii (frozen-gate drift vs activation-transition drift).  The record carries no kernel dimensions for W2/W3
(the build measures them); only W1's exclusion and the per-block p are asserted.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from flatocean import tier1_train as T
from flatocean.runs import run_dir

SUBSET_SEED = 30000
STAGES = {"late_endpoint": 500}
ANALYSIS = os.path.join(HERE, "results", "analysis")
ATOMIC = os.path.join(HERE, "results", "atomic")


def subset_indices(N):
    _, ytr, _, _, _ = T.load_raw("mnist")
    return T.stratified_indices(ytr, N // 10, SUBSET_SEED)


def atomic_record(run_name, rd):
    cfg = json.load(open(os.path.join(rd, "config.json")))
    widths = cfg["widths"]; q = cfg["q"]; d_out = cfg["d_out"]
    # p = n_out * (n_in + 1): W1 q->w, W2 w->w, W3 w->w, readout w->d_out
    dims = [q] + list(widths)
    layers = [{"layer": 1, "status": "excluded", "p": int(widths[0] * (q + 1)),
               "note": "first-layer-frozen: input layer held fixed (large counting kernel on a small subset)"}]
    for l in range(1, len(widths)):
        layers.append({"layer": l + 1, "p": int(widths[l] * (dims[l] + 1))})
    layers.append({"layer": "readout", "p": int(d_out * (widths[-1] + 1))})
    results = {st: {"epoch": ep, "layers": layers} for st, ep in STAGES.items()}
    return {"protocol": "p1_first_layer_frozen_cert", "run": rd, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tags", nargs="+", required=True, help="run tags to write atomic records for (e.g. '' p1_sub1000)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--indices-tag", default=None, help="name suffix for condition-(a) subsample builds on the full-data run (tag ''), matching multilayer-branch --indices-tag, so the atomic record name does not collide with the full-data record")
    a = ap.parse_args()
    os.makedirs(ANALYSIS, exist_ok=True); os.makedirs(ATOMIC, exist_ok=True)
    idx = subset_indices(a.N)
    idx_path = os.path.join(ANALYSIS, f"p1_subset_idx_N{a.N}.npy")
    np.save(idx_path, idx); print("wrote", idx_path, "size", len(idx))
    for tag in a.tags:
        for s in a.seeds:
            rd = run_dir("mnist", "mlp", s, tag)
            if not os.path.exists(os.path.join(rd, "config.json")):
                print(f"  skip (no run): {rd}"); continue
            name = f"mnist_mlp{'_' + tag if tag else ''}{'_' + a.indices_tag if a.indices_tag else ''}_seed{s}"
            rec = atomic_record(name, rd)
            p = os.path.join(ATOMIC, f"{name}.json")
            if os.path.exists(p):
                print(f"  EXISTS, not overwriting: {p}"); continue
            json.dump(rec, open(p, "w"), indent=1); print("wrote", p)


if __name__ == "__main__":
    main()
