"""
New-batch exposure of an atomic ocean (Claim G, plan amendment 2026-09-17, result 5).

For a block with Jacobian J_l, the old set X (the fixed 10k analysis set) and a new batch X', the number of
old-ocean dimensions the new batch exposes is

    rank( J_l(X') restricted to K_l(X) ) = rank J_l(X u X') - rank J_l(X) = dim K_l(X) - dim K_l(X u X').

The script computes the exact atomic decomposition of layers 2, 3 and the readout on X u X' for
X' = the disjoint 10k comparison set of the training data ("comparison") and X' = the 10k test set ("test"),
at the declared stages, and reads the X-only decomposition from results/atomic (same code, same rank rule).
Layer 1 is skipped: it has full column rank on X in every seed at these stages except seed 1 at epoch 500, whose
20k check is in results/atomic/mnist_mlp_seed1_saturation.json.  This is conditional exposure on one more finite
batch; it does not bound the ocean of the data distribution.

    OCEAN_THREADS=16 python -m flatocean new-batch-exposure --seeds 0 1 2 --device cpu
"""
# Ported unchanged from experiments/exp_new_batch_exposure.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean new-batch-exposure).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os
import time

import torch
from flatocean import records, runs

from flatocean.analyses import persistence as KP
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs

OUT = runs.out_dir("atomic")
STAGES = ("high_accuracy", "mse_interpolation", "late_endpoint")
TOL = "1e-09"


def block_records(st, s, device):
    recs = {}
    for l in (1, 2):
        t0 = time.time()
        rec, sv, _, _ = O.atomic_layer(st, l, s, device)
        rec["seconds"] = time.time() - t0; recs[str(l + 1)] = rec
        c = rec[f"components@{TOL}"]
        print(f"    layer {l + 1}: rank {rec['rank_J'][TOL]} dimK {c['dim_K']} = fn {c['feature_null']} + dn {c['downstream_null']} + def {c['deficit']} "
              f"dead {rec['dead_units']} gap {rec['gap_J'][TOL]} ({rec['seconds']:.0f}s)", flush=True)
    rec, sv, _ = O.atomic_readout(st, s, device); recs["readout"] = rec
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stages", nargs="+", default=list(STAGES))
    ap.add_argument("--batches", nargs="+", default=["comparison", "test"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--tag", default="", help="run tag (gelu, leaky, ...)")
    a = ap.parse_args()
    for seed in a.seeds:
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"; rd = O.run_dir(a.dataset, a.model, seed, a.tag)
        stages = stage_epochs(rd); base = json.load(open(os.path.join(OUT, f"{name}.json")))["results"]
        out_path = os.path.join(OUT, f"{name}_new_batch.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {
            "protocol": "EXPERIMENTS_PLAN.md amendment 2026-09-17, result 5; identifier new-batch-exposure-v1",
            "rank_tolerance": TOL, "stages": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            srec = res["stages"].setdefault(stage, {"epoch": ep, "batches": {}})
            old = base[stage]
            if "same_as_epoch" in old:
                old = next(v for v in base.values() if v.get("epoch") == old["same_as_epoch"])
            old = {str(L["layer"]): L for L in old["layers"]}
            for batch in a.batches:
                run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device, sets=("analysis", batch))
                Xa, _, ia = run["data"]["analysis"]; Xb, _, ib = run["data"][batch]
                # the baseline on X is the authenticated stored factor of every block (identity manifest checked against the current
                # checkpoint, preprocessing and ordered analysis samples); the atomic record supplies the component split only
                base_spec = {}
                for blk in ("2", "3", "readout"):
                    out_ = KP.factor_spectrum(name, ep, blk, run, rd)
                    if out_ is None:
                        raise P.ProvenanceError(f"{name} epoch {ep} block {blk}: no stored factor to serve as the authenticated baseline")
                    base_spec[blk] = out_
                ident = P.analysis_identity(run, rd, ep, {"protocol": "new-batch-exposure-v2", "rank_tolerance": TOL, "batch": batch,
                                                          "new_index_sha256": P.sha256_index(ib) if batch != "test" else "test set, natural order",
                                                          "baseline_factor_sha256": {b: base_spec[b][3] for b in base_spec}})
                prev = srec["batches"].get(batch)
                if prev and prev.get("identity", {}).get("digest") == ident["digest"]:
                    print(f"== {name} {stage} (epoch {ep}) X u {batch}: complete under the current identity", flush=True); continue
                print(f"== {name} {stage} (epoch {ep}) X u {batch}", flush=True)
                X = torch.cat([Xa, Xb], 0); st = O.mlp_structure(run["net"], X)
                recs = block_records(st, X.shape[0], a.device)
                summary = {}
                for blk, r in recs.items():
                    sv, p_, s_, sha, status = base_spec[blk]; lad = O.rank_ladder(sv); rank_old = lad[TOL]; dimK_old = p_ - rank_old
                    o = old.get(blk, {}); co = o.get(f"components@{TOL}"); cn = r[f"components@{TOL}"]
                    exposed = r["rank_J"][TOL] - rank_old
                    summary[blk] = {"dim_K_old": dimK_old, "dim_K_old_ladder": {t: p_ - lad[t] for t in lad}, "baseline_source": f"stored factor ({status})",
                                    "dim_K_union": cn["dim_K"], "exposed_by_new_batch": exposed, "exposed_fraction": exposed / dimK_old if dimK_old else None,
                                    "components_old": co, "components_old_source": "UNVERIFIED: results/atomic record without identity manifest (component split and dead-unit count only; the rank and dim K come from the authenticated factor)", "components_union": cn,
                                    "dead_units_old": o.get("dead_units"), "dead_units_union": r["dead_units"], "gap_union": r["gap_J"][TOL],
                                    "dim_K_union_ladder": {t: r[f"components@{t}"]["dim_K"] for t in ("1e-06", "1e-09", "1e-12")}}
                    if co is not None and co["dim_K"] != dimK_old:
                        summary[blk]["atomic_record_disagrees_with_factor"] = co["dim_K"]
                    print(f"    {blk}: dim K {dimK_old} -> {cn['dim_K']}  exposed {exposed}", flush=True)
                srec["batches"][batch] = {"n_new": int(Xb.shape[0]), "identity": ident,
                                          "new_index_sha256": P.sha256_index(ib) if batch != "test" else "test set, natural order",
                                          "checkpoint_sha256": ident["checkpoint_sha256"], "summary": summary, "layers_union": recs}
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
        print("saved", out_path, flush=True)


