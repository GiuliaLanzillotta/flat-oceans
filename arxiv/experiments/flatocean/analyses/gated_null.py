"""
Unit-wise gated feature-null space of a hidden block and its share of the exact atomic kernel.

For hidden block l with parameters A = [W_l b_l], the gated feature-null space is

    G^gate_l = { A : S_{l,i} A [h_{l-1,i}; 1] = 0 for every sample i }
             = (+)_j { a_j : a_j . [h_{l-1,i}; 1] = 0 for every sample i on which unit j is active },

one block of incoming weights per unit j, orthogonal to the features that unit sees while it is active.  Its
dimension is sum_j (m + 1 - rank H_{A_j}), with H_{A_j} the augmented features of the samples in the active set
A_j of unit j.  Since D_{l,i} contains the gate S_{l,i}, G^gate_l is contained in the atomic kernel K_l.  It
contains the feature-null space (rows orthogonal to all features) and the incoming weights of dead units.  A
perturbation in G^gate_l leaves every post-activation of the frozen network unchanged, not only the outputs, so
the direct sum of the G^gate_l over layers (with the feature-null space of the readout) is a multilayer frozen
branch: no displacement propagates (R_l = 0 in the highest-perturbed-layer recursion).

The script reports, per run, stage and hidden block: dim G^gate_l, its split by unit type (dead units; units
active on fewer samples than m + 1; units with a rank-deficient feature set on a larger active set: each unit's
whole nullity goes to its type, so the rare-unit part is not a pure counting count), its split by mechanism
(rows of dead units; the counting part beyond them, sum_j max(0, m + 1 - |A_j|) minus the dead rows; the part
beyond counting, dim - sum_j max(0, m + 1 - |A_j|), which is rank deficiency of active features), the
feature-null dimension, and dim K_l from results/atomic when available (the remainder dim K_l - dim G^gate_l is
the part of the kernel that needs cancellation through the downstream map).  Since 2026-09-26 (protocol v2)
every stage record carries a full identity (checkpoint, preprocessing, the sample sets and their index hash,
the configuration) and the consumed atomic record is bound by its file hash and its stage epoch.

    python -m flatocean gated-null --seeds 0 1 2                    # every hidden layer, the input layer included (CPU)
    python -m flatocean gated-null --tag w64 --seeds 0 1 2 --set train --layers 2 3
    python -m flatocean gated-null --tag w64 --seeds 0 1 2
"""
# Ported unchanged from experiments/exp_gated_null.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean gated-null).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os

import torch
from flatocean import records, runs

from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs

OUT = runs.out_dir("atomic")
RTOL = 1e-9


def gated_null(H, S):
    m1 = H.shape[1]; units = []
    svH = torch.linalg.svdvals(H); r_h = int((svH > RTOL * svH[0]).sum())
    for j in range(S.shape[1]):
        a = S[:, j] > 0; na = int(a.sum())
        if na == 0:
            units.append({"active": 0, "rank": 0}); continue
        sv = torch.linalg.svdvals(H[a]); r = int((sv > RTOL * sv[0]).sum())
        gap = float(sv[r - 1] / max(float(sv[r]), 1e-300)) if r < sv.numel() else None
        units.append({"active": na, "rank": r, "gap": gap})
    dim = sum(m1 - u["rank"] for u in units)
    split = {"dead_units": sum(m1 for u in units if u["active"] == 0),
             "rarely_active_units": sum(m1 - u["rank"] for u in units if 0 < u["active"] < m1),
             "rank_deficient_on_large_active_set": sum(m1 - u["rank"] for u in units if u["active"] >= m1)}
    unit_counting = sum(max(0, m1 - u["active"]) for u in units)           # forced by counting alone: unit j has m+1 incoming weights and |A_j| constraints
    mechanism = {"dead_rows": split["dead_units"], "counting_beyond_dead": unit_counting - split["dead_units"], "beyond_counting": dim - unit_counting}
    counts = {"dead_units": sum(u["active"] == 0 for u in units), "rarely_active_units": sum(0 < u["active"] < m1 for u in units),
              "units_with_deficient_large_set": sum(u["active"] >= m1 and u["rank"] < m1 for u in units)}
    gaps = [u["gap"] for u in units if u.get("gap")]
    return {"dim_gated_null": dim, "unit_level_counting_nullity": unit_counting, "beyond_unit_counting": dim - unit_counting, "split": split, "split_mechanism": mechanism, "unit_counts": counts, "feature_null": S.shape[1] * (m1 - r_h), "r_h": r_h,
            "min_unit_gap": min(gaps) if gaps else None, "units": units}


PROTOCOL = "gated-null-v2 (2026-09-26: full identity per stage, mechanism split, atomic record bound by hash and epoch)"


def sample_identity(run, rd, ep, sets, config):
    """Identity of a gated-null stage record: checkpoint, preprocessing, the sample sets in order with the hash of their
    concatenated indices, and the configuration (analysis_identity of provenance.py is for the analysis set alone)."""
    import numpy as np
    idx = np.concatenate([np.asarray(run["data"][k][2]) for k in sets])
    ident = {"identity_version": P.IDENTITY_VERSION, "checkpoint": f"checkpoints/epoch_{ep:04d}.pt",
             "checkpoint_sha256": P.sha256_file(os.path.join(rd, "checkpoints", f"epoch_{ep:04d}.pt")),
             "preprocessing_sha256": P.sha256_file(os.path.join(run["dir"], "preprocessing.pt")),
             "sample_sets": list(sets), "sample_index_sha256": P.sha256_index(idx), "sample_size": int(len(idx)), "analysis_dtype": "float64", "config": config}
    ident["digest"] = P.identity_digest(ident)
    return ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--set", default="analysis", choices=["analysis", "train", "analysis+comparison"],
                    help="samples defining the gates and features: the 10k analysis set, its union with the comparison set, or the full training set")
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="hidden layers to measure (1-based; default: every hidden layer, the input layer included since 2026-09-26)")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag); stages = stage_epochs(rd)
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        ap_ = os.path.join(OUT, f"{name}.json"); atomic = json.load(open(ap_))["results"] if os.path.exists(ap_) else {}
        atomic_sha = P.sha256_file(ap_) if os.path.exists(ap_) else None
        res = {"protocol": PROTOCOL, "rank_tolerance": RTOL, "stages": {}}
        for stage, info in stages.items():
            ep = info["checkpoint_epoch"]
            if ep is None or (a.stages and stage not in a.stages):
                continue
            sets = tuple(a.set.split("+"))
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=sets)
            X = torch.cat([run["data"][k][0] for k in sets], 0)
            if getattr(run["net"], "activation", "relu") != "relu":
                raise ValueError("the gated feature-null space is a ReLU object (active set = positive derivative); for a smooth "
                                 "activation every derivative is nonzero and the construction does not apply")
            st = O.mlp_structure(run["net"], X); srec = {"epoch": ep, "n_samples": int(X.shape[0]), "blocks": {}}
            srec["identity"] = sample_identity(run, rd, ep, sets, {"protocol": PROTOCOL, "set": a.set, "layers": a.layers, "rank_tolerance": RTOL})
            srec["checkpoint_sha256"] = srec["identity"]["checkpoint_sha256"]
            arec = atomic.get(stage)
            if a.set != "analysis":
                arec = None
            if arec and "same_as_epoch" in arec:
                arec = next(v for v in atomic.values() if v.get("epoch") == arec["same_as_epoch"])
            if arec:
                # the atomic records carry no identity of their own: bind the consumed record by its file hash and require the stage epoch to match
                if arec.get("epoch") != ep:
                    raise P.ProvenanceError(f"{name} {stage}: atomic record epoch {arec.get('epoch')} differs from checkpoint epoch {ep}")
                srec["atomic_source"] = {"file": os.path.relpath(ap_, HERE), "sha256": atomic_sha, "stage": stage, "epoch": ep, "identity": "none in the atomic record; bound by file hash and epoch"}
            for l in range(0, st["L"]):                              # l = 0 is the input layer (block 1); before 2026-09-26 the loop began at 1
                if a.layers and (l + 1) not in a.layers:
                    continue
                g = gated_null(st["H"][l], st["S"][l])
                if arec:
                    c = arec["layers"][l]["components@1e-09"]; g["dim_K"] = c["dim_K"]; g["components"] = c
                    g["kernel_outside_gated_null"] = c["dim_K"] - g["dim_gated_null"]
                srec["blocks"][str(l + 1)] = g
                print(f"{name} {stage} (epoch {ep}, s={X.shape[0]}) layer {l + 1}: unit-level counting {g['unit_level_counting_nullity']}; gated-null {g['dim_gated_null']} = dead {g['split']['dead_units']} + rare {g['split']['rarely_active_units']} "
                      f"+ deficient {g['split']['rank_deficient_on_large_active_set']}; feature-null {g['feature_null']}; dim K {g.get('dim_K')}; min unit gap {g['min_unit_gap']}", flush=True)
            res["stages"][stage] = srec
        suffix = "" if a.set == "analysis" else "_" + a.set.replace("+", "_")
        records.dump_json(os.path.join(OUT, f"{name}_gated_null{suffix}.json"), res)


