"""
Per-unit soft selectivity (GELU line, revised 2026-09-26, plan Section 0.8 item 10 (a)).

For hidden layer l with augmented inputs h_i (m + 1 coordinates) and units j, the single-unit block J_{l j} has the rows
sigma'_{ij} D_i[:, j] (x) h_i (m + 1 columns).  Its singular values are stored by the neuron-wise driver
(results/neuronwise/<run>_s<samples>_epoch<E>_unit_spectra.npz, key unit_sv_layer<l>, one row per unit).  Reported here:

  * the per-unit profile k_j(eps) = #{ sigma(J_{l j} / sqrt(s)) <= eps } for eps in a list, its sum over units, and the
    layer's absolute profile k(eps) on J_l / sqrt(s) from the atomic record (results/atomic/<run>.json, k_abs), so that
    sum_j k_j(eps) can be read against k(eps): for ReLU the exact kernel is unit-level (selective units) and the two
    agree up to the cross-neuron part; the question for GELU is whether its approximate ocean is unit-level too;
  * the per-unit count of tau-active samples, |sigma'_{ij}| > tau, for tau in a list (for ReLU sigma' is the gate, so the
    counts equal the active-set sizes for every tau < 1), with the derived tau-dead units (no tau-active sample), the
    tau-rare units (fewer tau-active samples than m + 1) and the counting nullity sum_j max(0, m + 1 - n_j(tau)).

Records results/neuronwise/<run>_soft_selectivity.json, one entry per stage and layer, with the analysis identity
(checkpoint, preprocessing, analysis samples, configuration), the hashes of the consumed spectra and atomic files, and
the neuron-wise record's identity digest.  The analysis set (10k) is used, as for the stored spectra.

    python -m flatocean soft-selectivity --tag gelu --seeds 0 1 2
    python -m flatocean soft-selectivity --seeds 0 1 2                 # ReLU reference
"""
import argparse
import json
import os

import numpy as np
import torch

from flatocean import provenance as P
from flatocean import blocks, records, runs
from flatocean.runs import stage_epochs

PROTOCOL = "soft-selectivity-v1 (2026-09-26)"
EPS = (1e-2, 1e-3, 1e-4, 1e-6)
TAUS = (1e-2, 1e-3, 1e-6)
STAGES = ("high_accuracy", "mse_interpolation", "epoch_250", "late_endpoint")


def key(x):
    return f"{x:g}"


def unit_profiles(unit_sv, s, eps=EPS):
    """unit_sv: n x (m + 1) singular values of the raw unit blocks; k_j(eps) counts those of J_{l j} / sqrt(s) at or below eps."""
    sv = np.asarray(unit_sv, dtype=np.float64) / np.sqrt(s)
    per_unit = {key(e): (sv <= e).sum(1).astype(int).tolist() for e in eps}
    return {"per_unit": per_unit, "sum_over_units": {k: int(sum(v)) for k, v in per_unit.items()}, "sigma1_unit_over_sqrt_s": sv.max(1).tolist()}


def tau_activity(S, m1, taus=TAUS):
    """S: s x n derivative values sigma'_{ij}.  Per tau: the tau-active count of every unit and the derived unit classes."""
    A = np.abs(np.asarray(S, dtype=np.float64)); out = {}
    for t in taus:
        n_act = (A > t).sum(0).astype(int)
        out[key(t)] = {"active_samples": n_act.tolist(), "dead_units": int((n_act == 0).sum()), "rare_units": int(((n_act > 0) & (n_act < m1)).sum()),
                       "counting_nullity": int(np.maximum(0, m1 - n_act).sum()), "median_active": float(np.median(n_act))}
    return out


def layer_profile(atomic_layer, eps=EPS):
    """The layer's absolute profile k(eps) from an atomic record's k_abs (keys are decimal strings), or None where a rung is missing."""
    ka = atomic_layer.get("k_abs") or {}
    by_val = {float(k): v for k, v in ka.items()}
    return {key(e): (int(by_val[e]) if e in by_val else None) for e in eps}


def main(argv=None):
    ap = argparse.ArgumentParser(description="per-unit soft selectivity from the stored unit spectra and a forward pass on the analysis set")
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="", help="'' = ReLU, gelu, leaky, ...")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3]); ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPS)); ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS))
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv); torch.set_num_threads(a.threads)
    out_dir = runs.out_dir("neuronwise")
    for seed in a.seeds:
        name = runs.run_name(a.dataset, a.model, seed, a.tag); rd = runs.run_dir(a.dataset, a.model, seed, a.tag)
        nw_path = os.path.join(out_dir, f"{name}_s{a.samples}.json"); at_path = os.path.join(runs.out_dir("atomic"), f"{name}.json")
        if not os.path.exists(nw_path):
            print(f"== {name}: no neuron-wise record {nw_path}; skipped", flush=True); continue
        nw = json.load(open(nw_path))["results"]; atomic = json.load(open(at_path))["results"] if os.path.exists(at_path) else {}
        at_sha = P.sha256_file(at_path) if os.path.exists(at_path) else None
        path = os.path.join(out_dir, f"{name}_soft_selectivity.json")
        rec = json.load(open(path)) if os.path.exists(path) and not a.overwrite else {"protocol": PROTOCOL, "run": name, "stages": {}}
        rec["protocol"] = PROTOCOL
        for stage in a.stages:
            nst = nw.get(stage)
            if not nst:
                print(f"== {name} {stage}: not in the neuron-wise record; skipped", flush=True); continue
            ep = int(nst["epoch"]); npz = os.path.join(out_dir, f"{name}_s{a.samples}_epoch{ep:04d}_unit_spectra.npz")
            if not os.path.exists(npz):
                print(f"== {name} {stage}: no unit spectra {npz}; skipped", flush=True); continue
            run = runs.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=("analysis",))
            X = run["data"]["analysis"][0]
            if X.shape[0] != a.samples:
                raise RuntimeError(f"{name}: the analysis set has {X.shape[0]} samples, the spectra were made with {a.samples}")
            config = {"protocol": PROTOCOL, "eps": [key(e) for e in a.eps], "taus": [key(t) for t in a.taus], "layers": a.layers, "samples": a.samples,
                      "spectra_sha256": P.sha256_file(npz), "atomic_sha256": at_sha, "neuronwise_digest": (nst.get("identity") or {}).get("digest")}
            ident = P.analysis_identity(run, rd, ep, config)
            old = rec["stages"].get(stage)
            if old and (old.get("identity") or {}).get("digest") == ident["digest"] and all(str(l) in old["layers"] for l in a.layers):
                print(f"== {name} {stage}: complete under the current identity", flush=True); continue
            st = blocks.mlp_structure(run["net"], X); srec = {"epoch": ep, "identity": ident, "activation": st["activation"], "layers": {}}
            ast = atomic.get(stage)
            if ast and "same_as_epoch" in ast:
                ast = next(v for v in atomic.values() if v.get("epoch") == ast["same_as_epoch"])
            if ast and int(ast.get("epoch", ep)) != ep:
                ast = None                                          # the atomic stage is a different checkpoint: no layer profile from it
            with np.load(npz) as z:
                for l in a.layers:
                    k = f"unit_sv_layer{l}"
                    if k not in z.files:
                        print(f"   {name} {stage} layer {l}: no unit spectra; skipped", flush=True); continue
                    sv = z[k]; S = st["S"][l - 1]; m1 = st["H"][l - 1].shape[1]
                    L = {"layer": l, "n_units": int(sv.shape[0]), "m_aug": int(m1), "samples": int(X.shape[0]), "profiles": unit_profiles(sv, X.shape[0], a.eps), "tau_activity": tau_activity(S.cpu().numpy(), m1, a.taus)}
                    al = next((x for x in (ast or {}).get("layers", []) if x.get("layer") == l), None)
                    L["layer_profile_k_abs"] = layer_profile(al, a.eps) if al else None
                    L["sum_over_layer"] = {e: (L["profiles"]["sum_over_units"][e] / L["layer_profile_k_abs"][e] if L["layer_profile_k_abs"] and L["layer_profile_k_abs"][e] else None) for e in L["profiles"]["sum_over_units"]}
                    nwl = (nst.get("layers") or {}).get(str(l)) or {}
                    L["neuronwise_reference"] = {"active_samples_recorded": (nwl.get("units") or {}).get("active_samples"), "selective_dim@1e-09": (nwl.get("units") or {}).get("selective_dim@1e-09")}
                    srec["layers"][str(l)] = L
                    ta = L["tau_activity"]; print(f"== {name} {stage} (ep {ep}) layer {l}: sum_j k_j(eps) {L['profiles']['sum_over_units']} against k(eps) {L['layer_profile_k_abs']}; "
                                                  f"tau-dead {[ta[key(t)]['dead_units'] for t in a.taus]}, tau-rare {[ta[key(t)]['rare_units'] for t in a.taus]}, counting nullity {[ta[key(t)]['counting_nullity'] for t in a.taus]}", flush=True)
            rec["stages"][stage] = srec; records.dump_json(path, rec)


if __name__ == "__main__":
    main()
