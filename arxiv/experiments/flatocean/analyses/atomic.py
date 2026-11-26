"""Claim B (and the spectra of Claim D) at the declared training stages of an MLP.

Protocol: memos/2026-09-10_protocol_stages_atomic_transition.md, Sections 2
and 3.  For every seed and stage, every hidden layer and the readout: feature
rank, downstream rank, exact rank of the layer Jacobian block by blocked QR,
the three components, dead units, spectral gaps, the predeclared k_rel(tau)
and k(eps) profiles, and the full spectrum (saved to .npz).

Usage
    python -m flatocean atomic-stages --dataset mnist --model mlp --seeds 0 1 2
    python -m flatocean atomic-stages --dataset mnist --model mlp --seeds 0 --saturation
"""
# Ported unchanged from experiments/exp_atomic_stages.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean atomic-stages).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os
import sys
import time

import numpy as np
from flatocean import records, runs
import torch

from flatocean import ocean_lib as O
from flatocean.runs import stage_epochs, EXTRA_EPOCHS, EARLY_ERR, HIGH_ERR, INTERP_MSE   # noqa: F401  (moved to the package, 2026-09-24)

OUT = runs.out_dir("atomic")


def decompose(run, s, device, block, tag, verbose=True, select=None):
    net = run["net"]; X = run["data"]["analysis"][0][:s]
    st = O.mlp_structure(net, X)
    layers, spectra = [], {}
    for l in range(st["L"]):
        if select is not None and str(l + 1) not in select:
            layers.append({"layer": l + 1, "status": "excluded",
                           "note": "frozen block: displacement fixed to zero, atomic factorization skipped (NOT a measured zero-dimensional kernel)"})
            if verbose:
                print(f"  [{tag}] layer {l+1}: EXCLUDED (frozen)", flush=True)
            continue
        t0 = time.time()
        rec, sv, sv_h, sv_d = O.atomic_layer(st, l, s, device, block=block, verbose=(l == 0))
        rec["seconds"] = time.time() - t0; rec["status"] = "computed"
        layers.append(rec); spectra[f"J{l+1}"] = sv; spectra[f"H{l+1}"] = sv_h; spectra[f"D{l+1}"] = sv_d
        if verbose:
            c = rec["components@1e-09"]
            gap = rec['gap_J']['1e-09']
            print(f"  [{tag}] layer {l+1}: p={rec['p']} r_h={rec['r_h']['1e-09']} r_d={rec['r_d']['1e-09']} rank={rec['rank_J']['1e-09']} "
                  f"gap={'full rank' if gap is None else '%.1e' % gap} dimK={c['dim_K']} = fn {c['feature_null']} + dn {c['downstream_null']} + def {c['deficit']} "
                  f"dead={rec['dead_units']} ({rec['seconds']:.0f}s)", flush=True)
    if select is None or "readout" in select:
        rec, sv, sv_h = O.atomic_readout(st, s, device, block=block)
        rec["status"] = "computed"
        layers.append(rec); spectra["Jreadout"] = sv; spectra["Hreadout"] = sv_h
        if verbose:
            c = rec["components@1e-09"]
            print(f"  [{tag}] readout: p={rec['p']} r_h={rec['r_h']['1e-09']} rank={rec['rank_J']['1e-09']} dimK={c['dim_K']} = fn {c['feature_null']} + def {c['deficit']}", flush=True)
    else:
        layers.append({"layer": "readout", "status": "excluded",
                       "note": "frozen block: displacement fixed to zero, atomic factorization skipped (NOT a measured zero-dimensional kernel)"})
    # dead-unit shares: dead units of layer l enter the downstream-null term of layer l and the feature-null term of layer l+1
    for l in range(st["L"]):
        if layers[l].get("status") != "computed":
            continue
        dead = layers[l]["dead_units"]; rh = layers[l]["r_h"]["1e-09"]
        share = {"downstream_null_from_dead": rh * dead}
        nxt = layers[l + 1] if l + 1 < len(layers) else None
        if nxt is not None and nxt.get("status") == "computed" and "n_out" in nxt:
            share["feature_null_next_from_dead"] = nxt["n_out"] * dead
        layers[l]["dead_share"] = share
    return layers, spectra


def _test_stage_discovery():
    """A history through epoch 10 of a 500-epoch run: reached stages resolve, unreached ones are None."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "checkpoints"))
        for e in (0, 1, 2, 5, 10):
            open(os.path.join(d, "checkpoints", f"epoch_{e:04d}.pt"), "w").close()
        gates = [{"dead_units": 0}]
        E = [{"epoch": e, "train_errors": 60000 if e == 0 else 3000, "train_acc": 0.0 if e == 0 else 0.95, "train_mse": 0.1, "test_acc": 0.9, "test_mse": 0.1,
              "param_norm": {"total": 1.0}, "path_length": {"total": float(e)}, "gates": gates} for e in range(11)]
        json.dump({"config": {"n_params": 10, "dataset": "mnist", "epochs": 500}, "epochs": E}, open(os.path.join(d, "history.json"), "w"))
        st = stage_epochs(d)
        assert st["initialization"]["checkpoint_epoch"] == 0 and st["early_fitting"]["checkpoint_epoch"] == 1
        assert all(st[k]["checkpoint_epoch"] is None for k in ("high_accuracy", "mse_interpolation", "late_endpoint", "epoch_100", "epoch_250"))
    print("stage-discovery test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--saturation", action="store_true", help="sample-size saturation at the late endpoint of the given seed")
    ap.add_argument("--declared-only", action="store_true", help="skip the descriptive epochs (100, 250); declared stages only")
    ap.add_argument("--stages", nargs="*", default=None, help="compute these stages only (names of the stage rule); the record keeps stages computed earlier")
    ap.add_argument("--blocks", nargs="+", default=None, help="hidden-layer labels (1-based) and/or 'readout' to analyze, e.g. --blocks 2 3 readout; UNLISTED blocks are recorded EXCLUDED (frozen: displacement zero, atomic factorization skipped). Default: all.")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="", help="run tag (e.g. gelu, w64)")
    a = ap.parse_args()
    if a.test:
        return _test_stage_discovery()
    os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, a.tag)
        stages = stage_epochs(rd)
        if a.stages:
            _unknown = [x for x in a.stages if x not in stages]
            if _unknown:
                sys.exit(f"--stages: unknown stage name(s) {_unknown}; valid stages for this run are {sorted(stages)}")
        name = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}_seed{seed}"
        if a.saturation:
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=500, device=a.device, sets=("analysis", "comparison"))
            Xa, ya, ia = run["data"]["analysis"]; Xc, yc, ic = run["data"]["comparison"]
            res = {"protocol": "memos/2026-09-10_protocol_stages_atomic_transition.md", "epoch": 500, "sizes": {}}
            for s in (2000, 5000, 10000, 20000):
                if s <= 10000:
                    per = s // 10
                    keep = np.concatenate([np.where(ya.cpu().numpy() == c)[0][:per] for c in range(10)])
                    keep = np.sort(keep); X = Xa[torch.from_numpy(keep).to(a.device)]
                else:
                    X = torch.cat([Xa, Xc], 0)
                run["data"]["analysis"] = (X, None, None)
                print(f"== saturation seed {seed} s={s}", flush=True)
                layers, spectra = decompose(run, X.shape[0], a.device, a.block, f"s={s}")
                res["sizes"][str(s)] = layers
                np.savez_compressed(os.path.join(OUT, f"{name}_saturation_s{s}_spectra.npz"), **spectra)
            records.dump_json(os.path.join(OUT, f"{name}_saturation.json"), res)
            continue
        out_json = os.path.join(OUT, f"{name}.json")
        if os.path.exists(out_json):                      # resume: keep stages already computed
            res = json.load(open(out_json)); res["stages"] = stages
        else:
            res = {"protocol": "memos/2026-09-10_protocol_stages_atomic_transition.md", "stages": stages, "results": {}}
        done = {v["epoch"] for v in res["results"].values() if "epoch" in v}
        for stage, info in stages.items():
            ep = info["checkpoint_epoch"]
            if ep is None:
                print(f"== seed {seed} stage {stage}: not reached", flush=True); continue
            if a.declared_only and stage.startswith("epoch_"):
                continue
            if a.stages and stage not in a.stages:
                continue
            if stage in res["results"] and "layers" in res["results"][stage]:
                print(f"== seed {seed} stage {stage}: already computed (epoch {ep})", flush=True); continue
            if ep in done:
                res["results"][stage] = {"same_as_epoch": ep}; continue
            print(f"== seed {seed} stage {stage}: epoch {ep} (milestone {info['milestone_epoch']})", flush=True)
            run = O.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device=a.device)
            _sel = set(a.blocks) if a.blocks else None
            layers, spectra = decompose(run, 10000, a.device, a.block, f"seed {seed} ep {ep}", select=_sel)
            res["results"][stage] = {"epoch": ep, "layers": layers,
                                     "computed_blocks": [ly.get("layer") for ly in layers if ly.get("status") == "computed"],
                                     "excluded_blocks": [ly.get("layer") for ly in layers if ly.get("status") == "excluded"],
                                     "block_selection": (list(a.blocks) if a.blocks else "all")}
            np.savez_compressed(os.path.join(OUT, f"{name}_epoch{ep:04d}_spectra.npz"), **spectra)
            done.add(ep)
            records.dump_json(os.path.join(OUT, f"{name}.json"), res)
        records.dump_json(os.path.join(OUT, f"{name}.json"), res)


