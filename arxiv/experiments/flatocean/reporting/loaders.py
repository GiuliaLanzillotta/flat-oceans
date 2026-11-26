"""Loaders of result records for the figures, tables and summaries (no plotting; importing this module draws nothing).
Extracted unchanged from make_tier1_figures.py and make_transition_v2_figures.py on 2026-09-25."""
import json
import os

import numpy as np

from flatocean import runs
from flatocean.reporting.style import RT, RA, STAGE_ORDER, C

def load_transition(dataset, model, seeds=(0, 1, 2)):
    runs = {}
    for s in seeds:
        p = os.path.join(RT, f"{dataset}_{model}_seed{s}.json")
        if os.path.exists(p):
            runs[s] = json.load(open(p))
    return runs

def stage_records(runs, stage):
    """Resolve 'same_as_epoch' aliases; returns list of (seed, record)."""
    out = []
    for s, r in runs.items():
        rec = r["results"].get(stage)
        if rec is None:
            continue
        if "same_as_epoch" in rec:
            rec = next(v for v in r["results"].values() if v.get("epoch") == rec["same_as_epoch"])
        out.append((s, rec))
    return out

def stages_present(runs):
    return [st for st in STAGE_ORDER if stage_records(runs, st)]

def load_atomic(dataset, model, seeds=(0, 1, 2)):
    runs = {}
    for s in seeds:
        p = os.path.join(RA, f"{dataset}_{model}_seed{s}.json")
        if os.path.exists(p):
            runs[s] = json.load(open(p))
    return runs

def atomic_stage_layers(runs, stage, tol="1e-09"):
    out = []
    for s, r in runs.items():
        rec = r["results"].get(stage)
        if rec is None:
            continue
        if "same_as_epoch" in rec:
            rec = next(v for v in r["results"].values() if v.get("epoch") == rec["same_as_epoch"])
        out.append((s, rec["layers"]))
    return out


# ---- transition v2 records (32-probe protocol)
RV = os.path.join(runs.RESULTS_ROOT, "transition_v2")

NBINS = 160

BIN_CENTERS = 10.0 ** (-15 + 0.1 * (np.arange(NBINS) + 0.5))

STATE_NAMES = ("always inactive", "always active", "mixed")

STATE_COLS = (C["ink3"], C["blue"], C["orange"])

def load_transition_v2(dataset, model, seeds=(0, 1, 2)):
    runs, counts = {}, {}
    for s in seeds:
        p = os.path.join(RV, f"{dataset}_{model}_seed{s}.json")
        if os.path.exists(p):
            runs[s] = json.load(open(p)); counts[s] = np.load(p.replace(".json", "_counts.npz"))
    return runs, counts

def stages_present_v2(runs):
    """Stages present in every run, in chronological order of their (median) checkpoint epoch; names are labels."""
    present = [st for st in STAGE_ORDER if all(st in r["results"] for r in runs.values())]
    def ep(st):
        return np.median([r["results"][st].get("epoch", r["results"][st].get("same_as_epoch", 0)) for r in runs.values()])
    return sorted(present, key=ep)

def probe_medians(rec, key, radii):
    """median over the 32 probes of the named norm, per radius."""
    return np.array([np.median([p[key] for p in rec["probes"][str(r)]]) for r in radii])

def probe_range(rec, key, radii):
    return (np.array([min(p[key] for p in rec["probes"][str(r)]) for r in radii]),
            np.array([max(p[key] for p in rec["probes"][str(r)]) for r in radii]))

def cond_flip(counts, st, l, ri, min_gates=50):
    """P(flip | sign, bin) at radius index ri, pooled over 32 probes: flips / (32 * gates)."""
    H = counts[f"{st}_hist_by_sign"][l].astype(float)                # 2 x NBINS
    Fl = counts[f"{st}_flips_by_sign_bin"][l, ri].sum(0).astype(float)   # 2 x NBINS (sum over probes)
    nprobe = counts[f"{st}_flips_by_sign_bin"].shape[2]
    P = np.full_like(H, np.nan)
    ok = H >= min_gates
    P[ok] = Fl[ok] / (nprobe * H[ok])
    return P, H
