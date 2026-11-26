"""Automated acceptance checks for the interpolation ablation (PI 2026-11-23).

Recomputes from the SAVED record + spectra (results/newinput_interp/): per condition per block the rank
ladder from the stored union singular values, dim K_union = p - rank, the exposed arithmetic
exposed = dim K_X - dim K_union = rank_union - rank_X, the exposed fraction, the spectral gap, and the
ladder dims; cross-checks dim K_X against the authenticated clean new-batch record; checks the alpha=0
reproduction of that reference (dims per block, exactly); verifies every gap < 10 cell carries its
cutoff-sensitivity qualification (ladder range + band count, recomputed). Monotonicity of exposure in
alpha is NOT assumed: it is reported only (INFO, never a failure). Result files are never mutated; on
completion a sidecar results/newinput_interp/check_interp_seed<sd>.json records every check with the
label "computed + automatically checked". Exit 0/1 with a one-line verdict.

usage: check_interp.py <seed>    LIGHT (seconds; JSON + npz only, no model loading)"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, glob
import numpy as np

HERE = str(PROJECT_ROOT)
os.chdir(HERE)
NI = os.path.join(HERE, "results", "newinput_interp")
TOL = "1e-09"; TOLS = (1e-6, 1e-9, 1e-12); LADDER = ("1e-06", "1e-09", "1e-12")
GAP_RULE = 10.0; CUT = 1e-9
BLOCKS = ("2", "3", "readout")

seed = int(sys.argv[1])
name = f"mnist_mlp_seed{seed}"
checks = {}; info = []


def ck(nm, ok, detail=""):
    checks[nm] = {"ok": bool(ok), "detail": str(detail)[:400]}
    if not ok:
        print(f"  FAIL {nm}: {detail}", flush=True)
    return bool(ok)


def rank_ladder(sv):
    """Ladder counts, same rule as ocean_lib.rank_ladder (ocean_lib.py lines 38-44)."""
    sv = np.asarray(sv, dtype=np.float64)
    if sv.size == 0 or sv[0] == 0:
        return {str(t): 0 for t in TOLS}
    return {str(t): int((sv > t * sv[0]).sum()) for t in TOLS}


def spectral_gap(sv, r):
    """sigma_r / sigma_{r+1} (flatocean/linalg.py spectral_gap)."""
    sv = np.asarray(sv, dtype=np.float64)
    if r == 0 or r >= sv.size:
        return None
    return float(sv[r - 1] / sv[r]) if sv[r] > 0 else float("inf")


# ---- load the record: the full record, else merged partials (validation stage) ----
full = os.path.join(NI, f"{name}_interp.json")
conds = {}; sources = []
if os.path.exists(full):
    d = json.load(open(full)); conds = dict(d["conditions"]); sources = [os.path.relpath(full, HERE)]
else:
    for pth in sorted(glob.glob(os.path.join(NI, f"{name}_interp_partial_*.json"))):
        d = json.load(open(pth)); conds.update(d["conditions"]); sources.append(os.path.relpath(pth, HERE))
ok = ck("record_exists", bool(conds), f"no interp record for seed {seed} under results/newinput_interp/")
if not conds:
    print(f"CHECK FAIL seed {seed}: no record"); sys.exit(1)
info.append(f"sources: {sources}; conditions present: {sorted(conds, key=float)}")

nb = json.load(open(os.path.join(HERE, "results", "atomic", f"{name}_new_batch.json")))
tb = nb["stages"]["late_endpoint"]["batches"]["test"]["summary"]

for ckey in sorted(conds, key=float):
    c = conds[ckey]
    ok &= ck(f"a{ckey}_identity", bool(c.get("identity", {}).get("digest")))
    ok &= ck(f"a{ckey}_input_shas", all(c.get("input_sha256", {}).get(k) for k in ("y_alpha", "noise_u01", "clean_y01")))
    zp = None
    for blk in BLOCKS:
        e = c["summary"].get(blk)
        if not ck(f"a{ckey}_{blk}_present", e is not None):
            ok = False; continue
        tag = f"a{ckey}_{blk}"
        # spectra: recompute rank, dims, gap, ladder from the saved union singular values
        zrel = e.get("spectra_file", ""); zpath = os.path.join(HERE, zrel)
        if not ck(f"{tag}_spectra_exists", os.path.exists(zpath), zrel):
            ok = False; continue
        if zp is None or zp[0] != zpath:
            zp = (zpath, np.load(zpath))
        sv = np.sort(np.asarray(zp[1][blk], float))[::-1]
        lad = rank_ladder(sv); r9 = lad[TOL]; p = int(e["p"])
        ok &= ck(f"{tag}_rank_union", r9 == e["rank_union"], f"recomputed {r9} != recorded {e['rank_union']}")
        ok &= ck(f"{tag}_dim_K_union", p - r9 == e["dim_K_union"], f"{p - r9} != {e['dim_K_union']}")
        ok &= ck(f"{tag}_ladder", {t: p - lad[t] for t in lad} == e["dim_K_union_ladder"],
                 f"{ {t: p - lad[t] for t in lad} } != {e['dim_K_union_ladder']}")
        # exposed arithmetic: both routes
        ok &= ck(f"{tag}_exposed_dims", e["exposed"] == e["dim_K_X"] - e["dim_K_union"],
                 f"{e['exposed']} != {e['dim_K_X']} - {e['dim_K_union']}")
        ok &= ck(f"{tag}_exposed_ranks", e["exposed"] == e["rank_union"] - e["rank_X"],
                 f"{e['exposed']} != {e['rank_union']} - {e['rank_X']}")
        ok &= ck(f"{tag}_dims_consistent", e["dim_K_X"] == p - e["rank_X"] and e["dim_K_union"] <= e["dim_K_X"]
                 and e["exposed"] >= 0)
        fr = e["exposed"] / e["dim_K_X"] if e["dim_K_X"] else None
        ok &= ck(f"{tag}_fraction", (fr is None and e["exposed_fraction"] is None)
                 or (fr is not None and abs(fr - e["exposed_fraction"]) <= 1e-15), f"{fr} != {e['exposed_fraction']}")
        g = spectral_gap(sv, r9)
        ok &= ck(f"{tag}_gap", (g is None and e["gap_union"] is None)
                 or (g is not None and e["gap_union"] is not None
                     and (g == e["gap_union"] or abs(g - e["gap_union"]) <= 1e-9 * abs(g))),
                 f"recomputed {g} != recorded {e['gap_union']}")
        # baseline cross-check against the authenticated clean new-batch record
        ok &= ck(f"{tag}_dim_K_X_vs_reference", e["dim_K_X"] == tb[blk]["dim_K_old"],
                 f"{e['dim_K_X']} != new_batch dim_K_old {tb[blk]['dim_K_old']}")
        # gap flags: unresolved cells must carry the qualification; band recomputed
        if e["gap_union"] is not None and e["gap_union"] < GAP_RULE:
            q = e.get("cutoff_sensitivity")
            if ck(f"{tag}_qualification_present", q is not None, "gap < 10 without cutoff_sensitivity"):
                rel = sv / max(sv[0], 1e-300)
                band = int(((rel >= CUT / 10) & (rel <= CUT * 10)).sum())
                ldims = list(e["dim_K_union_ladder"].values())
                ok &= ck(f"{tag}_qualification_values",
                         q["band_sv_within_decade_of_cut"] == band
                         and q["dim_K_union_range"] == [min(ldims), max(ldims)],
                         f"recorded {q} vs recomputed band {band}, range {[min(ldims), max(ldims)]}")
            else:
                ok = False
            info.append(f"QUALIFIED a{ckey}/{blk}: gap {e['gap_union']:.1f} < 10, "
                        f"ladder range {[min(list(e['dim_K_union_ladder'].values())), max(list(e['dim_K_union_ladder'].values()))]}")

# ---- alpha=0 reproduction of the clean new-batch reference (exact dims per block) ----
c0 = conds.get("0.0")
if ck("alpha0_present", c0 is not None):
    for blk in BLOCKS:
        e = c0["summary"][blk]; ref = tb[blk]
        ok &= ck(f"alpha0_{blk}_reproduces_reference",
                 e["dim_K_X"] == ref["dim_K_old"] and e["dim_K_union"] == ref["dim_K_union"]
                 and e["exposed"] == ref["exposed_by_new_batch"],
                 f"({e['dim_K_X']},{e['dim_K_union']},{e['exposed']}) != "
                 f"({ref['dim_K_old']},{ref['dim_K_union']},{ref['exposed_by_new_batch']})")
        info.append(f"alpha0/{blk}: dim_K_X {e['dim_K_X']} dim_K_union {e['dim_K_union']} exposed {e['exposed']} "
                    f"(reference {ref['dim_K_old']}/{ref['dim_K_union']}/{ref['exposed_by_new_batch']}); "
                    f"gap float-equal: {e['gap_union'] == ref['gap_union']}")
else:
    ok = False

# ---- monotonicity: REPORTED ONLY, never assumed, never a failure ----
keys = sorted(conds, key=float)
for blk in BLOCKS:
    vals = [(k, conds[k]["summary"][blk]["exposed"]) for k in keys if blk in conds[k]["summary"]]
    viol = [f"{a}->{b}: {ea}->{eb}" for (a, ea), (b, eb) in zip(vals, vals[1:]) if eb < ea]
    info.append(f"monotonicity {blk} (INFO only): exposed over alpha {[v for _, v in vals]}; "
                + ("nondecreasing" if not viol else f"NOT monotone: {viol}"))

rep = {"target": f"seed{seed}", "passed": bool(ok), "label": "computed + automatically checked",
       "sources": sources, "checks": checks, "info": info}
json.dump(rep, open(os.path.join(NI, f"check_interp_seed{seed}.json"), "w"), indent=1)
for line in info:
    print("  " + line, flush=True)
print(f"CHECK {'PASS' if ok else 'FAIL'} seed {seed} "
      f"({sum(1 for c in checks.values() if c['ok'])}/{len(checks)} checks ok)", flush=True)
sys.exit(0 if ok else 1)
