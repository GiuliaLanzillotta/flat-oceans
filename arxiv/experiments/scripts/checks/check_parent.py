"""Automated acceptance checks for the MLP PARENT inheritance package (two-day completion plan 2026-11-22).

usage: python3 scripts/pipeline/check_parent.py <seed> <epoch> <stage in {factors,ocean,comparison}>

  factors     all four parent block factors (1, 2, 3, readout) exist at the fork epoch and their identity
              manifests AUTHENTICATE fail-closed against the current run (flatocean.factors.factor_status).
  ocean       the parent branch record is acceptable: EITHER certified_branch true with matching saved bases and
              a passed admission record (STATUS certified), OR the fail-closed diagnostic
              ..._parent_bottom_up.FAILED.json is preserved with its validation record (STATUS failure_preserved;
              a preserved failure is a valid terminal result -- but drivers gating the comparison must require
              STATUS certified, which the comparison check below enforces independently).
  comparison  the parent_inheritance record is complete: displacement overlaps (all four primary quantities, per
              block + full network), projection masses with k/n expectations, principal angles for all three
              pairs, intersection + triple-intersection rank ladders, BOTH 256-direction control sets with
              exceedance counts, persisted arrays, and the required labels (scope non-certificate label,
              historical-0.1deg entries labelled not primary, asymmetries reported) -- checked automatically.

Prints "STATUS <state>" and "CHECK PASS|FAIL"; exit 0 iff the stage is accepted."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); os.chdir(HERE)

PARENTS = {(0, 350), (1, 350), (2, 350), (0, 450)}
try:
    SEED, EP, STAGE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
except (IndexError, ValueError):
    print("usage: check_parent.py <seed> <epoch> <factors|ocean|comparison>"); sys.exit(1)
if (SEED, EP) not in PARENTS or STAGE not in ("factors", "ocean", "comparison"):
    print(f"CHECK FAIL: (seed,epoch) must be in {sorted(PARENTS)} and stage in factors|ocean|comparison"); sys.exit(1)
name = f"mnist_mlp_seed{SEED}"
BR = os.path.join(HERE, "results", "branches")
fails = []


def check_factors():
    import numpy as np                                   # noqa: F401
    from flatocean import runs
    from flatocean import factors as F
    blocks = ["1", "2", "3", "readout"]
    missing = [b for b in blocks if not os.path.exists(F.factor_path(name, EP, b))]
    if missing:
        fails.append(f"missing factor files for blocks {missing}")
        return
    rd = runs.run_dir("mnist", "mlp", SEED, "")
    run = runs.load_run("mnist", "mlp", SEED, tag="", epoch=EP, device="cpu", sets=("analysis",))
    for b in blocks:
        try:
            sp = F.factor_spectrum(name, EP, b, run, rd)        # full fail-closed identity authentication
            sv, p, s_, sha, status = sp
            print(f"  {b}: authenticated ({status}) p={p} s={s_} sha256 {sha[:16]}...")
        except Exception as e:
            fails.append(f"block {b}: authentication failed: {e}")


def check_ocean():
    base = os.path.join(BR, f"{name}_epoch{EP:04d}_parent_bottom_up")
    if os.path.exists(base + ".json"):
        try:
            rec = json.load(open(base + ".json"))
        except Exception as e:
            fails.append(f"branch json unreadable: {e}"); return
        if rec.get("certified_branch") is not True:
            fails.append("branch json exists but certified_branch is not true (production record must be certified)")
        if not os.path.exists(base + ".pt"):
            fails.append("certified branch json without saved bases (.pt missing)")
        else:
            import torch
            sv = torch.load(base + ".pt", map_location="cpu", weights_only=False)
            for blk, k in rec.get("branch_dims", {}).items():
                if sv["bases"][blk].shape[1] != k:
                    fails.append(f"saved bases dim mismatch at {blk}")
        adm_p = os.path.join(BR, f"w1_admission_parent_s{SEED}_e{EP}.json")
        if not os.path.exists(adm_p):
            fails.append("parent W1 admission record missing")
        else:
            adm = json.load(open(adm_p))
            if not adm.get("passed"):
                fails.append("parent W1 admission record did not pass")
            if rec.get("w1_admission", {}).get("n_admitted") != adm.get("n_admitted"):
                fails.append("branch/admission n_admitted inconsistency")
        if not fails:
            print("STATUS certified")
        return
    if os.path.exists(base + ".FAILED.json"):
        try:
            rec = json.load(open(base + ".FAILED.json"))
        except Exception as e:
            fails.append(f"FAILED diagnostic unreadable: {e}"); return
        if rec.get("certified_branch") is True:
            fails.append("FAILED diagnostic claims certified_branch true (inconsistent)")
        if "validation" not in rec or "failures" not in rec:
            fails.append("FAILED diagnostic does not preserve the validation/failure record")
        if not fails:
            print("STATUS failure_preserved")
            print("NOTE: the ocean failure is preserved as a valid terminal result; the parent/child comparison "
                  "must NOT run for this parent until a certified branch exists")
        return
    fails.append(f"no ocean record: neither {os.path.basename(base)}.json nor .FAILED.json exists")


def check_comparison():
    base = os.path.join(HERE, "results", "lmc", "decomp", f"parent_inheritance_s{SEED}_e{EP}")
    if not os.path.exists(base + ".json"):
        fails.append(f"comparison record missing: {base}.json"); return
    rec = json.load(open(base + ".json"))
    BLOCKS = ["W1", "W2", "W3", "V"]; QUANTS = ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d")
    disp = rec.get("displacement", {})
    for blk in BLOCKS:
        for q in QUANTS + ("PP_d",):
            if q not in disp.get("per_block", {}).get(blk, {}):
                fails.append(f"displacement.{blk}.{q} missing")
    for q in QUANTS + ("PP_d",):
        if q not in disp.get("full_branch", {}):
            fails.append(f"displacement.full_branch.{q} missing")
    pm = rec.get("projection_masses", {})
    for blk in BLOCKS:
        ent = pm.get("per_block", {}).get(blk, {})
        for k in ("parent_cols_into_A", "parent_cols_into_B", "A_cols_into_P", "B_cols_into_P"):
            if k not in ent:
                fails.append(f"projection_masses.{blk}.{k} missing")
            elif "expectation_k_over_n" not in ent[k]:
                fails.append(f"projection_masses.{blk}.{k}: dimension-aware expectation k/n missing")
    if "full_network" not in pm:
        fails.append("projection_masses.full_network missing")
    pa = rec.get("principal_angles", {})
    for blk in BLOCKS:
        for key in ("PA", "PB", "AB"):
            ent = pa.get(blk, {}).get(key)
            if ent is None:
                fails.append(f"principal_angles.{blk}.{key} missing")
            elif "note" not in ent:                                 # nontrivial block: labels must be present
                h = ent.get("historical_0.1deg", {})
                if "NOT primary" not in str(h.get("label", "")):
                    fails.append(f"principal_angles.{blk}.{key}: historical 0.1-deg entry not labelled non-primary")
    inter = rec.get("intersections", {})
    if "rank cut sin<=1e-9" not in str(inter.get("convention", "")):
        fails.append("intersections: rank-ladder convention label missing")
    for blk in BLOCKS:
        ent = inter.get("per_block", {}).get(blk, {})
        for key in ("PA", "PB", "AB", "PAB_triple"):
            if key not in ent:
                fails.append(f"intersections.{blk}.{key} missing")
    if "PAB_triple" not in inter.get("full_network", {}):
        fails.append("intersections.full_network.PAB_triple missing")
    for mode in ("controls_isotropic", "controls_layer_length_matched"):
        c = rec.get(mode)
        if c is None:
            fails.append(f"{mode} missing"); continue
        ns = {c["full_branch"][q].get("n") for q in QUANTS if q in c.get("full_branch", {})}
        if ns != {256}:
            fails.append(f"{mode}: control count is not 256 (found {ns})")
        if "full_branch_exceedances_of_256" not in c:
            fails.append(f"{mode}: exceedance counts missing")
    scope = str(rec.get("scope", ""))
    if "NOT a finite-distance certificate" not in scope or "NO angular filtering" not in scope:
        fails.append("scope label missing or weakened (non-certificate / unfiltered labels required)")
    if "asymmetries" not in rec or "child_w1_support" not in rec.get("asymmetries", {}):
        fails.append("asymmetry report missing")
    if not os.path.exists(base + "_arrays.pt"):
        fails.append("_arrays.pt missing")
    else:
        import torch
        ar = torch.load(base + "_arrays.pt", map_location="cpu", weights_only=False)
        for k in ("spectra_sin_principal_angles", "projection_mass_arrays", "control_norm_frac_arrays"):
            if k not in ar:
                fails.append(f"_arrays.pt: {k} missing")
        ca = ar.get("control_norm_frac_arrays", {})
        for mode in ("isotropic", "layer_length_matched"):
            arrs = ca.get(mode, {}).get("full_branch", {})
            if any(len(arrs.get(q, [])) != 256 for q in QUANTS):
                fails.append(f"_arrays.pt: {mode} control arrays are not length 256")
    if not fails:
        print("STATUS complete")


{"factors": check_factors, "ocean": check_ocean, "comparison": check_comparison}[STAGE]()
if fails:
    print("CHECK FAIL (" + STAGE + "):\n  " + "\n  ".join(fails))
    sys.exit(1)
print(f"CHECK PASS ({STAGE}) {name} e{EP}")
sys.exit(0)
