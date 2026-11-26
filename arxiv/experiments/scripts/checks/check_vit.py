#!/usr/bin/env python3
"""Automated acceptance check for the COMMON ViT FAMILY STUDY (plan memo 2026-11-22).

usage: check_vit.py <seed>

Checks results/runs/ftvit_resisc45/common_family_s<seed>.{json,_arrays.pt} (+ per-layer chunked
vector files): record exists and is complete (all 3 layers x all family classes with an explicit
computed/empty status), rank statements carry recorded gaps, every nonempty family has its battery
with the correct probe counts in all three modes, battery arrays have the right shapes, the JVP
spot-agreement gates pass, the structural frozen-vs-first-order residual is at float-noise level,
and the record's own validation block says passed.

This is an AUTOMATED check; a pass is labelled "automatically checked" and is NEVER independent
review. On success a sidecar results/runs/ftvit_resisc45/check_vit_s<seed>.json is written (result
files are never mutated). Exit 0/1 with a one-line verdict + JSON summary."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time

HERE = str(PROJECT_ROOT)
RUN = os.environ.get("CHECK_VIT_DIR",                      # TEST ONLY; production checks the run dir
                     os.path.join(HERE, "results", "runs", "ftvit_resisc45"))
LAYERS = ["mlp0", "mlp3", "head"]
FAMS = ["fn_exact", "fn_approx", "dn_exact", "dn_approx", "sel"]
OK_STATUS = {"computed_nonempty", "computed_empty_genuine_zero", "structurally_empty",
             "structurally_coincides_with_FN"}
MODES = ["family", "randsub", "generic"]
NRAD, NCURVE, NMET = 10, 5, 3

checks = {}
def ck(nm, ok, detail=""):
    checks[nm] = {"ok": bool(ok), "detail": str(detail)[:400]}
    if not ok: print(f"  FAIL {nm}: {detail}", flush=True)
    return bool(ok)

def main():
    seed = int(sys.argv[1]); st = f"s{seed}"
    jp = os.path.join(RUN, f"common_family_{st}.json")
    apth = os.path.join(RUN, f"common_family_{st}_arrays.pt")
    ok = ck("json_exists", os.path.exists(jp), jp)
    ok &= ck("arrays_exist", os.path.exists(apth), apth)
    if not ok:
        print(f"CHECK_VIT s{seed}: FAIL (missing outputs) [automatically checked]"); sys.exit(1)
    rec = json.load(open(jp))
    ok &= ck("seed_tag", rec.get("seed_tag") == st, rec.get("seed_tag"))
    ok &= ck("N_10000", rec.get("N_examples") == 10000, rec.get("N_examples"))
    M = rec.get("M_probes_per_mode")
    ok &= ck("M_256", M == 256, M)
    ok &= ck("dtype_float64", rec.get("dtype") == "float64")
    ok &= ck("provenance_shas", bool(rec.get("ckpt_sha256")) and bool(rec.get("cache_sha256")))
    nd = rec.get("normalization_denominators", {})
    ok &= ck("normalization_denominators", nd.get("sqrt_N") and nd.get("logits_rms_absolute"))

    batteries = rec.get("batteries", {})
    expected_batteries = set()
    for L in LAYERS:
        lay = rec.get("layers", {}).get(L)
        if not ck(f"{L}_present", lay is not None): ok = False; continue
        # rank statements with recorded gaps
        for opk in ("operator_H", "operator_C"):
            op = lay.get(opk, {})
            has_gap = ("gap" in op and "gap_resolved_ge_10x" in op
                       and "ambiguity_band_sigmas_within_10x_of_cut" in op) or op.get("structural")
            ok &= ck(f"{L}_{opk}_rank_gap_recorded", "rank" in op and has_gap,
                     {k: op.get(k) for k in ("rank", "gap", "structural")})
            ok &= ck(f"{L}_{opk}_units_stated", bool(op.get("units")) and bool(op.get("operator")))
        fams = lay.get("families", {})
        for fk in FAMS:
            fam = fams.get(fk)
            if not ck(f"{L}_{fk}_present", fam is not None): ok = False; continue
            if fk == "sel":
                stat_ex = fam.get("exact", {}).get("status"); stat_ap = fam.get("approx", {}).get("status")
                ok &= ck(f"{L}_sel_statuses_explicit",
                         stat_ex in OK_STATUS and stat_ap in OK_STATUS, (stat_ex, stat_ap))
                if stat_ap == "computed_nonempty":
                    expected_batteries.add(fam["approx"].get("battery"))
                if L != "head":
                    ok &= ck(f"{L}_sel_census_present", bool(fam.get("census")))
            else:
                stat = fam.get("status")
                ok &= ck(f"{L}_{fk}_status_explicit", stat in OK_STATUS or stat == "unresolved", stat)
                if stat == "computed_nonempty":
                    ok &= ck(f"{L}_{fk}_battery_named", bool(fam.get("battery")), fam.get("battery"))
                    expected_batteries.add(fam.get("battery"))
                elif stat == "computed_empty_genuine_zero":
                    bnd = [v for k, v in fam.items() if k.startswith("boundary")]
                    ok &= ck(f"{L}_{fk}_empty_has_boundary_rate",
                             any(isinstance(b, (int, float)) for b in bnd) or fam.get("k_factor", fam.get("k")) == 0,
                             bnd)
        ok &= ck(f"{L}_N_marginal_present", "N_marginal" in lay and "counting_bounds_not_computed_ranks" in lay)
    expected_batteries.discard(None)

    # batteries: probe counts, per-direction arrays, seeds, jv rates
    for tag in sorted(expected_batteries):
        b = batteries.get(tag)
        if not ck(f"battery_{tag}_present", b is not None): ok = False; continue
        for mode in MODES:
            mo = b.get("modes", {}).get(mode)
            if not ck(f"battery_{tag}_{mode}_present", mo is not None): ok = False; continue
            ok &= ck(f"battery_{tag}_{mode}_M256", mo.get("M") == 256 and len(mo.get("per_direction", [])) == 256,
                     (mo.get("M"), len(mo.get("per_direction", []))))
            ok &= ck(f"battery_{tag}_{mode}_seed_recorded", isinstance(mo.get("generator_seed"), int))
            ok &= ck(f"battery_{tag}_{mode}_exceedance_counts",
                     len(mo.get("exceedance_count_rel_gt_1e-3_among_256", {})) == NRAD)
            pd0 = (mo.get("per_direction") or [{}])[0]
            ok &= ck(f"battery_{tag}_{mode}_jv_rates_per_probe",
                     "jv_abs_rate_X_per_unit_v" in pd0 and "jv_rel_rate" in pd0)
    ok &= ck("no_unexpected_batteries", set(batteries) == expected_batteries,
             {"extra": sorted(set(batteries)-expected_batteries),
              "missing": sorted(expected_batteries-set(batteries))})

    # arrays: shapes + structural frozen-vs-Jv residual recheck
    import torch
    arr = torch.load(apth, map_location="cpu", weights_only=False)
    worst_ji = 0.0; worst_exact = 0.0
    for tag in sorted(expected_batteries):
        for mode in MODES:
            key = f"battery_{tag}_{mode}_drift"
            t = arr.get(key)
            if not ck(f"array_{key}_shape", t is not None and tuple(t.shape) == (256, NCURVE, NMET, NRAD),
                      None if t is None else tuple(t.shape)):
                ok = False; continue
            worst_ji = max(worst_ji, float(t[:, 4, 1, :].max()))
            if tag.endswith("_exact") and mode == "family":
                worst_exact = max(worst_exact, float(t[:, 0, 1, :].max()))
    ok &= ck("frozen_minus_first_order_float_noise", worst_ji < 1e-8, worst_ji)
    ok &= ck("exact_family_actual_drift_machine_precision", worst_exact < 1e-9, worst_exact)
    for sk in ("sv_feature_mlp0_u_aug", "sv_feature_mlp3_a_aug", "sv_feature_head_f_aug",
               "sv_downstream_mlp0_z0", "sv_downstream_mlp3_z3_nonshift",
               "G0_downstream_gram_mlp0", "G3_downstream_gram_mlp3", "nu_u", "nu_f"):
        ok &= ck(f"array_{sk}_persisted", sk in arr)
    for L, vf in rec.get("chunked_vector_files", {}).items():
        ok &= ck(f"vector_file_{L}_exists", os.path.exists(vf), vf)

    # in-run gates: JVP spot agreement and the rest of the validation block
    jg = rec.get("jvp_spot_gates", {})
    ok &= ck("jvp_spot_gates_pass",
             all(jg.get(L, {}).get("pass") is True for L in LAYERS),
             {L: jg.get(L, {}).get("pass") for L in LAYERS})
    val = rec.get("validation", {})
    ok &= ck("validation_block_passed", val.get("passed") is True, val.get("passed"))
    ok &= ck("validation_label_automated", "automatically checked" in str(val.get("label", "")))
    gates = val.get("checks", {})
    for g in ("exact_null_machine_precision", "battery_basis_orthonormality",
              "exact_ranks_vs_head_multilayer", "tail_counts_vs_phase0f",
              "gpu_cpu_downstream_chunk_agreement", "rank_gaps_resolved"):
        ok &= ck(f"gate_{g}", gates.get(g, {}).get("pass") is True, gates.get(g, {}).get("pass"))

    summary = {"target": f"common_family_{st}", "ok": bool(ok),
               "label": "automatically checked (automated acceptance script; NOT independent review)",
               "n_batteries": len(batteries), "batteries": sorted(batteries),
               "family_statuses": {L: {fk: (rec["layers"][L]["families"][fk].get("status")
                                            if fk != "sel" else
                                            {"exact": rec["layers"][L]["families"][fk]["exact"].get("status"),
                                             "approx": rec["layers"][L]["families"][fk]["approx"].get("status")})
                                       for fk in FAMS}
                                   for L in LAYERS if L in rec.get("layers", {})},
               "n_checks": len(checks), "n_failed": sum(1 for c in checks.values() if not c["ok"]),
               "failed": [k for k, c in checks.items() if not c["ok"]],
               "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if ok:
        side = os.path.join(RUN, f"check_vit_{st}.json")
        with open(side, "w") as fo: json.dump({"summary": summary, "checks": checks}, fo, indent=1)
    print(f"CHECK_VIT s{seed}: {'PASS' if ok else 'FAIL'} "
          f"({summary['n_checks']} checks, {summary['n_failed']} failed) [automatically checked]")
    print(json.dumps(summary, indent=1))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
