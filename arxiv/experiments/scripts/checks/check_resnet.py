"""Automated acceptance checks for the COMMON ResNet family study (two-day plan 2026-11-22).

    python3 scripts/pipeline/check_resnet.py <target>      target in {cache, pilot, l42fc, l41, l40}

Checks completeness, identity SHAs (recomputed, fail closed), gap records, probe/battery shape
conventions, and the pilot's VJP agreement gates. Exit 0 on acceptance, 1 otherwise, with a one-line
verdict (label: automatically checked). This script only READS records; it never repairs them.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json
import numpy as np

HERE = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from resnet_common_lib import (CACHE_DIR, RESULTS_DIR, RGRID, METRICS, GROUPS, layer_label,
                               sha256_file, sha256_index)

FAM_CELLS = ("FN_X", "DN_X_additional", "FN_A", "DN_A", "SEL_additional")
STATUSES = {"computed", "genuine_zero", "structurally_empty", "structurally_coincides", "not_computed"}
ERRORS = []


def check(cond, msg):
    if not cond:
        ERRORS.append(msg)
    return bool(cond)


def check_cache():
    ip = os.path.join(CACHE_DIR, "cache_identity.json")
    if not check(os.path.exists(ip), f"missing {ip}"):
        return
    ident = json.load(open(ip))
    check(ident.get("n") == 10000, f"cache n = {ident.get('n')} != 10000")
    check(ident["index_sha256"] == sha256_index(np.arange(ident["n"], dtype=np.int64)),
          "index sha mismatch vs arange(n)")
    for key, entry in ident["files"].items():
        p = os.path.join(CACHE_DIR, entry["name"])
        if check(os.path.exists(p), f"missing cache file {p}"):
            check(sha256_file(p) == entry["sha256"], f"sha mismatch for cache file {key} ({p})")
    ckpt = ident["checkpoint"]["path"]
    check(os.path.exists(ckpt) and sha256_file(ckpt) == ident["checkpoint"]["sha256"],
          "checkpoint sha mismatch")
    st = ident.get("storage", {})
    check(st.get("xin_dtype") == "float32" and "fp64" in st.get("rationale", ""),
          "storage choice (fp32 disk + fp64-read path) not recorded")
    check("rounding_effect" in ident and "max_rel" in ident["rounding_effect"],
          "fp32-storage rounding effect not recorded")
    xin = np.load(os.path.join(CACHE_DIR, ident["files"]["xin"]["name"]), mmap_mode="r")
    check(xin.shape == (ident["n"], 1024, 14, 14) and xin.dtype == np.float32,
          f"xin shape/dtype wrong: {xin.shape} {xin.dtype}")
    import torch
    rl = torch.load(os.path.join(CACHE_DIR, ident["files"]["ref_local"]["name"]))
    check(tuple(rl.shape) == (ident["n"], 45) and rl.dtype == torch.float64,
          "ref_logits_local shape/dtype wrong")
    cen = torch.load(os.path.join(CACHE_DIR, ident["files"]["census"]["name"]))
    check(set(cen["counts"]) == {f"b{b}_{g}" for b in range(3) for g in ("g1", "g2", "gout")},
          "gate census incomplete")


def check_pilot():
    p = os.path.join(RESULTS_DIR, "contraction_pilot.json")
    if not check(os.path.exists(p), f"missing {p}"):
        return
    rep = json.load(open(p))
    check(rep.get("PASS") is True, "pilot PASS flag is not True")
    for g, v in rep.get("gates", {}).items():
        check(v is True, f"pilot gate failed: {g}")
    a = rep.get("a_frozen_r0_identity", {})
    check(a.get("max_rel", 1) <= 1e-12, f"r=0 identity max_rel {a.get('max_rel')}")
    check(a.get("frozen_forward_validation_record", {}).get("PASS") is True,
          "frozen_forward_validation.json not PASS")
    bb = rep.get("b_D_validation_vjp", {})
    check(bb.get("n_triples", 0) >= 8, f"only {bb.get('n_triples')} VJP triples (< 8)")
    check(bb.get("n_nonzero_columns", 0) >= 4, "fewer than 4 nonzero D columns among triples")
    check(bb.get("max_err_over_D_scale", 1) <= 1e-12, f"VJP err {bb.get('max_err_over_D_scale')}")
    cc = rep.get("c_spectra_reproducibility", {})
    for k in ("D_gram_l42conv3", "H_gram_l42conv3", "H_gram_fc_aug"):
        check(cc.get(k, 1) <= 1e-11, f"spectra repro {k} = {cc.get(k)}")
    ci = rep.get("cache_identity", {})
    ident = json.load(open(os.path.join(CACHE_DIR, "cache_identity.json")))
    check(ci.get("xin_sha256") == ident["files"]["xin"]["sha256"], "pilot cache xin sha != cache identity")
    check(ci.get("checkpoint_sha256") == ident["checkpoint"]["sha256"], "pilot ckpt sha != cache identity")
    check("projection_per_group" in rep and all(g in rep["projection_per_group"]
                                                for g in ("l42fc", "l41", "l40")),
          "per-group projections missing")
    check(os.path.exists(p.replace(".json", "_arrays.pt")), "pilot arrays file missing")


def check_group(group):
    import torch
    p = os.path.join(RESULTS_DIR, f"common_family_{group}.json")
    ap = p.replace(".json", "_arrays.pt")
    if not (check(os.path.exists(p), f"missing {p}") & check(os.path.exists(ap), f"missing {ap}")):
        return
    rep = json.load(open(p))
    arrays = torch.load(ap, weights_only=False)
    ident = json.load(open(os.path.join(CACHE_DIR, "cache_identity.json")))
    ci = rep.get("cache_identity", {})
    check(ci.get("xin_sha256") == ident["files"]["xin"]["sha256"], "group cache xin sha != cache identity")
    check(ci.get("checkpoint_sha256") == ident["checkpoint"]["sha256"], "group ckpt sha != cache identity")
    check(rep.get("N") == 10000, f"N = {rep.get('N')} != 10000")
    M = rep.get("M")
    check(M == 256, f"M = {M} != 256")
    check(rep.get("RGRID") == RGRID, "radius grid != common grid")
    for L in GROUPS[group]:
        lab = layer_label(L)
        if not check(lab in rep.get("layers", {}), f"layer {lab} missing"):
            continue
        lr = rep["layers"][lab]
        fams = lr.get("families", {})
        for cell in FAM_CELLS:
            if not check(cell in fams, f"{lab}: family cell {cell} missing"):
                continue
            check(fams[cell].get("status") in STATUSES,
                  f"{lab}/{cell}: status {fams[cell].get('status')} not in enum")
        check("counting_floor_bound" in lr, f"{lab}: counting floor bound missing")
        # gap records for every rank statement
        for opk in ("operator_H", "operator_D"):
            op = lr.get(opk, {})
            if "rank_record" in op:
                rk = op["rank_record"]
                check("gap" in rk and "gap_resolved" in rk, f"{lab}/{opk}: gap record missing")
            elif opk == "operator_D":
                check("statement" in op, f"{lab}/operator_D: neither rank_record nor structural statement")
            if "tail_info" in op:
                ti = op["tail_info"]
                check("tail_dim_abs_PRIMARY" in ti and "tail_dim_relX_recorded" in ti,
                      f"{lab}/{opk}: both T(ii) readings not recorded")
        # battery conventions
        computed = [c for c in FAM_CELLS if fams.get(c, {}).get("status") == "computed"]
        bat = lr.get("battery", {})
        if computed:
            check("generic" in bat, f"{lab}: generic control set missing")
        for c in computed:
            famsets = [k for k, v in bat.items() if v.get("family") == c]
            check(len(famsets) >= 2, f"{lab}/{c}: computed family needs family+randsub battery sets")
        for sname, srec in bat.items():
            check(srec.get("M") == M, f"{lab}/{sname}: M != {M}")
            check(len(srec.get("per_direction", [])) == M, f"{lab}/{sname}: per_direction != M")
            check(isinstance(srec.get("generator_seed"), int), f"{lab}/{sname}: seed missing")
            agg = srec.get("aggregates_per_radius", {})
            check(set(agg) == set(METRICS), f"{lab}/{sname}: aggregate metrics incomplete")
            for mname in METRICS:
                check(len(agg.get(mname, {})) == len(RGRID), f"{lab}/{sname}/{mname}: radii incomplete")
            for pd in srec.get("per_direction", [])[:3]:
                check("jv_rate_abs_X" in pd and "jv_rate_relX" in pd and "bracket_rel1e-3" in pd,
                      f"{lab}/{sname}: per-direction fields missing")
            akey = f"{lab}/{sname}_drift"
            if check(akey in arrays, f"arrays missing {akey}"):
                check(tuple(arrays[akey].shape) == (M, len(METRICS), len(RGRID)),
                      f"{akey} shape {tuple(arrays[akey].shape)}")
            jkey = f"{lab}/{sname}_jv_ex_norm"
            if check(jkey in arrays, f"arrays missing {jkey}"):
                check(tuple(arrays[jkey].shape) == (M, rep["N"]), f"{jkey} shape wrong")
        # spectra + Gram persistence
        if L != "fc":
            for k in ("sv_H", "sv_D", "G_H", "G_D", "gate_census"):
                check(f"{lab}/{k}" in arrays, f"arrays missing {lab}/{k}")
        else:
            check(f"{lab}/sv_H" in arrays and f"{lab}/G_H" in arrays, f"arrays missing {lab} spectra")
        # validation gates (when a battery ran on a conv layer)
        if computed and L != "fc":
            val = lr.get("validation", {})
            jr = val.get("jvp_route_check", {})
            check(jr.get("pass") is True, f"{lab}: jvp route check not passed")
            check(jr.get("n_probes", 0) >= 8, f"{lab}: < 8 jvp-route probes")
            check(val.get("affinity_pass") is True, f"{lab}: frozen-affinity (F0 + r*Jv) check failed")
        # direct re-measurement where an exact null was claimed
        for cell, opk in (("FN_X", "operator_H"), ("DN_X_additional", "operator_D")):
            f = fams.get(cell, {})
            if f.get("status") == "computed" and "direct_sigma_of_null_basis" in f:
                check(f["direct_sigma_of_null_basis"].get("pass") is True,
                      f"{lab}/{cell}: direct null re-measurement failed")
            elif f.get("status") == "computed" and cell == "FN_X":
                check(False, f"{lab}/{cell}: computed exact family without direct re-measurement")
    check(rep.get("meta", {}).get("status", "").startswith("computed"),
          "meta.status missing/not computed")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("cache", "pilot", "l42fc", "l41", "l40"):
        print("usage: check_resnet.py {cache|pilot|l42fc|l41|l40}")
        sys.exit(2)
    target = sys.argv[1]
    if target == "cache":
        check_cache()
    elif target == "pilot":
        check_pilot()
    else:
        check_group(target)
    if ERRORS:
        for e in ERRORS:
            print("FAIL:", e)
        print(f"VERDICT [{target}]: REJECTED (automatically checked) -- {len(ERRORS)} failure(s)")
        sys.exit(1)
    print(f"VERDICT [{target}]: ACCEPTED (automatically checked) -- completeness, identity SHAs, gap "
          f"records, probe/battery shapes{', VJP gates' if target == 'pilot' else ''} all pass")
    sys.exit(0)


if __name__ == "__main__":
    main()
