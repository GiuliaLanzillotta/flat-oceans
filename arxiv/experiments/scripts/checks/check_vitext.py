#!/usr/bin/env python3
"""Checker for the common-family RADIUS-GRID EXTENSION records (2026-11-22 radius extension).

usage: check_vitext.py <seed> [--dir DIR] [--expect-M 256]

Re-opens results/runs/ftvit_resisc45/common_family_s<seed>_radius_ext.{json,_arrays.pt}
(or the same pair under --dir, test only) and verifies, recomputing from the arrays:

  1. the in-run validation gate is recorded as PASSED (same directions / normalization /
     precision vs the stored battery arrays), with its margins within the fixed tolerance
     (abs <= 1e-12 OR rel <= 1e-9; never loosened here);
  2. every battery/mode of the SOURCE record is present with M == expect-M (default 256)
     and a drift_ext array of shape (M, 5, 3, 3) at radii [100, 300, 1000];
  3. the numerical-floor diagnostic is present for all three layers x three radii with
     finite floor_median <= floor_max, and the floor reference probes sit at machine
     floor (< 1e-9 relative) on the original grid;
  4. exceedance counts and floor-classification counts in the JSON are consistent with
     the persisted drift arrays (recomputed here from the arrays, tol 1e-3 relative,
     floor factor 10x as recorded);
  5. the ORIGINAL record files were not mutated: their sha256 NOW equals the pre-run
     sha256 persisted in the extension JSON.

Everything verified here is computed + automatically checked (NOT independent review).
Exit 0 with "CHECK PASS" on success; nonzero with "CHECK FAIL <name>" otherwise.
Writes check_vitext_s<seed>.json next to the checked record.
"""
import os, sys, json, hashlib, argparse
import numpy as np, torch

SP = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(SP))
RUN = os.path.join(EXP, "results", "runs", "ftvit_resisc45")
TOL = 1e-3
RGRID_EXT = [100.0, 300.0, 1000.0]
CURVES = ["actual", "frozen", "first_order", "actual_minus_frozen", "frozen_minus_first_order"]
METRICS = ["abs_rms_X", "rel_rms_X", "median_per_example_rel"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()

CHECKS = {}
def check(name, ok, detail):
    CHECKS[name] = {"pass": bool(ok), "detail": detail}
    print(("CHECK ok  " if ok else "CHECK FAIL ")+name+" :: "+str(detail)[:300], flush=True)
    return bool(ok)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int, choices=[0, 1, 2])
    ap.add_argument("--dir", default=RUN, help="TEST ONLY: directory holding the ext record pair")
    ap.add_argument("--expect-M", type=int, default=256)
    args = ap.parse_args(); st = f"s{args.seed}"; ok_all = True
    ext_json = os.path.join(args.dir, f"common_family_{st}_radius_ext.json")
    ext_arr = os.path.join(args.dir, f"common_family_{st}_radius_ext_arrays.pt")
    for p in (ext_json, ext_arr):
        if not os.path.exists(p):
            print(f"CHECK FAIL missing_output {p}"); sys.exit(1)
    j = json.load(open(ext_json))
    arrs = torch.load(ext_arr, map_location="cpu", weights_only=False)

    # 1. in-run validation gate recorded as passed, margins within the fixed tolerance
    v = j["validation"]
    g = v["checks"].get("validation_same_directions_all_batteries", {"pass": False, "detail": {}})
    ok_all &= check("validation_gate_recorded_passed",
                    v["passed"] is True and g["pass"] is True
                    and float(g["detail"]["max_reldev_absfail"]) <= 1e-9,
                    {"passed": v["passed"], "gate": g["detail"]})
    per_mode_gates = all(mrec["validation_gate"]["pass"] is True
                         and mrec["validation_gate"]["max_reldev_where_abs_exceeded"] <= 1e-9
                         for b in j["batteries_ext"].values() for mrec in b["modes"].values())
    ok_all &= check("validation_gate_per_mode", per_mode_gates, "all battery/mode gates pass")

    # 2. all source batteries/modes present with M == expect-M and correct array shapes
    src_json = j["provenance"]["source_record_json"]
    src = json.load(open(src_json))
    missing, badM, badshape = [], [], []
    for tag in src["batteries"]:
        if tag not in j["batteries_ext"]: missing.append(tag); continue
        for mode in ("family", "randsub", "generic"):
            mrec = j["batteries_ext"][tag]["modes"].get(mode)
            if mrec is None: missing.append(f"{tag}/{mode}"); continue
            if mrec["M"] != args.expect_M or len(mrec["per_direction"]) != args.expect_M:
                badM.append(f"{tag}/{mode}")
            t = arrs.get(f"battery_{tag}_{mode}_drift_ext")
            if t is None or tuple(t.shape) != (args.expect_M, 5, 3, 3): badshape.append(f"{tag}/{mode}")
            if j["batteries_ext"][tag]["layer"] == "mlp0":
                s = arrs.get(f"battery_{tag}_{mode}_partialfreeze_rel_ext")
                if s is None or tuple(s.shape) != (args.expect_M, 2, 3):
                    badshape.append(f"{tag}/{mode}/partialfreeze")
    ok_all &= check("batteries_modes_present_M",
                    not missing and not badM and not badshape
                    and j["M_probes_per_mode"] == args.expect_M
                    and list(np.asarray(arrs["RGRID_EXT"])) == RGRID_EXT
                    and list(arrs["curve_axis"]) == CURVES and list(arrs["metric_axis"]) == METRICS,
                    {"missing": missing, "badM": badM, "badshape": badshape,
                     "expect_M": args.expect_M})

    # 3. floor diagnostics present and sane
    fl = j["numerical_floor_diagnostic"]["layers"]; bad = []
    for L in ("mlp0", "mlp3", "head"):
        if L not in fl: bad.append(L); continue
        fa = arrs.get(f"floor_{L}_drift_fullgrid")
        if fa is None or tuple(fa.shape) != (fl[L]["K"], 5, 3, 13): bad.append(f"{L}/array")
        for r in RGRID_EXT:
            pr = fl[L]["per_radius"].get(f"r{r:g}")
            if pr is None or not np.isfinite(pr["floor_median"]) or not np.isfinite(pr["floor_max"]) \
               or pr["floor_median"] > pr["floor_max"]:
                bad.append(f"{L}/r{r:g}")
            elif fa is not None:   # recompute floor stats from the array
                ei = 10+RGRID_EXT.index(r); col = fa[:, 0, 1, ei].numpy()
                if abs(float(np.median(col))-pr["floor_median"]) > 1e-15*max(1.0, pr["floor_median"]) \
                   or abs(float(col.max())-pr["floor_max"]) > 1e-15*max(1.0, pr["floor_max"]):
                    bad.append(f"{L}/r{r:g}/recompute")
        if fl[L]["original_grid_actual_rel_max"] >= 1e-9: bad.append(f"{L}/orig_grid_floor")
    ok_all &= check("floor_diagnostics_present", not bad, {"bad": bad})

    # 4. exceedance + floor-classification counts recomputed from the arrays
    mism = []
    for tag, b in j["batteries_ext"].items():
        L = b["layer"]
        fmax = np.array([fl[L]["per_radius"][f"r{r:g}"]["floor_max"] for r in RGRID_EXT])
        for mode, mrec in b["modes"].items():
            d = arrs[f"battery_{tag}_{mode}_drift_ext"].numpy()
            afm = arrs[f"battery_{tag}_{mode}_at_floor_ext"].numpy().astype(bool)
            for ei, r in enumerate(RGRID_EXT):
                rel = d[:, 0, 1, ei]
                exm = rel > TOL
                if int(exm.sum()) != mrec["exceedance_count_rel_gt_1e-3_among_M"][f"r{r:g}"]:
                    mism.append(f"{tag}/{mode}/r{r:g}/exceed")
                af = rel <= 10.0*fmax[ei]
                cc = mrec["floor_classification_counts"][f"r{r:g}"]
                if (int(af.sum()) != cc["n_at_numerical_floor"]
                        or int((~af).sum()) != cc["n_genuine_output_change"]
                        or int((exm & ~af).sum()) != cc["n_exceed_genuine"]
                        or int((exm & af).sum()) != cc["n_exceed_at_floor"]
                        or not np.array_equal(af, afm[:, ei])):
                    mism.append(f"{tag}/{mode}/r{r:g}/classification")
            # per-direction classification strings consistent with the mask
            for i, pd in enumerate(mrec["per_direction"]):
                want = ["at_numerical_floor" if afm[i, ei] else "genuine_output_change"
                        for ei in range(3)]
                if pd["classification_ext"] != want:
                    mism.append(f"{tag}/{mode}/probe{i}/per_direction"); break
    ok_all &= check("counts_consistent_with_arrays", not mism, {"mismatches": mism[:20]})

    # 5. original record files not mutated (sha256 NOW vs pre-run sha persisted in the ext JSON)
    pj = j["provenance"]
    now_json = sha256(pj["source_record_json"]); now_arr = sha256(pj["source_record_arrays"])
    ok_all &= check("original_records_unmutated",
                    now_json == pj["source_record_json_sha256_pre_run"]
                    and now_arr == pj["source_record_arrays_sha256_pre_run"],
                    {"json_now": now_json[:16], "json_pre": pj["source_record_json_sha256_pre_run"][:16],
                     "arrays_now": now_arr[:16],
                     "arrays_pre": pj["source_record_arrays_sha256_pre_run"][:16]})

    out = {"checked": ext_json, "expect_M": args.expect_M,
           "label": "computed + automatically checked (checker recomputation; NOT independent review)",
           "checks": CHECKS, "passed": bool(ok_all)}
    rp = os.path.join(args.dir, f"check_vitext_{st}.json")
    with open(rp+".tmp", "w") as fo: json.dump(out, fo, indent=1)
    os.replace(rp+".tmp", rp)
    print(("CHECK PASS" if ok_all else "CHECK FAIL"), "->", rp, flush=True)
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
