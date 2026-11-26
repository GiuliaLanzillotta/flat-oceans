"""Intersection dimensions of the exact constructed sibling spaces.

Uses persisted principal-angle spectra, the registered 1e-9 cut and tolerance
ladder. Displacement projections continue to use the full endpoint spaces.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, json
import numpy as np, torch
HERE = str(PROJECT_ROOT)
os.chdir(HERE)
D = "results/lmc/decomp"
TOLS = (1e-12, 1e-9, 1e-7, 1e-6, 1e-4)
PRIM = 1e-9; GAP = 10.0
out = {"date": "2026-11-17",
       "convention": "rank cut sin<=1e-9 primary; resolved iff >=10x gap at the cut; ladder "
                     "{1e-12,1e-9,1e-7,1e-6,1e-4} gives the defensible range; NOT the 0.1-deg definition; "
                     "projection analyses unchanged (no filtering)",
       "A_exact": {}}


def count_ladder(sin_sorted, zeros_exact=0):
    ent = {"zeros_exact": int(zeros_exact), "counts_at_tol": {}, }
    for t in TOLS:
        ent["counts_at_tol"][f"{t:g}"] = int(zeros_exact + (sin_sorted <= t).sum())
    k = int((sin_sorted <= PRIM).sum())
    below = sin_sorted[k - 1] if k > 0 else None
    above = sin_sorted[k] if k < sin_sorted.size else None
    gap = (float(above / below) if (below is not None and above is not None and below > 0) else None)
    ent["primary_dim"] = int(zeros_exact + k)
    ent["gap_at_primary_cut"] = gap
    ent["resolved"] = bool(gap is None or gap >= GAP or k == 0 or k == sin_sorted.size)
    if k == 0 and above is not None:
        ent["resolved"] = bool(above > PRIM * GAP)          # clean: everything clearly above the cut
    ent["smallest_nonzero_sin"] = float(sin_sorted[sin_sorted > 0][0]) if (sin_sorted > 0).any() else None
    ent["range_1e-9_to_1e-4"] = [ent["counts_at_tol"]["1e-09"], ent["counts_at_tol"]["0.0001"]]
    return ent


# ---------------- A ----------------
for fam, seeds, E, blocks in (("cnn", (0, 1, 2), 200, ["conv1", "conv2", "conv3", "conv4", "conv5", "readout"]),
                              ("mlp", (0, 1, 2), 350, ["W1", "W2", "W3", "V"])):
    out["A_exact"][fam] = {}
    for SD in seeds:
        base = (f"{D}/cnn_fullspace_overlap_s{SD}_e200" if fam == "cnn" else f"{D}/w1branch_overlap_s{SD}_e{E}")
        r = json.load(open(base + ".json")); ar = torch.load(base + "_arrays.pt", map_location="cpu", weights_only=False)
        spectra = ar["spectra_sin_principal_angles"]
        pa = r.get("principal_angles", {})
        p_full = int(r["p"])
        ent = {"source": base + ".json", "p": p_full, "per_block": {}}
        dimsA_tot = dimsB_tot = inter_tot = 0; f2_tot = 0.0
        inter_tot_hi = 0
        for b in blocks:
            a_ = pa.get(b, {})
            dA, dB = a_.get("dim_A"), a_.get("dim_B")
            if dA is None:                                   # zero-dim blocks absent from pa in CNN records
                dA = dB = 0
            if min(dA, dB) == 0:
                ent["per_block"][b] = {"dims": [dA, dB], "intersection": 0, "genuine_zero": True,
                                       "frobenius_sq": 0.0}
                continue
            sin = np.sort(np.asarray(spectra[b], dtype=float))
            assert sin.size == min(dA, dB), (fam, SD, b, sin.size, dA, dB)
            lad = count_ladder(sin)
            f2 = float(min(dA, dB) - (sin ** 2).sum())
            ent["per_block"][b] = {"dims": [dA, dB], "intersection_ladder": lad,
                                   "intersection": lad["primary_dim"],
                                   "intersection_frac_of_p_block": None,
                                   "frobenius_sq": f2,
                                   "frobenius_sq_over_min_dim": f2 / min(dA, dB)}
            dimsA_tot += dA; dimsB_tot += dB; inter_tot += lad["primary_dim"]
            inter_tot_hi += lad["counts_at_tol"]["0.0001"]
            f2_tot += f2
        ent["full_network"] = {"dims": [dimsA_tot, dimsB_tot],
                               "intersection_primary": inter_tot,
                               "intersection_range_1e-9_to_1e-4": [inter_tot, inter_tot_hi],
                               "intersection_frac_of_p": inter_tot / p_full,
                               "frobenius_sq": f2_tot,
                               "frobenius_sq_over_min_dim": f2_tot / max(min(dimsA_tot, dimsB_tot), 1),
                               "frobenius_sq_over_p": f2_tot / p_full}
        out["A_exact"][fam][f"s{SD}"] = ent
        print(f"A {fam} s{SD}: dims {dimsA_tot}/{dimsB_tot} inter {inter_tot} "
              f"(range to 1e-4: {inter_tot_hi}) F2/min {ent['full_network']['frobenius_sq_over_min_dim']:.4f}")

json.dump(out, open(f"{D}/intersection_dims.json", "w"), indent=1)
print("WROTE", f"{D}/intersection_dims.json")
