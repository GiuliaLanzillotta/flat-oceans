"""Fork-time completion cell gatekeeper (two-day plan 2026-11-22, "MLP parents and fork time completion").

Two modes over one (family, seed, fork_epoch, kind) cell of the fork-time completion sweep
(kind fso = full-space continuous overlap, kind jtail = per-layer Jacobian-tail sweep):

  --pre      PREFLIGHT: authenticate that the inputs the analysis script will consume exist for this cell
             (child run dirs/checkpoints; certified branch records + basis files whose sha256 matches the
             certification record for fso; stored per-block factors with matching epoch/dimensions for jtail;
             the W1 admission record or its buildable inputs for mlp fso).  Exit 0 iff the cell can run.
  (default)  POSTCHECK: the cell's output record exists, is complete (all blocks / all eps-grid entries /
             all four overlap quantities + arrays file), carries the required exceedance fields, handles
             empty tails / empty components coherently (exact zeros labelled, degenerate exceedances None),
             and carries the required scope label.  Exit 0 iff the record passes.

Pure read-only checks; nothing is computed or modified.
usage: check_fork.py <fam in {cnn,mlp}> <seed> <fork_epoch> <kind in {fso,jtail}> [--pre]
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import hashlib
import json
import os
import sys

HERE = str(PROJECT_ROOT)
QS = ("PA_d", "PB_d", "PB_PA_d", "PA_PB_d")
EPS_KEYS = [f"{10.0 ** -k:g}" for k in range(1, 11)]
CELLS = {("cnn", "fso"): [(s, e) for s in (0, 1, 2) for e in (50, 100, 150, 200, 250)],
         ("cnn", "jtail"): [(s, e) for s in (0, 1, 2) for e in (50, 100, 150, 200, 250)],
         ("mlp", "fso"): [(0, 350), (2, 350), (0, 450)],          # (1, 350) is the original s1 record
         ("mlp", "jtail"): [(s, 350) for s in (0, 1, 2)] + [(0, 450)]}

errors = []


def need(cond, msg):
    if not cond:
        errors.append(msg)
    return bool(cond)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    args = [a for a in sys.argv[1:] if a != "--pre"]
    pre = "--pre" in sys.argv[1:]
    if len(args) != 4:
        print(__doc__)
        sys.exit(2)
    fam, sd, ep, kind = args[0], int(args[1]), int(args[2]), args[3]
    assert fam in ("cnn", "mlp") and kind in ("fso", "jtail", "adm"), (fam, kind)
    if kind == "adm":
        import json as _j
        ap = (f"results/branches/w1_admission_s{sd}.json" if ep == 350
              else f"results/branches/w1_admission_s{sd}_e{ep}.json")
        r = _j.load(open(ap))
        ok = bool(r.get("passed")) and all(c.get("n_admitted") is not None for c in r.get("children", {}).values())
        zero = all(c.get("n_admitted") == 0 for c in r.get("children", {}).values())
        adm_counts = {k: v.get("n_admitted") for k, v in r.get("children", {}).items()}
        tagz = "genuine zero" if zero else "nonzero candidates"
        print(f"[check_fork ADM] mlp s{sd} e{ep}: passed={r.get('passed')} "
              f"admitted={adm_counts} ({tagz}) -> " + ("OK" if ok else "FAIL"))
        raise SystemExit(0 if ok else 1)
    need((sd, ep) in CELLS[(fam, kind)], f"({fam}, s{sd}, e{ep}, {kind}) is not a cell of the completion sweep")
    if errors:
        finish(fam, sd, ep, kind, pre)

    if fam == "cnn":
        tags = [f"lmc_cnn_s{sd}_e{ep}_c{c}" for c in (0, 1)]
        blocks = ["conv1", "conv2", "conv3", "conv4", "conv5", "readout"]
        fac_blocks = blocks
    else:
        tags = [f"lmc_s{sd}_e{ep}_c{c}" for c in (0, 1)]
        blocks = ["W1", "W2", "W3", "V"]
        fac_blocks = ["1", "2", "3", "readout"]
    names = [f"mnist_{fam}_{t}_seed{sd}" for t in tags]

    if pre:
        preflight(fam, sd, ep, kind, tags, names, fac_blocks)
    else:
        postcheck(fam, sd, ep, kind, blocks if kind == "fso" else fac_blocks)
    finish(fam, sd, ep, kind, pre)


def finish(fam, sd, ep, kind, pre):
    mode = "PRE" if pre else "POST"
    if errors:
        print(f"[check_fork {mode}] {fam} s{sd} e{ep} {kind}: FAIL")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"[check_fork {mode}] {fam} s{sd} e{ep} {kind}: OK")
    sys.exit(0)


# --------------------------------------------------------------------------------------- preflight
def preflight(fam, sd, ep, kind, tags, names, fac_blocks):
    import numpy as np
    import torch

    # child run directories (both endpoints): config, manifest, preprocessing, epoch-500 checkpoint
    shapes = None
    for tag in tags:
        rd = os.path.join(HERE, "results", "tier1_models", f"mnist_{fam}_{tag}", f"seed_{sd}")
        for f in ("config.json", "data_manifest.json", "preprocessing.pt",
                  os.path.join("checkpoints", "epoch_0500.pt")):
            need(os.path.exists(os.path.join(rd, f)), f"run input missing: {os.path.join(rd, f)}")
        ck = os.path.join(rd, "checkpoints", "epoch_0500.pt")
        if shapes is None and os.path.exists(ck):
            st = torch.load(ck, map_location="cpu", weights_only=False)["model"]
            shapes = {k: tuple(v.shape) for k, v in st.items()}

    if kind == "fso" and fam == "cnn":
        for name, c in zip(names, ("c0", "c1")):
            bjp = os.path.join(HERE, "results", "branches", f"{name}_branch.json")
            btp = os.path.join(HERE, "results", "branches", f"{name}_epoch0500_branch_bottom_up.pt")
            if not need(os.path.exists(bjp), f"branch record missing: {bjp}"):
                continue
            bu = json.load(open(bjp))["results"]["late_endpoint"]["constructions"]["bottom_up"]
            need(bu.get("certified_branch") is True, f"{c}: bottom_up branch not certified in {bjp}")
            if need(os.path.exists(btp), f"branch basis missing: {btp}"):
                need(sha256_file(btp) == bu.get("basis_sha256"),
                     f"{c}: basis sha256 of {btp} differs from the certification record")
    elif kind == "fso":                                            # mlp
        for name, c in zip(names, ("c0", "c1")):
            base = os.path.join(HERE, "results", "branches", f"{name}_epoch0500_reuse_bottom_up")
            if need(os.path.exists(base + ".json"), f"reuse branch record missing: {base}.json"):
                bj = json.load(open(base + ".json"))
                need(bj.get("certified_branch") is True, f"{c}: reuse branch not certified")
                need(bj["authentication"]["W1"].get("status") == "excluded_frozen",
                     f"{c}: W1 not excluded_frozen in the reuse branch")
            need(os.path.exists(base + ".pt"), f"reuse branch basis missing: {base}.pt")
        adm = os.path.join(HERE, "results", "branches",
                           f"w1_admission_s{sd}.json" if ep == 350 else f"w1_admission_s{sd}_e{ep}.json")
        if os.path.exists(adm):
            a = json.load(open(adm))
            need(a.get("passed") is True, f"admission record {adm} exists but did not pass")
            need(a.get("seed") == sd and a.get("spawn") == ep, f"admission record {adm} belongs to another cell")
            print(f"  admission: record present and passed ({os.path.basename(adm)})")
        else:
            for name in names:                                      # buildable: W1 factors for w1_zero_admission.py
                fp = os.path.join(HERE, "results", "persistence", name, "epoch0500_1.npz")
                need(os.path.exists(fp), f"admission record missing AND its W1 factor input missing: {fp}")
            if not errors:
                print(f"  admission: record {os.path.basename(adm)} not yet written; W1 factor inputs present "
                      f"(run `w1_zero_admission.py {sd} {ep}` first)")
    else:                                                           # jtail: stored factors per block, both children
        for name in names:
            for blk in fac_blocks:
                fp = os.path.join(HERE, "results", "persistence", name, f"epoch0500_{blk}.npz")
                if not need(os.path.exists(fp), f"factor missing: {fp}"):
                    continue
                z = np.load(fp, allow_pickle=True)
                for fld in ("singular_values", "p", "s", "kernel_start", "epoch", "identity"):
                    need(fld in z.files, f"{fp}: stored factor lacks field '{fld}'")
                if "epoch" in z.files:
                    need(int(z["epoch"]) == 500, f"{fp}: stored epoch {int(z['epoch'])} != 500")
                if fam == "mlp" and "p" in z.files and shapes is not None:
                    wkey = {"1": "layers.0.weight", "2": "layers.1.weight", "3": "layers.2.weight",
                            "readout": "readout.weight"}[blk]
                    o_, i_ = shapes[wkey]
                    need(int(z["p"]) == o_ * i_ + o_,
                         f"{fp}: p {int(z['p'])} != block size {o_ * i_ + o_} of the checkpoint")


# --------------------------------------------------------------------------------------- postcheck
def postcheck(fam, sd, ep, kind, blocks):
    dec = os.path.join(HERE, "results", "lmc", "decomp")
    if kind == "fso":
        out = os.path.join(dec, (f"cnn_fullspace_overlap_s{sd}_e{ep}.json" if fam == "cnn"
                                 else f"w1branch_overlap_s{sd}_e{ep}.json"))
    else:
        out = os.path.join(dec, f"jtail_sweep_{fam}_s{sd}_e{ep}.json")
    if not need(os.path.exists(out), f"output record missing: {out}"):
        return
    need(os.path.exists(out.replace(".json", "_arrays.pt")), f"arrays file missing: {out.replace('.json', '_arrays.pt')}")
    r = json.load(open(out))
    need(r.get("seed") == sd and r.get("spawn") == ep, "record seed/spawn do not match the requested cell")
    need("runtime_s" in r, "record incomplete: no runtime_s (script did not reach the end)")

    def frac_ok(e, where, allow_zero_flag=False):
        if not need(isinstance(e, dict) and "norm_frac" in e and "sq_mass_frac" in e, f"{where}: norm/sq_mass fields missing"):
            return
        nf = e["norm_frac"]
        need(isinstance(nf, (int, float)) and nf == nf and -1e-12 <= nf <= 1 + 1e-9, f"{where}: norm_frac {nf} out of range")
        need(abs(e["sq_mass_frac"] - nf * nf) <= 1e-9 + 1e-6 * nf * nf, f"{where}: sq_mass_frac != norm_frac^2")

    if kind == "fso" and fam == "cnn":
        need("NO angular filtering" in r.get("scope", "") and "NOT intersection" in r.get("scope", ""),
             "scope label missing the no-filtering / not-intersection qualifications")
        for c in ("c0", "c1"):
            need(r.get("provenance", {}).get("endpoints", {}).get(c, {}).get("certified_branch") is True,
                 f"provenance: endpoint {c} not recorded as certified")
        disp = r.get("displacement", {})
        for q in QS:
            frac_ok(disp.get("full_network", {}).get(q), f"displacement.full_network.{q}")
        for b in blocks:
            for q in QS:
                frac_ok(disp.get("per_block", {}).get(b, {}).get(q), f"displacement.per_block.{b}.{q}")
        for mode in ("controls_isotropic", "controls_layer_length_matched"):
            cm = r.get(mode)
            if need(isinstance(cm, dict), f"{mode} missing"):
                need(cm.get("n") == 256, f"{mode}: n != 256")
                for q in QS:
                    need(q in cm.get("full_network", {}), f"{mode}.full_network.{q} missing")
        pa = r.get("principal_angles", {})
        for b in blocks:
            e = pa.get(b)
            if need(isinstance(e, dict), f"principal_angles.{b} missing"):
                # empty-component handling: a block with an empty endpoint basis must say so instead of angles
                if min(e.get("dim_A", 0), e.get("dim_B", 0)) == 0:
                    need("note" in e and "mean_sq_overlap" not in e,
                         f"principal_angles.{b}: empty endpoint basis without the explicit no-angles note")
                else:
                    need("mean_sq_overlap" in e and "smallest_angle_deg" in e, f"principal_angles.{b} incomplete")
    elif kind == "fso":                                             # mlp
        need("NOT a new shared subspace" in r.get("scope", ""), "scope label missing the diagnostics qualification")
        need("w1_component_note" in r, "mlp record lacks the W1 zero-component note")
        disp = r.get("displacement", {})
        for q in QS:
            frac_ok(disp.get("full_branch", {}).get(q), f"displacement.full_branch.{q}")
        for b in blocks:
            for q in QS:
                frac_ok(disp.get("per_block", {}).get(b, {}).get(q), f"displacement.per_block.{b}.{q}")
        w1 = disp.get("per_block", {}).get("W1", {})
        if w1.get("exact_zero") is True:                            # empty-component coherence (genuine zeros)
            for q in QS:
                need(w1.get(q, {}).get("norm_frac") == 0.0, f"W1.{q}: exact_zero flagged but norm_frac != 0")
            pw1 = r.get("principal_angles", {}).get("W1", {})
            need("note" in pw1 and pw1.get("dim_A") == 0 and pw1.get("dim_B") == 0,
                 "W1 principal angles: zero-dimensional component without the explicit note")
        for mode in ("controls_isotropic", "controls_layer_length_matched"):
            cm = r.get(mode, {})
            exc = cm.get("full_branch_exceedances_of_256")
            if need(isinstance(exc, dict), f"{mode}: exceedance counts missing"):
                for q in QS:
                    need(isinstance(exc.get(q), int) and 0 <= exc[q] <= 256, f"{mode} exceedance {q} invalid")
    else:                                                           # jtail
        need("FIRST-ORDER SENSITIVITY DIAGNOSTIC" in r.get("scope", "")
             and "does not establish finite-radius flatness" in r.get("scope", ""),
             "scope label missing the first-order / no-finite-radius qualifications")
        need("exceedance counts among 256" in r.get("statistics_convention", ""), "statistics convention label missing")
        need([f"{float(k):g}" for k in r.get("eps_grid", [])] == EPS_KEYS, "eps grid differs from the declared 1e-1..1e-10")
        for b in blocks:
            v = r.get("validation", {}).get(b)
            if need(isinstance(v, dict) and v, f"validation.{b} missing"):
                for k, e in v.items():
                    need(e.get("abs_diff_over_sigma0rate", 1) <= 3e-4, f"validation.{b}.{k} exceeds the fp32 floor")
            blk = r.get("blocks", {}).get(b)
            if not need(isinstance(blk, dict), f"blocks.{b} missing"):
                continue
            for key in EPS_KEYS:
                ent = blk.get("by_eps", {}).get(key)
                if not need(isinstance(ent, dict), f"blocks.{b}.by_eps[{key}] missing"):
                    continue
                td = ent.get("tail_dims")
                if not need(isinstance(td, list) and len(td) == 2 and all(isinstance(t, int) for t in td),
                            f"{b}@{key}: tail_dims missing/invalid"):
                    continue
                ktA, ktB = td
                disp = ent.get("displacement", {})
                need(all(q in disp for q in QS), f"{b}@{key}: displacement quantities incomplete")
                zero_qs = {q for q in QS if disp.get(q, {}).get("exact_zero")}
                if min(ktA, ktB) == 0:                              # empty-tail handling coherence
                    need("note" in ent and "zero-dimensional" in ent.get("note", ""),
                         f"{b}@{key}: empty tail without the explicit zero-dimensional note")
                    need("no principal angles" in str(ent.get("principal_angles", {}).get("note", "")),
                         f"{b}@{key}: empty tail but principal-angle note missing")
                    exp_zero = set(QS) if (ktA == 0 and ktB == 0) else \
                        ({"PA_d", "PB_PA_d", "PA_PB_d"} if ktA == 0 else {"PB_d", "PB_PA_d", "PA_PB_d"})
                    need(zero_qs == exp_zero, f"{b}@{key}: exact-zero pattern {sorted(zero_qs)} "
                                              f"incoherent with tail_dims {ent.get('tail_dims')}")
                else:
                    need(not zero_qs, f"{b}@{key}: exact_zero flagged although both tails are nonempty")
                for q in QS:
                    e = disp.get(q, {})
                    frac_ok(e, f"{b}@{key}.{q}")
                    exc = e.get("ctrl_exceedances_of_256", "MISSING")
                    if e.get("exact_zero"):
                        need(e.get("norm_frac") == 0.0, f"{b}@{key}.{q}: exact_zero but norm_frac != 0")
                        need(exc is None and "ctrl_note" in e,
                             f"{b}@{key}.{q}: degenerate exceedance must be None with a ctrl_note")
                    else:
                        need(isinstance(exc, int) and 0 <= exc <= 256, f"{b}@{key}.{q}: ctrl_exceedances_of_256 invalid")
        for key in EPS_KEYS:
            fn = r.get("full_network", {}).get(key)
            if not need(isinstance(fn, dict), f"full_network[{key}] missing"):
                continue
            for q in QS:
                e = fn.get(q)
                if need(isinstance(e, dict), f"full_network[{key}].{q} missing"):
                    frac_ok(e, f"full_network[{key}].{q}")
                    for cm in ("ctrl_isotropic", "ctrl_layer_length_matched"):
                        exc = e.get(cm, {}).get("exceedances_of_256", "MISSING")
                        need(isinstance(exc, int) and 0 <= exc <= 256,
                             f"full_network[{key}].{q}.{cm}: exceedances_of_256 invalid")


if __name__ == "__main__":
    main()
