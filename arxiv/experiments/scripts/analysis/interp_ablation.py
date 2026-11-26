"""Clean-to-pure-noise INTERPOLATION ablation (PI instruction 2026-11-23).

REPLACES the additive-noise sweep in the planned paper figure; the additive-noise + SVHN records under
results/newinput/ are preserved untouched (SVHN remains a separate comparison).

Spec (PI 2026-11-23): width-128 MNIST ReLU MLPs, seeds 0/1/2, epoch 500, NO training. X = the fixed
authenticated 10k analysis set. Endpoints: the 10k clean MNIST test images y_i (test images /255) and ONE
independent uniform-noise image z_i ~ U[0,1]^784 PER TEST IMAGE (torch.rand, single fixed Generator seed
20261123, fp64), REUSED across all interpolation steps AND all model seeds. Conditions:
x_i(alpha) = (1-alpha) * y_i + alpha * z_i for alpha = 0.0, 0.1, ..., 1.0 (11 steps), interpolated in RAW
[0,1] PIXEL coordinates, THEN the saved MNIST preprocessing applied unchanged (tier1_train.apply_preprocessing
with the run's own preprocessing.pt). Each condition uses the SAME ORIGINAL X with Y_alpha only (no
accumulation). Per block l in {2, 3, readout}: d_exposed_l(alpha) = dim K_l(X) - dim K_l(X u Y_alpha) and its
fraction of dim K_l(X); the baseline on X is the authenticated stored factor (results/atomic), as in the
new-batch protocol; rank rule relative 1e-9, union rank resolved iff gap >= 10; rank-sensitive cells keep
their cutoff-sensitivity RANGES (the 1e-6/1e-9/1e-12 ladder) and the band count, thresholds NEVER changed
(precedent: results/newinput/seed0_gate.json strict failure preserved, qualified cells accepted).

Machinery copied with maximum fidelity from scripts/analysis/newinput_ablation.py (PI spec 2026-11-16);
verbatim copies cite that template's line numbers.

Documented decisions (also in the manifest):
  * alpha=0 is RECOMPUTED (the gate then checks exact reproduction of the clean new-batch reference);
    the alpha=0 raw input is bitwise identical to apply_preprocessing(Xte): (1-0)*y01 + 0*z == y01 and the
    template's (raw01*255)/255 round-trip is bitwise exact (verified 2026-11-23).
  * Any `--alphas` invocation writes mnist_mlp_seed<sd>_interp_partial_<tag>.json, never the full-record
    filename (one job per output file); the full run recomputes every condition into its own file.
  * gate verdict: hard failures = alpha=0 non-reproduction, dim_K_union > dim_K_X, negative exposure,
    missing identity. Unresolved gaps (< 10) are recorded as QUALIFIED flags with their ladder ranges, not
    hard failures: at alpha=0 the gap equals the clean reference's own gap (seed 0 block 3: 6.3 < 10), which
    the PI already accepted with qualification (memos/2026-11-16_newinput_gate_acceptance.md); failing the
    gate on exact reproduction of an accepted record would be contradictory. The strict outcome
    (strict_pass, every flag listed) is preserved in the gate file.

usage: interp_ablation.py prepare | run <seed> [--alphas a,b,...] | gate <seed> | consolidate"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, glob, hashlib
import numpy as np, torch
# -- environment setup copied verbatim from scripts/analysis/newinput_ablation.py lines 19-26 --
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
os.chdir(HERE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
from flatocean import tier1_train as T
from flatocean import ocean_lib as O
from flatocean.analyses import persistence as KP
from flatocean import provenance as P
NI = os.path.join(HERE, "results", "newinput_interp"); os.makedirs(NI, exist_ok=True)
OUTA = os.path.join(HERE, "results", "atomic")
TOL = "1e-09"; LADDER = ("1e-06", "1e-09", "1e-12"); GAP_RULE = 10.0; CUT = 1e-9
ALPHAS = [i / 10 for i in range(11)]                      # 0.0, 0.1, ..., 1.0 (11 steps, PI 2026-11-23)
BLOCKS = ("2", "3", "readout")
MAN = os.path.join(NI, "manifest.json")
NOISE_SEED = 20261123


def akey(a):
    return f"{a:.1f}"


# -- copied verbatim from scripts/analysis/newinput_ablation.py lines 34-35 --
def sha_t(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy() if torch.is_tensor(t) else t).tobytes()).hexdigest()


def clean_y01():
    """The 10k clean MNIST test images in raw [0,1], exactly as the template builds y01
    (newinput_ablation.py lines 39-41)."""
    Xtr, ytr, Xte, yte, hashes = T.load_raw("mnist")
    assert Xte.shape == (10000, 784) and Xte.dtype == np.uint8
    return torch.from_numpy(Xte.astype(np.float64) / 255.0), hashes


def prepare():
    y01, hashes = clean_y01()
    g = torch.Generator().manual_seed(NOISE_SEED)
    z = torch.rand(10000, 784, generator=g, dtype=torch.float64)     # U[0,1]^784, one image per test image
    torch.save(z, os.path.join(NI, "noise_u01.pt"))
    man = {"spec": "PI 2026-11-23 interpolation ablation",
           "protocol": "interp-exposure-v1: x_i(alpha) = (1-alpha) * y_i + alpha * z_i in RAW [0,1] pixel "
                       "coordinates, THEN the saved MNIST centering + PCA-256 + scaling (apply_preprocessing "
                       "with the run's own preprocessing.pt); same original X every condition, Y_alpha only; "
                       "REPLACES the additive-noise sweep in the planned paper figure (results/newinput/ "
                       "preserved untouched; SVHN remains a separate comparison)",
           "alphas": [akey(a) for a in ALPHAS],
           "noise": "z_i ~ U[0,1]^784 per test image, torch.rand, fp64, single fixed Generator seed "
                    f"{NOISE_SEED}; REUSED across all interpolation steps AND all model seeds",
           "noise_seed": NOISE_SEED, "z_sha256": sha_t(z),
           "mnist_raw_sha": hashes, "clean_y01_sha256": sha_t(y01),
           "clean_reference": "results/atomic/mnist_mlp_seed<sd>_new_batch.json late_endpoint/batches/test "
                              "(authenticated; alpha=0 must reproduce it exactly, dims per block)",
           "rank_rule": f"relative {TOL}; union rank resolved iff gap >= {GAP_RULE:g}; unresolved cells keep "
                        "their ladder ranges (1e-6/1e-9/1e-12) and band counts, thresholds NEVER changed",
           "decisions": {
               "partial_runs": "any --alphas invocation writes mnist_mlp_seed<sd>_interp_partial_<tag>.json, "
                               "never the full-record filename (one job per output file); the full run "
                               "recomputes every condition into its own file",
               "alpha0_recomputed": "alpha=0 recomputed, not copied; its raw input is bitwise identical to "
                                    "the clean test batch of the new-batch records ((1-0)*y01 + 0*z == y01; "
                                    "(raw01*255)/255 round-trip bitwise exact, verified 2026-11-23)",
               "gate_qualification": "unresolved gaps are QUALIFIED flags with ladder ranges, not hard gate "
                                     "failures: at alpha=0 the gap equals the accepted clean reference's own "
                                     "gap (PI acceptance memos/2026-11-16_newinput_gate_acceptance.md); the "
                                     "strict outcome is preserved in seed<sd>_gate.json (strict_pass)"}}
    json.dump(man, open(MAN, "w"), indent=1)
    print("prepared:", {k: man[k] for k in ("z_sha256", "clean_y01_sha256", "noise_seed")}, flush=True)


def load_endpoints(man):
    """y01, z with SHA verification against the manifest (provenance: the saved z is reused everywhere)."""
    y01, hashes = clean_y01()
    assert sha_t(y01) == man["clean_y01_sha256"], "clean test endpoint SHA differs from the manifest"
    assert hashes == man["mnist_raw_sha"], "raw MNIST files differ from the manifest"
    z = torch.load(os.path.join(NI, "noise_u01.pt"), weights_only=False)
    assert sha_t(z) == man["z_sha256"], "noise endpoint SHA differs from the manifest"
    return y01, z


def parse_alphas(argv):
    """`--alphas a,b,...` -> (sorted canonical subset, tag) or (None, None) for the full grid."""
    if "--alphas" not in argv:
        return None, None
    toks = argv[argv.index("--alphas") + 1].split(",")
    sub = []
    for t in toks:
        a = float(t)
        match = [b for b in ALPHAS if abs(b - a) < 1e-12]
        assert match, f"alpha {t} is not on the fixed grid 0.0, 0.1, ..., 1.0"
        sub.append(match[0])                               # canonical grid value: bitwise identical y_alpha everywhere
    sub = sorted(set(sub))
    tag = "-".join(akey(a) for a in sub).replace(".", "p")
    return sub, tag


def run_seed(seed, alphas=None, tag=None):
    man = json.load(open(MAN))
    y01, z = load_endpoints(man)
    partial = alphas is not None
    conds = alphas if partial else ALPHAS
    # -- run/baseline loading copied verbatim from scripts/analysis/newinput_ablation.py lines 81-91 --
    name = f"mnist_mlp_seed{seed}"; rd = O.run_dir("mnist", "mlp", seed, "")
    run = O.load_run("mnist", "mlp", seed, tag="", epoch=500, device="cpu", sets=("analysis",))
    pre = torch.load(os.path.join(rd, "preprocessing.pt"), weights_only=False)
    assert pre.get("type", "pca") == "pca", "expected the saved MNIST PCA preprocessing"
    Xa = run["data"]["analysis"][0]
    base_spec = {}
    for blk in ("2", "3", "readout"):
        out_ = KP.factor_spectrum(name, 500, blk, run, rd)
        if out_ is None:
            raise P.ProvenanceError(f"{name} block {blk}: no stored factor baseline")
        base_spec[blk] = out_
    out_path = os.path.join(NI, f"{name}_interp_partial_{tag}.json" if partial else f"{name}_interp.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {
        "protocol": "interp-exposure-v1 (PI spec 2026-11-23; clean-to-pure-noise interpolation; "
                    "new-batch-exposure layer coverage, rank rule 1e-9)",
        "rank_tolerance": TOL, "epoch": 500, "manifest": man,
        "partial": ({"tag": tag, "alphas": [akey(a) for a in conds]} if partial else False),
        "conditions": {}}
    for a in conds:
        ck = akey(a)
        if ck in res["conditions"]:
            print(f"== {name} alpha={ck}: already recorded, skipping", flush=True); continue
        t0 = time.time()
        raw01 = (1.0 - a) * y01 + a * z                    # interpolation in RAW [0,1] pixel coordinates
        ysha = sha_t(raw01)
        # -- preprocessing + union + atomic analysis copied verbatim from newinput_ablation.py lines 101-110 --
        Y = T.apply_preprocessing((raw01 * 255.0).numpy(), pre, torch.float64)   # /255 inside reproduces raw01 to ~1e-16
        X = torch.cat([Xa, Y], 0); s = X.shape[0]
        st = O.mlp_structure(run["net"], X)
        recs = {}; spectra = {}; blk_secs = {}
        for l in (1, 2):
            tb = time.time()
            rec, sv, _, _ = O.atomic_layer(st, l, s, "cpu")
            recs[str(l + 1)] = rec; spectra[str(l + 1)] = np.asarray(sv); blk_secs[str(l + 1)] = round(time.time() - tb, 1)
        tb = time.time()
        rec, sv, _ = O.atomic_readout(st, s, "cpu")
        recs["readout"] = rec; spectra["readout"] = np.asarray(sv); blk_secs["readout"] = round(time.time() - tb, 1)
        spectra_file = f"{name}_a{ck}_spectra.npz"
        np.savez(os.path.join(NI, spectra_file), **spectra)
        summary = {}
        for blk, r in recs.items():
            # -- baseline + exposure arithmetic copied verbatim from newinput_ablation.py lines 113-116
            #    (exposed = rank_J_union - rank_old == dim K_l(X) - dim K_l(X u Y_alpha); both raw dims recorded) --
            sv0, p_, s0_, sha, status = base_spec[blk]
            lad = O.rank_ladder(sv0); rank_old = lad[TOL]; dimK_old = p_ - rank_old
            cn = r[f"components@{TOL}"]
            exposed = r["rank_J"][TOL] - rank_old
            assert r["p"] == p_, f"{blk}: union p {r['p']} != baseline factor p {p_}"
            ent = {"dim_K_X": dimK_old, "dim_K_X_ladder": {t: p_ - lad[t] for t in lad},
                   "rank_X": rank_old, "rank_union": r["rank_J"][TOL], "p": p_,
                   "dim_K_union": cn["dim_K"],
                   "dim_K_union_ladder": {t: r[f"components@{t}"]["dim_K"] for t in LADDER},
                   "exposed": exposed, "exposed_fraction": (exposed / dimK_old if dimK_old else None),
                   "components_union": cn, "gap_union": r["gap_J"][TOL],
                   "gap_note": "rank resolved iff gap >= 10",
                   "baseline_factor_sha256": sha, "baseline_status": status,
                   "spectra_file": f"results/newinput_interp/{spectra_file}", "seconds": blk_secs[blk]}
            if ent["gap_union"] is not None and ent["gap_union"] < GAP_RULE:
                # cutoff-sensitivity qualification: ladder RANGE + band count (rule of newinput_consolidate.py
                # lines 61-69); the threshold is NOT changed to force resolution
                svu = np.sort(np.asarray(spectra[blk], float))[::-1]
                rel = svu / max(svu[0], 1e-300)
                band = int(((rel >= CUT / 10) & (rel <= CUT * 10)).sum())
                ldims = list(ent["dim_K_union_ladder"].values())
                ent["cutoff_sensitivity"] = {
                    "dim_K_union_range": [min(ldims), max(ldims)],
                    "band_sv_within_decade_of_cut": band,
                    "exposed_fraction_uncertainty": band / max(dimK_old, 1),
                    "note": "union rank gap < 10: dimension determined only to within the ladder range/band; "
                            "QUALIFIED, threshold NOT changed (precedent results/newinput/seed0_gate.json)"}
            summary[blk] = ent
            print(f"    alpha={ck} blk {blk}: dimK_X {dimK_old} -> union {cn['dim_K']} exposed {exposed} "
                  f"({(exposed/dimK_old if dimK_old else 0):.3f}) gap {r['gap_J'][TOL]:.1f}", flush=True)
        # -- identity structure copied from newinput_ablation.py lines 123-127, condition fields adapted --
        ident = P.analysis_identity(run, rd, 500, {"protocol": "interp-exposure-v1", "alpha": ck,
                                                   "input_sha256": ysha,
                                                   "noise_u01_sha256": man["z_sha256"],
                                                   "clean_y01_sha256": man["clean_y01_sha256"],
                                                   "baseline_factor_sha256": {b: base_spec[b][3] for b in base_spec}})
        res["conditions"][ck] = {"alpha": a, "identity": ident, "summary": summary,
                                 "input_sha256": {"y_alpha": ysha, "noise_u01": man["z_sha256"],
                                                  "clean_y01": man["clean_y01_sha256"]},
                                 "seconds": round(time.time() - t0, 1)}
        json.dump(res, open(out_path + ".tmp", "w"), indent=1, default=str); os.replace(out_path + ".tmp", out_path)
        print(f"== {name} alpha={ck} done in {res['conditions'][ck]['seconds']}s; record updated", flush=True)
    print("WROTE", out_path, flush=True)


def collect_conditions(seed):
    """Conditions of one seed: the full record's, overlaid on any partial records' (gate works against
    partials before the full run exists; the full record takes precedence)."""
    name = f"mnist_mlp_seed{seed}"; conds = {}; sources = []
    for pth in sorted(glob.glob(os.path.join(NI, f"{name}_interp_partial_*.json"))):
        d = json.load(open(pth)); conds.update(d["conditions"]); sources.append(os.path.relpath(pth, HERE))
    full = os.path.join(NI, f"{name}_interp.json")
    if os.path.exists(full):
        d = json.load(open(full)); conds.update(d["conditions"]); sources.append(os.path.relpath(full, HERE))
    return conds, sources


def gate(seed):
    """Validation gate: alpha=0 must REPRODUCE the clean new-batch reference exactly (dims per block);
    standard checks dim_K_union <= dim_K_X, exposure >= 0, identity present. Unresolved gaps (< 10) are
    QUALIFIED flags with their ladder ranges, recorded not fixed (see the manifest's gate_qualification
    decision); strict_pass preserves the strict outcome."""
    fails = []; flags = []
    conds, sources = collect_conditions(seed)
    if not conds:
        print(f"GATE FAIL: no seed-{seed} interp record found"); sys.exit(1)
    nb = json.load(open(os.path.join(OUTA, f"mnist_mlp_seed{seed}_new_batch.json")))
    tb = nb["stages"]["late_endpoint"]["batches"]["test"]["summary"]
    repro = {}
    c0 = conds.get("0.0")
    if c0 is None:
        fails.append("alpha=0.0: missing (reproduction of the clean reference not checkable)")
    else:
        for blk in BLOCKS:
            e = c0["summary"][blk]; ref = tb[blk]
            cell = {"dim_K_X": [e["dim_K_X"], ref["dim_K_old"]],
                    "dim_K_union": [e["dim_K_union"], ref["dim_K_union"]],
                    "exposed": [e["exposed"], ref["exposed_by_new_batch"]],
                    "gap_union_equal_float": bool(e["gap_union"] == ref["gap_union"])}
            cell["exact"] = all(v[0] == v[1] for k, v in cell.items() if isinstance(v, list))
            repro[blk] = cell
            if not cell["exact"]:
                fails.append(f"alpha=0.0/{blk}: does NOT reproduce the clean new-batch reference: {cell}")
    for ck, c in sorted(conds.items(), key=lambda kv: float(kv[0])):
        if not c.get("identity", {}).get("digest"):
            fails.append(f"alpha={ck}: identity digest missing")
        for blk, e in c["summary"].items():
            if e["dim_K_union"] > e["dim_K_X"]:
                fails.append(f"alpha={ck}/{blk}: dim_K_union > dim_K_X")
            if e["exposed"] < 0:
                fails.append(f"alpha={ck}/{blk}: negative exposure")
            if e["gap_union"] is not None and e["gap_union"] < GAP_RULE:
                q = e.get("cutoff_sensitivity", {})
                flags.append(f"alpha={ck}/{blk}: union rank gap {e['gap_union']:.1f} < 10 (UNRESOLVED; "
                             f"ladder range {q.get('dim_K_union_range')}, band {q.get('band_sv_within_decade_of_cut')})"
                             if q else f"alpha={ck}/{blk}: union rank gap {e['gap_union']:.1f} < 10 (UNRESOLVED; no qualification recorded)")
                if not q:
                    fails.append(f"alpha={ck}/{blk}: unresolved gap without a recorded cutoff_sensitivity qualification")
    rep = {"passed": not fails, "strict_pass": not fails and not flags,
           "failures": fails, "qualified_flags": flags,
           "alpha0_reproduction": repro, "sources": sources,
           "label": "computed + automatically checked",
           "policy": "hard failures: alpha=0 non-reproduction, union > X, negative exposure, missing identity "
                     "or missing qualification; gap < 10 cells are QUALIFIED flags with ladder ranges "
                     "(recorded, thresholds unchanged; precedent results/newinput/seed0_gate.json)"}
    json.dump(rep, open(os.path.join(NI, f"seed{seed}_gate.json"), "w"), indent=1)
    print(json.dumps(repro, indent=1))
    print(("GATE PASS" + (" (with %d qualified flags)" % len(flags) if flags else " (strict)"))
          if not fails else "GATE FAIL:\n  " + "\n  ".join(fails), flush=True)
    if flags:
        print("QUALIFIED FLAGS:\n  " + "\n  ".join(flags), flush=True)
    sys.exit(0 if not fails else 1)


def consolidate():
    """After all three seeds: consolidated_interp.json (raw dims + per-seed exposed percentages per alpha per
    block + mean and SAMPLE standard deviation ddof=1) and the compact panel
    memos/figures/interp_exposed_kernel.png/.pdf + memos/2026-11-23_interp_ablation_table.md.
    Post-processing of completed records only; no recomputation. Missing conditions are reported as missing
    (distinguished from genuine zeros)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    man = json.load(open(MAN))
    seeds = (0, 1, 2)
    recs = {}
    for sd in seeds:
        pth = os.path.join(NI, f"mnist_mlp_seed{sd}_interp.json")
        assert os.path.exists(pth), f"full record missing for seed {sd}: {pth} (consolidate needs all three seeds)"
        recs[sd] = json.load(open(pth))
    out = {"protocol": "interp-exposure-v1 consolidation (PI 2026-11-23); post-processing only, no recomputation",
           "spec": man["spec"], "rank_rule": man["rank_rule"],
           "exposed_pct_definition": "100 * exposed / dim_K_X; mean and SAMPLE standard deviation (ddof=1) "
                                     "across seeds 0/1/2",
           "alphas": [akey(a) for a in ALPHAS], "blocks": list(BLOCKS), "cells": {}}
    missing = []
    for blk in BLOCKS:
        out["cells"][blk] = {}
        for a in ALPHAS:
            ck = akey(a); cell = {"per_seed": {}, "qualified_seeds": []}
            pcts = []
            for sd in seeds:
                c = recs[sd]["conditions"].get(ck)
                if c is None:
                    cell["per_seed"][f"seed{sd}"] = "MISSING"; missing.append(f"seed{sd}/alpha={ck}"); continue
                e = c["summary"][blk]
                ent = {k: e[k] for k in ("dim_K_X", "dim_K_union", "exposed", "exposed_fraction",
                                         "gap_union", "dim_K_union_ladder")}
                ent["exposed_pct"] = 100.0 * e["exposed"] / e["dim_K_X"] if e["dim_K_X"] else None
                ent["resolved"] = bool(e["gap_union"] is not None and e["gap_union"] >= GAP_RULE)
                if not ent["resolved"]:
                    ent["cutoff_sensitivity"] = e.get("cutoff_sensitivity")
                    cell["qualified_seeds"].append(sd)
                cell["per_seed"][f"seed{sd}"] = ent; pcts.append(ent["exposed_pct"])
            if len(pcts) == len(seeds):
                arr = np.asarray(pcts, float)
                cell["exposed_pct_mean"] = float(arr.mean())
                cell["exposed_pct_sd"] = float(arr.std(ddof=1))
            out["cells"][blk][ck] = cell
    out["missing_cells"] = missing                           # missing, as opposed to genuine zeros
    json.dump(out, open(os.path.join(NI, "consolidated_interp.json"), "w"), indent=1)

    # ---- ONE COMPACT PANEL: house style of scripts/analysis/hgn_fixed_ocean_figure.py (Okabe-Ito categorical
    #      order, white-edged markers, recessive grid) + the open-square qualified-cell convention of
    #      newinput_consolidate.py lines 104-106 ----
    os.makedirs(os.path.join(HERE, "build", "validation", "figures"), exist_ok=True)
    COLORS = {"2": "#0072B2", "3": "#D55E00", "readout": "#009E73"}   # CVD-safe, fixed order (house palette)
    LBL = {"2": "layer 2", "3": "layer 3", "readout": "readout"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for blk in BLOCKS:
        xs, mu, sdv, qual = [], [], [], []
        for a in ALPHAS:
            cell = out["cells"][blk][akey(a)]
            if "exposed_pct_mean" not in cell:
                continue
            xs.append(a); mu.append(cell["exposed_pct_mean"]); sdv.append(cell["exposed_pct_sd"])
            qual.append(bool(cell["qualified_seeds"]))
        xs, mu, sdv = np.asarray(xs), np.asarray(mu), np.asarray(sdv)
        ax.fill_between(xs, mu - sdv, mu + sdv, color=COLORS[blk], alpha=0.18, lw=0, zorder=2)
        ax.plot(xs, mu, color=COLORS[blk], lw=2, marker="o", ms=5, markerfacecolor=COLORS[blk],
                markeredgecolor="white", markeredgewidth=0.8, label=LBL[blk], zorder=3)
        for x, y, q in zip(xs, mu, qual):
            if q:
                ax.plot([x], [y], marker="s", ms=10, mfc="none", mec=COLORS[blk], mew=1.2, zorder=4)
        ax.annotate(LBL[blk], xy=(xs[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=COLORS[blk], va="center")
    ax.set_xlabel(r"interpolation strength $\alpha$  (clean $\rightarrow$ U[0,1] noise)")
    ax.set_ylabel(r"exposed kernel (% of dim $K_X$)")
    ax.set_xlim(-0.03, 1.16); ax.set_xticks(ALPHAS)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("Clean-to-noise interpolation: exposed kernel fraction\n"
                 "(mean $\\pm$ sample SD over seeds 0/1/2; open squares = cutoff-qualified cells)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "build", "validation", "figures", "interp_exposed_kernel.png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "build", "validation", "figures", "interp_exposed_kernel.pdf"), bbox_inches="tight")

    # ---- markdown table: ALL raw dims and per-seed values; qualified cells marked with their ranges ----
    lines = ["# Interpolation ablation (PI 2026-11-23): raw dims and per-seed exposure",
             "",
             "Clean-to-pure-noise interpolation x_i(alpha) = (1-alpha) y_i + alpha z_i (z ~ U[0,1]^784, seed "
             f"{NOISE_SEED}, reused across steps and model seeds); width-128 MNIST ReLU MLPs, epoch 500; "
             "X = the fixed authenticated 10k analysis set; rank rule relative 1e-9, union rank resolved iff "
             "gap >= 10. Cells with gap < 10 are QUALIFIED: marked `*` with the 1e-6/1e-9/1e-12 ladder range "
             "`[lo, hi]` of dim K_union (thresholds unchanged). exposed = dim K_X - dim K_union; "
             "pct = 100 exposed / dim K_X. Mean/SD = mean and sample standard deviation (ddof=1) across "
             "seeds. All values computed + automatically checked (scripts/pipeline/check_interp.py). "
             "Source records: results/newinput_interp/mnist_mlp_seed{0,1,2}_interp.json.", ""]
    for blk in BLOCKS:
        lines += [f"## Block {LBL[blk]}", "",
                  "| alpha | " + " | ".join(f"s{sd} dim K_X | s{sd} dim K_union | s{sd} exposed | s{sd} pct" for sd in seeds)
                  + " | mean pct | sd pct |",
                  "|" + "---|" * (4 * len(seeds) + 3)]
        for a in ALPHAS:
            ck = akey(a); cell = out["cells"][blk][ck]; row = [ck]
            for sd in seeds:
                e = cell["per_seed"][f"seed{sd}"]
                if e == "MISSING":
                    row += ["MISSING"] * 4; continue
                q = ""
                if not e["resolved"]:
                    cs = e.get("cutoff_sensitivity") or {}
                    q = f"\\* {cs.get('dim_K_union_range', list(e['dim_K_union_ladder'].values()))}"
                row += [str(e["dim_K_X"]), f"{e['dim_K_union']}{q}", str(e["exposed"]), f"{e['exposed_pct']:.3f}"]
            row += ([f"{cell['exposed_pct_mean']:.3f}", f"{cell['exposed_pct_sd']:.3f}"]
                    if "exposed_pct_mean" in cell else ["--", "--"])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    open(os.path.join(HERE, "build", "validation", "2026-11-23_interp_ablation_table.md"), "w").write("\n".join(lines) + "\n")
    print("WROTE results/newinput_interp/consolidated_interp.json + memos/figures/interp_exposed_kernel.{png,pdf} "
          "+ memos/2026-11-23_interp_ablation_table.md", flush=True)
    if missing:
        print("MISSING cells (not zeros):", missing, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        prepare()
    elif mode == "run":
        sub, tag = parse_alphas(sys.argv)
        run_seed(int(sys.argv[2]), sub, tag)
    elif mode == "gate":
        gate(int(sys.argv[2]))
    elif mode == "consolidate":
        consolidate()
    else:
        raise SystemExit(f"unknown mode {mode}")
