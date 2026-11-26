#!/usr/bin/env python3
"""RADIUS-GRID EXTENSION of the common ViT family batteries (2026-11-22 radius extension).

PI instruction (2026-11-22, verbatim): "append r=100, 300, 1000 to the common-family
batteries, reusing the same 256 directions, normalization, precision, controls and
paired-response diagnostics. Preserve existing results. Report crossings or continued
censoring, and distinguish numerical breakdown from genuine output change. Keep exact
structural invariance distinct from sampled finite-radius evidence."

usage: vit_radius_extension.py --seed {0,1,2} [--M 256] [--smoke --outdir DIR]

What this script does (NEW files only; it never touches an existing record):
  1. Rebuilds the base quantities (checkpoint, 10k cls_mid cache, forward activations
     u/a/z0/f/logits, gelu', LN statistics) with the setup code of
     scripts/analysis/vit_common_family_study.py copied verbatim (study lines 186-209,
     565-579; cited inline below). Checkpoint + cache SHA256 are authenticated against
     the stored common_family_s<seed>.json record (fail closed on mismatch).
  2. Loads the persisted orthonormal battery bases `basis_<tag>` from
     common_family_s<seed>_arrays.pt (family structure is NOT recomputed); checks
     orthonormality < 1e-10 and k_factor against the JSON record.
  3. Regenerates the SAME probe directions: stable_seed(SEED_BASE, "common_s<seed>_<tag>",
     mode) generator streams and make_raw (study lines 641-649) in the same per-probe
     order, with the study's projection convention reproduced exactly: tags ending in
     "_exact" are unprojected; approximate tags project the layer exact-null component
     out of family AND controls identically via the persisted nu_u / nu_f vectors and
     the analytic ones_768 mean-removal (study lines 350-352), cross-checked against
     the record's `exact_component_removal` field.
  4. VALIDATION GATE: every regenerated probe is re-evaluated on the ORIGINAL 10-radius
     grid with the study's evaluators (study lines 572-625 copied verbatim, the radius
     grid parametrized) and the drift (plus mlp.0 partial-freeze supplements, probe
     norms and jv Frobenius rates) must match the stored battery_<tag>_<mode>_drift
     rows [0:M] entrywise to <= 1e-12 absolute OR <= 1e-9 relative. Fail closed
     otherwise. This proves "same directions, normalization, precision".
  5. Evaluates the EXTENDED radii RGRID_EXT = [100, 300, 1000] with the identical
     evaluators: 5 paired curves x 3 metrics per probe (same axes), mlp.0 partial-freeze
     supplements, per-probe jv rates (unchanged by construction; gate-compared),
     exceedance counts at relative tol 1e-3 per new radius, and per-probe brackets over
     the CONCATENATED 13-radius grid (stored original rel drift ++ new rel drift),
     labelled as the extension bracket; the original record is left untouched.
  6. NUMERICAL-FLOOR DIAGNOSTIC: at each radius it also evaluates K=8 reference probes
     per layer drawn from that layer's EXACTLY structurally invariant family basis
     (mlp0_fn_exact, mlp3_dn_exact, head_fn_exact; measured at machine floor through
     r=30 in the study). Their measured actual-route relative drift at r=100/300/1000
     IS the fp64 rounding floor of the evaluation at that radius (cancellation in
     GELU/LN at large perturbation scale). floor_median / floor_max are recorded per
     layer/radius, and every battery probe at each new radius is classified:
     rel drift <= 10 x floor_max  ->  "at_numerical_floor"  (NOT evidence of genuine
     output change); above -> "genuine_output_change". Raw numbers kept regardless.

Outputs:
  results/runs/ftvit_resisc45/common_family_s<seed>_radius_ext.json
  results/runs/ftvit_resisc45/common_family_s<seed>_radius_ext_arrays.pt
Both are refused if they already exist. --smoke requires an explicit --outdir OUTSIDE
the canonical run directory, records "smoke": true, and compares against stored rows
[0:M] (the first M probes of each stream are the stream's first M draws).

Conventions are inherited unchanged from the study: fp64 everywhere, unit-Frobenius
[W|b] edits, relative visibility tolerance 1e-3 (comparison only, not retuned),
exceedance counts among M (no p-values), empty/censored results are valid results,
everything labelled computed + automatically checked (never "reviewed").
"""
import os, sys, json, time, hashlib, argparse, datetime
import numpy as np, torch, torch.nn as nn
import torchvision.models as tvm

SP = os.path.dirname(os.path.abspath(__file__))                       # scripts/analysis
EXP = os.path.dirname(os.path.dirname(SP))                            # experiments/
RUN = os.path.join(EXP, "results", "runs", "ftvit_resisc45")
sys.path.insert(0, os.path.join(RUN, "scripts"))
from ftvit_paths import cache_path, stable_seed

DEV = torch.device(os.environ.get("VIT_DEV", "cuda") if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(8, torch.get_num_threads()))                # shared-server rule: <= 8 CPU threads
RGRID = [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]     # study line 80 (unchanged)
RGRID_EXT = [100.0, 300.0, 1000.0]                                    # 2026-11-22 radius extension
RFULL = RGRID + RGRID_EXT                                             # concatenated 13-radius grid
NORIG = len(RGRID)
TOL = 1e-3                      # established RELATIVE visibility bracket (comparison only; not retuned)
SEED_BASE = 20261122            # study line 86 (unchanged)
M_PROBES = 256
CURVES = ["actual", "frozen", "first_order", "actual_minus_frozen", "frozen_minus_first_order"]
METRICS = ["abs_rms_X", "rel_rms_X", "median_per_example_rel"]
GATE_ABS, GATE_REL = 1e-12, 1e-9                                      # validation-gate tolerance (fixed)
K_FLOOR = 8                                                           # reference probes per layer
FLOOR_FACTOR = 10.0                                                   # "within 10x of the floor"
EXACT_REF = {"mlp0": "mlp0_fn_exact", "mlp3": "mlp3_dn_exact", "head": "head_fn_exact"}

def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))             # study line 92
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0)))+z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r, g, b, eps):                                             # study lines 94-96
    mu = r.mean(-1, keepdim=True); sd = torch.sqrt(r.var(-1, unbiased=False, keepdim=True)+eps)
    return g*(r-mu)/sd+b

def sha256(path):                                                     # study lines 98-102
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()

def dist(x):                                                          # study lines 104-107
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.10)),
            "q90": float(np.quantile(x, 0.90)), "min": float(x.min()), "max": float(x.max())}

def bracket_full(drift13, tol=TOL):
    """Extension bracket on the CONCATENATED 13-radius grid (stored 10 ++ new 3)."""
    d = np.asarray(drift13, float); cross = np.where(d >= tol)[0]
    if len(cross) == 0: return {"last_safe": RFULL[-1], "first_cross": None, "censored": True}
    j = int(cross[0])
    return {"last_safe": (RFULL[j-1] if j > 0 else 0.0), "first_cross": RFULL[j], "censored": False}

GATE_RESULTS = {}
def gate(name, ok, detail):                                           # study lines 141-146
    GATE_RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(("GATE ok  " if ok else "GATE FAIL ")+name+" :: "+str(detail)[:300], flush=True)
    if not ok:
        raise AssertionError(f"GATE FAIL {name}")

def make_raw(side, n, m, basis, mode, rng, k):                        # study lines 641-649 verbatim
    if mode == "generic": return torch.randn(n, m, generator=rng, dtype=torch.float64)
    if side == "left":
        U = basis if mode == "family" else torch.linalg.qr(
            torch.randn(n, k, generator=rng, dtype=torch.float64))[0]
        return U@torch.randn(k, m, generator=rng, dtype=torch.float64)
    B = basis if mode == "family" else torch.linalg.qr(
        torch.randn(m, k, generator=rng, dtype=torch.float64))[0]
    return torch.randn(n, k, generator=rng, dtype=torch.float64)@B.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--M", type=int, default=M_PROBES)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: requires an explicit --outdir outside the run directory")
    ap.add_argument("--outdir", default=RUN, help="TEST ONLY: redirect the two output files "
                    "(production records always go to the canonical run directory)")
    args = ap.parse_args(); t_start = time.time(); torch.manual_seed(0)
    st = f"s{args.seed}"
    if args.smoke and os.path.realpath(args.outdir) == os.path.realpath(RUN):
        print("FAIL CLOSED: --smoke requires an explicit --outdir outside the canonical run dir.")
        sys.exit(1)
    if not args.smoke and args.M != M_PROBES:
        print(f"FAIL CLOSED: production runs use M={M_PROBES}; use --smoke for reduced M.")
        sys.exit(1)
    out_json = os.path.join(args.outdir, f"common_family_{st}_radius_ext.json")
    out_arr = os.path.join(args.outdir, f"common_family_{st}_radius_ext_arrays.pt")
    for p in (out_json, out_arr):
        if os.path.exists(p):
            print(f"FAIL CLOSED: output {p} already exists; this script writes NEW records only.")
            sys.exit(1)

    # ------------- source record + pre-run SHA snapshot (mutation check baseline) ------------------
    src_json = os.path.join(RUN, f"common_family_{st}.json")
    src_arr = os.path.join(RUN, f"common_family_{st}_arrays.pt")
    for p in (src_json, src_arr):
        if not os.path.exists(p): print(f"FAIL CLOSED: missing source record {p}"); sys.exit(1)
    src_json_sha, src_arr_sha = sha256(src_json), sha256(src_arr)
    rec = json.load(open(src_json))
    arrs = torch.load(src_arr, map_location="cpu", weights_only=False)
    gate("source_record_valid",
         rec["validation"]["passed"] is True and rec["M_probes_per_mode"] == M_PROBES,
         {"validation_passed": rec["validation"]["passed"], "M": rec["M_probes_per_mode"]})
    gate("source_axes_match",
         list(arrs["curve_axis"]) == CURVES and list(arrs["metric_axis"]) == METRICS
         and np.allclose(arrs["RGRID"].double().numpy(), np.array(RGRID), rtol=1e-6)
         and rec["conventions"]["radius_grid_absolute"] == RGRID,
         {"curves": list(arrs["curve_axis"]), "metrics": list(arrs["metric_axis"])})

    # ------------- inputs + SHA authentication vs the stored record (fail closed) ------------------
    ckpt = os.path.join(RUN, f"ckpt_ftvit_lr3e-5_confirm_{st}.pt")
    cache_file = cache_path(f"clsmid_{st}_n{args.n}.pt")
    for p in (ckpt, cache_file):
        if not os.path.exists(p): print(f"FAIL CLOSED: missing input {p}"); sys.exit(1)
    ckpt_sha, cache_sha = sha256(ckpt), sha256(cache_file)
    gate("sha_authentication_vs_common_family_record",
         rec["ckpt_sha256"] == ckpt_sha and rec["cache_sha256"] == cache_sha
         and rec["ckpt"] == ckpt and os.path.realpath(rec["cache_file"]) == os.path.realpath(cache_file),
         {"ckpt": ckpt_sha[:16], "cache": cache_sha[:16]})

    # ------------- base quantities: study lines 186-209 copied verbatim ----------------------------
    ck = torch.load(ckpt, map_location="cpu"); ncls = ck["n_classes"]
    net = tvm.vit_b_16(weights=None); net.heads = nn.Sequential()
    net.heads.add_module("head", nn.Linear(768, ncls))
    net.load_state_dict(ck["state_dict"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    blk = net.encoder.layers[11]
    cls_mid = torch.load(cache_file)["cls_mid"].double().cpu(); N = cls_mid.shape[0]
    assert N == args.n, (N, args.n)
    g2, b2 = blk.ln_2.weight.double(), blk.ln_2.bias.double(); eps2 = blk.ln_2.eps
    W0, b0 = blk.mlp[0].weight.double(), blk.mlp[0].bias.double()
    W3, b3 = blk.mlp[3].weight.double(), blk.mlp[3].bias.double()
    gf_c, bf_c = net.encoder.ln.weight.double(), net.encoder.ln.bias.double(); epsf = net.encoder.ln.eps
    V_c, bV_c = net.heads.head.weight.double(), net.heads.head.bias.double(); d_out = V_c.shape[0]
    u = ln_fwd(cls_mid, g2, b2, eps2); z0 = u@W0.T+b0; a = gelu(z0); z3 = a@W3.T+b3
    bo = cls_mid+z3
    s_bo_c = torch.sqrt(bo.var(-1, unbiased=False, keepdim=True)+epsf)
    rhat_c = (bo-bo.mean(-1, keepdim=True))/s_bo_c
    f = ln_fwd(bo, gf_c, bf_c, epsf); logits = f@V_c.T+bV_c
    gp = gelup(z0); one = torch.ones(N, 1, dtype=torch.float64)
    logits_rms = float(torch.sqrt((logits**2).sum(1).mean()))
    sqrtN = float(np.sqrt(N))
    gate("base_forward_logits_rms_vs_record",
         abs(logits_rms-rec["normalization_denominators"]["logits_rms_absolute"])
         <= 1e-9*rec["normalization_denominators"]["logits_rms_absolute"],
         {"logits_rms": logits_rms,
          "record": rec["normalization_denominators"]["logits_rms_absolute"]})
    print(f"[{time.time()-t_start:7.1f}s] base forward done, N={N}, logits_rms={logits_rms:.6f}, "
          f"dev={DEV}", flush=True)

    # ------------- persisted bases + exact-null vectors (NOT recomputed) ---------------------------
    nu_u, nu_f = arrs["nu_u"].clone(), arrs["nu_f"].clone()
    ones768 = torch.ones(768, dtype=torch.float64)
    # study lines 350-352 verbatim (projection of exact components, approximate families only)
    def proj_mlp0(dA): return dA-torch.outer(dA@nu_u, nu_u)
    def proj_mlp3(dA): return dA-torch.outer(ones768, dA.mean(0))
    def proj_head(dA): return dA-torch.outer(dA@nu_f, nu_f)
    PROJ = {"mlp0": proj_mlp0, "mlp3": proj_mlp3, "head": proj_head}
    # consistency of persisted nu vectors with the base forward (same quality bound as the study)
    Pu = torch.cat([u, one], 1); Pf = torch.cat([f, one], 1)
    gate("persisted_nu_null_quality",
         float((Pu@nu_u).abs().max()) < 1e-9 and float((Pf@nu_f).abs().max()) < 1e-9,
         {"max_resid_nu_u": float((Pu@nu_u).abs().max()),
          "max_resid_nu_f": float((Pf@nu_f).abs().max())})
    del Pu, Pf

    LAY_NM = {"mlp0": (3072, 769), "mlp3": (768, 3073), "head": (d_out, 769)}
    batteries_todo = []     # (tag, layer, side, basis CPU (dim x k), proj) from the STORED record
    worst = 0.0
    for tag, brec in rec["batteries"].items():
        layer, side, k = brec["layer"], brec["factor_side"], brec["k_factor"]
        basis = arrs[f"basis_{tag}"].clone()
        gate_ok_k = (basis.shape[1] == k)
        if not gate_ok_k: gate(f"k_factor_vs_json_{tag}", False, {"basis_cols": basis.shape[1], "k": k})
        g = basis.T@basis-torch.eye(basis.shape[1], dtype=torch.float64)
        worst = max(worst, float(g.abs().max()))
        is_exact = tag.endswith("_exact")
        removal_says_none = brec["exact_component_removal"].startswith("none:")
        if is_exact != removal_says_none:
            gate(f"proj_convention_{tag}", False, {"tag": tag, "removal": brec["exact_component_removal"]})
        batteries_todo.append((tag, layer, side, basis, None if is_exact else PROJ[layer]))
    gate("loaded_basis_orthonormality_and_k", worst < 1e-10 and all(
         arrs[f"basis_{t}"].shape[1] == rec["batteries"][t]["k_factor"] for t in rec["batteries"]),
         {"max_abs_gram_err": worst, "batteries": [(t[0], t[3].shape[1]) for t in batteries_todo]})

    # ------------- DEV copies + evaluators: study lines 565-625 verbatim, grid parametrized --------
    gf_d, bf_d = gf_c.to(DEV), bf_c.to(DEV)
    rhat_d, s_bo_d = rhat_c.to(DEV), s_bo_c.to(DEV)
    gp_d = gp.to(DEV)
    u_d, a_d, z0_d = u.to(DEV), a.to(DEV), z0.to(DEV)
    cls_d, bo_d, f_d, logits_d = cls_mid.to(DEV), bo.to(DEV), f.to(DEV), logits.to(DEV)
    b3_d, bV_d = b3.to(DEV), bV_c.to(DEV)
    W3T_d = W3.T.contiguous().to(DEV); VT_d = V_c.T.contiguous().to(DEV)
    V_full_d = V_c.to(DEV)
    base_pern = logits_d.norm(dim=1)+1e-30

    def apply_JLN(delta):                                             # study lines 572-575
        c = delta-delta.mean(-1, keepdim=True)-(((rhat_d*delta).sum(-1, keepdim=True))/768)*rhat_d
        return (gf_d/s_bo_d)*c

    def met(dd):                                                      # study lines 577-580
        pern = dd.norm(dim=1)
        ab = float(torch.sqrt((pern**2).mean()))
        return (ab, ab/logits_rms, float((pern/base_pern).median()))

    # evaluators: study lines 581-625 verbatim with RGRID -> `grid` and the vector-capture
    # branches removed (no chunked validation vectors are produced by the extension)
    def eval_mlp0(dA, grid):
        dz0 = u_d@dA[:, :-1].T+dA[:, -1]
        d3lin = (gp_d*dz0)@W3T_d; jln = apply_JLN(d3lin); fo = jln@VT_d
        jf = float(fo.norm())
        C = np.zeros((5, 3, len(grid))); S = np.zeros((2, len(grid)))
        for ri, t in enumerate(grid):
            an = gelu(z0_d+t*dz0); bon = cls_d+an@W3T_d+b3_d
            d_act = ln_fwd(bon, gf_d, bf_d, epsf)@VT_d+bV_d-logits_d
            d_fro = (f_d+t*jln)@VT_d+bV_d-logits_d          # frozen GELU+LN forward (own float route)
            d_fo = t*fo
            for ci, dd in enumerate((d_act, d_fro, d_fo, d_act-d_fro, d_fro-d_fo)):
                C[ci, :, ri] = met(dd)
            S[0, ri] = met(ln_fwd(bo_d+t*d3lin, gf_d, bf_d, epsf)@VT_d+bV_d-logits_d)[1]  # frozenGELU only
            S[1, ri] = met(apply_JLN(bon-bo_d)@VT_d)[1]                                   # frozenLN only
        return jf, C, S

    def eval_mlp3(dA, grid):
        dz3 = a_d@dA[:, :-1].T+dA[:, -1]
        jln = apply_JLN(dz3); fo = jln@VT_d; jf = float(fo.norm())
        C = np.zeros((5, 3, len(grid)))
        for ri, t in enumerate(grid):
            d_act = ln_fwd(bo_d+t*dz3, gf_d, bf_d, epsf)@VT_d+bV_d-logits_d
            d_fro = (f_d+t*jln)@VT_d+bV_d-logits_d
            d_fo = t*fo
            for ci, dd in enumerate((d_act, d_fro, d_fo, d_act-d_fro, d_fro-d_fo)):
                C[ci, :, ri] = met(dd)
        return jf, C, None

    def eval_head(dA, grid):
        fo = f_d@dA[:, :-1].T+dA[:, -1]; jf = float(fo.norm())
        C = np.zeros((5, 3, len(grid)))
        for ri, t in enumerate(grid):
            d_act = f_d@(V_full_d+t*dA[:, :-1]).T+(bV_d+t*dA[:, -1])-logits_d
            d_fro = f_d@VT_d+bV_d+t*fo-logits_d
            d_fo = t*fo
            for ci, dd in enumerate((d_act, d_fro, d_fo, d_act-d_fro, d_fro-d_fo)):
                C[ci, :, ri] = met(dd)
        return jf, C, None

    EVAL = {"mlp0": eval_mlp0, "mlp3": eval_mlp3, "head": eval_head}

    # ------------- NUMERICAL-FLOOR DIAGNOSTIC (K=8 exactly invariant reference probes/layer) -------
    # Evaluated on the full 13-radius grid. The measured actual-route relative drift of these
    # EXACT directions at r=100/300/1000 IS the fp64 rounding floor of the evaluation at that
    # radius; through r<=30 it reproduces the study's machine-floor exact batteries.
    floors = {}; floor_arrays = {}
    for layer in ("mlp0", "mlp3", "head"):
        ref_tag = EXACT_REF[layer]
        if ref_tag not in rec["batteries"]:
            gate(f"floor_reference_present_{layer}", False, {"missing": ref_tag})
        brec = rec["batteries"][ref_tag]
        basis = arrs[f"basis_{ref_tag}"].clone()
        n, m = LAY_NM[layer]; k = basis.shape[1]; side = brec["factor_side"]
        fseed = stable_seed(SEED_BASE, f"commonext_{st}_floor_{layer}", "family")
        rngf = torch.Generator().manual_seed(fseed)
        FC = np.zeros((K_FLOOR, 5, 3, len(RFULL)))
        for i in range(K_FLOOR):
            raw = make_raw(side, n, m, basis, "family", rngf, k)
            dA = (raw/raw.norm()).to(DEV)
            _, Cc, _ = EVAL[layer](dA, RFULL)
            FC[i] = Cc
        floor_arrays[f"floor_{layer}_drift_fullgrid"] = torch.tensor(FC)
        floors[layer] = {
            "reference_battery": ref_tag, "K": K_FLOOR, "generator_seed": fseed,
            "basis": f"persisted basis_{ref_tag} (exact family, structurally invariant)",
            "per_radius": {f"r{r:g}": {"floor_median": float(np.median(FC[:, 0, 1, NORIG+ei])),
                                       "floor_max": float(FC[:, 0, 1, NORIG+ei].max())}
                           for ei, r in enumerate(RGRID_EXT)},
            "original_grid_actual_rel_max": float(FC[:, 0, 1, :NORIG].max()),
            "meaning": "measured actual-route rel_rms_X drift of exactly invariant directions = "
                       "fp64 rounding floor of the evaluation at that radius (cancellation in "
                       "GELU/LN at large perturbation scale); NOT genuine output change"}
        print(f"[{time.time()-t_start:7.1f}s] floor {layer}: "
              + " ".join(f"r{r:g}={floors[layer]['per_radius'][f'r{r:g}']['floor_max']:.3e}"
                         for r in RGRID_EXT), flush=True)
    gate("floor_reference_machine_floor_on_original_grid",
         all(floors[L]["original_grid_actual_rel_max"] < 1e-9 for L in floors),
         {L: floors[L]["original_grid_actual_rel_max"] for L in floors})
    floor_max_ref = {L: np.array([floors[L]["per_radius"][f"r{r:g}"]["floor_max"]
                                  for r in RGRID_EXT]) for L in floors}
    if DEV.type == "cuda": torch.cuda.empty_cache()

    # ------------- batteries: regenerate probes, gate on the original grid, extend -----------------
    batteries_ext = {}; arrays_out = {}; gate_global = {"max_absdev": 0.0, "max_reldev_absfail": 0.0}
    timing = {}
    for tag, layer, side, basis, proj in batteries_todo:
        n, m = LAY_NM[layer]; k = basis.shape[1]; ev = EVAL[layer]
        brec_src = rec["batteries"][tag]
        brec = {"layer": layer, "factor_side": side, "k_factor": k, "M": args.M,
                "exact_component_removal": brec_src["exact_component_removal"],
                "family_dim_param_space": brec_src["family_dim_param_space"],
                "modes": {}}
        for mode in ("family", "randsub", "generic"):
            t_mode = time.time()
            seed = stable_seed(SEED_BASE, f"common_{st}_{tag}", mode)
            gate_ok_seed = (seed == brec_src["modes"][mode]["generator_seed"])
            if not gate_ok_seed:
                gate(f"generator_seed_{tag}_{mode}", False,
                     {"regen": seed, "stored": brec_src["modes"][mode]["generator_seed"]})
            rng = torch.Generator().manual_seed(seed)
            stored = arrs[f"battery_{tag}_{mode}_drift"][:args.M].numpy()
            stored_supp = (arrs[f"battery_{tag}_{mode}_partialfreeze_rel"][:args.M].numpy()
                           if layer == "mlp0" else None)
            stored_per = brec_src["modes"][mode]["per_direction"][:args.M]
            drift_ext = np.zeros((args.M, 5, 3, len(RGRID_EXT)))
            supp_ext = np.zeros((args.M, 2, len(RGRID_EXT))) if layer == "mlp0" else None
            dev_abs = 0.0; dev_rel_absfail = 0.0; dev_aux = 0.0
            per = []; jv_abs = []; jv_rel = []; ls = []; fcs = []; cens = 0
            for mi in range(args.M):
                raw = make_raw(side, n, m, basis, mode, rng, k)
                nrm_raw = float(raw.norm())
                if proj is not None: raw = proj(raw)
                nrm = float(raw.norm()); dA = (raw/nrm).to(DEV)
                jf, Cc, Ss = ev(dA, RFULL)
                # ---- validation gate accumulation on the original 10 radii ----
                d = np.abs(Cc[:, :, :NORIG]-stored[mi])
                bad = d > GATE_ABS
                if bad.any():
                    rel = d[bad]/np.maximum(np.abs(stored[mi][bad]), 1e-300)
                    dev_rel_absfail = max(dev_rel_absfail, float(rel.max()))
                dev_abs = max(dev_abs, float(d.max()))
                if stored_supp is not None:
                    ds = np.abs(Ss[:, :NORIG]-stored_supp[mi])
                    bs = ds > GATE_ABS
                    if bs.any():
                        rels = ds[bs]/np.maximum(np.abs(stored_supp[mi][bs]), 1e-300)
                        dev_rel_absfail = max(dev_rel_absfail, float(rels.max()))
                    dev_abs = max(dev_abs, float(ds.max()))
                sp = stored_per[mi]
                for val, ref in ((jf, sp["jv_frob"]), (nrm_raw, sp["norm_before_projection"]),
                                 (nrm, sp["norm_before_unit"])):
                    dev_aux = max(dev_aux, abs(val-ref)/max(abs(ref), 1e-300))
                # ---- extension radii ----
                drift_ext[mi] = Cc[:, :, NORIG:]
                if supp_ext is not None: supp_ext[mi] = Ss[:, NORIG:]
                conc = np.concatenate([stored[mi, 0, 1, :], Cc[0, 1, NORIG:]])
                br = bracket_full(conc)
                jv_abs.append(jf/sqrtN); jv_rel.append(jf/sqrtN/logits_rms)
                ls.append(br["last_safe"]); cens += int(br["censored"])
                if br["first_cross"] is not None: fcs.append(br["first_cross"])
                at_floor = [bool(drift_ext[mi, 0, 1, ei] <= FLOOR_FACTOR*floor_max_ref[layer][ei])
                            for ei in range(len(RGRID_EXT))]
                per.append({"i": mi, "jv_abs_rate_X_per_unit_v": jv_abs[-1], "jv_rel_rate": jv_rel[-1],
                            "jv_frob": jf, "actual_rel_drift_ext": [float(x) for x in drift_ext[mi, 0, 1]],
                            "classification_ext": ["at_numerical_floor" if afl else "genuine_output_change"
                                                   for afl in at_floor],
                            "extension_bracket_concatenated_13radius_rel_tol1e-3": br})
            per_mode_gate = bool(dev_rel_absfail <= GATE_REL and dev_aux <= GATE_REL)
            gate_global["max_absdev"] = max(gate_global["max_absdev"], dev_abs)
            gate_global["max_reldev_absfail"] = max(gate_global["max_reldev_absfail"],
                                                    dev_rel_absfail, dev_aux)
            if not per_mode_gate:
                gate(f"validation_same_directions_{tag}_{mode}", False,
                     {"max_absdev": dev_abs, "max_reldev_where_abs_fails": dev_rel_absfail,
                      "max_reldev_jv_norms": dev_aux,
                      "tolerance": f"<= {GATE_ABS} abs OR <= {GATE_REL} rel, no loosening"})
            exceed = {f"r{r:g}": int((drift_ext[:, 0, 1, ei] > TOL).sum())
                      for ei, r in enumerate(RGRID_EXT)}
            at_floor_mask = drift_ext[:, 0, 1, :] <= FLOOR_FACTOR*floor_max_ref[layer][None, :]
            cls_counts = {}
            for ei, r in enumerate(RGRID_EXT):
                exm = drift_ext[:, 0, 1, ei] > TOL
                cls_counts[f"r{r:g}"] = {
                    "n_at_numerical_floor": int(at_floor_mask[:, ei].sum()),
                    "n_genuine_output_change": int((~at_floor_mask[:, ei]).sum()),
                    "n_exceed_rel_1e-3": int(exm.sum()),
                    "n_exceed_genuine": int((exm & ~at_floor_mask[:, ei]).sum()),
                    "n_exceed_at_floor": int((exm & at_floor_mask[:, ei]).sum())}
            agg = {cn: {me: {f"r{r:g}": dist(drift_ext[:, ci, mi_, ei]) for ei, r in enumerate(RGRID_EXT)}
                        for mi_, me in enumerate(METRICS)} for ci, cn in enumerate(CURVES)}
            brec["modes"][mode] = {
                "generator_seed": seed, "M": args.M,
                "validation_gate": {"pass": per_mode_gate, "max_absdev_vs_stored": dev_abs,
                                    "max_reldev_where_abs_exceeded": dev_rel_absfail,
                                    "max_reldev_jv_and_norms": dev_aux,
                                    "stored_rows_compared": f"[0:{args.M}]",
                                    "tolerance": f"abs <= {GATE_ABS} OR rel <= {GATE_REL}"},
                "jv_abs_rate_X_per_unit_v": dist(jv_abs), "jv_rel_rate": dist(jv_rel),
                "extension_last_safe_tested_rel_tol1e-3_13radius": dist(ls),
                "extension_first_cross_tested_13radius": (dist(fcs) if fcs else None),
                "censored_count_13radius": cens,
                "exceedance_count_rel_gt_1e-3_among_M": exceed,
                "floor_classification_counts": cls_counts,
                "aggregates_per_new_radius": agg, "per_direction": per,
                "seconds": time.time()-t_mode}
            timing[f"{tag}/{mode}"] = time.time()-t_mode
            arrays_out[f"battery_{tag}_{mode}_drift_ext"] = torch.tensor(drift_ext)
            if supp_ext is not None:
                arrays_out[f"battery_{tag}_{mode}_partialfreeze_rel_ext"] = torch.tensor(supp_ext)
            arrays_out[f"battery_{tag}_{mode}_at_floor_ext"] = torch.tensor(at_floor_mask)
            print(f"[{time.time()-t_start:7.1f}s] {tag:18s} {mode:8s} gate_absdev={dev_abs:.3e} "
                  f"cens13={cens} exceed={exceed} "
                  f"genuine@r1000={cls_counts[f'r{RGRID_EXT[-1]:g}']['n_genuine_output_change']}",
                  flush=True)
        batteries_ext[tag] = brec
        if DEV.type == "cuda": torch.cuda.empty_cache()
    gate("validation_same_directions_all_batteries",
         gate_global["max_reldev_absfail"] <= GATE_REL,
         {**gate_global, "statement": "every re-evaluated probe matches the stored "
          "battery_<tag>_<mode>_drift (and partial-freeze, jv, norms) rows [0:M] entrywise at "
          f"abs <= {GATE_ABS} OR rel <= {GATE_REL}: same directions, normalization, precision"})

    # ------------- persist (NEW files only) --------------------------------------------------------
    peak_mem = (torch.cuda.max_memory_allocated(DEV)/2**30 if DEV.type == "cuda" else None)
    note = ("Exact structural invariance is an algebraic, radius-independent statement: for a "
            "direction in a certified exact family the network output is unchanged at EVERY radius "
            "by construction, with no sampling involved. The finite-radius battery results recorded "
            "here are sampled evidence at the tested radii only (M probes per mode, 13 tested radii, "
            "exceedances counted at the fixed relative tolerance 1e-3, no p-values). At large radii "
            "the fp64 evaluation itself has a nonzero rounding floor (cancellation in the GELU and "
            "LayerNorm paths grows with the perturbation scale); the floor measured here from "
            "exactly invariant reference directions is the resolution limit of the evaluation at "
            "that radius, not a property of the probed family. A battery drift within 10x of that "
            "floor is flagged at_numerical_floor and is NOT evidence of genuine output change; "
            "censoring at or near the floor means 'no genuine change detected above the fp64 floor "
            "at the tested radii', not a certified invariance bound. Raw numbers are kept "
            "regardless of the flag.")
    validation = {"label": "automatically checked (in-run gates; NOT independent review)",
                  "checks": GATE_RESULTS,
                  "passed": bool(all(v["pass"] for v in GATE_RESULTS.values() if v["pass"] is not None))}
    arrays_out.update({"RGRID_ORIG": torch.tensor(RGRID, dtype=torch.float64),
                       "RGRID_EXT": torch.tensor(RGRID_EXT, dtype=torch.float64),
                       "RGRID_FULL": torch.tensor(RFULL, dtype=torch.float64),
                       "curve_axis": CURVES, "metric_axis": METRICS,
                       **floor_arrays})
    out = {
        "study": "COMMON ViT FAMILY STUDY -- RADIUS-GRID EXTENSION r in {100, 300, 1000}",
        "instruction": "2026-11-22 radius extension (PI): append r=100, 300, 1000 to the "
                       "common-family batteries, reusing the same 256 directions, normalization, "
                       "precision, controls and paired-response diagnostics; preserve existing "
                       "results; report crossings or continued censoring; distinguish numerical "
                       "breakdown from genuine output change; keep exact structural invariance "
                       "distinct from sampled finite-radius evidence",
        "seed_tag": st, "N_examples": N, "M_probes_per_mode": args.M, "smoke": bool(args.smoke),
        "device": str(DEV), "dtype": "float64", "torch_version": torch.__version__,
        "date": datetime.date.today().isoformat(),
        "radii_ext": RGRID_EXT, "radii_original": RGRID, "radii_full_concatenated": RFULL,
        "provenance": {
            "script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
            "source_record_json": src_json, "source_record_json_sha256_pre_run": src_json_sha,
            "source_record_arrays": src_arr, "source_record_arrays_sha256_pre_run": src_arr_sha,
            "ckpt": ckpt, "ckpt_sha256": ckpt_sha, "cache_file": cache_file, "cache_sha256": cache_sha,
            "probe_streams": "stable_seed(20261122, 'common_<seed>_<tag>', mode) regenerated; "
                             "first M draws of each stream; bases loaded from the stored arrays "
                             "file, never recomputed"},
        "conventions_inherited_unchanged": rec["conventions"],
        "normalization_denominators": rec["normalization_denominators"],
        "exact_vs_sampled_note": note,
        "numerical_floor_diagnostic": {
            "K_per_layer": K_FLOOR, "classification_rule":
                f"actual-route rel_rms_X drift <= {FLOOR_FACTOR} x floor_max(layer, radius) => "
                "at_numerical_floor; else genuine_output_change",
            "layers": floors},
        "batteries_ext": batteries_ext,
        "validation": validation,
        "timing": {"per_battery_mode_s": timing,
                   "total_runtime_s": time.time()-t_start,
                   "peak_gpu_mem_gib": peak_mem},
    }
    torch.save(arrays_out, out_arr+".tmp"); os.replace(out_arr+".tmp", out_arr)
    with open(out_json+".tmp", "w") as fo: json.dump(out, fo, indent=1)
    os.replace(out_json+".tmp", out_json)
    print("validation passed:", validation["passed"])
    print("wrote", out_json, "\nDONE", flush=True)

if __name__ == "__main__":
    main()
