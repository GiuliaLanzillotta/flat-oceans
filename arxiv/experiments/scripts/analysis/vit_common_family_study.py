#!/usr/bin/env python3
"""COMMON ViT FAMILY STUDY (two-day completion pipeline, GPU family lane; 2026-11-22).

usage: vit_common_family_study.py <seed>        (seed in {0,1,2}; one seed per invocation)

Authoritative specs: memos/2026-11-17_reviewer_protocol_corrections.md (SS1 norms/rates, SS2 marginal
family definitions), memos/2026-11-22_two_day_completion_plan.md ("Common ViT and ResNet measurements"),
memos/2026-11-17_vit_resnet_standardization_spec.md (SS2 common protocol). Machinery reused from
results/runs/ftvit_resisc45/scripts/{phase0f_families.py, recut_pilot_mlp3_dtail.py, vit_head_multilayer.py}.

Selected layers (block-11 of the fine-tuned ViT-B/16, RESISC45): mlp.0 (3072x769 aug), mlp.3 (768x3073
aug), readout head (45x769 aug). All three layers have bias parameters, so every input uses the augmented
homogeneous coordinate [p_i; 1]. Analysis set: N = 10,000 cached class-token residuals (cls_mid),
float64 throughout.

Per layer, with edit A in R^(n x m), inputs p_i in R^m and ORIGINAL downstream preactivation derivative
D_i in R^(45 x n) (mlp.0: through GELU -> mlp.3 -> +residual -> final LN -> head; mlp.3: through
+residual -> final LN -> head; head: D = I structurally):

    H = span{p_i},  C = span{D_i^T e_a}  (all 45 output channels),
    FN            = {Z U_Hperp^T}                       (exact feature-null),
    DN_additional = {U_Cperp Z U_H^T},                  N = FN (+) DN_additional (orthogonal).

Conventions (fixed; see plan memo "Fixed conventions"):
  T(i)  exact rank: sigma <= 1e-9 * sigma0 of THAT operator, with the gap-10 resolution check
        (RANK_GAP_RESOLVED = 10); unresolved ranks are reported as unresolved with their ambiguity band,
        never forced. Gram/eigh-derived sigmas additionally carry the eigh noise floor
        sqrt(eps_mach)*sigma0; a sigma statement below that floor is not certified by eigh alone.
  T(ii) approximate tails: ABSOLUTE singular RATE sigma/sqrt(N) <= 1e-3 in the factor operator's OWN
        units (N = examples). Feature-factor rates are input-activation RMS per unit right-factor
        direction; downstream-factor rates are logit RMS per unit activation direction. These are
        ACTIVATION-SPACE operators; they are never parameter-Jacobian spectra, and a marginal rate is
        not automatically a parameter-to-logit bound: the realized edits' original-network
        parameter-to-output JVP rates are reported separately per probe (reviewer SS1).
  EMPTY families are VALID results: k = 0 is recorded as a genuine zero with the boundary singular
  rate; structurally empty cells get no probes.
  Battery (every NONEMPTY family): M = 256 probes per mode, generator-seeded (seeds fixed and recorded
  before the sweep; draws are independent of any sweep result), unit-Frobenius [W|b] edits, absolute
  radius grid {1e-3,...,30}; dim-matched random-subspace + generic controls; for APPROXIMATE families
  the layer's exact-null components are projected out of family AND controls identically (phase0f
  convention); drift recorded as ABSOLUTE dataset-RMS ||Delta F||_X AND relative ||Delta F||_X/||F||_X
  (two labelled metrics) plus supplementary median per-example drift; paired VECTORS actual / frozen /
  Jv with vector residuals actual-minus-frozen and frozen-minus-Jv (never scalar-norm differences);
  exceedance counts among 256, no p-values. Frozen = the existing validated frozen forward (GELU gates
  and LN Jacobians fixed at base); for a SINGLE-layer edit the frozen response is linear in the radius
  and structurally coincides with the first-order JVP -- both are still computed through their own float
  routes and the vector residual is persisted.
  Selective families: the paper's constraints S_i A p_i = 0 with the precise operator recorded.
  mlp.0: S_i = diag(gelu'(z0_i)); the analytic GELU derivative vanishes only at the isolated point
  z* ~ -0.75179, so nonzero derivatives are NEVER rounded to exact dead units; the exact selective
  space coincides with FN and the census is a candidate screen only. mlp.3: selective columns are fed
  by low-activity GELU units; exactly-dead columns would lie in FN (which is empty). Approximate
  axis-aligned candidate screens use the same absolute 1e-3 rate in the respective factor operator's
  units. head: linear, no gates -> structurally empty.
  LN-shift exact directions (mlp.3 DN_additional = {ones_768 y^T}, dim 3073) are the separately
  validated mechanism: reused analytically and REVALIDATED (direct dataset JVP sigma + machine-precision
  nonlinear drift at r = 30), not recomputed from an eigenbasis.

Outputs (NEW files only; never touches an existing record):
  results/runs/ftvit_resisc45/common_family_s<seed>.json
  results/runs/ftvit_resisc45/common_family_s<seed>_arrays.pt
  results/runs/ftvit_resisc45/common_family_s<seed>_vectors_<layer>.pt   (chunked validation vectors)

Gates (fail-closed; a failed gate prints 'GATE FAIL' and exits nonzero): input SHA authentication
(where a prior reference exists), exact ranks vs head_multilayer_s<seed>.json, spectrum summaries vs
phase0d_s0.json (seed 0), bespoke tail counts vs phase0f_s<seed>_m256.json, GPU/CPU Gram chunk
agreement, basis orthonormality, exact-null machine precision, autograd-vs-analytic JVP spot agreement
(the validated batched-fp64 JVP route)."""
import os, sys, json, time, hashlib, argparse
import numpy as np, torch, torch.nn as nn
import torchvision.models as tvm

SP = os.path.dirname(os.path.abspath(__file__))                       # scripts/analysis
EXP = os.path.dirname(os.path.dirname(SP))                            # experiments/
RUN = os.path.join(EXP, "results", "runs", "ftvit_resisc45")
sys.path.insert(0, os.path.join(RUN, "scripts"))
from ftvit_paths import cache_path, stable_seed

DEV = torch.device(os.environ.get("VIT_DEV", "cuda") if torch.cuda.is_available() else "cpu")
RGRID = [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TOL = 1e-3                      # established RELATIVE visibility bracket (comparison only)
RATE_CUT = 1e-3                 # T(ii): absolute sigma/sqrt(N) in the factor operator's own units
RANK_REL = 1e-9                 # T(i)
GAP10 = 10.0
M_PROBES = 256
SEED_BASE = 20261122
CURVES = ["actual", "frozen", "first_order", "actual_minus_frozen", "frozen_minus_first_order"]
METRICS = ["abs_rms_X", "rel_rms_X", "median_per_example_rel"]
VEC_RIDX = (6, 8)               # capture radii RGRID[6]=1.0, RGRID[8]=10.0
NVEC, NVAL = 4, 256             # probes / examples in the chunked validation-vector subset

def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0)))+z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r, g, b, eps):
    mu = r.mean(-1, keepdim=True); sd = torch.sqrt(r.var(-1, unbiased=False, keepdim=True)+eps)
    return g*(r-mu)/sd+b

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()

def dist(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.10)),
            "q90": float(np.quantile(x, 0.90)), "min": float(x.min()), "max": float(x.max())}

def bracket(drift, tol=TOL):
    """Cumulative TESTED bracket on the common grid (phase0f convention)."""
    d = np.asarray(drift, float); cross = np.where(d >= tol)[0]
    if len(cross) == 0: return {"last_safe": RGRID[-1], "first_cross": None, "censored": True}
    j = int(cross[0])
    return {"last_safe": (RGRID[j-1] if j > 0 else 0.0), "first_cross": RGRID[j], "censored": False}

def rank_T1(sv, operator, units, gram_floor_sigma=None):
    """T(i) rank statement with the gap-10 resolution check; ambiguity bands retained."""
    sv = np.sort(np.asarray(sv, float))[::-1]
    s0 = float(sv[0]); cut = RANK_REL*s0
    nbelow = int((sv <= cut).sum()); r = len(sv)-nbelow
    small_ret = float(sv[r-1]) if r > 0 else 0.0
    large_null = float(sv[r]) if nbelow > 0 else None
    if nbelow > 0:
        gap = small_ret/max(large_null, 1e-300) if r > 0 else float("inf")
    else:
        gap = small_ret/cut
    resolved = bool(gap >= GAP10)
    floor_ok = True
    if gram_floor_sigma is not None:                     # eigh-route resolution qualification
        floor_ok = bool(small_ret >= GAP10*gram_floor_sigma)
        if nbelow == 0: resolved = resolved and floor_ok # a full-rank claim needs sigma_min above the floor
    band = [float(x) for x in sv[(sv > cut/GAP10) & (sv < cut*GAP10)]]
    return {"operator": operator, "units": units, "n_sigmas": int(len(sv)),
            "sigma0": s0, "sigma_min": float(sv[-1]), "cut_rel": RANK_REL, "cut_abs": cut,
            "rank": int(r), "null_dim": int(nbelow), "smallest_retained_sigma": small_ret,
            "largest_null_sigma": large_null, "gap": float(gap),
            "gap_resolved_ge_10x": resolved, "eigh_noise_floor_sigma": gram_floor_sigma,
            "floor_margin_ok_10x": floor_ok,
            "ambiguity_band_sigmas_within_10x_of_cut": band}

GATE_RESULTS = {}
def gate(name, ok, detail):
    GATE_RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(("GATE ok  " if ok else "GATE FAIL ")+name+" :: "+str(detail)[:300], flush=True)
    if not ok:
        raise AssertionError(f"GATE FAIL {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int, choices=[0, 1, 2])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--M", type=int, default=M_PROBES)
    ap.add_argument("--outdir", default=RUN, help="TEST ONLY: redirect the three output files "
                    "(production records always go to the canonical run directory)")
    args = ap.parse_args(); t_start = time.time(); torch.manual_seed(0)
    st = f"s{args.seed}"
    out_json = os.path.join(args.outdir, f"common_family_{st}.json")
    out_arr = os.path.join(args.outdir, f"common_family_{st}_arrays.pt")
    for p in (out_json, out_arr):
        if os.path.exists(p):
            print(f"FAIL CLOSED: output {p} already exists; this script writes NEW records only.")
            sys.exit(1)

    # ---------------- inputs + provenance (fail-closed where a prior reference exists) -------------
    ckpt = os.path.join(RUN, f"ckpt_ftvit_lr3e-5_confirm_{st}.pt")
    cache_file = cache_path(f"clsmid_{st}_n{args.n}.pt")
    for p in (ckpt, cache_file):
        if not os.path.exists(p): print(f"FAIL CLOSED: missing input {p}"); sys.exit(1)
    ckpt_sha, cache_sha = sha256(ckpt), sha256(cache_file)
    sha_ref = {"reference": None}
    if args.seed == 0:
        ref = json.load(open(os.path.join(RUN, "recut_pilot_mlp3_dtail_s0.json")))
        gate("sha_authentication_vs_recut_pilot",
             ref["ckpt_sha256"] == ckpt_sha and ref["cache_sha256"] == cache_sha,
             {"ckpt": ckpt_sha[:16], "cache": cache_sha[:16]})
        sha_ref = {"reference": "recut_pilot_mlp3_dtail_s0.json"}
    else:
        sha_ref = {"reference": None,
                   "note": "no prior SHA record for this seed; SHAs recorded here; inputs are "
                           "functionally authenticated by the exact-rank and tail-count gates below"}
    hm_ref = json.load(open(os.path.join(RUN, f"head_multilayer_{st}.json")))["exact_dims"]
    p0f_ref = {k: v["tail_dim_k"] for k, v in
               json.load(open(os.path.join(RUN, f"phase0f_{st}_m256.json")))["families"].items()}
    p0d_ref = (json.load(open(os.path.join(RUN, "phase0d_s0.json"))) if args.seed == 0 else None)

    ck = torch.load(ckpt, map_location="cpu"); ncls = ck["n_classes"]
    net = tvm.vit_b_16(weights=None); net.heads = nn.Sequential()
    net.heads.add_module("head", nn.Linear(768, ncls))
    net.load_state_dict(ck["state_dict"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    blk = net.encoder.layers[11]

    # ---------------- base forward, CPU float64 (phase0f lines verbatim) ---------------------------
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
    print(f"[{time.time()-t_start:7.1f}s] base forward done, N={N}, logits_rms={logits_rms:.6f}", flush=True)

    # ---------------- feature operators H (stacked [p;1], CPU LAPACK SVD) --------------------------
    Pu = torch.cat([u, one], 1); _, svu_t, VhU = torch.linalg.svd(Pu, full_matrices=False)
    Pa = torch.cat([a, one], 1); _, sva_t, VhA = torch.linalg.svd(Pa, full_matrices=False)
    Pf = torch.cat([f, one], 1); _, svf_t, VhF = torch.linalg.svd(Pf, full_matrices=False)
    sv_u, sv_a, sv_f = svu_t.numpy(), sva_t.numpy(), svf_t.numpy()
    print(f"[{time.time()-t_start:7.1f}s] feature SVDs done", flush=True)

    # ---------------- downstream operators C (GPU accumulation + CPU chunk gate, CPU eigh) ---------
    V_d, gf_d, bf_d = V_c.to(DEV), gf_c.to(DEV), bf_c.to(DEV)
    rhat_d, s_bo_d = rhat_c.to(DEV), s_bo_c.to(DEV)
    W3_d, gp_d = W3.to(DEV), gp.to(DEV)

    def VJ_batch(idx, dev):
        """d logits / d z3 for examples idx (b x 45 x 768); phase0f formula."""
        if dev:  Vg = V_d*gf_d;  rh = rhat_d[idx];  sb = s_bo_d[idx]
        else:    Vg = V_c*gf_c;  rh = rhat_c[idx];  sb = s_bo_c[idx]
        rsum = Vg.sum(1); Vgr = rh@Vg.T
        return ((Vg.unsqueeze(0)-rsum.view(1, -1, 1)/768)
                - torch.einsum("bc,bm->bcm", Vgr, rh)/768)/sb.view(-1, 1, 1)

    G0 = torch.zeros((3072, 3072), dtype=torch.float64, device=DEV)
    G3 = torch.zeros((768, 768), dtype=torch.float64, device=DEV)
    for i in range(0, N, 500):
        sl = torch.arange(i, min(i+500, N))
        VJ = VJ_batch(sl.to(DEV) if DEV.type == "cuda" else sl, True)
        G3 += VJ.reshape(-1, 768).T@VJ.reshape(-1, 768)
        D0 = torch.einsum("bcx,xy->bcy", VJ, W3_d)*gp_d[sl].unsqueeze(1)
        G0 += D0.reshape(-1, 3072).T@D0.reshape(-1, 3072)
        if i == 0:                                     # GPU/CPU agreement gate on the first chunk
            VJc = VJ_batch(sl, False)
            D0c = torch.einsum("bcx,xy->bcy", VJc, W3)*gp[sl].unsqueeze(1)
            e3 = float((VJ.cpu()-VJc).norm()/VJc.norm()); e0 = float((D0.cpu()-D0c).norm()/D0c.norm())
            gate("gpu_cpu_downstream_chunk_agreement", e3 < 1e-12 and e0 < 1e-12,
                 {"VJ_rel_err": e3, "D0_rel_err": e0, "device": str(DEV)})
    G0 = ((G0+G0.T)/2).cpu(); G3 = ((G3+G3.T)/2).cpu()
    ones768 = torch.ones(768, dtype=torch.float64)
    Qsh = torch.linalg.svd(torch.eye(768, dtype=torch.float64)
                           - torch.outer(ones768/np.sqrt(768.), ones768/np.sqrt(768.)))[0][:, :767]
    ev3, evec3 = torch.linalg.eigh(Qsh.T@G3@Qsh)
    sv_d3 = np.sqrt(ev3.flip(0).clamp(min=0).numpy())             # 767, desc, shift EXCLUDED
    evF3, _ = torch.linalg.eigh(G3); sv_d3_full = np.sqrt(evF3.flip(0).clamp(min=0).numpy())
    ev0, evec0 = torch.linalg.eigh(G0)
    sv_d0 = np.sqrt(ev0.flip(0).clamp(min=0).numpy()); V0 = evec0.flip(1)   # 3072, desc
    # direct dataset JVP of the LN-shift coordinate (eigh floor exceeds the T(i) cut; pilot route)
    shift_unit = (ones768/np.sqrt(768.)).to(DEV)
    acc = 0.0
    for i in range(0, N, 1000):
        sl = torch.arange(i, min(i+1000, N)).to(DEV) if DEV.type == "cuda" else torch.arange(i, min(i+1000, N))
        VJ = VJ_batch(sl, True)
        acc += float((torch.einsum("bcx,x->bc", VJ, shift_unit)**2).sum())
    sigma_shift_direct = float(np.sqrt(acc))
    print(f"[{time.time()-t_start:7.1f}s] downstream Grams + eigh done; sigma_shift={sigma_shift_direct:.3e}",
          flush=True)

    # ---------------- rank statements T(i) + reproduction gates ------------------------------------
    H_units = "input-activation RMS per unit right-factor direction (feature units; N examples stacked)"
    rk_Hu = rank_T1(sv_u, "stacked augmented mlp.0 input [u;1] (10000x769), direct SVD", H_units)
    rk_Ha = rank_T1(sv_a, "stacked augmented mlp.3 input [a;1] (10000x3073), direct SVD", H_units)
    rk_Hf = rank_T1(sv_f, "stacked augmented head input [f;1] (10000x769), direct SVD", H_units)
    eigh_floor0 = float(np.sqrt(np.finfo(np.float64).eps))*float(sv_d0[0])
    eigh_floor3 = float(np.sqrt(np.finfo(np.float64).eps))*float(sv_d3_full[0])
    C_units = "logit RMS per unit preactivation direction (activation-to-output rate; N examples stacked)"
    rk_C0 = rank_T1(sv_d0, "stacked downstream derivative D_i of z0 through GELU->mlp.3->+res->LN->head "
                           "(Gram sum D^T D, 3072x3072, eigh)", C_units, gram_floor_sigma=eigh_floor0)
    # mlp.3 C: analytic LN-shift candidate measured DIRECTLY; nonshift complement from the projected eigh
    nonshift_min = float(sv_d3[-1]); cut3 = RANK_REL*float(sv_d3[0])
    shift_is_null = bool(sigma_shift_direct <= cut3)
    nonshift_above = bool(nonshift_min > GAP10*max(cut3, 0.0))
    rk_C3 = {"operator": "stacked downstream derivative D_i of z3 through +res->final LN->head; "
                         "LN-shift coordinate measured by DIRECT dataset JVP (eigh noise floor "
                         "exceeds the 1e-9*sigma0 cut), nonshift complement from the shift-projected "
                         "Gram eigh (767 coords)",
             "units": C_units, "sigma0": float(sv_d3[0]), "cut_abs": cut3, "cut_rel": RANK_REL,
             "sigma_shift_direct": sigma_shift_direct,
             "sigma_shift_over_sigma0": sigma_shift_direct/float(sv_d3[0]),
             "eigh_noise_floor_sigma": eigh_floor3, "smallest_nonshift_sigma": nonshift_min,
             "gap": nonshift_min/max(sigma_shift_direct, 1e-300),
             "gap_resolved_ge_10x": bool(shift_is_null and nonshift_above
                                         and nonshift_min/max(sigma_shift_direct, 1e-300) >= GAP10),
             "rank": 767 if shift_is_null else 768, "null_dim": 1 if shift_is_null else 0,
             "ambiguity_band_sigmas_within_10x_of_cut":
                 [float(x) for x in sv_d3[(sv_d3 > cut3/GAP10) & (sv_d3 < cut3*GAP10)]]}
    rk_Cf = {"operator": "head downstream derivative D = I_45 (readout preactivation IS the output); "
                         "stacked operator has every sigma/sqrt(N) = 1 structurally",
             "units": C_units, "structural": True, "rank": 45, "null_dim": 0,
             "sigma_over_sqrtN_all": 1.0}

    gate("exact_ranks_vs_head_multilayer",
         rk_Hu["rank"] == hm_ref["r_hu"][0] and rk_Ha["rank"] == hm_ref["r_ha"][0]
         and rk_Hf["rank"] == hm_ref["r_hf"][0]
         and 3072*(769-rk_Hu["rank"]) == hm_ref["mlp0_feature_null"]
         and 768*(3073-rk_Ha["rank"]) == hm_ref["mlp3_feature_null"]
         and d_out*(769-rk_Hf["rank"]) == hm_ref["readout_feature_null"]
         and (3073 if shift_is_null else 0) == hm_ref["mlp3_LNshift_downstream_null_genuine"],
         {"r_hu": rk_Hu["rank"], "r_ha": rk_Ha["rank"], "r_hf": rk_Hf["rank"],
          "shift_null": shift_is_null})
    gate("rank_gaps_resolved",
         rk_Hu["gap_resolved_ge_10x"] and rk_Ha["gap_resolved_ge_10x"] and rk_Hf["gap_resolved_ge_10x"]
         and rk_C3["gap_resolved_ge_10x"],
         {"gap_Hu": rk_Hu["gap"], "gap_Ha": rk_Ha["gap"], "gap_Hf": rk_Hf["gap"], "gap_C3": rk_C3["gap"],
          "C0_resolved": rk_C0["gap_resolved_ge_10x"], "gap_C0": rk_C0["gap"]})
    # bespoke tail-count reproduction (phase0f conventions use strict '<'; counts must match exactly)
    rep_counts = {"mlp3_dtail_sig1e2": int((sv_d3 < 1e-2*sv_d3[0]).sum()),
                  "mlp0_dtail_sig1e2": int((sv_d0 < 1e-2*sv_d0[0]).sum()),
                  "mlp3_ftail_sig1e3": int((sv_a < 1e-3*sv_a[0]).sum())-int((sv_a < 1e-9*sv_a[0]).sum()),
                  "mlp0_ftail_sig1e2": int((sv_u < 1e-2*sv_u[0]).sum())-int((sv_u < 1e-9*sv_u[0]).sum())}
    gate("tail_counts_vs_phase0f", all(rep_counts[k] == p0f_ref[k] for k in rep_counts),
         {k: (rep_counts[k], p0f_ref[k]) for k in rep_counts})
    if p0d_ref is not None:
        sp = p0d_ref["spectra_dual_convention"]; ok, det = True, {}
        for key, sv in (("feature_mlp0_u", sv_u), ("feature_mlp3_a", sv_a),
                        ("feature_readout_f", sv_f), ("downstream_mlp3_nonshift", sv_d3)):
            for fld, val in (("sigma1", float(sv[0])), ("sigma_min_over_sigma1", float(sv[-1]/sv[0]))):
                ref = sp[key][fld]
                if fld == "sigma_min_over_sigma1" and ref < 1e-12:
                    # exact-null sigma at the SVD noise floor: different LAPACK routes (svdvals vs
                    # svd-with-vectors) differ freely at this level; both must be numerical zeros
                    o = val < 1e-12
                else:
                    o = abs(val-ref) <= 1e-6*max(abs(ref), 1e-12)
                ok &= o; det[f"{key}.{fld}"] = (ref, val, bool(o))
        o = abs(logits_rms-p0d_ref["logits_rms_absolute"]) <= 1e-9*p0d_ref["logits_rms_absolute"]
        ok &= o; det["logits_rms"] = (p0d_ref["logits_rms_absolute"], logits_rms, bool(o))
        gate("spectrum_reproduction_vs_phase0d_s0", ok, {k: v for k, v in det.items() if not v[2]} or "all match")
    else:
        GATE_RESULTS["spectrum_reproduction_vs_phase0d_s0"] = {
            "pass": None, "detail": "not applicable (phase0d exists for seed 0 only)"}

    # ---------------- family table ------------------------------------------------------------------
    nu_u = VhU[-1].clone(); nu_f = VhF[-1].clone()          # exact feature-null right-vectors
    def abs_rates(sv): return np.asarray(sv, float)/sqrtN

    def feat_tail_indices(sv, rk):
        """indices (descending order) with sigma/sqrt(N) <= RATE_CUT, excluding T(i)-exact nulls."""
        rates = abs_rates(sv); null_from = rk["rank"]
        return [int(j) for j in range(len(sv)) if rates[j] <= RATE_CUT and j < null_from]

    layers = {}; batteries_todo = []   # (tag, layer, side, basis CPU (dim x k), proj, family_dim)

    def proj_mlp0(dA): return dA-torch.outer(dA@nu_u, nu_u)
    def proj_mlp3(dA): return dA-torch.outer(ones768, dA.mean(0))
    def proj_head(dA): return dA-torch.outer(dA@nu_f, nu_f)

    # ---- mlp.0 (n=3072, m=769) ----
    kN0 = 769-rk_Hu["rank"]
    fn0 = {"definition": "FN = {Z U_Hperp^T}, U_Hperp = exact numerical kernel of [u;1]",
           "status": "computed_nonempty" if (kN0 > 0 and rk_Hu["gap_resolved_ge_10x"]) else "computed_empty_genuine_zero",
           "k_factor": kN0, "dim_param_space": 3072*kN0,
           "basis": "arrays: nu_u (769), family = {Z nu_u^T}",
           "null_quality_max_residual": float((Pu@nu_u).abs().max()),
           "battery": "mlp0_fn_exact" if kN0 > 0 else None}
    if kN0 > 0: batteries_todo.append(("mlp0_fn_exact", "mlp0", "right", nu_u.view(-1, 1), None, 3072*kN0))
    t0i = feat_tail_indices(sv_u, rk_Hu)
    fa0 = {"definition": "feature tail: right singular directions of [u;1] with ABSOLUTE rate "
                         "sigma/sqrt(N) <= 1e-3 (feature units), exact-null indices excluded",
           "cut": RATE_CUT, "k": len(t0i),
           "tail_rates": [float(abs_rates(sv_u)[j]) for j in t0i],
           "boundary_min_rate_present_excl_null": float(abs_rates(sv_u)[rk_Hu["rank"]-1]),
           "status": "computed_nonempty" if t0i else "computed_empty_genuine_zero",
           "battery": "mlp0_fn_approx" if t0i else None}
    if t0i:
        B = VhU[t0i].T.clone(); batteries_todo.append(("mlp0_fn_approx", "mlp0", "right", B, proj_mlp0, 3072*len(t0i)))
    dn0_null = rk_C0["null_dim"]
    dx0 = {"definition": "DN_additional = {U_Cperp Z U_H^T}; C from the stacked downstream derivative of z0",
           "status": ("computed_nonempty" if (dn0_null > 0 and rk_C0["gap_resolved_ge_10x"]) else
                      ("computed_empty_genuine_zero" if rk_C0["gap_resolved_ge_10x"] else "unresolved")),
           "k_left_factor": dn0_null, "dim_param_space": dn0_null*rk_Hu["rank"],
           "boundary_min_rate": float(abs_rates(sv_d0)[-1]), "rank_statement": "operator_C",
           "battery": "mlp0_dn_exact" if (dn0_null > 0 and rk_C0["gap_resolved_ge_10x"]) else None}
    if dx0["battery"]:
        batteries_todo.append(("mlp0_dn_exact", "mlp0", "left", V0[:, 3072-dn0_null:].clone(), None,
                               dn0_null*rk_Hu["rank"]))
    d0rates = abs_rates(sv_d0); k_d0 = int((d0rates <= RATE_CUT).sum()) - dn0_null
    da0 = {"definition": "downstream tail: left factors U from the smallest sigma of the stacked "
                         "downstream operator at ABSOLUTE rate sigma/sqrt(N) <= 1e-3 (logit RMS per unit "
                         "z0-activation direction); edits U Z with Z free over R^769",
           "cut": RATE_CUT, "k": int(k_d0), "boundary_min_rate": float(d0rates[-1]),
           "tail_rate_range": ([float(d0rates[3072-k_d0-dn0_null]), float(d0rates[3071-dn0_null])] if k_d0 > 0 else None),
           "status": "computed_nonempty" if k_d0 > 0 else "computed_empty_genuine_zero",
           "battery": "mlp0_dn_approx" if k_d0 > 0 else None}
    if k_d0 > 0:
        T0 = V0[:, 3072-k_d0-dn0_null:3072-dn0_null].clone() if dn0_null else V0[:, 3072-k_d0:].clone()
        batteries_todo.append(("mlp0_dn_approx", "mlp0", "left", T0, proj_mlp0, k_d0*769))
    # SEL mlp.0: S_i = diag(gelu'(z0_i)); census + exact statement + approximate axis-aligned screen
    gp_abs = gp.abs(); unit_max_gp = gp_abs.max(0).values; unit_min_gp = gp_abs.min(0).values
    n_exact_zero_entries = int((gp == 0.0).sum()); n_dead_units_fp64 = int((unit_max_gp == 0.0).sum())
    diagG0_rate = np.sqrt(np.clip(np.diag(G0.numpy()), 0, None))/sqrtN
    sel0_cand = [int(c) for c in np.where(diagG0_rate <= RATE_CUT)[0]]
    sel0 = {"constraint": "S_i A p_i = 0 with S_i = diag(gelu'(z0_i)) (precise operator; the analytic "
                          "GELU derivative vanishes only at the isolated point z* ~ -0.75179)",
            "census": {"min_abs_gelu_deriv_over_all_entries": float(gp_abs.min()),
                       "fp64_underflow_zero_entries": n_exact_zero_entries,
                       "units_with_all_entries_fp64_zero": n_dead_units_fp64,
                       "min_over_units_of_max_abs_deriv": float(unit_max_gp.min()),
                       "median_over_units_of_max_abs_deriv": float(unit_max_gp.median()),
                       "NOTE": "census is a candidate screen, not a selective-space computation; "
                               "fp64-underflowed entries are numerical, NOT exact dead units"},
            "exact": {"status": "structurally_coincides_with_FN",
                      "statement": "no unit has gelu'(z0)=0 at every example analytically, so the exact "
                                   "selective space is {A : A p_i = 0 for all i} = FN; no separate probes",
                      "evidence_min_over_units_max_abs_deriv": float(unit_max_gp.min())},
            "approx": {"definition": "axis-aligned candidate rows: units c with downstream rate "
                                     "sqrt((sum_i ||D0_i e_c||^2))/sqrt(N) <= 1e-3 (same units as DN tail); "
                                     "family = span{e_c} (x) R^769",
                       "k_units": len(sel0_cand), "candidate_units": sel0_cand,
                       "min_unit_rate": float(diagG0_rate.min()),
                       "status": "computed_nonempty" if sel0_cand else "computed_empty_genuine_zero",
                       "battery": "mlp0_sel_approx" if sel0_cand else None}}
    if sel0_cand:
        E0 = torch.zeros(3072, len(sel0_cand), dtype=torch.float64)
        for i, c in enumerate(sel0_cand): E0[c, i] = 1.0
        batteries_todo.append(("mlp0_sel_approx", "mlp0", "left", E0, proj_mlp0, len(sel0_cand)*769))
    layers["mlp0"] = {"shape_aug": [3072, 769], "params": 3072*769, "has_bias": True,
                      "operator_H": rk_Hu, "operator_C": rk_C0,
                      "families": {"fn_exact": fn0, "fn_approx": fa0, "dn_exact": dx0,
                                   "dn_approx": da0, "sel": sel0,
                                   "inv_note": "no layer-local exact invariance beyond FN; the LN-shift "
                                               "mechanism belongs to mlp.3's downstream family"},
                      "N_marginal": {"FN": 3072*kN0, "DN_additional": dn0_null*rk_Hu["rank"],
                                     "N_total_dedup": 3072*kN0+dn0_null*rk_Hu["rank"]}}

    # ---- mlp.3 (n=768, m=3073) ----
    kN3 = 3073-rk_Ha["rank"]
    fn3 = {"definition": "FN = {Z U_Hperp^T} on [a;1]",
           "status": "computed_nonempty" if kN3 > 0 else "computed_empty_genuine_zero",
           "k_factor": kN3, "dim_param_space": 768*kN3,
           "boundary_min_rate": float(abs_rates(sv_a)[-1]),
           "gap_evidence": {"sigma_min_over_sigma0": float(sv_a[-1]/sv_a[0]),
                            "vs_cut": RANK_REL, "gap_x_cut": rk_Ha["gap"]},
           "battery": "mlp3_fn_exact" if kN3 > 0 else None}
    if kN3 > 0: batteries_todo.append(("mlp3_fn_exact", "mlp3", "right", VhA[-kN3:].T.clone(), None, 768*kN3))
    # DN-X: LN-shift reused analytically, revalidated (not recomputed from an eigenbasis)
    dx3 = {"definition": "DN_additional = {U_Cperp Z U_H^T} = {ones_768 y^T : y in R^3073} (U_H = I, "
                         "r_h = 3073); the validated LN-shift mechanism, reused and revalidated",
           "status": "computed_nonempty" if shift_is_null else "unresolved",
           "k_left_factor": 1, "dim_param_space": 3073,
           "revalidation": {"sigma_shift_direct_dataset_JVP": sigma_shift_direct,
                            "sigma_shift_over_sigma0": sigma_shift_direct/float(sv_d3[0]),
                            "nonlinear_r30_check": "gates.exact_null_machine_precision"},
           "battery": "mlp3_dn_exact"}
    batteries_todo.append(("mlp3_dn_exact", "mlp3", "left", (ones768/np.sqrt(768.)).view(-1, 1), None, 3073))
    t3i = feat_tail_indices(sv_a, rk_Ha)
    fa3 = {"definition": "feature tail of [a;1] at ABSOLUTE rate sigma/sqrt(N) <= 1e-3 (feature units)",
           "cut": RATE_CUT, "k": len(t3i),
           "tail_rates": [float(abs_rates(sv_a)[j]) for j in t3i],
           "boundary_min_rate_present": float(abs_rates(sv_a)[-1]) if not t3i else None,
           "status": "computed_nonempty" if t3i else "computed_empty_genuine_zero",
           "battery": "mlp3_fn_approx" if t3i else None}
    if t3i:
        batteries_todo.append(("mlp3_fn_approx", "mlp3", "right", VhA[t3i].T.clone(), proj_mlp3, 768*len(t3i)))
    d3rates = abs_rates(sv_d3); k_d3 = int((d3rates <= RATE_CUT).sum())
    da3 = {"definition": "nonshift downstream tail of the stacked z3 derivative at ABSOLUTE rate "
                         "sigma/sqrt(N) <= 1e-3 (logit RMS per unit z3-activation direction); the exact "
                         "LN-shift coordinate is excluded as the separately validated exact mechanism",
           "cut": RATE_CUT, "k": int(k_d3), "boundary_min_rate": float(d3rates[-1]),
           "status": "computed_nonempty" if k_d3 > 0 else "computed_empty_genuine_zero",
           "NOTE": "genuine zero at the ABSOLUTE cut (reviewer SS1); the pilot's relative-selection "
                   "k=145 remains supplementary under its own stated convention",
           "battery": "mlp3_dn_approx" if k_d3 > 0 else None}
    if k_d3 > 0:
        T3 = (Qsh@evec3.flip(1))[:, 767-k_d3:].clone()
        batteries_todo.append(("mlp3_dn_approx", "mlp3", "left", T3, proj_mlp3, k_d3*3073))
    a_abs = a.abs(); col_rate = torch.sqrt((a**2).mean(0)).numpy()    # per-unit feature RMS (3072)
    n_dead_cols = int((a_abs.max(0).values == 0.0).sum())
    sel3_cand = [int(j) for j in np.where(col_rate <= RATE_CUT)[0]]
    sel3 = {"constraint": "selective feature constraint: columns of A fed by low-activity GELU units; "
                          "an exactly-dead unit (a_j = 0 for all examples) would put e_j in the feature "
                          "null; nonzero GELU outputs are never rounded to exact zeros",
            "census": {"min_over_units_of_max_abs_activation": float(a_abs.max(0).values.min()),
                       "units_all_examples_fp64_zero": n_dead_cols,
                       "min_unit_rms_rate": float(col_rate.min()),
                       "n_units_rms_rate_le_1e-3": len(sel3_cand)},
            "exact": {"status": "computed_empty_genuine_zero",
                      "statement": "no exactly dead unit (census above) and FN itself is empty "
                                   "(r_h = 3073); exact selective space = {0}",
                      "boundary": float(a_abs.max(0).values.min())},
            "approx": {"definition": "axis-aligned candidate columns: units j with feature rate "
                                     "||a_j||_rms <= 1e-3 (feature units; bias coordinate not a GELU "
                                     "unit); family = R^768 (x) span{e_j}",
                       "k_units": len(sel3_cand), "candidate_units": sel3_cand,
                       "status": "computed_nonempty" if sel3_cand else "computed_empty_genuine_zero",
                       "battery": "mlp3_sel_approx" if sel3_cand else None}}
    if sel3_cand:
        E3 = torch.zeros(3073, len(sel3_cand), dtype=torch.float64)
        for i, c in enumerate(sel3_cand): E3[c, i] = 1.0
        batteries_todo.append(("mlp3_sel_approx", "mlp3", "right", E3, proj_mlp3, 768*len(sel3_cand)))
    layers["mlp3"] = {"shape_aug": [768, 3073], "params": 768*3073, "has_bias": True,
                      "operator_H": rk_Ha, "operator_C": rk_C3,
                      "families": {"fn_exact": fn3, "fn_approx": fa3, "dn_exact": dx3,
                                   "dn_approx": da3, "sel": sel3,
                                   "inv_note": "final-LN shift removal IS dn_exact here (certified "
                                               "mechanism, revalidated)"},
                      "N_marginal": {"FN": 768*kN3, "DN_additional": 3073 if shift_is_null else 0,
                                     "N_total_dedup": 768*kN3+(3073 if shift_is_null else 0)}}

    # ---- head (n=45, m=769) ----
    kNf = 769-rk_Hf["rank"]
    fnh = {"definition": "FN = {Z U_Hperp^T} on [f;1]",
           "status": "computed_nonempty" if (kNf > 0 and rk_Hf["gap_resolved_ge_10x"]) else "computed_empty_genuine_zero",
           "k_factor": kNf, "dim_param_space": d_out*kNf,
           "basis": "arrays: nu_f (769), family = {Z nu_f^T}",
           "null_quality_max_residual": float((Pf@nu_f).abs().max()),
           "battery": "head_fn_exact" if kNf > 0 else None}
    if kNf > 0: batteries_todo.append(("head_fn_exact", "head", "right", nu_f.view(-1, 1), None, d_out*kNf))
    tfi = feat_tail_indices(sv_f, rk_Hf)
    fah = {"definition": "feature tail of [f;1] at ABSOLUTE rate sigma/sqrt(N) <= 1e-3 (feature units), "
                         "exact-null indices excluded",
           "cut": RATE_CUT, "k": len(tfi),
           "tail_rates": [float(abs_rates(sv_f)[j]) for j in tfi],
           "boundary_min_rate_present_excl_null": float(abs_rates(sv_f)[rk_Hf["rank"]-1]),
           "status": "computed_nonempty" if tfi else "computed_empty_genuine_zero",
           "battery": "head_fn_approx" if tfi else None}
    if tfi:
        batteries_todo.append(("head_fn_approx", "head", "right", VhF[tfi].T.clone(), proj_head, d_out*len(tfi)))
    dxh = {"definition": "DN_additional at a readout: D = I so C = R^45, C_perp = {0}",
           "status": "structurally_empty", "k_left_factor": 0, "dim_param_space": 0,
           "boundary_rate": 1.0,
           "statement": "structural: N = FN at the readout (spec SS2); no probes for a structurally empty cell"}
    dah = {"definition": "downstream activation tail at a readout", "status": "structurally_empty",
           "k": 0, "boundary_rate": 1.0,
           "statement": "every stacked-D singular rate equals 1 structurally (D = I); no tail at 1e-3"}
    selh = {"constraint": "none: linear layer, no elementwise gates",
            "exact": {"status": "structurally_empty"}, "approx": {"status": "structurally_empty"},
            "census": None}
    layers["head"] = {"shape_aug": [d_out, 769], "params": d_out*769, "has_bias": True,
                      "operator_H": rk_Hf, "operator_C": rk_Cf,
                      "families": {"fn_exact": fnh, "fn_approx": fah, "dn_exact": dxh,
                                   "dn_approx": dah, "sel": selh,
                                   "inv_note": "head linearity: actual == frozen == first_order "
                                               "structurally for head-only edits"},
                      "N_marginal": {"FN": d_out*kNf, "DN_additional": 0, "N_total_dedup": d_out*kNf}}

    for L, nm in (("mlp0", (3072, 769)), ("mlp3", (768, 3073)), ("head", (d_out, 769))):
        p_l = nm[0]*nm[1]; sd = N*d_out
        exact_marg = layers[L]["N_marginal"]["N_total_dedup"]
        layers[L]["counting_bounds_not_computed_ranks"] = {
            "atomic_floor_max0_p_minus_sd": int(max(0, p_l-sd)),
            "additional_deficit_floor_max0": int(max(0, p_l-sd-exact_marg)),
            "NOTE": "uniform counting lower bounds on ker(E) of the full atomic map; "
                    "NOT computed kernel dimensions and NOT marginal-family dimensions"}

    # SEL/DN overlap diagnostics (candidate screens vs computed tails)
    if sel0_cand and k_d0 > 0:
        T0m = T0  # noqa: F821
        layers["mlp0"]["families"]["sel"]["approx"]["overlap_proj_mass_onto_dn_tail"] = \
            [float((T0m.T[:, c]**2).sum()) for c in sel0_cand]
    if sel3_cand and t3i:
        TaB = VhA[t3i].T
        layers["mlp3"]["families"]["sel"]["approx"]["overlap_proj_mass_onto_fn_tail"] = \
            [float((TaB[c]**2).sum()) for c in sel3_cand]

    print(f"[{time.time()-t_start:7.1f}s] family table built; batteries: "
          f"{[(t[0], t[3].shape[1]) for t in batteries_todo]}", flush=True)

    # ---------------- DEV copies for the battery ----------------------------------------------------
    u_d, a_d, z0_d = u.to(DEV), a.to(DEV), z0.to(DEV)
    cls_d, bo_d, f_d, logits_d = cls_mid.to(DEV), bo.to(DEV), f.to(DEV), logits.to(DEV)
    b3_d, bV_d = b3.to(DEV), bV_c.to(DEV)
    W3T_d = W3.T.contiguous().to(DEV); VT_d = V_c.T.contiguous().to(DEV)
    base_pern = logits_d.norm(dim=1)+1e-30

    def apply_JLN(delta):
        c = delta-delta.mean(-1, keepdim=True)-(((rhat_d*delta).sum(-1, keepdim=True))/768)*rhat_d
        return (gf_d/s_bo_d)*c

    def met(dd):
        pern = dd.norm(dim=1)
        ab = float(torch.sqrt((pern**2).mean()))
        return (ab, ab/logits_rms, float((pern/base_pern).median()))

    def eval_mlp0(dA, capture):
        dz0 = u_d@dA[:, :-1].T+dA[:, -1]
        d3lin = (gp_d*dz0)@W3T_d; jln = apply_JLN(d3lin); fo = jln@VT_d
        jf = float(fo.norm())
        C = np.zeros((5, 3, len(RGRID))); S = np.zeros((2, len(RGRID))); caps = {}
        for ri, t in enumerate(RGRID):
            an = gelu(z0_d+t*dz0); bon = cls_d+an@W3T_d+b3_d
            d_act = ln_fwd(bon, gf_d, bf_d, epsf)@VT_d+bV_d-logits_d
            d_fro = (f_d+t*jln)@VT_d+bV_d-logits_d           # frozen GELU+LN forward (own float route)
            d_fo = t*fo
            for ci, dd in enumerate((d_act, d_fro, d_fo, d_act-d_fro, d_fro-d_fo)):
                C[ci, :, ri] = met(dd)
            S[0, ri] = met(ln_fwd(bo_d+t*d3lin, gf_d, bf_d, epsf)@VT_d+bV_d-logits_d)[1]  # frozenGELU only
            S[1, ri] = met(apply_JLN(bon-bo_d)@VT_d)[1]                                   # frozenLN only
            if capture and ri in VEC_RIDX:
                caps[ri] = torch.stack([d_act[:NVAL], d_fro[:NVAL], d_fo[:NVAL]]).cpu()
        return jf, C, S, caps

    def eval_mlp3(dA, capture):
        dz3 = a_d@dA[:, :-1].T+dA[:, -1]
        jln = apply_JLN(dz3); fo = jln@VT_d; jf = float(fo.norm())
        C = np.zeros((5, 3, len(RGRID))); caps = {}
        for ri, t in enumerate(RGRID):
            d_act = ln_fwd(bo_d+t*dz3, gf_d, bf_d, epsf)@VT_d+bV_d-logits_d
            d_fro = (f_d+t*jln)@VT_d+bV_d-logits_d
            d_fo = t*fo
            for ci, dd in enumerate((d_act, d_fro, d_fo, d_act-d_fro, d_fro-d_fo)):
                C[ci, :, ri] = met(dd)
            if capture and ri in VEC_RIDX:
                caps[ri] = torch.stack([d_act[:NVAL], d_fro[:NVAL], d_fo[:NVAL]]).cpu()
        return jf, C, None, caps

    V_full_d = V_c.to(DEV)
    def eval_head(dA, capture):
        fo = f_d@dA[:, :-1].T+dA[:, -1]; jf = float(fo.norm())
        C = np.zeros((5, 3, len(RGRID))); caps = {}
        for ri, t in enumerate(RGRID):
            d_act = f_d@(V_full_d+t*dA[:, :-1]).T+(bV_d+t*dA[:, -1])-logits_d
            d_fro = f_d@VT_d+bV_d+t*fo-logits_d
            d_fo = t*fo
            for ci, dd in enumerate((d_act, d_fro, d_fo, d_act-d_fro, d_fro-d_fo)):
                C[ci, :, ri] = met(dd)
            if capture and ri in VEC_RIDX:
                caps[ri] = torch.stack([d_act[:NVAL], d_fro[:NVAL], d_fo[:NVAL]]).cpu()
        return jf, C, None, caps

    EVAL = {"mlp0": eval_mlp0, "mlp3": eval_mlp3, "head": eval_head}
    LAY_NM = {"mlp0": (3072, 769), "mlp3": (768, 3073), "head": (d_out, 769)}

    def limiter(layer, C, S):
        la = bracket(C[0, 1])["last_safe"]; lb = bracket(C[2, 1])["last_safe"]
        if layer == "mlp0":
            lg = bracket(S[0])["last_safe"]; ll = bracket(S[1])["last_safe"]
            if lb <= la: return "none"
            if (lg-la) >= (ll-la) and (lg-la) > 0: return "GELU"
            if (ll-la) > 0: return "LN"
            return "GELU+LN"
        if layer == "mlp3": return "LN" if lb > la else "none"
        return "none"

    def make_raw(side, n, m, basis, mode, rng, k):
        if mode == "generic": return torch.randn(n, m, generator=rng, dtype=torch.float64)
        if side == "left":
            U = basis if mode == "family" else torch.linalg.qr(
                torch.randn(n, k, generator=rng, dtype=torch.float64))[0]
            return U@torch.randn(k, m, generator=rng, dtype=torch.float64)
        B = basis if mode == "family" else torch.linalg.qr(
            torch.randn(m, k, generator=rng, dtype=torch.float64))[0]
        return torch.randn(n, k, generator=rng, dtype=torch.float64)@B.T

    # basis orthonormality gate across all battery bases
    worst = 0.0
    for tag, layer, side, basis, proj, fdim in batteries_todo:
        g = basis.T@basis-torch.eye(basis.shape[1], dtype=torch.float64)
        worst = max(worst, float(g.abs().max()))
    gate("battery_basis_orthonormality", worst < 1e-10, {"max_abs_gram_err": worst})

    batteries = {}; arrays = {}
    vec_store = {L: {} for L in ("mlp0", "mlp3", "head")}
    LIMCATS = ("GELU", "LN", "GELU+LN", "none")
    for tag, layer, side, basis, proj, fdim in batteries_todo:
        n, m = LAY_NM[layer]; k = basis.shape[1]; ev = EVAL[layer]
        brec = {"layer": layer, "factor_side": side, "k_factor": k, "family_dim_param_space": fdim,
                "M": args.M,
                "exact_component_removal": ("layer exact nulls projected out of family AND controls "
                                            "identically" if proj is not None else
                                            "none: exact family battery (the family IS the exact "
                                            "component); controls are unprojected dim-matched subspaces"),
                "modes": {}}
        for mode in ("family", "randsub", "generic"):
            seed = stable_seed(SEED_BASE, f"common_{st}_{tag}", mode)
            rng = torch.Generator().manual_seed(seed)
            drift = np.zeros((args.M, 5, 3, len(RGRID)))
            supp = np.zeros((args.M, 2, len(RGRID))) if layer == "mlp0" else None
            per = []; jv_abs = []; jv_rel = []; ls = []; fcs = []; cens = 0; lims = []
            for mi in range(args.M):
                raw = make_raw(side, n, m, basis, mode, rng, k)
                nrm_raw = float(raw.norm())
                if proj is not None: raw = proj(raw)
                nrm = float(raw.norm()); dA = (raw/nrm).to(DEV)
                jf, Cc, Ss, caps = ev(dA, mi < NVEC)
                drift[mi] = Cc
                if supp is not None: supp[mi] = Ss
                br = bracket(Cc[0, 1]); lim = limiter(layer, Cc, Ss)
                jv_abs.append(jf/sqrtN); jv_rel.append(jf/sqrtN/logits_rms)
                ls.append(br["last_safe"]); cens += int(br["censored"]); lims.append(lim)
                if br["first_cross"] is not None: fcs.append(br["first_cross"])
                per.append({"i": mi, "jv_abs_rate_X_per_unit_v": jv_abs[-1],
                            "jv_rel_rate": jv_rel[-1], "jv_frob": jf,
                            "norm_before_projection": nrm_raw, "norm_before_unit": nrm,
                            "removed_mass_fraction": float(np.sqrt(max(0.0, 1.0-(nrm/nrm_raw)**2))),
                            "bracket_rel_tol1e-3": br, "limiter": lim})
                if caps:
                    vec_store[layer][f"{tag}/{mode}"] = vec_store[layer].get(f"{tag}/{mode}", {})
                    for ri, tens in caps.items():
                        vec_store[layer][f"{tag}/{mode}"].setdefault(ri, []).append(tens)
            exceed = {f"r{r:g}": int((drift[:, 0, 1, ri] > TOL).sum()) for ri, r in enumerate(RGRID)}
            agg = {}
            for ci, cn in enumerate(CURVES):
                agg[cn] = {me: {f"r{r:g}": dist(drift[:, ci, mi_, ri]) for ri, r in enumerate(RGRID)}
                           for mi_, me in enumerate(METRICS)}
            brec["modes"][mode] = {
                "generator_seed": seed, "M": args.M,
                "jv_abs_rate_X_per_unit_v": dist(jv_abs), "jv_rel_rate": dist(jv_rel),
                "last_safe_tested_rel_tol1e-3": dist(ls),
                "first_cross_tested": (dist(fcs) if fcs else None), "censored_count": cens,
                "limiter_counts": {c: int(sum(x == c for x in lims)) for c in LIMCATS},
                "exceedance_count_rel_gt_1e-3_among_256": exceed,
                "aggregates_per_radius": agg, "per_direction": per}
            arrays[f"battery_{tag}_{mode}_drift"] = torch.tensor(drift)
            if supp is not None: arrays[f"battery_{tag}_{mode}_partialfreeze_rel"] = torch.tensor(supp)
            print(f"[{time.time()-t_start:7.1f}s] {tag:18s} {mode:8s} jv_abs_med="
                  f"{np.median(jv_abs):.3e} last_safe_med={np.median(ls):g} cens={cens} "
                  f"exceed@r30={exceed['r30']}", flush=True)
        batteries[tag] = brec

    # ---------------- exact-null machine-precision gate --------------------------------------------
    rngx = torch.Generator().manual_seed(stable_seed(SEED_BASE, f"common_{st}", "exactspot"))
    spot = []
    for tag, layer, side, basis, proj, fdim in batteries_todo:
        if not tag.endswith("_exact"): continue
        n, m = LAY_NM[layer]
        for i in range(3):
            raw = make_raw(side, n, m, basis, "family", rngx, basis.shape[1])
            dA = (raw/raw.norm()).to(DEV)
            jf, Cc, Ss, _ = EVAL[layer](dA, False)
            spot.append({"family": tag, "i": i, "jv_rel_rate": jf/sqrtN/logits_rms,
                         "actual_rel_drift_r30": float(Cc[0, 1, -1])})
    gate("exact_null_machine_precision",
         all(s["jv_rel_rate"] < 1e-10 and s["actual_rel_drift_r30"] < 1e-9 for s in spot), spot)

    # ---------------- JVP spot agreement (validated batched-fp64 JVP route) -------------------------
    nsub = 200; sub = slice(0, nsub)
    cls_s, z0_s, bo_s = cls_mid[sub].to(DEV), z0[sub].to(DEV), bo[sub].to(DEV)
    f_s, u_s, a_s, gp_s = f[sub].to(DEV), u[sub].to(DEV), a[sub].to(DEV), gp[sub].to(DEV)
    rhat_s, sbo_s = rhat_c[sub].to(DEV), s_bo_c[sub].to(DEV)
    def JLN_s(delta):
        c = delta-delta.mean(-1, keepdim=True)-(((rhat_s*delta).sum(-1, keepdim=True))/768)*rhat_s
        return (gf_d/sbo_s)*c
    rngj = torch.Generator().manual_seed(stable_seed(SEED_BASE, f"common_{st}", "jvpspot"))
    jvp_gates = {}
    for layer in ("mlp0", "mlp3", "head"):
        n, m = LAY_NM[layer]; errs = []
        for i in range(3):
            dA = torch.randn(n, m, generator=rngj, dtype=torch.float64)
            dA = (dA/dA.norm()).to(DEV)
            if layer == "mlp0":
                fn = lambda z: ln_fwd(cls_s+gelu(z)@W3T_d+b3_d, gf_d, bf_d, epsf)@VT_d+bV_d
                tang = u_s@dA[:, :-1].T+dA[:, -1]
                _, jv_auto = torch.autograd.functional.jvp(fn, z0_s, tang, strict=True)
                jv_ana = JLN_s((gp_s*tang)@W3T_d)@VT_d
            elif layer == "mlp3":
                fn = lambda x: ln_fwd(x, gf_d, bf_d, epsf)@VT_d+bV_d
                tang = a_s@dA[:, :-1].T+dA[:, -1]
                _, jv_auto = torch.autograd.functional.jvp(fn, bo_s, tang, strict=True)
                jv_ana = JLN_s(tang)@VT_d
            else:
                fn = lambda tt: f_s@(V_full_d+tt*dA[:, :-1]).T+(bV_d+tt*dA[:, -1])
                _, jv_auto = torch.autograd.functional.jvp(
                    fn, torch.zeros((), dtype=torch.float64, device=DEV),
                    torch.ones((), dtype=torch.float64, device=DEV), strict=True)
                jv_ana = f_s@dA[:, :-1].T+dA[:, -1]
            errs.append(float((jv_auto-jv_ana).norm()/jv_ana.norm()))
        jvp_gates[layer] = {"rel_errs": errs, "pass": bool(all(e < 1e-12 for e in errs))}
    gate("jvp_autograd_vs_analytic_batched_fp64", all(v["pass"] for v in jvp_gates.values()), jvp_gates)

    # frozen-vs-first-order structural check from the recorded battery arrays
    worst_ji = 0.0
    for key, t in arrays.items():
        if key.endswith("_drift"): worst_ji = max(worst_ji, float(t[:, 4, 1, :].max()))
    gate("frozen_minus_first_order_structural", worst_ji < 1e-8,
         {"max_rel_rms_over_all_batteries": worst_ji,
          "statement": "single-layer frozen response is linear in r and coincides with the JVP "
                       "structurally; residual is float-route noise only"})

    # ---------------- persist ------------------------------------------------------------------------
    arrays.update({
        "RGRID": torch.tensor(RGRID), "curve_axis": CURVES, "metric_axis": METRICS,
        "sv_feature_mlp0_u_aug": torch.tensor(sv_u), "sv_feature_mlp3_a_aug": torch.tensor(sv_a),
        "sv_feature_head_f_aug": torch.tensor(sv_f),
        "sv_downstream_mlp0_z0": torch.tensor(sv_d0),
        "sv_downstream_mlp3_z3_nonshift": torch.tensor(sv_d3),
        "sv_downstream_mlp3_z3_full_eigh": torch.tensor(sv_d3_full),
        "G0_downstream_gram_mlp0": G0, "G3_downstream_gram_mlp3": G3,
        "nu_u": nu_u, "nu_f": nu_f, "ones768_shift_unit": ones768/np.sqrt(768.),
        "gelu_deriv_unit_max": unit_max_gp, "gelu_deriv_unit_min": unit_min_gp,
        "mlp3_unit_feature_rms_rate": torch.tensor(col_rate),
        "mlp0_unit_downstream_rate_diagG0": torch.tensor(diagG0_rate),
    })
    for tag, layer, side, basis, proj, fdim in batteries_todo:
        arrays[f"basis_{tag}"] = basis
    vec_files = {}
    for L in vec_store:
        if not vec_store[L]: continue
        pack = {}
        for key, rads in vec_store[L].items():
            pack[key] = {"radius_indices": sorted(rads.keys()),
                         "radii": [RGRID[ri] for ri in sorted(rads.keys())],
                         "types": ["actual", "frozen", "first_order"], "n_val_examples": NVAL,
                         "n_probes": NVEC,
                         "data": torch.stack([torch.stack(rads[ri], 0) for ri in sorted(rads.keys())], 1)}
        vf = os.path.join(args.outdir, f"common_family_{st}_vectors_{L}.pt")
        torch.save(pack, vf+".tmp"); os.replace(vf+".tmp", vf); vec_files[L] = vf

    validation = {"label": "automatically checked (in-run gates; NOT independent review)",
                  "checks": GATE_RESULTS,
                  "passed": bool(all(v["pass"] for v in GATE_RESULTS.values() if v["pass"] is not None))}
    out = {
        "study": "COMMON ViT FAMILY STUDY (marginal + selective families, common protocol)",
        "spec_sources": ["memos/2026-11-17_reviewer_protocol_corrections.md",
                         "memos/2026-11-22_two_day_completion_plan.md",
                         "memos/2026-11-17_vit_resnet_standardization_spec.md"],
        "seed_tag": st, "N_examples": N, "M_probes_per_mode": args.M, "device": str(DEV),
        "dtype": "float64",
        "ckpt": ckpt, "ckpt_sha256": ckpt_sha, "cache_file": cache_file, "cache_sha256": cache_sha,
        "sha_authentication": sha_ref,
        "normalization_denominators": {"sqrt_N": sqrtN, "logits_rms_absolute": logits_rms,
                                       "per_probe": "per_direction norm_before_unit"},
        "conventions": {
            "rank_T1": f"sigma <= {RANK_REL} * sigma0 of the named operator, gap-10 resolution check; "
                       "ambiguity bands retained; eigh-route floors recorded",
            "tail_T2": f"ABSOLUTE singular rate sigma/sqrt(N) <= {RATE_CUT} in the factor operator's own "
                       "units (N = examples); never relabelled as parameter-Jacobian spectra",
            "radius_grid_absolute": RGRID, "edit_normalization": "unit Frobenius norm of the [W|b] block",
            "drift_metrics": ["abs_rms_X: ||Delta F||_X (absolute dataset-RMS, logit units)",
                              "rel_rms_X: ||Delta F||_X / ||F(theta0)||_X (separately named relative metric)",
                              "median_per_example_rel (supplementary)"],
            "visibility_bracket": "relative 1e-3 at TESTED radii (established comparison brackets only)",
            "paired_vectors": "actual / frozen / first_order plus VECTOR residuals actual-minus-frozen "
                              "and frozen-minus-first_order; never scalar-norm differences",
            "frozen": "frozen forward = GELU gates and LN Jacobians fixed at base (existing validated "
                      "route); for single-layer edits frozen is linear in r and structurally coincides "
                      "with the JVP; both computed by their own float routes",
            "probes": "256 per mode, generator-seeded (recorded), fixed before the sweep; dim-matched "
                      "random-subspace + generic controls; no probes for structurally empty cells",
            "statistics": "exceedance counts among 256; zero exceedances among 256 controls reported as "
                          "such or as the plus-one estimate 1/257; no p-values",
            "jv_rates": "per-probe realized-edit original-network parameter-to-output JVP rates: "
                        "absolute ||J v||_X / ||v||_F and the separately named relative version"},
        "layers": layers, "batteries": batteries,
        "exact_spot_checks": spot, "jvp_spot_gates": jvp_gates,
        "chunked_vector_files": vec_files,
        "arrays_file": out_arr,
        "validation": validation,
        "runtime_s": time.time()-t_start,
    }
    torch.save(arrays, out_arr+".tmp"); os.replace(out_arr+".tmp", out_arr)
    with open(out_json+".tmp", "w") as fo: json.dump(out, fo, indent=1)
    os.replace(out_json+".tmp", out_json)
    print(json.dumps({L: {fk: (fv.get("status") if isinstance(fv, dict) and "status" in fv else
                               {sk: sv.get("status") for sk, sv in fv.items()
                                if isinstance(sv, dict) and "status" in sv} if isinstance(fv, dict) else fv)
                          for fk, fv in layers[L]["families"].items() if fk != "inv_note"}
                      for L in layers}, indent=1))
    print("validation passed:", validation["passed"])
    print("wrote", out_json, "\nDONE", flush=True)

if __name__ == "__main__":
    main()
