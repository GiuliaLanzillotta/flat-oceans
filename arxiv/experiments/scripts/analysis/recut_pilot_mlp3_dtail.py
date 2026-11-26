"""PILOT re-cut (2026-11-17 standardization spec, ViT work package item 4, ONE cell):
seed 0, block-11 mlp.3, approximate DOWNSTREAM-tail family, re-cut from the bespoke sigma-count
convention (phase0f `mlp3_dtail_sig1e2`, sigma_j < 1e-2*sigma_1 within the nonshift block, k=489)
to the COMMON cut T(ii): first-order rate sigma/sqrt(s) <= 1e-3 in ||.||_X units.

||.||_X convention (declared): the spec's Normalization row defines the output metric as dataset RMS
RELATIVE to ||F(theta0)||_X, so the PRIMARY common cut used here is
    (sigma/sqrt(s)) / logits_rms <= 1e-3        (sigma from the nonshift downstream GN spectrum).
The ABSOLUTE-logit-units reading sigma/sqrt(s) <= 1e-3 is ALSO evaluated and reported: the stored
spectrum summary (phase0d_s0.json) already shows it is EMPTY (sigma_min/sqrt(s) = 7.4e-3 > 1e-3).
Both dims are recorded; the convention sensitivity is a pilot finding for the PI.

Operator: the same downstream Gauss-Newton operator G3 the bespoke record used (z3 -> +residual ->
final LN -> head), accumulated with the phase0f code (VJ_batch, chunk 500). The stored record
persisted only summary sigma quantities (phase0d_s0.json spectra_dual_convention), NOT the full
spectrum, so the operator is RECOMPUTED here and validated against every stored quantity.

Battery (common protocol section 2): M=256 probes per mode (tail / dim-matched randsub / generic),
fresh stable seeds recorded, exact null (LN-shift) projected out of family AND controls (phase0f
convention), unit-Frobenius [W|b] normalization, common 10-point radius grid
{1e-3,3e-3,0.01,0.03,0.1,0.3,1,3,10,30}, PRIMARY dataset-RMS relative drift + SECONDARY median
per-example relative drift, paired actual / frozen(LN) / first_order with per-example VECTOR
residuals (transition = ||F_act - F_frozen||; joint_interaction = ||F_frozen - F_first_order||,
which STRUCTURALLY COINCIDES with 0 for mlp.3: the only downstream nonlinearity is the final LN and
the head is linear, so the existing validated frozen-LN forward IS the first-order response).
Cumulative tested brackets at tol 1e-3, censoring counts, exceedance counts among 256 (no p-values).

Gates: (1) spectrum reproduction vs phase0d_s0.json stored sigma summaries; (2) basis/probe
orthonormality + exact-null projection residuals; (3) bit-level reproduction of the first 3 stored
per-direction records of the bespoke family (same generator seeds, k=489 basis, 8-point grid);
(4) exact LN-shift null validated at machine precision (spot JVPs + full nonlinear drift at r=30)
and autograd-JVP spot check of the analytic LN Jacobian.

Cross-check: re-cut tail drift vs the bespoke k=489 record at the 8 shared radii (consistency, not
equality: the re-cut tail span is strictly contained in the bespoke tail span).

Writes results/runs/ftvit_resisc45/recut_pilot_mlp3_dtail_s0.json (+ _arrays.pt). NEW files only."""
import os, sys, json, argparse, time, hashlib
import numpy as np, torch, torch.nn as nn
import torchvision.models as tvm

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from ftvit_paths import cache_path, stable_seed

DEV = torch.device(os.environ.get("VIT_DEV", "cuda") if torch.cuda.is_available() else "cpu")

# ---- phase0f conventions, verbatim (GPU-ported as in vit_head_multilayer.py) ----
def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0)))+z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r,g,b,eps):
    mu=r.mean(-1,keepdim=True); sd=torch.sqrt(r.var(-1,unbiased=False,keepdim=True)+eps); return g*(r-mu)/sd+b

RGRID10 = [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]   # common grid (spec section 2, row R)
RGRID8  = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]               # bespoke phase0f grid (repro gate only)
TOL = 1e-3

# stored quantities to reproduce (gate 1): phase0d_s0.json spectra_dual_convention/downstream_mlp3_nonshift
STORED = {"sigma1": 233.4799046158332, "sigma1_over_sqrt_s": 2.334799046158332,
          "sigma_min_over_sigma1": 0.003185516327318493, "sigma_min_over_sqrt_s": 0.0074375404825450095,
          "REL_count_1e-2": 489, "ABS_count_0.1": 709, "ABS_count_0.01": 90, "ABS_count_0.001": 0,
          "logits_rms": 11.354487677569475}

def bracket(drift, grid, tol=TOL):
    """Cumulative tested bracket (phase0f convention): first tested radius at/above tol; a later
    re-entry cannot erase it. last_safe = previous tested radius (0.0 if first grid point crosses)."""
    d = np.asarray(drift, float); cross = np.where(d >= tol)[0]
    if len(cross) == 0: return {"last_safe": grid[-1], "first_cross": None, "censored": True}
    j = int(cross[0]); return {"last_safe": (grid[j-1] if j > 0 else 0.0), "first_cross": grid[j], "censored": False}

def dist(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.10)), "q90": float(np.quantile(x, 0.90)),
            "min": float(x.min()), "max": float(x.max())}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seed-tag", default="s0")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--out", required=True); ap.add_argument("--bespoke", required=True,
                    help="phase0f_s0_m256.json (read-only; cross-check + repro gate)")
    args = ap.parse_args(); torch.manual_seed(0); t_start = time.time()

    ck = torch.load(args.ckpt, map_location="cpu"); ncls = ck["n_classes"]
    net = tvm.vit_b_16(weights=None); net.heads = nn.Sequential(); net.heads.add_module("head", nn.Linear(768, ncls))
    net.load_state_dict(ck["state_dict"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    blk = net.encoder.layers[11]
    cache_file = cache_path(f"clsmid_{args.seed_tag}_n{args.n}.pt")
    # ---- OPERATOR SECTION ON CPU, bit-identical to phase0f (LAPACK eigenbasis is the one the
    # bespoke record's seeded probes were built from; a GPU eigh returns a rotated basis of the
    # same near-degenerate tail span and breaks bit-level reproduction of the stored records) ----
    cls_mid = torch.load(cache_file)["cls_mid"].double().cpu(); N = cls_mid.shape[0]; s = N
    g2, b2 = blk.ln_2.weight.double(), blk.ln_2.bias.double(); eps2 = blk.ln_2.eps
    W0, b0 = blk.mlp[0].weight.double(), blk.mlp[0].bias.double()
    W3, b3 = blk.mlp[3].weight.double(), blk.mlp[3].bias.double()
    gf_c, bf_c = net.encoder.ln.weight.double(), net.encoder.ln.bias.double(); epsf = net.encoder.ln.eps
    V_c, bV_c = net.heads.head.weight.double(), net.heads.head.bias.double(); d = V_c.shape[0]

    # base forward (phase0f lines, CPU)
    u = ln_fwd(cls_mid, g2, b2, eps2); z0 = u@W0.T+b0; a_c = gelu(z0); z3 = a_c@W3.T+b3; bo_c = cls_mid+z3
    mu = bo_c.mean(-1, keepdim=True); s_bo_c = torch.sqrt(bo_c.var(-1, unbiased=False, keepdim=True)+epsf)
    rhat_c = (bo_c-mu)/s_bo_c
    f_c = ln_fwd(bo_c, gf_c, bf_c, epsf); logits_c = f_c@V_c.T+bV_c
    logits_rms = float(torch.sqrt((logits_c**2).sum(1).mean()))

    def VJ_batch(idx):
        Vg = V_c*gf_c; rsum = Vg.sum(1); rh = rhat_c[idx]; sb = s_bo_c[idx]; Vgr = rh@Vg.T
        return ((Vg.unsqueeze(0)-rsum.view(1, -1, 1)/768)-torch.einsum("bc,bm->bcm", Vgr, rh)/768)/sb.view(-1, 1, 1)

    # ---- downstream GN operator G3 (phase0f accumulation, chunk 500, CPU) ----
    G3 = torch.zeros((768, 768), dtype=torch.float64)
    for i in range(0, N, 500):
        sl = torch.arange(i, min(i+500, N)); VJ = VJ_batch(sl); G3 += VJ.reshape(-1, 768).T@VJ.reshape(-1, 768)
    ones768_c = torch.ones(768, dtype=torch.float64)
    Qsh = torch.linalg.svd(torch.eye(768, dtype=torch.float64)
                           - torch.outer(ones768_c/np.sqrt(768.), ones768_c/np.sqrt(768.)))[0][:, :767]
    ev3, evec3 = torch.linalg.eigh(Qsh.T@G3@Qsh)
    sv_d3 = np.sqrt(ev3.flip(0).clamp(min=0).numpy()); V3 = (Qsh@evec3.flip(1))     # 768x767 desc, shift EXCLUDED
    # full-operator spectrum (shift INCLUDED) for the exact-rank T(i) statement
    evF, _ = torch.linalg.eigh(G3); sv_full = np.sqrt(evF.flip(0).clamp(min=0).numpy())  # 768 desc

    # ---- DEV copies for the battery (values derived from the single CPU base forward) ----
    a = a_c.to(DEV); bo = bo_c.to(DEV); f = f_c.to(DEV); logits = logits_c.to(DEV)
    rhat = rhat_c.to(DEV); s_bo = s_bo_c.to(DEV)
    gf, bf = gf_c.to(DEV), bf_c.to(DEV); V, bV = V_c.to(DEV), bV_c.to(DEV)
    ones768 = ones768_c.to(DEV); V3 = V3.contiguous()
    base_tok = logits.norm(dim=1) + 1e-30                              # per-example norms (secondary metric)

    def apply_JLN(delta):
        c = delta-delta.mean(-1, keepdim=True)-(((rhat*delta).sum(-1, keepdim=True))/768)*rhat; return (gf/s_bo)*c

    # ---- gate 1: spectrum reproduction vs stored summaries ----
    rep = {"sigma1": float(sv_d3[0]), "sigma1_over_sqrt_s": float(sv_d3[0]/np.sqrt(s)),
           "sigma_min_over_sigma1": float(sv_d3[-1]/sv_d3[0]), "sigma_min_over_sqrt_s": float(sv_d3[-1]/np.sqrt(s)),
           "REL_count_1e-2": int((sv_d3 < 1e-2*sv_d3[0]).sum()),
           "ABS_count_0.1": int((sv_d3/np.sqrt(s) < 0.1).sum()), "ABS_count_0.01": int((sv_d3/np.sqrt(s) < 0.01).sum()),
           "ABS_count_0.001": int((sv_d3/np.sqrt(s) < 0.001).sum()), "logits_rms": logits_rms}
    gate_spec = {}
    for k, vref in STORED.items():
        vnew = rep[k]
        ok = (vnew == vref) if isinstance(vref, int) else (abs(vnew-vref) <= 1e-6*max(abs(vref), 1e-12))
        gate_spec[k] = {"stored": vref, "recomputed": vnew, "pass": bool(ok)}
    spectrum_gate_pass = all(v["pass"] for v in gate_spec.values())

    # ---- exact-rank T(i) on the full operator ----
    # The Gram/eigh route cannot resolve sigma below the eigh noise floor sqrt(eps_mach*lambda_max)
    # (~1e-6 in sigma units here), which sits ABOVE the 1e-9*sigma0 cut (~2.3e-7): per T(i), a rank
    # statement without a resolved gap is UNRESOLVED through eigh alone.  The certified null
    # candidate (LN-shift coordinate) is therefore measured DIRECTLY: sigma(shift) = ||VJ.shift||_F
    # over the dataset (exact JVP), machine precision; the retained part comes from the eigh spectrum.
    shift_unit = ones768_c/np.sqrt(768.)
    sig_shift = 0.0
    for i in range(0, N, 1000):
        sl = torch.arange(i, min(i+1000, N)); VJ = VJ_batch(sl)
        sig_shift += float((torch.einsum("bcx,x->bc", VJ, shift_unit)**2).sum())
    sig_shift = float(np.sqrt(sig_shift))
    n_null = int(sig_shift <= 1e-9*sv_d3[0])
    gap = float(sv_d3[-1]/max(sig_shift, 1e-300)); gap_resolved = bool(gap >= 10.0)
    exact_rank = {"operator": "G3 full (shift included), 768 coords; shift coordinate measured by "
                              "DIRECT JVP (the eigh noise floor sqrt(eps*lambda_max) ~ 1e-6 sigma "
                              "exceeds the 1e-9*sigma0 cut, so the eigh spectrum alone leaves the "
                              "null UNRESOLVED; sv_full[-1] reported for the record)",
                  "sigma_shift_direct": sig_shift, "sigma_shift_direct_over_sigma0": sig_shift/float(sv_d3[0]),
                  "sigma_shift_eigh_noisefloor": float(sv_full[-1]),
                  "null_dim_coord": n_null, "null_dim_param_space": 3073*n_null,
                  "smallest_nonshift_sigma": float(sv_d3[-1]),
                  "gap": gap, "gap_resolved_ge_10x": gap_resolved,
                  "nonshift_all_above_cut": bool(sv_d3[-1] > 1e-9*sv_d3[0]*10),
                  "matches_certified_LNshift_DN": bool(3073*n_null == 3073)}

    # ---- cuts ----
    k_bespoke = int((sv_d3 < 1e-2*sv_d3[0]).sum())                                 # sigma-count 1e-2 (489)
    k_abs = int((sv_d3/np.sqrt(s) <= 1e-3).sum())                                  # absolute-logit reading (empty)
    thr_rel = 1e-3*logits_rms*np.sqrt(s)                                           # sigma threshold, primary reading
    k_rel = int((sv_d3 <= thr_rel).sum())                                          # ||.||_X-relative reading (PRIMARY)
    T3 = V3[:, 767-k_rel:].contiguous() if k_rel > 0 else V3[:, :0]

    # ---- probe construction (phase0f make_raw / proj_exact conventions, mlp3 dtail) ----
    def proj_exact(dA):                                   # remove LN-shift (all-ones-output); feature-null = 0
        return dA - torch.outer(ones768, dA.mean(0))
    def make_raw(basis, mode, rng, k):
        # draws AND the U@R product on CPU (phase0f parity); result moved to DEV for the battery
        n, m1 = 768, 3073
        if mode == "generic": return torch.randn(n, m1, generator=rng, dtype=torch.float64).to(DEV)
        U = basis if mode == "tail" else torch.linalg.qr(torch.randn(n, k, generator=rng, dtype=torch.float64))[0]
        return (U@torch.randn(k, m1, generator=rng, dtype=torch.float64)).to(DEV)

    one = torch.ones(N, 1, dtype=torch.float64, device=DEV)  # noqa (kept for parity with phase0f)
    def rms_rel(dl): return float(torch.sqrt((dl**2).sum(1).mean()))/logits_rms          # PRIMARY ||.||_X
    def med_rel(dl): return float((dl.norm(dim=1)/base_tok).median())                    # SECONDARY per-example

    def eval_dir(dA, grid):
        """Paired battery for one unit direction dA (768x3073). Returns jv_frob and per-radius curves."""
        dz3 = a@dA[:, :-1].T + dA[:, -1]; jln = apply_JLN(dz3); fo = jln@V.T             # first-order Dlogits/t
        jf = float(fo.norm())
        cur = {m: [] for m in ("actual_rms", "actual_med", "frozen_rms", "frozen_med",
                               "first_order_rms", "first_order_med", "transition_rms", "transition_med")}
        for t in grid:
            lg_act = ln_fwd(bo+t*dz3, gf, bf, epsf)@V.T+bV
            d_act = lg_act-logits; d_fro = t*fo                                          # frozen-LN == first-order (mlp3)
            cur["actual_rms"].append(rms_rel(d_act)); cur["actual_med"].append(med_rel(d_act))
            cur["frozen_rms"].append(rms_rel(d_fro)); cur["frozen_med"].append(med_rel(d_fro))
            cur["first_order_rms"].append(cur["frozen_rms"][-1]); cur["first_order_med"].append(cur["frozen_med"][-1])
            d_tr = d_act-d_fro                                                           # per-example VECTOR residual
            cur["transition_rms"].append(rms_rel(d_tr)); cur["transition_med"].append(med_rel(d_tr))
        return jf, cur

    # ---- gate 2: basis orthonormality + projection residuals ----
    gram_err = float((T3.T@T3 - torch.eye(k_rel, dtype=torch.float64)).abs().max()) if k_rel else 0.0
    shift_in_tail = float((T3.T@(ones768_c/np.sqrt(768.))).abs().max()) if k_rel else 0.0
    gate_ortho = {"tail_basis_gram_max_abs_err": gram_err, "tail_basis_shift_component_max": shift_in_tail,
                  "pass": bool(gram_err < 1e-10 and shift_in_tail < 1e-10)}

    # ---- gate 3: bit-level reproduction of the bespoke record (first 3 tail directions, k=489) ----
    besp = json.load(open(args.bespoke))
    bfam = besp["families"]["mlp3_dtail_sig1e2"]; kb = bfam["tail_dim_k"]
    assert kb == k_bespoke, (kb, k_bespoke)
    Tb = V3[:, 767-kb:].contiguous()
    rngb = torch.Generator().manual_seed(stable_seed(20260, "mlp3_dtail_sig1e2", "tail"))
    repro = []
    for mi in range(3):
        dA = proj_exact(make_raw(Tb, "tail", rngb, kb)); dA = dA/dA.norm()
        jfb, curb = eval_dir(dA, RGRID8)
        ref = bfam["tail"]["per_direction"][mi]
        ra = np.asarray(ref["drift"]["actual"], float); na = np.asarray(curb["actual_rms"], float)
        rf = np.asarray(ref["drift"]["frozenLN"], float); nf = np.asarray(curb["frozen_rms"], float)
        repro.append({"i": mi,
                      "jv_frob_stored": ref["jv_frob"], "jv_frob_recomputed": jfb,
                      "jv_rel_err": abs(jfb-ref["jv_frob"])/abs(ref["jv_frob"]),
                      "actual_max_rel_err": float(np.max(np.abs(na-ra)/np.maximum(np.abs(ra), 1e-30))),
                      "frozenLN_max_rel_err": float(np.max(np.abs(nf-rf)/np.maximum(np.abs(rf), 1e-30)))})
    gate_repro = {"per_direction": repro,
                  "pass": bool(all(r["jv_rel_err"] < 1e-6 and r["actual_max_rel_err"] < 1e-6
                                   and r["frozenLN_max_rel_err"] < 1e-6 for r in repro))}

    # ---- gate 4: exact LN-shift null at machine precision + autograd JVP spot check ----
    rngx = torch.Generator().manual_seed(stable_seed(20261117, "recut_pilot", "exactnull"))
    exact_checks = []
    for i in range(4):
        mix = torch.randn(3073, generator=rngx, dtype=torch.float64).to(DEV)
        dA = torch.outer(ones768, mix); dA = dA/dA.norm()
        dz3 = a@dA[:, :-1].T + dA[:, -1]; fo = apply_JLN(dz3)@V.T
        lg30 = ln_fwd(bo+30.0*dz3, gf, bf, epsf)@V.T+bV
        exact_checks.append({"i": i, "jv_RMS_rel": float(fo.norm())/np.sqrt(s)/logits_rms,
                             "actual_rms_rel_drift_r30": rms_rel(lg30-logits)})
    # autograd JVP vs analytic LN Jacobian (first 200 examples, 3 random tail probes)
    rngj = torch.Generator().manual_seed(stable_seed(20261117, "recut_pilot", "jvpspot"))
    jvp_errs = []
    for i in range(3):
        dA = proj_exact(make_raw(T3 if k_rel else V3, "tail" if k_rel else "generic", rngj, max(k_rel, 1)))
        dA = dA/dA.norm(); dz3 = (a@dA[:, :-1].T + dA[:, -1])[:200]
        fn = lambda x: ln_fwd(x, gf, bf, epsf)@V.T+bV
        _, jv_auto = torch.autograd.functional.jvp(fn, bo[:200], dz3, strict=True)
        jv_ana = apply_JLN_chunk = (lambda delta, idx: ((gf/s_bo[idx])*(delta-delta.mean(-1, keepdim=True)
                 - (((rhat[idx]*delta).sum(-1, keepdim=True))/768)*rhat[idx])))(dz3, slice(0, 200))@V.T
        jvp_errs.append(float((jv_auto-jv_ana).norm()/jv_ana.norm()))
    gate_exact = {"LNshift_spot_directions": exact_checks,
                  "autograd_vs_analytic_jvp_rel_err": jvp_errs,
                  "pass": bool(all(c["jv_RMS_rel"] < 1e-12 and c["actual_rms_rel_drift_r30"] < 1e-9
                                   for c in exact_checks) and all(e < 1e-12 for e in jvp_errs))}

    # ---- main battery: 256 probes x 3 modes on the common grid ----
    def limiter(cur):   # mlp3: only LN freeze available (phase0f attribution)
        la = bracket(cur["actual_rms"], RGRID10)["last_safe"]; lf = bracket(cur["frozen_rms"], RGRID10)["last_safe"]
        return "LN" if lf > la else "none"
    battery = {}; arrays = {}
    for mode in ("tail", "randsub", "generic"):
        seed = stable_seed(20261117, "recut_mlp3_dtail_common", mode)
        rng = torch.Generator().manual_seed(seed)
        per = []; drift_store = np.zeros((args.M, 8, len(RGRID10)))
        jv_rel = []; ls = []; fcs = []; cens = 0; lims = []
        for mi in range(args.M):
            dA = proj_exact(make_raw(T3, mode, rng, k_rel)); dA = dA/dA.norm()
            jf, cur = eval_dir(dA, RGRID10); br = bracket(cur["actual_rms"], RGRID10)
            jv_rel.append(jf/np.sqrt(s)/logits_rms); ls.append(br["last_safe"]); cens += int(br["censored"])
            lims.append(limiter(cur))
            if br["first_cross"] is not None: fcs.append(br["first_cross"])
            for ci, cname in enumerate(("actual_rms", "actual_med", "frozen_rms", "frozen_med",
                                        "first_order_rms", "first_order_med", "transition_rms", "transition_med")):
                drift_store[mi, ci] = cur[cname]
            per.append({"i": mi, "jv_RMS_over_v_relX": jv_rel[-1], "jv_frob": jf,
                        "bracket_primary": br, "limiter": lims[-1]})
        agg = {}
        for ci, cname in enumerate(("actual_rms", "actual_med", "frozen_rms", "frozen_med",
                                    "transition_rms", "transition_med")):
            ci_full = ("actual_rms", "actual_med", "frozen_rms", "frozen_med", "first_order_rms",
                       "first_order_med", "transition_rms", "transition_med").index(cname)
            agg[cname] = {f"r{r:g}": dist(drift_store[:, ci_full, ri]) for ri, r in enumerate(RGRID10)}
        exceed = {f"r{r:g}": int((drift_store[:, 0, ri] > TOL).sum()) for ri, r in enumerate(RGRID10)}
        battery[mode] = {"generator_seed": seed, "M": args.M,
                         "jv_operator_RMS_over_v_relX": dist(jv_rel),
                         "last_safe_tested_tol1e-3": dist(ls),
                         "first_cross_tested": (dist(fcs) if fcs else None),
                         "censored_count": cens,
                         "limiter_counts": {c: int(sum(l == c for l in lims)) for c in ("LN", "none")},
                         "exceedance_count_actual_rms_gt_tol_among_256": exceed,
                         "aggregates_per_radius": agg, "per_direction": per}
        arrays[f"{mode}_drift"] = torch.tensor(drift_store)
        print(f"{mode:8s} jvRMS_relX={battery[mode]['jv_operator_RMS_over_v_relX']['median']:.3e} "
              f"last_safe={battery[mode]['last_safe_tested_tol1e-3']['median']:g} cens={cens} "
              f"exceed@r1={exceed['r1']}", flush=True)

    # ---- cross-check vs bespoke record at the 8 shared radii ----
    besp_actual = np.array([pd_["drift"]["actual"] for pd_ in bfam["tail"]["per_direction"]])  # 256 x 8
    new_actual_shared = arrays["tail_drift"][:, 0, 2:].numpy()                                  # radii 0.01..30
    xc = {}
    for ri, r in enumerate(RGRID8):
        b_lo, b_hi = float(besp_actual[:, ri].min()), float(besp_actual[:, ri].max())
        inside = int(((new_actual_shared[:, ri] >= b_lo) & (new_actual_shared[:, ri] <= b_hi)).sum())
        xc[f"r{r:g}"] = {"recut_median": float(np.median(new_actual_shared[:, ri])),
                         "recut_range": [float(new_actual_shared[:, ri].min()), float(new_actual_shared[:, ri].max())],
                         "bespoke_median": float(np.median(besp_actual[:, ri])), "bespoke_range": [b_lo, b_hi],
                         "recut_within_bespoke_range_count_of_256": inside,
                         "recut_at_or_below_bespoke_max_count_of_256": int((new_actual_shared[:, ri] <= b_hi).sum())}

    out = {
        "pilot": "2026-11-17 ViT standardization re-cut PILOT: seed 0, block-11 mlp.3, downstream-tail family at the common T(ii) cut",
        "seed_tag": args.seed_tag, "N": N, "M": args.M, "device": str(DEV), "dtype": "float64",
        "ckpt": os.path.abspath(args.ckpt), "ckpt_sha256": sha256(args.ckpt),
        "cache_file": os.path.abspath(cache_file), "cache_sha256": sha256(cache_file),
        "logits_rms": logits_rms, "RGRID": RGRID10, "tol": TOL,
        "quadruple": {"operator": "downstream Gauss-Newton G3 of z3 through (+residual -> final LN -> head), "
                                  "nonshift spectrum (certified LN-shift exact null excluded and projected out)",
                      "normalization": "unit Frobenius norm of the [W|b] (768x3073) direction; output metric "
                                       "||.||_X = dataset RMS over N=10000, relative to ||F(theta0)||_X",
                      "radius_grid": RGRID10,
                      "threshold": "first-order rate (sigma/sqrt(s))/||F(theta0)||_X <= 1e-3 (PRIMARY reading of "
                                   "T(ii) 'sigma/sqrt(s) <= 1e-3 in ||.||_X units'); absolute reading also reported"},
        "operator_spectrum": {"source": "RECOMPUTED (stored records persist only summary sigma quantities in "
                                        "phase0d_s0.json spectra_dual_convention; full spectrum not persisted)",
                              "reproduction_gate": gate_spec, "pass": spectrum_gate_pass,
                              "sigma1": rep["sigma1"], "sigma1_over_sqrt_s": rep["sigma1_over_sqrt_s"],
                              "sigma_min_over_sigma1": rep["sigma_min_over_sigma1"],
                              "sigma_min_over_sqrt_s": rep["sigma_min_over_sqrt_s"]},
        "exact_rank_T1": exact_rank,
        "dims": {"k_common_relX_PRIMARY": k_rel,
                 "k_common_absolute_logits": k_abs,
                 "k_bespoke_sigma_count_1e-2_this_seed": k_bespoke,
                 "bespoke_family_range_across_seeds": "466-489 (s1/s2 per phase0f records)",
                 "sigma_threshold_relX_in_logit_units": float(thr_rel),
                 "NOTE": "ABSOLUTE reading of T(ii) gives an EMPTY tail (sigma_min/sqrt(s)=7.4e-3 > 1e-3): "
                         "convention decision flagged for the PI; PRIMARY here follows the spec's ||.||_X "
                         "definition ('relative to ||F(theta0)||_X')."},
        "frozen_convention": "frozen = frozen-LN forward (existing validated phase0f code). For mlp.3 the "
                             "downstream nonlinearity is the final LN alone and the head is linear, so frozen "
                             "STRUCTURALLY COINCIDES with first_order (JVP); joint_interaction = "
                             "||F_frozen - F_first_order|| = 0 STRUCTURALLY. transition = ||F_actual - F_frozen|| "
                             "per-example VECTOR residual, persisted. No frozen-GELU mode exists downstream of mlp.3.",
        "battery": battery,
        "crosscheck_vs_bespoke_shared_radii": {"bespoke_record": os.path.abspath(args.bespoke),
                                               "bespoke_family": "mlp3_dtail_sig1e2 (k=489)", "per_radius": xc,
                                               "NOTE": "consistency, not equality: different dims (re-cut tail span "
                                                       "strictly contained in bespoke tail span)"},
        "gates": {"1_spectrum_reproduction": {"pass": spectrum_gate_pass},
                  "2_probe_orthonormality": gate_ortho,
                  "3_bespoke_record_reproduction": gate_repro,
                  "4_exact_null_machine_precision": gate_exact,
                  "5_frozen_vs_actual": "existing battery includes frozen-LN; reproduced bit-level in gate 3; "
                                        "frozen==first_order structural statement recorded"},
        "statistical_wording": "exceedance counts among 256 probes; zero exceedances among 256 controls reported "
                               "as such or as the plus-one estimate 1/257; no p-values",
        "runtime_s": time.time()-t_start,
    }
    arrays.update({"sv_d3_nonshift": torch.tensor(sv_d3), "sv_full": torch.tensor(sv_full),
                   "tail_basis_T3": T3.cpu(), "RGRID": torch.tensor(RGRID10),
                   "drift_axis": ["actual_rms", "actual_med", "frozen_rms", "frozen_med",
                                  "first_order_rms", "first_order_med", "transition_rms", "transition_med"]})
    torch.save(arrays, args.out.replace(".json", "_arrays.pt"))
    with open(args.out, "w") as fo: json.dump(out, fo, indent=1)
    print("k_rel=", k_rel, "k_abs=", k_abs, "k_bespoke=", k_bespoke,
          "gates:", {g: (v if isinstance(v, str) else v.get("pass")) for g, v in out["gates"].items()})
    print("wrote", args.out, "\nDONE", flush=True)

if __name__ == "__main__": main()
