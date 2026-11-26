"""Matched-threshold Hessian/GGN near-null comparison (PLAN_ARXIV_COMPLETION_2026-11-11, Track 4; PI corrections 2026-11-11).

For each common absolute curvature threshold tau in {1e-8, 1e-6, 1e-4}: H_tau = span{v_i : |lambda_H,i| <= tau},
G_tau = span{u_j : |lambda_G,j| <= tau}, both from the SAVED eigensystems (NO H/G rebuild). Before any sub-selection the
column <-> eigenvalue mapping of every saved basis is verified: (a) column counts vs eigenvalue-mask counts, (b) nesting
residuals between bases that must be nested, (c) sampled ON-CHECKPOINT eigenvalue checks (v'Gv = (2/s)||Jv||^2 by JVP,
||Gv - lambda v|| by JVP+VJP, v'Hv by HVP). Selected column masks and dimensions are SAVED per (seed, tau, side).
Reports: dims, H_tau-by-G_tau coverage AND the reverse, principal angles where tractable (min dim <= SVLIM), dimension-
aware random expectations, PSD/noise checks on G. Numerical-zero classification is separate and labelled (the 1e-14
numerical GGN kernel stays a reference, not a matched threshold). Residual analysis: directions that REMAIN inside
H_tau (they are H-eigenbasis columns) with the LOWEST G_tau projection mass; per-direction v'Gv, v'Hv, v'Rv by JVP/HVP.
Coverage in float32 (TF32 off for reproducibility) with float64 spot-validation; eigen-classification in float64.
Light GPU job (single GPU, 8 threads)."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from flatocean import frozen_lib as FL
from flatocean import runs
from torch.func import functional_call, jvp as fjvp, vjp as fvjp, grad as fgrad

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]; EPOCH = 500
TAUS = [1e-8, 1e-6, 1e-4]              # matched absolute curvature grid (both sides)
GLOOSE_REL = 1e-9                       # the saved loose-kernel rule: |lambda_G| <= 1e-9 * max|lambda_G|
KER_NUM = 1e-14                         # numerical GGN kernel (REFERENCE ONLY, labelled; not a matched threshold)
SVLIM = 17000                           # principal angles only when min(dim) <= SVLIM (GPU svdvals budget)
NSAMP = 8                               # sampled eigencheck: NSAMP boundary + NSAMP random columns per basis
NRES = 32                               # residual-analysis directions per seed (lowest G_tau mass inside H_tau)
OUT = os.path.join(HERE, "results", "hessian_gn")

def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    res = {"protocol": {"taus": TAUS, "ker_numerical_reference": KER_NUM, "svlim": SVLIM,
                        "objective": "L = sum_i ||f(x_i)-y_i||^2 / s (one-hot MSE, no reg); G = 2 J^T J / s; H = G + R",
                        "normalization": "identical examples/outputs/coordinates for H and G (same saved eigensystems)"},
           "seeds": {}}
    for sd in SEEDS:
        t0 = time.time(); S = {"verify": {}, "matched": {}, "residual_analysis": {}, "numerical_reference": {}}
        m = torch.load(os.path.join(OUT, f"hgn_cov_w64_seed{sd}_mats.pt"), weights_only=False)
        evH = m["evH"].numpy(); evG = m["evG"].numpy(); p = evH.shape[0]
        Hn4 = m["Hnull_1e-4"]; oc = m["oceans"]
        gmax = float(np.abs(evG).max())

        # ---------- (a) column-count vs eigenvalue-mask verification ----------
        hmask4 = np.abs(evH) <= 1e-4; h_ev4 = evH[hmask4]
        loosemask = np.abs(evG) <= GLOOSE_REL * gmax; loose_ev = evG[loosemask]
        counts = {"Hnull_1e-4": (int(hmask4.sum()), int(Hn4.shape[1])),
                  "kerG_loose": (int(loosemask.sum()), int(oc["kerG_combined_exact"].shape[1]))}
        gt_bases = {}
        for tg, key in [(1e-6, "GN_tail|1e-06"), (1e-4, "GN_tail|0.0001"), (1e-2, "GN_tail|0.01")]:
            counts[key] = (int((np.abs(evG) <= tg).sum()), int(oc[key].shape[1])); gt_bases[tg] = oc[key]
        S["verify"]["count_match"] = {k: {"mask": a, "cols": b, "match": a == b} for k, (a, b) in counts.items()}

        # ---------- (b) nesting residuals (fp32 GPU; subspace containment of the smaller in the larger) ----------
        def nest_resid(Qsmall, Qbig, nsub=2048):
            cols = torch.randperm(Qsmall.shape[1])[:min(nsub, Qsmall.shape[1])]
            A = Qsmall[:, cols].to(DEV, torch.float32); B = Qbig.to(DEV, torch.float32)
            R = A - B @ (B.T @ A); r = float(R.norm() / A.norm()); del A, B, R; torch.cuda.empty_cache(); return r
        S["verify"]["nesting_resid_sampled"] = {
            "kerG_loose_in_GNtail1e-6": nest_resid(oc["kerG_combined_exact"], gt_bases[1e-6]),
            "GNtail1e-6_in_1e-4": nest_resid(gt_bases[1e-6], gt_bases[1e-4]),
            "GNtail1e-4_in_1e-2": nest_resid(gt_bases[1e-4], gt_bases[1e-2])}

        # ---------- PSD / noise checks on G ----------
        S["verify"]["G_psd"] = {"evG_min": float(evG.min()), "evG_max": gmax,
                                "n_negative": int((evG < 0).sum()),
                                "most_negative": float(evG.min()),
                                "neg_floor_note": "negative values are eigensolver noise; magnitude sets the resolvable floor"}
        S["verify"]["H_context"] = {"evH_min": float(evH.min()), "H_neg_frac_1e-8": float((evH < -1e-8).mean())}

        # ---------- (c) sampled on-checkpoint eigenvalue verification (JVP/VJP/HVP) ----------
        run = runs.load_run("mnist", "mlp", sd, tag="w64", epoch=EPOCH, device="cpu", sets=("analysis",))
        net = run["net"].eval()
        for pr in net.parameters(): pr.requires_grad_(False)
        theta = FL.flat_params(net).double().to(DEV)
        Xc, y, _ = run["data"]["analysis"]; X = Xc.double().to(DEV); s_n = X.shape[0]
        Y = torch.nn.functional.one_hot(y.long(), 10).double().to(DEV)
        ps = list(FL.param_slices(net)); shp = {nm: pm.shape for nm, pm in net.named_parameters()}
        def F(th): return functional_call(net, {nm: th[a:b].view(shp[nm]) for nm, a, b in ps}, (X,))
        def Lf(th): r = F(th) - Y; return (r * r).sum() / s_n
        def Jv(v): _, jv = fjvp(F, (theta,), (v,)); return jv
        def JTw(w): _, vj = fvjp(F, theta); return vj(w)[0]
        def Gv(v): return (2.0 / s_n) * JTw(Jv(v))
        def Hv(v): _, hv = fjvp(fgrad(Lf), (theta,), (v,)); return hv
        gen = torch.Generator().manual_seed(1111 + sd)
        def eigcheck_G(Q, lam, label):
            k = Q.shape[1]
            idx = list(range(max(0, k - NSAMP), k)) + torch.randint(0, k, (NSAMP,), generator=gen).tolist()
            dev_abs = []; op_res = []
            for i in sorted(set(idx)):
                v = Q[:, i].double().to(DEV); lv = float(lam[i])
                gv = Gv(v); dev_abs.append(abs(float(v @ gv) - lv)); op_res.append(float((gv - lv * v).norm()))
            return {"label": label, "k": k, "max_abs_dev_vGv_vs_lambda": max(dev_abs),
                    "max_opnorm_resid_Gv_minus_lv": max(op_res), "n_checked": len(set(idx))}
        def eigcheck_H(Q, lam):
            k = Q.shape[1]
            idx = list(range(max(0, k - NSAMP), k)) + torch.randint(0, k, (NSAMP,), generator=gen).tolist()
            dev = []
            for i in sorted(set(idx)):
                v = Q[:, i].double().to(DEV); dev.append(abs(float(v @ Hv(v)) - float(lam[i])))
            return {"k": k, "max_abs_dev_vHv_vs_lambda": max(dev), "n_checked": len(set(idx))}
        S["verify"]["eigencheck"] = {
            "H(Hnull_1e-4)": eigcheck_H(Hn4, h_ev4),
            "G(kerG_loose)": eigcheck_G(oc["kerG_combined_exact"], loose_ev, "kerG_loose"),
            "G(GN_tail|1e-6)": eigcheck_G(gt_bases[1e-6], evG[np.abs(evG) <= 1e-6], "GN_tail1e-6"),
            "G(GN_tail|1e-4)": eigcheck_G(gt_bases[1e-4], evG[np.abs(evG) <= 1e-4], "GN_tail1e-4")}

        # ---------- matched-threshold sub-selection; SAVE masks ----------
        masks = {}
        def H_tau(tau):
            sub = np.abs(h_ev4) <= tau; masks[f"H|{tau:g}"] = ("subset_of_Hnull_1e-4_cols", sub.copy())
            return Hn4[:, sub], int(sub.sum())
        def G_tau(tau):
            if tau <= GLOOSE_REL * gmax:
                sub = np.abs(loose_ev) <= tau; masks[f"G|{tau:g}"] = ("subset_of_kerG_loose_cols", sub.copy())
                return oc["kerG_combined_exact"][:, sub], int(sub.sum())
            base = gt_bases[tau]                              # exact saved threshold base (count-verified above)
            masks[f"G|{tau:g}"] = (f"whole_GN_tail|{tau:g}", np.ones(base.shape[1], dtype=bool))
            return base, int(base.shape[1])

        def cov_pair(QA, QB):                                  # returns mass, both coverages, angles if tractable
            A = QA.to(DEV, torch.float32); B = QB.to(DEV, torch.float32)
            M = A.T @ B; mass = float((M * M).sum())
            out = {"mass": mass, "cov_B_by_A": mass / B.shape[1], "cov_A_by_B": mass / A.shape[1]}
            if min(M.shape) <= SVLIM:
                cs = torch.linalg.svdvals(M).clamp(0, 1)
                out["pa_cos_gt_0.99"] = int((cs > 0.99).sum()); out["pa_cos_gt_0.9"] = int((cs > 0.9).sum())
                out["pa_top1"] = float(cs[0])
            else:
                out["pa_note"] = f"principal angles skipped (min dim > {SVLIM})"
            del A, B, M; torch.cuda.empty_cache(); return out

        for tau in TAUS:
            QH, dh = H_tau(tau); QG, dg = G_tau(tau)
            c = cov_pair(QG, QH)
            ent = {"dim_H": dh, "dim_G": dg, "random_H_side": dg / p, "random_G_side": dh / p,
                   "cov_H_by_G": c["cov_B_by_A"], "cov_G_by_H": c["cov_A_by_B"]}
            ent.update({k: v for k, v in c.items() if k.startswith("pa")})
            # exact atomic components + atomic union on the same H_tau (branch/atomic coverage on the matched grid)
            for nm in ("exact_feature_null", "exact_downstream_null", "exact_joint_deficit", "exact_branch_union"):
                Q = oc[nm]; A = Q.to(DEV, torch.float32); B = QH.to(DEV, torch.float32)
                ent[f"covH_{nm}"] = float(((A.T @ B) ** 2).sum()) / max(dh, 1); del A, B; torch.cuda.empty_cache()
            S["matched"][f"{tau:g}"] = ent

        # ---------- numerical-zero reference (separate, labelled) ----------
        sub14 = np.abs(loose_ev) <= KER_NUM
        S["numerical_reference"] = {"kerG_dim_1e-14": int(sub14.sum()),
            "note": "numerical GGN kernel at the 1e-14 classification threshold; NOT a matched curvature threshold"}

        # ---------- residual analysis: inside H_tau, lowest G_tau projection (tau = 1e-8) ----------
        tau = 1e-8; QH, dh = H_tau(tau); QG, dg = G_tau(tau)
        A = QG.to(DEV, torch.float32); B = QH.to(DEV, torch.float32)
        colmass = ((A.T @ B) ** 2).sum(0).cpu()               # per-H_tau-column G_tau projection mass (in [0,1])
        del A, B; torch.cuda.empty_cache()
        order = torch.argsort(colmass)[:NRES]
        rows = []
        hsub = np.where(np.abs(h_ev4) <= tau)[0]
        for i in order.tolist():
            v = QH[:, i].double().to(DEV)
            vhv = float(v @ Hv(v)); jv = Jv(v); vgv = float((2.0 / s_n) * (jv * jv).sum())
            rows.append({"col": int(i), "lambda_H_recorded": float(h_ev4[hsub[i]]), "Gtau_mass": float(colmass[i]),
                         "vHv": vhv, "vGv": vgv, "vRv": vhv - vgv, "inside_H_tau": abs(vhv) <= tau * 1.5})
        S["residual_analysis"] = {"tau": tau, "n": len(rows), "selection": "H_tau eigenbasis columns with lowest G_tau projection mass (remain inside H near-null by construction; verified via vHv)",
                                  "rows": rows,
                                  "summary": {"median_vGv": float(np.median([r["vGv"] for r in rows])),
                                              "median_|vRv|": float(np.median([abs(r["vRv"]) for r in rows])),
                                              "all_inside_H_tau": all(r["inside_H_tau"] for r in rows)}}

        # ---------- float64 spot-validation of two coverage cells ----------
        spots = {}
        for tau in (1e-8, 1e-4):
            QH, dh = H_tau(tau); QG, dg = G_tau(tau)
            A32 = QG.to(DEV, torch.float32); B32 = QH.to(DEV, torch.float32)
            c32 = float(((A32.T @ B32) ** 2).sum()) / dh; del A32, B32; torch.cuda.empty_cache()
            A64 = QG.to(DEV, torch.float64); B64 = QH.to(DEV, torch.float64)
            c64 = float(((A64.T @ B64) ** 2).sum()) / dh; del A64, B64; torch.cuda.empty_cache()
            spots[f"{tau:g}"] = {"fp32": c32, "fp64": c64, "abs_diff": abs(c32 - c64)}
        S["verify"]["fp64_spot_covH_by_G"] = spots

        # ---------- save masks + per-seed timing ----------
        torch.save({k: {"rule": r, "mask": torch.from_numpy(mm)} for k, (r, mm) in masks.items()},
                   os.path.join(OUT, f"hgn_matched_indices_seed{sd}.pt"))
        S["runtime_s"] = round(time.time() - t0, 1)
        res["seeds"][str(sd)] = S
        print(f"seed {sd} done in {S['runtime_s']}s: " + ", ".join(
            f"tau={t}: dH={S['matched'][f'{float(t):g}']['dim_H']} dG={S['matched'][f'{float(t):g}']['dim_G']} "
            f"H|G={S['matched'][f'{float(t):g}']['cov_H_by_G']:.3f} G|H={S['matched'][f'{float(t):g}']['cov_G_by_H']:.3f}"
            for t in TAUS), flush=True)
        del m, Hn4, oc, run, net; torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(OUT, "hgn_matched_threshold_3seed.json"), "w"), indent=1, default=str)
    print("wrote results/hessian_gn/hgn_matched_threshold_3seed.json + hgn_matched_indices_seed{0,1,2}.pt", flush=True)

if __name__ == "__main__":
    main()
