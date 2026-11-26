#!/usr/bin/env python
"""Hessian-near-null COVERAGE by independently-constructed oceans — STANDARD MNIST width-64 ReLU MLP (256-64-64-64-10,
25,418 params, digits 0-9), 3 seeds (2026-10-07). This is the full-MNIST analogue of the task-A Hessian/GN pilot
(hgn_ocean_poc.py), with NO CL/task-A dependency: the oceans are constructed here from the authenticated checkpoint + the
10k analysis set via the standard atomic-geometry primitives (branch.py / comps), independent of the Hessian spectrum.

CENTRAL QUESTION (per PI): how much of the Hessian near-null space do the oceans EXPLAIN?
  coverage(O) = ||Q_O^T Q_H||_F^2 / dim(H_null)        (Q_H from the Hessian spectrum; Q_O constructed independently)
  random expectation for a dim-k ocean = k / p.
Report, across an eigenvalue-threshold sweep |lambda|<=tau:
  - coverage by each INDIVIDUAL exact family (feature-null / downstream-null / joint-deficit),
  - coverage by the COMBINED exact ocean: (a) structural union = branch, (b) the data-Jacobian kernel ker(G)=ker(J_X),
  - ADDITIONAL coverage from approximate J/GN tails: G near-null at 0<|lambda_G|<=tau_G (exact kernel + approximate shell),
  - dimension-aware comparison (ocean dim, H_null dim, random k/p),
  - PRINCIPAL-ANGLE / intersection diagnostics (singular values of Q_O^T Q_H: genuinely shared dims cos~1 vs diffuse mass),
  - and the UNEXPLAINED fraction 1 - coverage(best construction).
SUPPORTING diagnostics (not the headline): ocean INCLUSION (||Q_O^T Q_H||^2/dim_O) and curvature probes ||Jv||, v'Gv, v'Hv,
||Hv||, ||(H-G)v|| (256 probes). Keep v'Hv~0 DISTINCT from Hv~0 (a direction can have ~0 directional quadratic curvature
yet Hv!=0, i.e. NOT a Hessian-kernel direction). MSE data loss L=(1/m)sum||F-Y||^2, no reg; G=(2/m)JtJ; H=grad^2 L.
Each seed reported separately (eigenspaces NOT pooled). float64, GPU, memory-safe (G parked on CPU during the H build)."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from flatocean import tier1_train as T; from flatocean import frozen_lib as FL; from flatocean import ocean_lib as O; from flatocean import coordinates as CP
from flatocean.analyses import branch as B
from flatocean import runs
SEEDS = [int(x) for x in os.environ.get("HGN_SEEDS", "0").split(",")]
EPOCH = int(os.environ.get("HGN_EPOCH", "500")); NPROBE = int(os.environ.get("HGN_NPROBE", "256"))
TOLS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]        # H near-null absolute-eigenvalue thresholds
GEXACT = 1e-9                                  # ker(G): |lambda_G| <= GEXACT*lambda_max (the exact data-Jacobian kernel)
GTAIL = [1e-6, 1e-4, 1e-2]                     # approximate GN-tail thresholds (exact kernel + small-but-nonzero lambda_G)
OUT = os.path.join(HERE, "results", "hessian_gn")


def build_exact_oceans(net, X, nt=None):
    """feature-null / downstream-null / joint-deficit flat bases (25418 x dim), via the standard atomic primitives."""
    if nt is None: nt = B.Net(net, X)
    K, _ = B.atomic_kernels(nt, known_full_rank=("W1",)); RTOL = B.RTOL; lay = {"W2": 1, "W3": 2, "V": nt.L}
    def comps(blk, l):
        n, m1 = nt.shape[blk]; H = nt.H[l]
        _, svH, VhH = torch.linalg.svd(torch.linalg.qr(H, mode="r").R, full_matrices=True); rh = int((svH > RTOL * svH[0]).sum())
        Phi = VhH[:rh].T; Phic = VhH[rh:].T
        Fb = torch.einsum("jb,kc->jkbc", torch.eye(n, dtype=H.dtype), Phic).reshape(n * m1, n * Phic.shape[1])
        if blk == "V": DN = torch.zeros(n * m1, 0, dtype=H.dtype)
        else:
            D = nt.D[l].reshape(-1, n); _, svD, VhD = torch.linalg.svd(torch.linalg.qr(D, mode="r").R, full_matrices=True); rd = int((svD > RTOL * svD[0]).sum()); Om = VhD[rd:].T
            DN = torch.einsum("ja,kc->jkac", Om, Phi).reshape(n * m1, Om.shape[1] * Phi.shape[1]) if Om.shape[1] and Phi.shape[1] else torch.zeros(n * m1, 0, dtype=H.dtype)
        FDN = torch.cat([Fb, DN], 1)
        if K[blk].shape[1] == 0: JD = torch.zeros(n * m1, 0, dtype=H.dtype)
        else:
            coords = K[blk].T @ FDN; Q = torch.linalg.qr(coords, mode="complete").Q[:, coords.shape[1]:] if coords.shape[1] else torch.eye(K[blk].shape[1], dtype=H.dtype); JD = K[blk] @ Q
        return {"feature_null": Fb, "downstream_null": DN, "joint_deficit": JD}
    EC = {b: comps(b, lay[b]) for b in ("W2", "W3", "V")}
    EC["W1"] = {c: torch.zeros(nt.shape["W1"][0] * nt.shape["W1"][1], 0, dtype=torch.float64) for c in ("feature_null", "downstream_null", "joint_deficit")}
    oc = {}
    for comp in ("feature_null", "downstream_null", "joint_deficit"):
        oc["exact_" + comp] = CP.subspace_basis(net, nt, {b: EC[b][comp] for b in nt.blocks}, "cpu")
    oc["exact_branch_union"] = CP.subspace_basis(net, nt, {b: K[b] for b in nt.blocks}, "cpu")   # structural combined exact ocean
    return oc


def main():
    os.makedirs(OUT, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype = torch.float64
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    for sd in SEEDS:
        t0 = time.time()
        run = runs.load_run("mnist", "mlp", sd, tag="w64", epoch=EPOCH, device="cpu", sets=("analysis",))
        net = run["net"].eval()
        for pr in net.parameters(): pr.requires_grad_(False)
        theta = FL.flat_params(net).double().to(dev); p = theta.numel()
        Xc, y, idx = run["data"]["analysis"]; X = Xc.double().to(dev); m = X.shape[0]
        Y = torch.nn.functional.one_hot(y.long(), 10).double().to(dev)
        cfg = json.load(open(os.path.join(HERE, f"results/tier1_models/mnist_mlp_w64/seed_{sd}/config.json")))
        pslices = list(FL.param_slices(net)); pshapes = {nm: pm.shape for nm, pm in net.named_parameters()}
        from torch.func import functional_call, jvp as fjvp, vjp as fvjp, grad as fgrad, jacrev
        def F(th): return functional_call(net, {nm: th[a:b].view(pshapes[nm]) for nm, a, b in pslices}, (X,))
        def Lf(th): r = F(th) - Y; return (r * r).sum() / m
        L0 = float(Lf(theta)); gnorm = float(fgrad(Lf)(theta).norm())
        def Jv(v): _, jv = fjvp(F, (theta,), (v,)); return jv
        def JTw(w): _, vj = fvjp(F, theta); return vj(w)[0]
        def Gv(v): return (2.0 / m) * JTw(Jv(v))
        def Hv(v): _, hv = fjvp(fgrad(Lf), (theta,), (v,)); return hv

        # ---- independently-constructed exact oceans (on CPU), authenticated by ||Jv|| ----
        oc_cpu = build_exact_oceans(net.to("cpu"), X.cpu()); net.to(dev)
        auth = {}
        for nm, Q in oc_cpu.items():
            v = (Q @ torch.randn(Q.shape[1], dtype=dtype)); v = (v / v.norm()).to(dev)
            auth[nm] = {"dim": int(Q.shape[1]), "Jv_rms_sample": float(Jv(v).norm()) / m ** 0.5}
        print(f"s{sd} EPOCH{EPOCH} p={p} m={m} cfg_widths={cfg.get('widths')} L={L0:.3e} | exact-ocean auth(||Jv||/sqrt(s)): "
              + " ".join(f"{k.replace('exact_','')}={auth[k]['Jv_rms_sample']:.1e}(d{auth[k]['dim']})" for k in oc_cpu), flush=True)

        # ---- curvature probes (SUPPORTING): ||Jv|| v'Gv v'Hv ||Hv|| ||(H-G)v|| ----
        gen = torch.Generator(device="cpu").manual_seed(13 + sd)
        def unit_in(Q): w = Q @ torch.randn(Q.shape[1], generator=gen, dtype=dtype).to(Q.device); return (w / w.norm()).to(dev)
        def rand_full(): w = torch.randn(p, generator=gen, dtype=dtype); return (w / w.norm()).to(dev)
        def probe(v):
            jv = Jv(v); hv = Hv(v); gv = Gv(v)
            return {"Jv": float(jv.norm()) / m ** 0.5, "vGv": float((2.0 / m) * (jv * jv).sum()), "vHv": float(v @ hv), "Hv": float(hv.norm()), "HmG_v": float((hv - gv).norm())}
        def dist(rows, k): a = np.array([r[k] for r in rows], float); return {"p50": float(np.percentile(a, 50)), "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}
        curv = {}
        for nm, Q in list(oc_cpu.items()) + [("random", None)]:
            Qd = Q.to(dev) if Q is not None else None
            rows = [probe(unit_in(Qd) if Qd is not None else rand_full()) for _ in range(NPROBE)]
            curv[nm] = {k: dist(rows, k) for k in ["Jv", "vGv", "vHv", "Hv", "HmG_v"]}
            print(f"  probe {nm:22s} Jv {curv[nm]['Jv']['p50']:.2e} vHv {curv[nm]['vHv']['p50']:+.2e} Hv {curv[nm]['Hv']['p50']:.2e} HmGv {curv[nm]['HmG_v']['p50']:.2e}", flush=True)
            del Qd
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---- explicit G then H (G parked on CPU during the H build) ----
        tcon = time.time()
        G = torch.zeros(p, p, dtype=dtype, device=dev); bs = int(os.environ.get("HGN_BS", "256"))
        def Fi(th, xi): return functional_call(net, {nm: th[a:b].view(pshapes[nm]) for nm, a, b in pslices}, (xi.unsqueeze(0),)).squeeze(0)
        jr = torch.func.vmap(torch.func.jacrev(Fi, argnums=0), in_dims=(None, 0))
        for i in range(0, m, bs):
            Jc = jr(theta, X[i:i + bs]).reshape(-1, p); G += Jc.T @ Jc
        G *= (2.0 / m); Gn = float(G.norm()); Gcpu = G.cpu(); del G
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        H = jacrev(fgrad(Lf), chunk_size=int(os.environ.get("HGN_HCHUNK", "128")))(theta)
        sym = float((H - H.T).norm() / max(H.norm(), 1e-30)); H = 0.5 * (H + H.T); Hn = float(H.norm())
        G = Gcpu.to(dev); del Gcpu;
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        R_over_H = float((H - G).norm() / max(Hn, 1e-30)); G_over_H = float(Gn / max(Hn, 1e-30))
        teig = time.time()
        evG, UG = torch.linalg.eigh(G); del G
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        evH, UH = torch.linalg.eigh(H); del H
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        evG_c = evG.cpu().numpy(); evH_c = evH.cpu().numpy()

        # ---- oceans as orthonormal bases on GPU; add G-derived combined exact + approximate-tail oceans ----
        gmax = float(evG.abs().max())
        oceans = {k: v.to(dev) for k, v in oc_cpu.items()}
        kerG_mask = torch.abs(evG) <= GEXACT * gmax
        oceans["kerG_combined_exact"] = UG[:, kerG_mask].contiguous()               # ker(J_X): the true combined exact ocean
        for tg in GTAIL:
            oceans[f"GN_tail|{tg:g}"] = UG[:, torch.abs(evG) <= tg].contiguous()      # exact kernel + approximate GN shell

        # save the expensive eig + ocean bases BEFORE the coverage loop, so a kill preserves the H/G/eig work
        torch.save({"evH": evH.cpu(), "evG": evG.cpu(), "oceans": {k: v.cpu() for k, v in oceans.items()},
                    "Hnull_1e-4": UH[:, torch.abs(evH) <= 1e-4].cpu(), "meta": {"seed": sd, "epoch": EPOCH, "p": p, "m": m}},
                   os.path.join(OUT, f"hgn_cov_w64_seed{sd}_mats.pt"))
        # ---- CENTRAL: coverage (mass) + inclusion + mean_cos^2 are SVD-free; principal-angle svdvals only for small oceans ----
        def frob2(A): return float((A * A).sum())
        PA_LIMIT = int(os.environ.get("HGN_PA_LIMIT", "4000"))   # exact principal angles (svdvals, on GPU) only when ocean dim k<=PA_LIMIT (covers all constructed oceans; skips huge GN-tail)
        spaces = {}
        for tol in TOLS:
            selH = torch.abs(evH) <= tol; QH = UH[:, selH]; dH = int(QH.shape[1])
            ent = {"Hnull_dim": dH, "Hnull_negative_dim": int(((torch.abs(evH) <= tol) & (evH < 0)).sum()), "oceans": {}}
            for nm, QO in oceans.items():
                k = int(QO.shape[1])
                if k == 0 or dH == 0:
                    ent["oceans"][nm] = {"ocean_dim": k, "coverage": 0.0, "inclusion": 0.0, "random_coverage_k_over_p": k / p, "mean_cos2": 0.0, "shared_dims_cos>0.99": 0}
                    continue
                M = QO.T @ QH; mass = frob2(M)                                       # ||Q_O^T Q_H||_F^2 = sum cos^2(principal angles)
                e = {"ocean_dim": k, "coverage": mass / dH, "inclusion": mass / k, "random_coverage_k_over_p": k / p, "mean_cos2": mass / min(k, dH)}
                if k <= PA_LIMIT:                                                     # genuine-intersection diagnostic (basis-independent), GPU svdvals
                    cos = torch.linalg.svdvals(M).clamp(0, 1); e["shared_dims_cos>0.99"] = int((cos > 0.99).sum()); e["shared_dims_cos>0.9"] = int((cos > 0.9).sum())
                else:
                    e["shared_dims_cos>0.99"] = None; e["shared_dims_cos>0.9"] = None   # too large for exact principal angles; coverage+mean_cos2 reported
                ent["oceans"][nm] = e
            spaces[f"{tol:g}"] = ent
            dd = ent["oceans"]
            print(f"  COVERAGE tol={tol:g} Hnull={dH}: branch={dd['exact_branch_union']['coverage']:.3f} kerG={dd['kerG_combined_exact']['coverage']:.3f} "
                  f"GN|1e-4={dd['GN_tail|0.0001']['coverage']:.3f} (randk/p branch={dd['exact_branch_union']['random_coverage_k_over_p']:.3f}) | feat={dd['exact_feature_null']['coverage']:.3f} down={dd['exact_downstream_null']['coverage']:.3f} joint={dd['exact_joint_deficit']['coverage']:.3f}", flush=True)

        res = {"seed": sd, "model": "mnist_mlp_w64 (standard, digits 0-9)", "epoch": EPOCH, "p": p, "m": m, "loss_mse": L0, "grad_norm": gnorm,
               "provenance": {"config_widths": cfg.get("widths"), "config_nparams": cfg.get("n_params"), "dataset": cfg.get("dataset"), "analysis_n": m, "exact_ocean_auth_Jv": {k: auth[k]["Jv_rms_sample"] for k in auth}},
               "validation": {"H_symmetry_rel": sym, "R_over_H": R_over_H, "G_over_H": G_over_H},
               "eig_summary": {"H_min": float(evH_c.min()), "H_max": float(evH_c.max()), "H_neg_frac": float((evH_c < -1e-8).mean()),
                               "G_min": float(evG_c.min()), "G_max": float(evG_c.max()), "kerG_dim": int(kerG_mask.sum())},
               "ocean_dims": {k: int(v.shape[1]) for k, v in oceans.items()},
               "coverage_sweep": spaces, "curvature_probes_supporting": curv,
               "runtime_s": {"total": time.time() - t0, "matrices": teig - tcon, "eig": time.time() - teig},
               "notes": "coverage=||Q_O^T Q_H||_F^2/dim(H_null) is the headline; inclusion+curvature are supporting; v'Hv~0 is NOT Hv~0; kerJX FILE on disk was stale (not flat for this checkpoint) and is NOT used."}
        json.dump(res, open(os.path.join(OUT, f"hgn_cov_w64_seed{sd}.json"), "w"), indent=1, default=str)   # mats saved before the coverage loop
        del UH, UG, evH, evG, oceans
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"s{sd} DONE L {L0:.3e} sym {sym:.1e} R/H {R_over_H:.4f} Hneg {res['eig_summary']['H_neg_frac']:.3f} kerG {res['eig_summary']['kerG_dim']} | {res['runtime_s']['total']:.0f}s", flush=True)
    print("HGN COVERAGE W64 DONE", flush=True)


if __name__ == "__main__":
    main()
