"""W1 candidate ADMISSION check, seed-1 endpoints (reviewer decision 5, 2026-11-11). Runs BEFORE any W1-unfrozen build.

For each child (c0, c1) of s1_e350: authenticate the stored W1 persistence factor (fail-closed on checkpoint /
preprocessing / analysis-index identity and p/analysis-size), verify the saved basis itself (orthonormality,
coordinate embedding via the detected bias-column convention), and validate EVERY candidate column against the
ORIGINAL network: analytic exposure (branch.Net operator) AND autodiff torch.func.jvp exposure, with per-column
agreement asserted. Admission: robust candidates = autodiff exposure <= 1e-12; expected counts are ASSERTED
(c0: 3899 robust + 1 boundary reported separately; c1: 4166 robust). These are NUMERICALLY VALIDATED KERNEL
CANDIDATES, not mathematically established exact kernels. Any failure -> recorded + exit 1 (builds must not start).
Writes results/branches/w1_admission_s1.json with the admitted column indices per child."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time
import numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); os.chdir(HERE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
from flatocean import runs
from flatocean.analyses import branch as B
from flatocean import provenance as P
from torch.func import functional_call, jvp as fjvp

SEED, SPAWN = 1, 350; GATE = 1e-12; ORTHO_TOL = 1e-8
EXPECT = {0: 3899, 1: 4166}
failures = []; out = {"seed": SEED, "spawn": SPAWN, "gate_autodiff_exposure": GATE,
                      "label": "numerically validated kernel candidates (W1); NOT mathematically established exact kernels",
                      "children": {}}

for CHILD in (0, 1):
    t0 = time.time(); tag = f"lmc_s{SEED}_e{SPAWN}_c{CHILD}"; name = f"mnist_mlp_{tag}_seed{SEED}"
    run = runs.load_run("mnist", "mlp", SEED, tag=tag, epoch=500, device="cpu", sets=("analysis",))
    rd = runs.run_dir("mnist", "mlp", SEED, tag)
    net = run["net"].eval().double(); X = run["data"]["analysis"][0].double(); nt = B.Net(net, X)
    n, m1 = nt.shape["W1"]; l = 0
    cur_ck = P.sha256_file(os.path.join(rd, "checkpoints", "epoch_0500.pt"))
    cur_pre = P.sha256_file(os.path.join(run["dir"], "preprocessing.pt"))
    cur_idx = P.sha256_index(run["data"]["analysis"][2])

    fp = os.path.join(HERE, "results", "persistence", name, "epoch0500_1.npz")
    z = np.load(fp, allow_pickle=True); ident = json.loads(str(z["identity"]))
    for fld, cur in [("checkpoint_sha256", cur_ck), ("preprocessing_sha256", cur_pre), ("analysis_index_sha256", cur_idx)]:
        if ident.get(fld) != cur:
            failures.append(f"c{CHILD}: identity {fld} mismatch (FAIL CLOSED)")
    if int(z["p"]) != n * m1 or int(ident["analysis_size"]) != nt.s:
        failures.append(f"c{CHILD}: p/analysis_size mismatch (FAIL CLOSED)")

    Vk = torch.from_numpy(np.asarray(z["V_kernel64"], dtype=np.float64)); k = Vk.shape[1]
    Vrow = Vk.reshape(m1, n, k).transpose(0, 1).reshape(n * m1, k).contiguous()
    ortho = float((Vrow.T @ Vrow - torch.eye(k, dtype=torch.float64)).abs().max()) if k else 0.0
    if ortho > ORTHO_TOL:
        failures.append(f"c{CHILD}: orthonormality {ortho:.2e} > {ORTHO_TOL:.0e}")

    # coordinate embedding: the [W b] layout's bias column must be the constant-1 column of H[0]
    Hl = nt.H[l]
    const_cols = [j for j in range(Hl.shape[1]) if torch.allclose(Hl[:, j], torch.ones_like(Hl[:, j]))]
    if const_cols != [m1 - 1]:
        failures.append(f"c{CHILD}: bias-column convention violated (constant cols {const_cols})")

    lin = net.layers[l]
    wname = next(nm for nm, pm in net.named_parameters() if pm is lin.weight)
    bname = next(nm for nm, pm in net.named_parameters() if pm is lin.bias)
    base = {nm: pm.detach() for nm, pm in net.named_parameters()}

    def expo_analytic(v):
        A = v.reshape(n, m1)
        dz = torch.einsum("nm,im->in", A, nt.H[l])
        return float((torch.einsum("idn,in->id", nt.D[l], dz).pow(2).sum(1).mean()).sqrt())

    def expo_autodiff(v):
        A = v.reshape(n, m1); dW = A[:, :-1].contiguous(); db = A[:, -1].contiguous()
        def G(w, b):
            pr = dict(base); pr[wname] = w; pr[bname] = b
            return functional_call(net, pr, (X,))
        _, tang = fjvp(G, (base[wname], base[bname]), (dW, db))
        return float((tang.pow(2).sum(1).mean()).sqrt())

    e_ad = np.empty(k); e_an = np.empty(k); agree_fail = 0
    for i in range(k):
        v = Vrow[:, i]
        e_an[i] = expo_analytic(v); e_ad[i] = expo_autodiff(v)
        if abs(e_an[i] - e_ad[i]) > 1e-12 + 1e-6 * e_ad[i]:
            agree_fail += 1
    if agree_fail:
        failures.append(f"c{CHILD}: analytic-vs-autodiff disagreement on {agree_fail}/{k} columns")

    robust = np.where(e_ad <= GATE)[0]; boundary = np.where(e_ad > GATE)[0]
    if len(robust) != EXPECT[CHILD]:
        failures.append(f"c{CHILD}: robust count {len(robust)} != expected {EXPECT[CHILD]}")
    out["children"][f"c{CHILD}"] = {
        "factor": fp, "factor_sha256": P.sha256_file(fp), "stored_kernel_cols": int(k),
        "orthonormality_maxdev": ortho, "bias_column_ok": const_cols == [m1 - 1],
        "autodiff_exposure_max_robust": float(e_ad[robust].max()) if len(robust) else None,
        "admitted_robust_indices": robust.tolist(), "n_robust": int(len(robust)),
        "boundary_reported_separately": [{"col": int(i), "autodiff": float(e_ad[i]), "analytic": float(e_an[i])}
                                          for i in boundary.tolist()],
        "agreement_failures": agree_fail, "runtime_s": round(time.time() - t0, 1)}
    print(f"c{CHILD}: k={k} robust={len(robust)} (expected {EXPECT[CHILD]}) boundary={len(boundary)} "
          f"ortho={ortho:.1e} max robust autodiff expo={out['children'][f'c{CHILD}']['autodiff_exposure_max_robust']} "
          f"({out['children'][f'c{CHILD}']['runtime_s']}s)", flush=True)

out["failures"] = failures; out["passed"] = not failures
outp = os.path.join(B.OUT, "w1_admission_s1.json")
json.dump(out, open(outp, "w"), indent=2)
print(("ADMISSION PASS" if not failures else "ADMISSION FAIL:\n  " + "\n  ".join(failures)), flush=True)
print("WROTE", outp, flush=True)
sys.exit(0 if not failures else 1)
