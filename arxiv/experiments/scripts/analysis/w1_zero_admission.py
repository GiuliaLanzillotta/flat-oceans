"""W1 candidate admission check for seeds 0 and 2 (directive A, 2026-11-16): establishes whether the W1
branch component is GENUINELY ZERO-DIMENSIONAL (zero admitted candidates) at these endpoints, under the
identical convention as the seed-1 admission (w1_admission_s1.py: robust candidate = autodiff JVP exposure
<= 1e-12), so that a zero W1 component is documented as a genuine zero rather than missing/uncomputed.

Per child (c0, c1): authenticate the stored W1 persistence factor fail-closed (checkpoint / preprocessing /
analysis-index identity, p / analysis-size); record declared rank and kernel dims; take the 8 SMALLEST-exposure
stored columns and cross-check their exposures against the ORIGINAL network by autodiff torch.func.jvp
(||J v||_X, fp64); admitted = #(jvp exposure <= 1e-12). Expected here: kernel_dims = 0 and min exposure
~1e-7 >> gate at every endpoint, hence 0 admitted. Any identity/agreement failure -> recorded + exit 1.
Writes results/branches/w1_admission_s<seed>.json (fork 350) or w1_admission_s<seed>_e<epoch>.json (other forks).
usage: w1_zero_admission.py <seed in {0,2}> [fork_epoch; default 350, {350} or (seed 0) {350,450}]"""

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
from flatocean import frozen_lib as FL
from flatocean import provenance as P
from torch.func import functional_call, jvp as fjvp

SEED = int(sys.argv[1]); assert SEED in (0, 2)
SPAWN = int(sys.argv[2]) if len(sys.argv) > 2 else 350
assert SPAWN == 350 or (SEED, SPAWN) == (0, 450), f"(seed, fork) = ({SEED}, {SPAWN}) is not an admitted cell"
GATE = 1e-12
failures = []
out = {"seed": SEED, "spawn": SPAWN, "gate_autodiff_exposure": GATE,
       "convention": "identical to w1_admission_s1.py (robust candidate = autodiff JVP exposure <= 1e-12); "
                     "run on the 8 smallest-exposure stored columns per child, which bound all others",
       "label": "zero admitted candidates => the W1 branch component is GENUINELY ZERO-DIMENSIONAL at these "
                "endpoints (a real zero under the admission convention, not missing/uncomputed); the certified "
                "reuse_bottom_up branch (W1 excluded, dim 0) is therefore the complete-layer construction here",
       "children": {}}

for CHILD in (0, 1):
    t0 = time.time(); tag = f"lmc_s{SEED}_e{SPAWN}_c{CHILD}"; name = f"mnist_mlp_{tag}_seed{SEED}"
    run = runs.load_run("mnist", "mlp", SEED, tag=tag, epoch=500, device="cpu", sets=("analysis",))
    rd = runs.run_dir("mnist", "mlp", SEED, tag)
    net = run["net"].eval().double(); X = run["data"]["analysis"][0].double(); s_n = X.shape[0]
    cur_ck = P.sha256_file(os.path.join(rd, "checkpoints", "epoch_0500.pt"))
    cur_pre = P.sha256_file(os.path.join(run["dir"], "preprocessing.pt"))
    cur_idx = P.sha256_index(run["data"]["analysis"][2])

    fp = os.path.join(HERE, "results", "persistence", name, "epoch0500_1.npz")
    z = np.load(fp, allow_pickle=True); ident = json.loads(str(z["identity"]))
    for fld, cur in [("checkpoint_sha256", cur_ck), ("preprocessing_sha256", cur_pre), ("analysis_index_sha256", cur_idx)]:
        if ident.get(fld) != cur:
            failures.append(f"c{CHILD}: identity {fld} mismatch (FAIL CLOSED)")
    p = int(z["p"]); ks = int(z["kernel_start"]); s_ = int(z["s"])
    W = net.layers[0].weight; o_, i_ = W.shape
    if p != o_ * i_ + o_ or s_ != s_n:
        failures.append(f"c{CHILD}: p/analysis_size mismatch (FAIL CLOSED)")
    sv = np.asarray(z["singular_values"], dtype=float)
    expo = np.concatenate([sv, np.zeros(max(0, p - sv.size))]) / np.sqrt(s_)
    order = np.argsort(expo); cols = np.sort(order[:8])

    # columns are stored in column-major [W b] coordinates; map to flat row-major [W rows; b]
    wsz = o_ * i_
    flat_w = torch.arange(wsz).view(o_, i_)
    perm = torch.cat([flat_w.T.reshape(-1), torch.arange(wsz, wsz + o_)])
    bsl = {nm: (a, b) for nm, a, b in FL.block_slices(net)}
    a0, _ = bsl["W1"]
    theta = FL.flat_params(net).detach().double()
    ps = list(FL.param_slices(net)); shp = {nm: pm.shape for nm, pm in net.named_parameters()}

    got = []
    for j in cols:
        vcol = torch.from_numpy(np.ascontiguousarray(z["V"][:, int(j):int(j) + 1])).double()[:, 0]
        v = torch.zeros_like(theta); v[a0 + perm] = vcol

        def F(t_):
            return functional_call(net, {nm: t_[a:b].view(shp[nm]) for nm, a, b in ps}, (X,))
        J = fjvp(F, (theta,), (v,))[1]
        got.append(float((J.pow(2).sum(-1).mean()).sqrt()))
    got = np.asarray(got); stored = expo[cols]
    sig0r = float(expo[0])
    agree = float(np.max(np.abs(got - stored)) / max(sig0r, 1e-300))
    if agree > 3e-4:
        failures.append(f"c{CHILD}: JVP-vs-stored disagreement {agree:.2e} (fp32 floor 3e-4)")
    n_adm = int((got <= GATE).sum())
    if n_adm != 0:
        failures.append(f"c{CHILD}: unexpectedly admitted {n_adm} candidates")
    out["children"][f"c{CHILD}"] = {
        "p": p, "declared_rank": ks, "declared_kernel_dims": p - ks,
        "min_stored_exposure": float(expo.min()), "smallest8_stored_exposure": stored.tolist(),
        "smallest8_jvp_exposure": got.tolist(), "jvp_agreement_over_sigma0rate": agree,
        "n_admitted": n_adm, "min_exposure_over_gate": float(expo.min() / GATE),
        "factor_sha256": P.sha256_file(fp), "runtime_s": round(time.time() - t0, 1)}
    print(f"c{CHILD}: kernel_dims {p-ks} min expo {expo.min():.3e} (= {expo.min()/GATE:.1e} x gate) "
          f"jvp agree {agree:.1e} admitted {n_adm} [{time.time()-t0:.0f}s]", flush=True)

out["passed"] = not failures; out["failures"] = failures
op = os.path.join(HERE, "results", "branches",
                  f"w1_admission_s{SEED}.json" if SPAWN == 350 else f"w1_admission_s{SEED}_e{SPAWN}.json")
json.dump(out, open(op, "w"), indent=1)
print(("ADMISSION RECORD PASS (zero candidates -> W1 component genuinely zero-dimensional)" if out["passed"]
       else "ADMISSION FAILURES: " + "; ".join(failures)), flush=True)
print("WROTE", op, flush=True)
sys.exit(0 if out["passed"] else 1)
