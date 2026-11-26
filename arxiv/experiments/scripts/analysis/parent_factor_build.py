"""MLP PARENT factor build at the fork checkpoints (two-day completion plan 2026-11-22, "MLP parents and fork
time completion").  Parents = the BASE width-128 runs mnist_mlp_seed{0,1,2} AT THE FORK EPOCHS: (s0,e350),
(s1,e350), (s2,e350), (s0,e450).  Builds ONLY the missing per-block SVD factors (blocks 1, 2, 3, readout --
W1 INCLUDED, p = 32896) by calling the CANONICAL builder verbatim: flatocean.analyses.persistence.factor_block
(the extracted body of the `kernel-persistence factor` phase that produced every stored
results/persistence/<run>/epochXXXX_<block>.npz, identical protocol identifier kernel-persistence-v1, identity
manifest P.analysis_identity, rank rule sigma > 1e-9 sigma_1, gesdd driver with the >26k openblas64 path for W1).
Storage follows factor_path(): results/persistence/mnist_mlp_seed<s>/epoch0350_<blk>.npz (resp. epoch0450) --
the PARENT's own epoch tag under the base run's dir (the children's factors say epoch0500 because the children
END at 500; a parent factor at the fork is a different checkpoint of a different run).  A factor that already
exists is authenticated fail-closed (factor_status) and reused, never rebuilt.  After each build the file is
re-authenticated through factor_spectrum.

usage:  python3 parent_factor_build.py <seed> <epoch> [--dry]
  --dry: NO build, NO run loading.  Verifies the fork checkpoint exists (sha256 recorded), lightly authenticates
  any existing factor (stored epoch/block/p + identity checkpoint_sha256 against the actual checkpoint file),
  lists the missing blocks and prints honest memory/time estimates (SVD workspace 8 n^2 * 3 bytes; Jacobian
  assembly 8 s d p bytes).  Exit 0 iff all checks pass.
HEAVY in real mode (W1 SVD ~26 GB workspace + ~26 GB assembly; declare JOB_MEM_GB>=60, 16 threads)."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time
import numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); os.chdir(HERE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
from flatocean import provenance as P
from flatocean import runs
from flatocean import factors as F
from flatocean.analyses import persistence as PERS

PARENTS = {(0, 350), (1, 350), (2, 350), (0, 450)}
BLOCKS = ["2", "3", "readout", "1"]            # cheap blocks first; the heavy W1 (block "1") SVD last
SEED, EP = int(sys.argv[1]), int(sys.argv[2]); DRY = "--dry" in sys.argv[3:]
assert (SEED, EP) in PARENTS, f"(seed, epoch) must be one of {sorted(PARENTS)}"
name = f"mnist_mlp_seed{SEED}"; rd = runs.run_dir("mnist", "mlp", SEED, "")
ck_path = os.path.join(rd, "checkpoints", f"epoch_{EP:04d}.pt")


def block_sizes():
    """(p, n, m1) per block and (s, d) from the checkpoint state dict and the data manifest (no run load)."""
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    Ws = [sd[k] for k in sorted(sd) if k.endswith("weight") and "readout" not in k]
    Vw = next(sd[k] for k in sd if "readout" in k and k.endswith("weight"))
    man = json.load(open(os.path.join(rd, "data_manifest.json")))
    s = len(man["analysis_indices"]); d = Vw.shape[0]
    out = {}
    for l, W in enumerate(Ws):
        out[str(l + 1)] = (W.shape[0] * (W.shape[1] + 1), W.shape[0], W.shape[1] + 1)
    out["readout"] = (Vw.shape[0] * (Vw.shape[1] + 1), Vw.shape[0], Vw.shape[1] + 1)
    return out, s, d


def dry():
    fails = []
    if not os.path.exists(ck_path):
        print(f"DRY FAIL: fork checkpoint missing: {ck_path}", flush=True); sys.exit(1)
    ck_sha = P.sha256_file(ck_path)
    print(f"== DRY {name} epoch {EP}: checkpoint {ck_path} sha256 {ck_sha[:16]}... EXISTS", flush=True)
    sizes, s, d = block_sizes()
    missing = []
    for blk in BLOCKS:
        p, n, m1 = sizes[blk]
        path = F.factor_path(name, EP, blk)
        if os.path.exists(path):
            try:
                z = np.load(path, allow_pickle=True)
                ident = json.loads(str(z["identity"]))
                ok = (int(z["epoch"]) == EP and str(z["block"]) == blk and int(z["p"]) == p
                      and "V" in z.files and ident.get("checkpoint_sha256") == ck_sha)
                if ok:
                    print(f"   {blk}: factor EXISTS and light-authenticates (p={p}, kernel dim {p - int(z['kernel_start'])}); "
                          f"full factor_status authentication runs in the real build", flush=True)
                else:
                    fails.append(f"{blk}: existing factor FAILS light authentication (epoch/block/p/checkpoint_sha256)")
            except Exception as e:
                fails.append(f"{blk}: existing factor unreadable: {e}")
            continue
        svd_gb = 8 * p * p * 3 / 1e9; asm_gb = 8 * s * d * p / 1e9
        # time: QR ~2 (sd) p^2 flops + gesdd-with-vectors ~20 p^3 flops at an effective ~5e10 flop/s on 16 threads
        t_est = (2 * s * d * p * p + 20 * p ** 3) / 5e10
        missing.append(blk)
        print(f"   {blk}: MISSING -> build p={p} (n={n}, m1={m1}); est SVD workspace 8n^2*3 = {svd_gb:.1f} GB, "
              f"assembly 8*s*d*p = {asm_gb:.1f} GB (s={s}, d={d}); rough time ~{t_est/60:.0f}-{3*t_est/60:.0f} min "
              f"@16 threads", flush=True)
    try:
        from flatocean.analyses.persistence import factor_block  # noqa: F401  (builder reuse import check)
        print("   builder reuse import OK: flatocean.analyses.persistence.factor_block", flush=True)
    except Exception as e:
        fails.append(f"builder import failed: {e}")
    print(f"== DRY summary {name} e{EP}: missing blocks {missing or 'NONE'}"
          + (f"; FAILURES: {fails}" if fails else ""), flush=True)
    sys.exit(1 if fails else 0)


def main():
    if not os.path.exists(ck_path):
        print(f"FAIL: fork checkpoint missing: {ck_path}", flush=True); sys.exit(1)
    t0 = time.time()
    run = runs.load_run("mnist", "mlp", SEED, tag="", epoch=EP, device="cpu", sets=("analysis",))
    for blk in BLOCKS:
        PERS.factor_block(run, rd, name, EP, blk, "cpu")        # canonical builder; skips+authenticates existing
        sp = F.factor_spectrum(name, EP, blk, run, rd)          # post-build fail-closed authentication
        assert sp is not None, f"{blk}: factor file missing after build"
        sv, p, s_, sha, status = sp
        print(f"   {blk}: authenticated ({status}) p={p} s={s_} sha256 {sha[:16]}...", flush=True)
    print(f"== {name} epoch {EP}: all {len(BLOCKS)} parent factors present and authenticated "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    dry() if DRY else main()
