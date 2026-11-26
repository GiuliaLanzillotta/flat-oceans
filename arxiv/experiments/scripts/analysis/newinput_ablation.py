"""SVHN evaluation with saved MNIST preprocessing. Usage: prepare | run <seed> | gate."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from flatocean.paths import ROOT as PROJECT_ROOT, dataset
import os, sys, json, time, hashlib
import numpy as np, torch
HERE = str(PROJECT_ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
os.chdir(HERE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
from flatocean import tier1_train as T
from flatocean import ocean_lib as O
from flatocean.analyses import persistence as KP
from flatocean import provenance as P
NI = os.path.join(HERE, "results", "newinput"); os.makedirs(NI, exist_ok=True)
OUTA = os.path.join(HERE, "results", "atomic")
TOL = "1e-09"; CONDS = ["svhn"]
MAN = os.path.join(NI, "manifest.json")


def sha_t(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy() if torch.is_tensor(t) else t).tobytes()).hexdigest()


def prepare():
    Xtr, ytr, Xte, yte, hashes = T.load_raw("mnist")
    assert Xte.shape == (10000, 784) and Xte.dtype == np.uint8
    import scipy.io as sio
    m = sio.loadmat(str(dataset("SVHN") / "test_32x32.mat"))
    Xs = m["X"]                                             # (32, 32, 3, N) uint8, natural file order
    sel = np.arange(10000)                                  # FIRST 10,000 in natural order (fixed, saved)
    img = torch.from_numpy(np.ascontiguousarray(Xs[:, :, :, sel]).astype(np.float64) / 255.0).permute(3, 2, 0, 1)
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]            # BT.601 luma, (N, 32, 32)
    g28 = torch.nn.functional.adaptive_avg_pool2d(gray.unsqueeze(1), (28, 28)).squeeze(1)
    svhn01 = g28.reshape(10000, 784).clamp(0.0, 1.0).contiguous()
    torch.save(svhn01, os.path.join(NI, "svhn_gray28_01.pt"))
    np.save(os.path.join(NI, "svhn_selection_indices.npy"), sel)
    man = {"spec": "SVHN first 10000 test images, BT.601 grayscale, adaptive-avg-pool 32->28; saved MNIST preprocessing",
           "mnist_raw_sha": hashes,
           "svhn_file": str(dataset("SVHN") / "test_32x32.mat"),
           "svhn_file_sha256": T.sha256(str(dataset("SVHN") / "test_32x32.mat")),
           "svhn_selection": "first 10000, natural order", "svhn_gray28_sha256": sha_t(svhn01),
           "clean_arm": "REUSED authenticated reference: results/atomic/*_new_batch.json (batch 'test')"}
    json.dump(man, open(MAN, "w"), indent=1)
    print("prepared:", {k: v for k, v in man.items() if k.endswith("sha256")}, flush=True)


def cond_raw01(cond):
    return torch.load(os.path.join(NI, "svhn_gray28_01.pt"), weights_only=False)


def run_seed(seed):
    man = json.load(open(MAN))
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
    out_path = os.path.join(OUTA, f"{name}_newinput.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {
        "protocol": "newinput-exposure-v1 (PI spec 2026-11-16; new-batch-exposure layer coverage, rank rule 1e-9)",
        "rank_tolerance": TOL, "epoch": 500, "manifest": man, "conditions": {}}
    for cond in CONDS:
        if cond in res["conditions"]:
            print(f"== {name} {cond}: already recorded, skipping", flush=True); continue
        t0 = time.time()
        raw01 = cond_raw01(cond)
        Y = T.apply_preprocessing((raw01 * 255.0).numpy(), pre, torch.float64)   # /255 inside reproduces raw01 to ~1e-16
        X = torch.cat([Xa, Y], 0); s = X.shape[0]
        st = O.mlp_structure(run["net"], X)
        recs = {}; spectra = {}
        for l in (1, 2):
            rec, sv, _, _ = O.atomic_layer(st, l, s, "cpu")
            recs[str(l + 1)] = rec; spectra[str(l + 1)] = np.asarray(sv)
        rec, sv, _ = O.atomic_readout(st, s, "cpu")
        recs["readout"] = rec; spectra["readout"] = np.asarray(sv)
        np.savez(os.path.join(NI, f"{name}_{cond}_spectra.npz"), **spectra)
        summary = {}
        for blk, r in recs.items():
            sv0, p_, s0_, sha, status = base_spec[blk]
            lad = O.rank_ladder(sv0); rank_old = lad[TOL]; dimK_old = p_ - rank_old
            cn = r[f"components@{TOL}"]
            exposed = r["rank_J"][TOL] - rank_old
            summary[blk] = {"dim_K_X": dimK_old, "dim_K_union": cn["dim_K"],
                            "exposed": exposed, "exposed_fraction": (exposed / dimK_old if dimK_old else None),
                            "components_union": cn, "gap_union": r["gap_J"][TOL], "gap_note": "rank resolved iff gap >= 10",
                            "baseline_factor_sha256": sha, "baseline_status": status, "seconds": r.get("seconds")}
            print(f"    {cond} blk {blk}: dimK_X {dimK_old} -> union {cn['dim_K']} exposed {exposed} "
                  f"({(exposed/dimK_old if dimK_old else 0):.3f}) gap {r['gap_J'][TOL]:.1f}", flush=True)
        ident = P.analysis_identity(run, rd, 500, {"protocol": "newinput-exposure-v1", "condition": cond,
                                                    "input_sha256": (man["noisy_sha256"][cond[5:]] if cond.startswith("noisy")
                                                                     else man["svhn_gray28_sha256"]),
                                                    "noise_z_sha256": man["z_sha256"],
                                                    "baseline_factor_sha256": {b: base_spec[b][3] for b in base_spec}})
        res["conditions"][cond] = {"identity": ident, "summary": summary, "seconds": round(time.time() - t0, 1)}
        json.dump(res, open(out_path, "w"), indent=1, default=str)
        print(f"== {name} {cond} done in {res['conditions'][cond]['seconds']}s; record updated", flush=True)
    print("WROTE", out_path, flush=True)


def gate():
    """Seed-0 validation before the other seeds are queued."""
    fails = []
    p0 = os.path.join(OUTA, "mnist_mlp_seed0_newinput.json")
    if not os.path.exists(p0):
        print("GATE FAIL: seed-0 record missing"); sys.exit(1)
    d = json.load(open(p0))
    nb = json.load(open(os.path.join(OUTA, "mnist_mlp_seed0_new_batch.json")))
    for cond in CONDS:
        c = d["conditions"].get(cond)
        if not c:
            fails.append(f"{cond}: missing"); continue
        if not c.get("identity", {}).get("digest"):
            fails.append(f"{cond}: identity digest missing")
        for blk, e in c["summary"].items():
            if e["dim_K_union"] > e["dim_K_X"]:
                fails.append(f"{cond}/{blk}: dim_K_union > dim_K_X")
            if e["exposed"] < 0:
                fails.append(f"{cond}/{blk}: negative exposure")
            if e["gap_union"] is not None and e["gap_union"] < 10:
                fails.append(f"{cond}/{blk}: union rank gap {e['gap_union']:.1f} < 10 (UNRESOLVED)")
    # baseline consistency with the existing clean-arm record (same authenticated baseline)
    try:
        tb = nb["stages"]["late_endpoint"]["batches"]["test"]["summary"]
        for blk in ("2", "3", "readout"):
            a = d["conditions"][CONDS[0]]["summary"][blk]["dim_K_X"]; b = tb[blk]["dim_K_old"]
            if a != b:
                fails.append(f"baseline dim_K_X mismatch blk {blk}: {a} != existing record {b}")
    except KeyError as e:
        fails.append(f"clean-arm cross-check unavailable ({e}); verify manually")
    rep = {"passed": not fails, "failures": fails}
    json.dump(rep, open(os.path.join(NI, "seed0_gate.json"), "w"), indent=1)
    print(("GATE PASS" if not fails else "GATE FAIL:\n  " + "\n  ".join(fails)), flush=True)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        prepare()
    elif mode == "gate":
        gate()
    elif mode == "run":
        run_seed(int(sys.argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode}")
