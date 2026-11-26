"""Build + authenticate the 10,000-example input cache for the COMMON ResNet family study
(two-day plan 2026-11-22, "a correctly authenticated 10k cache"; previously measured 4k results stay
separate and untouched).

What is cached (results/resnet/cache10k/, NEW directory):
  * xin_fp32.npy            layer4 input activations = layer3 output at theta0 for the first 10,000
                            RESISC45 train images (N,1024,14,14).  STORAGE CHOICE (recorded): float32 on
                            disk (~8 GB; float64 would be ~16 GB); every consumer promotes chunks to
                            float64 on read and forms ALL operators in float64
                            (resnet_common_lib.open_xin / xin_chunk is the fp64-read path).  The fp32
                            file is the DEFINED study input: downstream fixed quantities below are
                            computed in float64 FROM the fp32-rounded activations, so the frozen-forward
                            r=0 identity holds at fp64 machine precision against this cache.
  * ref_logits_local_fp64.pt   F(theta0) of the study: float64 logits of the layer4+fc ACTUAL local
                            forward (FrozenLayer4Head path) applied to the fp32-rounded X_in.  This is
                            the base-output reference for every drift/normalization denominator.
  * ref_logits_fullnet_fp64.pt float64 logits of the ORIGINAL full-network fp64 forward (unrounded
                            internal activations), kept to record the fp32-storage rounding effect.
  * gate_census.pt          per-channel base-gate active counts (frozen-forward gate convention) for
                            every gate layer of blocks 4.0-4.2, from the cached X_in: the selective
                            (SEL) candidate screen input.  Counts only; the complete selective-space
                            computation lives in the family study.
  * cache_identity.json     SHA-stamped identity: checkpoint bytes, source image array, preprocessing,
                            ordered example indices, storage dtype, file SHAs, rounding-effect record.

Checkpoint/data/preprocessing conventions are those of scripts/analysis/resnet_frozen_forward.py
(= resnet_cand_val_v2.py): seed-0 fine-tuned ResNet-50/RESISC45 checkpoint, first N train images,
torchvision IMAGENET1K_V2 eval transforms, fp64 full-network forward in eval mode.

Launch (light GPU job, ~10-20 min, JOB_MEM_GB ~ 20):
    run_light resnet_cache10k 56-63 env CUDA_VISIBLE_DEVICES=1 python3 scripts/analysis/resnet_cache10k.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, argparse, datetime
import numpy as np
import torch

HERE = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from resnet_common_lib import (CACHE_DIR, NCLS, sha256_file, sha256_index, load_model, CKPT,
                               FrozenLayer4Head, block_states, base_gates_of_states, run_block,
                               x_norm, meta_block)
from ftvit_paths import data_path
from torchvision.models import ResNet50_Weights


def main():
    ap = argparse.ArgumentParser(description="ResNet 10k input cache for the common family study")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out-dir", default=CACHE_DIR)
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    net, ck = load_model(dev)
    net = net.to(torch.float64)
    tfm = ResNet50_Weights.IMAGENET1K_V2.transforms()
    imgs_path = data_path("resisc45_train_imgs.npy")
    imgs = np.load(imgs_path, mmap_mode="r")
    n = min(args.n, imgs.shape[0])
    idx = np.arange(n, dtype=np.int64)                       # ordered example identifiers: first n, order fixed

    # ---- pass 1: fp64 full-network forward; capture layer3 output (layer4 input) + full-net logits ----
    xin_path = os.path.join(args.out_dir, "xin_fp32.npy")
    xin_mm = np.lib.format.open_memmap(xin_path, mode="w+", dtype=np.float32, shape=(n, 1024, 14, 14))
    ref_full = torch.zeros(n, NCLS, dtype=torch.float64)
    h = {}
    hnd = net.layer3.register_forward_hook(lambda m, i, o: h.__setitem__("x", o.detach()))
    with torch.no_grad():
        for i in range(0, n, args.batch):
            jhi = min(i + args.batch, n)
            b = np.asarray(imgs[i:jhi])
            x = torch.from_numpy(b).to(dev).float().div(255).permute(0, 3, 1, 2).contiguous()
            x = tfm(x).to(torch.float64)
            ref_full[i:jhi] = net(x).cpu()
            xin_mm[i:jhi] = h["x"].to(torch.float32).cpu().numpy()
            if i % 1000 == 0:
                print(f"forward {i}/{n} ({time.time() - t0:.0f}s)", flush=True)
    hnd.remove()
    xin_mm.flush()
    del xin_mm
    print(f"pass 1 done ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 2: fp64 local actual forward FROM the fp32-rounded X_in (the study's F(theta0)),
    #              plus the base-gate census of blocks 4.0-4.2 ----
    head = FrozenLayer4Head(net, dtype=torch.float64, device=dev)
    xin = np.load(xin_path, mmap_mode="r")
    ref_local = torch.zeros(n, NCLS, dtype=torch.float64)
    census = {f"b{b}_{g}": None for b in range(3) for g in ("g1", "g2", "gout")}
    counts = {}
    n_pos = {}
    for i in range(0, n, 100):
        jhi = min(i + 100, n)
        xb = torch.from_numpy(np.asarray(xin[i:jhi])).to(device=dev, dtype=torch.float64)
        hcur = xb
        for b in range(3):
            st = block_states(head, hcur, b)
            g1, g2, go = base_gates_of_states(head, st, b)
            for tag, g in (("g1", g1), ("g2", g2), ("gout", go)):
                key = f"b{b}_{tag}"
                c = g.sum((0, 2, 3)).to(torch.int64).cpu()       # per-channel active count over (i, xi)
                counts[key] = c if key not in counts else counts[key] + c
                n_pos[key] = g.shape[2] * g.shape[3]
            hcur = st["out"]
        ref_local[i:jhi] = (hcur.mean((2, 3)) @ head.theta0["fc.weight"].T + head.theta0["fc.bias"]).cpu()
    census = {k: counts[k] for k in counts}
    rounding = ref_local - ref_full
    scale = float(ref_full.abs().max())
    rounding_rec = {
        "max_abs": float(rounding.abs().max()), "max_rel": float(rounding.abs().max()) / scale,
        "rms_rel_X": x_norm(rounding) / max(x_norm(ref_full), 1e-300),
        "note": "fp64 local forward from fp32-stored X_in vs fp64 full-network forward (unrounded): "
                "pure fp32-storage rounding effect; the cache DEFINES the study input, so "
                "ref_logits_local_fp64 is the base reference for all common-study measurements"}
    print("fp32-storage rounding effect:", json.dumps(rounding_rec), flush=True)

    torch.save(ref_local, os.path.join(args.out_dir, "ref_logits_local_fp64.pt"))
    torch.save(ref_full, os.path.join(args.out_dir, "ref_logits_fullnet_fp64.pt"))
    torch.save({"counts": census, "n_examples": n, "spatial_positions": n_pos,
                "total_patches": {k: int(n * n_pos[k]) for k in n_pos},
                "convention": "frozen-forward base gates 1{u>0} after eval-BN affine / residual addition "
                              "(resnet_common_lib.base_gates_of_states), from the cached fp32 X_in read "
                              "in fp64; SEL candidate screen only (census != selective-space computation)"},
               os.path.join(args.out_dir, "gate_census.pt"))

    # ---- identity (SHA-stamped, fail-closed consumers recompute these) ----
    files = {}
    for key, name in (("xin", "xin_fp32.npy"), ("ref_local", "ref_logits_local_fp64.pt"),
                      ("ref_fullnet", "ref_logits_fullnet_fp64.pt"), ("census", "gate_census.pt")):
        p = os.path.join(args.out_dir, name)
        files[key] = {"name": name, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    ident = {
        "cache": "common ResNet family study input cache, N=10000 (two-day plan 2026-11-22)",
        "n": int(n), "indices": "arange(0, n) of the train array (first n, stored order)",
        "index_sha256": sha256_index(idx),
        "checkpoint": {"path": CKPT, "sha256": sha256_file(CKPT),
                       "val_acc": ck.get("val_acc"), "test_acc": ck.get("test_acc")},
        "images": {"path": imgs_path, "sha256": sha256_file(imgs_path),
                   "shape": list(imgs.shape), "dtype": str(imgs.dtype)},
        "preprocessing": {"transform": "torchvision ResNet50_Weights.IMAGENET1K_V2.transforms()",
                          "repr": repr(tfm).replace("\n", " "),
                          "pipeline": "uint8 -> float/255 -> NCHW -> transforms (resize 232, center-crop "
                                      "224, ImageNet mean/std) -> float64 full-net eval forward"},
        "storage": {"xin_dtype": "float32",
                    "rationale": "10000 x 1024 x 14 x 14 fp32 = 8.0 GB on disk (fp64 would be 16 GB); "
                                 "consumers use the fp64-read path (resnet_common_lib.open_xin/xin_chunk) "
                                 "and form all operators in float64. The fp32 file is the defined study "
                                 "input; see rounding_effect for the (recorded) difference to the "
                                 "unrounded fp64 forward.",
                    "forward_dtype": "float64 (full net and local layer4+fc path)"},
        "rounding_effect": rounding_rec,
        "files": files,
        "base_output": {"ref_local_rms_X": x_norm(ref_local), "ref_fullnet_rms_X": x_norm(ref_full)},
        "meta": meta_block("scripts/analysis/resnet_cache10k.py",
                           {"device": str(dev), "elapsed_s": round(time.time() - t0, 1),
                            "batch": args.batch}),
    }
    with open(os.path.join(args.out_dir, "cache_identity.json"), "w") as f:
        json.dump(ident, f, indent=1)
    print("WROTE", os.path.join(args.out_dir, "cache_identity.json"))
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
