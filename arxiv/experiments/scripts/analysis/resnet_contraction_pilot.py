"""DOWNSTREAM-CONTRACTION PILOT for the common ResNet family study (two-day plan 2026-11-22).
This pilot GATES the l42fc/l41/l40 production groups.

Scope (bounded): layer4.2.conv3 (cheapest downstream) + the fc readout, on the authenticated 10k cache.

Verifies, with recorded numbers:
  (a) frozen-forward reuse: r=0 identity of the validated frozen forward (FrozenLayer4Head, fixed base
      gates) against the cache's fp64 base reference on ALL 10,000 examples, and cross-reference to the
      existing record results/resnet/frozen_forward_validation.json (PASS required);
  (b) D validation: the batched 45-VJP backward contraction rows g[a,i,c,xi] = e_a^T D_(i,xi) agree with
      independent forward-mode autograd JVPs through the real remaining network on >= 8 sampled
      (example, position, channel) triples, at fp64 precision (sampled half among ACTIVE columns, half
      uniformly: gated-off positions have exactly-zero columns in both routes);
  (c) reproducibility: the H (patch) and C (downstream marginal) Gram spectra from two independent chunk
      orders agree to ~1e-12 relative.

Also benchmarks (for the production release decision): contraction cost/example and peak GPU memory at
layer4.2.conv3, timing probes of the g-pass at layer4.2.conv1 and layer4.0.conv1, fp32 batched battery
forwards from those layers, and a projected per-group runtime/memory table.

The downstream marginal Gram is sum_(i,xi) D_(i,xi)^T D_(i,xi) in ACTIVATION space (n x n), computed with
convolutional weight sharing respected -- NOT the parameter-space Gram and NOT the withdrawn full-GGN
route. For the readout, D = I (structural statement; no contraction needed).

Writes results/resnet/contraction_pilot.json (+ _arrays.pt with the pilot spectra). NEW files only.

Launch (light GPU job, ~20-40 min, JOB_MEM_GB ~ 16):
    run_light resnet_contraction_pilot 56-63 env CUDA_VISIBLE_DEVICES=1 \
        python3 scripts/analysis/resnet_contraction_pilot.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from resnet_common_lib import (CACHE_DIR, RESULTS_DIR, NCLS, RANK_REL, RATE_ABS, authenticate_cache,
                               open_xin, xin_chunk, load_model, FrozenLayer4Head, states_to_block,
                               block_states, base_gates_of_states, downstream_g, conv_patches,
                               im2col_consistency, tail_from_z, rank_gap_record, x_norm, meta_block,
                               stable_seed, _aff)


def spectra_rel_diff(GA, GB):
    evA = torch.linalg.eigvalsh(GA.cpu())
    evB = torch.linalg.eigvalsh(GB.cpu())
    mx = float(evA.abs().max())
    return float((evA - evB).abs().max()) / max(mx, 1e-300), evA


def accumulate_l42conv3(head, xin, order, chunk, dev, with_H=True, timing=None):
    """One full pass over the cache in the given chunk order: downstream marginal Gram Gd (2048x2048),
    patch Gram Gh (512x512) for layer4.2.conv3, pooled-feature Gram Gf (2049x2049) for fc, base logits."""
    n = xin.shape[0]
    Gd = torch.zeros(2048, 2048, dtype=torch.float64, device=dev)
    Gh = torch.zeros(512, 512, dtype=torch.float64, device=dev)
    Gf = torch.zeros(2049, 2049, dtype=torch.float64, device=dev)
    base = torch.zeros(n, NCLS, dtype=torch.float64)
    npat = 0
    tg = 0.0
    starts = list(range(0, n, chunk))
    if order == "reversed":
        starts = starts[::-1]
    for i0 in starts:
        i1 = min(i0 + chunk, n)
        xb = xin_chunk(xin, i0, i1, dev)
        x2 = states_to_block(head, xb, 2)
        st = block_states(head, x2, 2)
        t0 = time.time()
        g, lg0 = downstream_g(head, st, 2, 3)                   # (45, B, 2048, 49)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        tg += time.time() - t0
        rows = g.permute(0, 1, 3, 2).reshape(-1, 2048)          # rows e_a^T D_(i,xi)
        Gd += rows.T @ rows
        if with_H:
            P = conv_patches(head, st, 2, 3)                    # (B, 512, 49) -- 1x1 conv: patches = a2
            Pm = P.permute(0, 2, 1).reshape(-1, 512)
            Gh += Pm.T @ Pm
            npat += Pm.shape[0]
            pooled = st["out"].mean((2, 3))
            aug = torch.cat([pooled, torch.ones(pooled.shape[0], 1, dtype=pooled.dtype, device=dev)], 1)
            Gf += aug.T @ aug
        base[i0:i1] = lg0.cpu()
    if timing is not None:
        timing["g_pass_s"] = tg
    return Gd, Gh, Gf, base, npat


def main():
    ap = argparse.ArgumentParser(description="ResNet downstream-contraction pilot (gates production)")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-triples", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "contraction_pilot.json"))
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    t_start = time.time()
    rep = {"pilot": "ResNet downstream-contraction pilot: layer4.2.conv3 + fc on the authenticated 10k "
                    "cache (gates l42fc/l41/l40 production)"}

    # ---- 0. authenticate the cache (fail closed) --------------------------------------------------
    t0 = time.time()
    ident, auth = authenticate_cache(args.cache_dir)
    rep["cache_identity"] = {"n": ident["n"], "index_sha256": ident["index_sha256"],
                            "checkpoint_sha256": ident["checkpoint"]["sha256"],
                            "xin_sha256": ident["files"]["xin"]["sha256"],
                            "storage": ident["storage"], "auth_elapsed_s": round(time.time() - t0, 1),
                            "authentication": auth}
    print(f"cache authenticated ({time.time() - t0:.0f}s)", flush=True)
    xin, n = open_xin(args.cache_dir)
    ref_local = torch.load(os.path.join(args.cache_dir, ident["files"]["ref_local"]["name"]))
    net, ck = load_model(dev)
    head = FrozenLayer4Head(net, dtype=torch.float64, device=dev)
    scale = float(ref_local.abs().max())
    logits_rms = x_norm(ref_local)

    # ---- (a) frozen-forward reuse: r=0 identity on all n examples ---------------------------------
    t0 = time.time()
    max_abs = 0.0
    for i0 in range(0, n, args.chunk):
        i1 = min(i0 + args.chunk, n)
        xb = xin_chunk(xin, i0, i1, dev)
        gates, base_loc = head.base_gates(xb)
        F0 = head.frozen(xb, gates)
        max_abs = max(max_abs, float((F0 - ref_local[i0:i1].to(dev)).abs().max()))
        max_abs = max(max_abs, float((F0 - base_loc).abs().max()))
    ffv_path = os.path.join(RESULTS_DIR, "frozen_forward_validation.json")
    ffv = json.load(open(ffv_path))
    rep["a_frozen_r0_identity"] = {
        "n": n, "max_abs": max_abs, "max_rel": max_abs / scale,
        "pass": bool(max_abs / scale <= 1e-12),
        "note": "frozen forward (fixed base gates) at r=0 from the cached fp32 X_in (fp64-read path) vs "
                "the cache's fp64 local base reference AND vs the same-pass actual local forward",
        "frozen_forward_validation_record": {"path": ffv_path, "PASS": ffv.get("PASS"),
                                             "frozen_base_vs_original_fullnet": ffv["frozen_base_vs_original_fullnet"],
                                             "ckpt": ffv["meta"]["ckpt"]},
        "elapsed_s": round(time.time() - t0, 1)}
    assert ffv.get("PASS") is True, "frozen_forward_validation.json must be PASS (fail closed)"
    print(f"(a) r=0 identity: max_rel {max_abs / scale:.3e} ({time.time() - t0:.0f}s)", flush=True)

    # ---- contraction pass 1 (forward chunk order) + benchmark -------------------------------------
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timing = {}
    t0 = time.time()
    Gd_A, Gh_A, Gf_A, base_A, npat = accumulate_l42conv3(head, xin, "forward", args.chunk, dev, timing=timing)
    t_pass = time.time() - t0
    base_err = float((base_A - ref_local).abs().max()) / scale
    assert base_err <= 1e-13, f"g-pass base logits mismatch vs cache reference: {base_err:.3e}"
    mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else None
    rep["benchmark_l42conv3"] = {
        "chunk": args.chunk, "full_pass_s": round(t_pass, 1),
        "g_contraction_s": round(timing["g_pass_s"], 1),
        "g_contraction_ms_per_example": 1000.0 * timing["g_pass_s"] / n,
        "full_pass_ms_per_example": 1000.0 * t_pass / n,
        "peak_gpu_mem_GB": mem_gb, "n_patches": npat,
        "base_logits_vs_cache_max_rel": base_err,
        "note": "one full 10k pass: fp64 states walk + 45-VJP downstream contraction (real remaining "
                "network) + H/fc Gram accumulation; downstream marginal Gram is 2048x2048 activation-space"}
    print(f"pass A: {t_pass:.0f}s total, g {timing['g_pass_s']:.0f}s, peak GPU {mem_gb:.1f} GB", flush=True)

    # ---- (c) pass 2 in reversed chunk order; spectra reproducibility ------------------------------
    t0 = time.time()
    Gd_B, Gh_B, Gf_B, _, _ = accumulate_l42conv3(head, xin, "reversed", args.chunk, dev)
    dd, ev_d = spectra_rel_diff(Gd_A, Gd_B)
    dh, ev_h = spectra_rel_diff(Gh_A, Gh_B)
    df, ev_f = spectra_rel_diff(Gf_A, Gf_B)
    rep["c_spectra_reproducibility"] = {
        "definition": "max |lambda_A - lambda_B| / lambda_max over the full eigenvalue spectrum "
                      "(CPU/LAPACK eigvalsh), two independent chunk orders (forward vs reversed)",
        "D_gram_l42conv3": dd, "H_gram_l42conv3": dh, "H_gram_fc_aug": df,
        "pass": bool(max(dd, dh, df) <= 1e-11), "elapsed_s": round(time.time() - t0, 1)}
    print(f"(c) spectra repro: D {dd:.2e}  H {dh:.2e}  fc {df:.2e}", flush=True)

    # ---- (b) D columns vs independent forward-mode autograd JVPs ----------------------------------
    t0 = time.time()
    from torch.func import jvp as fjvp
    rng = np.random.default_rng(stable_seed(20261122, "resnet_pilot", "vjp_triples"))
    i_chunk0 = 0
    xb = xin_chunk(xin, i_chunk0, i_chunk0 + args.chunk, dev)
    x2 = states_to_block(head, xb, 2)
    st = block_states(head, x2, 2)
    g, _ = downstream_g(head, st, 2, 3)                          # (45, B, 2048, 49)
    colnorm = g.permute(1, 2, 3, 0)                              # (B, 2048, 49, 45)
    colnorm = colnorm.norm(dim=3)
    D_scale = float(colnorm.max())
    active = torch.nonzero(colnorm > 0.1 * D_scale, as_tuple=False).cpu().numpy()
    triples = []
    z0 = st["z3"]
    B = z0.shape[0]
    for t in range(args.n_triples):
        if t % 2 == 0 and len(active) > 0:                       # half among clearly active columns
            i, c, xi = [int(v) for v in active[rng.integers(len(active))]]
            kind = "active"
        else:                                                    # half uniform (may hit gated-off zeros)
            i, c, xi = int(rng.integers(B)), int(rng.integers(2048)), int(rng.integers(49))
            kind = "uniform"
        tang = torch.zeros_like(z0)
        tang.view(B, 2048, -1)[i, c, xi] = 1.0
        _, dlg = fjvp(lambda z: tail_from_z(head, 2, 3, z, st["s"]), (z0,), (tang,))
        err = float((dlg[i] - g[:, i, c, xi]).abs().max())
        leak = float(dlg[torch.arange(B, device=dev) != i].abs().max())
        triples.append({"example_in_chunk": i, "channel": c, "position": xi, "kind": kind,
                        "col_norm": float(g[:, i, c, xi].norm()), "max_abs_err": err,
                        "err_over_D_scale": err / D_scale, "cross_example_leakage": leak})
    n_active = sum(1 for r in triples if r["col_norm"] > 0)
    vmax = max(r["err_over_D_scale"] for r in triples)
    lmax = max(r["cross_example_leakage"] for r in triples)
    rep["b_D_validation_vjp"] = {
        "n_triples": len(triples), "n_nonzero_columns": n_active, "D_column_scale": D_scale,
        "max_err_over_D_scale": vmax, "max_cross_example_leakage": lmax, "triples": triples,
        "pass": bool(len(triples) >= 8 and n_active >= 4 and vmax <= 1e-12 and lmax == 0.0),
        "note": "backward 45-VJP contraction rows vs independent forward-mode autograd JVP columns "
                "through the real remaining network, fp64; gated-off positions have exactly-zero "
                "columns in both routes", "elapsed_s": round(time.time() - t0, 1)}
    print(f"(b) VJP triples: max err/scale {vmax:.2e}, {n_active}/{len(triples)} nonzero cols", flush=True)

    # ---- pilot-level rank/tail summaries (informative; production restates these) -----------------
    sv_d = np.sqrt(np.clip(ev_d.numpy()[::-1], 0, None))
    sv_h = np.sqrt(np.clip(ev_h.numpy()[::-1], 0, None))
    sv_f = np.sqrt(np.clip(ev_f.numpy()[::-1], 0, None))
    sqn = np.sqrt(n)
    def opsum(sv, name, units):
        return {"operator": name, "units": units, "dim": int(sv.size),
                "sigma0": float(sv[0]), "sigma_min": float(sv[-1]),
                "rank_T1": rank_gap_record(sv),
                "tail_dim_abs_rate_1e-3_PRIMARY": int((sv / sqn <= RATE_ABS).sum()),
                "tail_dim_relX_rate_1e-3_recorded": int((sv / sqn / logits_rms <= RATE_ABS).sum()),
                "sigma_min_over_sqrtN": float(sv[-1] / sqn)}
    rep["pilot_operator_summaries"] = {
        "l42conv3_downstream_marginal": opsum(sv_d, "stacked D_(i,xi) over (i,xi,class), layer4.2.conv3 "
                                              "preactivation -> logits", "activation-to-output rate"),
        "l42conv3_patch_H": opsum(sv_h, "stacked input patches p_(i,xi), layer4.2.conv3 (1x1: patches = "
                                  "a2 activations)", "feature-space rate"),
        "fc_patch_H_aug": opsum(sv_f, "stacked augmented [pooled;1] features, fc readout", "feature-space rate"),
        "fc_downstream": {"statement": "STRUCTURALLY D = I at the readout: DN_additional = 0, N = FN "
                                       "(reviewer correction Sec. 2); no contraction needed"},
        "logits_rms_X": logits_rms, "N_examples": n,
        "normalization": "rates sigma/sqrt(N), N = 10000 EXAMPLES (never patches)"}

    # ---- timing probes at other depths + projections ----------------------------------------------
    t0 = time.time()
    probes = {}
    for (b, j), tag in [((2, 1), "l42_conv1"), ((0, 1), "l40_conv1")]:
        xb = xin_chunk(xin, 0, args.chunk, dev)
        x_b = states_to_block(head, xb, b)
        st_b = block_states(head, x_b, b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tt = time.time()
        for _ in range(2):
            g_b, _ = downstream_g(head, st_b, b, j)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        g_ms = (time.time() - tt) / 2 / args.chunk * 1000
        # fp32 battery-style forward: K probes batched through the real downstream
        head32 = FrozenLayer4Head(net, dtype=torch.float32, device=dev)
        st32 = {k: v.float() for k, v in st_b.items()}
        K = 32
        z32 = st32[f"z{j}"]
        dz = torch.randn(K, *z32.shape, dtype=torch.float32, device=dev) * 1e-3
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tt = time.time()
        zb = (z32.unsqueeze(0) + dz).reshape(K * z32.shape[0], *z32.shape[1:])
        lg = tail_from_z(head32, b, j, zb, st32["s"])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        f32_ms = (time.time() - tt) / (K * args.chunk) * 1000
        probes[tag] = {"g_pass_ms_per_example": g_ms,
                       "fp32_tail_forward_ms_per_probe_example": f32_ms}
        del g_b, dz, zb, lg, head32
    # projections: per group, per layer class
    nex = 10000
    bat_forwards = 256 * len([1e-3, 3e-3, .01, .03, .1, .3, 1, 3, 10, 30])    # per battery set
    g_l42c3 = rep["benchmark_l42conv3"]["g_contraction_ms_per_example"]
    proj = {}
    for grp, tags in [("l42fc", ["l42_conv1"]), ("l41", ["l42_conv1"]), ("l40", ["l40_conv1"])]:
        gm = probes[tags[0]]["g_pass_ms_per_example"]
        fm = probes[tags[0]]["fp32_tail_forward_ms_per_probe_example"]
        n_layers = 3
        proj[grp] = {
            "assumptions": "3 conv layers/group (fc analytic); 2 g-passes per layer (operator + Jv); "
                           "battery sets per layer = 1 generic + 2 per nonempty approximate family "
                           "(worst case FN-A + DN-A nonempty -> 5 sets); timing from the measured "
                           "per-example probes at the group's deepest downstream (upper bound)",
            "g_passes_h": round(2 * n_layers * gm * nex / 3.6e6, 2),
            "battery_per_set_h": round(bat_forwards * fm * nex / 1000 / 3600, 2),
            "battery_worst_case_5sets_per_layer_h": round(5 * n_layers * bat_forwards * fm * nex / 1000 / 3600, 2),
        }
        proj[grp]["worst_case_total_h"] = round(proj[grp]["g_passes_h"]
                                                + proj[grp]["battery_worst_case_5sets_per_layer_h"] + 2.0, 1)
    rep["timing_probes"] = {"per_example_ms": probes, "l42conv3_g_ms_per_example": g_l42c3,
                            "elapsed_s": round(time.time() - t0, 1)}
    rep["projection_per_group"] = proj
    rep["projection_note"] = ("worst_case_total_h adds ~2 h for operators/eigh/SEL certification; battery "
                              "hours scale DOWN with genuinely empty families (genuine zeros cost ~0); "
                              "SEL certification cost depends on the gate census and is bounded by the "
                              "subset-certificate design in the study script")

    # ---- verdict ----------------------------------------------------------------------------------
    gates = {"cache_authentication": auth["pass"],
             "a_frozen_r0_identity": rep["a_frozen_r0_identity"]["pass"],
             "b_D_validation_vjp": rep["b_D_validation_vjp"]["pass"],
             "c_spectra_reproducibility": rep["c_spectra_reproducibility"]["pass"]}
    rep["gates"] = gates
    rep["PASS"] = bool(all(gates.values()))
    rep["meta"] = meta_block("scripts/analysis/resnet_contraction_pilot.py",
                             {"device": str(dev), "chunk": args.chunk,
                              "elapsed_s": round(time.time() - t_start, 1)})
    torch.save({"sv_D_l42conv3": torch.from_numpy(sv_d.copy()), "sv_H_l42conv3": torch.from_numpy(sv_h.copy()),
                "sv_H_fc_aug": torch.from_numpy(sv_f.copy()),
                "note": "pilot spectra (sigma, descending, unnormalized stacked operators; divide by "
                        "sqrt(N)=100 for rates); production re-derives and persists its own"},
               args.out.replace(".json", "_arrays.pt"))
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("gates:", json.dumps(gates), "\nPASS:", rep["PASS"], "\nWROTE", args.out, flush=True)
    if not rep["PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
