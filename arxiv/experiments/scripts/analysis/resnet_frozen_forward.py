"""Frozen forward for ResNet-50 layer4 (blocks 4.0-4.2) + avgpool + fc under eval-mode BatchNorm.

Implements items A and B of memos/2026-11-16_resnet_implementation_map.md (closes the implementation half of
mismatch M5 of memos/2026-11-13_resnet_correspondence.md).  Conventions mirrored EXACTLY from the MLP/CNN
frozen implementation (frozen_lib.py, protocol memos/2026-09-10_protocol_stages_atomic_transition.md Sec. 4/6):

  * Gates.  The base gates S are the per-sample, per-unit, per-position sign masks 1{u > 0} of every ReLU
    preactivation, computed ONCE at the base parameters theta_0 (frozen_lib.base_masks).  The frozen forward
    F^frz(w) applies the EDITED affine maps of theta_0 + w with the FIXED masks S in place of ReLU
    (frozen_lib.frozen_forward: `h = S * z`); hidden values are recomputed, derivatives/masks are not.  The
    frozen network is therefore the piecewise-linear branch of the original network at theta_0, evaluated at
    the perturbed weights; it coincides with the original network until a gate flips.
  * Eval-mode BatchNorm.  BN is a fixed per-channel affine u = a * z + off with
    a = gamma / sqrt(running_var + eps), off = beta - a * running_mean (frozen_lib.bn_affine, identical
    formula).  Running statistics are frozen; BN affine parameters (gamma/beta) are HELD FIXED and are not
    part of the perturbable set (frozen_lib.block_slices convention: BN params are not a block of the
    conditional analysis).  `bn_affine` below is the ONE shared helper (memo item A): the feature-space Gram
    assembly must import it from here rather than reimplementing the affine.
  * Residual paths.  The identity shortcuts (blocks 4.1, 4.2) and the FIXED projection shortcut of block 4.0
    (downsample conv -> eval-BN affine) are linear maps; they pass through the frozen forward unchanged.  The
    downsample conv weight belongs to the perturbable set (its BN affine does not).
  * Perturbable set (any subset may be perturbed; everything upstream of layer4 stays at theta_0 and its
    activations are cached as the fixed input X_in = layer3 output):
        layer4.{0,1,2}.conv{1,2,3}.weight, layer4.0.downsample.0.weight, fc.weight, fc.bias.
  * Structure of the frozen map.  With fixed gates every block is a composition of linear maps, so along a
    direction that perturbs a SINGLE layer the frozen map is affine; along a multi-layer direction it is a
    POLYNOMIAL in t of degree <= the maximum number of perturbed layers on any input->logit path (products
    across serially composed layers; parallel residual branches ADD and do not multiply degrees).
  * Norms: ||V||_X = (s^-1 sum_i ||v_i||^2)^(1/2) over stacked outputs (frozen_lib.x_norm).

Checkpoint/data paths follow scripts/analysis/resnet_cand_val_v2.py (seed-0 fine-tuned ResNet-50/RESISC45).
The CLI runs the item-B validation (fp64, small batch): frozen_base_err at r=0 against the original
torchvision forward, gating consistency over r in {1e-6, 1e-4, 1e-2}, single-layer affinity, and multi-layer
polynomial-degree checks.  Writes results/resnet/frozen_forward_validation.json (new record).  Light job
(minutes, one GPU or CPU).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, argparse, datetime
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = str(PROJECT_ROOT)
RD = os.path.join(HERE, "results", "runs", "ftresnet_resisc45")
CKPT = os.path.join(RD, "ckpt_ft_resnet_lr1e-4_confirm_s0.pt")

# perturbable parameter names, in a fixed order (weights of the three convs per block, the projection
# shortcut conv of block 0, and the fc head; conv biases do not exist in ResNet-50, BN params are fixed)
PARAM_NAMES = tuple(
    [f"layer4.{b}.conv{j}.weight" for b in range(3) for j in (1, 2, 3)]
    + ["layer4.0.downsample.0.weight", "fc.weight", "fc.bias"]
)


def bn_affine(bn, dtype=None):
    """SHARED eval-mode BatchNorm affine helper (memo 2026-11-16 item A; formula of frozen_lib.bn_affine).

    Returns (a, off) with u = a * z + off applied per channel to the pre-BN value z:
    a = gamma / sqrt(running_var + eps), off = beta - a * running_mean.  Running stats and gamma/beta are
    taken detached at their current (base) values.  The feature-space Gram and the frozen forward must both
    use THIS function so the affine enters identically (one implementation, not two)."""
    w, b = bn.weight.detach(), bn.bias.detach()
    m, v = bn.running_mean.detach(), bn.running_var.detach()
    if dtype is not None:
        w, b, m, v = w.to(dtype), b.to(dtype), m.to(dtype), v.to(dtype)
    a = w / torch.sqrt(v + bn.eps)
    return a, b - a * m


def x_norm(V):
    """||V||_X for V of shape (s, d): root mean over samples of the squared row norm (frozen_lib.x_norm)."""
    return float(V.double().pow(2).sum(1).mean().sqrt())


def _aff(a, off, z):
    return a.view(1, -1, 1, 1) * z + off.view(1, -1, 1, 1)


class FrozenLayer4Head:
    """Actual and frozen forward of layer4 (blocks 4.0-4.2) + avgpool + fc on cached layer3 activations.

    Built from a loaded torchvision ResNet-50 (eval mode).  All base weights and BN affines are detached
    copies in `dtype`.  Perturbations enter as a dict {name: delta_tensor} over any subset of PARAM_NAMES,
    added to the base weights; gates stay fixed in the frozen forward."""

    def __init__(self, net, dtype=torch.float64, device=None):
        assert not net.training, "network must be in eval mode (frozen running stats)"
        dev = device if device is not None else next(net.parameters()).device
        self.dtype, self.device = dtype, dev

        def cp(t):
            return t.detach().to(device=dev, dtype=dtype).clone()

        self.theta0, self.blocks = {}, []
        for b in range(3):
            blk = net.layer4[b]
            info = {"conv_cfg": [], "bn": []}
            for j, (conv, bn) in enumerate([(blk.conv1, blk.bn1), (blk.conv2, blk.bn2), (blk.conv3, blk.bn3)]):
                name = f"layer4.{b}.conv{j + 1}.weight"
                assert conv.bias is None and conv.groups == 1
                self.theta0[name] = cp(conv.weight)
                info["conv_cfg"].append({"stride": conv.stride, "padding": conv.padding, "dilation": conv.dilation})
                info["bn"].append(tuple(t.to(dev) for t in bn_affine(bn, dtype)))
            if blk.downsample is not None:
                dconv, dbn = blk.downsample[0], blk.downsample[1]
                assert dconv.bias is None
                self.theta0["layer4.0.downsample.0.weight"] = cp(dconv.weight)
                info["down_cfg"] = {"stride": dconv.stride, "padding": dconv.padding, "dilation": dconv.dilation}
                info["down_bn"] = tuple(t.to(dev) for t in bn_affine(dbn, dtype))
            else:
                info["down_cfg"] = None
            self.blocks.append(info)
        self.theta0["fc.weight"] = cp(net.fc.weight)
        self.theta0["fc.bias"] = cp(net.fc.bias)
        assert set(self.theta0) == set(PARAM_NAMES)

    # ------------------------------------------------------------------ core walk (actual / frozen unified)
    @torch.no_grad()
    def _run(self, Xin, delta=None, gates=None, frozen=False, count_flips=False):
        """One pass over blocks 4.0-4.2 + avgpool + fc.

        frozen=False, gates=None       : actual forward; returns (logits, collected_gates, None)
        frozen=False, gates given      : actual forward; returns (logits, None, flip_counts) if count_flips
        frozen=True,  gates given      : frozen forward; returns (logits, None, None)
        """
        assert not (frozen and gates is None), "frozen forward needs base gates"
        delta = delta or {}
        for k in delta:
            assert k in self.theta0, f"unknown perturbed parameter {k}"
        W = lambda name: (self.theta0[name] + delta[name].to(self.theta0[name])) if name in delta else self.theta0[name]
        h = Xin.to(device=self.device, dtype=self.dtype)
        collected, flips = ([] if gates is None else None), ([] if count_flips else None)

        def act(u, g):
            if frozen:
                return u * g.to(u.dtype)
            if gates is None:
                collected[-1].append(u > 0)
            elif count_flips:
                flips.append(int(((u > 0) != g).sum()))
            return Fn.relu(u)

        for b, info in enumerate(self.blocks):
            if gates is None:
                collected.append([])
            gb = gates[b] if gates is not None else (None, None, None)
            c1, c2, c3 = info["conv_cfg"]
            (a1, o1), (a2, o2), (a3, o3) = info["bn"]
            h1 = act(_aff(a1, o1, Fn.conv2d(h, W(f"layer4.{b}.conv1.weight"), **c1)), gb[0])
            h2 = act(_aff(a2, o2, Fn.conv2d(h1, W(f"layer4.{b}.conv2.weight"), **c2)), gb[1])
            u3 = _aff(a3, o3, Fn.conv2d(h2, W(f"layer4.{b}.conv3.weight"), **c3))
            if info["down_cfg"] is not None:                       # FIXED projection shortcut (block 4.0)
                ad, od = info["down_bn"]
                s = _aff(ad, od, Fn.conv2d(h, W("layer4.0.downsample.0.weight"), **info["down_cfg"]))
            else:                                                  # identity shortcut (blocks 4.1, 4.2)
                s = h
            h = act(u3 + s, gb[2])                                 # block-output ReLU gates the SUM
        logits = h.mean((2, 3)) @ W("fc.weight").T + W("fc.bias")
        if gates is None:
            collected = [tuple(gs) for gs in collected]
        return logits, collected, flips

    @torch.no_grad()
    def base_gates(self, Xin):
        """Base gate masks S at theta_0: list over blocks of (g1, g2, g_out) boolean per-sample masks
        (1{u > 0} of each ReLU preactivation, AFTER the eval-BN affine / residual addition, which is where
        the gate acts).  Also returns the base logits of this local forward."""
        logits, gates, _ = self._run(Xin)
        return gates, logits

    @torch.no_grad()
    def actual(self, Xin, delta=None, gates=None):
        """Original (real-ReLU) local forward at theta_0 + delta.  With `gates` given, also returns the
        per-gate-layer flip counts against the base masks."""
        logits, _, flips = self._run(Xin, delta=delta, gates=gates, frozen=False, count_flips=gates is not None)
        return (logits, flips) if gates is not None else logits

    @torch.no_grad()
    def frozen(self, Xin, gates, delta=None):
        """Frozen forward F^frz at theta_0 + delta: edited weights, fixed base masks in place of ReLU."""
        return self._run(Xin, delta=delta, gates=gates, frozen=True)[0]

    # ------------------------------------------------------------------ directions
    def random_direction(self, names, seed, device=None):
        """Unit-norm (global Euclidean) Gaussian direction supported on the given parameter subset."""
        g = torch.Generator().manual_seed(seed)
        d = {n: torch.randn(self.theta0[n].shape, generator=g, dtype=torch.float64) for n in names}
        nrm = torch.sqrt(sum(t.pow(2).sum() for t in d.values()))
        return {n: (t / nrm).to(device=device or self.device, dtype=self.dtype) for n, t in d.items()}


# ---------------------------------------------------------------------------- model / data loading (paths
# follow scripts/analysis/resnet_cand_val_v2.py)
def load_model(device, ckpt=CKPT):
    import torchvision.models as tvm
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = tvm.resnet50(weights=None)
    net.fc = nn.Linear(2048, ck["n_classes"])
    net.load_state_dict(ck["state_dict"])
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ck


def cache_base_inputs(net, n, device, dtype=torch.float64, batch=16):
    """Earlier layers at theta_0: full-network fp64 forward on the first n RESISC45 train images, caching the
    layer3 output X_in (the fixed input to layer4) and the reference logits of the ORIGINAL network."""
    import numpy as np
    from torchvision.models import ResNet50_Weights
    sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
    from ftvit_paths import data_path
    tfm = ResNet50_Weights.IMAGENET1K_V2.transforms()
    imgs = np.load(data_path("resisc45_train_imgs.npy"), mmap_mode="r")
    n = min(n, imgs.shape[0])
    net = net.to(dtype)
    h, Xin, ref = {}, [], []
    hnd = net.layer3.register_forward_hook(lambda m, i, o: h.__setitem__("x", o.detach()))
    with torch.no_grad():
        for i in range(0, n, batch):
            b = np.asarray(imgs[i:min(i + batch, n)])
            x = torch.from_numpy(b).to(device).float().div(255).permute(0, 3, 1, 2).contiguous()
            x = tfm(x).to(dtype)
            ref.append(net(x).cpu())
            Xin.append(h["x"].cpu())
    hnd.remove()
    return torch.cat(Xin).to(device), torch.cat(ref).to(device), n


# ---------------------------------------------------------------------------- validation (memo item B)
def _fd_table(head, Xin, gates, delta, ts):
    """Frozen outputs g(t) along theta_0 + t * delta at the given nodes, stacked (len(ts), s, nc)."""
    outs = []
    for t in ts:
        outs.append(head.frozen(Xin, gates, {k: t * v for k, v in delta.items()}))
    return torch.stack(outs)


def _fd_orders(vals):
    """Successive forward differences of vals (m, s, nc) along dim 0; returns [max|Delta^k|] for k=1..m-1."""
    out = []
    d = vals
    while d.shape[0] > 1:
        d = d[1:] - d[:-1]
        out.append(float(d.abs().max()))
    return out


def validate(n=64, device="cuda", out_path=os.path.join(HERE, "results", "resnet", "frozen_forward_validation.json")):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    t0 = time.time()
    net, ck = load_model(dev)
    Xin, ref_logits, n = cache_base_inputs(net, n, dev)
    head = FrozenLayer4Head(net, dtype=torch.float64, device=dev)
    gates, base_local = head.base_gates(Xin)
    F0 = head.frozen(Xin, gates)                                   # frozen at theta_0 (r = 0)
    rep = {"n_images": n, "device": str(dev), "dtype": "float64",
           "n_gate_units_total": int(sum(int(g.numel()) for gs in gates for g in gs))}
    scale = float(ref_logits.abs().max())
    print(f"setup: n={n} dev={dev} ({time.time() - t0:.0f}s); |ref| max {scale:.3f}", flush=True)

    # (a) frozen_base_err at r=0 -------------------------------------------------------------------------
    d_full = F0 - ref_logits
    rep["frozen_base_vs_original_fullnet"] = {
        "max_abs": float(d_full.abs().max()), "max_rel": float(d_full.abs().max()) / scale,
        "rms_rel_X": x_norm(d_full) / max(x_norm(ref_logits), 1e-300),
        "note": "original torchvision fp64 forward vs frozen forward (shared bn_affine); differs only by "
                "BN-affine rounding (a*z+off vs fused batch_norm)"}
    d_loc = F0 - base_local
    rep["frozen_base_vs_local_actual"] = {
        "max_abs": float(d_loc.abs().max()),
        "note": "same code path with real ReLU vs fixed base masks at theta_0; expected exactly 0"}
    assert rep["frozen_base_vs_original_fullnet"]["max_rel"] <= 1e-12, rep["frozen_base_vs_original_fullnet"]
    assert rep["frozen_base_vs_local_actual"]["max_abs"] <= 1e-14 * scale, rep["frozen_base_vs_local_actual"]
    print("(a) frozen_base_err:", json.dumps(rep["frozen_base_vs_original_fullnet"]), flush=True)
    print("    frozen vs local actual at theta0: max_abs", rep["frozen_base_vs_local_actual"]["max_abs"], flush=True)

    # (b) gating consistency: frozen vs actual over r ----------------------------------------------------
    rep["gating_consistency"] = {}
    for dir_name, names, seed in [("layer4.1.conv2.weight", ["layer4.1.conv2.weight"], 11),
                                  ("layer4.2.conv3.weight", ["layer4.2.conv3.weight"], 12)]:
        v = head.random_direction(names, seed)
        rows = []
        for r in (1e-6, 1e-4, 1e-2):
            dlt = {k: r * t for k, t in v.items()}
            fa, flips = head.actual(Xin, dlt, gates=gates)
            fz = head.frozen(Xin, gates, dlt)
            drift = x_norm(fa - F0)
            rows.append({"r": r, "flips_total": int(sum(flips)), "flips_per_gate_layer": flips,
                         "actual_drift_X": drift, "frozen_vs_actual_X": x_norm(fa - fz),
                         "frozen_vs_actual_rel_drift": x_norm(fa - fz) / max(drift, 1e-300)})
            print(f"(b) {dir_name} r={r:g}: flips={rows[-1]['flips_total']} drift_X={drift:.3e} "
                  f"|frz-act|_X={rows[-1]['frozen_vs_actual_X']:.3e}", flush=True)
        rep["gating_consistency"][dir_name] = rows
        assert rows[0]["flips_total"] == 0, ("expected no gate flips at r=1e-6", rows[0])
        assert rows[0]["frozen_vs_actual_X"] <= 1e-13 * scale, ("no-flip frozen/actual mismatch", rows[0])

    # (c) affinity in single-layer directions ------------------------------------------------------------
    rep["single_layer_affinity"] = {}
    a, b = 0.5, 1.5
    for name in ["layer4.0.conv1.weight", "layer4.1.conv2.weight", "layer4.2.conv3.weight",
                 "layer4.0.downsample.0.weight", "fc.weight"]:
        v = head.random_direction([name], 20)
        vals = _fd_table(head, Xin, gates, v, [a, (a + b) / 2, b])  # F(av) + F(bv) - 2 F(((a+b)/2) v)
        resid = float((vals[0] + vals[2] - 2 * vals[1]).abs().max())
        gmax = float(vals.abs().max())
        rep["single_layer_affinity"][name] = {"max_abs": resid, "rel": resid / gmax, "scale": gmax}
        assert resid / gmax <= 1e-11, (name, rep["single_layer_affinity"][name])
        print(f"(c) single-layer {name}: affinity rel {resid / gmax:.2e}", flush=True)

    # (c') multi-layer directions: polynomial of degree <= #perturbed serial layers ----------------------
    rep["multi_layer_polynomial"] = {}
    cases = [
        ("serial_deg3", ["layer4.0.conv2.weight", "layer4.1.conv3.weight", "fc.weight"], 3, 0.5, 30),
        ("parallel_deg1", ["layer4.0.conv3.weight", "layer4.0.downsample.0.weight"], 1, 0.5, 31),
        ("all_layers_deg10", list(PARAM_NAMES), 10, 0.3, 32),
    ]
    for tag, names, deg, hstep, seed in cases:
        v = head.random_direction(names, seed)
        ts = [j * hstep for j in range(deg + 2)]
        vals = _fd_table(head, Xin, gates, v, ts)
        orders = _fd_orders(vals)                                  # max|Delta^k|, k = 1..deg+1
        gmax = float(vals.abs().max())
        row = {"names": names, "bound_degree": deg, "nodes_h": hstep,
               "fd_max_abs_by_order": orders, "rel_Delta_top": orders[-1] / gmax,
               "rel_Delta_deg": orders[-2] / gmax if len(orders) > 1 else None, "scale": gmax}
        rep["multi_layer_polynomial"][tag] = row
        tol = 1e-9 if deg >= 10 else 1e-10
        assert row["rel_Delta_top"] <= tol, (tag, row)
        print(f"(c') {tag}: |Delta^{deg + 1}|/max|g| = {row['rel_Delta_top']:.2e} "
              f"(|Delta^{deg}|/max|g| = {row['rel_Delta_deg']:.2e}, scale {gmax:.2e})", flush=True)
    # the parallel case is affine although two tensors are perturbed (residual branches ADD):
    par = rep["multi_layer_polynomial"]["parallel_deg1"]
    par["note"] = "degree bound #layers=2, measured affine (Delta^2 at machine floor): parallel residual branches add"
    # nontriviality: along a serial multi-layer direction the frozen map is NOT affine and the top polynomial
    # coefficient is resolved -- Delta^2 and Delta^deg must sit far above the Delta^(deg+1) machine floor.
    # (Unit directions are SMALL relative to the ResNet weight norms, so these are relative-to-floor gates,
    # not the absolute 1e-3 cut of frozen_lib's toy-scale check.)
    ml = rep["multi_layer_polynomial"]["serial_deg3"]
    fd = ml["fd_max_abs_by_order"]
    rep["serial_deg3_nontriviality"] = {"Delta2_over_floor": fd[1] / max(fd[-1], 1e-300),
                                        "Delta3_over_floor": fd[2] / max(fd[-1], 1e-300)}
    assert rep["serial_deg3_nontriviality"]["Delta2_over_floor"] > 1e4, ("frozen map looks affine along a serial multi-layer direction", ml)
    assert rep["serial_deg3_nontriviality"]["Delta3_over_floor"] > 1e4, ("degree-3 coefficient not resolved above the FD floor", ml)
    rep["multi_layer_polynomial"]["all_layers_deg10"]["note"] = (
        "degree bound holds (Delta^11 at floor); at these amplitudes the degree-10 coefficient (product of "
        "all ten small per-layer components) is itself numerically at the floor, so only the BOUND is "
        "informative for this case -- degree resolution is demonstrated by serial_deg3")

    rep["PASS"] = True
    rep["meta"] = {"script": "scripts/analysis/resnet_frozen_forward.py", "ckpt": CKPT,
                   "val_acc": ck.get("val_acc"), "test_acc": ck.get("test_acc"),
                   "date": datetime.date.today().isoformat(),
                   "torch": torch.__version__, "elapsed_s": round(time.time() - t0, 1),
                   "convention": "frozen_lib.py frozen-gate convention (fixed base masks 1{u>0}, edited "
                                 "affine maps); eval-BN as fixed per-channel affine via the shared bn_affine "
                                 "helper; BN gamma/beta and running stats held fixed; perturbable set = "
                                 "layer4 conv weights + projection-shortcut conv + fc (W, b); earlier layers "
                                 "cached at theta_0 (layer3 output as fixed input)"}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rep, open(out_path, "w"), indent=2)
    print("WROTE", out_path, "\nALL CHECKS PASSED", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ResNet layer4 frozen-forward validation (memo 2026-11-16 item B)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "resnet", "frozen_forward_validation.json"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    validate(n=args.n, device=args.device, out_path=args.out)
