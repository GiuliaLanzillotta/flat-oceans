"""Shared machinery for the COMMON ResNet marginal/selective family study (two-day plan 2026-11-22).

Authoritative specs: memos/2026-11-17_reviewer_protocol_corrections.md Sec. 2 (marginal family operators),
memos/2026-11-22_two_day_completion_plan.md ("Common ViT and ResNet measurements": fixed conventions),
memos/2026-11-17_vit_resnet_standardization_spec.md Sec. 2 (common protocol T/M/R rows).

Everything here REUSES the validated frozen forward (scripts/analysis/resnet_frozen_forward.py:
FrozenLayer4Head, bn_affine, x_norm) and its checkpoint/data conventions; the downstream derivative D is
taken through the REAL remaining network by autograd (eval-BN fixed affines, residual paths including the
block-4.0 projection shortcut, avgpool, fc), in float64.

Conventions fixed by the plan (declared once, used by the pilot and the study):
  * N = 10,000 examples (the authenticated cache10k); spatial rows stacked, rates normalized by sqrt(N)
    with N = EXAMPLES, never patches.
  * exact rank: sigma <= 1e-9 * sigma0 of the declared operator, gap-10 resolution recorded; exact-null
    candidates re-measured DIRECTLY (fresh contraction pass), never certified from the eigh floor alone
    (ViT recut pilot carry-over rule).
  * approximate tails: absolute singular RATE sigma/sqrt(N) <= 1e-3 in the operator's own units
    (PRIMARY; reviewer correction). The ||.||_X-relative reading is also recorded, never primary.
  * operator eigendecompositions on CPU/LAPACK (basis-level reproducibility; recut-pilot rule).
  * radius grid {1e-3,3e-3,0.01,0.03,0.1,0.3,1,3,10,30}; unit-Frobenius directions; 256 probes per
    nonempty family, dimension-matched randsub + one shared generic control per layer; exact components
    removed from approximate families AND controls; brackets on the RELATIVE dataset-RMS drift at 1e-3,
    absolute dataset-RMS drift recorded separately; median per-example drift supplementary.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, hashlib, datetime
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from resnet_frozen_forward import (FrozenLayer4Head, load_model, bn_affine, x_norm, _aff,  # noqa: F401
                                   CKPT, PARAM_NAMES)
from ftvit_paths import data_path, stable_seed  # noqa: F401

CACHE_DIR = os.path.join(HERE, "results", "resnet", "cache10k")
RESULTS_DIR = os.path.join(HERE, "results", "resnet")
RGRID = [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TOL_REL = 1e-3                     # T(iii) visibility bracket on dataset-RMS RELATIVE drift
RANK_REL = 1e-9                    # T(i) relative singular cutoff
RANK_GAP = 10.0                    # registered gap-10 resolution check
RATE_ABS = 1e-3                    # T(ii) absolute singular rate sigma/sqrt(N), operator's own units
NCLS = 45

GROUPS = {"l42fc": [(2, 1), (2, 2), (2, 3), "fc"],
          "l41": [(1, 1), (1, 2), (1, 3)],
          "l40": [(0, 1), (0, 2), (0, 3)]}

METRICS = ("actual_abs", "actual_rel", "actual_med",
           "frozen_abs", "frozen_rel", "frozen_med",
           "transition_abs", "transition_rel", "transition_med")


def layer_label(L):
    return "readout_fc" if L == "fc" else f"layer4.{L[0]}.conv{L[1]}"


def layer_pname(L):
    return "fc.weight" if L == "fc" else f"layer4.{L[0]}.conv{L[1]}.weight"


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_index(idx):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes()).hexdigest()


def bracket(drift, grid, tol=TOL_REL):
    """Cumulative tested bracket (phase0f/recut convention): first tested radius at/above tol."""
    d = np.asarray(drift, float)
    cross = np.where(d >= tol)[0]
    if len(cross) == 0:
        return {"last_safe": grid[-1], "first_cross": None, "censored": True}
    j = int(cross[0])
    return {"last_safe": (grid[j - 1] if j > 0 else 0.0), "first_cross": grid[j], "censored": False}


def dist(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.10)),
            "q90": float(np.quantile(x, 0.90)), "min": float(x.min()), "max": float(x.max())}


def rank_gap_record(sv, rel=RANK_REL, gap=RANK_GAP):
    """Numerical rank at sigma <= rel*sigma0 with the registered gap-10 resolution record.
    sv: descending singular values (numpy). Full-rank statements record sigma_min/cutoff as the gap."""
    sv = np.asarray(sv, float)
    s0 = float(sv[0]) if sv.size else 0.0
    cut = rel * s0
    r = int((sv > cut).sum())
    rec = {"sigma0": s0, "cutoff_rel": rel, "cutoff": cut, "rank": r, "null_dim": int(sv.size - r)}
    if 0 < r < sv.size:
        g = float(sv[r - 1] / max(sv[r], 1e-300))
        rec.update(gap=g, gap_resolved=bool(g >= gap),
                   sigma_at_rank=float(sv[r - 1]), sigma_below_rank=float(sv[r]))
    elif r == sv.size and r > 0:
        g = float(sv[-1] / max(cut, 1e-300))
        rec.update(gap=g, gap_resolved=bool(g >= gap), sigma_min=float(sv[-1]),
                   note="full numerical rank; gap = sigma_min/cutoff")
    else:
        rec.update(gap=None, gap_resolved=False, note="operator numerically zero")
    return rec


# --------------------------------------------------------------------------------------- cache handling
def cache_paths():
    return {"identity": os.path.join(CACHE_DIR, "cache_identity.json"),
            "xin": os.path.join(CACHE_DIR, "xin_fp32.npy"),
            "ref_local": os.path.join(CACHE_DIR, "ref_logits_local_fp64.pt"),
            "ref_fullnet": os.path.join(CACHE_DIR, "ref_logits_fullnet_fp64.pt"),
            "census": os.path.join(CACHE_DIR, "gate_census.pt")}


def load_cache_identity(cache_dir=None):
    p = os.path.join(cache_dir or CACHE_DIR, "cache_identity.json")
    with open(p) as f:
        return json.load(f)


def authenticate_cache(cache_dir=None, check_bytes=True):
    """Fail-closed authentication of the 10k cache: recompute the SHA-256 of every stamped file and the
    checkpoint, compare with the stored identity. Returns (identity, report). Raises on mismatch."""
    cd = cache_dir or CACHE_DIR
    ident = load_cache_identity(cd)
    rep = {"cache_dir": cd, "checked": {}, "pass": True}
    checks = []
    if check_bytes:
        for key, entry in ident["files"].items():
            p = os.path.join(cd, entry["name"])
            got = sha256_file(p)
            ok = (got == entry["sha256"])
            checks.append(ok)
            rep["checked"][key] = {"file": p, "stored": entry["sha256"], "recomputed": got, "pass": ok}
        ck_got = sha256_file(ident["checkpoint"]["path"])
        ok = (ck_got == ident["checkpoint"]["sha256"])
        checks.append(ok)
        rep["checked"]["checkpoint"] = {"file": ident["checkpoint"]["path"],
                                        "stored": ident["checkpoint"]["sha256"],
                                        "recomputed": ck_got, "pass": ok}
    rep["pass"] = all(checks) if checks else True
    if not rep["pass"]:
        raise RuntimeError(f"cache authentication FAILED (fail closed): {json.dumps(rep)}")
    return ident, rep


def open_xin(cache_dir=None):
    """fp32 on-disk cache with the fp64-read path: returns (memmap, n). Chunks are promoted to float64
    immediately on read; every derived operator is formed in float64 (recorded storage choice)."""
    cd = cache_dir or CACHE_DIR
    ident = load_cache_identity(cd)
    mm = np.load(os.path.join(cd, ident["files"]["xin"]["name"]), mmap_mode="r")
    assert mm.dtype == np.float32
    return mm, mm.shape[0]


def xin_chunk(mm, i0, i1, device, dtype=torch.float64):
    return torch.from_numpy(np.asarray(mm[i0:i1])).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- base-state walk (actual)
@torch.no_grad()
def states_to_block(head, x, b):
    """Run blocks 0..b-1 at theta0 (actual forward) and return the input of block b."""
    for bb in range(b):
        x = run_block(head, x, bb)
    return x


@torch.no_grad()
def run_block(head, h, b):
    info = head.blocks[b]
    W = head.theta0
    c1, c2, c3 = info["conv_cfg"]
    bn1, bn2, bn3 = info["bn"]
    h1 = Fn.relu(_aff(*bn1, Fn.conv2d(h, W[f"layer4.{b}.conv1.weight"], **c1)))
    h2 = Fn.relu(_aff(*bn2, Fn.conv2d(h1, W[f"layer4.{b}.conv2.weight"], **c2)))
    u3 = _aff(*bn3, Fn.conv2d(h2, W[f"layer4.{b}.conv3.weight"], **c3))
    if info["down_cfg"] is None:
        s = h
    else:
        s = _aff(*info["down_bn"], Fn.conv2d(h, W["layer4.0.downsample.0.weight"], **info["down_cfg"]))
    return Fn.relu(u3 + s)


@torch.no_grad()
def block_states(head, h, b):
    """All base quantities of block b from its input h: conv outputs z1..z3 (pre-BN), activations a1,a2,
    shortcut branch s, block output. Mirrors FrozenLayer4Head._run stage by stage (actual forward)."""
    info = head.blocks[b]
    W = head.theta0
    c1, c2, c3 = info["conv_cfg"]
    bn1, bn2, bn3 = info["bn"]
    z1 = Fn.conv2d(h, W[f"layer4.{b}.conv1.weight"], **c1)
    a1 = Fn.relu(_aff(*bn1, z1))
    z2 = Fn.conv2d(a1, W[f"layer4.{b}.conv2.weight"], **c2)
    a2 = Fn.relu(_aff(*bn2, z2))
    z3 = Fn.conv2d(a2, W[f"layer4.{b}.conv3.weight"], **c3)
    u3 = _aff(*bn3, z3)
    if info["down_cfg"] is None:
        s = h
    else:
        s = _aff(*info["down_bn"], Fn.conv2d(h, W["layer4.0.downsample.0.weight"], **info["down_cfg"]))
    return {"x": h, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "u3": u3, "s": s,
            "out": Fn.relu(u3 + s)}


def base_gates_of_states(head, st, b):
    """Base gate masks of block b in frozen-forward convention (1{u>0} AFTER the eval-BN affine /
    residual addition): (g1, g2, g_out)."""
    info = head.blocks[b]
    bn1, bn2, _ = info["bn"]
    return (_aff(*bn1, st["z1"]) > 0, _aff(*bn2, st["z2"]) > 0, (st["u3"] + st["s"]) > 0)


def tail_from_z(head, b, j, z, s_short):
    """Logits from the conv-j output z of block b (pre-BN), through the REAL remaining network at theta0:
    eval-BN affines, real ReLU, the block's residual addition (s_short precomputed from the block input;
    for block 4.0 this is the fixed projection-shortcut branch), remaining blocks, avgpool, fc.
    Differentiable in z (used for the downstream derivative D by autograd). Batch of z may be a probe
    multiple of s_short's batch (s_short is broadcast by repeat)."""
    info = head.blocks[b]
    W = head.theta0
    cfgs = info["conv_cfg"]
    bns = info["bn"]
    h = z
    if j <= 1:
        h = Fn.conv2d(Fn.relu(_aff(*bns[0], h)), W[f"layer4.{b}.conv2.weight"], **cfgs[1])
    if j <= 2:
        h = Fn.conv2d(Fn.relu(_aff(*bns[1], h)), W[f"layer4.{b}.conv3.weight"], **cfgs[2])
    u3 = _aff(*bns[2], h)
    if u3.shape[0] != s_short.shape[0]:
        rep = u3.shape[0] // s_short.shape[0]
        s_short = s_short.repeat(rep, 1, 1, 1)
    h = Fn.relu(u3 + s_short)
    for bb in range(b + 1, 3):
        info2 = head.blocks[bb]
        W1 = W[f"layer4.{bb}.conv1.weight"]
        c1, c2, c3 = info2["conv_cfg"]
        b1, b2, b3 = info2["bn"]
        h1 = Fn.relu(_aff(*b1, Fn.conv2d(h, W1, **c1)))
        h2 = Fn.relu(_aff(*b2, Fn.conv2d(h1, W[f"layer4.{bb}.conv2.weight"], **c2)))
        u = _aff(*b3, Fn.conv2d(h2, W[f"layer4.{bb}.conv3.weight"], **c3))
        h = Fn.relu(u + h)                                     # blocks 4.1/4.2: identity shortcut
    return h.mean((2, 3)) @ W["fc.weight"].T + W["fc.bias"]


def downstream_g(head, st, b, j, ncls=NCLS):
    """Stacked downstream derivative for the chunk: g[a, i, c, xi] = d logits[i, a] / d z_j[i, c, xi]
    through the REAL remaining network (autograd at the base point). Returns (g, base_logits) with
    g of shape (ncls, B, n, P) and P = flattened spatial positions. Rows of g are exactly the rows
    e_a^T D_(i,xi) of the common downstream marginal operator."""
    z = st[f"z{j}"].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        lg = tail_from_z(head, b, j, z, st["s"])
        gs = [torch.autograd.grad(lg[:, a].sum(), z, retain_graph=(a < ncls - 1))[0] for a in range(ncls)]
    G = torch.stack(gs)
    B, n = z.shape[0], z.shape[1]
    return G.reshape(ncls, B, n, -1), lg.detach()


def conv_patches(head, st, b, j):
    """im2col patches of the conv-j input (B, m, P): column xi is p_(i,xi) with the C*k*k flattening
    that matches weight.view(n, m). Convolutional weight sharing is respected by construction."""
    inp = st["x"] if j == 1 else st[f"a{j - 1}"]
    cfg = head.blocks[b]["conv_cfg"][j - 1]
    W = head.theta0[f"layer4.{b}.conv{j}.weight"]
    return Fn.unfold(inp, kernel_size=(W.shape[2], W.shape[3]), padding=cfg["padding"],
                     stride=cfg["stride"], dilation=cfg["dilation"])


def im2col_consistency(head, st, b, j, tol=1e-10):
    """One-time check per layer: conv2d == W.view(n,m) @ patches (ordering/stride/padding agreement)."""
    W = head.theta0[f"layer4.{b}.conv{j}.weight"]
    n = W.shape[0]
    P = conv_patches(head, st, b, j)
    z_ref = st[f"z{j}"]
    z_mat = (W.view(n, -1) @ P).view(z_ref.shape)
    err = float((z_mat - z_ref).abs().max() / max(float(z_ref.abs().max()), 1e-300))
    assert err <= tol, f"im2col/conv mismatch at layer4.{b}.conv{j}: rel {err:.3e}"
    return err


def shapes_of(head, L):
    """(n, m, k, bias) of the selected layer's weight operator."""
    if L == "fc":
        W = head.theta0["fc.weight"]
        return W.shape[0], W.shape[1] + 1, 1, True            # augmented [W|b]
    W = head.theta0[f"layer4.{L[0]}.conv{L[1]}.weight"]
    return W.shape[0], W.shape[1] * W.shape[2] * W.shape[3], W.shape[2], False


def delta_from_A(head, L, A):
    """Parameter-space delta dict for FrozenLayer4Head from the (n x m) operator edit A (float64 CPU/GPU).
    For fc the last column is the bias coordinate (augmented convention; fc HAS a bias)."""
    if L == "fc":
        return {"fc.weight": A[:, :-1].contiguous(), "fc.bias": A[:, -1].contiguous()}
    name = f"layer4.{L[0]}.conv{L[1]}.weight"
    W = head.theta0[name]
    return {name: A.reshape(W.shape).contiguous()}


def git_rev():
    try:
        import subprocess
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def meta_block(script, extra=None):
    m = {"script": script, "date": datetime.date.today().isoformat(), "torch": torch.__version__,
         "git_rev": git_rev(),
         "spec": ["memos/2026-11-17_reviewer_protocol_corrections.md#2",
                  "memos/2026-11-22_two_day_completion_plan.md#common-vit-and-resnet-measurements",
                  "memos/2026-11-17_vit_resnet_standardization_spec.md#2"]}
    if extra:
        m.update(extra)
    return m
