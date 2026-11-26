"""Polar radius sweep of a last-hidden-layer ocean direction, for every width and depth MLP (and reusable for any tag).
Builds two selective-unit directions of the last hidden layer (constructed from features and gates; no Jacobian), then
sweeps angle x radius recording true/frozen output change and gate flips.  Saves results/paper_slices/polar_<label>.npz.
Run: CUDA_VISIBLE_DEVICES=1 python paper_figures/gen_polar.py"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import frozen_lib as FL
from flatocean import blocks as BLK, runs
import slices_compute as SC

DEV = SC.DEV; OUT = SC.OUT
RADII = np.logspace(-3, 0.6, 41); ANGLES = 64
GEN = torch.Generator().manual_seed(5)
CONFIGS = [("w64", 305, "w64"), ("", 40, "w128"), ("w256", 20, "w256"), ("w512", 11, "w512"),
           ("d2", 286, "d2"), ("d4", 26, "d4"), ("d5", 33, "d5")]


def right_null(M, rtol=1e-9):
    Mc = M.double().cpu()
    if Mc.shape[0] == 0:
        return torch.eye(Mc.shape[1], dtype=torch.float64)
    _, s, Vh = torch.linalg.svd(Mc, full_matrices=Mc.shape[0] < Mc.shape[1])
    r = int((s > rtol * s[0]).sum()) if s.numel() and s[0] > 0 else 0
    return Vh[r:].conj().T.contiguous()


def last_hidden_dirs(net, X, theta0):
    """Two orthonormal selective directions of the last hidden layer, as flat parameter vectors."""
    st = BLK.mlp_structure(net, X); L = st["L"]
    H = st["H"][L - 1]; masks = FL.base_masks(net, X)[L - 1]                 # augmented input, gates of last hidden layer
    sl = {n: (a, b) for n, a, b in FL.param_slices(net)}
    wkey, bkey = f"layers.{L - 1}.weight", f"layers.{L - 1}.bias"
    aw, _ = sl[wkey]; ab, _ = sl[bkey]; nprev = net.layers[L - 1].weight.shape[1]
    act = masks.sum(0); order = torch.argsort(act)                           # rarely active first
    cols = []
    for j in order.tolist():
        nul = right_null(H[masks[:, j]])
        if nul.shape[1] >= 1:
            v = torch.zeros_like(theta0)
            v[aw + j * nprev: aw + (j + 1) * nprev] = nul[:nprev, 0].to(DEV)
            v[ab + j] = nul[nprev, 0].to(DEV)
            cols.append(v)
        if len(cols) >= 2:
            break
    Q, _ = torch.linalg.qr(torch.stack(cols, 1))
    return Q[:, 0], Q[:, 1]


def polar_sweep(net, theta0, masks, X, F0, u1, u2, label):
    th = np.linspace(0, 2 * np.pi, ANGLES, endpoint=False)
    pt = np.zeros((ANGLES, len(RADII))); pf = np.zeros_like(pt); pfl = np.zeros_like(pt)
    for i, t in enumerate(th):
        d = float(np.cos(t)) * u1 + float(np.sin(t)) * u2
        for jr, r in enumerate(RADII):
            tr, fz, fl = SC.evaluate(net, theta0, masks, X, F0, float(r) * d)
            pt[i, jr] = tr; pf[i, jr] = fz; pfl[i, jr] = fl
    np.savez(os.path.join(OUT, f"polar_{label}.npz"), angles=th, radii=RADII, pol_true=pt, pol_frozen=pf, pol_flips=pfl, label=label)
    print(f"polar_{label}: inradius(gate) ~ {RADII[int(np.median([np.argmax(pfl[i]>0) if (pfl[i]>0).any() else len(RADII)-1 for i in range(ANGLES)]))]:.2e}", flush=True)


def main():
    torch.set_num_threads(4)
    for tag, ep, label in CONFIGS:
        if os.path.exists(os.path.join(OUT, f"polar_{label}.npz")):
            print("skip", label, flush=True); continue
        net, X, theta0, masks, F0, nt, _ = SC.setup(tag, ep)
        u1, u2 = last_hidden_dirs(net, X, theta0)
        _, fz, _ = SC.evaluate(net, theta0, masks, X, F0, 0.05 * u1)
        print(f"{label}: frozen check @0.05 = {fz:.1e}", flush=True)
        polar_sweep(net, theta0, masks, X, F0, u1, u2, label)
    print("done", flush=True)


if __name__ == "__main__":
    main()
