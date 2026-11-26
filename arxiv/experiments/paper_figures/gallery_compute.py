"""X-ray gallery: one 2D parameter slice per source of functional flatness, width-64 ReLU MLP, seed 0, high accuracy.
Each panel is a plane spanned by two directions of the SAME source subspace (constructed from the base features H,
downstream maps D and gate masks; no Jacobian assembly), so the flat region shows as a 2D island whose extent is the
radius.  slice_grid records the true/frozen output change and gate crossings on the 10k analysis set.
Writes results/paper_slices/gal_<source>.npz.  Run from experiments/ (venv): CUDA_VISIBLE_DEVICES=2 python paper_figures/gallery_compute.py"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import frozen_lib as FL
from flatocean import runs
from flatocean.analyses import branch as B
from flatocean.analyses import branch_rays as BR
from flatocean import blocks as BLK
import slices_compute as SC

DEV = SC.DEV; OUT = SC.OUT; RES = os.path.join(runs.HERE, "results")
GEN = torch.Generator().manual_seed(7)


def right_null(M, rtol=1e-9):
    if M.shape[0] == 0:
        return torch.eye(M.shape[1], dtype=torch.float64)
    Mc = M.double().cpu(); wide = Mc.shape[0] < Mc.shape[1]
    _, s, Vh = torch.linalg.svd(Mc, full_matrices=wide)
    r = int((s > rtol * s[0]).sum()) if s.numel() and s[0] > 0 else 0
    return Vh[r:].conj().T.contiguous()


def top_right(M):
    _, _, Vh = torch.linalg.svd(M.double().cpu(), full_matrices=False)
    return Vh[0].contiguous()


def zeros(nt):
    return {b: torch.zeros(nt.shape[b], dtype=torch.float64) for b in nt.blocks}


def emb(net, nt, M):
    return B.embed(net, nt, M).to(DEV)


def ortho(cols):
    """Orthonormal flat basis (P x k) from a list of flat direction tensors."""
    Q, _ = torch.linalg.qr(torch.stack(cols, 1))
    return Q


def build_bases(net, X, nt):
    """name -> (P x k) orthonormal flat basis of the source subspace (k >= 2)."""
    st = BLK.mlp_structure(net, X)
    masks = [m for m in FL.base_masks(net, X)]
    bank = {}; info = {}
    def rc(n):
        return torch.randn(n, generator=GEN, dtype=torch.float64)
    # feature-null of the readout: rows in the null of the last-layer features
    N = right_null(st["H"][3])
    if N.shape[1] >= 2:
        cols = [emb(net, nt, {**zeros(nt), "V": (lambda M: (M.__setitem__(0, N[:, 0]) or M))(torch.zeros(nt.shape["V"], dtype=torch.float64))}),
                emb(net, nt, {**zeros(nt), "V": (lambda M: (M.__setitem__(1, N[:, 1]) or M))(torch.zeros(nt.shape["V"], dtype=torch.float64))})]
        bank["feature_null"] = ortho(cols); info["feature_null"] = dict(dim=int(N.shape[1]))
    # zero-feature readout column of a dead layer-3 unit
    dd = torch.where(masks[2].sum(0) == 0)[0]
    if len(dd):
        j = int(dd[0]); cols = []
        for a in (0, 1):
            M = zeros(nt); M["V"][a, j] = 1.0; cols.append(emb(net, nt, M))
        bank["zero_feature_out"] = ortho(cols); info["zero_feature_out"] = dict(unit=j, block="readout")
    # downstream-null of layer 3: A = c a^T, c in C^perp, a a fixed feature direction
    C_perp = right_null(st["D"][2].reshape(-1, nt.shape["W3"][0]))
    if C_perp.shape[1] >= 2:
        a = top_right(st["H"][2]); cols = [emb(net, nt, {**zeros(nt), "W3": torch.outer(C_perp[:, k], a)}) for k in (0, 1)]
        bank["downstream_null"] = ortho(cols); info["downstream_null"] = dict(dim=int(C_perp.shape[1]))
    # selective unit: rarely active layer-3 unit, incoming row orthogonal to its active features
    act3 = masks[2].sum(0); live = torch.where(act3 > 0)[0]
    if len(live):
        j = int(live[act3[live].argmin()]); nul = right_null(st["H"][2][masks[2][:, j]])
        if nul.shape[1] >= 2:
            cols = []
            for k in (0, 1):
                M = zeros(nt); M["W3"][j] = nul[:, k]; cols.append(emb(net, nt, M))
            bank["selective"] = ortho(cols); info["selective"] = dict(unit=j, active=int(masks[2][:, j].sum()), freed=int(nul.shape[1]))
    # dead-unit incoming row (layer 3)
    dead3 = torch.where(act3 == 0)[0]
    if len(dead3):
        j = int(dead3[0]); cols = []
        for _ in (0, 1):
            M = zeros(nt); M["W3"][j] = rc(nt.shape["W3"][1]); cols.append(emb(net, nt, M))
        bank["dead_unit"] = ortho(cols); info["dead_unit"] = dict(unit=j)
    # certified multilayer branch
    bs = torch.load(os.path.join(RES, "branches", "mnist_mlp_w64_seed0_epoch0305_bottom_up.pt"), weights_only=False)["bases"]
    Bb = SC.full_basis(nt, bs)
    bank["branch"] = torch.stack([SC.unit(emb(net, nt, BR.split_blocks(nt, Bb @ rc(Bb.shape[1])))) for _ in (0, 1)], 1)
    bank["branch"], _ = torch.linalg.qr(bank["branch"]); info["branch"] = dict(dim=int(Bb.shape[1]))
    # tangent-flat: ker J_X off the branch (not an ocean)
    Kx = torch.load(os.path.join(RES, "branches", "mnist_mlp_w64_seed0_epoch0305_kerJX.pt"), weights_only=False)["basis"].double()
    cols = []
    for _ in (0, 1):
        v = Kx @ rc(Kx.shape[1]); v = v - Bb @ (Bb.T @ v); v = Kx @ (Kx.T @ v); v = v - Bb @ (Bb.T @ v)
        cols.append(SC.unit(emb(net, nt, BR.split_blocks(nt, v))))
    bank["tangent_flat"], _ = torch.linalg.qr(torch.stack(cols, 1)); info["tangent_flat"] = dict()
    return bank, info


def main():
    torch.set_num_threads(4)
    SC.R_CART = 1.5; SC.N_CART = 101; SC.RADII = np.logspace(-3, 0.6, 41)
    net, X, theta0, masks, F0, nt, _ = SC.setup("w64", 305)
    bank, info = build_bases(net, X, nt)
    print("sources:", json.dumps(info), flush=True)
    for name, Q in bank.items():
        c = torch.randn(Q.shape[1], 2, generator=GEN, dtype=torch.float64).to(DEV)
        P, _ = torch.linalg.qr(Q @ c)                                # two orthonormal directions spanning a random plane in the subspace
        SC.slice_grid(net, theta0, masks, X, F0, SC.unit(P[:, 0]), SC.unit(P[:, 1]), f"gal_{name}", dict(source=name, **info.get(name, {})))
    json.dump(info, open(os.path.join(OUT, "gallery_info.json"), "w"))


if __name__ == "__main__":
    main()
