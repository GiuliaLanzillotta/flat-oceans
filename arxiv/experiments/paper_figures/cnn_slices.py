"""2D parameter slices for the MNIST CNN.  Frozen forward = edited conv weights with the base ReLU gates fixed per
position (average pooling is linear).  Sources: the certified selective-sum branch (an ocean), the readout feature-null
(infinite-radius ocean), a generic direction and a random plane.  Writes results/paper_slices/cnn_<name>.npz.
Run: CUDA_VISIBLE_DEVICES=2 python paper_figures/cnn_slices.py"""
import os, sys, json
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import ocean_lib as O
from flatocean import runs
import slices_compute as SC

DEV = SC.DEV; OUT = SC.OUT; RES = os.path.join(runs.HERE, "results")
GEN = torch.Generator().manual_seed(3)
EPOCH = 500


def flat_params(net):
    return torch.cat([p.reshape(-1) for p in net.parameters()])


@torch.no_grad()
def set_flat(net, vec):
    i = 0
    for p in net.parameters():
        n = p.numel(); p.copy_(vec[i:i + n].reshape(p.shape)); i += n


@torch.no_grad()
def base_masks(net, X, block=2000):
    masks = None
    for i in range(0, X.shape[0], block):
        _, Z, _, _ = _states(net, X[i:i + block])
        g = [(z > 0) for z in Z]
        masks = g if masks is None else [torch.cat([a, b]) for a, b in zip(masks, g)]
    return masks


def _states(net, x):
    Z, H = [], []; h = x; ci = 0
    for a in net.arch:
        if a[0] == "conv":
            H.append(h); z = net.convs[ci](h); Z.append(z); h = torch.relu(z); ci += 1
        else:
            h = F.avg_pool2d(h, 2)
    return net.readout(h.flatten(1)), Z, H, h.flatten(1)


@torch.no_grad()
def frozen_forward(net, masks, x, idx):
    """Edited weights, base gates fixed per position."""
    h = x[idx]; ci = 0
    for a in net.arch:
        if a[0] == "conv":
            z = net.convs[ci](h); h = z * masks[ci][idx].to(z.dtype); ci += 1
        else:
            h = F.avg_pool2d(h, 2)
    return net.readout(h.flatten(1))


def xnorm(V):
    return float(V.pow(2).sum(1).mean().sqrt())


@torch.no_grad()
def evaluate(net, theta0, masks, X, F0, w, block=2000):
    set_flat(net, theta0 + w)
    s = X.shape[0]; dt = 0.0; dfz = 0.0; flips = 0
    dt_v = torch.zeros_like(F0); dz_v = torch.zeros_like(F0)
    for i in range(0, s, block):
        idx = torch.arange(i, min(i + block, s), device=X.device)
        out, Z, _, _ = _states(net, X[idx]); frz = frozen_forward(net, masks, X, idx)
        dt_v[idx] = out - F0[idx]; dz_v[idx] = frz - F0[idx]
        flips += sum(int(((z > 0) != masks[l][idx]).sum()) for l, z in enumerate(Z))
    set_flat(net, theta0)
    return xnorm(dt_v), xnorm(dz_v), flips


# ---- map a block-coordinate basis vector to a flat parameter perturbation (net.parameters() order)
def block_to_flat(net, blk, col):
    d = net.readout.out_features
    parts = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    if blk == "readout":
        m = net.readout.in_features
        A = col.reshape(m + 1, d)                                     # index k*d + c
        parts["readout.weight"] += A[:m].T
        parts["readout.bias"] += A[m]
    else:
        li = int(blk[4:]) - 1; conv = net.convs[li]; C = conv.out_channels; cin = conv.in_channels
        A = col.reshape(9 * cin + 1, C)                               # index k*C + c
        w = A[:9 * cin].reshape(cin, 3, 3, C).permute(3, 0, 1, 2)     # k = cin*9 + (kh*3+kw)
        parts[f"convs.{li}.weight"] += w
        parts[f"convs.{li}.bias"] += A[9 * cin]
    return torch.cat([parts[n].reshape(-1) for n, _ in net.named_parameters()])


def branch_flat(net, bases, gen):
    """Random combination of a stored block-coordinate branch basis -> flat parameter direction."""
    v = torch.zeros(sum(p.numel() for p in net.parameters()), dtype=torch.float64, device=DEV)
    for blk, B in bases.items():
        if B.shape[1] == 0:
            continue
        c = torch.randn(B.shape[1], generator=gen, dtype=torch.float64)
        col = (B.double() @ c)
        v = v + block_to_flat(net, blk, col.to(DEV))
    return v


def unit(w):
    return w / w.norm()


def main():
    torch.set_num_threads(4)
    SC.R_CART = 0.5; SC.N_CART = 61; SC.RADII = np.logspace(-3, 0.6, 31); SC.ANGLES = 48
    run = runs.load_run("mnist", "cnn", 0, epoch=EPOCH, device=DEV)
    net = run["net"]; X = run["data"]["analysis"][0][:2000]   # 2000 images suffice for the picture; CNN forward is heavy
    theta0 = flat_params(net).clone(); masks = base_masks(net, X)
    with torch.no_grad():
        F0, _, _, hf = _states(net, X)
    bases = torch.load(os.path.join(RES, "branches", f"mnist_cnn_seed0_epoch{EPOCH:04d}_branch_bottom_up.pt"), weights_only=False)["bases"]
    # validation: a branch direction must be flat in the frozen model
    ub = unit(branch_flat(net, bases, GEN))
    tr, fz, fl = evaluate(net, theta0, masks, X, F0, 0.05 * ub)
    print(f"CNN branch check: frozen@0.05={fz:.1e} true@0.05={tr:.1e} flips={fl}", flush=True)
    # readout feature-null: rows of the readout in the null of the pooled features -> infinite-radius ocean
    Hf = torch.cat([hf, torch.ones(X.shape[0], 1, dtype=hf.dtype, device=DEV)], 1)
    _, sv, Vh = torch.linalg.svd(Hf.double().cpu(), full_matrices=False); r = int((sv > 1e-9 * sv[0]).sum())
    N = Vh[r:].T                                                       # (m+1) x k, k = pooled-feature nullity
    def readout_dir(kk):
        A = torch.zeros(net.readout.in_features + 1, net.readout.out_features, dtype=torch.float64)
        A[:, kk % net.readout.out_features] = N[:, kk]
        return unit(block_to_flat(net, "readout", A.reshape(-1).to(DEV)))
    slices = {}
    # branch x branch (CNN ocean island)
    u1 = ub; u2 = unit(branch_flat(net, bases, GEN)); u2 = unit(u2 - float(u2 @ u1) * u1)
    slices["cnn_branch"] = (u1, u2)
    # feature-null x branch
    if N.shape[1] >= 2:
        slices["cnn_featurenull"] = (readout_dir(0), u1)
    # branch x generic
    g = unit(torch.randn(theta0.numel(), generator=GEN, dtype=torch.float64).to(DEV))
    slices["cnn_branch_generic"] = (u1, unit(g - float(g @ u1) * u1))
    # random slice
    g2 = unit(torch.randn(theta0.numel(), generator=GEN, dtype=torch.float64).to(DEV))
    slices["cnn_random"] = (g, unit(g2 - float(g2 @ g) * g))
    for name, (a, b) in slices.items():
        SC.slice_grid(net, theta0, masks, X, F0, a, b, name, dict(epoch=EPOCH))
    print("done", flush=True)


# make slices_compute use the CNN forward for the grid
def _patch_slicegrid():
    import types
    def evaluate_cnn(net, theta0, masks, X, F0, w):
        return evaluate(net, theta0, masks, X, F0, w)
    SC.evaluate = evaluate_cnn


if __name__ == "__main__":
    _patch_slicegrid()
    main()
