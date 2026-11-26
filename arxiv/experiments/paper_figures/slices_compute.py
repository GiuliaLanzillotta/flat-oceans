"""Two-dimensional parameter slices for the paper's ray figures (paper helper, 2026-09-26).

For a trained MLP checkpoint and two unit Euclidean directions (u, v) in parameter space, evaluate on the 10k analysis
set the true output change ||F(theta0 + a u + b v) - F(theta0)||_X, the frozen (base-gate) change and the number of
gates that differ from the base point, on
  * a Cartesian grid (a, b) in [-R, R]^2 (island pictures), and
  * a polar grid: angle x log-spaced radius (radius maps).
Slices (all seed 0, high-accuracy checkpoints):
  relu_w64:  branch x branch, branch x tangent-flat complement, branch x generic, feature-null x branch, rescaling plane
  leaky_w64: branch x branch
  relu_w128 / leaky_w128 / gelu_w128, layer 3 only: kernel x kernel (ReLU, leaky) and 1e-3 tail x tail (GELU)
Writes results/paper_slices/<slice>.npz.  Reads only stored bases and factors; writes nothing else.
Run from experiments/ (venv): CUDA_VISIBLE_DEVICES=2 python paper_figures/slices_compute.py
"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import runs
from flatocean.analyses import branch as B
from flatocean.analyses import branch_rays as BR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(runs.HERE, "results", "paper_slices"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(runs.HERE, "results")
R_CART = 0.5; N_CART = 101
ANGLES = 64; RADII = np.logspace(-3, 1, 41)
GEN = torch.Generator().manual_seed(2026)


def x_norm(V):
    return float(V.pow(2).sum(1).mean().sqrt())


@torch.no_grad()
def evaluate(net, theta0, masks, X, F0, w):
    FL.set_flat(net, theta0 + w)
    f, Z, _ = net.forward(X, return_states=True)
    flips = sum(int(((z > 0) != m).sum()) for z, m in zip(Z, masks))
    frz = FL.frozen_forward(net, masks, X, torch.arange(X.shape[0], device=X.device))
    FL.set_flat(net, theta0)
    return x_norm(f - F0), x_norm(frz - F0), flips


@torch.no_grad()
def slice_grid(net, theta0, masks, X, F0, u, v, name, extra=None):
    t0 = time.time()
    a = np.linspace(-R_CART, R_CART, N_CART); cart = np.zeros((3, N_CART, N_CART))
    for i, ai in enumerate(a):
        for j, bj in enumerate(a):
            cart[:, i, j] = evaluate(net, theta0, masks, X, F0, ai * u + bj * v)
    th = np.linspace(0, 2 * np.pi, ANGLES, endpoint=False); pol = np.zeros((3, ANGLES, len(RADII)))
    for i, t in enumerate(th):
        d = float(np.cos(t)) * u + float(np.sin(t)) * v
        for j, r in enumerate(RADII):
            pol[:, i, j] = evaluate(net, theta0, masks, X, F0, float(r) * d)
    meta = dict(axis=a, angles=th, radii=RADII, F0_norm=x_norm(F0), theta_norm=float(theta0.norm()), seconds=time.time() - t0)
    if extra: meta.update(extra)
    np.savez(os.path.join(OUT, name + ".npz"), cart_true=cart[0], cart_frozen=cart[1], cart_flips=cart[2],
             pol_true=pol[0], pol_frozen=pol[1], pol_flips=pol[2], **meta)
    print(f"{name}: {meta['seconds']:.0f}s; centre true {cart[0, N_CART//2, N_CART//2]:.1e}; corner true {cart[0, 0, 0]:.1e}", flush=True)


def unit(w):
    return w / w.norm()


def random_in(basis_flat, embed):
    """Random unit direction in the span of the block-coordinate basis (p x k), embedded into flat parameters."""
    c = torch.randn(basis_flat.shape[1], generator=GEN, dtype=torch.float64)
    return unit(embed(basis_flat @ c))


def full_basis(nt, bases):
    return BR.torch.cat([torch.cat([torch.zeros(sum(nt.p(x) for x in nt.blocks[:j]), bases[b].shape[1], dtype=torch.float64), bases[b].double(),
                                    torch.zeros(sum(nt.p(x) for x in nt.blocks[j + 1:]), bases[b].shape[1], dtype=torch.float64)], 0)
                         for j, b in enumerate(nt.blocks) if bases[b].shape[1]], 1)


def setup(tag, epoch):
    run = runs.load_run("mnist", "mlp", 0, tag=tag, epoch=epoch, device=DEV)
    net = run["net"]; X = run["data"]["analysis"][0]
    theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X)
    with torch.no_grad():
        F0 = net(X)
    nt = B.Net(net, X)
    embed = lambda vblock: B.embed(net, nt, BR.split_blocks(nt, vblock.cpu())).to(DEV)
    return net, X, theta0, masks, F0, nt, embed


def w64_slices(tag, epoch, with_controls):
    name = f"mnist_mlp_{tag}_seed0"
    net, X, theta0, masks, F0, nt, embed = setup(tag, epoch)
    bases = torch.load(os.path.join(RES, "branches", f"{name}_epoch{epoch:04d}_bottom_up.pt"), weights_only=False)["bases"]
    Bb = full_basis(nt, bases)
    u1 = random_in(Bb, embed); u2 = random_in(Bb, embed); u2 = unit(u2 - float(u2 @ u1) * u1)
    dims = {b: int(bases[b].shape[1]) for b in bases}
    # sanity: a branch direction is frozen-flat
    _, frz, _ = evaluate(net, theta0, masks, X, F0, 0.1 * u1); print(f"{name} branch frozen change at r=0.1: {frz:.1e}", flush=True)
    slice_grid(net, theta0, masks, X, F0, u1, u2, f"{tag}_branch_branch", dict(dims=json.dumps(dims)))
    if with_controls:
        Kx = torch.load(os.path.join(RES, "branches", f"{name}_epoch{epoch:04d}_kerJX.pt"), weights_only=False)["basis"].double()
        c = torch.randn(Kx.shape[1], generator=GEN, dtype=torch.float64); v = Kx @ c
        v = v - Bb @ (Bb.T @ v); v = Kx @ (Kx.T @ v); v = v - Bb @ (Bb.T @ v)
        t = unit(embed(v)); t = unit(t - float(t @ u1) * u1)
        slice_grid(net, theta0, masks, X, F0, u1, t, f"{tag}_branch_tangent")
        g = unit(torch.randn(theta0.numel(), generator=GEN, dtype=torch.float64).to(DEV)); g = unit(g - float(g @ u1) * u1)
        slice_grid(net, theta0, masks, X, F0, u1, g, f"{tag}_branch_generic")
        fn = torch.load(os.path.join(RES, "branches", f"{name}_epoch{epoch:04d}_feature_null.pt"), weights_only=False)["bases"]
        f = random_in(full_basis(nt, fn), embed)
        slice_grid(net, theta0, masks, X, F0, f, u1, f"{tag}_featurenull_branch")
        # rescaling plane of one layer-2 unit: incoming row (with bias) up, outgoing column down
        sl = {n_: (a_, b_) for n_, a_, b_ in FL.param_slices(net)}; j = 5
        W2 = net.layers[1].weight; b2 = net.layers[1].bias; W3 = net.layers[2].weight
        e_in = torch.zeros_like(theta0); e_out = torch.zeros_like(theta0)
        a_, b_ = sl["layers.1.weight"]; e_in[a_ + j * W2.shape[1]: a_ + (j + 1) * W2.shape[1]] = W2[j]
        a_, b_ = sl["layers.1.bias"]; e_in[a_ + j] = b2[j]
        a_, b_ = sl["layers.2.weight"]; e_out[a_ + j: b_: W3.shape[1]] = W3[:, j]
        nin, nout = float(e_in.norm()), float(e_out.norm())
        slice_grid(net, theta0, masks, X, F0, unit(e_in), unit(e_out), f"{tag}_rescaling_unit{j}", dict(norm_in=nin, norm_out=nout, unit=j))


def w128_layer3(tag, epoch, mode):
    name = f"mnist_mlp{('_' + tag) if tag else ''}_seed0"
    net, X, theta0, masks, F0, nt, embed = setup(tag, epoch)
    z = np.load(os.path.join(RES, "persistence", name, f"epoch{epoch:04d}_3.npz"))
    s = int(z["s"]); sv = z["singular_values"]
    if mode == "kernel" and "V_kernel64" in z.files:
        V = torch.from_numpy(z["V_kernel64"]).double()
    elif mode == "kernel":                               # older factor format: kernel columns of V (float32) beyond the 1e-9 rank
        r = int((sv > 1e-9 * sv[0]).sum()); V = torch.from_numpy(z["V"][:, r:]).double()
    else:
        cols = np.where(sv / np.sqrt(s) <= 1e-3)[0]; V = torch.from_numpy(z["V"][:, cols]).double()
    m1, n = nt.shape["W3"][1], nt.shape["W3"][0]
    def to_flat(col):                                    # factor vectorization: index k*n + j <-> A[j, k]
        A = col.reshape(m1, n).T
        mats = {b: torch.zeros(nt.shape[b], dtype=torch.float64) for b in nt.blocks}; mats["W3"] = A
        return B.embed(net, nt, mats).to(DEV)
    u1 = unit(to_flat(V @ torch.randn(V.shape[1], generator=GEN, dtype=torch.float64)))
    u2 = unit(to_flat(V @ torch.randn(V.shape[1], generator=GEN, dtype=torch.float64))); u2 = unit(u2 - float(u2 @ u1) * u1)
    _, frz, _ = evaluate(net, theta0, masks, X, F0, 0.1 * u1); print(f"{name} layer-3 {mode} frozen change at r=0.1: {frz:.1e}", flush=True)
    slice_grid(net, theta0, masks, X, F0, u1, u2, f"{tag or 'relu'}_w128_L3_{mode}", dict(dim=int(V.shape[1]), mode=mode))


if __name__ == "__main__":
    torch.set_num_threads(4)
    if not os.path.exists(os.path.join(OUT, "w64_rescaling_unit5.npz")):
        w64_slices("w64", 305, True)
    if not os.path.exists(os.path.join(OUT, "leaky_w64_branch_branch.npz")):
        w64_slices("leaky_w64", 138, False)
    w128_layer3("", 40, "kernel")
    w128_layer3("leaky", 40, "kernel")
    w128_layer3("gelu", 22, "tail")
    w128_layer3("", 40, "tail")
