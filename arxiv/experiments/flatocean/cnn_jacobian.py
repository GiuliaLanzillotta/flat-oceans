"""Tied convolutional Jacobian blocks for the Tier 1 CNN (TIER1_PIPELINE.md, "The theory extends to a convolution").

For gated conv layer l with shared kernel and bias (parameters A = [W | b], column-major
coordinates k * c_out + c with k the patch coordinate (ci * 9 + kh * 3 + kw, then the constant 1
for the bias) and c the output channel), the row (i, a) of J_l is

    sum_r p_{i,r}^T kron (D_{i,r}^T e_a),

with p_{i,r} the input patch at position r (with the constant 1) and D_{i,r} the linear map from
that position's preactivation to the output through the frozen tail (base masks, pooling, readout).
D is obtained by autodiff of the frozen tail (one backward pass per output coordinate a, all
samples of a chunk at once, since samples do not interact), patches by `unfold`.

The readout block has rows h_i^T kron e_a^T with h_i the flattened pooled features and the
constant 1 (index k * d + c), as for the MLP.

Validation (`python cnn_jacobian.py test`, float64): on a tiny CNN with padding and pooling,
rows against per-sample autodiff of the true network (kernel-only and bias-only columns),
the adjoint identity <J u, v> = <u, J^T v> with J^T v from a vector-Jacobian product of the
true network, and the tied/untied identity J_CNN = J_local T with T the replication of a
shared kernel into locally connected coordinates; then rows of the production CNN (seed 0,
epoch 500) against per-sample autodiff on a handful of samples for every block.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
def conv_blocks(net):
    return [f"conv{i + 1}" for i in range(len(net.convs))] + ["readout"]


def block_size(net, blk):
    if blk == "readout":
        return (net.readout.in_features + 1) * net.readout.out_features
    conv = net.convs[int(blk[4:]) - 1]
    return (9 * conv.in_channels + 1) * conv.out_channels


def _forward_states(net, x):
    """True forward on x: output, preactivations Z, layer inputs H, pooled-flattened readout input."""
    Z, H = [], []; h = x; ci = 0
    for a in net.arch:
        if a[0] == "conv":
            H.append(h); z = net.convs[ci](h); Z.append(z); h = torch.relu(z); ci += 1
        else:
            h = F.avg_pool2d(h, 2)
    hf = h.flatten(1)
    return net.readout(hf), Z, H, hf


def downstream(net, masks, idx, l, z_shape, device):
    """D for the chunk idx: (b, d, C, H, W) = Jacobian of the output with respect to the
    preactivation of gated layer l, through the frozen tail (base masks), by d backward passes."""
    b = len(idx); d = net.readout.out_features
    z = torch.zeros((b,) + tuple(z_shape[1:]), dtype=torch.float64, device=device, requires_grad=True)
    with torch.enable_grad():
        h = z * masks[l][idx].to(z.dtype)
        out = _tail(net, masks, idx, l, h)
        D = torch.stack([torch.autograd.grad(out[:, a].sum(), z, retain_graph=(a < d - 1))[0] for a in range(d)], 1)
    return D                                                    # b x d x C x H x W


def _tail(net, masks, idx, l, v):
    """Differentiable copy of frozen_lib.frozen_tail_linear for the CNN (no no_grad)."""
    h = v; ci = 0; started = False
    for a in net.arch:
        if a[0] == "conv":
            if started:
                h = F.conv2d(h, net.convs[ci].weight, padding=1) * masks[ci][idx].to(h.dtype)
            if ci == l:
                started = True
            ci += 1
        else:
            if started:
                h = F.avg_pool2d(h, 2)
    return F.linear(h.flatten(1), net.readout.weight)


@torch.no_grad()
def cnn_rows(net, masks, X, idx, blk):
    """Rows (i, a) for samples idx of the tied Jacobian block `blk`: (b*d) x p, column-major coordinates."""
    x = X[idx]; d = net.readout.out_features; b = len(idx)
    if blk == "readout":
        _, _, _, hf = _forward_states(net, x)
        h = torch.cat([hf, torch.ones(b, 1, dtype=hf.dtype, device=hf.device)], 1)         # b x (m+1)
        eye = torch.eye(d, dtype=h.dtype, device=h.device)
        return torch.einsum("ik,ac->iakc", h, eye).reshape(b * d, -1)                        # index k*d + c
    l = int(blk[4:]) - 1
    _, Z, H, _ = _forward_states(net, x)
    P = F.unfold(H[l], 3, padding=1)                                                          # b x 9c_in x R
    P = torch.cat([P, torch.ones(b, 1, P.shape[2], dtype=P.dtype, device=P.device)], 1)      # b x (9c_in+1) x R
    with torch.enable_grad():
        D = downstream(net, masks, idx, l, Z[l].shape, x.device)                              # b x d x C x H x W
    D = D.reshape(b, d, D.shape[2], -1)                                                       # b x d x C x R
    rows = torch.einsum("ikr,iacr->iakc", P, D)                                               # b x d x (9c_in+1) x C
    return rows.reshape(b * d, -1)                                                            # index k*C + c


def block_index_map(net, blk):
    """Flat-parameter index of every block coordinate (column-major order above)."""
    sl = {name: (s, e) for name, s, e in FL.param_slices(net)}
    if blk == "readout":
        W = net.readout.weight; d, m = W.shape; ws, _ = sl["readout.weight"]; bs, _ = sl["readout.bias"]
        idx = np.empty((m + 1) * d, dtype=np.int64)
        for k in range(m):
            idx[k * d: (k + 1) * d] = ws + np.arange(d) * m + k
        idx[m * d:] = bs + np.arange(d)
        return idx
    l = int(blk[4:]) - 1; conv = net.convs[l]
    C, cin = conv.out_channels, conv.in_channels; K = 9 * cin
    ws, _ = sl[f"convs.{l}.weight"]; bs, _ = sl[f"convs.{l}.bias"]
    idx = np.empty((K + 1) * C, dtype=np.int64)
    for k in range(K):
        idx[k * C: (k + 1) * C] = ws + np.arange(C) * K + k              # weight (C, cin, 3, 3) row-major: c*K + k
    idx[K * C:] = bs + np.arange(C)
    return idx


# ----------------------------------------------------------------------------- validation
def _block_param_slices(net, blk):
    """(name, start, end) of the block's parameter tensors in the flat vector, and the tensors."""
    prefix = "readout" if blk == "readout" else f"convs.{int(blk[4:]) - 1}."
    sl = [t for t in FL.param_slices(net) if t[0].startswith(prefix)]
    params = dict(net.named_parameters())
    return sl, [params[t[0]] for t in sl]


def _gather_block(net, blk, grads, device):
    """Scatter per-tensor gradients into the flat vector and read them out in block coordinates."""
    sl, _ = _block_param_slices(net, blk)
    full = torch.zeros(sum(q.numel() for q in net.parameters()), dtype=torch.float64, device=device)
    for (name, s, e), g in zip(sl, grads):
        full[s:e] = g.reshape(-1)
    return full[torch.as_tensor(block_index_map(net, blk), device=device)]


def _per_sample_autodiff_rows(net, X, idx, blk):
    """Rows by autodiff of the TRUE network output with respect to the block's parameters, one (i, a) at a time."""
    _, params = _block_param_slices(net, blk); d = net.readout.out_features; rows = []
    for i in idx.tolist():
        out = net(X[i:i + 1])
        for a in range(d):
            rows.append(_gather_block(net, blk, torch.autograd.grad(out[0, a], params, retain_graph=True), X.device))
    return torch.stack(rows)


def _tiny_net(device):
    from flatocean import tier1_train as T
    arch = [("conv", 2, 3), ("pool",), ("conv", 3, 4), ("pool",)]
    net = T.CNN(arch, 3, init_seed=1, dtype=torch.float64, in_hw=8).to(device)
    with torch.no_grad():
        for c in net.convs:
            c.bias.copy_(0.1 * torch.randn(c.bias.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2)))
    return net


def _test_tiny(device):
    torch.manual_seed(0)
    net = _tiny_net(device); X = torch.randn(6, 2, 8, 8, dtype=torch.float64, device=device)
    masks = FL.base_masks(net, X); idx = torch.arange(6, device=device)
    worst = 0.0
    for blk in conv_blocks(net):
        J = cnn_rows(net, masks, X, idx, blk)
        Ja = _per_sample_autodiff_rows(net, X, idx, blk)
        rel = ((J - Ja).norm() / Ja.norm()).item(); worst = max(worst, rel)
        # kernel-only and bias-only columns
        p = J.shape[1]; C = net.readout.out_features if blk == "readout" else net.convs[int(blk[4:]) - 1].out_channels
        bias_cols = torch.arange(p - C, p, device=device); ker_cols = torch.arange(0, p - C, device=device)
        rb = ((J[:, bias_cols] - Ja[:, bias_cols]).norm() / Ja[:, bias_cols].norm()).item()
        rk = ((J[:, ker_cols] - Ja[:, ker_cols]).norm() / Ja[:, ker_cols].norm()).item()
        # adjoint test: <J u, v> = <u, J^T v>, J^T v from a VJP of the true network
        g = torch.Generator().manual_seed(3)
        u = torch.randn(p, dtype=torch.float64, generator=g).to(device); v = torch.randn(J.shape[0], dtype=torch.float64, generator=g).to(device)
        lhs = (J @ u) @ v
        _, params = _block_param_slices(net, blk)
        out = net(X); JTv = _gather_block(net, blk, torch.autograd.grad((out.reshape(-1) * v).sum(), params), device)
        ra = (abs(lhs - u @ JTv) / abs(lhs)).item(); worst = max(worst, ra, rb, rk)
        print(f"  tiny {blk}: p={p} rows-vs-autodiff {rel:.1e} (kernel {rk:.1e}, bias {rb:.1e}), adjoint {ra:.1e}")
    # tied/untied identity on conv1: J_local has a separate kernel per position; J_CNN = J_local T
    blk = "conv1"; l = 0
    x = X; d = net.readout.out_features; b = x.shape[0]
    _, Z, H, _ = _forward_states(net, x)
    P = F.unfold(H[l], 3, padding=1); P = torch.cat([P, torch.ones(b, 1, P.shape[2], dtype=P.dtype, device=device)], 1)
    with torch.enable_grad():
        D = downstream(net, masks, idx, l, Z[l].shape, device).reshape(b, d, Z[l].shape[1], -1)
    R = P.shape[2]; K1 = P.shape[1]; C = D.shape[2]
    J_local = torch.einsum("ikr,iacr->iarkc", P, D).reshape(b * d, R * K1 * C)     # untied coordinates (r, k, c)
    T = torch.eye(K1 * C, dtype=torch.float64, device=device).repeat(R, 1)          # replication: (r, k, c) <- (k, c)
    J_cnn = cnn_rows(net, masks, X, idx, blk)
    rt = ((J_local @ T - J_cnn).norm() / J_cnn.norm()).item(); worst = max(worst, rt)
    # K_CNN = {a : T a in K_local}: check on a basis of ker J_cnn (tiny case may have a large kernel)
    U, S, Vh = torch.linalg.svd(J_cnn); r = int((S > 1e-12 * S[0]).sum()); Kc = Vh[r:].T
    resid = (J_local @ (T @ Kc)).norm().item() / max((T @ Kc).norm().item(), 1e-300)
    print(f"  tied/untied: ||J_local T - J_CNN||/||J_CNN|| = {rt:.1e}; dim K_CNN = {Kc.shape[1]}, ||J_local T a|| / ||T a|| on K_CNN = {resid:.1e}")
    assert worst < 1e-11 and resid < 1e-10, (worst, resid)
    print("cnn_jacobian: tiny-case tests passed")


def _test_production(device):
    run = O.load_run("cifar10", "cnn", 0, epoch=500, device=device); net = run["net"]
    X = run["data"]["analysis"][0][:3]; masks = FL.base_masks(net, X); idx = torch.arange(3, device=device)
    for blk in conv_blocks(net):
        J = cnn_rows(net, masks, X, idx, blk); Ja = _per_sample_autodiff_rows(net, X, idx, blk)
        rel = ((J - Ja).norm() / Ja.norm()).item()
        print(f"  production {blk}: p={J.shape[1]} rows-vs-autodiff {rel:.1e}")
        assert rel < 1e-11, rel
    print("cnn_jacobian: production-row tests passed")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _test_tiny(dev); _test_production(dev)
