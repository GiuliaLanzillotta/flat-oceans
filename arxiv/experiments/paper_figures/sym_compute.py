"""Rescaling-symmetry planes: for a hidden unit, the plane of its incoming row (+bias) against its outgoing column.
For a homogeneous activation the exact rescaling orbit is flat; for GELU it is not a symmetry.  One slice per config,
saved to results/paper_slices/sym_<label>.npz with the incoming/outgoing norms so the orbit hyperbola can be drawn.
Run: CUDA_VISIBLE_DEVICES=2 python paper_figures/sym_compute.py"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import frozen_lib as FL
import slices_compute as SC

CONFIGS = [("leaky_w64", 138, "leaky_w64"), ("", 40, "relu_w128"), ("leaky", 40, "leaky_w128"), ("gelu", 22, "gelu_w128")]
J = 5   # unit index in layer 2


def build_rescaling(net, theta0, j):
    sl = {n_: (a_, b_) for n_, a_, b_ in FL.param_slices(net)}
    W2, b2, W3 = net.layers[1].weight, net.layers[1].bias, net.layers[2].weight
    e_in = torch.zeros_like(theta0); e_out = torch.zeros_like(theta0)
    a_, _ = sl["layers.1.weight"]; e_in[a_ + j * W2.shape[1]: a_ + (j + 1) * W2.shape[1]] = W2[j]
    a_, _ = sl["layers.1.bias"]; e_in[a_ + j] = b2[j]
    a_, b_ = sl["layers.2.weight"]; e_out[a_ + j: b_: W3.shape[1]] = W3[:, j]
    return e_in, e_out, float(e_in.norm()), float(e_out.norm())


def main():
    torch.set_num_threads(4)
    SC.R_CART = 0.6; SC.N_CART = 121; SC.RADII = np.logspace(-3, 0.6, 41)
    for tag, ep, label in CONFIGS:
        out = os.path.join(SC.OUT, f"sym_{label}.npz")
        if os.path.exists(out):
            print("skip", label, flush=True); continue
        net, X, theta0, masks, F0, nt, _ = SC.setup(tag, ep)
        e_in, e_out, nin, nout = build_rescaling(net, theta0, J)
        SC.slice_grid(net, theta0, masks, X, F0, SC.unit(e_in), SC.unit(e_out), f"sym_{label}",
                      dict(unit=J, norm_in=nin, norm_out=nout, label=label))
    print("done", flush=True)


if __name__ == "__main__":
    main()
