"""Rescaling orbit for ReLU width 64 on the gallery grid (radius 1.5) so it matches the ocean/random panels."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slices_compute as SC, sym_compute as SY
torch.set_num_threads(4)
SC.R_CART = 1.5; SC.N_CART = 101; SC.RADII = np.logspace(-3, 0.6, 41)
net, X, theta0, masks, F0, nt, _ = SC.setup("w64", 305)
e_in, e_out, nin, nout = SY.build_rescaling(net, theta0, SY.J)
SC.slice_grid(net, theta0, masks, X, F0, SC.unit(e_in), SC.unit(e_out), "sym_relu_w64_wide", dict(norm_in=nin, norm_out=nout))
