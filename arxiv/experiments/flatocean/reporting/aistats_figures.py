"""[ported from make_aistats_figures.py on 2026-09-25; command: python -m flatocean aistats-figures]
Single-column figures for flat_ocean_aistats.tex from the Tier 1 MNIST results (shared style of
make_tier1_figures.py).

    python make_aistats_figures.py atomic rays transition persistence
"""
import json
import os
import sys

import numpy as np
from flatocean import runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flatocean.reporting.style import C, save

R = runs.RESULTS_ROOT
SEEDS = (0, 1, 2)
STAGES = [("high_accuracy", "0.1% err."), ("mse_interpolation", "interp."), ("late_endpoint", "ep. 500")]
ALL_STAGES = ["initialization", "early_fitting", "high_accuracy", "mse_interpolation", "epoch_100", "epoch_250", "late_endpoint"]
STAGE_TICKS = ["init.", "early", "0.1%", "interp.", "100", "250", "500"]
COLW = 3.25






# ----------------------------------------------------------------------------- atomic decomposition


# ----------------------------------------------------------------------------- structured rays




# ----------------------------------------------------------------------------- transition potential (absolute radius, provisional)


# ----------------------------------------------------------------------------- transition potential at block-relative radii




# ----------------------------------------------------------------------------- normalized margins (appendix, full width)
def fig_margins():
    """Sign-conditional cumulative distribution of the normalized margin: share of the ACTIVE (resp. INACTIVE) gates of
    a layer with |m| <= x, each histogram divided by its own sign total, so a change of the active population does
    not masquerade as a change of the margin distribution.  The active share is printed in each panel."""
    stages = [("initialization", "init.", 0.25), ("high_accuracy", "0.1% err.", 0.6), ("late_endpoint", "ep. 500", 1.0)]
    edges = 10.0 ** (np.arange(161) / 10.0 - 15.0)
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0), sharey=True); summary = {}
    for l, ax in enumerate(axes):
        note = []
        for st, lab, shade in stages:
            H = np.array([np.load(os.path.join(R, "transition_v2", f"mnist_mlp_seed{s}_counts.npz"))[f"{st}_hist_by_sign"][l] for s in SEEDS], dtype=float)
            prev = H[:, 1].sum(1) / H.sum(axis=(1, 2))                  # active share per seed
            note.append(f"{lab}: {100 * np.median(prev):.0f}%")
            for sign, col, ls in ((1, C["blue"], "-"), (0, C["orange"], "--")):
                cdf = np.cumsum(H[:, sign], 1) / H[:, sign].sum(1, keepdims=True)
                ax.plot(edges[1:], np.median(cdf, 0), color=col, ls=ls, alpha=shade, lw=1.2,
                        label=f"{'active' if sign else 'inactive'}, {lab}" if l == 0 else None)
                j = int(np.argmin(np.abs(edges[1:] - 1e-2)))
                summary[(l + 1, st, "active" if sign else "inactive")] = (float(np.median(cdf[:, j])), float(cdf[:, j].min()), float(cdf[:, j].max()), float(np.median(prev)))
        ax.text(0.03, 0.97, "active share\n" + "\n".join(note), transform=ax.transAxes, fontsize=5.8, va="top", color=C["ink2"])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-6, 1); ax.set_ylim(1e-7, 1.5); ax.set_xticks([1e-6, 1e-4, 1e-2])
        ax.set_title(f"L{l + 1}", pad=2); ax.set_xlabel(r"normalized margin $|m|$")
    axes[0].set_ylabel(r"share of same-sign gates below $|m|$", fontsize=7)
    axes[0].legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.8, labelspacing=0.15)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "paper_margins_mnist")
    print("sign-conditional share of gates with |m| <= 1e-2 (median [min, max] over seeds) and active share:")
    for k, v in summary.items():
        print(f"  L{k[0]} {k[1]:15s} {k[2]:9s} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]   active share {v[3]:.3f}")


# ----------------------------------------------------------------------------- persistence geometry
PAIRS = [("high_accuracy->mse_interpolation", r"0.1%$\to$interp."), ("mse_interpolation->epoch_100", r"interp.$\to$100"),
         ("epoch_100->epoch_250", r"100$\to$250"), ("epoch_250->late_endpoint", r"250$\to$500"),
         ("high_accuracy->late_endpoint", r"0.1%$\to$500")]








