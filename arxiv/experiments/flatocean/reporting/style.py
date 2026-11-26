"""Shared figure style of the paper and the memos (WRITING_STYLE.md): colours, stage names, rcParams, save and panel
helpers.  Extracted unchanged from make_tier1_figures.py on 2026-09-25.  Output directories: figures/ and tables/ under the
experiments directory, or FLATOCEAN_FIGURES_DIR / FLATOCEAN_TABLES_DIR."""
import argparse
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from flatocean import runs

HERE = runs.HERE
F = os.environ.get("FLATOCEAN_FIGURES_DIR", os.path.join(HERE, "build", "figures")); os.makedirs(F, exist_ok=True)
TABLES = os.environ.get("FLATOCEAN_TABLES_DIR", os.path.join(HERE, "build", "tables")); os.makedirs(TABLES, exist_ok=True)
RT = os.path.join(runs.RESULTS_ROOT, "transition"); RA = os.path.join(runs.RESULTS_ROOT, "atomic")

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", purple="#7b5cd6",
         ink="#0b0b0b", ink2="#52514e", ink3="#8a8985", grid="#e6e5e1", pale="#eef3fb", vis="#d9d8d4")

LAYER_COLS = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["purple"]]

STAGE_RAMP = matplotlib.colors.LinearSegmentedColormap.from_list("stage", ["#c9dcf5", "#6da7ec", "#2a78d6", "#154c93", "#0d2c58"])

STAGE_ORDER = ["initialization", "early_fitting", "high_accuracy", "mse_interpolation", "epoch_100", "epoch_250", "late_endpoint"]

STAGE_LABEL = {"initialization": "init.", "early_fitting": "early fit", "high_accuracy": "99.9% acc.",
               "mse_interpolation": "MSE $<10^{-4}$", "epoch_100": "ep. 100", "epoch_250": "ep. 250", "late_endpoint": "ep. 500"}

DATASET_LABEL = {("mnist", "mlp"): "MNIST MLP", ("cifar10", "cnn"): "CIFAR-10 CNN", ("cifar10", "mlp"): "CIFAR-10 MLP (PCA-256)"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"))
    plt.close(fig); print("wrote", name)

def panel(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")

def layer_name(model, l, L):
    if model == "cnn":
        return f"conv{l+1}"
    return f"L{l+1}"

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "cm", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["ink3"], "axes.linewidth": 0.6,
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "grid.color": C["grid"], "grid.linewidth": 0.5, "axes.grid": False, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "legend.frameon": False,
})
