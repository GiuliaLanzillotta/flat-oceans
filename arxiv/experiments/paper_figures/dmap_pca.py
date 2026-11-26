"""The downstream-map clusters, shown as the pairwise-distance matrix itself (ReLU w128,
seed 0, layer 3): rows and columns are 500 samples (50 per class, ordered by class) of the
stratified dcluster subsample; colour = distance between the unit-normalized vec(D_i)
(dark = close).  At high accuracy the ten class blocks are already visible; by epoch 500
they are near-zero blocks against a uniform far background --- the clusters need no
projection to be seen.  Reads figures/atomic/dcluster_cache.npz.
Writes figures/atomic/dmap_pca.{png,pdf}."""
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S

OUT = os.path.join(S.F, "atomic"); os.makedirs(OUT, exist_ok=True)
z = np.load(os.path.join(S.HERE,"results","visualization","dcluster_cache.npz"))
y = z["y"]; n = len(y)
PER = 50
idx = np.concatenate([np.where(y == c)[0][:PER] for c in range(10)])

CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "dist", ["#f5d9a0", "#e8a23c", "#2a78d6", "#0d2c58"])  # sand = close, deep navy = far

fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.78), gridspec_kw=dict(
    wspace=0.10, left=0.055, right=0.88, top=0.86, bottom=0.115))
for ax, st, title in [(axs[0], "high_accuracy", "high accuracy"),
                      (axs[1], "late_endpoint", "epoch 500")]:
    D = squareform(z[f"d_{st}_3"])[np.ix_(idx, idx)]
    im = ax.imshow(D, cmap=CMAP, vmin=0, vmax=1.45, interpolation="nearest",
                   rasterized=True)
    ticks = np.arange(10) * PER + PER / 2
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(c) for c in range(10)], fontsize=4.6)
    if ax is axs[0]:
        ax.set_yticklabels([str(c) for c in range(10)], fontsize=4.6)
        ax.set_ylabel("samples by class", fontsize=5.6, labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("samples by class", fontsize=5.6, labelpad=1)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(S.C["ink3"]); sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=6.4, pad=2.5)
cax = fig.add_axes([0.905, 0.115, 0.022, 0.745])
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([0, 0.7, 1.4]); cb.ax.tick_params(labelsize=4.8, length=1.5)
cb.set_label(r"distance of $\mathrm{vec}\,D_i$", fontsize=5.4, labelpad=1.5)
cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(S.C["ink3"])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"dmap_pca.{ext}"), dpi=300)
print("wrote atomic/dmap_pca")
