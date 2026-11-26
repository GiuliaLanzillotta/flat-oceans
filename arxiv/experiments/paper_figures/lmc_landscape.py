"""2D loss and function landscape of an LMC pair across spawn epochs.  Rows: spawn epoch (parent P trained that long
before the split).  Columns: MSE loss; function change ||F-F_A||_X; frozen part C; activation transition T; all on a
log colour scale.  Plane through the two siblings A, B and the parent P; white curves are the two real training
trajectories P->A and P->B, projected onto the plane; markers/labels A, B, P.  Early spawns: the parent still has loss,
and the two children pull far apart.  Late spawns: the parent already sits in the basin and the children stay close.
Reads results/paper_slices/landscape_<model>_e*.npz.  Writes figures/lmc/landscape_<model>.{png,pdf}."""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S
SL = os.path.join(S.HERE, "results", "paper_slices"); OUT = os.path.join(S.HERE, "figures", "lmc"); os.makedirs(OUT, exist_ok=True)
COLS = [("mse", "MSE loss", "magma", LogNorm(1e-5, 1e-1)),
        ("fun", r"function $\|F-F_A\|_X$", "viridis", LogNorm(1e-2, 0.7)),
        ("froz", r"frozen part $C$", "viridis", LogNorm(1e-2, 0.7)),
        ("trans", r"transition $T$", "viridis", LogNorm(1e-2, 0.7))]
OUTLINE = [pe.withStroke(linewidth=1.6, foreground="black")]


def draw(ax, z, key, cmap, norm):
    xs, ys = z["xs"], z["ys"]; R = float(z["R"]); Px, Py = float(z["Px"]), float(z["Py"])
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    im = ax.imshow(np.maximum(z[key], norm.vmin), origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=norm)
    ax.contour(xs, ys, z[key], levels=np.geomspace(norm.vmin * 3, norm.vmax, 6), colors="white", linewidths=0.25, alpha=0.3)
    ax.plot(z["tAx"], z["tAy"], color="white", lw=1.2, alpha=0.9, zorder=4)
    ax.plot(z["tBx"], z["tBy"], color="white", lw=1.2, alpha=0.9, zorder=4)
    for (px, py), mk, name in [((0, 0), "o", "A"), ((R, 0), "s", "B"), ((Px, Py), "*", "P")]:
        ax.scatter([px], [py], s=(52 if mk == "*" else 30), marker=mk, c="white", edgecolors="black", linewidths=0.6, zorder=6)
        ax.annotate(name, (px, py), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="white", zorder=7, path_effects=OUTLINE)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
    return im


EPOCHS = [100, 150, 250]


def build(model):
    files = {int(os.path.basename(f).split("_e")[1].split(".")[0]): f for f in glob.glob(os.path.join(SL, f"landscape_{model}_e*.npz"))}
    spawns = [e for e in EPOCHS if e in files]
    fig, axs = plt.subplots(len(spawns), 4, figsize=(6.2, 1.6 * len(spawns) + 0.45),
                            gridspec_kw=dict(wspace=0.06, hspace=0.06, left=0.06, right=0.98, top=0.9, bottom=0.14))
    axs = np.atleast_2d(axs)
    ims = {}
    for r, E in enumerate(spawns):
        z = np.load(files[E])
        for c, (key, lab, cmap, norm) in enumerate(COLS):
            ims[c] = draw(axs[r, c], z, key, cmap, norm)
            if r == 0:
                axs[r, c].set_title(lab, fontsize=8.5, pad=3)
        axs[r, 0].set_ylabel(f"spawn ep. {E}", fontsize=8.5, color=S.C["ink2"])
    cb1 = fig.colorbar(ims[0], ax=axs[:, 0].tolist(), location="bottom", fraction=0.05, pad=0.04, aspect=16, shrink=0.8)
    cb1.set_label("MSE loss", fontsize=7.5); cb1.ax.tick_params(labelsize=6); cb1.outline.set_visible(False)
    cb2 = fig.colorbar(ims[3], ax=axs[:, 1:].ravel().tolist(), location="bottom", fraction=0.05, pad=0.04, aspect=50, shrink=0.85)
    cb2.set_label(r"distance from $F_A$ (function, frozen, transition)", fontsize=7.5); cb2.ax.tick_params(labelsize=6); cb2.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"landscape_{model}.{ext}"))
    print(f"wrote lmc/landscape_{model}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["mlp", "cnn"]:
        if glob.glob(os.path.join(SL, f"landscape_{m}_e*.npz")):
            build(m)
