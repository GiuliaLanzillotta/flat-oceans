"""Ocean X-ray figures.  Each plot TYPE writes into its own folder under figures/xray/<type>/, and every figure is
rendered in the candidate palettes and (for slices) both orientations, on one fixed value->colour mapping
(LogNorm 1e-12..1) so all figures are comparable.  Reads results/paper_slices/*.npz.
Run from experiments/: python paper_figures/xray.py"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S

D = os.path.join(S.HERE, "results", "paper_slices")
XRAY = os.path.join(S.HERE, "figures", "xray")
FLOOR = 1e-16; VIS = 1e-3; VMIN = 1e-12; VMAX = 1.0
SHORE = "white"

PALETTES = {
    # Ab: brighter bathymetry (deep ocean is a bright medium blue, not dark); this is the default for the main text.
    "Ab": LinearSegmentedColormap.from_list("Ab", [(0,"#1f6fb0"),(0.28,"#4fa3d6"),(0.52,"#8fd0e8"),(0.70,"#cdeef2"),
            (0.75,"#f4f8ea"),(0.80,"#f8e08c"),(0.88,"#f3c64e"),(0.94,"#ea8f3c"),(1,"#c63a1e")]),
    "A": LinearSegmentedColormap.from_list("A", [(0,"#123f6e"),(0.30,"#2472ad"),(0.55,"#57add4"),(0.70,"#a8e0ea"),
            (0.75,"#eaf6f2"),(0.80,"#f2df9a"),(0.88,"#edc24a"),(0.94,"#df8a3a"),(1,"#9c2f1a")]),
    "G": LinearSegmentedColormap.from_list("G", [(0,"#12508f"),(0.35,"#3f86c4"),(0.60,"#8fc7e8"),(0.72,"#cfe6f2"),
            (0.75,"#f6f4ef"),(0.82,"#eda06a"),(0.90,"#d64b2c"),(1,"#6e1414")]),
    "E": LinearSegmentedColormap.from_list("E", [(0,"#0f6157"),(0.30,"#1f9c88"),(0.55,"#4fc39c"),(0.72,"#bfe6b0"),
            (0.78,"#f0ead0"),(0.86,"#e8c46a"),(0.94,"#cf7b3a"),(1,"#7a2f1a")]),
}
MAIN_KEYS = ["Ab", "A"]


def load(name):
    z = np.load(os.path.join(D, name + ".npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def save(fig, subdir, name):
    d = os.path.join(XRAY, subdir); os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    plt.close(fig); print("wrote", subdir + "/" + name)


def _shore(ax, X, Y, F):
    if F is not None and F.max() > 0:
        cs = ax.contour(X, Y, F, levels=[0.5], colors=[SHORE], linewidths=1.1)
        cs.set_path_effects([pe.Stroke(linewidth=1.9, foreground="#222"), pe.Normal()])


def _rotate(field, a, phi):
    n = field.shape[0]; g = np.linspace(a[0], a[-1], n); A, B = np.meshgrid(g, g)
    x = A * np.cos(phi) - B * np.sin(phi); y = A * np.sin(phi) + B * np.cos(phi)
    ci = (x - a[0]) / (a[-1] - a[0]) * (n - 1); ri = (y - a[0]) / (a[-1] - a[0]) * (n - 1)
    return map_coordinates(field, [ri.ravel(), ci.ravel()], order=1, mode="nearest").reshape(n, n)


def _canon_phi(z):
    th = z["angles"]; r = z["radii"]; fl = z["pol_flips"]
    rg = np.array([r[np.argmax(fl[i] > 0)] if (fl[i] > 0).any() else r[-1] for i in range(len(th))])
    return float(th[int(np.argmin(rg))])


def island(ax, z, cm, title=None, orient="random"):
    a = z["axis"]; T = np.maximum(z["cart_true"], FLOOR).T; F = z["cart_flips"].T.astype(float)
    if orient == "canonical" and z["cart_flips"].max() > 0:
        phi = _canon_phi(z); T = _rotate(T, a, phi); F = _rotate(F, a, phi)
    im = ax.pcolormesh(a, a, T, norm=LogNorm(VMIN, VMAX), cmap=cm, shading="gouraud", rasterized=True)
    _shore(ax, a, a, F)
    ax.set_aspect("equal"); t = round(float(a[-1]) * 0.66, 1)
    ax.set_xticks([-t, 0, t]); ax.set_yticks([-t, 0, t]); ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    if title:
        ax.set_title(title, fontsize=8.5, pad=3)
    return im


def logpolar(ax, z, cm, title=None):
    r = z["radii"]; th = np.degrees(z["angles"]); T = np.maximum(z["pol_true"], FLOOR)
    im = ax.pcolormesh(np.log10(r), th, T, norm=LogNorm(VMIN, VMAX), cmap=cm, shading="gouraud", rasterized=True)
    _shore(ax, np.log10(r), th, z["pol_flips"])
    ax.set_yticks([0, 90, 180, 270, 360]); ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    if title:
        ax.set_title(title, fontsize=8.5, pad=3)
    return im


def cbar(fig, im, axs, **kw):
    cb = fig.colorbar(im, ax=axs, **kw); cb.set_label("output change $\\|\\Delta F_X\\|_X$", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_visible(False); cb.ax.axhline(VIS, color="#222", lw=0.8)
    return cb


GALLERY = [
    ("gal_feature_null", "feature-null", "inf"), ("gal_zero_feature_out", "zero-feature readout", "inf"),
    ("gal_downstream_null", "downstream-null", "fin"), ("gal_selective", "selective unit", "fin"),
    ("gal_dead_unit", "dead-unit row", "fin"), ("gal_branch", "multilayer branch", "fin"),
    ("gal_tangent_flat", "tangent-flat", "zero"), ("gal_random", "random slice", "zero"),
]
TAG = {"inf": ("$r=\\infty$", "#123f6e"), "fin": ("$r>0$", "#12508f"), "zero": ("$r=0$", "#9c2f1a")}


def _slice_stats(z):
    """Per-slice info line: dimension of the source space (if known) and median shore / visible radius."""
    import json as _json
    r = np.asarray(z["radii"]); T = np.asarray(z["pol_true"], float); F = np.asarray(z["pol_flips"], float)
    def first(M, thr):
        out = []
        for a in range(M.shape[0]):
            idx = np.nonzero(M[a] > thr)[0]
            if idx.size:
                out.append(r[idx[0]])
        return float(np.median(out)) if out else float("nan")
    return first(F, 0), first(T, VIS)


def gallery_cartesian(key, orient):
    import json as _json
    cm = PALETTES[key]
    try:
        info = _json.load(open(os.path.join(D, "gallery_info.json")))
    except Exception:
        info = {}
    fig, axs = plt.subplots(2, 4, figsize=(6.0, 3.35), gridspec_kw=dict(wspace=0.22, hspace=0.60, left=0.05, right=0.89, top=0.91, bottom=0.125))
    axf = axs.ravel(); im = None
    for k, (name, title, rad) in enumerate(GALLERY):
        try:
            z = load(name)
        except FileNotFoundError:
            axf[k].set_visible(False); continue
        im = island(axf[k], z, cm, title, orient)
        txt, col = TAG[rad]
        axf[k].text(0.06, 0.94, txt, transform=axf[k].transAxes, ha="left", va="top", fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="white", lw=0.4))
        sh, vi = _slice_stats(z)
        dim = (info.get(name.replace("gal_", "")) or {}).get("dim")
        parts = ([f"dim {dim:,}"] if dim else []) +                 ([f"shore {sh:.2g}"] if np.isfinite(sh) else ["no crossing"]) +                 ([f"visible {vi:.2g}"] if np.isfinite(vi) else ["visible $>10$"])
        axf[k].text(0.5, -0.085, "  $\cdot$  ".join(parts), transform=axf[k].transAxes, ha="center", va="top",
                    fontsize=4.6, color="#5a6270")
    cbar(fig, im, list(axf), fraction=0.014, pad=0.02)
    save(fig, "gallery", f"gallery_{orient}_{key}")


def gallery_polar(key):
    cm = PALETTES[key]
    fig, axs = plt.subplots(2, 4, figsize=(6.0, 3.0), gridspec_kw=dict(wspace=0.30, hspace=0.42, left=0.06, right=0.89, top=0.91, bottom=0.12))
    axf = axs.ravel(); im = None
    for k, (name, title, rad) in enumerate(GALLERY):
        try:
            z = load(name)
        except FileNotFoundError:
            axf[k].set_visible(False); continue
        im = logpolar(axf[k], z, cm, title)
        if k % 4 == 0:
            axf[k].set_ylabel("angle (deg)", fontsize=7)
        if k >= 4:
            axf[k].set_xlabel("$\\log_{10}$ radius", fontsize=7)
    cbar(fig, im, list(axf), fraction=0.014, pad=0.02)
    save(fig, "polar", f"gallery_polar_{key}")


def taxonomy(key, orient):
    cm = PALETTES[key]
    items = [("w64_branch_branch", "exactly flat ocean"), ("w64_branch_tangent", "flat only to first order"), ("w64_branch_generic", "not flat")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.28, left=0.06, right=0.88, bottom=0.16, top=0.9))
    for ax, (n, t) in zip(axs, items):
        im = island(ax, load(n), cm, t, orient); ax.set_xlabel("branch direction", fontsize=7.5)
    axs[0].set_ylabel("second direction", fontsize=7.5)
    cbar(fig, im, list(axs), fraction=0.02, pad=0.02)
    save(fig, "taxonomy", f"taxonomy_{orient}_{key}")


# ---- activations: polar maps (top) and radius curves (bottom) for ReLU / leaky / GELU at width 128, layer 3
ACT = [("relu_w128_L3_kernel", "ReLU (exact ocean)"), ("leaky_w128_L3_kernel", "leaky ReLU (exact ocean)"), ("gelu_w128_L3_tail", "GELU (approx. ocean)")]


def activations_polar(key):
    cm = PALETTES[key]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.28, left=0.07, right=0.88, bottom=0.2, top=0.88))
    for k, (ax, (n, t)) in enumerate(zip(axs, ACT)):
        im = logpolar(ax, load(n), cm, t); ax.set_xlabel("$\\log_{10}$ radius", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("angle (deg)", fontsize=7.5)
    cbar(fig, im, list(axs), fraction=0.02, pad=0.02)
    save(fig, "activations", f"activations_polar_{key}")


def activations_radius():
    """Output change vs radius (true, frozen, gates) for the three activations; palette-independent line plot."""
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.32, left=0.07, right=0.93, bottom=0.2, top=0.88))
    for k, (ax, (n, t)) in enumerate(zip(axs, ACT)):
        z = load(n); r = z["radii"]; T = np.maximum(z["pol_true"], FLOOR)
        ax.fill_between(r, T.min(0), T.max(0), color=S.C["blue"], alpha=0.18, lw=0)
        ax.plot(r, np.median(T, 0), color=S.C["blue"], label="true")
        ax.plot(r, np.median(np.maximum(z["pol_frozen"], FLOOR), 0), color=S.C["orange"], ls=":", label="frozen")
        ax2 = ax.twinx(); ax2.plot(r, np.maximum(np.median(z["pol_flips"], 0), 0.5), color=S.C["ink3"], lw=0.9)
        ax2.set_yscale("log"); ax2.set_ylim(0.5, 1e6); ax2.spines["right"].set_visible(True); ax2.tick_params(axis="y", colors=S.C["ink3"], labelsize=6)
        ax2.set_ylabel("gates crossed", color=S.C["ink3"], fontsize=7) if k == 2 else ax2.set_yticklabels([])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-16, 10); ax.set_xlabel("radius $r$"); ax.set_title(t, fontsize=8.5)
        ax.axhline(VIS, color="#222", lw=0.5, ls="--"); ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
        if k == 0:
            ax.set_ylabel("output change"); ax.legend(loc="upper left", fontsize=6.5)
        else:
            ax.set_yticklabels([])
    save(fig, "activations", "activations_radius")


# ---- structural symmetry vs ocean: the rescaling-orbit plane (incoming row vs outgoing column of a unit)
SYM = [("sym_relu_w128", "ReLU, width 128"), ("sym_leaky_w128", "leaky ReLU, width 128"),
       ("sym_gelu_w128", "GELU, width 128"), ("sym_leaky_w64", "leaky ReLU, width 64")]


def symmetry(key):
    cm = PALETTES[key]
    have = [(n, t) for n, t in SYM if os.path.exists(os.path.join(D, n + ".npz"))]
    if not have:
        print("symmetry: no sym_*.npz yet"); return
    fig, axs = plt.subplots(1, len(have), figsize=(2.5 * len(have), 2.7), gridspec_kw=dict(wspace=0.3, left=0.06, right=0.9, bottom=0.16, top=0.88), squeeze=False)
    for ax, (n, t) in zip(axs[0], have):
        z = load(n); a = z["axis"]; T = np.maximum(z["cart_true"], FLOOR).T
        im = ax.pcolormesh(a, a, T, norm=LogNorm(VMIN, VMAX), cmap=cm, shading="gouraud", rasterized=True)
        nin, nout = float(z["norm_in"]), float(z["norm_out"])
        c = np.linspace(0.25, 4, 500); xa = (c - 1) * nin; yb = (1 / c - 1) * nout; m = (np.abs(xa) <= a[-1]) & (np.abs(yb) <= a[-1])
        ax.plot(xa[m], yb[m], color="#111", lw=1.4, label="exact orbit")
        tt = np.linspace(-a[-1], a[-1], 50); ax.plot(tt, -tt * nout / nin, color="#111", lw=0.9, ls="--", label="tangent")
        ax.set_aspect("equal"); ax.set_xlim(a[0], a[-1]); ax.set_ylim(a[0], a[-1]); ax.set_title(t, fontsize=8.5)
        ax.set_xlabel("incoming row", fontsize=7.5); ax.tick_params(labelsize=6.5)
    axs[0, 0].set_ylabel("outgoing column", fontsize=7.5); axs[0, 0].legend(loc="lower left", fontsize=6)
    cbar(fig, im, list(axs[0]), fraction=0.02, pad=0.02)
    save(fig, "symmetry", f"symmetry_{key}")




# ---- MNIST CNN slices (built by cnn_slices.py)
CNN = [("cnn_branch", "branch $\\times$ branch", "fin"), ("cnn_featurenull", "feature-null $\\times$ branch", "inf"),
       ("cnn_branch_generic", "branch $\\times$ generic", "fin"), ("cnn_random", "random slice", "zero")]


def cnn_gallery(key, orient):
    cm = PALETTES[key]
    have = [(n, t, r) for n, t, r in CNN if os.path.exists(os.path.join(D, n + ".npz"))]
    if not have:
        print("cnn: no cnn_*.npz yet"); return
    fig, axs = plt.subplots(1, len(have), figsize=(2.4 * len(have), 2.7), gridspec_kw=dict(wspace=0.3, left=0.05, right=0.9, bottom=0.16, top=0.86), squeeze=False)
    im = None
    for ax, (n, t, rad) in zip(axs[0], have):
        im = island(ax, load(n), cm, t, orient)
        txt, col = TAG[rad]
        ax.text(0.06, 0.94, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7, color="white", bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="white", lw=0.4))
    cbar(fig, im, list(axs[0]), fraction=0.02, pad=0.02)
    save(fig, "cnn", f"cnn_{orient}_{key}")


def cnn_polar(key):
    cm = PALETTES[key]
    have = [(n, t) for n, t, r in CNN if os.path.exists(os.path.join(D, n + ".npz"))]
    if not have:
        return
    fig, axs = plt.subplots(1, len(have), figsize=(2.4 * len(have), 2.5), gridspec_kw=dict(wspace=0.32, left=0.06, right=0.9, bottom=0.2, top=0.86), squeeze=False)
    for k, (ax, (n, t)) in enumerate(zip(axs[0], have)):
        im = logpolar(ax, load(n), cm, t); ax.set_xlabel("$\\log_{10}$ radius", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("angle (deg)", fontsize=7.5)
    cbar(fig, im, list(axs[0]), fraction=0.02, pad=0.02)
    save(fig, "polar", f"cnn_polar_{key}")



# ---- radius-against-angle polar LINE plots (visible-drift radius and first-crossing radius as closed curves)
def radius_curves(z):
    th = z["angles"]; r = z["radii"]; fl = z["pol_flips"]; tr = z["pol_true"]
    rg = np.array([r[np.argmax(fl[i] > 0)] if (fl[i] > 0).any() else r[-1] for i in range(len(th))])
    rv = np.array([r[np.argmax(tr[i] > VIS)] if (tr[i] > VIS).any() else r[-1] for i in range(len(th))])
    thc = np.append(th, th[0])
    return thc, np.log10(np.append(rg, rg[0])), np.log10(np.append(rv, rv[0]))


def _polar_axes(ax):
    ax.set_rlim(-3.2, 0.8); ax.set_rticks([-3, -2, -1, 0]); ax.set_yticklabels(["$10^{-3}$", "", "$10^{-1}$", "$1$"], fontsize=6)
    ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(["0", "90", "180", "270"], fontsize=6.5); ax.tick_params(pad=2)
    ax.set_rlabel_position(45)


def radius_polar_one(ax, z, color, label):
    thc, rg, rv = radius_curves(z)
    ax.plot(thc, rv, color=color, lw=1.5, label=label)                       # solid = drift > 1e-3 (visible radius)
    ax.plot(thc, rg, color=color, lw=1.0, ls=(0, (3, 2)), alpha=0.85)         # dashed = first gate crossing
    _polar_axes(ax)


def _ramp(n, cmap="viridis"):
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.15, 0.85, n)]


def GROUPS():
    w = _ramp(4, "viridis"); dp = _ramp(4, "plasma")
    return [
        ("activations (width 128)", [("relu_w128_L3_kernel", "ReLU", S.C["blue"]), ("leaky_w128_L3_kernel", "leaky", S.C["orange"]), ("gelu_w128_L3_tail", "GELU", S.C["aqua"])]),
        ("width (ReLU MLP)", [("polar_w64", "64", w[0]), ("polar_w128", "128", w[1]), ("polar_w256", "256", w[2]), ("polar_w512", "512", w[3])]),
        ("depth (ReLU MLP)", [("polar_d2", "2", dp[0]), ("polar_w128", "3", dp[1]), ("polar_d4", "4", dp[2]), ("polar_d5", "5", dp[3])]),
        ("architecture", [("polar_w128", "MLP", S.C["blue"]), ("cnn_branch", "CNN", S.C["orange"])]),
    ]


def radius_polar_groups():
    groups = [(t, [(n, lab, c) for n, lab, c in items if os.path.exists(os.path.join(D, n + ".npz"))]) for t, items in GROUPS()]
    groups = [(t, it) for t, it in groups if it]
    fig, axs = plt.subplots(1, len(groups), figsize=(2.7 * len(groups), 3.1), subplot_kw=dict(projection="polar"),
                            gridspec_kw=dict(wspace=0.55, bottom=0.24, top=0.78, left=0.03, right=0.97))
    if len(groups) == 1:
        axs = [axs]
    for ax, (t, items) in zip(axs, groups):
        for n, lab, c in items:
            radius_polar_one(ax, load(n), c, lab)
        ax.set_title(t, fontsize=8.5, pad=20)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(len(items), 4), fontsize=6.3, handlelength=1.4, columnspacing=1.0)
    fig.suptitle("radius against angle:  solid = visible drift $>10^{-3}$,  dashed = first gate crossing", fontsize=8.5, y=1.0)
    save(fig, "radius_polar", "radius_polar_by_group")


def radius_polar_gallery():
    fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.6), subplot_kw=dict(projection="polar"),
                            gridspec_kw=dict(wspace=0.5, hspace=0.55, bottom=0.06, top=0.86, left=0.04, right=0.96))
    axf = axs.ravel()
    for k, (name, title, rad) in enumerate(GALLERY):
        try:
            z = load(name)
        except FileNotFoundError:
            axf[k].set_visible(False); continue
        radius_polar_one(axf[k], z, S.C["blue"], "")
        axf[k].set_title(title, fontsize=8, pad=16)
    fig.suptitle("radius against angle per source:  solid = visible drift $>10^{-3}$,  dashed = first gate crossing", fontsize=8.5, y=0.99)
    save(fig, "radius_polar", "radius_polar_gallery")


# ---- symmetry vs ocean vs random slice (one network)
def symmetry_vs_ocean(key):
    cm = PALETTES[key]
    sname = "sym_relu_w64_wide" if os.path.exists(os.path.join(D, "sym_relu_w64_wide.npz")) else None
    if sname is None:
        print("sym_vs_ocean: no wide symmetry slice yet"); return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), gridspec_kw=dict(wspace=0.3, left=0.05, right=0.9, bottom=0.16, top=0.88))
    z = load(sname); a = z["axis"]; T = np.maximum(z["cart_true"], FLOOR).T
    im = axs[0].pcolormesh(a, a, T, norm=LogNorm(VMIN, VMAX), cmap=cm, shading="gouraud", rasterized=True)
    nin, nout = float(z["norm_in"]), float(z["norm_out"])
    c = np.linspace(0.2, 5, 600); xa = (c - 1) * nin; yb = (1 / c - 1) * nout; m = (np.abs(xa) <= a[-1]) & (np.abs(yb) <= a[-1])
    axs[0].plot(xa[m], yb[m], color="#111", lw=1.5, label="exact orbit")
    tt = np.linspace(-a[-1], a[-1], 50); axs[0].plot(tt, -tt * nout / nin, color="#111", lw=0.9, ls="--", label="tangent")
    axs[0].set_aspect("equal"); axs[0].set_xlim(a[0], a[-1]); axs[0].set_ylim(a[0], a[-1]); axs[0].legend(loc="lower left", fontsize=6)
    axs[0].set_title("structural symmetry", fontsize=8.5); axs[0].set_xlabel("incoming row", fontsize=7.5); axs[0].set_ylabel("outgoing column", fontsize=7.5)
    island(axs[1], load("gal_branch"), cm, "flat ocean")
    island(axs[2], load("gal_random"), cm, "random slice")
    cbar(fig, im, list(axs), fraction=0.02, pad=0.02)
    save(fig, "symmetry", f"sym_vs_ocean_{key}")


if __name__ == "__main__":
    for key in MAIN_KEYS:
        for orient in ("canonical", "random"):
            gallery_cartesian(key, orient); taxonomy(key, orient); cnn_gallery(key, orient)
        gallery_polar(key); activations_polar(key); symmetry(key); cnn_polar(key); symmetry_vs_ocean(key)
    activations_radius(); radius_polar_groups(); radius_polar_gallery()
