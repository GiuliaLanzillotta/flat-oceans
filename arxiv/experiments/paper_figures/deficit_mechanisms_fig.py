"""The deficit mechanism per activation family (Giulia 2026-10-04): ReLU's deficit is
selective units; leaky ReLU's is cross-neuron readout-coupled cancellation; GELU has no
exact kernel but class-clustered downstream maps.

(a) ReLU family: selective space against the exact atomic kernel, every (layer, stage, seed)
    with a nonzero kernel, MLP L2/L3 and CNN conv5/readout -- all on the diagonal.
(b) Composition of the last hidden-layer kernel: unit-level (sand) against cross-neuron
    (navy), ReLU against leaky ReLU.
(c) Class purity of the downstream k-means (G=10), first fitted stage to epoch 500, per
    activation and layer: class structure needs no exact kernel.

Reads results/atomic/mnist_mlp_seed*_{gated_null,}.json, results/branches/mnist_cnn_seed*_
{patch_spaces,branch}.json, results/neuronwise/*_s10000.json, results/clusters/*_partitions.json.
Writes figures/atomic/deficit_mechanisms.{png,pdf}."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S

RA = os.path.join(S.HERE, "results", "atomic"); RB = os.path.join(S.HERE, "results", "branches")
RN = os.path.join(S.HERE, "results", "neuronwise"); RC = os.path.join(S.HERE, "results", "clusters")
OUT = os.path.join(S.F, "atomic"); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.95))
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.40, wspace=0.42)

# ---- (a) ReLU: selective space vs exact kernel, on the diagonal
ax = axes[0]
pts = {"mlp": [], "conv5": [], "readout": []}
TAGS = ["", "_w32", "_w64", "_w256", "_w512", "_d2", "_d4", "_d5"]
for tag in TAGS:
    for s in SEEDS:
        fg = os.path.join(RA, f"mnist_mlp{tag}_seed{s}_gated_null.json")
        fa = os.path.join(RA, f"mnist_mlp{tag}_seed{s}.json")
        if not (os.path.exists(fg) and os.path.exists(fa)):
            continue
        G = json.load(open(fg))["stages"]
        A = json.load(open(fa))["results"]
        for st, e in A.items():
            if st not in G or not isinstance(e, dict) or "layers" not in e:
                continue
            lays = e["layers"] if isinstance(e["layers"], list) else list(e["layers"].values())
            for lay in lays:
                L = str(lay.get("layer"))
                if L == "readout" or "components@1e-09" not in lay:
                    continue
                K = lay["components@1e-09"]["dim_K"]
                sel = G[st]["blocks"].get(L, {}).get("dim_gated_null")
                if K and sel:
                    pts["mlp"].append((sel, K))
for s in SEEDS:
    P = json.load(open(os.path.join(RB, f"mnist_cnn_seed{s}_patch_spaces.json")))["stages"]
    for st, e in P.items():
        if not isinstance(e, dict) or "blocks" not in e:
            continue
        for blk, key in (("conv5", "conv5"), ("readout", "readout")):
            b = e["blocks"].get(blk, {})
            sel, ak = b.get("dim"), (b.get("atomic_kernel") or {}).get("dim_K")
            if sel and ak:
                pts[key].append((sel, ak))
g = np.array([20, 3e5])
ax.plot(g, g, color=S.C["ink3"], lw=0.7, ls=":")
for key, col, mk, lab in (("mlp", S.C["blue"], "o", "MLP hidden layers"),
                          ("conv5", S.C["purple"], "s", "CNN conv5"), ("readout", S.C["aqua"], "s", "CNN readout")):
    if pts[key]:
        x, y = zip(*pts[key])
        ax.scatter(x, y, s=10, marker=mk, color=col, alpha=0.8, linewidths=0, zorder=3, label=lab)
    print(key, len(pts[key]), "points")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(20, 3e5); ax.set_ylim(20, 3e5)
ax.set_xlabel("selective space (dims)")
ax.set_ylabel("exact kernel (dims)")
ax.set_title("(a) ReLU: selectivity is the kernel", loc="left", fontsize=8)

# ---- (b) cross-neuron share of the kernel, all stages, grouped bars with seed whiskers
ax = axes[1]
STAGES4 = ["high_accuracy", "mse_interpolation", "epoch_250", "late_endpoint"]
SLAB4 = ["high\nacc.", "interp.", "ep.\n250", "ep.\n500"]
SERIES = [("", "2", "#6da7ec"), ("", "3", S.C["ink"]), ("_leaky", "3", S.C["blue"])]
Wb = 0.26
x = np.arange(len(STAGES4))
for j, (tag, L, col) in enumerate(SERIES):
    med, lo, hi = [], [], []
    for st in STAGES4:
        v = []
        for sd in SEEDS:
            f = os.path.join(RN, f"mnist_mlp{tag}_seed{sd}_s10000.json")
            if not os.path.exists(f):
                continue
            try:
                d = json.load(open(f))["results"][st]["layers"][L]["decomposition"]["1e-09"]
                v.append(d["cross_neuron"] / max(d["dim_K"], 1))
            except KeyError:
                pass
        med.append(np.median(v) if v else np.nan)
        lo.append(min(v) if v else np.nan); hi.append(max(v) if v else np.nan)
    med, lo, hi = map(np.array, (med, lo, hi))
    xo = x + (j - 1) * Wb
    ax.bar(xo, med, width=Wb * 0.9, color=col)
    ax.errorbar(xo, med, yerr=[med - lo, hi - med], fmt="none", ecolor=S.C["ink2"],
                elinewidth=0.7, capsize=1.3)
ax.set_xticks(x); ax.set_xticklabels(SLAB4, fontsize=6.2)
ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("cross-neuron share")
ax.set_title("(b) cross-neuron share", loc="left", fontsize=8)

# ---- (c) class structure across the stages: ARI of the diameter-controlled grouping at eps=1
ax = axes[2]
EPS0 = 1.0
ST3 = ["high_accuracy", "epoch_250", "late_endpoint"]
FAMC = {"": S.C["ink"], "_leaky": S.C["blue"], "_gelu": S.C["purple"]}
for tag in ("", "_leaky", "_gelu"):
    R = json.load(open(os.path.join(RC, f"mnist_mlp{tag}_seed0_dmap_eps.json")))["stages"]
    for L, ls in (("layer3", "-"), ("layer2", (0, (2, 1.4)))):
        ys = []
        for st in ST3:
            g = next(e for e in R[st][L]["eps_grid"] if abs(e["epsilon"] - EPS0) < 1e-9)
            ys.append(g["ARI_labels"])
        ax.plot(range(3), ys, color=FAMC[tag], ls=ls, lw=1.3, marker="o", ms=2.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["high\nacc.", "ep.\n250", "ep.\n500"], fontsize=6.2)
ax.set_xlim(-0.2, 2.2); ax.set_ylim(0, 1.06)
ax.set_ylabel("ARI vs labels")
ax.set_title("(c) class structure, all activations", loc="left", fontsize=8)

hs = [Line2D([], [], marker="o", ls="", ms=3, color=S.C["blue"]),
      Line2D([], [], marker="s", ls="", ms=3, color=S.C["purple"]),
      Line2D([], [], marker="s", ls="", ms=3, color=S.C["aqua"]),
      Rectangle((0, 0), 1, 1, facecolor="#6da7ec"), Rectangle((0, 0), 1, 1, facecolor=S.C["ink"]),
      Rectangle((0, 0), 1, 1, facecolor=S.C["blue"]),
      Line2D([], [], color=S.C["ink"], lw=1.3), Line2D([], [], color=S.C["blue"], lw=1.3),
      Line2D([], [], color=S.C["purple"], lw=1.3),
      Line2D([], [], color=S.C["ink2"], lw=1.3, ls="-"), Line2D([], [], color=S.C["ink2"], lw=1.3, ls=(0, (2, 1.4)))]
fig.legend(hs, ["MLP hidden layers", "CNN conv5", "CNN readout",
                "ReLU L2", "ReLU L3", "leaky L3",
                "ReLU", "leaky ReLU", "GELU", "L3", "L2"],
           loc="lower center", bbox_to_anchor=(0.5, -0.095), ncol=6, frameon=False,
           fontsize=5.4, columnspacing=0.7, handlelength=1.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"deficit_mechanisms.{ext}"))
plt.close(fig); print("wrote atomic/deficit_mechanisms")
