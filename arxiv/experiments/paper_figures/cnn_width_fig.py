"""CNN width ablation (P2 memos 2026-10-02): the selective-patch ocean across conv widths
0.5x / 1x / 2x of the MNIST CNN, with the width-matched native-color SVHN CNN as the data
contrast.  Two figures.

width_scaling: (a) selective ocean against parameter count (log-log, slope-1 guide),
(b) ocean share of the parameters, (c) test error -- accuracy saturates while the ocean grows.
width_channels: conv5 per-channel anatomy, freed dimensions against how often the channel
fires; the ocean concentrates in dead and rarely-active channels.

Reads results/branches/*_patch_spaces.json and results/tier1_models/*/seed_*/history.json.
The two width-2x cells without a stored record (seeds 1, 2) are the matrix-free selective-sum
dimensions from the 2026-10-02 memo, drawn hollow.
Writes figures/cnn/width_scaling.{png,pdf} and width_channels.{png,pdf}."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S

RB = os.path.join(S.HERE, "results", "branches"); RT = os.path.join(S.HERE, "results", "tier1_models")
OUT = os.path.join(S.F, "cnn"); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]

NETS = [("mnist_cnn_w0p5", "mnist_cnn_w0p5", 0.5, S.C["blue"]),
        ("mnist_cnn", "mnist_cnn", 1.0, S.C["blue"]),
        ("mnist_cnn_w2", "mnist_cnn_w2", 2.0, S.C["blue"]),
        ("svhn_cnn_svhn_cnn", "svhn_cnn_svhn_cnn", 2.0, S.C["orange"])]
MATRIX_FREE = {("mnist_cnn_w2", 1): 64416, ("mnist_cnn_w2", 2): 45004}  # memo 2026-10-02, no record file


def patch(tag, seed):
    p = os.path.join(RB, f"{tag}_seed{seed}_patch_spaces.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["stages"]["late_endpoint"]


def final_metrics(model, seed):
    h = json.load(open(os.path.join(RT, model, f"seed_{seed}", "history.json")))
    e = h["epochs"][-1]
    return h["config"]["n_params"], e["test_acc"]


data = {}
for tag, model, w, col in NETS:
    rows = []
    for s in SEEDS:
        n_par, acc = final_metrics(model, s)
        st = patch(tag, s)
        if st is not None:
            dim = st["selective_sum_dim"]; mf = False
        else:
            dim = MATRIX_FREE[(tag, s)]; mf = True
        rows.append(dict(seed=s, n_params=n_par, acc=acc, dim=dim, mf=mf))
    data[tag] = rows
    print(tag, [(r["dim"], round(r["acc"], 4), r["mf"]) for r in rows])

# ---------------------------------------------------------------- scaling
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.0))
fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.34, wspace=0.42)

ax = axes[0]
for tag, model, w, col in NETS:
    for r in data[tag]:
        ax.scatter(r["n_params"], r["dim"], s=16, facecolor="none" if r["mf"] else col,
                   edgecolor=col, linewidth=0.9, zorder=3)
mn = [np.mean([r["dim"] for r in data[t]]) for t in ("mnist_cnn_w0p5", "mnist_cnn", "mnist_cnn_w2")]
px = [data[t][0]["n_params"] for t in ("mnist_cnn_w0p5", "mnist_cnn", "mnist_cnn_w2")]
ax.plot(px, mn, color=S.C["blue"], lw=1.2)
g = np.array([1.1e4, 2.2e5])
ax.plot(g, mn[0] * (g / px[0]), color=S.C["ink3"], lw=0.7, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("parameters"); ax.set_ylabel("selective ocean (dims)")
ax.set_title("ocean against network size", loc="left", fontsize=8)

ax = axes[1]
for tag, model, w, col in NETS:
    for r in data[tag]:
        ax.scatter(w, r["dim"] / r["n_params"], s=16, facecolor="none" if r["mf"] else col,
                   edgecolor=col, linewidth=0.9, zorder=3)
fr = [np.mean([r["dim"] / r["n_params"] for r in data[t]]) for t in ("mnist_cnn_w0p5", "mnist_cnn", "mnist_cnn_w2")]
ax.plot([0.5, 1.0, 2.0], fr, color=S.C["blue"], lw=1.2)
ax.set_xscale("log")
ax.set_xticks([0.5, 1.0, 2.0]); ax.set_xticklabels(["0.5x", "1x", "2x"])
ax.set_xlim(0.37, 2.7); ax.set_ylim(0, 0.45)
ax.minorticks_off()
ax.set_xlabel("conv width multiplier"); ax.set_ylabel("ocean / parameters")
ax.set_title("ocean share of the block budget", loc="left", fontsize=8)

ax = axes[2]
for tag, model, w, col in NETS:
    for r in data[tag]:
        ax.scatter(w, 1 - r["acc"], s=16, color=col, zorder=3)
er = [np.mean([1 - r["acc"] for r in data[t]]) for t in ("mnist_cnn_w0p5", "mnist_cnn", "mnist_cnn_w2")]
ax.plot([0.5, 1.0, 2.0], er, color=S.C["blue"], lw=1.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.5, 1.0, 2.0]); ax.set_xticklabels(["0.5x", "1x", "2x"])
ax.set_xlim(0.37, 2.7); ax.set_ylim(3e-3, 2e-1)
ax.minorticks_off()
ax.set_xlabel("conv width multiplier"); ax.set_ylabel("test error")
ax.set_title("accuracy saturates", loc="left", fontsize=8)

for ax, letter in zip(axes, "abc"):
    S.panel(ax, f"({letter})")
hs = [Line2D([], [], marker="o", ls="", color=S.C["blue"], ms=4),
      Line2D([], [], marker="o", ls="", color=S.C["orange"], ms=4),
      Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=S.C["blue"], ms=4),
      Line2D([], [], color=S.C["ink3"], lw=0.7, ls=":")]
fig.legend(hs, ["MNIST CNN", "SVHN CNN (width-matched)", "matrix-free cell", "ocean $\\propto$ parameters"],
           loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=6.6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"width_scaling.{ext}"))
plt.close(fig); print("wrote cnn/width_scaling")

# ---------------------------------------------------------------- channel anatomy
PANELS = [("mnist_cnn_w0p5", "MNIST 0.5x (32 ch.)"), ("mnist_cnn", "MNIST 1x (64 ch.)"),
          ("mnist_cnn_w2", "MNIST 2x (128 ch.)"), ("svhn_cnn_svhn_cnn", "SVHN (128 ch.)")]
FULL = {"mnist_cnn_w0p5": 9 * 16 + 1, "mnist_cnn": 9 * 32 + 1,
        "mnist_cnn_w2": 9 * 64 + 1, "svhn_cnn_svhn_cnn": 9 * 64 + 1}
NCOL = {"mnist_cnn_w0p5": S.C["aqua"], "mnist_cnn": "#6da7ec", "mnist_cnn_w2": S.C["blue"],
        "svhn_cnn_svhn_cnn": S.C["orange"]}
NLAB = {"mnist_cnn_w0p5": "MNIST 0.5x", "mnist_cnn": "MNIST 1x", "mnist_cnn_w2": "MNIST 2x",
        "svhn_cnn_svhn_cnn": "SVHN"}
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.15))
fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.34, wspace=0.34)

# (a) each conv5 channel: how often it fires vs how much of its incoming space it frees
ax = axes[0]
for tag, _ in PANELS:
    st = patch(tag, 0)
    chans = st["blocks"]["conv5"]["channels"]
    act = np.clip(np.array([c["active_positions"] for c in chans], float), 0.5, None)
    share = np.array([c["free"] for c in chans], float) / FULL[tag]
    ax.scatter(act, share, s=8, color=NCOL[tag], alpha=0.8, linewidths=0, zorder=3,
               label=NLAB[tag])
    print(tag, "conv5 dims", st["blocks"]["conv5"]["dim"], "min act", int(act.min()))
ax.set_xscale("log"); ax.set_xlim(0.7, 1e5); ax.set_ylim(-0.04, 1.09)
ax.axhline(1.0, color=S.C["ink3"], lw=0.6, ls=":")
ax.set_xlabel("activations of the channel on $X$ (sample, position)")
ax.set_ylabel("freed share of the\nchannel's incoming space")
ax.set_title("each point is one conv5 channel (seed 0)", loc="left", fontsize=8)

# (b) channels sorted least-active first: cumulative share of the conv5 ocean they carry
ax = axes[1]
for tag, _ in PANELS:
    st = patch(tag, 0)
    tot = st["blocks"]["conv5"]["dim"]
    if tot == 0:
        continue
    chans = sorted(st["blocks"]["conv5"]["channels"], key=lambda c: c["active_positions"])
    frac = np.arange(1, len(chans) + 1) / len(chans)
    cum = np.cumsum([c["free"] for c in chans]) / tot
    ax.plot(frac, cum, color=NCOL[tag], lw=1.5, label=NLAB[tag])
ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
ax.set_xlabel("channels included, least-active first (fraction)")
ax.set_ylabel("share of the conv5\nocean carried")
ax.set_title("the rarely-active channels carry it", loc="left", fontsize=8)
ax.legend(fontsize=6.4, loc="lower right", handlelength=1.2, labelspacing=0.3)
S.panel(axes[0], "(a)"); S.panel(axes[1], "(b)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False,
           fontsize=6.6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"width_channels.{ext}"))
plt.close(fig); print("wrote cnn/width_channels")
