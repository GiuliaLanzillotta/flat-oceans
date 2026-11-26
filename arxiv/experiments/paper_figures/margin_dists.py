"""Margin arms, under the hood (appendix).  (a) Distribution of the absolute preactivations |z| at epoch 500 for the
control and the two margin arms (all hidden layers and analysis samples pooled, per-arm density on log |z|): the
penalty empties the near-zero region without moving the bulk.  (b) Gates flipped by the time the ray first becomes
visible (true/F0 >= 1e-3), per arm and seed, median over the 128 branch probes: hundreds of flips at visibility in
every arm.  Reads the stored checkpoints (forward pass) and results/branches/*_rays.json.
Writes figures/radius/margin_dists.{png,pdf}."""
import os, sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S
from flatocean import runs as R
RB = os.path.join(S.HERE, "results", "branches"); OUT = os.path.join(S.F, "radius"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(S.HERE,"results","visualization","margin_dists_cache.npz")
ARMS = [("w64", "control", S.C["ink3"]), ("mgn_l0", r"margin $\lambda_0$", S.C["orange"]), ("mgn_l1", r"margin $\lambda_1$", S.C["ink"])]

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); Z = {k: z[k] for k in z.files}
else:
    Z = {}
    for tag, _, _ in ARMS:
        vals = []
        for sd in (0, 1, 2):
            run = R.load_run("mnist", "mlp", sd, tag=tag, epoch=500, device="cpu")
            X = run["data"]["analysis"][0]
            with torch.no_grad():
                _, Zs, _ = run["net"](X, return_states=True)
            vals.append(np.abs(torch.cat([q.flatten() for q in Zs]).numpy()))
        Z[tag] = np.concatenate(vals)
    np.savez(CACHE, **Z)

fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.55), gridspec_kw=dict(wspace=0.40, left=0.13, right=0.985, top=0.84, bottom=0.30))
ax = axs[0]
bins = np.logspace(-4, 2.2, 70)
for tag, lab, c in ARMS:
    h, e = np.histogram(Z[tag], bins=bins, density=True)
    ax.plot(np.sqrt(e[1:] * e[:-1]), h * np.sqrt(e[1:] * e[:-1]), color=c, lw=1.0, label=lab)
ax.set_xscale("log")
ax.set_xlabel("$|z|$ (all hidden gates)", fontsize=5.6, labelpad=1)
ax.set_ylabel("density per log $|z|$", fontsize=5.4)
ax.set_title("(a) preactivation margins", loc="left", fontsize=6.4, pad=3)
ax.tick_params(labelsize=5)
ax.legend(fontsize=4.4, handlelength=1.2, labelspacing=0.18, borderpad=0.15, loc="upper left", frameon=False)

ax = axs[1]
for k, (tag, lab, c) in enumerate(ARMS):
    pooled = []
    for sd in (0, 1, 2):
        f = os.path.join(RB, f"mnist_mlp_{tag}_seed{sd}_rays.json")
        d = json.load(open(f))["results"]["late_endpoint"]["sets"]["branch:bottom_up"]["absolute"]
        rs = sorted(d, key=float)
        flips_at_vis = []
        for probe in range(len(d[rs[0]])):
            for x in rs:
                q = d[x][probe]
                if q["true"] / max(q.get("base_output_norm_X", 1.0), 1e-12) >= 1e-3:
                    flips_at_vis.append(sum(q["flips"]))
                    break
        pooled.append(flips_at_vis)
        ax.scatter(k + (sd - 1) * 0.17, np.median(flips_at_vis), s=12, color=S.C["ink"], edgecolors="white",
                   linewidths=0.3, zorder=4, marker="D")
    allp = np.array([max(v, 0.5) for seed in pooled for v in seed])
    bp = ax.boxplot(allp, positions=[k], widths=0.62, showfliers=False, patch_artist=True, whis=(10, 90),
                    medianprops=dict(color=c, lw=1.2), boxprops=dict(facecolor=c, alpha=0.22, edgecolor=c, lw=0.8),
                    whiskerprops=dict(color=c, lw=0.8), capprops=dict(color=c, lw=0.8))
ax.set_yscale("log"); ax.set_ylim(0.4, 3e3)
ax.axhline(0.5, color=S.C["ink3"], lw=0.4, ls=":")
ax.text(0.03, 0.035, "floor $=0$ flips", transform=ax.transAxes, fontsize=4.2, color=S.C["ink3"])
ax.set_xticks(range(3)); ax.set_xticklabels([a[1] for a in ARMS], fontsize=4.8); ax.set_xlim(-0.6, 2.6)
ax.set_ylabel("gates flipped at visibility", fontsize=5.4)
ax.set_title("(b) flips at visibility", loc="left", fontsize=6.4, pad=3)
ax.tick_params(labelsize=5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"margin_dists.{ext}"))
print("wrote radius/margin_dists")
