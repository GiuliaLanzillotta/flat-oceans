"""P1 appendix figures (planning-agent request 2026-10-03): sample size, geometry, radius and
exposure, from the corrected records with the bracket/censoring treatment.

p1_radii: first gate-crossing radius and 1e-3 output-visibility radius against the training
sample size N, both evaluated ON THE FITTED SUBSET; per-seed trajectories, grid brackets
[last-below, first-above], censored cells drawn as up-arrows at the grid maximum.
p1_exposure: (a) median actual drift against the radius on the fitted subset against the
excluded rows (same directions); (b) excluded-row drift at matched radius 1e-3 against N;
(c) the counting-induced W2/W3 atomic kernels against N.

Reads results/analysis/p1_rays_summary.json, results/branches/mnist_mlp{_p1,}_sub*_seed*
(rays / ext_rays / exposure_rays / branch records).  Writes figures/p1/*.{png,pdf}."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S

RB = os.path.join(S.HERE, "results", "branches")
OUT = os.path.join(S.F, "p1"); os.makedirs(OUT, exist_ok=True)
SUM = json.load(open(os.path.join(S.HERE, "results", "analysis", "p1_rays_summary.json")))
NS = [1000, 2000, 5000]
SEEDS = [0, 1, 2]
CONDS = [("retrain", "retrained on the subset", S.C["blue"]),
         ("subsample", "full-data model, subset rows", S.C["orange"])]
GRID_MAX = 100.0
SET = "branch:bottom_up"


def cell(cond, N, s):
    return SUM[f"{cond}_N{N}_seed{s}"]


def bracket(e, key):
    b = e["fitted_sets"][SET][key]
    cens = b["first_above_median_is_censored"] or b["first_above_median"] is None
    hi = GRID_MAX if cens else b["first_above_median"]
    lo = b["lower_median"] if b["lower_median"] is not None else hi
    return lo, hi, cens


# ---------------------------------------------------------------- radii vs N
def name(cond, N, s):
    return f"mnist_mlp{'_p1' if cond == 'retrain' else ''}_sub{N}_seed{s}"

def med(e, setname, key):
    b = e["fitted_sets"][setname][key]
    v = b["first_above_median"]
    return GRID_MAX if (b["first_above_median_is_censored"] or v is None) else v

fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.95))
fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.37, wspace=0.42)
for ax, key, title in ((axes[0], "r_gate", "(a) first gate crossing (fitted set)"),
                       (axes[1], "r_vis_0.001", r"(b) visibility $10^{-3}$ (fitted set)")):
    for setname, ls in (("branch:bottom_up", "-"), ("generic", (0, (2, 1.5)))):
        for cond, col in (("retrain", S.C["blue"]), ("subsample", S.C["orange"])):
            V = np.array([[med(cell(cond, N, s), setname, key) for N in NS] for s in SEEDS])
            for s_ in range(3):
                ax.plot(NS, V[s_], color=col, ls=ls, lw=0.6, alpha=0.3)
            ax.plot(NS, np.median(V, 0), color=col, ls=ls, lw=1.5, marker="o", ms=2.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(NS); ax.set_xticklabels(["1000", "2000", "5000"]); ax.minorticks_off()
    ax.set_xlim(850, 5900); ax.set_ylim(8e-7, 2.5e2)
    ax.axhline(GRID_MAX, color=S.C["ink3"], lw=0.5, ls=":")
    ax.set_xlabel("constraint-set size $N$")
    ax.set_title(title, loc="left", fontsize=8)
axes[0].set_ylabel("radius (median)")

# (c) single-layer kernels at N=1000: first gate-crossing radius per block
ax = axes[2]
BLOCKS = ["atomic_W2", "atomic_W3", "atomic_V"]
for j, (cond, mk, dx) in enumerate((("retrain", "o", -0.13), ("subsample", "s", 0.13))):
    for s in SEEDS:
        p = os.path.join(RB, name(cond, 1000, s) + "_atomic_rays.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))["results"]["late_endpoint"]["sets"]
        for bi, blk in enumerate(BLOCKS):
            if blk not in d or "absolute" not in d[blk]:
                continue
            ab = d[blk]["absolute"]
            ks = sorted(ab, key=float)
            firsts = []
            for i in range(len(ab[ks[0]])):
                f = [float(k) for k in ks if sum(ab[k][i]["flips"]) > 0]
                firsts.append(f[0] if f else np.inf)
            m = np.median(firsts)
            x = bi + dx + (s - 1) * 0.045
            if np.isinf(m):
                ax.scatter([x], [GRID_MAX], marker="^", s=14, facecolor="none",
                           edgecolor=S.C["blue"] if cond == "retrain" else S.C["orange"], linewidths=0.8, zorder=3)
            else:
                ax.scatter([x], [m], marker=mk, s=12,
                           color=S.C["blue"] if cond == "retrain" else S.C["orange"],
                           zorder=3, linewidths=0)
ax.set_yscale("log"); ax.set_ylim(8e-7, 2.5e2)
ax.axhline(GRID_MAX, color=S.C["ink3"], lw=0.5, ls=":")
ax.set_xticks(range(3)); ax.set_xticklabels(["W2", "W3", "readout"], fontsize=6.6)
ax.set_xlim(-0.5, 2.5)
ax.set_title("(c) single-layer kernels, $N{=}1000$", loc="left", fontsize=8)
from matplotlib.lines import Line2D as _L
hs = [_L([], [], color=S.C["blue"], lw=1.5), _L([], [], color=S.C["orange"], lw=1.5),
      _L([], [], color=S.C["ink2"], lw=1.2, ls="-"), _L([], [], color=S.C["ink2"], lw=1.2, ls=(0, (2, 1.5))),
      _L([], [], marker="^", ls="", ms=4, markerfacecolor="none", markeredgecolor=S.C["orange"])]
fig.legend(hs, ["subset-trained", "full-data", "branch ray", "generic ray", "never crosses"],
           loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False,
           fontsize=5.8, columnspacing=0.8, handlelength=1.3)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"p1_radii_consistent.{ext}"))
plt.close(fig); print("wrote p1/p1_radii")

# ---------------------------------------------------------------- exposure
FLOOR = 1e-17

def med_curve(path, setname, key="true"):
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path))["results"]["late_endpoint"]["sets"]
    if setname not in d:
        return None, None
    ab = d[setname]["absolute"]
    ks = sorted(ab, key=float)
    med = [np.median([max(e[key], FLOOR) for e in ab[k]]) for k in ks]
    return np.array([float(k) for k in ks]), np.array(med)



fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.05))
fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.36, wspace=0.42)

ax = axes[0]
for cond, lab, col in CONDS:
    for kind, ls in (("fitted", "-"), ("excluded", (0, (2, 1.5)))):
        curves = []
        for s in SEEDS:
            base = os.path.join(RB, name(cond, 1000, s))
            if kind == "fitted":
                r1, m1 = med_curve(base + "_ext_rays.json", SET)
                r2, m2 = med_curve(base + "_rays.json", SET)
                if r1 is None or r2 is None:
                    continue
                r = np.concatenate([r1, r2[r2 > r1.max()]])
                m = np.concatenate([m1, m2[r2 > r1.max()]])
            else:
                r, m = med_curve(base + "_exposure_rays.json", SET)
                if r is None:
                    continue
            curves.append((r, m))
        rs = curves[0][0]
        ms = [np.interp(rs, rr, mm) for rr, mm in curves]
        ax.plot(rs, np.median(np.stack(ms), 0), color=col, ls=ls, lw=1.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-6, 1.0); ax.set_ylim(3e-18, 3e-1)
ax.set_xlabel("absolute radius $r$")
ax.set_ylabel(r"output change $\|\Delta F\|_X$")
ax.set_title("branch directions, $N=1000$", loc="left", fontsize=8)

ax = axes[1]
for cond, lab, col in CONDS:
    for s in SEEDS:
        y = [max(cell(cond, N, s)["exposure_sets"][SET]["median_true_drift_at_1e-3"], FLOOR)
             for N in NS]
        ax.plot(NS, y, color=col, lw=0.7, alpha=0.35)
    med = [np.median([cell(cond, N, s)["exposure_sets"][SET]["median_true_drift_at_1e-3"]
                      for s in SEEDS]) for N in NS]
    ax.plot(NS, med, color=col, lw=1.6, marker="o", ms=2.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(NS); ax.set_xticklabels(["1000", "2000", "5000"]); ax.minorticks_off()
ax.set_xlabel("constraint-set size $N$")
ax.set_title("excluded-row change at $r=10^{-3}$", loc="left", fontsize=8)

ax = axes[2]
REC = json.load(open(os.path.join(S.HERE, "results", "analysis", "p1_rank_reconciled.json")))
for cond, lab, col in CONDS:
    for blk, ls in (("layer2", "-"), ("layer3", (0, (2, 1.5)))):
        for s in SEEDS:
            dims = []
            for N in NS:
                c = REC[cond].get(f"N{N}_seed{s}")
                e = c["fit"][blk] if c else None
                dims.append(max(e["dim_K"], 1) if e and e.get("dim_K_status") == "exact" else np.nan)
            ax.plot(NS, dims, color=col, ls=ls, lw=0.8, alpha=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(NS); ax.set_xticklabels(["1000", "2000", "5000"]); ax.minorticks_off()
ax.set_ylim(80, 2.5e4)
ax.set_ylabel("dimensions")
ax.set_xlabel("constraint-set size $N$")
ax.set_title("exact atomic kernel of W2 / W3", loc="left", fontsize=8)
for ax, letter in zip(axes, "abc"):
    S.panel(ax, f"({letter})")
hs = [Line2D([], [], color=CONDS[0][2], lw=1.5), Line2D([], [], color=CONDS[1][2], lw=1.5)]
fig.legend(hs, ["retrained on the subset", "full-data model, subset rows"],
           loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=6.6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"p1_exposure_consistent.{ext}"))
plt.close(fig); print("wrote p1/p1_exposure")
