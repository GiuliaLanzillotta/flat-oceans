"""Is the epsilon-ocean unit-level or cross-unit?  ratio = sum of per-unit ocean counts / layer ocean count, at
eps=1e-3, against training stage, for ReLU / leaky / GELU (L3).  ratio->1: the ocean is the union of single-unit
oceans (unit-level, like ReLU selective units); ratio->0: it exists only through cancellation across units (cross-unit,
like leaky readout-null).  Second panel: GELU tau-active unit populations at epoch 500.
Writes figures/xray/gelu/soft_selectivity.{png,pdf}."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S
NW = os.path.join(S.HERE, "results", "neuronwise"); OUT = os.path.join(S.F, "gelu"); os.makedirs(OUT, exist_ok=True)
STAGES = [("high_accuracy", "99.9%\nacc."), ("mse_interpolation", "MSE\ninterp."), ("epoch_250", "ep. 250"), ("late_endpoint", "ep. 500")]
ACTS = [("", "ReLU", S.C["blue"]), ("leaky", "leaky", S.C["orange"]), ("gelu", "GELU", S.C["aqua"])]

fig, axs = plt.subplots(1, 4, figsize=(6.3, 1.8), gridspec_kw=dict(wspace=0.30, left=0.08, right=0.99, bottom=0.30, top=0.85, width_ratios=[1, 1, 1, 1.35]))
EPSL = ["1e-06", "0.0001", "0.001", "0.01"]
ERAMP = ["#c9dcf5", "#6da7ec", "#2a78d6", "#0d2c58"]
for ax, (tag, lab, col) in zip(axs[:3], ACTS):
    d = json.load(open(os.path.join(NW, f"mnist_mlp{('_'+tag) if tag else ''}_seed0_soft_selectivity.json")))["stages"]
    for e, ec in zip(EPSL, ERAMP):
        xs, ys = [], []
        for i, (st, _) in enumerate(STAGES):
            if st not in d or "3" not in d[st]["layers"]:
                continue
            v = d[st]["layers"]["3"]["sum_over_layer"].get(e)
            if v is not None:
                xs.append(i); ys.append(v)
        ax.plot(xs, ys, color=ec, lw=1.2, marker="o", ms=2.2,
                label=(r"$\epsilon=10^{%d}$" % round(np.log10(float(e)))) if tag == "" else None)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(["99.9%", "MSE", "250", "500"], fontsize=5.4)
    ax.set_xlim(-0.25, 3.25); ax.set_ylim(-0.05, 1.08)
    ax.axhline(1, color=S.C["ink3"], lw=0.5, ls=":")
    ax.tick_params(labelsize=5.6)
    ax.set_title(("(a) " if tag == "" else "") + lab, loc="left", fontsize=7.5)
    if tag != "":
        ax.set_yticklabels([])
axs[0].set_ylabel("unit-level share of the\napprox. ocean (L3)", fontsize=6.4)
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=5.8,
           handlelength=1.3, columnspacing=1.0, frameon=False)

ax = axs[3]
d = json.load(open(os.path.join(NW, "mnist_mlp_gelu_seed0_soft_selectivity.json")))["stages"]["late_endpoint"]["layers"]
taus = ["1e-09", "1e-06", "0.001", "0.01"]
RAMP = ["#c9dcf5", "#6da7ec", "#2a78d6", "#0d2c58"]
for l, ls in [("2", "-"), ("3", (0, (2, 1.4)))]:
    for t, col in zip(taus, RAMP):
        act = np.sort(np.array(d[l]["tau_activity"][t]["active_samples"], float))
        ax.plot(np.arange(1, len(act) + 1), np.clip(act, 0.6, None), color=col, ls=ls, lw=1.1,
                label=(r"$\tau=10^{%d}$" % round(np.log10(float(t)))) if l == "2" else None)
ax.set_yscale("log"); ax.set_ylim(0.5, 1.5e4)
ax.set_xlim(1, 64)
ax.set_xlabel("units, least $\\tau$-active first", fontsize=6.4, labelpad=1)
ax.set_ylabel("$\\tau$-active samples", fontsize=6.4)
ax.set_title("(b) GELU, epoch 500", loc="left", fontsize=7.5)
ax.tick_params(labelsize=5.6)
ax.legend(loc="upper left", fontsize=5.2, handlelength=1.2, labelspacing=0.25, frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"soft_selectivity.{ext}"))
print("wrote xray/gelu/soft_selectivity")
