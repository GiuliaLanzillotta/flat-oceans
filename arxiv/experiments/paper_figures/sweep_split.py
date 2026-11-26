"""Width and depth sweep: the selective-unit space of every hidden layer on the 60k training set, split into dead rows,
rarely active units (counting) and rank-deficient units, at high accuracy and epoch 500; the live input rank of every
layer.  Reads results/atomic/mnist_mlp_<tag>_seed<s>_gated_null_train.json.  Writes figures/paper_sweep_split.{png,pdf}.
Run from experiments/: python paper_figures/sweep_split.py"""
import glob, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean.reporting import style as S

RA = os.path.join(S.HERE, "results", "atomic")
WIDTH = {"w32": (32, 3), "w64": (64, 3), "w128": (128, 3), "w256": (256, 3), "w512": (512, 3), "d2": (128, 2), "d4": (128, 4), "d5": (128, 5)}
STAGES = [("high_accuracy", "0.1% train error"), ("late_endpoint", "epoch 500")]
COMP = [("dead_units", "dead rows", S.C["blue"]), ("rarely_active_units", "rarely active (counting)", S.C["orange"]),
        ("rank_deficient_on_large_active_set", "rank-deficient active set", S.C["aqua"])]

def load(tag):
    pat = f"mnist_mlp_seed*_gated_null_train.json" if tag == "w128" else f"mnist_mlp_{tag}_seed*_gated_null_train.json"
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RA, pat)))]

def per_layer(tag, stage):
    """(layers, comps[3][seeds][layers] fractions, rh[seeds][layers] live-rank fraction, dead unit fraction[seeds][layers])"""
    w, L = WIDTH[tag]; recs = [r["stages"][stage] for r in load(tag) if stage in r["stages"]]
    if not recs: return None
    layers = list(range(2, L + 1)); n_in = lambda l: (256 if l == 1 else w) + 1
    comps = np.zeros((3, len(recs), len(layers))); rh = np.zeros((len(recs), len(layers))); dead = np.zeros_like(rh)
    for si, st in enumerate(recs):
        for li, l in enumerate(layers):
            b = st["blocks"][str(l)]; p = n_in(l) * w
            for ci, (k, _, _) in enumerate(COMP): comps[ci, si, li] = b["split"][k] / p
            rh[si, li] = b["r_h"] / n_in(l); dead[si, li] = b["unit_counts"]["dead_units"] / w
    return layers, comps, rh, dead

def stacked(ax, tags, stage, xlabel):
    x0 = 0; ticks = []; labels = []; totals = []
    for tag in tags:
        out = per_layer(tag, stage)
        if out is None:
            ax.text(x0 + 0.5, 0.02, "stage not\nreached", ha="center", va="bottom", fontsize=5.5, color=S.C["ink2"]); x0 += 1.8; ticks.append(x0 - 1.3); labels.append(tag[1:]); continue
        layers, comps, rh, dead = out
        for li, l in enumerate(layers):
            x = x0 + li * 0.8; bottom = 0.0
            m = comps[:, :, li].mean(1)
            for ci, (_, _, col) in enumerate(COMP):
                ax.bar(x, m[ci], 0.7, bottom=bottom, color=col, linewidth=0); bottom += m[ci]
            tot = comps[:, :, li].sum(0)
            ax.errorbar(x, tot.mean(), yerr=[[tot.mean() - tot.min()], [tot.max() - tot.mean()]], color=S.C["ink"], lw=0.7, capsize=1.5)
            ax.text(x, -0.045, f"L{l}", ha="center", va="top", fontsize=6, color=S.C["ink2"])
        ticks.append(x0 + 0.8 * (len(layers) - 1) / 2); labels.append(tag[1:] if tag[0]=="w" else "depth "+tag[1:]); x0 += 0.8 * len(layers) + 0.7
    ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.tick_params(axis="x", pad=9, length=0)
    ax.set_ylim(0, 1.02); ax.set_xlabel(xlabel)

def ranks(ax, tags, xlabel):
    xs = np.arange(len(tags))
    for si_, (stage, sl) in enumerate(STAGES):
        for li in range(4):
            ys = []; lo = []; hi = []
            for tag in tags:
                out = per_layer(tag, stage)
                if out is None or li >= len(out[0]): ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                r = out[2][:, li]; ys.append(r.mean()); lo.append(r.min()); hi.append(r.max())
            ys, lo, hi = map(np.array, (ys, lo, hi))
            if np.all(np.isnan(ys)): continue
            ax.errorbar(xs + (si_ - 0.5) * 0.12, ys, yerr=[ys - lo, hi - ys], color=S.LAYER_COLS[li], marker="^o"[si_], ms=3.5, lw=0.9,
                        ls=":-"[si_], capsize=1.2, label=f"L{li + 2}, {sl}")
    ax.set_xticks(xs); ax.set_xticklabels([t[1:] if t[0]=="w" else "depth "+t[1:] for t in tags]); ax.set_ylim(0, 1.05); ax.set_xlabel(xlabel); ax.set_ylabel("$r_h/(n_{\\ell-1}+1)$")

fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), gridspec_kw=dict(width_ratios=[1.35, 1.35, 1.0], hspace=0.9, wspace=0.3, bottom=0.24, top=0.95, left=0.08, right=0.99))
W = ["w32", "w64", "w128", "w256", "w512"]; D = ["d2", "w128", "d4", "d5"]
for row, (tags, xl) in enumerate([(W, "width (depth 3)"), (D, "depth (width 128)")]):
    for col, (stage, sl) in enumerate(STAGES):
        ax = axes[row, col]; stacked(ax, tags, stage, xl); ax.set_title(f"selective fraction, {sl}")
        if col == 0: ax.set_ylabel("dim $G^{\\rm sel}_\\ell / p_\\ell$")
    ax = axes[row, 2]; ranks(ax, tags, xl); ax.set_title("live input rank")
for ax in axes[1, :]:
    ax.set_xticklabels(["2", "3", "4", "5"])
h1 = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in COMP]
fig.legend(h1, [n for _, n, _ in COMP], loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=1, title="components of the selective space")
hh, ll = axes[1, 2].get_legend_handles_labels()
fig.legend(hh, ll, loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=2, title="live input rank: layer, stage")
S.save(fig, "paper_sweep_split")
