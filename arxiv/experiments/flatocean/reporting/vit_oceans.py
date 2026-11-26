"""Compact ViT appendix figure from the completed three-seed JSON records.

No checkpoint loading or new measurements. Corrected head_multilayer records
supply vector residuals; *_prebugfix records are deliberately excluded.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import style


def main():
    root = Path(style.HERE)
    source = root / 'results/runs/ftvit_resisc45'
    families = [json.loads((source / f'phase0f_s{s}_m256.json').read_text()) for s in range(3)]
    joint = [json.loads((source / f'head_multilayer_s{s}.json').read_text()) for s in range(3)]
    for record in families + joint:
        assert record['N'] == 10000 and record['M'] == 256
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45))
    fig.subplots_adjust(left=.075, right=.99, top=.89, bottom=.40, wspace=.75)
    for ax, title in zip(axes, ['(a) Atomic rays', '(b) Exact branches', '(c) Joint tail rays', '(d) Joint response']):
        ax.set_title(title, fontsize=8.5)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=7.5)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
    colors = style.C
    modes = [('tail', 'Tail', colors['blue'], 'o'),
             ('randsub', 'Random subspace', colors['orange'], 's'),
             ('generic', 'Full-block random', colors['ink3'], '^')]
    fam_names = list(families[0]['families'])
    labels = [r'$D_3(722)$', r'$D_3(567)$', r'$D_3(\tau)$',
              r'$D_0(\tau)$', r'$F_3(\tau)$', r'$F_0(\tau)$']
    for j, (mode, label, color, marker) in enumerate(modes):
        values = np.array([[d['families'][f][mode]['last_safe_tested_tol1e-3']['median']
                            for f in fam_names] for d in families])
        avg = values.mean(0)
        axes[0].errorbar(avg, np.arange(6)+(j-1)*.19,
                        xerr=np.maximum(0, [avg-values.min(0), values.max(0)-avg]),
                        fmt=marker, ms=3.7, capsize=2.5, color=color, label=label)
    axes[0].set_yticks(range(6), labels)
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    axes[0].set_xlim(.06, 5)
    axes[0].set_xticks([.1, 1, 3], ['0.1', '1', '3'])
    axes[0].set_xlabel('Last-safe radius')

    def curve(ax, radius, data, label, color, marker='o', ls='-'):
        values = np.asarray(data)
        avg = values.mean(0)
        ax.plot(radius, avg, label=label, color=color, marker=marker, ms=3.5, lw=1.5, ls=ls)
        ax.fill_between(radius, values.min(0), values.max(0), color=color, alpha=.16, linewidth=0)

    branches = [
        ('featurenull_mlp0+readout', 'Feature-null', colors['blue'], 'o'),
        ('mixed_featurenull_mlp0+readout_plus_mlp3LNshift', '+ LN shifts', colors['aqua'], 's'),
        ('CONTROL_generic_mlp0+mlp3+readout', 'Random control', colors['ink3'], '^')]
    for key, label, color, marker in branches:
        curve(axes[1], joint[0]['EXT_radii'],
              [d['exact_multilayer_branches'][key]['rel_logit_drift'] for d in joint], label, color, marker)
    axes[1].set_xlabel('Edit amplitude')
    axes[1].set_ylabel('Relative logit drift')
    axes[1].set_yticks([1e-15, 1e-10, 1e-5, 1])

    def median_curves(mode, field):
        return [np.median([v['drift'][field] for v in d['approx_joint'][mode]['per_direction']], axis=0)
                for d in joint]

    for mode, label, color, marker in modes:
        curve(axes[2], joint[0]['RGRID'], median_curves(mode, 'actual'), label, color, marker)
    components = [('actual', 'Actual', colors['blue'], 'o', '-'),
                  ('first_order', 'First order', colors['ink2'], 's', '--'),
                  ('transition', 'Transition', colors['orange'], '^', ':'),
                  ('joint_interaction', 'Mixed weights', colors['purple'], 'd', '-.')]
    for key, label, color, marker, ls in components:
        curve(axes[3], joint[0]['RGRID'], median_curves('tail', key), label, color, marker, ls)
    for ax in axes[1:]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([.01, 1, 100] if ax is axes[1] else [.01, 1, 30])
    for ax in axes[2:]:
        ax.set_xlabel('Joint radius')
        ax.axhline(1e-3, color=colors['ink3'], ls='--', lw=.65)
    axes[2].set_yticks([1e-6, 1e-4, 1e-2, 1])
    axes[3].set_yticks([1e-11, 1e-8, 1e-5, 1e-2])
    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.51), fontsize=7.4,
                  handlelength=1.7, borderaxespad=0, labelspacing=.25)
    dest = Path(style.F) / 'early_draft/section_d'
    dest.mkdir(parents=True, exist_ok=True)
    style.save(fig, 'early_draft/section_d/vit_oceans')


