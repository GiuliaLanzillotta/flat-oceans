"""Full atomic-kernel scaling on the saved 10k-image factorizations only."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import style as S

TAGS = ['w32', 'w64', 'w128', 'w256', 'd2', 'd4', 'd5']
STAGES = ['high_accuracy', 'late_endpoint']


def collect():
    rows = []
    root = Path(S.HERE)
    for tag in TAGS:
        for seed in range(3):
            stem = 'mnist_mlp' + ('' if tag == 'w128' else '_' + tag)
            path = root / 'results/atomic' / f'{stem}_seed{seed}.json'
            results = json.loads(path.read_text())['results']
            for stage in STAGES:
                record = results.get(stage, {})
                if 'same_as_epoch' in record:
                    record = next(v for v in results.values()
                                  if v.get('epoch') == record['same_as_epoch'])
                for layer in record.get('layers', []):
                    ell = layer['layer']
                    if not isinstance(ell, int) or ell < 2:
                        continue
                    assert layer['rows'] == 100_000, (path, stage, ell)
                    p = layer['p']
                    dim = layer['components@1e-09']['dim_K']
                    assert dim == p - layer['rank_J']['1e-09']
                    gap = layer['gap_J']['1e-09']
                    rows.append(dict(tag=tag, seed=seed, stage=stage, layer=ell,
                                     epoch=record['epoch'], p=p, dimension=dim,
                                     percent=100*dim/p, gap=gap,
                                     sensitive=gap is not None and gap < 10,
                                     percent_at_1e12=100*(p-layer['rank_J']['1e-12'])/p,
                                     kind='full_atomic', source=str(path.relative_to(root))))
    return rows


def collect_selective512():
    rows = []
    root = Path(S.HERE)
    for seed in range(3):
        path = root/'results/atomic'/f'mnist_mlp_w512_seed{seed}_gated_null.json'
        data = json.loads(path.read_text())['stages']
        for stage in STAGES:
            record = data[stage]
            assert record['n_samples'] == 10_000
            for ell in [2, 3]:
                block = record['blocks'][str(ell)]
                dim = block['dim_gated_null']
                assert dim == sum(block['split'].values())
                rows.append(dict(tag='w512', seed=seed, stage=stage, layer=ell,
                                 epoch=record['epoch'], p=512*513, dimension=dim,
                                 percent=100*dim/(512*513), sensitive=False,
                                 kind='selective_lower_bound', source=str(path.relative_to(root))))
    return rows


def summary(rows, tag, stage, layer):
    cells = [r for r in rows if (r['tag'], r['stage'], r['layer']) == (tag, stage, layer)]
    if not cells:
        return np.nan, np.nan, False
    assert {r['seed'] for r in cells} == {0, 1, 2}
    values = [r['percent'] for r in cells]
    return np.mean(values), np.std(values, ddof=1), any(r['sensitive'] for r in cells)


def main():
    rows = collect()
    selective = collect_selective512()
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.90), sharey=True)
    fig.subplots_adjust(left=.13, right=.975, bottom=.36, top=.85, wspace=.24)
    panels = [(['w32','w64','w128','w256','w512'], ['32','64','128','256','512'], 'Depth 3', 'Width'),
              (['d2','w128','d4','d5'], ['2','3','4','5'], 'Width 128', 'Depth')]
    for ax, (tags, labels, title, xlabel) in zip(axes, panels):
        xs = np.arange(len(tags))
        for ell in range(2, 6):
            col = S.LAYER_COLS[ell-2]
            for j, stage in enumerate(STAGES):
                stats = [summary(rows, t, stage, ell) for t in tags]
                mean = np.array([s[0] for s in stats]); sd = np.array([s[1] for s in stats])
                if not np.isfinite(mean).any():
                    continue
                ax.fill_between(xs, mean-sd, mean+sd, color=col, alpha=.12, lw=0)
                ax.plot(xs, mean, color=col, marker='^o'[j], ms=3,
                        mfc='white' if j == 0 else col, mew=.7, lw=1,
                        ls='--' if j == 0 else '-')
                for k in np.flatnonzero(np.isfinite(mean)):
                    isolated = ((k == 0 or not np.isfinite(mean[k-1])) and
                                (k == len(xs)-1 or not np.isfinite(mean[k+1])))
                    if isolated:
                        ax.errorbar(xs[k], mean[k], yerr=sd[k], color=col, lw=.7, capsize=2)
        ax.set(xticks=xs, xticklabels=labels, xlabel=xlabel, ylim=(-3,105), yticks=[0,50,100],
               xlim=(-.2, len(tags)-.65))
        ax.set_title(title, fontsize=7, pad=3)
        ax.tick_params(labelsize=6, pad=1)
        ax.xaxis.label.set_size(7)
    for ell in [2, 3]:
        col = S.LAYER_COLS[ell-2]
        for j, stage in enumerate(STAGES):
            mean, sd, _ = summary(selective, 'w512', stage, ell)
            axes[0].errorbar(4 + (-.10 if ell == 2 else .10), mean, yerr=sd,
                             fmt='s', ms=3.3, mfc='white' if j == 0 else col,
                             mec=col, mew=.7, color=col, lw=.7, capsize=2)
    axes[0].set_ylabel('Dimension (%)', fontsize=6.5)
    fig.legend([Line2D([],[],color=S.LAYER_COLS[i],lw=1.3) for i in range(4)],
               ['L2','L3','L4','L5'],loc='lower center',ncol=4,fontsize=6.5,
               columnspacing=1,handlelength=1.2,bbox_to_anchor=(.5,.065))
    fig.legend([Line2D([],[],color=S.C['ink2'],ls='--',marker='^',mfc='white',ms=3,lw=1),
                Line2D([],[],color=S.C['ink2'],ls='-',marker='o',mfc=S.C['ink2'],ms=3,lw=1)],
               ['High accuracy','Epoch 500'],loc='lower center',ncol=2,fontsize=6.5,
               columnspacing=1.4,handlelength=2,bbox_to_anchor=(.5,-.015))
    target = Path(S.F)/'early_draft/main_compact'; target.mkdir(parents=True,exist_ok=True)
    S.save(fig, str(target/'atomic_scaling'))
    (target/'atomic_scaling_data.json').write_text(json.dumps(rows+selective,indent=2)+'\n')

    table = [r'\begin{table}[!htbp]\centering',
             r'\caption{Invariant-space dimensions as percentages of layer parameters on 10,000 analysis images: full atomic kernels, except for the selective lower bounds at width 512. Entries give means $\pm$ one sample standard deviation over three seeds at the relative singular-value cutoff $10^{-9}$. Daggers mark a full-kernel gap below ten at this cutoff in at least one seed. A dash means that the layer or milestone does not exist.}',
             r'\label{tab:atomic-scaling-full}', r'\small\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{llrrrr}\toprule',
             r'Architecture & Stage & L2 & L3 & L4 & L5\\\midrule']
    for tag in ['w32','w64','w128','w256','w512','d2','d4','d5']:
        arch = ('Width '+tag[1:]) if tag.startswith('w') else ('Depth '+tag[1:])
        if tag == 'w512': arch += ' (selective)'
        for j, stage in enumerate(STAGES):
            values=[]
            for ell in range(2,6):
                mean,sd,flag=summary(rows+selective,tag,stage,ell)
                values.append('---' if not np.isfinite(mean) else
                              f'${mean:.1f}\\pm{sd:.1f}'+(r'^{\dagger}' if flag else '')+'$')
            table.append((arch if j==0 else '')+' & '+('High accuracy' if j==0 else 'Epoch 500')+
                         ' & '+' & '.join(values)+r'\\')
    table += [r'\bottomrule\end{tabular}',r'\end{table}']
    Path(S.TABLES,'section_d_atomic_scaling.tex').write_text('\n'.join(table)+'\n')
    print('Saved',len(rows),'source-linked layer records; largest cutoff-tightening change (pp):',
          max(abs(r['percent']-r['percent_at_1e12']) for r in rows))


