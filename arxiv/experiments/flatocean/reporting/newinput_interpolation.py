"""Paper figure and seed table from the reviewed clean-to-noise records."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from . import style as S


def main():
    data = json.loads((Path(S.HERE) / 'results/newinput_interp/consolidated_interp.json').read_text())
    alphas = data['alphas']
    x = np.array(alphas, dtype=float)
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    for layer, label, color in zip(data['blocks'], ['L2', 'L3', 'Readout'],
                                    [S.C['blue'], S.C['orange'], S.C['purple']]):
        cells = [data['cells'][layer][a] for a in alphas]
        values = np.array([[c['per_seed'][f'seed{s}']['exposed_pct'] for s in range(3)] for c in cells])
        mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        np.testing.assert_allclose(mean, [c['exposed_pct_mean'] for c in cells], atol=1e-12, rtol=0)
        np.testing.assert_allclose(sd, [c['exposed_pct_sd'] for c in cells], atol=1e-12, rtol=0)
        mean = 100 - mean
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.15, linewidth=0)
        ax.plot(x, mean, 'o-', ms=2.8, color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(72, 100), xlabel=r'Noise fraction $\alpha$', ylabel='Space surviving (%)')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(range(75, 101, 5))
    ax.grid(axis='y')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(.56, 1.02))
    fig.subplots_adjust(left=.15, right=.98, bottom=.22, top=.86)
    (Path(S.F) / 'ablations').mkdir(parents=True, exist_ok=True)
    S.save(fig, 'ablations/newinput_interpolation')

    rows = []
    qualified = 0
    for a in alphas:
        entries = []
        for layer in data['blocks']:
            vals = []
            for seed in range(3):
                c = data['cells'][layer][a]['per_seed'][f'seed{seed}']
                qualified += not c['resolved']
                vals.append(f"{100-c['exposed_pct']:.2f}" + (r'^{\dagger}' if not c['resolved'] else ''))
            entries.append('$' + ' / '.join(vals) + '$')
        rows.append(a + ' & ' + ' & '.join(entries) + r'\\')
    assert qualified == 13
    text = r'''% Generated from consolidated_interp.json by newinput_interpolation.py.
\begin{table}[!htbp]\centering
\caption{Clean-to-noise interpolation: percentage of the original atomic invariant subspace that survives, for seeds 0/1/2. Daggers mark the 13 cutoff-sensitive L3 cells. The original kernel dimensions are 9,583/10,543/10,936 (L2), 14,696/15,331/15,005 (L3), and 780/880/760 (readout).}
\label{tab:newinput-interpolation}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrr}\toprule
$\alpha$ & L2 & L3 & Readout\\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}
\end{table}
'''
    Path(S.TABLES, 'section_d_newinput_interpolation_table.tex').write_text(text)


