"""Compact main-paper figures from saved results; no model fitting or measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.special import ndtr
from . import style as S

OUT = Path(S.F) / 'early_draft/main_compact'
RESULTS = Path(S.HERE) / 'results'

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    S.save(fig, str(OUT / name))

def activations():
    x = np.linspace(-4, 4, 500)
    panels = [('ReLU', np.maximum(x, 0), 0.), ('Leaky ReLU', np.maximum(x, .15*x), .15),
              ('Identity', x, 1.), ('GELU', x*ndtr(x), 0.)]
    fig, axes = plt.subplots(1, 4, figsize=(3.35, .9))
    fig.subplots_adjust(left=.015, right=.99, bottom=.03, top=.79, wspace=.19)
    for j, (ax, (name, y, left_slope)) in enumerate(zip(axes.flat, panels)):
        ax.axhline(0, color=S.C['grid'], lw=.6); ax.axvline(0, color=S.C['grid'], lw=.6)
        ax.plot(x, y, color=S.C['blue'], lw=1.6)
        # Affine tangents at the marked inputs, including exact GELU derivatives.
        for side, z0, slope, color, label in (
                (np.linspace(-4, 0, 250), -3., left_slope, S.C['orange'], r'$z_-$'),
                (np.linspace(0, 4, 250), 3., 1., S.C['purple'], r'$z_+$')):
            value = slope*z0
            if name == 'GELU':
                value = z0*ndtr(z0)
                slope = ndtr(z0) + z0*np.exp(-z0*z0/2)/np.sqrt(2*np.pi)
            ax.plot(side, value+slope*(side-z0), ls=(0, (3, 2)), color=color, lw=1., zorder=3)
            ax.plot(z0, 0, marker='o', ms=2.5, color=color, zorder=4)
            ax.annotate(label, (z0, 0), xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', color=color, fontsize=6)
        ax.set(xlim=(-4,4), ylim=(-1.3 if j!=2 else -4.2,4.2), xticks=[], yticks=[])
        ax.set_title(name, fontsize=6.5, pad=2)
        for spine in ax.spines.values(): spine.set_visible(False)
    save(fig, 'activation_tails')

def polynomial():
    groups = [[set(), {1}, {2}, {3}], [{1,2}, {1,3}, {2,3}, {1,2,3}]]
    titles = [r'Base and first order ($J_Xw$)', r'Higher orders ($P_X^{\rm frz}$)']
    fig, axes = plt.subplots(1,2,figsize=(3.35,1.18))
    fig.subplots_adjust(left=.005,right=.995,bottom=.015,top=.82,wspace=.08)
    for ax, terms, title in zip(axes,groups,titles):
        ax.set_title(title,fontsize=7,pad=5)
        for row, edits in enumerate(terms):
            y = 3-row
            label = r'$\Psi_{\emptyset}$' if not edits else '$\\Psi_{\\{'+','.join(map(str,sorted(edits)))+'\\}}$'
            ax.text(0,y+.31,label,fontsize=7,va='center')
            left=1.25
            for kind, layer, width in [('W',3,.83),('S',2,.56),('W',2,.83),('S',1,.56),('W',1,.83),('x',0,.45)]:
                edited=kind=='W' and layer in edits
                col=S.LAYER_COLS[layer-1] if edited else ('#e4e7ec' if kind=='W' else 'white')
                ax.add_patch(Rectangle((left,y),width,.62,facecolor=col,edgecolor=S.C['ink3'] if kind!='W' else 'none',lw=.5))
                if kind=='W':
                    txt=('$B$' if edited else '$V^0$') if layer==3 else (f'$A_{layer}$' if edited else f'$W_{layer}^0$')
                else: txt=rf'$S_{{{layer},i}}$' if kind=='S' else '$x_i$'
                ax.text(left+width/2,y+.31,txt,fontsize=6.8 if kind!='S' else 5.8,ha='center',va='center',color='white' if edited else S.C['ink'])
                left+=width+.08
        ax.set(xlim=(0,5.87),ylim=(-.08,3.75));ax.axis('off')
    save(fig,'frozen_terms')






