"""Conceptual selective-unit and population-pathway panels (no measured data)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

from . import style as S


def main():
    blue, orange = S.C['blue'], S.C['orange']
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.28))
    fig.subplots_adjust(left=.015, right=.995, bottom=.025, top=.79, wspace=.20)
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
    a, b = axes
    a.set_title('(a) Selective units', fontsize=7.2, loc='left', pad=4)
    b.set_title('(b) Population pathways', fontsize=7.0, loc='left', pad=4)

    # Overlapping active input sets, each associated with a different ReLU unit.
    for x, col in [(.36, blue), (.64, orange)]:
        a.add_patch(Ellipse((x, .60), .55, .48, facecolor=col, alpha=.13,
                            edgecolor='none'))
        a.add_patch(Ellipse((x, .60), .55, .48, facecolor='none',
                            edgecolor=col, lw=.85))
    pts = [(.22,.60,blue),(.33,.72,blue),(.32,.48,blue),
           (.50,.66,S.C['purple']),(.50,.54,S.C['purple']),
           (.68,.71,orange),(.78,.59,orange),(.68,.48,orange)]
    for x, y, c in pts:
        a.plot(x,y,'o',ms=2.7,color=c,zorder=5)
    a.text(.5,.97,'Active input sets',ha='center',va='top',fontsize=6.6)
    for x, col, unit in [(.26, blue, r'ReLU $u_1$'), (.74, orange, r'ReLU $u_2$')]:
        a.add_patch(FancyArrowPatch((x,.36),(x,.22),arrowstyle='-|>',
                                    mutation_scale=7,color=col,lw=.9))
        a.add_patch(FancyBboxPatch((x-.18,.035),.36,.17,
                                  boxstyle='round,pad=.008',
                                  facecolor='white',edgecolor=col,lw=.8))
        a.text(x,.105,unit,ha='center',va='center',fontsize=7)

    # Both populations use every input coordinate in the same input layer.
    # Their routes separate only at the hidden units.
    xin,x1,x2,xout=.10,.40,.68,.94
    ys=[.75,.49,.23]
    for y in ys:
        for z in ys:
            b.plot([xin,x1],[y,z],color=S.C['grid'],lw=.55,zorder=0)
            b.plot([x1,x2],[y,z],color=S.C['grid'],lw=.55,zorder=0)
        b.plot([x2,xout],[y,.49],color=S.C['grid'],lw=.55,zorder=0)
    for y,col,name,label_y in [(.75,blue,'A',.87),(.23,orange,'B',.10)]:
        for input_y in ys:
            b.plot([xin,x1],[input_y,y],color=col,lw=.95,zorder=2)
        b.plot([x1,x2,xout],[y,y,.49],color=col,lw=1.15,zorder=2)
        b.text((x1+x2)/2,label_y,name,ha='center',va='center',
               fontsize=6.7,color=col)
    b.scatter([xin]*len(ys),ys,s=30,facecolors='white',
              edgecolors=S.C['ink'],linewidths=.8,zorder=4)
    for x in [x1,x2]:
        for y,col in zip(ys,[blue,S.C['ink3'],orange]):
            b.scatter([x],[y],s=30,facecolors='white',edgecolors=col,linewidths=.8,zorder=4)
    b.scatter([xout],[.49],s=34,facecolors='white',edgecolors=S.C['ink'],linewidths=.8,zorder=4)
    (Path(S.F)/'schematics').mkdir(parents=True,exist_ok=True)
    S.save(fig,'schematics/cooccurrence_mechanisms')


