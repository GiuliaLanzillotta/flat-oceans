"""Paper loss-geometry summaries from reviewed scalar records; no new experiments."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import style as S

ROOT=Path(S.HERE)
OUT=Path(S.F)/'early_draft/applications'

def main():
    source=ROOT/'results/hessian_gn/fixed_ocean_figure_table.json'
    d=json.loads(source.read_text());ts=np.array(d['protocol']['thresholds'])
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.65))
    fig.subplots_adjust(left=.10,right=.98,bottom=.26,top=.85,wspace=.30)
    stats={}
    for op,label,col in [('H','Hessian','blue'),('G','GGN','orange')]:
        cov=[];dims=[]
        for seed in ['0','1','2']:
            r=d['seeds'][seed];cells=r['sides'][op]['cells']
            keyed={float(k):v for k,v in cells.items()}
            cov.append([100*keyed[t]['coverage'] for t in ts])
            dims.append([100*keyed[t]['target_dim']/r['p'] for t in ts])
        stats[op]={}
        for ax,key,values in zip(axs,['coverage_percent','target_dimension_percent'],[cov,dims]):
            values=np.array(values);mu=values.mean(0);sd=values.std(0,ddof=1)
            stats[op][key]=dict(mean=mu.tolist(),sd=sd.tolist())
            ax.plot(ts,mu,color=S.C[col],marker='o',ms=3.8,lw=1.4)
            ax.fill_between(ts,np.maximum(0,mu-sd),np.minimum(100,mu+sd),color=S.C[col],alpha=.18,lw=0)
    baseline=100*np.array([d['seeds'][s]['k_over_p_baseline'] for s in ['0','1','2']])
    axs[0].axhline(baseline.mean(),color=S.C['ink2'],ls=':',lw=1)
    axs[0].axhspan(baseline.mean()-baseline.std(ddof=1),baseline.mean()+baseline.std(ddof=1),color=S.C['ink2'],alpha=.08,lw=0)
    for ax,title in zip(axs,['Ocean coverage','Target size']):
        ax.set(xscale='log',xlim=(5e-15,2e-4),ylim=(0,103),xticks=[1e-14,1e-9,1e-4],yticks=[0,25,50,75,100])
        ax.set_title(title,fontsize=10);ax.set_box_aspect(1);ax.minorticks_off()
        ax.set_xlabel(r'Threshold $\tau$');ax.grid(axis='y')
    axs[0].set_ylabel('Coverage (%)');axs[1].set_ylabel('Parameters (%)')
    handles=[Line2D([],[],color=S.C['blue'],marker='o',ms=4,label='Hessian'),
             Line2D([],[],color=S.C['orange'],marker='o',ms=4,label='GGN'),
             Line2D([],[],color=S.C['ink2'],ls=':',label='Random coverage')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.53,.01))
    OUT.mkdir(parents=True,exist_ok=True)
    S.save(fig,str(OUT/'fixed_ocean_coverage'))
    (OUT/'fixed_ocean_coverage.json').write_text(json.dumps(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),thresholds=ts.tolist(),stats=stats,ddof=1),indent=2)+'\n')

