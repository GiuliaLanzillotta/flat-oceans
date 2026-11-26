"""Ray previews: mean and sample SD of within-seed medians (three seeds)."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE))
from flatocean.reporting import style as S
DEST=Path(S.F)/'ray_extensions';DEST.mkdir(parents=True,exist_ok=True)

def aggregate(zs,names,p):
 result={}
 for name in names:
  qs=[z['records'][name] for z in zs];dims=[q['dimension'] for q in qs]
  if any(d==0 for d in dims):continue
  # NumPy's even-n median averages the two central order statistics.
  a=np.array([np.median(q['drift'].numpy(),axis=0) for q in qs]);assert np.isfinite(a).all()
  for row,rec in zip(a,qs):
   ordered=np.sort(rec['drift'].numpy(),axis=0);n=len(ordered)
   np.testing.assert_array_equal(row,(ordered[n//2-1]+ordered[n//2])/2)
  mu=a.mean(0);sd=a.std(0,ddof=1)
  result[name]={'dimensions':dims,'percent_parameters':(100*np.array(dims)/p).tolist(),'mean_percent_parameters':float(100*np.mean(dims)/p),'seed_median_drift':a.tolist(),'mean_seed_median_drift':mu.tolist(),'sd_seed_median_drift':sd.tolist(),'mean_minus_sd':(mu-sd).tolist(),'mean_plus_sd':(mu+sd).tolist()}

 return result

def axes(fig,ax):
 ax.set_box_aspect(1);ax.set_xscale('log');ax.set_yscale('log');ax.minorticks_off();ax.tick_params(labelsize=8);ax.set_xlabel('Radius $r$',fontsize=9);ax.set_ylabel(r'Output drift $\|\Delta F\|_X$',fontsize=9);ax.axhline(.001,color=S.C['ink2'],ls=':',lw=.7)

def gelu():
 stages=['high_accuracy','mse_interpolation','late_endpoint'];titles=['High accuracy','Interpolation','Epoch 500'];names=['approx_0.001','generic_L3'];summ={}
 from matplotlib.lines import Line2D
 fig,ax=plt.subplots(figsize=(3.5,3.7));fig.subplots_adjust(left=.19,right=.98,bottom=.29,top=.85);axes(fig,ax)
 allvals=[]
 for stage,title,col in zip(stages,titles,[S.C['orange'],S.C['blue'],S.C['purple']]):
  zs=[torch.load(HERE/f'results/gelu_extended_rays/production/seed{s}_{stage}.pt',weights_only=False) for s in range(3)];p=zs[0]['parameter_dimension'];r=np.asarray(zs[0]['radii']);q=aggregate(zs,names,p)
  summ[stage]={'epochs':[z['epoch'] for z in zs],'radii':r.tolist(),'parameter_dimension':p,'layer_dimension':16512,'families':q}
  for name in names:
   mu=np.array(q[name]['mean_seed_median_drift']);sd=np.array(q[name]['sd_seed_median_drift']);generic=name=='generic_L3';allvals.append(mu+sd)
   color=S.C['ink3'] if generic else col
   ax.fill_between(r,np.maximum(mu-sd,1e-15),np.maximum(mu+sd,1e-15),color=color,alpha=.10,lw=0)
   ax.plot(r,np.maximum(mu,1e-15),color=color,ls='--' if generic else '-',marker='s' if generic else 'o',mfc='white' if generic else color,ms=3,lw=1.4)
 ax.set_xlim(.001,1000);ax.set_xticks([.001,.1,10,1000]);ax.set_ylim(1e-8,10**np.ceil(np.log10(np.max(allvals))));ax.set_yticks([1e-8,1e-5,1e-2,10,1000]);ax.set_title('GELU MLP · L3',fontsize=10.5,pad=7)
 stagehandles=[Line2D([],[],color=S.C[c],marker='o',lw=0,label=t) for t,c in zip(titles,['orange','blue','purple'])]
 fig.legend(handles=stagehandles,loc='upper center',bbox_to_anchor=(.57,.99),ncol=3,fontsize=7,handlelength=.6,columnspacing=.8,handletextpad=.3)
 familyhandles=[Line2D([],[],color=S.C['ink'],label='Approximate kernel'),Line2D([],[],color=S.C['ink3'],ls='--',marker='s',mfc='white',label='Generic L3')]
 fig.legend(handles=familyhandles,loc='lower center',bbox_to_anchor=(.5,.07),ncol=2,fontsize=7.5,handlelength=1.5,columnspacing=1)
 # Match the physical data-panel size used throughout the appendix.
 from flatocean.reporting.appendix_layout import standard
 standard(fig,'gelu_rays_standard')
 # A compact two-column key fits the single-panel figure.
 for leg in list(fig.legends):leg.remove()
 fig.legend(handles=stagehandles+familyhandles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=2,fontsize=6.4,handlelength=1.3,columnspacing=.8)
 for ext in ['png','pdf']:fig.savefig(DEST/f'gelu_rays_standard.{ext}',bbox_inches='tight',pad_inches=.04)
 plt.close(fig);(DEST/'gelu_summary.json').write_text(json.dumps({'statistic':'median within seed, mean and sample SD over three seeds','spectral_rate_cutoff':.001,'stages':'ReLU-matched checkpoints','data':summ},indent=2))

if __name__ == '__main__':
 gelu()
