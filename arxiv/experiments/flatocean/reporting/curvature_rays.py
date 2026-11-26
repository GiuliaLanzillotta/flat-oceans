"""Matched cumulative Hessian/GGN rays from saved three-seed measurements."""
from pathlib import Path
import json,re,subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import style as S
from .appendix_layout import standard,records
ROOT=Path(S.HERE);SRC=ROOT.parent;DEST=Path(S.F)/'applications'
TAUS=[1e-14,1e-12,1e-10,1e-8,1e-6,1e-4]
def main():
 torch.set_num_threads(2);result={};fig,axes=plt.subplots(1,2,figsize=(4.1,2.8));colors=list(plt.get_cmap('plasma')(np.linspace(.06,.88,6)))+[S.C['ink'],S.C['ink3']]
 for ax,op,prefix in zip(axes,['H','G'],['hessian','ggn']):
  zs=[torch.load(ROOT/f'results/{prefix}_threshold_rays/production/seed{s}.pt',weights_only=False,mmap=True) for s in range(3)];r=np.asarray(zs[0]['radii']);names=[f'{op}_{t:g}' for t in TAUS]+['ocean','generic'];result[op]={}
  for name,col in zip(names,colors):
   assert all(np.array_equal(r,z['radii']) for z in zs)
   a=np.array([np.median(z['records'][name]['drift'].numpy(),axis=0) for z in zs]);mu=a.mean(0);sd=a.std(0,ddof=1);dims=[z['records'][name]['dimension'] for z in zs];generic=name=='generic'
   result[op][name]=dict(dimensions=dims,percent_parameters=float(100*np.mean(dims)/25418),seed_medians=a.tolist(),mean=mu.tolist(),sample_sd=sd.tolist())
   ax.plot(r,np.maximum(mu,1e-15),color=col,ls='--' if generic else '-',lw=1.5 if name=='ocean' else 1.,marker='o',ms=2,markevery=3)
   ax.fill_between(r,np.maximum(mu-sd,1e-15),np.maximum(mu+sd,1e-15),color=col,alpha=.10,lw=0)
  ax.set(xscale='log',yscale='log',xlim=(.001,100),ylim=(1e-15,1e4),xticks=[.001,.1,100],yticks=[1e-15,1e-9,1e-3,1e3],xlabel='Radius $r$');ax.set_title('Hessian' if op=='H' else 'GGN');ax.minorticks_off();ax.axhline(.001,ls=':',lw=.7,color=S.C['ink2'])
 axes[0].set_ylabel(r'Output drift $\|\Delta F\|_X$')
 labels=[]
 for t in TAUS:
  h=result['H'][f'H_{t:g}']['percent_parameters'];g=result['G'][f'G_{t:g}']['percent_parameters'];labels.append(r'$10^{'+str(int(np.log10(t)))+r'}$'+f' ({h:.1f}% / {g:.1f}%)')
 labels+=['Ocean (16.1%)','Generic (100%)']
 handles=[Line2D([],[],color=c,ls='--' if i==7 else '-',label=lab) for i,(c,lab) in enumerate(zip(colors,labels))]
 fig.legend(handles=handles,ncol=2);standard(fig,'curvature_cumulative_rays')
 for leg in list(fig.legends):leg.remove()
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,fontsize=6.4,handlelength=1.5,columnspacing=1.3,labelspacing=.3,title=r'$\tau$ (Hessian / GGN % of parameters)',title_fontsize=6.4)
 for ext in ['pdf','png']:fig.savefig(DEST/f'curvature_cumulative_rays.{ext}',bbox_inches='tight',pad_inches=.04,dpi=220)
 plt.close(fig)
 out=Path(S.F)/'applications';(out/'curvature_cumulative_summary.json').write_text(json.dumps(dict(radii=r.tolist(),parameter_dimension=25418,n_rays=256,n_seeds=3,statistic='median within seed, mean and sample SD across seeds',families=result,panel_layout=records['curvature_cumulative_rays']),indent=2))
 info=subprocess.check_output(['pdfinfo',str(DEST/'curvature_cumulative_rays.pdf')],text=True);w=float(re.search(r'Page size:\s*([\d.]+)',info)[1])*.9/72
 print('TeX width',f'{w:.3f}in')
 for op in ['H','G']:
  for name,q in result[op].items():
   safe=[float(r[np.maximum.accumulate(a)<=.001][-1]) for a in np.array(q['seed_medians'])];print(op,name,round(q['percent_parameters'],2),safe)
