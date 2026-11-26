"""Appendix figures/tables from saved measurements only; no network evaluations."""
from pathlib import Path
import json
import re
import numpy as np
import torch
from scipy.stats import beta,binom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import style as S
from . import radius_samples as RS
ROOT=Path(S.HERE);SRC=ROOT.parent;DEST=Path(S.F)/'appendix_revision'
STAGES=[('high_accuracy','High accuracy','orange'),('mse_interpolation','Interpolation','blue'),('late_endpoint','Epoch 500','purple')]
COL=[S.C['blue'],S.C['orange'],S.C['ink3']]
def read(p):return json.loads(p.read_text())
def layout(n,columns=3):
 cols=min(n,columns);rows=(n+cols-1)//cols
 fig,axes=plt.subplots(rows,cols,figsize=(2.05*cols,2*rows+.65),squeeze=False)
 fig.subplots_adjust(left=.55/(2.05*cols),right=1-.12/(2.05*cols),bottom=.60/(2*rows+.65),top=1-.35/(2*rows+.65),wspace=.65,hspace=.72)
 for ax in axes.flat:ax.set_box_aspect(1);ax.tick_params(labelsize=7);ax.grid(axis='y',alpha=.5)
 for ax in list(axes.flat)[n:]:ax.set_visible(False)
 return fig,list(axes.flat)[:n]
def save(fig,name):
 DEST.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','png']:fig.savefig(DEST/f'{name}.{ext}')
 plt.close(fig)
def vit_arrays(seed,family,mode,metric='abs_rms_X'):
 root=ROOT/'results/runs/ftvit_resisc45';e=read(root/f'common_family_s{seed}_radius_ext.json')
 a=torch.load(root/f'common_family_s{seed}_arrays.pt',weights_only=False,mmap=True,map_location='cpu');b=torch.load(root/f'common_family_s{seed}_radius_ext_arrays.pt',weights_only=False,mmap=True,map_location='cpu')
 r=np.array(e['radii_full_concatenated']);ci=a['curve_axis'].index('actual');mi=a['metric_axis'].index(metric);key=f'battery_{family}_{mode}_drift'
 y=np.concatenate([a[key][:,ci,mi,:].numpy(),b[key+'_ext'][:,ci,mi,:].numpy()],axis=1);assert y.shape==(256,len(r))
 return r,y

def family_figures():
 fams=[('mlp0_fn_exact','First MLP\nFeature-null'),('mlp3_dn_exact','Second MLP\nDownstream-null'),('head_fn_exact','Readout feature-null'),('mlp0_dn_approx','First MLP\nDownstream tail'),('mlp3_fn_approx','Second MLP\nFeature tail')]
 fig,axes=layout(6);summaries={}
 for ax,(family,title) in zip(axes,fams):
  for mode,col,ls,label in zip(['family','randsub','generic'],COL,['-','--',':'],['Constructed','Random subspace','Generic']):
   vals=[]
   for seed in range(3):
    d=read(ROOT/f'results/runs/ftvit_resisc45/common_family_s{seed}.json');b=d['batteries'].get(family,{})
    if not b.get('modes') or mode not in b['modes']:continue
    r,y=vit_arrays(seed,family,mode,'rel_rms_X');vals.append(np.median(y,axis=0))
   if not vals:continue
   vals=np.array(vals);mu=vals.mean(0);sd=vals.std(0,ddof=1) if len(vals)>1 else np.zeros(len(r))
   ax.plot(r,np.maximum(mu,1e-17),color=col,ls=ls,marker='o',ms=2.8,label=label)
   if len(vals)>1:ax.fill_between(r,np.maximum(mu-sd,1e-17),mu+sd,color=col,alpha=.13,lw=0)
   summaries[family+'/'+mode]=dict(r=r.tolist(),mean=mu.tolist(),sd=sd.tolist(),seeds=len(vals))
  ax.set(xscale='log',yscale='log',xlabel='Radius $r$',ylim=(1e-17,1e2));ax.set_title(title,fontsize=8.5);ax.set_xticks([.001,1,1000]);ax.set_yticks([1e-15,1e-9,1e-3]);ax.axhline(1e-3,color=S.C['ink2'],ls=':',lw=.8)
 axes[5].set_visible(False)
 for ax in [axes[0],axes[3]]:ax.set_ylabel('Relative output drift')
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,0));save(fig,'vit_common_rays')
 d=read(ROOT/'results/resnet/common_family_l42fc.json');a=torch.load(ROOT/'results/resnet/common_family_l42fc_arrays.pt',weights_only=False,mmap=True,map_location='cpu');r=np.array(d['RGRID']);mi=a['metric_axis'].index('actual_abs');fig,axes=layout(2);res={}
 for mode,col,ls,label in zip(['SEL','SEL_randsub','generic'],COL,['-','--',':'],['Selective','Random subspace','Generic']):
  y=a[f'layer4.2.conv2/{mode}_drift'][:,mi,:].numpy()/d['logits_rms_X'];mu=np.median(y,axis=0);frac=(np.maximum.accumulate(y,axis=1)>1e-3).mean(0)
  axes[0].plot(r,mu,color=col,ls=ls,marker='o',ms=3,label=label);axes[1].plot(r,frac,color=col,ls=ls,marker='o',ms=3);res[mode]=dict(r=r.tolist(),median=mu.tolist(),failure_fraction=frac.tolist())
  for j,rr in enumerate(r):assert int((y[:,j]>1e-3).sum())==d['layers']['layer4.2.conv2']['battery'][mode]['exceedance_count_actual_rel_gt_tol_among_M'][f'r{rr:g}']
 axes[0].set(yscale='log',ylabel='Relative output drift');axes[0].axhline(1e-3,color=S.C['ink2'],ls=':',lw=.8);axes[1].set(ylabel='Observed failure fraction',ylim=(-.02,1.02))
 for ax,title in zip(axes,['Selective-family rays','Grid failures']):ax.set(xscale='log',xlabel='Radius $r$',xticks=[.001,.1,10]);ax.set_title(title,fontsize=8.5)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,0));save(fig,'resnet_common_rays');(DEST/'family_curves.json').write_text(json.dumps(dict(vit=summaries,resnet=res),indent=2))

def certificate_figures():
 panels=[]
 for title,stem,inv in [('ReLU MLP','mnist_mlp_w64_seed{}_rays.json','branch:bottom_up'),('CNN','mnist_cnn_seed{}_patch_rays.json','constructed'),('Leaky-ReLU MLP','mnist_mlp_leaky_w64_seed{}_rays.json','branch:bottom_up'),('BatchNorm MLP','mnist_mlp_bn_w64_seed{}_rays.json','branch:bottom_up'),('SVHN baseline','svhn_mlp_w64_seed{}_rays.json','branch:bottom_up')]:
  panel=RS.saved_ray_panel(title,[(s,stem.format(s)) for s in range(3)],inv,STAGES)
  if panel['rows']:panels.append(panel)
 panel=RS.saved_ray_panel('SVHN selected',[(s,f'svhn_mlp_w64_v2_seed{s}_rays.json') for s in range(3)],'branch:bottom_up',[('best_val','Validation-selected','aqua')])
 if panel['rows']:panels.append(panel)
 rows=[]
 for seed in range(3):
  for stage,label,color in STAGES:
   d=torch.load(ROOT/f'results/gelu_extended_rays/production/seed{seed}_{stage}.pt',weights_only=False,map_location='cpu')
   r=np.asarray(d['radii']);assert d['n']==256 and r[-1]==1000
   for family in ['approx_0.001','generic_L3']:
    q=d['records'][family];assert q['dimension']>0
    y=q['drift'].numpy();assert y.shape==(256,len(r)) and np.isfinite(y).all()
    rows.append(dict(seed=seed,stage=stage,color=S.C[color],radii=r,y=np.maximum.accumulate(y,axis=1),is_generic=family=='generic_L3'))
 panels.append(dict(title='GELU MLP: L3 tail',rows=rows))
 for fam,title in [('mlp0_fn_exact','ViT first MLP'),('mlp3_dn_exact','ViT second MLP'),('head_fn_exact','ViT readout')]:
  rows=[]
  for seed in range(3):
   for mode in ['family','generic']:
    r,y=vit_arrays(seed,fam,mode);rows.append(dict(seed=seed,stage='fine_tuned',color=S.C['aqua'],radii=r,y=np.maximum.accumulate(y,axis=1),is_generic=mode=='generic'))
  panels.append(dict(title=title,rows=rows))
 p=RS.resnet_selective_panel();p['title']='ResNet conv2'
 for row in p['rows']:row['seed']=0
 panels.append(p);K=sum(len(row['radii']) for p in panels for row in p['rows']);eta=.05/K;fig,axes=layout(len(panels),columns=4);allrows=[]
 for ax,p in zip(axes,panels):
  groups={}
  for row in p['rows']:
   y=row['y'];n=len(y);k=(y>1e-3).sum(0);u=np.array([beta.ppf(1-eta,int(x)+1,n-int(x)) if x<n else 1. for x in k]);assert np.all(u+1e-14>=k/n)
   for x,b in zip(k,u):
    if x<n:assert np.isclose(binom.cdf(int(x),n,b),eta,rtol=1e-8,atol=1e-12)
   groups.setdefault((row['stage'],row['is_generic']),[]).append((row,u));allrows.append(dict(model=p['title'],seed=row['seed'],stage=row['stage'],generic=row['is_generic'],radii=row['radii'].tolist(),failures=k.tolist(),n=n,upper=u.tolist()))
  drawn=[]
  for (stage,generic),group in groups.items():
   row=group[0][0];r=row['radii'];values=np.array([u for _,u in group]);mu=values.mean(0);sd=values.std(0,ddof=1) if len(group)>1 else np.zeros(len(r));col=S.C['ink3'] if generic else row['color']
   if generic and any(np.array_equal(mu,a) and np.array_equal(sd,b) for a,b in drawn):continue
   if generic:drawn.append((mu,sd))
   ax.plot(r,mu,color=col,ls='--' if generic else '-',marker='s' if generic else 'o',mfc='white' if generic else col,ms=3)
   if len(group)>1:ax.fill_between(r,np.maximum(0,mu-sd),np.minimum(1,mu+sd),color=col,alpha=.14,lw=0)
  ax.set(xscale='log',ylim=(-.03,1.03),xlabel='Radius $r$',xticks=[.001,1,1000],yticks=[0,.1,.5,1]);ax.set_title(p['title'],fontsize=8.5);ax.axhline(.1,color=S.C['ink2'],ls=':',lw=.8);ax.minorticks_off()
 for i in range(0,len(axes),4):axes[i].set_ylabel('Failure probability')
 handles=[Line2D([],[],color=S.C[c],marker='o',label=lab) for _,lab,c in STAGES+[('fine_tuned','Fine-tuned / selected','aqua')]]+[Line2D([],[],color=S.C['ink3'],marker='s',mfc='white',ls='--',label='Generic')]
 fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=7,bbox_to_anchor=(.5,0))
 from .appendix_layout import standard
 standard(fig,'radius_certificates')
 save(fig,'radius_certificates');(DEST/'radius_certificates.json').write_text(json.dumps(dict(K=K,alpha=.05,epsilon=.001,rows=allrows),indent=2))

def evaluation():
 fig,axes=layout(1);ax=axes[0]
 for block,label,col in zip(['1','2','3','readout'],['L1','L2','L3','Readout'],S.LAYER_COLS):
  points={}
  for seed in range(3):
   p=ROOT/f'results/atomic/mnist_mlp_seed{seed}_saturation.json'
   if not p.exists():continue
   for n,ls in read(p)['sizes'].items():
    for b in ls:
     if str(b['layer'])==block and 'components@1e-09' in b:points.setdefault(int(n),[]).append(b['components@1e-09']['dim_K'])
  ns=sorted(points);mean=np.array([np.mean(points[n]) for n in ns]);sd=np.array([np.std(points[n],ddof=1) if len(points[n])>1 else 0 for n in ns]);x=np.array(ns)/1000;ax.plot(x,mean,color=col,marker='o',ms=3,label=label);ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=col,alpha=.15,lw=0)
 ax.set(xlabel='Images (thousands)',ylabel='Atomic dimension',xticks=[2,10,20]);ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True);fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=7,bbox_to_anchor=(.5,0));save(fig,'evaluation_dimensions')
