"""Section-C coverage atlas from saved records only; no model loading or experiments."""
from pathlib import Path
from collections import defaultdict
import json
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator, ScalarFormatter, SymmetricalLogLocator
from . import style as S

ROOT=Path(S.HERE); PAPER=Path(S.TABLES); RESULTS=ROOT/'results'
OUT=Path(S.F)/'section_c'
LABELS={
'mnist_mlp':'MNIST ReLU, width 128', 'mnist_mlp_w32':'MNIST ReLU, width 32',
'mnist_mlp_w64':'MNIST ReLU, width 64','mnist_mlp_w256':'MNIST ReLU, width 256','mnist_mlp_w512':'MNIST ReLU, width 512',
'mnist_mlp_d2':'MNIST ReLU, depth 2','mnist_mlp_d4':'MNIST ReLU, depth 4','mnist_mlp_d5':'MNIST ReLU, depth 5',
'mnist_mlp_leaky':'MNIST leaky ReLU, width 128','mnist_mlp_leaky_w64':'MNIST leaky ReLU, width 64',
'mnist_mlp_gelu':'MNIST GELU, width 128','mnist_mlp_identity':'MNIST identity, width 128',
'mnist_mlp_bn_w64':'MNIST BatchNorm, width 64','mnist_mlp_raw':'MNIST raw pixels, width 64',
'svhn_mlp_w64':'SVHN MLP, baseline','svhn_mlp_w64_v2':'SVHN MLP, validation selected',
'mnist_cnn':'MNIST CNN, base','mnist_cnn_w0p5':'MNIST CNN, half width','mnist_cnn_w2':'MNIST CNN, double width','svhn_cnn_svhn_cnn':'SVHN CNN',
'mnist_mlp_mgn_l0':r'Margin penalty $0.01$','mnist_mlp_mgn_l1':r'Margin penalty $0.1$',
'mnist_mlp_gain_d0':r'Downstream decay $10^{-4}$','mnist_mlp_gain_d1':r'Downstream decay $10^{-3}$',
}
for n in [1000,2000,5000]:
 LABELS[f'mnist_mlp_p1_sub{n}']=f'Retrained on {n:,} images'
 LABELS[f'mnist_mlp_sub{n}']=f'Full-data model, {n:,} constraints'
STAGES=['initialization','early_fitting','high_accuracy','mse_interpolation','epoch_50','epoch_100','epoch_150','epoch_200','epoch_250','late_endpoint','best_val']
SHORT={'initialization':'Init','early_fitting':'Early','high_accuracy':'HA','mse_interpolation':'Int','late_endpoint':'500','best_val':'Val'}
SLONG={'initialization':'Initialization','early_fitting':'Early fitting','high_accuracy':'High accuracy','mse_interpolation':'Interpolation','late_endpoint':'Epoch 500','best_val':'Validation selected'}
for k in STAGES:
 if k.startswith('epoch_'):SHORT[k]=k[6:];SLONG[k]='Epoch '+k[6:]
SOURCES=set(); EXCLUSIONS=[]

def read(p):
 SOURCES.add(str(p.relative_to(ROOT)));return json.loads(p.read_text())
def stage_order(st):return STAGES.index(st) if st in STAGES else 99
def layer_order(b):return (99 if b in ['V','readout'] else int(re.sub('[^0-9]','',str(b)) or 98))
def layer_name(b):return 'Readout' if b in ['V','readout'] else ('Conv '+str(b)[4:] if str(b).startswith('conv') else 'L'+re.sub('[^0-9]','',str(b)))
def layer_key(b):return 'readout' if b in ['V','readout'] else str(b).removeprefix('W')
def parse(f,suffix=''):
 m=re.fullmatch(r'(.+)_seed(\d+)'+suffix+r'\.json',f.name)
 return (m[1],int(m[2])) if m and m[1] in LABELS else None

def save(fig,name):
 fig.canvas.draw()
 for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
 plt.close(fig)

def texfig(name,caption,label=None,height='.84'):
 return '\n'.join([r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth,height='+height+r'\textheight,keepaspectratio]{early_draft/section_c/'+name+'.pdf}',r'\caption{'+caption+'}'+(r'\label{'+label+'}' if label else ''),r'\end{figure}'])+'\n'

def write_csv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/f'{name}.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def resolved(records):
 for st,v in records.items():
  if 'same_as_epoch' in v:
   v=next((x for x in records.values() if x.get('epoch')==v['same_as_epoch']),v)
  yield st,v

def collect():
 atomic={}; selective=[];branches=[];neuron=[]
 for f in sorted((RESULTS/'atomic').glob('*.json')):
  tag=parse(f)
  if not tag:continue
  model,seed=tag;d=read(f)
  for st,v in resolved(d.get('results',{})):
   for b in v.get('layers',[]):
    c=b.get('components@1e-09');blk=layer_key(b['layer'])
    if not c:continue
    if min(c.get(k,0) for k in ['feature_null','downstream_null','deficit'])<0:
     EXCLUSIONS.append(dict(source=str(f),stage=st,block=blk,reason='inconsistent numerical decomposition'));continue
    assert sum(c[k] for k in ['feature_null','downstream_null','deficit'])==c['dim_K']
    atomic[model,seed,st,blk]=dict(model=model,seed=seed,stage=st,epoch=v.get('epoch'),block=blk,p=b['p'],kernel=c['dim_K'],feature=c['feature_null'],downstream=c['downstream_null'],deficit=c['deficit'],unsplit=0,approx=b.get('k_rel',{}).get('0.001'),rank6=b.get('rank_J',{}).get('1e-06'),rank12=b.get('rank_J',{}).get('1e-12'),gap=b.get('gap_J',{}).get('1e-09'),source=str(f.relative_to(ROOT)))
 for f in sorted((RESULTS/'branches').glob('*.json')):
  tag=parse(f) or parse(f,'_branch')
  if not tag:continue
  model,seed=tag;d=read(f)
  for st,v in resolved(d.get('results',{})):
   if 'blocks' not in v:continue
   for name,b in v['blocks'].items():
    if b.get('kernel_source')=='excluded_frozen':continue
    k=b.get('dim_atomic_kernel',b.get('dim'));blk=layer_key(name)
    key=model,seed,st,blk
    if k is not None and key not in atomic:
     atomic[key]=dict(model=model,seed=seed,stage=st,epoch=v.get('epoch'),block=blk,p=b['p'],kernel=k,feature=0,downstream=0,deficit=0,unsplit=k,approx=None,gap=b.get('gap'),source=str(f.relative_to(ROOT)))
   for order,c in v.get('constructions',{}).items():
    if not c.get('certified_branch',c.get('validation',{}).get('passed',False)):continue
    if c.get('validation',{}).get('gram_fallback_used',False):continue
    dims=c.get('dims',{});total=c.get('total',sum(dims.values()))
    if not isinstance(total,(int,float)):continue
    branches.append(dict(model=model,seed=seed,stage=st,epoch=v.get('epoch'),order=order,dimension=total,atomic_sum=v.get('sum_atomic'),dims=dims,source=str(f.relative_to(ROOT))))
 for f in sorted((RESULTS/'atomic').glob('*gated_null*.json')):
  tag=parse(f,'_gated_null') or parse(f,'_gated_null_train')
  if not tag:continue
  model,seed=tag
  for st,v in read(f).get('stages',{}).items():
   for blk,b in v.get('blocks',{}).items():
    if 'dim_gated_null' not in b:continue
    n=v['n_samples'];dim=b['dim_gated_null'];sp=b.get('split_mechanism',{})
    # Incoming block sizes are available from the authenticated checkpoint configuration.
    cfg=v.get('identity',{}).get('config',{})
    width=(32 if '_w32' in model else 64 if '_w64' in model else 256 if '_w256' in model else 512 if '_w512' in model else 128)
    param=width*(257 if str(blk)=='1' else width+1)
    selective.append(dict(model=model,seed=seed,stage=st,epoch=v.get('epoch'),block=blk,samples=n,p=param,dimension=dim,dead=sp.get('dead_rows'),sample=sp.get('counting_beyond_dead'),span=sp.get('beyond_counting'),source=str(f.relative_to(ROOT))))
 for f in sorted((RESULTS/'branches').glob('*patch_spaces.json')):
  tag=parse(f,'_patch_spaces')
  if not tag:continue
  model,seed=tag
  for st,v in read(f).get('stages',{}).items():
   for blk,b in v.get('blocks',{}).items():
    if 'dim' not in b:continue
    selective.append(dict(model=model,seed=seed,stage=st,epoch=v.get('epoch'),block=blk,samples=v.get('identity',{}).get('analysis_size',10000),p=b['p'],dimension=b['dim'],dead=None,sample=None,span=None,source=str(f.relative_to(ROOT))))
   val=v.get('selective_sum_validation',{})
   if val.get('passed'):
    branches.append(dict(model=model,seed=seed,stage=st,epoch=v.get('epoch'),order='selective_sum',dimension=v['selective_sum_dim'],atomic_sum=v.get('atomic_sum_dim'),dims={k:b['dim'] for k,b in v['blocks'].items()},source=str(f.relative_to(ROOT))))
 for f in sorted((RESULTS/'neuronwise').glob('*_s10000.json')):
  tag=parse(f,'_s10000')
  if not tag:continue
  model,seed=tag
  for st,v in read(f).get('results',{}).items():
   for blk,b in v.get('layers',{}).items():
    c=b.get('decomposition',{}).get('1e-09',{})
    if not c or not c.get('consistent',True):continue
    neuron.append(dict(model=model,seed=seed,stage=st,block=blk,kernel=c['dim_K'],unit=c['neuronwise'],cross=c['cross_neuron'],source=str(f.relative_to(ROOT))))
 # Read saved CNN spectra without loading their large singular-vector arrays.
 for seed in range(3):
  model='mnist_cnn'
  epochs={r['epoch']:r['stage'] for r in atomic.values() if r['model']==model and r['seed']==seed}
  epochs.update({0:'initialization',1:'early_fitting'})
  for f in sorted((RESULTS/'persistence'/f'{model}_seed{seed}').glob('epoch*.npz')):
   SOURCES.add(str(f.relative_to(ROOT)))
   with np.load(f) as z:
    epoch=int(z['epoch']);blk=str(z['block']);param=int(z['p']);sv=z['singular_values'];rank=int(np.count_nonzero(sv>1e-9*sv[0]));approx=param-int(np.count_nonzero(sv>1e-3*sv[0]))
   st=epochs.get(epoch,'epoch_'+str(epoch));key=model,seed,st,blk
   if key not in atomic:
    atomic[key]=dict(model=model,seed=seed,stage=st,epoch=epoch,block=blk,p=param,kernel=param-rank,feature=0,downstream=0,deficit=0,unsplit=param-rank,source=str(f.relative_to(ROOT)))
   assert atomic[key]['kernel']==param-rank,(key,atomic[key]['kernel'],param-rank)
   atomic[key]['approx']=approx
 # A construction recorded by both the branch and patch-space analyses is one seed.
 unique={}
 for r in branches:
  key=tuple(r[k] for k in ['model','seed','stage','order'])
  if key in unique:assert unique[key]['dimension']==r['dimension'],key
  else:unique[key]=r
 branches=list(unique.values())
 f=RESULTS/'gain_maskedbound/mnist_mlp_gain_d1_seed1_branchrepair_aligned_1e-10.json'
 d=read(f);c=d['orders']['top_down']
 if c['certifies_at_fixed_1e-9_fresh'] and not c['gram_fallback_used']:
  branches.append(dict(model='mnist_mlp_gain_d1',seed=1,stage='late_endpoint',epoch=500,order='top_down_tight',dimension=c['retained_total'],atomic_sum=d['atomic_sum'],dims=c['retained_dims'],source=str(f.relative_to(ROOT))))
 return list(atomic.values()),selective,branches,neuron

# Grids are drawn near their printed width: six-point ticks remain six-point ticks.
PLOT_LABELS={m: label.replace('MNIST ', '').replace(', width ', ', w').replace(', depth ', ', d').replace('MLP, validation selected','MLP, validation').replace('Full-data model, ', 'Full-data, ').replace(' constraints',' constraints') for m,label in LABELS.items()}
for n in [1000,2000,5000]:
 PLOT_LABELS[f'mnist_mlp_p1_sub{n}']=f'Train on {n:,}'
 PLOT_LABELS[f'mnist_mlp_sub{n}']=f'Full-data, n={n:,}'

BLOCK_COLS=S.LAYER_COLS+[S.C['ink2']]
BLOCK_KEYS=['1','2','3','4','5','readout']

def block_colour(b):
 return BLOCK_COLS[5 if b=='readout' else (layer_order(b)-1)%5]

def dense_grid(count,columns=3,row_height=1.35,legend_height=.42):
 rows=int(np.ceil(count/columns));height=rows*row_height+legend_height+.18
 fig,axes=plt.subplots(rows,columns,figsize=(6.9,height),squeeze=False)
 fig.subplots_adjust(left=.055,right=.995,bottom=.27/height,top=1-(legend_height+.2)/height,hspace=.67,wspace=.28)
 for ax in axes.flat:
  ax.tick_params(labelsize=6.2,pad=1.5)
  ax.title.set_fontsize(7)
 for ax in list(axes.flat)[count:]:ax.set_visible(False)
 return fig,axes

def shared_legend(fig,handles,labels,columns=3):
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,1),ncol=columns,fontsize=6.8,columnspacing=1.25,handlelength=1.6,handletextpad=.45,labelspacing=.35)

def block_legend(fig,blocks):
 blocks=sorted({'readout' if b=='readout' else str(layer_order(b)) for b in blocks},key=layer_order)
 shared_legend(fig,[Line2D([],[],color=block_colour(b)) for b in blocks],['Readout' if b=='readout' else 'L'+str(layer_order(b)) for b in blocks],columns=6)

def stage_ticks(ax,stages,counts=None):
 labels=[{'initialization':'I','early_fitting':'E'}.get(st,SHORT.get(st,st)) for st in stages]
 if counts is not None:labels=[s+'\n'+str(n) for s,n in zip(labels,counts)]
 ax.set_xticks(range(len(stages)));ax.set_xticklabels(labels,fontsize=5.7 if len(stages)>8 else 6)
 if len(stages)>8:ax.tick_params(axis='x',labelrotation=55)

def finish_fragment(filename,figures):
 PAPER.mkdir(parents=True,exist_ok=True)
 (PAPER/filename).write_text(''.join(figures)+r'\FloatBarrier'+'\n')

def atomic_figures(rows):
 tex=[];data_tex=[]
 model_groups=[['mnist_mlp','mnist_mlp_w64','mnist_mlp_w32','mnist_mlp_w256'],
 ['mnist_mlp_d2','mnist_mlp_d4','mnist_mlp_d5'],
 ['mnist_mlp_leaky','mnist_mlp_leaky_w64','mnist_mlp_gelu','mnist_mlp_identity'],
 ['mnist_mlp_raw','mnist_mlp_bn_w64','mnist_cnn','svhn_mlp_w64','svhn_mlp_w64_v2'],
 [],[f'mnist_mlp_{prefix}sub{n}' for n in [1000,2000,5000] for prefix in ['p1_','']]]
 palette=[S.C['blue'],S.C['aqua'],S.C['orange'],S.C['purple']]
 for gi,models in enumerate(model_groups):
  panels=[]
  for m in models:
   model_rows=[r for r in rows if r['model']==m]
   if not model_rows:continue
   if len({r['stage'] for r in model_rows})==1:panels.append((m,None))
   else:panels.extend((m,b) for b in sorted({r['block'] for r in model_rows},key=layer_order))
  if not panels:continue
  fig,axes=dense_grid(len(panels),4,row_height=1.4,legend_height=.45)
  fig.supylabel('Dimension / layer parameters',fontsize=7,x=.001)
  for ax,(model,blk) in zip(axes.flat,panels):
   rr=[r for r in rows if r['model']==model and (blk is None or r['block']==blk)]
   if blk is None:
    sts=sorted({r['block'] for r in rr},key=layer_order);groups=[[r for r in rr if r['block']==b] for b in sts];subtitle=SLONG[rr[0]['stage']]
   else:
    sts=sorted({r['stage'] for r in rr},key=stage_order);groups=[[r for r in rr if r['stage']==st] for st in sts];subtitle=layer_name(blk)
   x=np.arange(len(sts));base=np.zeros(len(x))
   for k,col in zip(['feature','downstream','deficit','unsplit'],palette):
    y=np.array([np.mean([r[k]/r['p'] for r in g]) for g in groups]);ax.bar(x,y,bottom=base,color=col,width=.78,linewidth=0,hatch='///' if k=='unsplit' else None);base+=y
   ax.bar(x,1-base,bottom=base,color=S.C['grid'],width=.78,linewidth=0)
   low=np.array([min(r['kernel']/r['p'] for r in g) for g in groups]);hi=np.array([max(r['kernel']/r['p'] for r in g) for g in groups]);ax.errorbar(x,base,yerr=[np.maximum(base-low,0),np.maximum(hi-base,0)],fmt='.',color=S.C['ink'],lw=.65,ms=1.5,capsize=1)
   ax.set(ylim=(0,1.07),yticks=[0,.5,1])
   if blk is None:
    ax.set_xticks(x);ax.set_xticklabels([layer_name(b).replace('Layer ','L').replace('Conv ','C').replace('Readout','Out')+'\n'+str(len(g)) for b,g in zip(sts,groups)],fontsize=6)
   else:stage_ticks(ax,sts,[len(g) for g in groups])
   ax.set_title(PLOT_LABELS[model]+'\n'+subtitle,fontsize=7,loc='left',pad=3);ax.grid(axis='y',alpha=.3)
  shared_legend(fig,[Patch(color=c) for c in palette[:3]]+[Patch(facecolor=palette[3],hatch='///'),Patch(color=S.C['grid'])],['Feature-null','Downstream-null','Co-occurrence deficit','Kernel; split unavailable','Non-kernel'],columns=3)
  name=f'compact_atomic_{gi:02d}';save(fig,name)
  caption=r'Atomic decomposition across the indicated models and blocks. Models use MNIST unless labelled SVHN; w and d denote width and hidden depth. Panels show a block across stages, or all editable blocks at the stated endpoint. Heights are fractions of block parameters; bars average seeds and whiskers span their kernel fractions. Numbers below tick labels are seed counts. I, E, HA, Int and Val denote initialization, early fitting, high accuracy, interpolation and validation selection; numbers denote fixed epochs. Stages are grouped by milestone and then fixed epoch. For reduced-data models, $n$ is the retraining or analysis-set size as titled. Hatched kernels have no measured component split. Counts use the $10^{-9}$ rank rule; GELU late-stage counts and blocks without a spectral gap remain threshold-dependent.'
  (data_tex if gi==5 else tex).append(texfig(name,caption,'fig:c-atomic-'+str(gi)))
 finish_fragment('section_c_atomic_figures.tex',tex)
 finish_fragment('section_d_data_atomic.tex',data_tex)
 # Every spectral panel, with one consistent colour per block and one shared key.
 models=[m for m in LABELS if len({r['stage'] for r in rows if r['model']==m and r.get('approx') is not None})>1]
 fig,axes=dense_grid(len(models),3,row_height=1.13,legend_height=.32)
 fig.supylabel('Dimension / layer parameters',fontsize=7,x=.001)
 for i,(ax,m) in enumerate(zip(axes.flat,models)):
  rr=[r for r in rows if r['model']==m and r.get('approx') is not None];sts=sorted({r['stage'] for r in rr},key=stage_order)
  for b in sorted({r['block'] for r in rr},key=layer_order):
   for key,ls in [('kernel',':'),('approx','-')]:
    vals=[[r[key]/r['p'] for r in rr if r['stage']==st and r['block']==b] for st in sts];mean=[np.mean(v) if v else np.nan for v in vals];lo=[min(v) if v else np.nan for v in vals];hi=[max(v) if v else np.nan for v in vals]
    col=block_colour(b);ax.plot(range(len(sts)),mean,ls=ls,color=col,marker='o',ms=1.5,lw=.85);ax.fill_between(range(len(sts)),lo,hi,color=col,alpha=.08)
  ax.set_title(PLOT_LABELS[m],fontsize=7,loc='left',pad=3);ax.set(ylim=(-.03,1.03),yticks=[0,.5,1]);stage_ticks(ax,sts)
 block_legend(fig,[r['block'] for r in rows if r.get('approx') is not None])
 name='compact_tails';save(fig,name)
 finish_fragment('section_c_tail_figures.tex',[texfig(name,r'Approximate atomic dimensions for models measured at multiple training stages, with all measured blocks shown. MNIST is used unless labelled SVHN; block numbers refer to hidden layers or convolutional blocks. Solid curves count singular values at most $10^{-3}\sigma_1$ and dotted curves use $10^{-9}\sigma_1$, each divided by block parameter count. Lines are seed means and bands are seed ranges. Missing measurements remain gaps. Stage positions are categorical, with abbreviations as in the atomic panels.')])

def selective_figures(rows,atomic,neuron):
 tex=[];cases=[(m,n) for m in LABELS for n in sorted({r['samples'] for r in rows if r['model']==m})]
 cases=[(m,n) for m,n in cases if len({r['stage'] for r in rows if r['model']==m and r['samples']==n})>1]
 fig,axes=dense_grid(len(cases),4,row_height=1.3,legend_height=.32)
 fig.supylabel('Selective dimension / layer parameters',fontsize=7,x=.001)
 for ax,(m,n) in zip(axes.flat,cases):
  rr=[r for r in rows if r['model']==m and r['samples']==n];sts=sorted({r['stage'] for r in rr},key=stage_order)
  for b in sorted({r['block'] for r in rr},key=layer_order):
   gs=[[r['dimension']/r['p'] for r in rr if r['stage']==st and r['block']==b] for st in sts];mean=[np.mean(g) if g else np.nan for g in gs];col=block_colour(b)
   ax.plot(range(len(sts)),mean,color=col,marker='o',ms=1.5,lw=.9);ax.fill_between(range(len(sts)),[min(g) if g else np.nan for g in gs],[max(g) if g else np.nan for g in gs],color=col,alpha=.1)
  ax.set_title(PLOT_LABELS[m]+f'\n{n//1000}k analysis images',fontsize=7,loc='left',pad=3);ax.set(ylim=(-.03,1.03),yticks=[0,.5,1]);stage_ticks(ax,sts)
 block_legend(fig,[r['block'] for r in rows]);name='compact_selective';save(fig,name)
 tex.append(texfig(name,r'Selective dimension as a fraction of block parameters, across all tested stages and analysis-set sizes. MNIST is used unless labelled SVHN. Colours identify hidden layers or convolutional blocks; lines are seed means and bands span seeds. MLP spaces constrain each unit on its active samples; CNN spaces constrain each channel on its active input patches. A zero identifies an empty selective construction; the full atomic kernel can still contain other directions. Stage abbreviations follow the atomic panels.'))
 models=[m for m in LABELS if any(r['model']==m for r in neuron)]
 fig,axes=dense_grid(len(models),6,row_height=1.2,legend_height=.42)
 fig.subplots_adjust(wspace=.65)
 fig.supylabel('Kernel dimensions',fontsize=7,x=.001)
 for ax,m in zip(axes.flat,models):
  rr=[r for r in neuron if r['model']==m];sts=sorted({r['stage'] for r in rr},key=stage_order)
  for b,ls in [('2','-'),('3','--')]:
   for key,col in [('unit',S.C['blue']),('cross',S.C['orange'])]:
    gs=[[r[key] for r in rr if r['stage']==st and r['block']==b] for st in sts]
    mean=[np.mean(g) if g else np.nan for g in gs]
    ax.plot(range(len(sts)),mean,color=col,ls=ls,marker='o',ms=2,lw=1)
    ax.fill_between(range(len(sts)),[min(g) if g else np.nan for g in gs],[max(g) if g else np.nan for g in gs],color=col,alpha=.07)
  ax.set_title(PLOT_LABELS[m].replace(', ','\n'),loc='left',fontsize=6,pad=3);stage_ticks(ax,sts);ax.set_ylim(bottom=0);ax.set_box_aspect(1)
  ax.tick_params(axis='x',labelrotation=45,labelsize=4.7);ax.yaxis.set_major_locator(MaxNLocator(3));ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True)
 shared_legend(fig,[Line2D([],[],color=S.C['blue']),Line2D([],[],color=S.C['orange']),Line2D([],[],color=S.C['ink'],ls='-'),Line2D([],[],color=S.C['ink'],ls='--')],['Within-unit','Cross-unit','L2','L3'],columns=4)
 name='compact_neuron';save(fig,name)
 tex.append(texfig(name,r'Within-unit and cross-unit kernel dimensions for MNIST MLPs. Each panel combines L2 and L3, distinguished by line style. Curves are seed means and bands span seeds. ReLU within-unit directions agree with the selective construction where containment checks pass. Leaky-ReLU and GELU per-unit operators use their actual slopes; late GELU counts remain numerical threshold-dependent quantities.'))
 finish_fragment('section_c_selective_figures.tex',tex)






def response_figures(branches, margin=False):
 # Compress only stored probe arrays. Preserve model, seed, stage, scope and radius convention.
 data={}; sources=[]
 valid={(r['model'],r['seed'],r['stage'],r['order']) for r in branches}
 for f in sorted((RESULTS/'branches').glob('*rays.json')):
  tag=None
  for suffix in ['_rays','_ext_rays','_patch_rays']:
   tag=parse(f,suffix)
   if tag:break
  if not tag:continue
  m,seed=tag
  if 'gain_' in m:continue
  if margin and m not in ['mnist_mlp_w64','mnist_mlp_mgn_l0','mnist_mlp_mgn_l1']:continue
  if not margin and 'mgn_' in m:continue
  d=read(f)
  for st,v in resolved(d.get('results',{})):
   if margin and st!='late_endpoint':continue
   for name,sets in v.get('sets',{}).items():
    if name.startswith('branch:') and (m,seed,st,name.split(':',1)[1]) not in valid:continue
    if name=='branch' and 'cnn' in m and (m,seed,st,'selective_sum') not in valid:continue
    for mode,rr in sets.items():
     if mode not in ['absolute','relative'] or not isinstance(rr,dict):continue
     for radius,probes in rr.items():
      if not isinstance(probes,list) or not probes or 'true' not in probes[0]:continue
      data[m,seed,st,name,mode,float(radius)]=dict(epoch=v['epoch'],ten_percent=.1*parameter_norm(m,seed,v['epoch']),true=float(np.median([q['true'] for q in probes])),frozen=float(np.median([q['frozen'] for q in probes])),transition=float(np.median([q.get('transition',0) for q in probes])),probes=len(probes),source=str(f.relative_to(ROOT)))
 render_responses(data,margin)
 write_csv('margin_rays' if margin else 'branch_rays',[dict(model=k[0],seed=k[1],stage=k[2],direction_set=k[3],mode=k[4],radius=k[5],**v) for k,v in data.items()])


def transition_figures():
 rows=[]
 for folder in ['transition_v3','transition_bn']:
  for f in sorted((RESULTS/folder).glob('*.json')):
   pattern=r'(.+)_seed(\d+)_(absolute|relative)_(all|W\d+|V)\.json' if folder=='transition_v3' else r'(.+)_seed(\d+)\.json'
   match=re.fullmatch(pattern,f.name)
   if not match or match[1] not in LABELS:continue
   model,seed=match[1],int(match[2]);mode=match[3] if folder=='transition_v3' else 'absolute';scope=match[4] if folder=='transition_v3' else 'all';d=read(f)
   for st,v in resolved(d.get('results',{})):
    for r,qq in v.get('probes',{}).items():
     if not qq or 'true' not in qq[0]:continue
     vals=[q['transition']/q['true'] for q in qq if q['true']>0]
     if vals:rows.append(dict(model=model,seed=seed,stage=st,mode=mode,scope=scope,radius=float(r),share=float(np.median(vals)),probes=len(qq),ten_percent=ten_percent_radius(v,qq,float(r),mode),source=str(f.relative_to(ROOT))))
 stage_colours={st:S.STAGE_RAMP(i/(len(STAGES)-1)) for i,st in enumerate(STAGES)}
 for mode in ['absolute','relative']:
  cases=sorted({(r['model'],r['mode'],r['scope']) for r in rows if r['mode']==mode},key=lambda k:(list(LABELS).index(k[0]),k[2]));tex=[]
  for start in range(0,len(cases),16):
   cc=cases[start:start+16];fig,axes=dense_grid(len(cc),4,row_height=1.42,legend_height=.44)
   fig.supylabel('Transition norm / output-change norm',fontsize=7,x=.001)
   fig.supxlabel('Radius ('+mode+')',fontsize=7,y=0)
   shown=sorted({r['stage'] for r in rows if (r['model'],r['mode'],r['scope']) in cc},key=stage_order)
   for ax,(m,mode,scope) in zip(axes.flat,cc):
    radius_background(ax,mode)
    rr=[r for r in rows if (r['model'],r['mode'],r['scope'])==(m,mode,scope)];sts=sorted({r['stage'] for r in rr},key=stage_order)
    for st in sts:
     radii=sorted({r['radius'] for r in rr if r['stage']==st});vv=[[r['share'] for r in rr if r['stage']==st and r['radius']==radius] for radius in radii];col=stage_colours[st];ax.plot(radii,[max(np.mean(v),1e-12) for v in vv],color=col,marker='o',ms=1.5,lw=.85);ax.fill_between(radii,[max(min(v),1e-12) for v in vv],[max(max(v),1e-12) for v in vv],color=col,alpha=.08)
    limits=ax.get_xlim()
    for st in sts:
     refs=[r['ten_percent'] for r in rr if r['stage']==st and r['ten_percent'] is not None]
     if refs:
      ref=float(np.median(refs))
      if limits[0]<=ref<=limits[1]:ax.axvline(ref,color=stage_colours[st],ls=':',lw=.8,alpha=.85)
    ax.set(xscale='log',yscale='log');ax.xaxis.set_major_locator(LogLocator(numticks=4));ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_major_locator(LogLocator(numticks=4));ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(PLOT_LABELS[m]+'\n'+('All layers' if scope=='all' else layer_name(scope))+'; '+mode,loc='left',fontsize=7,pad=3)
   shared_legend(fig,[Line2D([],[],color=stage_colours[st]) for st in shown],[SHORT.get(st,st) for st in shown],columns=6)
   name=f'{mode}_transition_{start//16:02d}';save(fig,name)
   tex.append(texfig(name,r'Activation-transition contribution, with 256 directions per checkpoint and radius, at every tested radius and stage for the indicated editable blocks. MNIST is used unless labelled SVHN. Blue backgrounds mark absolute radii and orange backgrounds block-relative radii. Vertical dotted lines mark edits of norm $0.1\|\theta\|_2$, using the median scale conversion across seeds at each colour-coded stage; markers outside the scanned interval are omitted. The horizontal axis is radius, with the convention in each title. The vertical axis is $\|A_X^{\rm tr}\|_X/\|\Delta F_X\|_X$: medians over directions within seeds, then means across seeds; bands span seeds. Values below $10^{-12}$ use that plotting floor. Stage abbreviations follow the atomic panels. Ratios can exceed one when the transition and frozen responses cancel.'))
  finish_fragment('section_c_transition_figures.tex' if mode=='absolute' else 'section_f_transition_figures.tex',tex);write_csv('transition',rows)
 
def radius_background(ax,mode):
 from matplotlib.colors import to_rgba
 ax.set_facecolor(to_rgba(S.C['blue'] if mode=='absolute' else S.C['orange'],.055))

def ten_percent_radius(stage,probes,radius,mode):
 ratios=[q['radius_over_param_norm'] for q in probes if q.get('radius_over_param_norm',0)>0]
 if ratios:return float(.1*radius/np.median(ratios))
 if mode=='absolute' and stage.get('param_norm'):return .1*stage['param_norm']
 return None

def branch_attempt_figures():
 from matplotlib.colors import LogNorm
 attempts={}
 for f in sorted((RESULTS/'branches').glob('*.json')):
  tag=parse(f) or parse(f,'_branch')
  if not tag or any(k in tag[0] for k in ['gain_','mgn_']):continue
  m,seed=tag
  for st,v in resolved(read(f).get('results',{})):
   for order,c in v.get('constructions',{}).items():
    if 'total' not in c:continue
    val=c.get('validation',{});passed=bool(c.get('certified_branch',val.get('passed',False))) and not val.get('gram_fallback_used',False)
    attempts[m,seed,st,order]=dict(model=m,seed=seed,stage=st,epoch=v['epoch'],order=order,passed=passed,raw_dimension=c['total'],dimension=c['total'] if passed else 0,source=str(f.relative_to(ROOT)))
 for f in sorted((RESULTS/'branches').glob('*patch_spaces.json')):
  tag=parse(f,'_patch_spaces')
  if not tag:continue
  m,seed=tag
  for st,v in read(f).get('stages',{}).items():
   val=v.get('selective_sum_validation',{})
   if 'passed' not in val:continue
   attempts.setdefault((m,seed,st,'selective_sum'),dict(model=m,seed=seed,stage=st,epoch=v['epoch'],order='selective_sum',passed=val['passed'],raw_dimension=v['selective_sum_dim'],dimension=v['selective_sum_dim'] if val['passed'] else 0,source=str(f.relative_to(ROOT))))
 rows=list(attempts.values());write_csv('construction_attempts',rows)
 render_constructions(rows)


def transition_details():
 rows=[]
 models=['mnist_mlp','mnist_mlp_leaky','mnist_mlp_gelu','mnist_cnn']
 for m in models:
  for mode,target in [('absolute',1),('relative',.01)]:
   for seed in range(3):
    f=RESULTS/'transition_v3'/f'{m}_seed{seed}_{mode}_all.json'
    if not f.exists():continue
    for st,v in resolved(read(f)['results']):
     qq=next((q for r,q in v.get('probes',{}).items() if float(r)==target),[])
     if not qq:continue
     row=dict(model=m,mode=mode,seed=seed,stage=st,epoch=v['epoch'])
     for k in ['true','frozen','transition','propagated_sum_of_norms']:row[k]=float(np.median([q[k] for q in qq]))
     row['layers']=np.median([q['propagated_norms'] for q in qq],axis=0).tolist();rows.append(row)
 for mode in ['absolute','relative']:
  fig,axes=dense_grid(8,4,row_height=1.5,legend_height=.5)
  for index,m in enumerate(models):
   rr=[r for r in rows if r['model']==m and r['mode']==mode];sts=sorted({r['stage'] for r in rr},key=stage_order)
   for kind,ax in [('response',axes.flat[index]),('propagation',axes.flat[index+4])]:
    radius_background(ax,mode);ax.set_box_aspect(1)
    series=[('true',S.C['ink'],'-'),('frozen',S.C['blue'],'--'),('transition',S.C['orange'],':')] if kind=='response' else [('transition',S.C['ink'],'-'),('propagated_sum_of_norms',S.C['ink'],'--')]+[(i,block_colour(str(i+1)),'-') for i in range(len(rr[0]['layers']))]
    for key,col,ls in series:
     vals=[[r['layers'][key] if isinstance(key,int) else r[key] for r in rr if r['stage']==st] for st in sts]
     ax.plot(range(len(sts)),[max(np.mean(v),1e-16) for v in vals],color=col,ls=ls,lw=1)
    stage_ticks(ax,sts);ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(numticks=3));ax.yaxis.set_minor_formatter(NullFormatter());ax.set_title(PLOT_LABELS[m]+'\n'+kind,fontsize=7,loc='left')
  fig.supylabel('RMS output norm',fontsize=7,x=.001)
  shared_legend(fig,[Line2D([],[],color=S.C['ink']),Line2D([],[],color=S.C['blue'],ls='--'),Line2D([],[],color=S.C['orange'],ls=':')],['True (top)','Frozen (top)','Transition (top)'],columns=3)
  name=mode+'_transition_details';save(fig,name)
  cap=(r'Response magnitudes and layerwise propagation on 256 full-network directions per seed at '+('absolute radius 1.' if mode=='absolute' else r'block-relative radius $0.01$.')+r' Upper panels separate true, frozen and transition norms. Lower panels show each propagated layer residual in the atomic-plot block colours, their sum of norms (black dashed), and the norm after vector summation (black solid). Their separation measures cancellation across layers. Curves average within-seed medians over three seeds. Values below $10^{-16}$ use that plotting floor.')
  finish_fragment('section_c_transition_details.tex' if mode=='absolute' else 'section_f_transition_details.tex',[texfig(name,cap,'fig:c-transition-details' if mode=='absolute' else 'fig:f-transition-details')])
 write_csv('transition_details',rows)


def background_figure():
 cats=[('background','bg','Interior background'),('boundary','boundary','Padding boundary'),('digit','digit','Digit patches')]
 stages=['high_accuracy','epoch_250','mse_interpolation','late_endpoint'];marks=['o','^','s','D'];rows=[]
 for seed in range(3):
  f=RESULTS/'transition_v3'/f'mnist_cnn_seed{seed}_background_split.json'
  for st,v in read(f)['stages'].items():
   qq=v.get('probes',{}).get('0.01',[])
   if not qq:continue
   for flips,prop,label in cats:rows.append(dict(seed=seed,stage=st,kind=label,gates=float(np.median([q['flips_'+flips] for q in qq])),propagated=float(np.median([q['propagated_'+prop] for q in qq]))))
 fig,axes=dense_grid(3,3,row_height=1.8,legend_height=.4)
 for ax,(_,_,label) in zip(axes.flat,cats):
  ax.set_box_aspect(1)
  rr=[r for r in rows if r['kind']==label]
  for seed in range(3):
   qq=[next(r for r in rr if r['seed']==seed and r['stage']==st) for st in stages]
   ax.plot([r['gates'] for r in qq],[max(r['propagated'],1e-12) for r in qq],color=S.C['blue'],alpha=.18,lw=.8)
  means=[]
  for st,mk in zip(stages,marks):
   qq=[r for r in rr if r['stage']==st];x=np.mean([r['gates'] for r in qq]);y=max(np.mean([r['propagated'] for r in qq]),1e-12);means.append((x,y));ax.plot(x,y,marker=mk,color=S.C['blue'],ms=4)
  ax.plot(*np.array(means).T,color=S.C['blue'],lw=1);ax.set_xscale('symlog',linthresh=1);ax.set_yscale('log');ax.set_title(label,fontsize=7,loc='left');ax.set_xlabel('Gates crossed',fontsize=7)
  ax.yaxis.set_major_locator(LogLocator(numticks=4));ax.yaxis.set_minor_formatter(NullFormatter())
  ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10,linthresh=1));ax.xaxis.get_major_locator().set_params(numticks=4)
  xs=[r['gates'] for r in rr];ys=[r['propagated'] for r in rr]
  if min(xs)>0 and max(xs)/min(xs)<10:
   ax.set_xscale('linear');ax.xaxis.set_major_locator(MaxNLocator(3));ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
  if min(ys)>0 and max(ys)/min(ys)<10:
   ax.set_yscale('linear');ax.yaxis.set_major_locator(MaxNLocator(3));ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
 axes.flat[0].set_ylabel('Propagated RMS norm',fontsize=7)
 shared_legend(fig,[Line2D([],[],color=S.C['blue'],marker=mk,ls='') for mk in marks],['High accuracy','Epoch 250','Interpolation','Epoch 500'],columns=4)
 name='background_trajectory';save(fig,name)
 finish_fragment('section_c_background.tex',[texfig(name,r'Conv1 gate crossings and propagated residuals in the base MNIST CNN at block-relative radius $0.01$. Each point is a training stage: within-seed medians over 128 matched directions, then means over three seeds; faint paths show individual seeds. Paths follow training order. Background and boundary panels use a horizontal scale linear up to one crossing and logarithmic above it; the digit panel uses linear axes to resolve its narrower range. Zero propagated contributions are drawn at $10^{-12}$.','fig:c-cnn-background').replace('width=\\linewidth','width=.72\\linewidth')]);write_csv('background_trajectory',rows)


def square_sweeps():
 fig,ax=plt.subplots(figsize=(3.4,3.6));fig.subplots_adjust(left=.18,right=.96,bottom=.23,top=.79);ax.set_box_aspect(1)
 for axis,names,x,label,col in [(ax,['w64','w128','w256','w512'],[64,128,256,512],'Hidden width',S.C['blue']),(ax,['d2','w128','d4','d5'],[64,128,256,512],'Hidden depth',S.C['orange'])]:
  crossing=[];visible=[]
  for name in names:
   f=RESULTS/'paper_slices'/f'polar_{name}.npz';SOURCES.add(str(f.relative_to(ROOT)))
   with np.load(f) as z:
    rs=z['radii'];vals=[]
    for key,threshold in [('pol_flips',0),('pol_true',1e-3)]:vals.append(np.array([rs[np.flatnonzero(q>threshold)[0]] if np.any(q>threshold) else np.nan for q in z[key]]))
    crossing.append(vals[0]);visible.append(vals[1])
  for yy,ls in [(crossing,'-'),(visible,'--')]:
   med=np.array([np.nanmedian(q) for q in yy]);lo=np.array([np.nanquantile(q,.1) for q in yy]);hi=np.array([np.nanquantile(q,.9) for q in yy]);axis.errorbar(x,med,yerr=[med-lo,hi-med],color=col,ls=ls,marker='o',ms=3,capsize=2)
 ax.set_xlabel('Hidden width',fontsize=7,color=S.C['blue']);ax.tick_params(axis='x',colors=S.C['blue'],labelsize=7)
 ax.set_xscale('log',base=2);ax.set_xticks([64,128,256,512]);ax.set_xticklabels([64,128,256,512]);ax.minorticks_off();ax.set_yscale('log');ax.set_ylabel('Absolute radius',fontsize=7)
 top=ax.secondary_xaxis('top',functions=(lambda x:np.log2(np.maximum(x,1e-30))-4,lambda x:2.**(x+4)));top.set_xticks([2,3,4,5]);top.set_xticklabels([2,3,4,5]);top.set_xlabel('Hidden depth',fontsize=7,color=S.C['orange']);top.tick_params(axis='x',colors=S.C['orange'],labelsize=7);top.minorticks_off()
 fig.legend([Line2D([],[],color=S.C['ink']),Line2D([],[],color=S.C['ink'],ls='--')],['First crossing','First visible change'],loc='lower center',ncol=2,fontsize=6.5)
 save(fig,'square_sweeps')




_NORM_CACHE={}
def parameter_norm(model,seed,epoch):
 # Retrieve the norm from training history at the exact checkpoint, without loading a model.
 base='mnist_mlp' if re.fullmatch(r'mnist_mlp_sub\d+',model) else model
 key=(base,seed)
 if key not in _NORM_CACHE:
  f=RESULTS/'tier1_models'/base/f'seed_{seed}'/'history.json'
  d=read(f);_NORM_CACHE[key]={r['epoch']:r['param_norm']['total'] if isinstance(r['param_norm'],dict) else r['param_norm'] for r in d['epochs']}
 return _NORM_CACHE[key][epoch]


def render_responses(data,margin):
 from matplotlib.colors import to_rgba
 display={'atomic_sum':'Atomic sum (CNN)','branch':'Selective branch','branch:bottom_up':'Bottom-up branch','branch:top_down':'Top-down branch','branch:feature_null':'Feature-null branch','constructed':'Constructed branch','generic':'Generic','naive_atomic_sum':'Atomic sum (MLP)','tangent_flat':'Tangent control'}
 groups=[('margin',['mnist_mlp_w64','mnist_mlp_mgn_l0','mnist_mlp_mgn_l1'])] if margin else [
  ('activation',['mnist_mlp_w64','mnist_mlp_leaky_w64','mnist_mlp_leaky']),
  ('setting',['mnist_cnn','mnist_mlp_bn_w64','mnist_mlp_raw','svhn_mlp_w64','svhn_mlp_w64_v2']),
  ('data',[m for m in LABELS if '_sub' in m])]
 colour_map={m:S.LAYER_COLS[i%5] for _,models in groups for i,m in enumerate(models)}
 # Width distinguishes the two leaky models; activation retains its colour.
 colour_map.update(mnist_mlp_w64=S.C['blue'],mnist_mlp_leaky_w64=S.C['orange'],mnist_mlp_leaky=S.C['orange'])
 if margin:colour_map.update(mnist_mlp_w64=S.C['ink2'],mnist_mlp_mgn_l0=S.C['blue'],mnist_mlp_mgn_l1=S.C['orange'])
 outputs=defaultdict(list)
 for mode in ['absolute','relative']:
  for group,models in groups:
   subset={k:v for k,v in data.items() if k[0] in models and k[4]==mode}
   if not subset:continue
   # Two data conditions occupy separate panels; colour identifies constraint-set size.
   kinds=['trained_subset','fixed_network'] if group=='data' else [group]
   for kind in kinds:
    dd={k:v for k,v in subset.items() if group!='data' or ('_p1_' in k[0])==(kind=='trained_subset')}
    if not dd:continue
    cases=sorted({(k[2],k[3]) for k in dd},key=lambda k:(stage_order(k[0]),list(display).index(k[1]) if k[1] in display else 99))
    present=[m for m in models if any(k[0]==m for k in dd)]
    if group=='data':
     for m in present:colour_map[m]=S.LAYER_COLS[[1000,2000,5000].index(int(m.rsplit('sub',1)[1]))]
    for start in range(0,len(cases),16):
     cc=cases[start:start+16];columns=min(4,len(cc));fig,axes=dense_grid(len(cc),columns,row_height=1.65,legend_height=.58)
     print_width=columns/4
     fig.set_size_inches(6.9*print_width,fig.get_figheight())
     fig.subplots_adjust(left=.055/print_width,wspace=.45)
     fig.supylabel('RMS output change',fontsize=7,x=.001);fig.supxlabel('Absolute radius' if mode=='absolute' else 'Block-relative radius',fontsize=7,y=0)
     for ax,(st,kind_set) in zip(axes.flat,cc):
      ax.set_box_aspect(1)
      bg='blue' if kind_set.startswith('branch') or kind_set=='constructed' else ('orange' if kind_set in ['naive_atomic_sum','atomic_sum'] else 'ink3')
      ax.set_facecolor(to_rgba(S.C[bg],.045))
      for m in present:
       rr={k:v for k,v in dd.items() if k[0]==m and k[2]==st and k[3]==kind_set}
       if not rr:continue
       radii=sorted({k[-1] for k in rr});col=colour_map[m];mk='s' if m=='mnist_mlp_leaky' else 'o'
       for metric,ls in [('true','-'),('frozen','--')]:
        vals=[[v[metric] for k,v in rr.items() if k[-1]==r] for r in radii]
        ax.plot(radii,[max(np.mean(v),1e-16) for v in vals],color=col,ls=ls,lw=1.3,marker=mk if metric=='true' else None,ms=3)
        if metric=='true':ax.fill_between(radii,[max(min(v),1e-16) for v in vals],[max(max(v),1e-16) for v in vals],color=col,alpha=.1)
       if mode=='absolute':ax.axvline(np.median([v['ten_percent'] for v in rr.values()]),color=col,ls=':',lw=.65,alpha=.75)
      ax.axhline(1e-3,color=S.C['ink3'],ls=':',lw=.6)
      ax.set(xscale='log',yscale='log',ylim=(1e-16,None));ax.xaxis.set_major_locator(LogLocator(numticks=3));ax.yaxis.set_major_locator(LogLocator(numticks=4));ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
      counts=[len({k[1] for k in dd if k[0]==m and k[2]==st and k[3]==kind_set}) for m in present]
      title=display.get(kind_set,kind_set)+'\n'+SHORT.get(st,st)+'; n='+ '/'.join(str(n) if n else '–' for n in counts)
      ax.set_title(title,loc='left',fontsize=6.6,pad=3)
     shared_legend(fig,[Line2D([],[],color=colour_map[m],marker='s' if m=='mnist_mlp_leaky' else 'o',ms=3) for m in present],[PLOT_LABELS[m] for m in present],columns=min(len(present),columns))
     name=f'rays_{mode}_{kind}_{start//16:02d}';save(fig,name)
     caption=(r'Finite-distance '+('absolute-radius' if mode=='absolute' else 'block-relative-radius')+r' responses. Colours identify the models in the legend. '+(r'Square markers distinguish width-128 leaky ReLU from width 64. ' if group=='activation' else '')+r'Each panel fixes the direction family and training stage. Solid curves show true responses and dashed curves frozen responses; means and ranges summarize seed medians. Blue-tinted panels show branch candidates, orange atomic sums and grey controls. '+(r'Vertical dotted lines mark $0.1\|\theta\|_2$, using the median network norm across the contributing seeds. ' if mode=='absolute' else '')+r'The horizontal dotted line marks output change $10^{-3}$; values below $10^{-16}$ use that floor. The $n$ values in each title give seed counts in legend order; a dash denotes an absent curve. Unvalidated atomic sums and tangent directions are controls. ')
     if group=='data':caption+=r'``Train on $n$'' means training from initialization on that subset; ``Full-data, $n$ constraints'' keeps the network trained on all images and restricts its analysis set. Input blocks stay fixed.'
     dest=('section_f_margin_rays.tex' if margin else 'section_f_ray_figures.tex') if mode=='relative' else ('section_c_margin_rays.tex' if margin else ('section_d_data_rays.tex' if group=='data' else 'section_c_ray_figures.tex'))
     outputs[dest].append(texfig(name,caption).replace(r'width=\linewidth',f'width={print_width:.2f}'+r'\linewidth'))
 for dest,figs in outputs.items():finish_fragment(dest,figs)


def render_constructions(rows):
 for row in rows:
  for key in ["seed", "epoch", "dimension"]: row[key] = int(float(row[key]))
 orders=[o for o in ['bottom_up','top_down','naive_atomic_sum','feature_null','selective_sum']+[f'random_{i}' for i in range(6)] if any(r['order']==o for r in rows)]
 labels={'bottom_up':'Bottom-up','top_down':'Top-down','naive_atomic_sum':'Atomic sum','feature_null':'Feature-null','selective_sum':'Selective'}
 colours={o:plt.get_cmap('tab20')(i) for i,o in enumerate(orders)}
 for data_group,dest in [(False,'section_c_branch_figures.tex'),(True,'section_d_data_branches.tex')]:
  models=[m for m in LABELS if ('_sub' in m)==data_group and any(len({r['epoch'] for r in rows if r['model']==m and r['seed']==seed})>1 for seed in {r['seed'] for r in rows if r['model']==m})]
  tex=[]
  for start in range(0,len(models),12):
   cc=models[start:start+12];fig,axes=dense_grid(len(cc),6 if len(cc)==6 else 4,row_height=1.35,legend_height=.5)
   fig.supylabel('Validated branch dimension',fontsize=7,x=.001);fig.supxlabel('Training epoch',fontsize=7,y=0)
   for ax,m in zip(axes.flat,cc):
    ax.set_box_aspect(1);rr=[r for r in rows if r['model']==m]
    for o in orders:
     # Milestone labels differ in epoch by seed, so plot each seed at its actual epoch.
     for seed in sorted({r['seed'] for r in rr}):
      qq=sorted([r for r in rr if r['order']==o and r['seed']==seed],key=lambda r:r['epoch'])
      if not qq:continue
      ax.plot([r['epoch'] for r in qq],[r['dimension'] for r in qq],color=colours[o],ls='--' if o.startswith('random') else '-',lw=1.6,alpha=.85,marker='o',ms=4.2)
      for r in qq:
       if r['dimension']==0:ax.plot(r['epoch'],0,marker='x',ms=5.5,color=colours[o],mew=1.3,clip_on=False)
    ax.set_ylim(bottom=0);ax.set_xlim(-10,520);ax.set_xticks([0,250,500]);ax.yaxis.set_major_locator(MaxNLocator(3));ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True);ax.set_title(PLOT_LABELS[m],fontsize=6.8,loc='left')
   shown=[o for o in orders if any(r['model'] in cc and r['order']==o for r in rows)]
   shared_legend(fig,[Line2D([],[],color=colours[o],ls='--' if o.startswith('random') else '-') for o in shown],[labels.get(o,o.replace('random_','R')) for o in shown],columns=6)
   name=f'construction_time_{"data" if data_group else "general"}_{start//12:02d}';save(fig,name)
   tex.append(texfig(name,r'Multilayer branch dimensions versus actual checkpoint epoch. Each curve is one seed; colours identify constructions, and dashed curves denote random layer orders R0--R5. Coincident curves have equal retained dimensions. Crosses mark zero retained dimension, including failed validations; unattempted constructions have no point. MNIST is used unless labelled SVHN. Only models with a trajectory measured within at least one seed are shown.'))
  finish_fragment(dest,tex)



