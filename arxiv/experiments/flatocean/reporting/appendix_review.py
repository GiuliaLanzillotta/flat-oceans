"""Compact appendix figures from existing scalar records and slice arrays only."""
from . import section_c as C
from .section_c import np, plt, S, RESULTS, OUT, PAPER, read, save, dense_grid, shared_legend, texfig, finish_fragment, Line2D, LogLocator, NullFormatter


def save(fig, name):
 if name in {'activation_rays_square', 'gelu_rays_norm', 'nc_square'}:
  from matplotlib.text import Text
  for text in fig.findobj(Text):
   text.set_fontsize(max(text.get_fontsize(), 8))
  if name=='activation_rays_square':
   fig.subplots_adjust(left=.14,right=.99,bottom=.2,top=.72,wspace=.48)
   for legend in fig.legends: legend.remove()
   shared_legend(fig,[Line2D([],[],color=S.C[k],marker=mk,ms=4) for k,mk in [('blue','o'),('ink','^'),('purple','s'),('ink3','')]],['L2 kernel','L3 kernel',r'L3 tail $10^{-3}$','Generic'],columns=2)
  if name=='nc_square':fig.subplots_adjust(left=.15,right=.99,bottom=.27,top=.77,wspace=.55)
 C.save(fig,name)


def normline(ax,norm,col=S.C['ink3']):
 ax.axvline(.1*norm,color=col,ls=':',lw=1.1,alpha=.7)


def atomic_rays():
 stages=['high_accuracy','late_endpoint'];cats=[('kernel_2','Kernel L2','blue','-'),('kernel_3','Kernel L3','ink','-'),('kernel_readout','Readout kernel','aqua','-'),('tail_3_0.001',r'Tail $10^{-3}$ L3','purple','-'),('dead_in_3','Dead-unit incoming','orange','--'),('dead_out_3','Zero-feature outgoing','yellow','--')]
 fig,axes=dense_grid(4,4,row_height=1.8,legend_height=.65)
 for ax,st in zip(axes.flat,stages):
  for cat,lab,key,ls in cats+[('generic','Generic','ink3','-')]:
   curves=[];norms=[]
   for seed in range(3):
    d=read(RESULTS/'transition_v2'/f'mnist_mlp_seed{seed}{"" if cat=="generic" else "_structured"}.json')['results'].get(st,{})
    v=d if cat=='generic' else d.get('categories',{}).get(cat,{})
    probes=v.get('probes',v.get('radii',{}))
    if not probes or not any(probes.values()):continue
    probes={r:q for r,q in probes.items() if q}
    rs=sorted(probes,key=float);curves.append([np.median([q['true'] for q in probes[r]]) for r in rs]);norms.append(d['param_norm'])
   if not curves:continue
   yy=np.maximum(curves,1e-17);ax.plot(list(map(float,rs)),np.median(yy,axis=0),color=S.C[key],ls=ls,lw=1.4)
   ax.fill_between(list(map(float,rs)),yy.min(0),yy.max(0),color=S.C[key],alpha=.1)
  normline(ax,np.median(norms));ax.axhline(1e-3,color=S.C['ink3'],lw=.6,ls=':');ax.set(xscale='log',yscale='log',ylim=(1e-17,2));ax.set_title('ReLU w128\n'+C.SLONG[st],loc='left',fontsize=7)
 for ax,pre,title in [(axes.flat[2],['w64_branch_branch','w64_featurenull_branch','w64_branch_generic'],'ReLU w64; zoom'),(axes.flat[3],['cnn_branch','cnn_featurenull','cnn_random'],'CNN; zoom')]:
  for name,lab,key in [(pre[2],'Generic','ink3'),(pre[1],'Feature-null','aqua'),(pre[0],'Branch','blue')]:
   with np.load(RESULTS/'paper_slices'/f'{name}.npz') as z:
    r=z['radii'];y=z['pol_true'];flips=z['pol_flips'];norm=z['theta_norm'].item()
    if 'featurenull' in name:
     a=z['angles'];distance=np.minimum(abs(a),np.minimum(abs(a-np.pi),abs(a-2*np.pi)));y=y[distance<1.5*2*np.pi/len(a)]
    ax.plot(r,np.median(y,axis=0),color=S.C[key],lw=1.4);ax.fill_between(r,np.quantile(y,.1,axis=0),np.quantile(y,.9,axis=0),color=S.C[key],alpha=.1)
    if name==pre[0]:
     crossing=[r[np.flatnonzero(f>0)[0]] for f in flips if np.any(f>0)]
     ax.axvline(np.median(crossing),color=S.C['blue'],ls='--',lw=1.1);normline(ax,norm)
  ax.set(xlim=(0,max(r.max(),.105*norm)),ylim=(0,None));ax.set_title(title,fontsize=7,loc='left');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,3))
 for ax in axes.flat:ax.set_box_aspect(1);ax.set_xlabel('Absolute radius',fontsize=6);ax.yaxis.set_major_locator(LogLocator(numticks=4) if ax.get_yscale()=='log' else C.MaxNLocator(3))
 axes.flat[0].set_ylabel('RMS output change',fontsize=6)
 keys=[('Generic','ink3','-')]+[(lab,k,ls) for _,lab,k,ls in cats]+[('Branch (zoom)','blue','-'),('Feature-null (zoom)','aqua','-')]
 shared_legend(fig,[Line2D([],[],color=S.C[k],ls=ls) for lab,k,ls in keys],[lab for lab,k,ls in keys],columns=4)
 save(fig,'atomic_rays_zoom')
 finish_fragment('section_c_atomic_rays.tex',[texfig('atomic_rays_zoom',r'Atomic rays and zoomed branch responses on MNIST. Left pair: width-128 ReLU single-block kernels, approximate tails and dead-unit controls, with medians of 32 probes per seed and ranges across three seeds; the generic reference edits the whole network. Right pair: seed-0 width-64 MLP and CNN slices, with medians and 10th--90th percentiles over angles; feature-null curves use their own slice axis. These panels use linear axes to resolve visible drift after a first crossing (blue dashed marker). Grey vertical dotted markers give $0.1\|\theta\|_2$; the left panels also mark output visibility at $10^{-3}$. Blue and green retain the invariant-direction and feature-null roles across panels, while the legend distinguishes atomic kernels from multilayer branches.','fig:rays-overview').replace(r'\label{fig:rays-overview}',r'\label{fig:rays-overview}\label{fig:zoom-rays}')])


def activation_rays():
 fig,axes=dense_grid(2,2,row_height=2.2,legend_height=.65);fig.set_size_inches(4.4,2.85)
 cats=[('kernel_2','L2 kernel','blue','o'),('kernel_3','L3 kernel','ink','^'),('tail_3_0.001',r'L3 tail $10^{-3}$','purple','s')]
 for ax,tag in zip(axes.flat,['leaky','gelu']):
  data=read(RESULTS/'transition_v2'/f'mnist_mlp_{tag}_seed0_structured.json')['results']
  generic=read(RESULTS/'transition_v3/summary'/f'mnist_mlp_{tag}_seed0_absolute_all.json')['stages']
  for st,alpha in [('high_accuracy',.38),('late_endpoint',1)]:
   v=data[st]
   for cat,lab,key,mk in cats:
    probes=v['categories'].get(cat,{}).get('probes',{})
    if not probes:continue
    rr=sorted(probes,key=float)
    for metric,ls in [('true','-'),('frozen','--')]:ax.plot(list(map(float,rr)),[max(np.median([q[metric] for q in probes[r]]),1e-16) for r in rr],color=S.C[key],ls=ls,alpha=alpha,marker=mk if metric=='true' else None,ms=3.4,lw=1.35)
   rad=generic.get(st,{}).get('radii',{})
   if rad:
    rr=sorted(rad,key=float);ax.plot(list(map(float,rr)),[rad[r]['true']['50'] if isinstance(rad[r]['true'],dict) else rad[r]['true'] for r in rr],color=S.C['ink3'],alpha=alpha,lw=1.4)
   normline(ax,v['param_norm'])
  ax.set(xscale='log',yscale='log',ylim=(1e-16,10));ax.set_box_aspect(1);ax.set_title('Leaky ReLU' if tag=='leaky' else 'GELU',fontsize=7,loc='left');ax.set_xlabel('Absolute radius',fontsize=6);ax.xaxis.set_major_locator(LogLocator(numticks=3));ax.yaxis.set_major_locator(LogLocator(numticks=4))
 axes.flat[0].set_ylabel('RMS output change',fontsize=6)
 shared_legend(fig,[Line2D([],[],color=S.C[k],marker=mk,ms=3.4) for _,lab,k,mk in cats]+[Line2D([],[],color=S.C['ink3'])],[lab for _,lab,k,mk in cats]+['Generic'],columns=4)
 save(fig,'activation_rays_square')






def persistence():
 models=['mnist_mlp','mnist_mlp_leaky','mnist_mlp_gelu'];rows=[]
 for m in models:
  for seed in range(3):
   d=read(RESULTS/'persistence'/f'{m}_seed{seed}_geometry.json')
   for b,block in d['blocks'].items():
    for pair,v in block['pairs'].items():
     for family,q in v['families'].items():
      if 'to_later_kernel' not in q:continue
      z=q['to_later_kernel'];a=q.get('amplification') or {}
      rows.append(dict(model=m,seed=seed,block=b,pair=pair,family=family,dimension=q['dim_source'],Q=z['Q'],lost=z['d_lost'],gain=a.get('G_over_generic_gain'),hist=z['hist_sin2'],edges=d['sin2_hist_edges'],large=z.get('d_lost_share_from_sin2>=0.5'),**(q.get('retained_mass_split') or {})))
 pairs=['high_accuracy->mse_interpolation','mse_interpolation->epoch_100','epoch_100->epoch_250','epoch_250->late_endpoint','high_accuracy->late_endpoint','mse_interpolation->late_endpoint'];labs=['HA→Int','Int→100','100→250','250→500','HA→500','Int→500'];tex=[]
 cases=[(m,b) for m in models for b in ['2','3','readout'] if any(r['model']==m and r['block']==b for r in rows)]
 fig,axes=dense_grid(len(cases)*3,6,row_height=1.13,legend_height=.30)
 fig.subplots_adjust(wspace=.50,hspace=.45)
 for row,(metric,ylabel) in enumerate([('Q','Retained mass'),('lost','Lost dimensions'),('gain','Gain / generic')]):
  for j,(m,b) in enumerate(cases):
   ax=axes[row,j];rr=[r for r in rows if r['model']==m and r['block']==b]
   for fam,col,ls in [('K',S.C['blue'],'-'),('S',S.C['orange'],'--'),('D',S.C['aqua'],':')]:
    vv=[[r[metric] for r in rr if r['family']==fam and r['pair']==pair and r[metric] is not None] for pair in pairs]
    yy=[np.median(v) if v else np.nan for v in vv];lo=[min(v) if v else np.nan for v in vv];hi=[max(v) if v else np.nan for v in vv]
    ax.plot(range(6),yy,color=col,ls=ls,marker='o',ms=3,lw=1.3);ax.fill_between(range(6),lo,hi,color=col,alpha=.08)
   ax.set_xticks(range(6));ax.set_xticklabels(range(1,7),fontsize=5)
   act='GELU' if 'gelu' in m else ('Leaky' if 'leaky' in m else 'ReLU')
   ax.set_title(act+'; '+('readout' if b=='readout' else 'L'+b),fontsize=6,loc='left');ax.set_box_aspect(1)
   if metric=='Q':ax.set_ylim(-.02,1.02);ax.set_yticks([0,.5,1])
   else:
    ax.set_yscale('symlog',linthresh=.001 if metric=='gain' else 1);ax.yaxis.set_major_locator(C.SymmetricalLogLocator(base=10,linthresh=.001 if metric=='gain' else 1));ax.yaxis.get_major_locator().set_params(numticks=3);ax.set_ylim(bottom=0)
   if j==0:ax.set_ylabel(ylabel,fontsize=6)
   ax.tick_params(axis='y',labelsize=5)
 shared_legend(fig,[Line2D([],[],color=S.C[c],ls=ls) for c,ls in [('blue','-'),('orange','--'),('aqua',':')]],['Full kernel','Dead coordinates removed','Co-occurrence subspace'],columns=3)
 save(fig,'persistence_summary')
 cap=(r'Checkpoint persistence in width-128 MNIST MLPs. Rows show retained mass, effective lost dimension, and lost-component gain relative to a generic direction. Columns retain every measured nonempty block. Pair indices 1--6 denote HA$\to$Int, Int$\to$100, 100$\to$250, 250$\to$500, HA$\to$500 and Int$\to$500; HA and Int are high accuracy and interpolation. Curves show medians and ranges across three seeds. Component curves appear only where their projector split resolves. Empty source spaces have no retention value. GELU kernels remain numerical threshold-dependent quantities. The lower two rows use a scale linear near zero and logarithmic above it.')
 tex.append(texfig('persistence_summary',cap,'fig:persistence'))
 # Principal angles, including every measured pair and seed, for full kernels.
 fig,axes=dense_grid(len(cases),6,row_height=1.25,legend_height=.36)
 for ax,(m,b) in zip(axes.flat,cases):
  for pi,pair in enumerate(pairs):
   rr=[r for r in rows if r['model']==m and r['block']==b and r['family']=='K' and r['pair']==pair]
   if not rr:continue
   edges=np.r_[rr[0]['edges'],1+1e-12];ys=np.array([r['hist'] for r in rr],float);ys=ys/ys.sum(1)[:,None]
   ax.plot(np.log10(np.maximum((edges[:-1]+edges[1:])/2,1e-16)),np.mean(ys,0),color=plt.get_cmap('tab10')(pi),lw=1.3)
  ax.set(yscale='log',ylim=(1e-5,1.1),xticks=[-16,-8,0]);ax.set_box_aspect(1);ax.set_title(('GELU' if 'gelu' in m else ('Leaky' if 'leaky' in m else 'ReLU'))+'; '+('out' if b=='readout' else 'L'+b),fontsize=6.5,loc='left')
 fig.supylabel('Fraction of principal angles',fontsize=7,x=.001);fig.supxlabel(r'$\log_{10}\sin^2\theta$',fontsize=7,y=0);shared_legend(fig,[Line2D([],[],color=plt.get_cmap('tab10')(i)) for i in range(6)],labs,columns=6)
 save(fig,'persistence_angles');tex.append(texfig('persistence_angles',r'Distribution of principal angles for every checkpoint pair in the preceding figures. Histograms are normalized within a seed, then averaged across seeds. The left edge collects angles at numerical zero; the right edge represents directions nearly orthogonal to the later kernel. ReLU persistence combines a large nearly aligned part with a much smaller set of large-angle directions. Leaky ReLU loses more alignment late in training.','fig:persistence-angles'))
 tex[0]=tex[0].replace(r'\end{figure}', '')
 tex[1]=tex[1].replace(r'\begin{figure}[!htbp]\centering',r'\par\smallskip')
 finish_fragment('section_d_persistence.tex',tex);C.write_csv('persistence_geometry',rows)
 persistence_tails()


def persistence_tails():
 records={seed:read(RESULTS/'persistence'/f'mnist_mlp_seed{seed}_overlaps.json') for seed in range(3)}
 blocks=['2','3','readout'];fig,display_axes=dense_grid(12,6,row_height=1.38,legend_height=.48)
 axes=display_axes.reshape(4,3)
 fig.subplots_adjust(wspace=.62,hspace=.5)
 for j,b in enumerate(blocks):
  for seed,d in records.items():
   epochs=sorted(int(e) for e in d['per_stage'][b]);pairs=[p for p in d['pairs'] if p['block']==b];col=C.block_colour(b)
   for p in pairs:
    e1,e2=p['epochs'];consecutive=epochs.index(e2)==epochs.index(e1)+1
    x=e1;mk='o' if consecutive else 's'
    for row,key in [(0,'Q_V_U'),(1,'projector_distance')]:
     y=p['K_K'][key]
     if y is not None:axes.flat[row*3+j].plot(x,y,marker=mk,color=col,ms=3.4,alpha=.65,ls='')
   # All positive-tail thresholds for each consecutive pair; seed paths remain visible.
   for p in pairs:
    e1,e2=p['epochs']
    if epochs.index(e2)!=epochs.index(e1)+1:continue
    colour=S.STAGE_RAMP(.15+.85*epochs.index(e1)/max(1,len(epochs)-2));eps=sorted(map(float,p['Eplus_K']))
    yy=[p['Eplus_K'][str(e)]['Q_U_V'] for e in eps];dims=[p['Eplus_K'][str(e)]['dim_Eplus'] for e in eps]
    axes.flat[6+j].plot(eps,[np.nan if y is None else y for y in yy],color=colour,lw=1.1,alpha=.7)
    axes.flat[9+j].plot(eps,dims,color=colour,lw=1.1,alpha=.7)
  for row in range(4):
   ax=axes.flat[row*3+j];ax.set_box_aspect(1);ax.set_title(['Backward mass','Distance','Tail alignment','Tail dimension'][row]+'; '+('out' if b=='readout' else 'L'+b),fontsize=6.4,loc='left')
   if row<2:ax.set_xlabel('Earlier epoch',fontsize=6)
   else:ax.set_xscale('log');ax.set_xlabel('Tail threshold',fontsize=6);ax.xaxis.set_major_locator(LogLocator(numticks=3))
   if row in [0,2]:ax.set_ylim(-.02,1.02)
   else:ax.set_yscale('symlog',linthresh=1);ax.yaxis.set_major_locator(C.SymmetricalLogLocator(base=10,linthresh=1));ax.yaxis.get_major_locator().set_params(numticks=4);ax.set_ylim(bottom=0)
 for row,label in enumerate(['Backward mass','Projector distance','Tail alignment','Tail dimension']):axes.flat[row*3].set_ylabel(label,fontsize=6)
 shared_legend(fig,[Line2D([],[],marker='o',ls='',color=S.C['ink']),Line2D([],[],marker='s',ls='',color=S.C['ink']),Line2D([],[],color=S.STAGE_RAMP(.2)),Line2D([],[],color=S.STAGE_RAMP(1.0))],['Successive checkpoints','Earlier → endpoint','Earlier tail pair','Later tail pair'],columns=4)
 save(fig,'persistence_tails')
 finish_fragment('section_d_persistence_tails.tex',[texfig('persistence_tails',r'Growth and spectral-tail alignment in the width-128 ReLU MLP, all three seeds. Panels 1--3 show how much of the later kernel was already present earlier; panels 4--6 show the projector distance, for every recorded checkpoint pair. Panels 7--9 and 10--12 show the positive spectral tail, excluding the current numerical kernel: its mean squared projection into the next kernel and its dimension, over the complete threshold grid on $J/\sqrt{s}$. Each line is one seed and consecutive checkpoint pair; darker lines denote later pairs. Alignment is projected mass, not an exact intersection dimension.','fig:persistence-tails')])


def residual_rays():
 data={act:read(RESULTS/'transition_v2'/f'mnist_mlp{tag}_seed0_epsilon_flat.json')['stages']['late_endpoint'] for act,tag in [('ReLU',''),('GELU','_gelu')]}
 fig,axes=dense_grid(3,3,row_height=1.9,legend_height=.5)
 ge=data['GELU']['categories'];re=data['ReLU']['categories'];colours=[S.C[k] for k in ['aqua','blue','purple','orange']]
 def curve(cat,key):
  v=cat['survival'];rs=sorted(v,key=float)
  return list(map(float,rs)),[v[r][key] for r in rs]
 for cat,col,lab in [(ge['tail_3_0.001'],colours[0],'GELU tail'),(re['kernel_3'],colours[1],'ReLU kernel')]:
  for key,ls in [('true_median','-'),('frozen_median','--')]:
   x,y=curve(cat,key);axes.flat[0].plot(x,np.maximum(y,1e-16),color=col,ls=ls,lw=1.4)
 axes.flat[0].axhline(1e-3,color=S.C['ink3'],ls=':',lw=1.1)
 table=data['GELU']['bound']['modes']['absolute']['table'];rs=sorted(table,key=float)
 for key,col,ls in [('transition_median',S.C['aqua'],'-'),('bound_rms_median',S.C['ink3'],'--')]:axes.flat[1].plot(list(map(float,rs)),[table[r][key] for r in rs],color=col,ls=ls,lw=1.4)
 for cat,col,lab in [(ge['tail_3_0.001'],colours[0],r'GELU tail $10^{-3}$'),(re['kernel_3'],colours[1],'ReLU kernel'),(ge['tail_3_1e-06'],colours[2],r'GELU tail $10^{-6}$'),(re['tail_3_0.001'],colours[3],r'ReLU tail $10^{-3}$')]:
  x,y=curve(cat,'survival');axes.flat[2].plot(x,[v['0.001'] for v in y],color=col,marker='o',ms=3.4,lw=1.4,label=lab)
 for j,ax in enumerate(axes.flat):
  ax.set_box_aspect(1);ax.set_xscale('log');ax.set_xlabel('Absolute radius',fontsize=6)
  for m,col in [('mnist_mlp',S.C['blue']),('mnist_mlp_gelu',S.C['aqua'])]:
   if j==1 and m=='mnist_mlp':continue
   normline(ax,C.parameter_norm(m,0,500),col)
  ax.set_title(['True and frozen','Transition and proxy','Below visibility threshold'][j],loc='left',fontsize=7)
  if j<2:ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(numticks=4))
  ax.xaxis.set_major_locator(LogLocator(numticks=3));ax.set_ylabel(['RMS output change','RMS transition','Probe fraction'][j],fontsize=6)
 handles,labels=axes.flat[2].get_legend_handles_labels();shared_legend(fig,handles,labels,columns=4);save(fig,'gelu_rays_norm')
 # Preserve the separate SVHN tangent-complement comparison.
 fig,ax=plt.subplots(figsize=(2.8,2.9));fig.subplots_adjust(left=.21,right=.97,top=.84,bottom=.30);ax.set_box_aspect(1)
 for name,col,lab in [('branch:bottom_up',S.C['blue'],'Branch'),('generic',S.C['ink3'],'Generic'),('tangent_flat',S.C['purple'],'Tangent complement')]:
  curves=[]
  for seed in range(3 if name!='tangent_flat' else 1):
   sets=read(RESULTS/'branches'/f'svhn_mlp_w64_seed{seed}_rays.json')['results']['late_endpoint']['sets'];rr=sets[name]['absolute'];rs=sorted(rr,key=float);y=[np.median([q['true'] for q in rr[r]]) for r in rs];curves.append(y);ax.plot(list(map(float,rs)),np.maximum(y,1e-17),color=col,alpha=.3,lw=1.1)
   if seed==0 and name=='branch:bottom_up':ax.plot(list(map(float,rs)),[max(np.median([q['frozen'] for q in rr[r]]),1e-17) for r in rs],color=col,ls='--',lw=1.4)
  ax.plot(list(map(float,rs)),np.maximum(np.median(curves,axis=0),1e-17),color=col,lw=1,label=lab)
 normline(ax,np.median([C.parameter_norm('svhn_mlp_w64',seed,500) for seed in range(3)]));ax.axhline(1e-3,color=S.C['ink3'],ls=':',lw=1.1)
 ax.set(xscale='log',yscale='log',xlabel='Absolute radius',ylabel='RMS output change',title='SVHN, width 64');ax.yaxis.set_major_locator(LogLocator(numticks=4));ax.xaxis.set_major_locator(LogLocator(numticks=3));h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=6);save(fig,'svhn_rays_norm')


def persistence_diagnostics():
 rec={seed:read(RESULTS/'persistence'/f'mnist_mlp_seed{seed}_geometry.json') for seed in range(3)}
 blocks=['2','3','readout'];fig,display_axes=dense_grid(12,6,row_height=1.38,legend_height=.48)
 axes=display_axes.reshape(4,3)
 fig.subplots_adjust(wspace=.62,hspace=.5)
 families=[('K','Full kernel','blue'),('Cb','Dead coordinates','orange'),('S','Dead coordinates removed','aqua'),('D','Co-occurrence subspace','purple')]
 for j,b in enumerate(blocks):
  for seed,d in rec.items():
   block=d['blocks'][b];epochs=sorted(map(int,block['stages']))
   for family,lab,col in families:axes.flat[j].plot(epochs,[block['stages'][str(e)]['dims'].get(family,np.nan) for e in epochs],color=S.C[col],lw=1.1,alpha=.65)
   for pair,v in block['pairs'].items():
    q=v['families'].get('S',{});split=q.get('retained_mass_split')
    if split and 'to_later_kernel' in q:
     x=v['epochs'][0];mk='s' if v['epochs'][1]==500 else 'o'
     for key,col in [('in_later_same_family','aqua'),('in_later_dead_coordinates','orange')]:axes.flat[3+j].plot(x,split[key],color=S.C[col],marker=mk,ms=3.4,ls='',alpha=.6)
    amp=v['families']['K'].get('amplification')
    if amp:
     vals=amp['lost_mass_share_by_sigma_rel'];xs=sorted(float(k[2:]) for k in vals);axes.flat[6+j].plot(xs,[vals['>='+str(x)] if '>='+str(x) in vals else next(v for k,v in vals.items() if float(k[2:])==x) for x in xs],color=S.C['blue'],lw=1.1,alpha=.45)
  overlaps={seed:read(RESULTS/'persistence'/f'mnist_mlp_seed{seed}_overlaps.json') for seed in range(3)}
  for seed,d in overlaps.items():
   epochs=sorted(map(int,d['per_stage'][b]));pairs=[p for p in d['pairs'] if p['block']==b and epochs.index(p['epochs'][1])==epochs.index(p['epochs'][0])+1]
   for cat,col in [('K','blue'),('Eplus_0.001','orange'),('Eplus_0.0001','aqua'),('Eplus_1e-06','purple')]:
    qq=[p for p in pairs if p.get('sampled_'+cat,{}).get('dim',0)>0];axes.flat[9+j].plot([p['epochs'][0] for p in qq],[np.median(p['sampled_'+cat]['exposure_later']) for p in qq],color=S.C[col],lw=1.1,alpha=.6,marker='o',ms=3)
  for row in range(4):
   ax=axes.flat[row*3+j];ax.set_box_aspect(1);ax.set_title(['Dimensions','Destination','Lost spectrum','Sensitivity'][row]+'; '+('out' if b=='readout' else 'L'+b),fontsize=6.4,loc='left');ax.set_xlabel('Checkpoint epoch' if row==0 else ('Relative singular value' if row==2 else 'Earlier epoch'),fontsize=6)
   if row in [1,2]:ax.set_ylim(-.02,1.02)
   if row==2:ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(numticks=3))
   if row==3:ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(numticks=4))
   if row==0:ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True)
 for row,lab in enumerate(['Dimension','Retained mass','Lost mass','Later sensitivity']):axes.flat[row*3].set_ylabel(lab,fontsize=6)
 shared_legend(fig,[Line2D([],[],color=S.C[c]) for _,l,c in families],[l for _,l,c in families],columns=4);save(fig,'persistence_diagnostics')
 tex=texfig('persistence_diagnostics',r'Further ReLU persistence diagnostics, all measured blocks, pairs and three seeds. Panels 1--3: dimensions of the full kernel and its component spaces. Panels 4--6: the earlier kernel with dead coordinates removed, split into mass retained in later non-dead directions (green) and later dead coordinates (orange); circles denote consecutive pairs and squares comparisons ending at epoch 500. Panels 7--9: cumulative lost mass carried by later singular directions above each relative singular-value threshold. Panels 10--12: median later sensitivity of 32 sampled directions from the earlier kernel (blue) and its positive tails at thresholds $10^{-3}$ (orange), $10^{-4}$ (green) and $10^{-6}$ (purple). Missing curves denote empty or unresolved spaces.','fig:persistence-diagnostics')
 lines=[r'\begin{table}[!htbp]\centering',r'\caption{GELU numerical dead-coordinate sensitivity at epoch 500, width 128. A unit is counted when its derivative magnitude is at most $\tau$ on every analysis image. Coordinate dimensions are seed ranges. The last columns count seeds whose dead-coordinate and co-occurrence projector splits resolve, out of three. The co-occurrence split remains unresolved at the endpoint.}\label{tab:persistence-gelu-tau}',r'\small\begin{tabular}{llrrr}\toprule',r'Layer & $\tau$ & Coordinate dimension & Dead split & Deficit split\\\midrule']
 for tau in ['1e-09','1e-06','0.001']:
  for b in ['2','3']:
   rr=[read(RESULTS/'persistence'/f'mnist_mlp_gelu_seed{seed}_geometry_tau{tau}.json')['blocks'][b]['stages']['500'] for seed in range(3)];vals=[r['dim_C_coordinates'] for r in rr];t={'1e-09':r'10^{-9}','1e-06':r'10^{-6}','0.001':r'10^{-3}'}[tau]
   lines.append(f"{b} & ${t}$ & {min(vals)}--{max(vals)} & {sum(r['split_resolved']['S'] for r in rr)}/3 & {sum(r['split_resolved']['D'] for r in rr)}/3"+r'\\')
 lines.append(r'\bottomrule\end{tabular}\end{table}')
 finish_fragment('section_d_persistence_diagnostics.tex',[tex,'\n'.join(lines)])


