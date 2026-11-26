"""Render appendix revisions from saved scalar results and slice arrays only."""
import csv
from pathlib import Path
from . import section_c as C
from . import appendix_comments as A
from .section_c import np,plt,S,RESULTS,OUT,PAPER,read,save,dense_grid,shared_legend,texfig,finish_fragment,Line2D


def records(name):
 return list(csv.DictReader((OUT/(name+'.csv')).open()))


def stats(values):
 v=np.asarray(values,float);return float(np.mean(v)),float(np.std(v,ddof=1)) if len(v)>1 else 0.






def cluster_stages():
 rec=A.partition_records();fig,axes=dense_grid(2,2,row_height=1.75,legend_height=.4);fig.set_size_inches(3.8,2.3);fig.subplots_adjust(left=.13,right=.99,wspace=.45)
 for row,ax in enumerate(axes.flat):
  for b,col,ls in [('2','blue','--'),('3','orange','-')]:
   for seed in range(3):
    qq=[rec['',seed][st]['layers'][b]['10']['labels' if row==0 else 'kmeans_D'] for st in A.STAGES];yy=[A.valid_share(q) if row==0 else q['purity'] for q in qq]
    ax.plot(range(5),yy,color=S.C[col],ls=ls,marker=['o','s','^'][seed],ms=3.5,lw=1.4,alpha=.8)
    if row==0:
     for x,y,q in zip(range(5),yy,qq):
      if not q['dim_P_resolved']:ax.plot(x,y,marker=['o','s','^'][seed],mfc='white',mec=S.C[col],ms=3.5)
    elif seed==0:ax.plot(range(5),[q['purity_random_baseline'] for q in qq],color=S.C[col],ls=':',lw=1)
  ax.set_box_aspect(1);ax.set_ylim(-.03,1.04);C.stage_ticks(ax,A.STAGES);ax.set_title(['Class-explained deficit','Downstream-group purity'][row],loc='left',fontsize=7)
 shared_legend(fig,[Line2D([],[],color=S.C['blue'],ls='--'),Line2D([],[],color=S.C['orange'])],['L2','L3'],columns=2);save(fig,'cluster_relu_stages')
 finish_fragment('section_d_cluster_stages.tex',[texfig('cluster_relu_stages',r'Class organization in the width-128 ReLU MLP across five fitted stages. Left: share of the deficit explained by class groups. Right: label purity of ten requested downstream k-means groups; dotted lines give matched random-label purity. Markers distinguish seeds 0--2. Open markers flag tolerance-sensitive partition ranks. HA and Int denote high accuracy and interpolation.','fig:cluster-stages').replace(r'width=\linewidth',r'width=.58\linewidth')])


def nc():
 acts=A.ACTS+[('_identity','Identity',S.C['aqua'])];stages=['initialization']+A.STAGES
 fig,axes=dense_grid(4,4,row_height=1.8,legend_height=.4);fig.subplots_adjust(wspace=.48)
 metrics=[('NC1','NC1','log'),('NC2_norm_cv','Mean-norm variation','linear'),('NC2_gram_etf','Class-mean geometry','linear'),('NC4_vs_label','NCC disagreement','log')]
 for ax,(metric,title,scale) in zip(axes.flat,metrics):
  for tag,lab,col in acts:
   data=[read(RESULTS/'neural_collapse'/f'mnist_mlp{tag}_seed{seed}.json')['results'] for seed in range(3)]
   for b,ls in [('3','-'),('2','--')]:
    vv=[[d[st]['layers'][b][metric] for d in data if st in d] for st in stages];mm=np.array([stats(v)[0] if v else np.nan for v in vv]);sd=np.array([stats(v)[1] if v else np.nan for v in vv]);floor=1e-5 if scale=='log' else 0
    ax.plot(range(6),np.maximum(mm,floor),color=col,ls=ls,lw=1.4,marker='o',ms=2.8);ax.fill_between(range(6),np.maximum(mm-sd,floor),mm+sd,color=col,alpha=.12)
  ax.set_yscale(scale);ax.set_title(title,fontsize=7,loc='left');C.stage_ticks(ax,stages);ax.set_xticklabels(['Init','HA','Int','100','250','500']);ax.tick_params(axis='x',labelrotation=45);ax.set_box_aspect(1)
 shared_legend(fig,[Line2D([],[],color=c) for _,_,c in acts],[lab for _,lab,_ in acts],columns=4);save(fig,'nc_full_sd')
 fig,axes=dense_grid(4,4,row_height=1.8,legend_height=.4);fig.subplots_adjust(wspace=.5)
 for tag,lab,col in A.ACTS:
  eps=read(RESULTS/'clusters'/f'mnist_mlp{tag}_seed0_dmap_eps.json')['stages'];ncdata=read(RESULTS/'neural_collapse'/f'mnist_mlp{tag}_seed0.json')['results']
  for bi,b in enumerate([3,2]):
   for st,mk in [('high_accuracy','o'),('epoch_250','s'),('late_endpoint','^')]:
    q=eps[st][f'layer{b}'];dd=q['distance_distributions_within_between'];x=ncdata[st]['layers'][str(b-1)]['NC1'];pur=next(v['purity'] for v in q['eps_grid'] if abs(v['epsilon']-1)<1e-8)
    for j,y in enumerate([dd['within_class']['mean']/dd['between_class']['mean'],pur]):axes.flat[2*bi+j].scatter(x,y,s=24,color=col,marker=mk,edgecolor='white',linewidth=.3,zorder=3)
 for bi,b in enumerate([3,2]):
  for j in [0,1]:
   ax=axes.flat[2*bi+j];ax.set_title(f'L{b}; '+('distance ratio' if j==0 else 'purity'),fontsize=7,loc='left');ax.set_xlabel('Input-feature NC1',fontsize=7);ax.set_ylabel('Within / between' if j==0 else 'Label purity',fontsize=7);ax.set_box_aspect(1);ax.margins(y=.05);ax.autoscale(enable=True,axis='y');ax.yaxis.set_major_locator(C.MaxNLocator(4));ax.xaxis.set_major_locator(C.MaxNLocator(3))
 shared_legend(fig,[Line2D([],[],color=col,marker='o',ls='') for _,_,col in A.ACTS],[lab for _,lab,_ in A.ACTS],columns=3);save(fig,'nc_pairings_scatter')
 finish_fragment('section_d_nc_support.tex',[texfig('nc_pairings_scatter',r'Feature concentration versus downstream clustering, seed 0 of the width-128 MNIST activation comparison. Each point pairs NC1 of the features entering the indicated layer with either its mean within-class/between-class downstream-map distance ratio or its diameter-cluster label purity at $\varepsilon=1$. Circles, squares and triangles denote high accuracy, epoch 250 and epoch 500. The left pair uses L3 maps and L2 features; the right pair uses L2 maps and L1 features. Colours identify activations. Each vertical axis fits its own data range with a small margin. Lower distance ratios indicate stronger class separation; high purity alone can also result from many small groups. These are correlational measurements.','fig:nc-pairings')])


def data_dimensions():
 rows=records('atomic');branches=records('construction_attempts');fig,axes=dense_grid(4,4,row_height=1.7,legend_height=.46);fig.subplots_adjust(wspace=.48)
 for j,b in enumerate(['2','3','readout']):
  ax=axes.flat[j]
  for prefix,col,dx in [('p1_','blue',-.18),('','orange',.18)]:
   vv=[[float(r['kernel']) for r in rows if r['model']==f'mnist_mlp_{prefix}sub{n}' and r['block']==b] for n in [1000,2000,5000]];mm,sd=np.array([stats(v) for v in vv]).T
   ax.bar(np.arange(3)+dx,mm,width=.34,color=S.C[col],yerr=sd,error_kw=dict(lw=.8,capsize=2))
  ax.set_xticks(range(3),['1','2','5']);ax.set_title(C.layer_name(b),fontsize=7,loc='left');ax.set_xlabel('Subset size (thousands)',fontsize=6.5);ax.set_box_aspect(1);ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True)
 ax=axes.flat[3]
 for prefix,col in [('p1_','blue'),('','orange')]:
  for order,ls,mk in [('bottom_up','-','o'),('top_down','--','s'),('naive_atomic_sum',':','x')]:
   for seed in range(3):
    vv=[next((float(r['dimension']) for r in branches if r['model']==f'mnist_mlp_{prefix}sub{n}' and r['order']==order and int(r['seed'])==seed),np.nan) for n in [1000,2000,5000]]
    ax.plot(range(3),vv,color=S.C[col],ls=ls,marker=mk,lw=1.3,ms=3.5,alpha=.8,clip_on=False)
 ax.set_title('Compatible invariant subspaces',fontsize=7,loc='left');ax.set_box_aspect(1);ax.set_xticks(range(3),['1','2','5']);ax.set_xlabel('Subset size (thousands)',fontsize=6.5);ax.set_ylim(bottom=0);ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True)
 axes.flat[0].set_ylabel('Dimension',fontsize=7)
 shared_legend(fig,[Line2D([],[],color=S.C['blue']),Line2D([],[],color=S.C['orange'])],['Train on subset','Train on all data; evaluate on subset'],columns=2);save(fig,'data_dimensions_four')
 finish_fragment('section_d_data_atomic.tex',[texfig('data_dimensions_four',r'Dimensions at epoch 500 versus subset size, width-128 ReLU MLPs. Colours distinguish subset-trained and full-data networks throughout. First three panels: total atomic dimensions, with seed means and sample-standard-deviation whiskers. Fourth: compatible branch dimensions, with one curve per seed; solid/circles denote bottom-up construction, dashed/squares top-down, and dotted/crosses the atomic sum. Zero means no validated direction, including a failed certificate. Unmeasured constructions have no point. L1 is fixed.','fig:data-atomic').replace(r'\label{fig:data-atomic}',r'\label{fig:data-atomic}\label{fig:data-branches}')])
 (PAPER/'section_d_data_branches.tex').write_text('% Integrated into section_d_data_atomic.tex.\n')




def ray_grid():
 raw=records('branch_rays');rows=[r for r in raw if r['mode']=='absolute']
 def draw(ax,rr,models,cols):
  for m,col in zip(models,cols):
   qq=[r for r in rr if r['model']==m];xs=sorted({float(r['radius']) for r in qq})
   if not xs:continue
   for metric,ls in [('true','-'),('frozen','--')]:
    vv=[[float(r[metric]) for r in qq if float(r['radius'])==x] for x in xs];mm=np.array([np.mean(v) for v in vv]);ax.plot(xs,np.maximum(mm,1e-16),color=col,ls=ls,lw=1.4,marker='o' if metric=='true' else None,ms=2.8)
    if metric=='true':ax.fill_between(xs,[max(min(v),1e-16) for v in vv],[max(max(v),1e-16) for v in vv],color=col,alpha=.12)
   ax.axvline(np.median([float(r['ten_percent']) for r in qq]),color=col,ls=':',lw=.9)
  ax.axhline(1e-3,color=S.C['ink3'],ls=':',lw=.8);ax.set(xscale='log',yscale='log',ylim=(1e-16,None));ax.xaxis.set_major_locator(C.LogLocator(numticks=3));ax.yaxis.set_major_locator(C.LogLocator(numticks=4));ax.set_box_aspect(1)
 # Merge architecture and activation comparisons; keep one leaky width.
 models=['mnist_mlp_w64','mnist_mlp_leaky_w64','mnist_cnn','mnist_mlp_bn_w64','svhn_mlp_w64','svhn_mlp_w64_v2']
 cols=[S.C[k] for k in ['blue','orange','aqua','purple','yellow','ink3']]
 groups={'Branch':['branch:bottom_up','constructed'],'Top-down':['branch:top_down'],'Feature-null':['branch:feature_null'],'Selective':['branch'],'Atomic sum':['naive_atomic_sum','atomic_sum'],'Generic':['generic'],'Tangent':['tangent_flat']}
 # CNN constructed branches join the MLP bottom-up panels; selective candidates retain their own column.
 cases=[(st,lab) for st in ['high_accuracy','mse_interpolation','epoch_250','late_endpoint','best_val'] for lab,names in groups.items() if any(r['stage']==st and r['direction_set'] in names and r['model'] in models for r in rows)]
 tex=[]
 for start in range(0,len(cases),12):
  cc=cases[start:start+12];fig,axes=dense_grid(len(cc),4,row_height=1.55,legend_height=.55);fig.subplots_adjust(wspace=.4)
  for ax,(st,lab) in zip(axes.flat,cc):
   rr=[r for r in rows if r['stage']==st and r['direction_set'] in groups[lab] and r['model'] in models];draw(ax,rr,models,cols);ax.set_title(lab+'; '+C.SHORT.get(st,st),fontsize=7,loc='left')
  shared_legend(fig,[Line2D([],[],color=c) for c in cols],['ReLU w64','Leaky w64','CNN','BatchNorm w64','SVHN baseline','SVHN selected'],columns=3);fig.supylabel('RMS output change',fontsize=7,x=.002);fig.supxlabel('Absolute radius',fontsize=7,y=0);name=f'rays_joint_{start//12}';save(fig,name)
  tex.append(texfig(name,r'Absolute-radius responses across architectures and activations. Each panel fixes the training stage and direction family; only width 64 is shown for leaky ReLU. The Branch panels combine MLP bottom-up constructions with the CNN received-space construction; Selective panels retain the separate CNN selective candidates. Solid and dashed curves give actual and frozen responses, averaged over seed medians, with seed ranges shaded. Dotted vertical lines mark one tenth of the network norm; the horizontal dotted line marks output change $10^{-3}$. Missing curves indicate unmeasured model--stage combinations.'))
 finish_fragment('section_c_ray_figures.tex',tex)
 fig,axes=dense_grid(4,4,row_height=1.65,legend_height=.42);fig.subplots_adjust(wspace=.43)
 for i,(prefix,ds) in enumerate([('p1_','branch:bottom_up'),('p1_','generic'),('','branch:bottom_up'),('','generic')]):
  ax=axes.flat[i];models=[f'mnist_mlp_{prefix}sub{n}' for n in [1000,2000,5000]];rr=[r for r in rows if r['stage']=='late_endpoint' and r['direction_set']==ds];draw(ax,rr,models,[S.C[k] for k in ['blue','orange','aqua']]);ax.set_title(('Subset-trained' if prefix else 'Full-data')+'\n'+('Branch' if 'branch' in ds else 'Generic'),fontsize=7,loc='left')
 shared_legend(fig,[Line2D([],[],color=S.C[k]) for k in ['blue','orange','aqua']],['1,000 images','2,000 images','5,000 images'],columns=3);fig.supylabel('RMS output change',fontsize=7,x=.001);fig.supxlabel('Absolute radius',fontsize=7,y=0);save(fig,'data_rays_four')
 finish_fragment('section_d_data_rays.tex',[texfig('data_rays_four',r'Absolute-radius branch and generic probes at epoch 500, for subset-trained networks (left pair) and full-data networks evaluated on the same subsets (right pair). Colours denote subset size. Solid/dashed lines show actual/frozen response; means and ranges summarize three seed medians. L1 is fixed. Vertical dotted lines mark one tenth of the network norm; the horizontal line marks output change $10^{-3}$.')])


def p1_summary():
 import runpy
 runpy.run_path(str(RESULTS.parent/'paper_figures/p1_figs.py'),run_name='__main__')


def xray_catalogue():
 from matplotlib.colors import LogNorm,LinearSegmentedColormap
 import matplotlib.patheffects as pe
 setups=[
 ('gal_feature_null','ReLU w64 e305','Feature-null'),('gal_zero_feature_out','ReLU w64 e305','Zero-feature readout'),
 ('gal_downstream_null','ReLU w64 e305','Downstream-null'),('gal_selective','ReLU w64 e305','Selective unit'),
 ('gal_dead_unit','ReLU w64 e305','Dead-unit row'),('gal_branch','ReLU w64 e305','Multilayer branch'),
 ('gal_tangent_flat','ReLU w64 e305','Tangent-flat'),('gal_random','ReLU w64 e305','Random'),
 ('w64_branch_branch','ReLU w64 e305','Branch / branch'),('w64_branch_generic','ReLU w64 e305','Branch / generic'),
 ('w64_branch_tangent','ReLU w64 e305','Branch / tangent'),('w64_featurenull_branch','ReLU w64 e305','Feature-null / branch'),
 ('w64_rescaling_unit5','ReLU w64 e305','Unit-5 rescaling'),('leaky_w64_branch_branch','Leaky w64 e138','Branch / branch'),
 ('cnn_branch','CNN e500','Branch / branch'),('cnn_branch_generic','CNN e500','Branch / generic'),
 ('cnn_featurenull','CNN e500','Feature-null'),('cnn_random','CNN e500','Random'),
 ('relu_w128_L3_kernel','ReLU w128 e40','L3 kernel'),('relu_w128_L3_tail','ReLU w128 e40','L3 tail'),
 ('leaky_w128_L3_kernel','Leaky w128 e40','L3 kernel'),('gelu_w128_L3_tail','GELU w128 e22','L3 tail'),
 ('sym_relu_w64_wide','ReLU w64 e305','Rescaling'),('sym_leaky_w64','Leaky w64 e138','Rescaling'),
 ('sym_relu_w128','ReLU w128 e40','Rescaling'),('sym_leaky_w128','Leaky w128 e40','Rescaling'),
 ('sym_gelu_w128','GELU w128 e22','Rescaling control')]
 polar=[('w64',305),('w128',40),('w256',20),('w512',11),('d2',286),('d4',26),('d5',33)]
 panels=[(name,model,family,view) for name,model,family in setups for view in ['Cartesian','Angle-radius']]
 panels += [('polar_'+name,'ReLU '+name+' e'+str(ep),'Last-hidden selective','Angle-radius') for name,ep in polar]
 available={p.stem for p in (RESULTS/'paper_slices').glob('*.npz') if not p.stem.startswith(('landscape_','selective_'))}
 assert {p[0] for p in panels}.issubset(available)
 cm=LinearSegmentedColormap.from_list('appendix_ocean',[(0,'#123f6e'),(.30,'#2472ad'),(.55,'#57add4'),(.70,'#a8e0ea'),(.75,'#eaf6f2'),(.80,'#f2df9a'),(.88,'#edc24a'),(.94,'#df8a3a'),(1,'#9c2f1a')])
 tex=[];manifest=[]
 for start in range(0,len(panels),16):
  cc=panels[start:start+16];fig,axes=dense_grid(len(cc),4,row_height=1.85,legend_height=.45);fig.subplots_adjust(left=.065,right=.99,top=.965,bottom=.09,wspace=.45,hspace=.7)
  for i,(ax,(name,model,family,view)) in enumerate(zip(axes.flat,cc)):
   with np.load(RESULTS/'paper_slices'/f'{name}.npz',allow_pickle=False) as z:
    if view=='Cartesian':x=y=z['axis'];v=z['cart_true'].T;flips=z['cart_flips'].T;ax.set_xlabel('Edit 1',fontsize=6);ax.set_ylabel('Edit 2',fontsize=6)
    else:x=np.log10(z['radii']);y=np.degrees(z['angles']);v=z['pol_true'];flips=z['pol_flips'];ax.set_xlabel(r'$\log_{10}$ radius',fontsize=6);ax.set_ylabel('Angle (deg)',fontsize=6);ax.set_yticks([0,180,360])
    im=ax.pcolormesh(x,y,np.maximum(v,1e-16),norm=LogNorm(1e-12,1),cmap=cm,shading='auto',rasterized=True)
    if 'gelu' not in name and flips.min()<.5<flips.max():
     contour=ax.contour(x,y,flips,levels=[.5],colors=['white'],linewidths=.8);contour.set_path_effects([pe.Stroke(linewidth=1.3,foreground='#333'),pe.Normal()])
    if v.min()<1e-3<v.max():ax.contour(x,y,v,levels=[1e-3],colors=['#222'],linestyles='--',linewidths=.7)
   ax.set_box_aspect(1);ax.set_title(model+'\n'+family,fontsize=6.6,loc='left',pad=3);ax.tick_params(labelsize=5.8,pad=1,length=2);ax.xaxis.set_major_locator(C.MaxNLocator(3))
   if view=='Cartesian':ax.yaxis.set_major_locator(C.MaxNLocator(3))
   manifest.append(dict(file=name+'.npz',view=view,page=start//16+1,panel=i+1,model=model,direction=family))
  cax=fig.add_axes([.23,.034,.54,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[1e-12,1e-9,1e-6,1e-3,1]);cb.ax.tick_params(labelsize=6,length=2);cb.set_label(r'Output change $\|\Delta F_X\|_X$',fontsize=7,labelpad=1)
  name=f'xray_catalogue_{start//16+1}';save(fig,name)
  caption=r'Ocean x-ray catalogue, page '+str(start//16+1)+r'. All panels use MNIST and seed 0; titles give width (w), hidden depth (d), and checkpoint epoch (e). Adjacent Cartesian and angle--radius panels describe the same two-dimensional parameter plane. The final seven angle--radius panels compare selective directions in width and depth sweeps; width sweeps have depth three and depth sweeps have width 128. White contours separate the sampled zero-gate-change region from changed gates, and black dashed contours mark output change $10^{-3}$. All panels share the colour scale; values below $10^{-12}$ use its lowest colour. See the text for the scope of these sampled views.'
  tex.append(texfig(name,caption,'fig:xray-catalogue-'+str(start//16+1),height='.78'))
  tex.append(r'\FloatBarrier')
 finish_fragment('section_f_xray_catalogue.tex',tex);C.write_csv('xray_catalogue_manifest',manifest)




