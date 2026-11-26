"""Appendix comment revisions, rendered exclusively from existing results/caches."""
from . import section_c as C
from .section_c import np, plt, S, RESULTS, OUT, read, save, dense_grid, shared_legend, texfig, finish_fragment, Line2D

ACTS=[('', 'ReLU', S.C['blue']),('_leaky','Leaky ReLU',S.C['orange']),('_gelu','GELU',S.C['purple'])]
STAGES=['high_accuracy','mse_interpolation','epoch_100','epoch_250','late_endpoint']


def axes_style(axes):
 for ax in axes.flat:
  ax.tick_params(labelsize=6.5)
  ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7)






def partition_records():
 records={}
 for tag,_,_ in ACTS:
  for seed in range(3):
   f=RESULTS/'clusters'/f'mnist_mlp{tag}_seed{seed}_partitions.json'
   if f.exists():records[tag,seed]=read(f)['results']
 return records


def valid_share(q):
 v=q.get('across_share_of_delta')
 return v if q.get('inequalities_hold') and v is not None and 0<=v<=1 else np.nan


def clustering():
 records=partition_records();fig,axes=dense_grid(6,3,row_height=1.65,legend_height=.5)
 for j,(tag,lab,_) in enumerate(ACTS):
  for b,col,ls in [('2',S.C['blue'],'--'),('3',S.C['orange'],'-')]:
   for seed in range(3):
    rec=records.get((tag,seed),{})
    for row in [0,1]:
     xx=[];yy=[];qq=[]
     for i,st in enumerate(STAGES):
      q=rec.get(st,{}).get('layers',{}).get(b,{}).get('10',{}).get('labels' if row==0 else 'kmeans_D')
      if q:xx.append(i);yy.append(valid_share(q) if row==0 else q['purity']);qq.append(q)
     axes[row,j].plot(xx,yy,color=col,ls=ls,lw=1.5,alpha=.75,marker=['o','s','^'][seed],ms=4)
     if row==0:
      for x,y,q in zip(xx,yy,qq):
       if np.isfinite(y) and not q.get('dim_P_resolved'):axes[row,j].plot(x,y,marker=['o','s','^'][seed],mfc='white',mec=col,ms=4)
     elif seed==0:
      axes[row,j].plot(xx,[q['purity_random_baseline'] for q in qq],color=col,ls=':',lw=1.1)
  for row in [0,1]:
   ax=axes[row,j];ax.set_title(lab+(' (3 seeds)' if not tag else ' (seed 0)'),fontsize=7,loc='left');C.stage_ticks(ax,STAGES);ax.set_ylim(-.03,1.05)
 axes[0,0].set_ylabel('Class-explained share');axes[1,0].set_ylabel('Downstream-group purity');axes_style(axes)
 shared_legend(fig,[Line2D([],[],color=S.C['blue'],ls='--'),Line2D([],[],color=S.C['orange']),Line2D([],[],color=S.C['ink3'],ls=':')],['L2','L3','Matched random purity'],columns=3);save(fig,'cluster_stages')
 caption=r'Class structure across fitted stages in three-hidden-layer, width-128 MNIST MLPs. Top: the fraction of the deficit explained by the class partition. Bottom: label purity of ten requested downstream k-means groups. ReLU has three seeds (circle, square, triangle); leaky ReLU and GELU have seed 0. Every recorded stage is shown; lines connect measured points. Open markers flag tolerance-sensitive partition ranks. Invalid GELU endpoint rank shares are omitted, while their purity remains measurable. HA and Int denote high accuracy and interpolation. Purity and explained deficit are distinct quantities.'
 tex=[texfig('cluster_stages',caption,'fig:cluster-stages')]
 fig,axes=dense_grid(4,4,row_height=1.6,legend_height=.4);fig.subplots_adjust(wspace=.45)
 for j,b in enumerate(['2','3']):
  ax=axes.flat[j];rec=records['',0]
  for st,alpha in [('high_accuracy',.5),('late_endpoint',1)]:
   for kind,col,ls in [('kmeans_D','blue','-'),('joint','purple','--'),('random_kmeans_D','ink3',':')]:
    layer=rec[st]['layers'][b];gs=[g for g in [5,10,20,40,80] if kind in layer.get(str(g),{})]
    ax.plot(gs,[valid_share(layer[str(g)][kind]) for g in gs],color=S.C[col],ls=ls,lw=1.5,alpha=alpha,marker='o',ms=4)
   ax.plot(10,valid_share(layer['10']['labels']),marker='*',color=S.C['orange'],ms=7,alpha=alpha)
   q=layer['10'].get('patterns_L3',{})
   if q and q['n_groups']<=100:ax.plot(q['n_groups'],valid_share(q),marker='D',color=S.C['ink'],ms=4,alpha=alpha)
  ax.set(xscale='log',xticks=[5,20,80],xticklabels=['5','20','80'],ylim=(-.04,1.04),xlabel='Requested groups');ax.set_title('Explained share; L'+b,fontsize=7,loc='left')
  ax=axes.flat[j+2];shuffle=read(RESULTS/'clusters/mnist_mlp_seed0_shuffle.json')['results']
  kinds=['global','labels','kmeans_D','kmeans_Dn','gates_suffix']
  for st,col in [('high_accuracy',S.C['blue']),('late_endpoint',S.C['orange'])]:
   ax.plot(range(5),[shuffle[st]['layers'][b][k]['delta_preserved_fraction'] for k in kinds],color=col,lw=1.5,marker='o',ms=4)
  ax.set_xticks(range(5),['All','Class','D','D norm','Gates'],rotation=35,ha='right');ax.set_ylim(-.04,1.04);ax.set_title('Shuffle survival; L'+b,fontsize=7,loc='left')
 axes.flat[0].set_ylabel('Explained fraction');axes.flat[2].set_ylabel('Deficit retained');axes_style(axes)
 shared_legend(fig,[Line2D([],[],color=S.C[c],ls=ls) for c,ls in [('blue','-'),('purple','--'),('ink3',':')]],['Downstream groups','Joint groups','Matched random groups'],columns=3);save(fig,'cluster_pairings')
 tex.append(texfig('cluster_pairings',r'Grouping and pairing tests in the width-128 ReLU MLP, seed 0. Left pair: deficit explained as the requested group count changes; light and dark curves denote high accuracy and epoch 500. Stars mark class groups and diamonds suffix-gate patterns, at their actual group count. Right pair: deficit remaining after global shuffles (All) or shuffles within classes, downstream groups (D), normalized downstream groups (D norm), or suffix-gate groups; blue is high accuracy and orange epoch 500. The L2 deficit survives almost entirely; L3 is more sensitive to global shuffling than to shuffling within groups.','fig:mechanism'))
 finish_fragment('section_d_clusters.tex',tex)
 # Both layers, every measured stage and the full diameter grid.
 fig,axes=dense_grid(12,4,row_height=1.55,legend_height=.5)
 metrics=[('n_groups','Groups'),('purity','Purity'),('ARI_labels','ARI'),('singleton_sample_fraction','Singleton fraction')]
 for row,(tag,lab,col) in enumerate(ACTS):
  d=read(RESULTS/'clusters'/f'mnist_mlp{tag}_seed0_dmap_eps.json')['stages']
  for j,(metric,title) in enumerate(metrics):
   ax=axes[row,j]
   for bi,b in enumerate([2,3]):
    for st,alpha in [('high_accuracy',.4),('epoch_250',.7),('late_endpoint',1)]:
     g=d[st][f'layer{b}']['eps_grid'];ax.plot([q['epsilon'] for q in g],[q[metric] for q in g],color=col,ls='--' if b==2 else '-',alpha=alpha,lw=1.5,marker='o' if st=='late_endpoint' else None,ms=3)
     if metric=='purity' and st=='late_endpoint':ax.plot([q['epsilon'] for q in g],[q['purity_matched_size_random_mean'] for q in g],color=S.C['ink3'],ls=':',lw=1)
   ax.set_title(lab.replace('Leaky ReLU','Leaky')+'; '+title,fontsize=7,loc='left');ax.set_xlabel(r'Diameter threshold $\varepsilon$',fontsize=7)
   if j==0:ax.set_yscale('log')
   else:ax.set_ylim(-.03,1.04)
 shared_legend(fig,[Line2D([],[],color=S.C['ink'],ls='--'),Line2D([],[],color=S.C['ink']),Line2D([],[],color=S.C['ink'],alpha=.4),Line2D([],[],color=S.C['ink'],alpha=.7)],['L2','L3','High accuracy','Epoch 250'],columns=4);save(fig,'cluster_diameters')
 finish_fragment('section_d_cluster_diameters.tex',[texfig('cluster_diameters',r'Diameter-controlled clustering in the same width-128 networks, seed 0. Rows compare activations; columns show group count, label purity, adjusted Rand index and the fraction of samples in singleton groups. Solid and dashed curves denote L3 and L2. Opacity increases from high accuracy to epoch 250 and epoch 500; endpoint curves have markers. Dotted purity controls shuffle labels at fixed group sizes. L3 forms class-aligned groups for every activation, whereas L2 behaviour depends on activation. High purity carried by singletons is distinguished from grouping whole classes.','fig:cluster-diameters')])




def cnn_units():
 from matplotlib.colors import to_rgb
 z=np.load(RESULTS/'visualization/cnn_units_cache.npz');frac=z['frac5'];order=np.argsort(frac.mean(1));p=frac/np.maximum(frac.sum(1,keepdims=True),1e-12);ent=-(p*np.log(np.maximum(p,1e-12))).sum(1);sel=list(np.argsort(ent)[:3])+[int(np.argsort(ent)[-1])]
 z=np.load(RESULTS/'visualization/cnn_units_cache2.npz')
 # Right-hand image grid retains its original printed size; the heatmap gets more width.
 fig=plt.figure(figsize=(7.8,3));gs=fig.add_gridspec(1,2,width_ratios=[2.05,1.72],wspace=.18,left=.06,right=.99,top=.88,bottom=.1)
 ax=fig.add_subplot(gs[0]);im=ax.imshow(frac[order].T,aspect='auto',cmap='Blues',vmin=0,vmax=1);ax.set_yticks(range(10));ax.set_xticks([0,31,63],['1','32','64']);ax.set(xlabel='Channel (sorted by active fraction)',ylabel='Digit');ax.set_title('Class activity',loc='left');cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.02);cb.ax.tick_params(labelsize=7)
 gsr=gs[1].subgridspec(4,4,wspace=.08,hspace=.26)
 for k,ch in enumerate(sel):
  for j in range(4):
   ax=fig.add_subplot(gsr[j,k]);g=z['imgs'][k,j];g=(g-g.min())/max(np.ptp(g),1e-9);ax.imshow(1-g,cmap='gray',vmin=0,vmax=1);a=z['acts'][k,j]/max(z['acts'][k,j].max(),1e-12);rgba=np.zeros((32,32,4));rgba[...,:3]=to_rgb(S.C['orange']);rgba[...,3]=.8*a;ax.imshow(rgba);ax.set_xticks([]);ax.set_yticks([]);ax.set_title((f'ch {ch}' if j==0 else f'top {j+1}')+f' · {z["labs"][k,j]}',fontsize=5,pad=1)
 save(fig,'cnn_units_wide')


