"""Figures requested in the October 7 comment pass, from stored measurements only."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
from . import style as S
from . import main_polish as M
from . import appendix_comments as A
from . import section_c as C

RESULTS=M.RESULTS

def read(p):return json.loads(p.read_text())

def transition():
    stages=S.STAGE_ORDER
    fig,axes=plt.subplots(1,3,figsize=(3.35,1.35),sharey=True)
    fig.subplots_adjust(left=.13,right=.99,top=.86,bottom=.37,wspace=.18)
    for ax,(tag,title,_) in zip(axes,A.ACTS):
        data=[dict(C.resolved(read(RESULTS/'transition_v3'/f'mnist_mlp{tag}_seed{sd}_relative_all.json')['results'])) for sd in range(3)]
        curves={}
        for key,color,ls in [('true','ink','-'),('frozen','blue','--'),('transition','orange','-')]:
            values=[]
            for st in stages:
                vals=[]
                for d in data:
                    q=next((q for r,q in d.get(st,{}).get('probes',{}).items() if float(r)==.01),[])
                    if q:vals.append(np.median([v[key] for v in q]))
                values.append(vals)
            mean=np.array([np.mean(v) if v else np.nan for v in values]); sd=np.array([np.std(v,ddof=1) if len(v)>1 else np.nan for v in values]); lo=mean-sd; hi=mean+sd
            curves[key]=mean
            ax.fill_between(range(len(stages)),lo,hi,color=S.C[color],alpha=.12,lw=0)
            ax.plot(range(len(stages)),mean,color=S.C[color],ls=ls,lw=1.15,marker='o' if key=='transition' else None,ms=2.5)
        print(title,'endpoint transition / frozen:',curves['transition'][-1]/curves['frozen'][-1])
        ax.set(yscale='log',xticks=range(7),xticklabels=['Init','Fit','HA','Int','100','250','500'])
        ax.set_title(title.replace('Leaky ReLU','Leaky'),fontsize=7,pad=3)
        ax.tick_params(labelsize=5.5,pad=1);ax.tick_params(axis='x',labelrotation=55)
        ax.yaxis.set_major_locator(LogLocator(numticks=4));ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('RMS response',fontsize=7)
    fig.legend([Line2D([],[],color=S.C[c],ls=ls) for c,ls in [('ink','-'),('blue','--'),('orange','-')]],['True','Frozen','Transition'],loc='lower center',ncol=3,fontsize=6.5,columnspacing=1,handlelength=1.5,bbox_to_anchor=(.5,-.005))
    M.save(fig,'transition')



def hero_shared_axes(output_name='hero', third='symmetry'):
    """Selected centered ocean 03 and controls on one common physical window."""
    from paper_figures import xray as X
    assert third in ('symmetry', 'ggn')
    panels = [('shared_generic', 'Generic'), ('ocean_03_centered', 'Flat ocean'),
              ('shared_' + third, 'Symmetry' if third == 'symmetry' else 'GGN kernel')]
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.99, top=.85, bottom=.32, wspace=.17)
    reference = X.load('hero_fresh24/ocean_03_centered')
    xr = reference['x_axis'][[0, -1]]
    yr = reference['y_axis'][[0, -1]]
    for ax, (name, title) in zip(axes, panels):
        z = X.load('hero_fresh24/' + name)
        x, y = z['x_axis'], z['y_axis']
        assert np.allclose(x[[0, -1]], xr) and np.allclose(y[[0, -1]], yr)
        assert np.isclose(z['theta_norm'], reference['theta_norm'])
        im = ax.pcolormesh(x, y, np.maximum(z['cart_true'], X.FLOOR).T,
                          norm=plt.matplotlib.colors.LogNorm(X.VMIN, X.VMAX),
                          cmap=X.PALETTES['A'], shading='nearest', rasterized=True)
        ax.set(xlim=xr, ylim=yr, xticks=[-2, 0, 2], yticks=[-2, 0, 2])
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.plot(0, 0, 'o', color='white', mec='#333', mew=.4, ms=2.2, zorder=5)
        for spine in ax.spines.values():
            spine.set_color('#b9b8b4'); spine.set_linewidth(.4)
    if third == 'symmetry':
        ni, no = float(z['norm_in']), float(z['norm_out'])
        c = np.geomspace(.01, 10, 1800)
        xx, yy = (c-1)*ni, (1/c-1)*no
        axes[2].plot(xx, yy, color='#111', lw=1.0)
        axes[2].plot(xr, -xr*no/ni, color='#111', lw=.75, ls='--')
    axes[2].set(xlim=xr, ylim=yr)
    cax = fig.add_axes([.11, .19, .80, .045])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[1e-12, 1e-8, 1e-4, 1])
    cb.set_label(r'Output change $\|\Delta F_X\|_X$', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2, pad=1); cb.outline.set_visible(False)
    M.save(fig, output_name)
    return fig







def atomic():
    colors=['#ead9a0','#a8cbe8','#4b93d1','#0d4a8f']
    rows=[('Width',[('w32','32'),('w64','64'),('','128'),('w256','256')]),('Depth',[('d2','2'),('d4','4'),('d5','5')]),('Activation',[('','ReLU'),('leaky','Leaky'),('identity','Identity')])]
    fig,ax=plt.subplots(figsize=(3.35,2.35));fig.subplots_adjust(left=.09,right=.995,top=.97,bottom=.13)
    records=[]
    for ri,(family,cases) in enumerate(rows):
        y=-ri*1.9;x=0
        for tag,name in cases:
            data=read(RESULTS/'atomic'/f'mnist_mlp{("_"+tag) if tag else ""}_seed0.json')['results']['late_endpoint']['layers'];data=data if isinstance(data,list) else data.values()
            x0=x
            for q in data:
                layer=str(q['layer'])
                if layer in ['1','readout']:continue
                c=q['components@1e-09'];parts=[c[k] for k in ['feature_null','downstream_null','deficit']]
                assert min(parts)>=0 and sum(parts)==c['dim_K'] and sum(parts)<=q['p']
                vals=[q['p']-sum(parts)]+parts;records.append(dict(model=tag or 'w128',layer=layer,p=q['p'],components=vals))
                ax.pie(vals,colors=colors,center=(x,y),radius=.42,startangle=90,counterclock=False,frame=True,wedgeprops={'edgecolor':'white','linewidth':.4})
                ax.text(x,y-.52,'L'+layer,fontsize=5.5,ha='center',va='top');ax.text(x,y-.79,f"{q['p']/1000:.1f}k",fontsize=5.3,ha='center',va='top',color=S.C['ink2']);x+=1
            ax.text((x0+x-1)/2,y+.53,name,fontsize=6.5,ha='center',va='bottom');x+=.4
        ax.text(-.75,y,family,fontsize=6.5,rotation=90,va='center',ha='center')
    ax.set(xlim=(-1.1,9.25),ylim=(-4.85,.88));ax.axis('off');ax.set_aspect('equal')
    fig.legend([Rectangle((0,0),1,1,color=c) for c in colors],['Exposed','Feature-null','Downstream-null','Co-occurrence'],loc='lower center',ncol=2,fontsize=6,handlelength=1.1,columnspacing=.8,bbox_to_anchor=(.52,-.01))
    M.save(fig,'atomic');(M.OUT/'atomic_source_check.json').write_text(json.dumps(records,indent=2)+'\n')

def clustering():
    fig,axes=plt.subplots(1,2,figsize=(3.35,1.5));fig.subplots_adjust(left=.12,right=.99,bottom=.31,top=.86,wspace=.36)
    detail,da=plt.subplots(3,4,figsize=(7,4.8));detail.subplots_adjust(left=.075,right=.99,bottom=.14,top=.94,wspace=.42,hspace=.62)
    records=[]
    metrics=[('n_groups','Groups'),('purity','Label purity'),('ARI_labels','Adjusted Rand index'),('singleton_sample_fraction','Singletons (%)')]
    for ri,(tag,label,col) in enumerate(A.ACTS):
        grids=[read(RESULTS/'clusters'/f'mnist_mlp{tag}_seed{sd}_dmap_eps.json')['stages']['late_endpoint']['layer3']['eps_grid'] for sd in range(3)]
        x=np.array([q['epsilon'] for q in grids[0]])
        assert all(np.array_equal(x,[q['epsilon'] for q in grid]) for grid in grids)
        for j,(key,title) in enumerate(metrics):
            values=np.array([[q[key] for q in grid] for grid in grids]);mu=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
            if j<2:
                ax=axes[j];ax.fill_between(x,mu-sd,mu+sd,color=col,alpha=.15,lw=0);ax.plot(x,mu,color=col,marker='o',ms=3,lw=1.1)
            for seed,(color,marker) in enumerate(zip(['blue','orange','aqua'],['o','s','^'])):
                da[ri,j].plot(x,values[seed]*(100 if j==3 else 1),color=S.C[color],marker=marker,ms=3,lw=1)
                if j==1:
                    control=[q['purity_matched_size_random_mean'] for q in grids[seed]]
                    da[ri,j].plot(x,control,color=S.C[color],ls=':',lw=.7)
            ax=da[ri,j];ax.set_title(title,fontsize=7,pad=3);ax.tick_params(labelsize=6,pad=1);ax.xaxis.set_major_locator(MaxNLocator(3))
            if j==0:
                ax.set(yscale='log');ax.yaxis.set_major_locator(LogLocator(numticks=3));ax.yaxis.set_minor_formatter(NullFormatter());ax.set_ylabel(label,fontsize=7)
            elif j==3:
                ax.set(ylim=(0,max(.05,112*values.max())));ax.yaxis.set_major_locator(MaxNLocator(3))
            else:ax.set(ylim=(-.04,1.04),yticks=[0,.5,1])
            if ri==2:ax.set_xlabel(r'Diameter $\varepsilon$',fontsize=7)
            records.append(dict(activation=label,metric=key,seeds=[0,1,2],epsilon=x.tolist(),values=values.tolist(),mean=mu.tolist(),sample_sd=sd.tolist()))
        control=np.array([[q['purity_matched_size_random_mean'] for q in grid] for grid in grids]).mean(axis=0)
        axes[1].plot(x,control,color=col,ls=':',lw=.9)
    axes[0].set(yscale='log',ylabel='Groups');axes[0].yaxis.set_major_locator(LogLocator(numticks=3));axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set(ylim=(0,1.06),yticks=[0,.5,1],ylabel='Label purity')
    for ax in axes:
        ax.set_xlabel(r'Diameter $\varepsilon$',fontsize=7);ax.yaxis.label.set_size(7);ax.tick_params(labelsize=6,pad=1);ax.xaxis.set_major_locator(MaxNLocator(3));ax.set_title('L3, epoch 500',fontsize=7,pad=3)
    fig.legend([Line2D([],[],color=c,marker='o',ms=3) for _,_,c in A.ACTS],['ReLU','Leaky','GELU'],loc='lower center',ncol=3,fontsize=6.5,handlelength=1.5,columnspacing=1,bbox_to_anchor=(.5,-.01))
    detail.legend([Line2D([],[],color=S.C[c],marker=m,ms=4) for c,m in zip(['blue','orange','aqua'],['o','s','^'])],['Seed 0','Seed 1','Seed 2'],loc='lower center',ncol=3,fontsize=7)
    M.save(fig,'clustering');M.save(detail,'clustering_seeds');(M.OUT/'clustering_source_check.json').write_text(json.dumps(records,indent=2)+'\n')
    (C.PAPER/'section_d_cluster_seed_support.tex').write_text(r"""\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth,height=.84\textheight,keepaspectratio]{early_draft/main_compact/clustering_seeds.pdf}
\caption{Diameter-controlled downstream clustering across seeds 0--2 in width-128 MNIST MLPs, L3 at epoch 500. Rows compare activations; columns show group count, label purity, adjusted Rand index and the percentage of samples in singleton groups. Colours and markers identify seeds; dotted purity curves give matched-size shuffled-label means. All seeds use the same analysis size, distance normalization and diameter grid.}\label{fig:cluster-seeds}
\end{figure}
\FloatBarrier
""")


def nc_pairings():
    fig,axes=plt.subplots(1,4,figsize=(7,2.9));fig.subplots_adjust(left=.065,right=.995,bottom=.46,top=.90,wspace=.55)
    correlation=[]
    for ai,(tag,label,col) in enumerate(A.ACTS):
        data=read(RESULTS/'clusters'/f'mnist_mlp{tag}_seed0_dmap_eps.json')['stages'];nc=read(RESULTS/'neural_collapse'/f'mnist_mlp{tag}_seed0.json')['results']
        for bi,b in enumerate([3,2]):
            x=[];ys=[[],[]]
            for stage,mk in [('high_accuracy','o'),('epoch_250','s'),('late_endpoint','^')]:
                q=data[stage][f'layer{b}'];dd=q['distance_distributions_within_between'];strength=1/nc[stage]['layers'][str(b-1)]['NC1'];x.append(np.log10(strength))
                values=[dd['within_class']['mean']/dd['between_class']['mean'],next(v['purity'] for v in q['eps_grid'] if abs(v['epsilon']-1)<1e-8)]
                for j,y in enumerate(values):
                    axes[2*bi+j].scatter(strength,y,s=54,color=col,marker=mk,edgecolor='white',linewidth=.5,zorder=3);ys[j].append(y)
            for j,y in enumerate(ys):
                r=float(np.corrcoef(x,y)[0,1]) if np.ptp(y)>1e-12 and np.ptp(x)>1e-12 else None
                correlation.append(dict(activation=label,layer=b,metric=['distance_ratio','purity'][j],n=3,pearson_log_inverse_nc1=r))
                ax=axes[2*bi+j];txt=f'{label}: r = {r:+.2f}' if r is not None else f'{label}: r undefined'
                ax.text(0,-.47-ai*.15,txt,transform=ax.transAxes,color=col,fontsize=6.5,ha='left',va='top')
    for bi,b in enumerate([3,2]):
        for j in [0,1]:
            ax=axes[2*bi+j];ax.set(xscale='log',xlabel='Collapse strength (1/NC1)',ylabel='Within / between' if j==0 else 'Label purity');ax.set_title(f'L{b}; '+('distance ratio' if j==0 else 'purity'),fontsize=7,loc='left');ax.tick_params(labelsize=6.5);ax.xaxis.label.set_size(6.5);ax.yaxis.label.set_size(7);ax.xaxis.set_major_locator(LogLocator(numticks=3));ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_major_locator(MaxNLocator(4));ax.margins(y=.10)
            lo,hi=ax.get_xlim();ticks=np.geomspace(lo*1.02,hi/1.02,3);ax.set_xticks(ticks,[f'{v:.2g}' for v in ticks])
    fig.legend([Line2D([],[],color=S.C['ink'],marker=mk,ls='',ms=5) for mk in ['o','s','^']],['High accuracy','Epoch 250','Epoch 500'],loc='lower center',ncol=3,fontsize=7,bbox_to_anchor=(.5,-.005))
    M.save(fig,'nc_pairings');(M.OUT/'nc_correlations.json').write_text(json.dumps(correlation,indent=2)+'\n')
    caption=r'''Feature concentration versus downstream clustering in width-128 MNIST MLPs, seed 0. Larger $1/\mathrm{NC1}$ means stronger input-feature collapse; the horizontal scale is logarithmic. Points compare within-class/between-class downstream-map distance or diameter-cluster label purity at $\varepsilon=1$, at three matched checkpoints. Left pair: L3 maps and L2 features; right pair: L2 maps and L1 features. Colors identify activations and marker shapes identify stages. Each panel reports the within-activation Pearson correlation with $\log_{10}(1/\mathrm{NC1})$, using only these three points; a constant response has undefined correlation. These descriptive correlations do not establish a causal relationship.'''
    (C.PAPER/'section_d_nc_support.tex').write_text(r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\linewidth]{early_draft/main_compact/nc_pairings.pdf}'+'\n'+r'\caption{'+caption+r'}\label{fig:nc-pairings}'+'\n'+r'\end{figure}'+'\n'+r'\FloatBarrier'+'\n')

