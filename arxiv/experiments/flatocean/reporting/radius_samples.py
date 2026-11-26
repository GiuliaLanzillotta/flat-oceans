"""Radius plots from saved independent rays; no network evaluations."""
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
from scipy.stats import binom, beta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import style as S

OUT=Path(S.F)/'early_draft/main_compact'
PAPER=Path(S.HERE).parent


def resnet_selective_panel():
    """Reviewed layer4.2.conv2 battery, not the linear classifier's empty kernel."""
    import torch
    root=Path(S.HERE)/'results/resnet'
    source=root/'common_family_l42fc.json';arrays=root/'common_family_l42fc_arrays.pt'
    d=json.loads(source.read_text())
    a=torch.load(arrays,map_location='cpu',mmap=True,weights_only=False)
    layer='layer4.2.conv2';rec=d['layers'][layer]
    radii=np.array(d['RGRID'],float)
    assert np.allclose(a['RGRID'].numpy(),radii)
    mi=a['metric_axis'].index('actual_abs')
    # Independent review corrected the reported 10,139 to 10,171 total;
    # the persisted, actually sampled subspace has 10,170 columns.
    basis_dim=sum(v.shape[1] for k,v in a.items() if k.startswith(layer+'/SEL_basis_ch'))
    assert basis_dim==10170
    rows=[]
    for mode in ['SEL','generic']:
        record=rec['battery'][mode]
        assert record['M']==256 and [v['i'] for v in record['per_direction']]==list(range(256))
        raw=a[f'{layer}/{mode}_drift'][:,mi,:].numpy()
        assert raw.shape==(256,10)
        for j,r in enumerate(radii):
            agg=record['aggregates_per_radius']['actual_abs'][f'r{r:g}']
            assert np.isclose(np.median(raw[:,j]),agg['median'],rtol=1e-12,atol=1e-20)
            relative_count=int((raw[:,j]/d['logits_rms_X']>d['tol_rel']).sum())
            assert relative_count==record['exceedance_count_actual_rel_gt_tol_among_M'][f'r{r:g}']
        y=np.maximum.accumulate(raw,axis=1)
        assert np.isfinite(y).all() and (y>=0).all()
        rows.append(dict(stage='fine_tuned',stage_index=0,epoch=None,label='Fine-tuned',color=S.C['aqua'],
                         family='conv2_selective' if mode=='SEL' else 'generic_conv2',is_generic=mode=='generic',
                         dim=basis_dim if mode=='SEL' else None,parameter_norm=None,radii=radii,y=y))
    return dict(title='Resnet head',architecture='ResNet layer4.2.conv2',source=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                extra_sources=[dict(path=str(arrays),sha256=hashlib.file_digest(arrays.open('rb'),'sha256').hexdigest())]
                if hasattr(hashlib,'file_digest') else [dict(path=str(arrays))],
                scope='Fine-tuned ResNet-50, RESISC45, seed0. User-requested panel title Resnet head denotes final convolutional block layer4.2.conv2, not the fc classifier. 10,170-dimensional sampled selective space, full reviewed family 10,171. Actual fp32 forwards with fp64 validation; absolute RMS units.',rows=rows)

def saved_ray_panel(title, names, invariant, stages):
    """Load all available requested stages, preserving the checkpoint as sampling unit."""
    rows=[];sources=[];missing=[]
    for seed,name in names:
        path=Path(S.HERE)/'results/branches'/name
        d=json.loads(path.read_text())['results']
        for si,(st,label,color) in enumerate(stages):
            if st not in d:
                missing.append(dict(seed=seed,stage=st))
                continue
            rec=d[st]
            # Generic relative probes move every layer by the same fraction of
            # its norm, so ||edit|| / amplitude recovers the full parameter norm.
            rel=rec['sets']['generic']['relative'];first=min(rel,key=float)
            amp=float(first)
            assert all(np.isclose(f,amp,rtol=1e-9,atol=1e-12)
                       for probe in rel[first] for f in probe['block_fractions'].values())
            norms=np.array([probe['euclidean_norm']/amp for probe in rel[first]])
            assert np.ptp(norms)<1e-9
            parameter_norm=float(norms.mean())
            for family in [invariant,'generic']:
                data=rec['sets'][family]
                if family==invariant:
                    assert data.get('dim',0)>0
                    if family.startswith('branch:'):assert data['validation_passed']
                grid=data['absolute'];keys=sorted(grid,key=float)
                ids=sorted(p['direction_seed'] for p in grid[keys[0]])
                assert len(ids)==len(set(ids))==128
                a=[]
                for k in keys:
                    byid={p['direction_seed']:p for p in grid[k]}
                    assert len(grid[k])==len(byid)==128 and sorted(byid)==ids
                    assert all(np.isclose(p['euclidean_norm'],float(k),rtol=1e-9,atol=1e-12) for p in byid.values())
                    a.append([byid[i]['true'] for i in ids])
                y=np.maximum.accumulate(np.array(a).T,axis=1)
                assert np.isfinite(y).all() and (y>=0).all()
                rows.append(dict(seed=seed,stage=st,stage_index=si,epoch=rec['epoch'],label=label,color=S.C[color],
                                 family=family,is_generic=family=='generic',dim=data.get('dim'),
                                 parameter_norm=parameter_norm,
                                 radii=np.array(keys,float),y=y))
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    return dict(title=title,source=sources[0]['path'],sha256=sources[0]['sha256'],
                extra_sources=sources[1:],missing=missing,rows=rows)

def probability_figure(panels, stages, name="radius_probability"):
    """Fixed-tolerance, simultaneous one-sided bounds from the saved iid rays."""
    cells=sum(len(row['radii']) for panel in panels for row in panel['rows'])
    alpha,epsilon=.05,1e-3
    fig,axs=plt.subplots(1,len(panels),figsize=(3.35,2.05 if len(panels)==3 else 2.55),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.99,bottom=.30,top=.77,wspace=.22)
    report=[];aggregates=[]
    for pi,(ax,panel) in enumerate(zip(axs,panels)):
        title=panel.get('display_title',['MLP','CNN','Resnet head'][pi])
        ax.set_title(title,fontsize=9,pad=5)
        ax.set(xscale='log',xlim=(.0006,170),ylim=(-.035,1.035))
        ax.set_box_aspect(1)
        ax.set_xticks([.001,.1,100])
        ax.set_yticks([0,.1,.5,1],['0','0.1','0.5','1'])
        ax.tick_params(labelsize=6.1,pad=1.5)
        ax.set_xlabel('Radius $r$',fontsize=7,labelpad=2)
        ax.minorticks_off();ax.grid(axis='y',alpha=.65)
        ax.axhline(.1,color=S.C['ink3'],ls=':',lw=.8)
        grouped={}
        for row in panel['rows']:
            r,y=row['radii'],row['y'];n=len(y)
            k=(y<=epsilon).sum(axis=0)
            lower=np.array([beta.ppf(alpha/cells,int(b),n-int(b)+1) if b else 0. for b in k])
            # Independent formula check: the lower bound inverts the binomial tail.
            for b,l in zip(k,lower):
                if b:assert np.isclose(binom.sf(int(b)-1,n,l),alpha/cells,rtol=1e-9,atol=1e-12)
            assert np.all(np.diff(k)<=0) and np.all(lower<=k/n+1e-14)
            upper=1-lower
            failures=n-k
            direct=np.array([beta.ppf(1-alpha/cells,int(f)+1,n-int(f)) if f<n else 1. for f in failures])
            assert np.allclose(upper,direct,rtol=1e-12,atol=1e-13)
            group=grouped.setdefault((row['stage'],row['is_generic']),[])
            group.append((row,upper))
            safe=r[lower>=.9];cert=float(safe.max()) if len(safe) else 0.
            report.append(dict(architecture=panel.get('architecture',title),seed=row['seed'],stage=row['label'],epoch=row['epoch'],
                               family=row['family'],dimension=row['dim'],n=n,radii=r.tolist(),
                               successes=k.tolist(),estimated_success=(k/n).tolist(),lower=lower.tolist(),
                               failure_upper=upper.tolist(),
                               certified_grid_radius=cert,parameter_norm=row['parameter_norm'],
                               certified_percent_parameter_norm=100*cert/row['parameter_norm'] if row['parameter_norm'] else None,
                               reaches_grid_end=bool(cert==r[-1])))
        # Statistics are over checkpoints, not over pooled ray observations.
        plotted_generics=[]
        for (stage,is_generic),members in grouped.items():
            row=members[0][0];r=row['radii']
            assert all(np.array_equal(m['radii'],r) for m,_ in members)
            assert len({m['seed'] for m,_ in members})==len(members)
            values=np.stack([u for _,u in members]);mean=values.mean(axis=0)
            sd=values.std(axis=0,ddof=1) if len(members)>1 else None
            band=np.zeros_like(mean) if sd is None else sd
            aggregates.append(dict(architecture=panel.get('architecture',title),stage=row['label'],
                                   is_generic=is_generic,seeds=[m['seed'] for m,_ in members],
                                   n_seeds=len(members),radii=r.tolist(),mean_failure_upper=mean.tolist(),
                                   sd_failure_upper=None if sd is None else sd.tolist(),
                                   band_lower=np.maximum(0,mean-band).tolist(),
                                   band_upper=np.minimum(1,mean+band).tolist()))
            if is_generic:
                # Deduplicate only identical displayed means AND seed spreads.
                if any(np.array_equal(mean,m) and np.array_equal(band,b) for m,b in plotted_generics):
                    continue
                plotted_generics.append((mean.copy(),band.copy()))
            color=S.C['ink2'] if is_generic else row['color']
            if sd is not None:
                ax.fill_between(r,np.maximum(0,mean-sd),np.minimum(1,mean+sd),
                                color=color,alpha=.13 if is_generic else .20,lw=0,zorder=1)
            ax.plot(r,mean,ls='--' if is_generic else '-',lw=1.35 if is_generic else 1.25,
                    marker='s' if is_generic else 'o',ms=3.6 if is_generic else 4.1,
                    mfc='white' if is_generic else color,mec=color if is_generic else 'white',
                    mew=.8 if is_generic else .4,color=color,zorder=5 if is_generic else 3)
    axs[0].set_ylabel('Failure probability',fontsize=7,labelpad=3)
    present={row['stage'] for p in panels for row in p['rows']}
    legend_stages=[t for t in stages+[('fine_tuned','Fine-tuned','aqua')] if t[0] in present]
    fig.legend([Line2D([],[],marker='o',ls='',color=S.C[c],ms=3.5) for _,_,c in legend_stages],
               [label for _,label,_ in legend_stages],ncol=4,loc='upper center',bbox_to_anchor=(.53,1),
               fontsize=6.1,handlelength=.5,handletextpad=.3,columnspacing=.65)
    key=[Line2D([],[],marker='o',ls='-',mfc=S.C['ink'],color=S.C['ink'],ms=4),
         Line2D([],[],marker='s',ls='--',mfc='white',color=S.C['ink2'],ms=3.5)]
    fig.legend(key,['Invariant subspace','Generic'],
               ncol=2,loc='lower center',bbox_to_anchor=(.53,.055),fontsize=6.7,
               columnspacing=1.1,handlelength=1.9,handletextpad=.4)
    fig.text(.53,.025,r'Mean $\pm$ seed SD; $\varepsilon=10^{-3}$; dotted: $\delta=0.1$',
             ha='center',fontsize=6.1)
    fig.canvas.draw()
    assert all(abs(ax.get_window_extent().width/ax.get_window_extent().height-1)<1e-10 for ax in axs)
    for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}')
    plt.close(fig)
    metadata=dict(epsilon=epsilon,confidence=1-alpha,simultaneous_cells=cells,
                  method='One-sided Clopper-Pearson failure upper bound, alpha/cells per cell; grid-prefix response.',
                  display='Mean of per-checkpoint upper bounds, with sample SD (ddof=1) across available training seeds. SD bands clipped to [0,1] for display only; no band for single-seed stages. Identical generic mean/SD curves drawn once. Lines guide the eye; no between-radius certificate.',
                  scope='Fixed tolerance; simultaneous coverage over all included checkpoints, spaces and tested radii. The mean upper bounds the mean unsafe probability over included checkpoints, not each seed or unseen training seeds. Rays are not pooled. Seed SD is descriptive, not a confidence interval.',
                  parameter_norm_source='Generic block-balanced relative probes: Euclidean edit norm / common layer-relative amplitude.',
                  sources=[dict(path=p['source'],sha256=p['sha256'],extra_sources=p.get('extra_sources',[]),scope=p.get('scope'),missing=p.get('missing',[])) for p in panels],rows=report,aggregates=aggregates)
    (OUT/f'{name}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Probability figure:',cells,'simultaneous cells; all binomial inversions verified')

def main(argv=None):
    parser=argparse.ArgumentParser(description="Failure-probability curves for the published MLP, CNN and ResNet spaces.")
    parser.add_argument('--probability',action='store_true',help=argparse.SUPPRESS)
    parser.parse_args(argv)
    OUT.mkdir(parents=True,exist_ok=True)
    stages=[('high_accuracy','High accuracy','orange'),('mse_interpolation','Interpolation','blue'),('late_endpoint','Epoch 500','purple')]
    models=[('ReLU MLP · width 64','mnist_mlp_w64_seed{}_rays.json','branch:bottom_up'),('CNN','mnist_cnn_seed{}_patch_rays.json','constructed')]
    panels=[saved_ray_panel(title,[(seed,name.format(seed)) for seed in range(3)],invariant,stages) for title,name,invariant in models]
    panel=resnet_selective_panel()
    for row in panel['rows']:row['seed']=0
    panels.append(panel)
    probability_figure(panels,stages)
