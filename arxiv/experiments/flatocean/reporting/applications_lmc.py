"""Appendix connectivity figures from saved measurements only.

No model evaluation, training, or factorization. All inherited-Adam sibling and independent-parent segment records are
retained; optimizer-reset runs are excluded; older 128-probe ray summaries are superseded by their 256-probe reruns.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import NullFormatter
from .applications import ROOT, PAPER, OUT, read, save
from . import style as S

PAIRS = [(0,350),(1,350),(0,450)]
MARKERS = ['o','s','^']

def segments():
    records=[]
    for f in sorted((ROOT/'results/lmc/segments').glob('*.json')):
        if not re.search(r'_s(\d)',f.stem):
            continue
        d=read('lmc/segments/'+f.name)
        spawn=(d.get('spawn') or {}).get('A') or {}
        model='CNN' if 'cnn' in f.name else 'MLP'
        rec=dict(name=f.stem,model=model,policy=spawn.get('adam_policy_kind','independent'),epoch=spawn.get('epoch',0),data=d)
        if rec['policy']=='reset': continue
        rec['seed']=int(re.search(r'_s(\d)',f.stem)[1]);records.append(rec)
    return records

def square(ax):
    ax.set_box_aspect(1);ax.tick_params(labelsize=6.5)

def cohorts_and_curves(records):
    cohorts=[('MLP','inherited'),('CNN','inherited')]
    norm=Normalize(0,450); cmap=plt.get_cmap('viridis')
    for split in ['analysis','test']:
        fig,axs=plt.subplots(2,4,figsize=(7,3.9))
        fig.subplots_adjust(left=.085,right=.98,top=.94,bottom=.13,hspace=.43,wspace=.48)
        for row,(model,policy) in enumerate(cohorts):
            chosen=[r for r in records if r['model']==model and r['policy'] in [policy,'independent']]
            for r in sorted(chosen,key=lambda r:r['policy']!='independent'):
                a=r['data']['sets'][split];ps=a['points'];xs=np.array([p['alpha'] for p in ps])
                mean=(a['loss_A']['mse']+a['loss_B']['mse'])/2
                chord=(1-xs)*a['loss_A']['mse']+xs*a['loss_B']['mse']
                vals=[(np.array([p['loss']['mse'] for p in ps])-chord)/mean,
                      [100*p['loss']['error_rate'] for p in ps],
                      [(p['change_from_A']+p['change_from_B'])/2 for p in ps],
                      [p['interpolation_deviation'] for p in ps]]
                independent=r['policy']=='independent'
                for col,y in enumerate(vals):
                    axs[row,col].plot(xs,y,color=S.C['ink3'] if independent else cmap(norm(r['epoch'])),ls='--' if independent else '-',lw=1.6,alpha=.55 if independent else .8)
            for col,ax in enumerate(axs[row]):
                square(ax);ax.set(xlim=(0,1),xticks=[0,.5,1]);ax.set_xlabel(r'$\alpha$',labelpad=0)
                ax.set_title(['MSE excess / endpoint MSE','Classification error (%)',r'Endpoint distance $E(\alpha)$',r'Chord deviation $\|D_\alpha\|_X$'][col],fontsize=7)
                if col==0:
                    ax.set_yscale('symlog',linthresh=1);ax.axhline(10,color=S.C['orange'],ls=':',lw=1)
                    ax.set_ylabel(model+'\n'+('Adam inherited' if policy=='inherited' else 'Adam reset'),fontsize=7)
                elif col==1:ax.set_yscale('symlog',linthresh=.1)
                else: ax.set_yscale('symlog',linthresh=.01)
                values=np.concatenate([np.asarray(line.get_ydata(),float) for line in ax.lines])
                ax.set_ylim(min(-1,float(values.min())*1.1) if col==0 else 0,max(float(values.max())*1.15,.001))
        cax=fig.add_axes([.22,.048,.40,.015]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal');cb.set_label('Fork epoch',fontsize=7,labelpad=1);cb.ax.tick_params(labelsize=6)
        fig.legend(handles=[Line2D([],[],color=S.C['ink3'],ls='--',label='Independent parents')],loc='lower right',bbox_to_anchor=(.99,.02),fontsize=7)
        save(fig,'lmc_segments_'+split)
    rows=[]
    for model,policy in cohorts+[('MLP','independent'),('CNN','independent')]:
        rr=[r for r in records if (r['model'],r['policy'])==(model,policy)]
        epochs=', '.join(map(str,sorted({r['epoch'] for r in rr}))) if policy!='independent' else '--'
        n=len(rr);counts=[]
        for split in ['analysis','test']:
            counts.append(str(sum(r['data']['sets'][split]['barriers']['loss_connected']['value'] for r in rr))+f'/{n}')
        rows.append(' & '.join([model,policy,epochs,str(n)]+counts)+r' \\')
    (PAPER/'section_e_lmc_cohorts.tex').write_text(r'''\begin{table}[!htbp]\centering\small
\caption{All segment comparisons, with each pair counted once. All cohorts cover seeds 0--2. MLP forks at epochs 350 and 450 have three and one pairs respectively. Other fork epochs have three pairs each. Siblings inherit the Adam state. A pass means that the sampled MSE barrier is less than ten times the mean endpoint MSE, evaluated separately on each set.}\label{tab:lmc-cohorts}
\begin{tabular}{llp{.30\linewidth}rrr}\toprule
Network & Adam state & Fork epochs & Pairs & Analysis pass & Test pass\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}'+'\n')




def gates(records):
    fig,flat=plt.subplots(1,6,figsize=(7,1.85));axs=flat.reshape(2,3)
    fig.subplots_adjust(left=.055,right=.99,top=.83,bottom=.34,wspace=.55)
    for row,model in enumerate(['MLP','CNN']):
        for col,policy in enumerate(['inherited','independent']):
            rr=[r for r in records if r['model']==model and r['policy']==policy]
            ps=[r['data']['sets']['analysis']['points'] for r in rr]
            alpha=[q['alpha'] for q in ps[0]]
            for layer in range(len(ps[0][0]['gates'])):
                vals=np.array([[q['gates'][layer]['agree_A'] for q in qs] for qs in ps])
                axs[row,col].plot(alpha,np.median(vals,axis=0),color=S.LAYER_COLS[layer])
                axs[row,col].fill_between(alpha,vals.min(axis=0),vals.max(axis=0),color=S.LAYER_COLS[layer],alpha=.12,lw=0)
            axs[row,col].set(title=model+(' siblings' if col==0 else ' parents'),xlabel=r'$\alpha$',ylabel='Gate agreement',ylim=(.3,1.01))
        z=np.load(ROOT/f'results/paper_slices/selective_{model.lower()}.npz');mid=np.argmin(abs(z['alphas']-.5))
        for l in range(int(z['nblk'])):axs[row,2].scatter(z[f'act_{l}'],z[f'agree_{l}'][mid],s=4,color=S.LAYER_COLS[l],alpha=.65)
        axs[row,2].set(title=model+' selective',xlabel='Activity at A',ylabel='Gate agreement',ylim=(.3,1.01),xlim=(-.02,1.02))
    for ax in axs.ravel():
        square(ax);ax.tick_params(labelsize=5);ax.xaxis.label.set_size(5.5);ax.yaxis.label.set_size(5.5);ax.title.set_size(6)
        ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
    fig.legend(handles=[Line2D([],[],color=S.LAYER_COLS[l],label=f'L{l+1}') for l in range(5)],loc='lower center',ncol=5,fontsize=6)
    save(fig,'lmc_gates_all')

def landscapes():
    from paper_figures.lmc_landscape import draw
    for model in ['mlp','cnn']:
        files=sorted((ROOT/'results/paper_slices').glob(f'landscape_{model}_e*.npz'),key=lambda f:int(f.stem.split('_e')[1]))
        files=[f for f in files if int(f.stem.split('_e')[1]) in {r['epoch'] for r in segments() if r['model']==model.upper() and r['policy']=='inherited'}]
        fig,axs=plt.subplots(len(files),4,figsize=(7,1.7*len(files)+.6))
        fig.subplots_adjust(left=.075,right=.985,top=.95,bottom=.09,wspace=.055,hspace=.075)
        ims={}
        for row,f in enumerate(files):
            z=np.load(f)
            for col,(key,title) in enumerate([('mse','MSE per output'),('fun','Function'),('froz','Frozen'),('trans','Transition')]):
                norm=LogNorm(1e-5,.1) if col==0 else LogNorm(.01,.7)
                ims[col]=draw(axs[row,col],z,key,'magma' if col==0 else 'viridis',norm);axs[row,col].set_box_aspect(1)
                if row==0:axs[row,col].set_title(title,fontsize=11)
                for label in axs[row,col].texts:
                    label.set_fontsize(10)
                    label.set_position((-9,3) if label.get_text()=='B' else (3,-10) if label.get_text()=='P' else (3,3))
            axs[row,0].set_ylabel('Fork '+f.stem.split('_e')[1],fontsize=11)
        for col,(x,w,label) in enumerate([(.09,.19,'MSE'),(.34,.63,r'$\|\mathrm{response}\|_X$')]):
            cb=fig.colorbar(ims[0 if col==0 else 3],cax=fig.add_axes([x,.035,w,.013]),orientation='horizontal');cb.set_label(label,fontsize=10,labelpad=1);cb.ax.tick_params(labelsize=9)
        save(fig,'lmc_planes_'+model)


