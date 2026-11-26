"""Finite-radius tests of cumulative Hessian/GGN eigenspaces and controls."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse,os,sys,time,json,hashlib
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE))
from flatocean import runs
from flatocean import frozen_lib as FL
from torch.func import functional_call,jvp,vjp,grad
from ray_preview_common import evaluate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='0,1,2');ap.add_argument('--pilot',action='store_true');ap.add_argument('--n',type=int,default=256);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--operator',choices=['H','G'],default='H');ap.add_argument('--cumulative',action='store_true');args=ap.parse_args()
 torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')));torch.set_grad_enabled(False)
 kind='threshold' if args.cumulative else 'shell'
 out=HERE/f"results/{'hessian' if args.operator=='H' else 'ggn'}_{kind}_rays"/('pilot' if args.pilot else 'production');out.mkdir(parents=True,exist_ok=True)
 taus=[1e-14,1e-12,1e-10,1e-8,1e-6,1e-4];radii=np.array([.001,.003,.01,.03,.1,.2,.3,.5,.7,1.,1.5,2.,3.,5.,7.,10.,15.,20.,30.,50.,100.,200.,300.,500.,700.,1000.])
 if args.cumulative:radii=radii[radii<=100.]
 if args.pilot:radii=np.array([.1,1.,10.])
 n=args.n
 for seed in map(int,args.seeds.split(',')):
  start=time.time();dst=out/f'seed{seed}.pt';assert not dst.exists()
  run=runs.load_run('mnist','mlp',seed,tag='w64',epoch=500,device='cpu',sets=('analysis',));net=run['net'].eval()
  for pr in net.parameters():pr.requires_grad_(False)
  X,y,idx=run['data']['analysis'];Y=torch.nn.functional.one_hot(y.long(),10).double();F0=net(X);s=len(X);theta=FL.flat_params(net).double();p=theta.numel()
  assert s==10000 and p==25418 and net.activation=='relu' and not net.batchnorm
  ps=list(FL.param_slices(net));shapes={k:v.shape for k,v in net.named_parameters()}
  def params(v):return {k:v[...,a:b].reshape(*v.shape[:-1],*shapes[k]) for k,a,b in ps}
  base=params(theta);L=len(net.layers)
  def F(t):return functional_call(net,params(t),(X,))
  def objective(t):return (F(t)-Y).square().sum()/s
  path=HERE/f'results/hessian_gn/hgn_cov_w64_seed{seed}_mats.pt';m=torch.load(path,mmap=True,weights_only=False,map_location='cpu');ev=m['ev'+args.operator];mask=ev.abs()<=1e-4;vals=ev[mask];Q=m['Hnull_1e-4'] if args.operator=='H' else m['oceans']['GN_tail|0.0001']
  assert Q.dtype==torch.float64 and Q.shape==(p,int(mask.sum()))
  assert m['meta']['seed']==seed and m['meta']['epoch']==500 and m['meta']['m']==s
  branch_path=HERE/f'results/branches/mnist_mlp_w64_seed{seed}_epoch0500_bottom_up.pt'
  branch=torch.load(branch_path,weights_only=False,map_location='cpu')['bases']
  O=torch.zeros(p,sum(b.shape[1] for b in branch.values()),dtype=torch.float64);row=col=0
  for label,nout,nin in [('W1',64,256),('W2',64,64),('W3',64,64),('V',10,64)]:
   b=branch[label];k=b.shape[1];a=b.reshape(nout,nin+1,k)
   O[row:row+nout*nin,col:col+k]=a[:,:nin,:].reshape(nout*nin,k)
   O[row+nout*nin:row+nout*(nin+1),col:col+k]=a[:,nin,:]
   row+=nout*(nin+1);col+=k
  assert O.shape[1]==[3619,4573,4072][seed] and row==p
  # Published seeds 0/1 used an equivalent atomic basis realization; seed 2
  # used the compatible basis. Preserve the recorded sampling coordinates.
  if seed in (0, 1):
   O = m['oceans']['exact_branch_union']
  gen=torch.Generator().manual_seed(20261125+seed);coef=torch.randn(Q.shape[1],n,generator=gen,dtype=torch.float64)
  records={};validation={};prov={'checkpoint_sha256':sha(HERE/f'results/tier1_models/mnist_mlp_w64/seed_{seed}/checkpoints/epoch_0500.pt'),'analysis_indices_sha256':hashlib.sha256(np.asarray(idx).tobytes()).hexdigest(),'eigensystem':str(path.relative_to(HERE)),'eigensystem_bytes':path.stat().st_size,'eigenvalues_sha256':hashlib.sha256(ev.numpy().tobytes()).hexdigest(),'script_sha256':sha(__file__),'compatible_ocean_basis_sha256':sha(branch_path)}
  names=[f'{args.operator}_{t:g}' for t in taus]+['ocean','generic']
  if args.pilot:names=[args.operator+'_1e-14',args.operator+'_0.0001','ocean','generic']
  for name in names:
   t0=time.time()
   if name.startswith(args.operator+'_'):
    upper=float(name[2:]);lower=0. if args.cumulative or upper==taus[0] else taus[taus.index(upper)-1];take=(vals.abs()<=upper)&((vals.abs()>lower) if lower else torch.ones_like(vals,dtype=torch.bool));c=coef[take];b=Q[:,take];d=int(take.sum());raw=b@c;err=float((raw.norm(dim=0)/c.norm(dim=0)-1).abs().max());assert err<1e-10
    V=(raw/raw.norm(dim=0)).T.contiguous();del b,raw,c
    with torch.enable_grad():
     if args.operator=='H':hv=jvp(grad(objective),(theta,),(V[0],))[1]
     else:
      jv=jvp(F,(theta,),(V[0],))[1];_,pullback=vjp(F,theta);hv=(2./s)*pullback(jv)[0]
    c=coef[take,0];expected=Q[:,take]@(vals[take]*c/c.norm());errh=float((hv-expected).norm());assert errh<1e-10,(name,errh)
    validation[name]={'combination_norm_error':err,'operator_vector_residual':errh,'max_abs_eigenvalue':float(vals[take].abs().max())}
   elif name=='ocean':
    d=O.shape[1];c=torch.randn(d,n,generator=gen,dtype=torch.float64);raw=O@c;err=float((raw.norm(dim=0)/c.norm(dim=0)-1).abs().max());assert err<1e-10
    V=(raw/raw.norm(dim=0)).T.contiguous();del c,raw;validation[name]={'combination_norm_error':err}
   else:
    d=p;V=torch.randn(n,p,generator=gen,dtype=torch.float64);V/=V.norm(dim=1,keepdim=True);validation[name]={}
   assert (V.norm(dim=1)-1).abs().max()<1e-12
   # Recreate the published shared controls without a previous ray-result file.
   if args.operator == 'G' and name in ('ocean', 'generic'):
    control_gen = torch.Generator().manual_seed(20261125 + seed)
    torch.randn(m['Hnull_1e-4'].shape[1], n, generator=control_gen, dtype=torch.float64)
    control_coeff = torch.randn(O.shape[1], n, generator=control_gen, dtype=torch.float64)
    if name == 'ocean':
     raw = O @ control_coeff
     V = (raw / raw.norm(dim=0)).T.contiguous()
    else:
     V = torch.randn(n, p, generator=control_gen, dtype=torch.float64)
     V /= V.norm(dim=1, keepdim=True)
   drift, loss, checks = evaluate(net, X, y, V, radii, args.batch)
   validation[name]['independent_forward_checks'] = checks
   records[name]={'dimension':d,'directions':V,'drift':drift,'loss_mse':loss}
   print(f'seed={seed} {name} dim={d} rays={n} radii={len(radii)} seconds={time.time()-t0:.1f} drift@1={float(drift[:,list(radii).index(1.)].mean()):.6g}',flush=True)
   torch.save({'seed':seed,'radii':radii,'n':n,'records':records,'validation':validation,'provenance':prov,'base_loss_mse':float((F0-Y).square().mean()),'base_output_rms':float(F0.norm()/s**.5)},out/f'seed{seed}_partial.pt')
  (out/f'seed{seed}_partial.pt').replace(dst)
  protocol={'directions':('normalized iid Gaussian coordinates in nested cumulative bands' if args.cumulative else 'normalized iid Gaussian coordinates in disjoint spectral shells')+'; shared controls regenerated from the Hessian random stream','spectral_band':(f'abs(lambda_{args.operator}) <= tau' if args.cumulative else f'first numerical kernel abs(lambda_{args.operator}) <= 1e-14, then disjoint shells previous_tau < abs(lambda_{args.operator}) <= tau')+'; loss=sum_outputs/s; G=2 J^T J/s','drift':'sqrt(sum_i ||F(theta+rv)-F(theta)||^2/s)','loss_mse':'mean over samples and outputs','precision':'float64','epsilon':.001,'failure':'any tested radius <=r exceeds epsilon; no between-grid guarantee'}
  (out/f'seed{seed}.json').write_text(json.dumps({'seed':seed,'n':n,'radii':radii.tolist(),'taus':taus,'dimensions':{k:v['dimension'] for k,v in records.items()},'validation':validation,'provenance':prov,'runtime_s':time.time()-start,'protocol':protocol},indent=2))
  print(f'SEED {seed} COMPLETE in {time.time()-start:.1f}s',flush=True)
 print(args.operator+' '+kind.upper()+' RAYS COMPLETE',flush=True)
if __name__=='__main__':main()
