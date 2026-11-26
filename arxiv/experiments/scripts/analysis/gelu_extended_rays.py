"""GELU L3 numerical/approximate kernels, same checkpoints/data, extended true-network rays."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse,os,sys,json,time,hashlib
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE))
from flatocean import runs
from flatocean.factors import load_factor,factor_path
from flatocean.analyses.structured_directions import block_index_map
from flatocean import frozen_lib as FL
from torch.func import functional_call,jvp
from ray_preview_common import evaluate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='0,1,2');ap.add_argument('--stages',default='high_accuracy,mse_interpolation,late_endpoint');ap.add_argument('--pilot',action='store_true');ap.add_argument('--n',type=int,default=256);a=ap.parse_args()
 torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')));torch.set_grad_enabled(False)
 out=HERE/'results/gelu_extended_rays'/('pilot' if a.pilot else 'production');out.mkdir(parents=True,exist_ok=True)
 radii=np.array([.001,.003,.01,.03,.1,.3,1.,3.,10.,30.,100.,300.,1000.])
 if a.pilot:radii=np.array([.1,1.,10.,1000.])
 for seed in map(int,a.seeds.split(',')):
  oldpath=HERE/f'results/transition_v2/mnist_mlp_gelu_seed{seed}_structured.json';old=json.loads(oldpath.read_text())
  for stage in a.stages.split(','):
   start=time.time();dst=out/f'seed{seed}_{stage}.pt';assert not dst.exists();epoch=old['results'][stage]['epoch']
   run=runs.load_run('mnist','mlp',seed,tag='gelu',epoch=epoch,device='cpu');net=run['net'].eval();X,y,idx=run['data']['analysis'];s=len(X);theta=FL.flat_params(net).double();p=theta.numel()
   assert net.activation=='gelu' and not net.batchnorm and s==10000
   sv,B,sf,pb,status=load_factor(f'mnist_mlp_gelu_seed{seed}',epoch,'3',run,runs.run_dir('mnist','mlp',seed,'gelu'),return_status=True);assert sf==s
   indices=torch.as_tensor(block_index_map(net,'3'));assert pb==len(indices)==16512
   rates=np.asarray(sv)/s**.5;kernel=np.asarray(sv)<=1e-9*sv[0]
   ds=old['direction_seeds'][:a.n];ambient=torch.stack([torch.randn(p,generator=torch.Generator().manual_seed(int(d)),dtype=torch.float64) for d in ds]);ambient/=ambient.norm(dim=1,keepdim=True);ub=ambient[:,indices].T
   shapes={k:v.shape for k,v in net.named_parameters()};sl=list(FL.param_slices(net))
   def F(t):return functional_call(net,{k:t[b:e].reshape(shapes[k]) for k,b,e in sl},(X,))
   records={};checks={};fpath=factor_path(f'mnist_mlp_gelu_seed{seed}',epoch,'3')
   prov={'checkpoint_sha256':sha(Path(runs.run_dir('mnist','mlp',seed,'gelu'))/f'checkpoints/epoch_{epoch:04d}.pt'),'factor_sha256':sha(fpath),'factor_status':status,'old_record_sha256':sha(oldpath),'analysis_indices_sha256':hashlib.sha256(np.asarray(idx).tobytes()).hexdigest(),'script_sha256':sha(__file__),'evaluator_sha256':sha(Path(__file__).with_name('ray_preview_common.py'))}
   assert prov['checkpoint_sha256']==old['results'][stage]['checkpoint_sha256']
   for name in ['kernel','approx_1e-06','approx_0.0001','approx_0.001','generic_L3']:
    t=time.time();ch={}
    if name=='generic_L3':vb=ub.clone();dim=pb
    else:
     take=kernel if name=='kernel' else rates<=float(name[7:]);dim=int(take.sum())
     if dim==0:records[name]={'dimension':0,'empty':True};print(seed,stage,name,'EMPTY',flush=True);continue
     basis=B[:,take];coef=basis.T@ub;vb=basis@coef
     normerr=float((vb.norm(dim=0)/coef.norm(dim=0)-1).abs().max());assert normerr<1e-6
     ch['stored_basis_combination_norm_error']=normerr;ch['largest_stored_rate']=float(rates[take].max())
     # fp32 positive-tail columns, fp64 kernel columns: preserve and quantify precision.
     ch['basis_storage']='fp64 numerical-kernel columns; fp32 positive-spectrum columns promoted to fp64'
     del basis,coef
    vb/=vb.norm(dim=0);V=torch.zeros(a.n,p,dtype=torch.float64);V[:,indices]=vb.T
    assert float((V.norm(dim=1)-1).abs().max())<1e-12
    with torch.enable_grad():_,jv=jvp(F,(theta,),(V[0],))
    expo=float(jv.norm()/s**.5);ch['first_direction_JVP_exposure']=expo
    if name.startswith('approx_'):assert expo<=float(name[7:])*(1+1e-3)+1e-10,(name,expo)
    if name=='kernel':assert expo<1e-9*float(sv[0])/s**.5+1e-10
    drift,loss,forward=evaluate(net,X,y,V,radii,batch=8,first_layer=2);ch['independent_forward_checks']=forward
    if name=='kernel':
     oldcat=old['results'][stage]['categories']['kernel_3'];assert dim==oldcat['dim'];errs=[]
     for j,r in enumerate(radii):
      if str(float(r)) not in oldcat['probes']:continue
      byid={v['direction_seed']:v['true'] for v in oldcat['probes'][str(float(r))]};ref=torch.tensor([byid[d] for d in ds],dtype=torch.float64)
      err=float((drift[:,j]-ref).abs().max());assert err<1e-10,(stage,r,err);errs.append(err)
     ch['original_kernel_drift_max_error']=max(errs)
    records[name]={'dimension':dim,'directions':V,'drift':drift,'loss_mse':loss};checks[name]=ch
    print(f's{seed} {stage} e{epoch} {name} dim={dim} time={time.time()-t:.1f}s mean@1={float(drift[:,list(radii).index(1.)].mean()):.5g}',flush=True)
   z={'seed':seed,'stage':stage,'epoch':epoch,'n':a.n,'radii':radii,'parameter_dimension':p,'layer_dimension':pb,'records':records,'validation':checks,'provenance':prov,'direction_seeds':ds,'protocol':{'scope':'only L3 edited; true nonlinear network; unit total Euclidean parameter norm','approximate_kernel':'sigma(J_L3)/sqrt(s)<=eta, including numerical kernel when its rates satisfy eta','sampling':'normalized projection of independent full-parameter Gaussian probes into each selected stored basis; same direction seeds as original','kernel':'sigma<=1e-9*sigma_max','drift':'sqrt(sum_i ||F(theta+rv)-F(theta)||^2/s)','no_uniform_finite_radius_claim':True}}
   torch.save(z,dst);(out/f'seed{seed}_{stage}.json').write_text(json.dumps({k:v for k,v in z.items() if k not in ('records','radii')}|{'radii':radii.tolist(),'dimensions':{k:v['dimension'] for k,v in records.items()},'runtime_s':time.time()-start},indent=2))
   print('COMPLETE',seed,stage,time.time()-start,flush=True)
if __name__=='__main__':main()
