"""Read-only postprocessing of saved eigensystems and compatible bases. CPU float64; no model evaluations."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from flatocean.paths import RESULTS
import json,time,torch,argparse

def main():
 torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
 base=RESULTS
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=RESULTS/'hessian_gn')
 args=parser.parse_args()
 out=args.output
 out.mkdir(parents=True,exist_ok=True)
 taus=[10.**(-j) for j in range(14,3,-1)]
 result={'protocol':{'thresholds':taus,'space':'fixed saved bottom-up compatible invariant subspace; no approximate construction','coverage':'sum ||Q_O^T q_j||^2 / target dimension','selection':'absolute eigenvalue <= tau for both H and G','precision':'float64 CPU; classifications float64','new_experiments':False},'seeds':{}}
 arrays={}
 for seed in range(3):
  t=time.time();m=torch.load(base/f'hessian_gn/hgn_cov_w64_seed{seed}_mats.pt',map_location='cpu',weights_only=False,mmap=True)
  saved=torch.load(base/f'branches/mnist_mlp_w64_seed{seed}_epoch0500_bottom_up.pt',map_location='cpu',weights_only=False);bs=saved['bases']
  parts=[];start=0;k=sum(v.shape[1] for v in bs.values());O=torch.zeros(25418,k,dtype=torch.float64);col=0
  for name,n,d in [('W1',64,256),('W2',64,64),('W3',64,64),('V',10,64)]:
   B=bs[name];assert B.shape[0]==n*(d+1);parts.append((name,n,d,start,B));a=B.reshape(n,d+1,-1);r=B.shape[1]
   O[start:start+n*d,col:col+r]=a[:,:d,:].reshape(n*d,r);O[start+n*d:start+n*(d+1),col:col+r]=a[:,d,:]
   start+=n*(d+1);col+=r
  assert start==25418
  gramerr=float((O.T@O-torch.eye(k,dtype=torch.float64)).abs().max());assert gramerr<1e-8
  e={'ocean_dimension':k,'p':25418,'k_over_p_baseline':k/25418,'basis_orthogonality_max':gramerr,'sides':{}}
  for side in ['G','H']:
   all_ev=m['ev'+side];mask=all_ev.abs()<=1e-4;ev=all_ev[mask];Q=m['oceans']['GN_tail|0.0001'] if side=='G' else m['Hnull_1e-4'];assert Q.shape==(25418,len(ev))
   masses=torch.zeros(len(ev),dtype=torch.float64);normsq=torch.zeros_like(masses);independent_error=0.
   gen=torch.Generator().manual_seed(917+seed);sel=torch.randperm(len(ev),generator=gen)[:64];tight=torch.where(ev.abs()<=1e-14)[0];sel=torch.unique(torch.cat([sel,tight[:8],tight[-8:]]));Qc=Q[:,sel]
   check_ortho=float((Qc.T@Qc-torch.eye(len(sel),dtype=torch.float64)).abs().max());assert check_ortho<1e-8
   direct=(O.T@Qc).square().sum(0)
   for a in range(0,len(ev),256):
    V=Q[:,a:a+256];normsq[a:a+256]=V.square().sum(0)
    for name,n,d,start,B in parts:
     if B.shape[1]==0:continue
     # Independent evaluation in augmented local coordinates: no global embedding matrix in this path.
     w=V[start:start+n*d].reshape(n,d,-1);bias=V[start+n*d:start+n*(d+1)].reshape(n,1,-1);vaug=torch.cat([w,bias],dim=1).reshape(n*(d+1),-1)
     masses[a:a+256]+=(B.T@vaug).square().sum(0)
   independent_error=float((masses[sel]-direct).abs().max());assert independent_error<1e-10
   assert float((normsq-1).abs().max())<1e-8
   assert float(masses.min())>=-1e-10 and float((masses-normsq).max())<1e-8
   tight_sel=tight[torch.linspace(0,len(tight)-1,min(64,len(tight))).long()];V=Q[:,tight_sel];R=V-O@(O.T@V);direct_residual_sq=R.square().sum(0)
   complement_error=float((direct_residual_sq-(normsq[tight_sel]-masses[tight_sel])).abs().max());assert complement_error<1e-10
   cells={}
   for tau in taus:
    keep=ev.abs()<=tau;dim=int(keep.sum());mass=float(masses[keep].sum());cells[f'{tau:g}']={'target_dim':dim,'projection_mass':mass,'coverage':mass/dim,'ocean_in_target_mass_fraction':mass/k}
   e['sides'][side]={'cells':cells,'checks':{'parent_count_mask_matches':True,'all_column_norm_max_error':float((normsq-1).abs().max()),'sampled_basis_orthogonality_max_error':check_ortho,'independent_local_vs_global_mass_max_error':independent_error,'sampled_tight_projection_complement_identity_max_error':complement_error,'sampled_tight_outside_norm_max':float(R.norm(dim=0).max())},'negative_eigenvalue_min':float(all_ev.min())}
   arrays[f's{seed}_{side}']={'source_selected_indices':torch.where(mask)[0],'eigenvalues':ev.clone(),'projection_mass_by_column':masses,'normsq':normsq,'independent_check_columns':sel,'tight_residual_columns':tight_sel,'tight_residual_sq':direct_residual_sq}
   print('DONE',seed,side,'checks',e['sides'][side]['checks'],flush=True)
  e['runtime_seconds']=time.time()-t;result['seeds'][str(seed)]=e
  (out/'fixed_ocean_figure_table.json').write_text(json.dumps(result,indent=2));torch.save(arrays,out/'fixed_ocean_coverage_arrays.pt')
  print('SEED',seed,'seconds',e['runtime_seconds'],flush=True)
 print('ALL DONE',flush=True)

if __name__ == "__main__":
 main()
