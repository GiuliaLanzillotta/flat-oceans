"""Shared true-network evaluator for the November 25 ray previews."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import torch
from flatocean import frozen_lib as FL
from torch.func import functional_call

def evaluate(net,X,y,V,radii,batch=16,first_layer=0):
 theta=FL.flat_params(net).double(); shapes={k:v.shape for k,v in net.named_parameters()}; slices=list(FL.param_slices(net))
 def params(v):return {k:v[...,a:b].reshape(*v.shape[:-1],*shapes[k]) for k,a,b in slices}
 base=params(theta); F0=net(X); Y=torch.nn.functional.one_hot(y.long(),F0.shape[1]).double(); s=len(X)
 h0=X
 for l in range(first_layer):h0=net.act(net.layers[l](h0))
 # The prefix is cached only when its parameter edit is identically zero.
 for k,a,b in slices:
  if k.startswith('layers.') and int(k.split('.')[1])<first_layer:assert torch.count_nonzero(V[:,a:b])==0
 z0=net.layers[first_layer](h0); drift=torch.zeros(len(V),len(radii),dtype=torch.float64); loss=torch.zeros_like(drift); checks={}
 for lo in range(0,len(V),batch):
  hi=min(lo+batch,len(V)); vp=params(V[lo:hi]); ell=first_layer
  dz=torch.matmul(h0,vp[f'layers.{ell}.weight'].transpose(-1,-2))+vp[f'layers.{ell}.bias'][:,None,:]
  for j,r in enumerate(radii):
   h=net.act(z0[None]+r*dz)
   for ell in range(first_layer+1,len(net.layers)):
    w=base[f'layers.{ell}.weight'][None]+r*vp[f'layers.{ell}.weight'];b=base[f'layers.{ell}.bias'][None]+r*vp[f'layers.{ell}.bias'];h=net.act(torch.bmm(h,w.transpose(1,2))+b[:,None,:])
   w=base['readout.weight'][None]+r*vp['readout.weight'];b=base['readout.bias'][None]+r*vp['readout.bias'];f=torch.bmm(h,w.transpose(1,2))+b[:,None,:]
   drift[lo:hi,j]=((f-F0[None]).square().sum((1,2))/s).sqrt();loss[lo:hi,j]=(f-Y[None]).square().mean((1,2))
   if lo==0 and j in (0,len(radii)-1):
    ref=functional_call(net,params(theta+float(r)*V[0]),(X,));err=float((f[0]-ref).abs().max());rel=err/max(1.,float(ref.abs().max()))
    assert rel<1e-11,(r,err,rel);checks[str(r)]={'absolute_max_error':err,'relative_max_error':rel}
 assert torch.isfinite(drift).all() and torch.isfinite(loss).all()
 return drift,loss,checks
