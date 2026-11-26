"""ViT Phase-0f: approximate-null FAMILY characterization, CORRECTED reporting/sampling (bounded; heavy exact dim_P HELD).
Fixes (Giulia): (1) report TESTED brackets [last_safe, first_cross] and last tested-safe radius (interpolated crossing kept
only as descriptive); (2) first-crossing is CUMULATIVE (the first tested radius at/above tol; a later re-entry cannot erase
it); (3) every sampled dW has ALL exact-null components projected out -- BOTH marginals': mlp3 removes the LN-shift
(all-ones-output) [its feature-null is 0]; mlp0 removes the feature-null (nu_u component) [its downstream-null is 0]; applied
to family AND matched-random samples; (4) stable explicit seeds; persist per-direction drift curves, jv, brackets, censoring;
(5) limiter attributed by freeze-one-at-a-time with a 'none' (no appreciable curvature limitation) category -- NOT auto-LN.
Scope: SAMPLED directions at ONE checkpoint (seed 0) vs matched random controls; distributions are NOT uniform guarantees.
Exact 6,190 baseline + co-occurrence bounds + multilayer compatibility stand; explicit co-occurrence construction + heavy
factorization remain HELD."""
import os, sys, json, argparse, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ViT_B_16_Weights
SP=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SP)
from ftvit_paths import cache_path, stable_seed
def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0)))+z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r,g,b,eps):
    mu=r.mean(-1,keepdim=True); sd=torch.sqrt(r.var(-1,unbiased=False,keepdim=True)+eps); return g*(r-mu)/sd+b
RGRID=[0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]
def bracket(drift,tol):
    d=np.asarray(drift,float); cross=np.where(d>=tol)[0]
    if len(cross)==0: return {"last_safe":RGRID[-1],"first_cross":None,"censored":True,"interp":None}
    j=int(cross[0]); ls=RGRID[j-1] if j>0 else 0.0; fc=RGRID[j]
    if j>0:
        lr=np.log([RGRID[j-1],RGRID[j]]); ld=np.log(np.clip([d[j-1],d[j]],1e-30,None)); lt=np.log(tol)
        interp=float(np.exp(lr[0]+(lt-ld[0])*(lr[1]-lr[0])/(ld[1]-ld[0])))
    else: interp=None
    return {"last_safe":ls,"first_cross":fc,"censored":False,"interp":interp}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--seed-tag",required=True)
    ap.add_argument("--n",type=int,default=10000); ap.add_argument("--M",type=int,default=10); ap.add_argument("--out",required=True)
    args=ap.parse_args(); torch.manual_seed(0)
    ck=torch.load(args.ckpt,map_location="cpu"); ncls=ck["n_classes"]
    net=tvm.vit_b_16(weights=None); net.heads=nn.Sequential(); net.heads.add_module("head",nn.Linear(768,ncls))
    net.load_state_dict(ck["state_dict"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    blk=net.encoder.layers[11]
    cls_mid=torch.load(cache_path(f"clsmid_{args.seed_tag}_n{args.n}.pt"))["cls_mid"].double().cpu(); N=cls_mid.shape[0]; s=N
    g2,b2=blk.ln_2.weight.double(),blk.ln_2.bias.double(); eps2=blk.ln_2.eps
    W0,b0=blk.mlp[0].weight.double(),blk.mlp[0].bias.double(); W3,b3=blk.mlp[3].weight.double(),blk.mlp[3].bias.double()
    gf,bf=net.encoder.ln.weight.double(),net.encoder.ln.bias.double(); epsf=net.encoder.ln.eps
    V,bV=net.heads.head.weight.double(),net.heads.head.bias.double(); d=V.shape[0]
    u=ln_fwd(cls_mid,g2,b2,eps2); z0=u@W0.T+b0; a=gelu(z0); z3=a@W3.T+b3; bo=cls_mid+z3
    mu=bo.mean(-1,keepdim=True); s_bo=torch.sqrt(bo.var(-1,unbiased=False,keepdim=True)+epsf); rhat=(bo-mu)/s_bo
    f=ln_fwd(bo,gf,bf,epsf); logits=f@V.T+bV; gp=gelup(z0); one=torch.ones(N,1,dtype=torch.float64)
    logits_rms=float(torch.sqrt((logits**2).sum(1).mean()))
    def apply_JLN(delta):
        c=delta-delta.mean(-1,keepdim=True)-(((rhat*delta).sum(-1,keepdim=True))/768)*rhat; return (gf/s_bo)*c
    def VJ_batch(idx):
        Vg=V*gf; rsum=Vg.sum(1); rh=rhat[idx]; sb=s_bo[idx]; Vgr=rh@Vg.T
        return ((Vg.unsqueeze(0)-rsum.view(1,-1,1)/768)-torch.einsum("bc,bm->bcm",Vgr,rh)/768)/sb.view(-1,1,1)
    # subspaces
    G0=torch.zeros((3072,3072),dtype=torch.float64); G3=torch.zeros((768,768),dtype=torch.float64)
    for i in range(0,N,500):
        sl=torch.arange(i,min(i+500,N)); VJ=VJ_batch(sl); G3+=VJ.reshape(-1,768).T@VJ.reshape(-1,768)
        D0=torch.einsum("bcx,xy->bcy",VJ,W3)*gp[sl].unsqueeze(1); G0+=D0.reshape(-1,3072).T@D0.reshape(-1,3072)
    ones768=torch.ones(768,dtype=torch.float64); Qsh=torch.linalg.svd(torch.eye(768,dtype=torch.float64)-torch.outer(ones768/np.sqrt(768.),ones768/np.sqrt(768.)))[0][:,:767]
    ev3,evec3=torch.linalg.eigh(Qsh.T@G3@Qsh); sv_d3=np.sqrt(ev3.flip(0).clamp(min=0).numpy()); V3=(Qsh@evec3.flip(1))
    ev0,evec0=torch.linalg.eigh(G0); sv_d0=np.sqrt(ev0.flip(0).clamp(min=0).numpy()); V0=evec0.flip(1)
    Vu=torch.linalg.svd(torch.cat([u,one],1),full_matrices=False)[2]; svu=np.asarray(torch.linalg.svdvals(torch.cat([u,one],1)))
    Va=torch.linalg.svd(torch.cat([a,one],1),full_matrices=False)[2]; sva=np.asarray(torch.linalg.svdvals(torch.cat([a,one],1)))
    nu_u=Vu[-1]                                       # exact feature-null of [u;1] (1-dim)
    def proj_exact(block,dA):                         # remove ALL exact-null components (both marginals)
        if block=="mlp3": return dA-torch.outer(ones768,dA.mean(0))                 # remove LN-shift (all-ones-output); feature-null=0
        else: return dA-torch.outer(dA@nu_u,nu_u)                                   # remove feature-null (nu_u); downstream-null=0
    def feat_tail(Vrows,sv,tau):
        lo=int((sv<1e-9*sv[0]).sum()); hi=int((sv<tau*sv[0]).sum()); idx=list(range(len(sv)-hi,len(sv)-lo))
        return Vrows[idx].T if hi>lo else Vrows[:0].T
    nd3_2=int((sv_d3<1e-2*sv_d3[0]).sum()); nd0_2=int((sv_d0<1e-2*sv_d0[0]).sum())
    fam={"mlp3_dtail_k45":("dtail","mlp3",V3[:,767-722:]),"mlp3_dtail_k200":("dtail","mlp3",V3[:,767-567:]),
         "mlp3_dtail_sig1e2":("dtail","mlp3",V3[:,767-nd3_2:]),"mlp0_dtail_sig1e2":("dtail","mlp0",V0[:,3072-nd0_2:]),
         "mlp3_ftail_sig1e3":("ftail","mlp3",feat_tail(Va,sva,1e-3)),"mlp0_ftail_sig1e2":("ftail","mlp0",feat_tail(Vu,svu,1e-2))}
    def jv_from_z3img(dz3img):
        tot=0.0
        for i in range(0,N,1000):
            sl=torch.arange(i,min(i+1000,N)); VJ=VJ_batch(sl); jv=torch.einsum("bcx,bx->bc",VJ,dz3img[sl]); tot+=float((jv**2).sum())
        return np.sqrt(tot)
    def rms_rel(lg): dl=lg-logits; return float(torch.sqrt((dl**2).sum(1).mean()))/logits_rms
    def eval_dir(block,dA):
        if block=="mlp3":
            dz3=a@dA[:,:-1].T+dA[:,-1]; jln=apply_JLN(dz3); jf=jv_from_z3img(dz3); act=[]; fln=[]
            for t in RGRID:
                act.append(rms_rel(ln_fwd(bo+t*dz3,gf,bf,epsf)@V.T+bV)); fln.append(rms_rel((f+t*jln)@V.T+bV))
            return jf,{"actual":act,"frozenLN":fln,"frozenboth":fln}     # no GELU downstream
        else:
            dz0=u@dA[:,:-1].T+dA[:,-1]; d3lin=(gp*dz0)@W3.T; jln=apply_JLN(d3lin); jf=jv_from_z3img(d3lin)
            act=[]; fg=[]; fl=[]; fb=[]
            for t in RGRID:
                an=gelu(z0+t*dz0); z3a=an@W3.T; bon=cls_mid+z3a+b3            # expensive matmul once/radius (shared)
                act.append(rms_rel(ln_fwd(bon,gf,bf,epsf)@V.T+bV)); fl.append(rms_rel((f+apply_JLN(bon-bo))@V.T+bV))
                fg.append(rms_rel(ln_fwd(bo+t*d3lin,gf,bf,epsf)@V.T+bV)); fb.append(rms_rel((f+t*jln)@V.T+bV))
            return jf,{"actual":act,"frozenGELU":fg,"frozenLN":fl,"frozenboth":fb}
    def make_raw(typ,block,basis,mode,rng):
        n=3072 if block=="mlp0" else 768; m1=769 if block=="mlp0" else 3073; k=basis.shape[1]
        if mode=="generic": return torch.randn(n,m1,generator=rng,dtype=torch.float64)
        if typ=="dtail":
            U=basis if mode=="tail" else torch.linalg.qr(torch.randn(n,k,generator=rng,dtype=torch.float64))[0]
            return U@torch.randn(k,m1,generator=rng,dtype=torch.float64)
        Vt=basis if mode=="tail" else torch.linalg.qr(torch.randn(m1,k,generator=rng,dtype=torch.float64))[0]
        return torch.randn(n,k,generator=rng,dtype=torch.float64)@Vt.T
    def dist(x): x=np.asarray(x,float); return {"median":float(np.median(x)),"min":float(x.min()),"max":float(x.max())}
    def limiter(curves):
        la=bracket(curves["actual"],1e-3)["last_safe"]; ll=bracket(curves["frozenLN"],1e-3)["last_safe"]; lb=bracket(curves["frozenboth"],1e-3)["last_safe"]
        if "frozenGELU" in curves:
            lg=bracket(curves["frozenGELU"],1e-3)["last_safe"]; gg=lg-la; gl=ll-la
            if lb<=la: return "none"
            if gg>=gl and gg>0: return "GELU"
            if gl>0: return "LN"
            return "GELU+LN"
        return "LN" if ll>la else "none"
    out={"seed_tag":args.seed_tag,"N":N,"M":args.M,"logits_rms":logits_rms,"RGRID":RGRID,"tol":1e-3,
         "CAVEATS":"SAMPLED directions at ONE checkpoint vs matched random controls; brackets are TESTED radii (interp is descriptive only); NOT uniform guarantees; APPROXIMATE-marginal additions only (all exact nulls projected out, both marginals); co-occurrence LBs stand, explicit construction + heavy dim_P HELD.","families":{}}
    for name,(typ,block,basis) in fam.items():
        k=basis.shape[1]; rec={"type":typ,"block":block,"tail_dim_k":int(k)}
        for mode in ("tail","randsub","generic"):
            rng=torch.Generator().manual_seed(stable_seed(20260,name,mode))
            per=[]; jvr=[]; ls=[]; fcs=[]; cens=0; lims=[]
            for mi in range(args.M):
                dA=proj_exact(block,make_raw(typ,block,basis,mode,rng)); dA=dA/dA.norm()
                jf,cv=eval_dir(block,dA); br=bracket(cv["actual"],1e-3)
                jvr.append(jf/np.sqrt(s)); ls.append(br["last_safe"]); cens+=int(br["censored"]); lims.append(limiter(cv))
                if br["first_cross"] is not None: fcs.append(br["first_cross"])
                per.append({"i":mi,"jv_RMS":jf/np.sqrt(s),"jv_frob":jf,"bracket":br,"limiter":lims[-1],
                            "drift":{kk:[float(x) for x in vv] for kk,vv in cv.items()}})
            rec[mode]={"jv_operator_RMS_over_v":dist(jvr),"last_safe_tested_tol1e-3":dist(ls),
                       "first_cross_tested":(dist(fcs) if fcs else None),"censored_count":cens,
                       "limiter_counts":{c:int(sum(l==c for l in lims)) for c in ("GELU","LN","GELU+LN","none")},
                       "per_direction":per}
        out["families"][name]=rec
        t=rec["tail"]; rs=rec["randsub"]; ge=rec["generic"]
        print(f"{name:20s} k={k:4d} | TAIL jvRMS={t['jv_operator_RMS_over_v']['median']:.2e} last_safe={t['last_safe_tested_tol1e-3']['median']:.2f}[{t['last_safe_tested_tol1e-3']['min']:.2f},{t['last_safe_tested_tol1e-3']['max']:.2f}] cens={t['censored_count']} lim={t['limiter_counts']} | RANDSUB last_safe={rs['last_safe_tested_tol1e-3']['median']:.2f} jvRMS={rs['jv_operator_RMS_over_v']['median']:.2e} | GEN last_safe={ge['last_safe_tested_tol1e-3']['median']:.2f}",flush=True)
    with open(args.out,"w") as fo: json.dump(out,fo,indent=2)
    print("wrote",args.out,"\nDONE",flush=True)
if __name__=="__main__": main()
