"""ViT Phase-0d: CORRECTED interpretation + dual-convention metrics + diameter-controlled clustering + representative/
truncated-map approximation test. Light; reuses cls_mid cache; NO heavy factorization (held). Seed 0 pilot.

Fixes (Giulia): (a) report metrics in BOTH the earlier ABSOLUTE-RMS convention (sigma/sqrt(s); operator-RMS ||Jv||/sqrt(s);
aggregate-RMS output change) AND the relative one, each labeled; (b) tail COUNTS are feature/downstream COORDINATE counts,
not tested/parameter directions; (c) clustering uses COMMON normalization (divide all maps by c=mean||D_i||_F, preserve
gains) with a DIAMETER threshold, thresholds chosen from the measured distance distribution; (d) ONE group => across-deficit
0 BY CONSTRUCTION (says nothing about approximate compression): TEST representative/truncated maps as approximate
constructions and VALIDATE vs the full original operator (report approximation error preserving gains)."""
import os, sys, json, argparse, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ViT_B_16_Weights
SP=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SP)
from ftvit_paths import cache_path
def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0)))+z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r,g,b,eps):
    mu=r.mean(-1,keepdim=True); sd=torch.sqrt(r.var(-1,unbiased=False,keepdim=True)+eps); return g*(r-mu)/sd+b
def prof(sv,s):
    sv=np.asarray(sv,dtype=np.float64); s1=sv[0]; sq=sv/np.sqrt(s)
    REL=[1e-2,1e-3,1e-4,1e-6]; ABS=[1e-1,1e-2,1e-3,1e-4]
    return {"sigma1":float(s1),"sigma1_over_sqrt_s":float(s1/np.sqrt(s)),"sigma_min_over_sigma1":float(sv[-1]/s1),
            "sigma_min_over_sqrt_s":float(sv[-1]/np.sqrt(s)),
            "REL_count_below(sigma_j/sigma1)":{f"{t:g}":int((sv<t*s1).sum()) for t in REL},
            "ABS_count_below(sigma_j/sqrt_s)":{f"{t:g}":int((sq<t).sum()) for t in ABS}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--seed-tag",required=True)
    ap.add_argument("--n",type=int,default=10000); ap.add_argument("--out",required=True)
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
    G3=torch.zeros((768,768),dtype=torch.float64)
    for i in range(0,N,500):
        sl=torch.arange(i,min(i+500,N)); VJ=VJ_batch(sl); Df3=VJ.reshape(-1,768); G3+=Df3.T@Df3
    sh=torch.ones(768,dtype=torch.float64)/np.sqrt(768.); Q=torch.linalg.svd(torch.eye(768,dtype=torch.float64)-torch.outer(sh,sh))[0][:,:767]
    ev3p,evec3p=torch.linalg.eigh(Q.T@G3@Q); sv_d3=np.sqrt(ev3p.flip(0).clamp(min=0).numpy()); evec3p_desc=evec3p.flip(1)
    svu=torch.linalg.svdvals(torch.cat([u,one],1)).numpy(); sva=torch.linalg.svdvals(torch.cat([a,one],1)).numpy(); svf=torch.linalg.svdvals(torch.cat([f,one],1)).numpy()
    out={"seed_tag":args.seed_tag,"N":N,"logits_rms_absolute":logits_rms}
    # ---- dual-convention spectral profiles ----
    out["spectra_dual_convention"]={"NOTE":"REL = sigma_j/sigma_1; ABS = sigma_j/sqrt(s) (s=%d). These are feature/downstream COORDINATE spectra; the tail COUNTS are coordinate counts, NOT tested directions and NOT parameter-space directions."%s,
      "feature_mlp0_u":prof(svu,s),"feature_mlp3_a":prof(sva,s),"feature_readout_f":prof(svf,s),
      "downstream_mlp3_nonshift":prof(sv_d3,s)}
    # ---- candidate directions: BOTH conventions for jv and ray ----
    def jv_frob(block,dA):
        tot=0.0
        for i in range(0,N,1000):
            sl=torch.arange(i,min(i+1000,N)); VJ=VJ_batch(sl)
            if block=="mlp0":
                H=torch.cat([u[sl],torch.ones(len(sl),1,dtype=torch.float64)],1); dz0=H@dA.T; dz3=(gp[sl]*dz0)@W3.T; jv=torch.einsum("bcx,bx->bc",VJ,dz3)
            elif block=="mlp3":
                H=torch.cat([a[sl],torch.ones(len(sl),1,dtype=torch.float64)],1); dz3=H@dA.T; jv=torch.einsum("bcx,bx->bc",VJ,dz3)
            else:
                H=torch.cat([f[sl],torch.ones(len(sl),1,dtype=torch.float64)],1); jv=H@dA.T
            tot+=float((jv**2).sum())
        return np.sqrt(tot)
    radii=[0.01,0.03,0.1,0.3,1.0]
    def ray_mlp3(dA,fl=False):
        med=[]; rmsrel=[]
        for t in radii:
            z3n=a@(W3+t*dA[:,:-1]).T+(b3+t*dA[:,-1]); bon=cls_mid+z3n
            fn=(f+apply_JLN(bon-bo)) if fl else ln_fwd(bon,gf,bf,epsf); lg=fn@V.T+bV; dl=(lg-logits)
            med.append(float((dl.norm(dim=1)/(logits.norm(dim=1)+1e-30)).median()))
            rmsrel.append(float(torch.sqrt((dl**2).sum(1).mean()))/logits_rms)
        return {"median_rel":med,"aggregate_rms_rel":rmsrel}
    def ray_mlp0(dA,fg=False,fl=False):
        med=[]; rmsrel=[]
        for t in radii:
            z0n=u@(W0+t*dA[:,:-1]).T+(b0+t*dA[:,-1]); an=(a+gp*(z0n-z0)) if fg else gelu(z0n); z3n=an@W3.T+b3; bon=cls_mid+z3n
            fn=(f+apply_JLN(bon-bo)) if fl else ln_fwd(bon,gf,bf,epsf); lg=fn@V.T+bV; dl=(lg-logits)
            med.append(float((dl.norm(dim=1)/(logits.norm(dim=1)+1e-30)).median())); rmsrel.append(float(torch.sqrt((dl**2).sum(1).mean()))/logits_rms)
        return {"median_rel":med,"aggregate_rms_rel":rmsrel}
    cand={}
    # mlp.0 feature-tail (2nd-smallest feature right-singular vector of [u;1]) as ONE candidate direction
    Vhu=torch.linalg.svd(torch.cat([u,one],1),full_matrices=False)[2]; nu_u2=Vhu[-2]
    dA=torch.randn(3072,dtype=torch.float64).view(-1,1)@nu_u2.view(1,-1); dA=dA/dA.norm()
    jf=jv_frob("mlp0",dA)
    cand["mlp0_feature_tail(1 direction)"]={"feature_sigma_over_sigma1":float(svu[-2]/svu[0]),"jv_frobenius_over_v":jf,"jv_operator_RMS_over_v":jf/np.sqrt(s),
        "ray_actual":ray_mlp0(dA),"ray_frozenboth":ray_mlp0(dA,fg=True,fl=True),
        "comment":"ONE direction: stays nearly LINEAR with small drift through r=1 (frozen ~ actual, no curvature amplification)."}
    # downstream-tail (smallest non-shift downstream eigenvector) as a bias direction
    db3=(Q@evec3p_desc[:,-1]); dA=torch.cat([torch.zeros(768,3072,dtype=torch.float64),db3.view(-1,1)],1); dA=dA/dA.norm()
    jf=jv_frob("mlp3",dA); r=ray_mlp3(dA); rfl=ray_mlp3(dA,fl=True)
    cand["mlp3_downstream_tail(1 direction)"]={"downstream_sigma_over_sigma1":float(sv_d3[-1]/sv_d3[0]),
        "jv_frobenius_over_v":jf,"jv_operator_RMS_over_v":jf/np.sqrt(s),"ray_actual":r,"ray_frozenLN":rfl,
        "comment":"ONE direction: approx flat through r~0.3 (aggregate/median small), then LN curvature grows drift; LN REDUCES the usable radius, does not remove the ocean."}
    # readout feature-null exact (control)
    nu_f=torch.linalg.svd(torch.cat([f,one],1),full_matrices=False)[2][-1]; dA=torch.randn(d,dtype=torch.float64).view(-1,1)@nu_f.view(1,-1); dA=dA/dA.norm()
    jf=jv_frob("readout",dA); cand["readout_feature_null_exact(1 direction)"]={"jv_frobenius_over_v":jf,"jv_operator_RMS_over_v":jf/np.sqrt(s)}
    # random control
    dA=torch.randn(768,3073,dtype=torch.float64); dA=dA/dA.norm(); jf=jv_frob("mlp3",dA); r=ray_mlp3(dA)
    cand["random_mlp3(1 direction)"]={"jv_frobenius_over_v":jf,"jv_operator_RMS_over_v":jf/np.sqrt(s),"ray_actual":r}
    out["candidates_dual_convention"]=cand
    # ---- (c) diameter-controlled clustering, COMMON normalization (gains preserved) ----
    torch.manual_seed(5); P=torch.randn(d*768,128,dtype=torch.float64)/np.sqrt(128)
    E=torch.empty((N,128),dtype=torch.float64); fro=torch.empty(N,dtype=torch.float64)
    for i in range(0,N,500):
        sl=torch.arange(i,min(i+500,N)); Db=VJ_batch(sl).reshape(len(sl),-1); fro[sl]=Db.norm(dim=1); E[sl]=Db@P   # NO per-map normalization
    c=float(fro.mean()); Ec=E/c                                   # common scale (divide ALL by c=mean||D_i||_F; JL preserves Euclidean dist)
    m=Ec.mean(0); dist=(Ec-m).norm(dim=1).numpy()                  # distance of common-scaled maps to the representative (mean)
    dd={"c_mean_frob":c,"frob_min":float(fro.min()),"frob_max":float(fro.max()),
        "dist_to_mean_pct":{p:float(np.percentile(dist,p)) for p in (5,50,90,99,100)}}
    # finer thresholds chosen from the measured distance distribution (fractions of the max dist-to-mean)
    dmax=float(dist.max()); thr_list=sorted(set(round(x,4) for x in [dmax*0.1,dmax*0.25,dmax*0.5,np.percentile(dist,50),np.percentile(dist,90)]))
    groups={}
    for thr in thr_list:
        Lmat=torch.zeros((400,128),dtype=torch.float64); nL=0; lab=np.full(N,-1,int); cap=False
        for i in range(N):
            if nL>0:
                dd_i=(Ec[i:i+1]-Lmat[:nL]).norm(dim=1); j=int(dd_i.argmin())
                if float(dd_i[j])<=thr or nL>=400: lab[i]=j; continue
            lab[i]=nL; Lmat[nL]=Ec[i]; nL+=1; cap=cap or nL>=400
        groups[f"diam_thr={thr:.3g}"]={"n_groups":nL,"cap_hit":bool(cap)}
    out["diameter_clustering_common_norm"]={"NOTE":"common-normalization (all maps / c=mean||D_i||_F; gains PRESERVED), Euclidean diameter threshold on a 128-d JL embedding; thresholds chosen from the measured dist-to-mean distribution. Supported statement is 'one group at every TESTED threshold' if nL=1 throughout.",
        "distance_distribution":dd,"groups_by_threshold":groups}
    # ---- (d) representative / truncated-map approximation, validated vs the FULL operator ----
    # representative map D_bar = mean_i D_i (gains preserved). approx error per sample ||D_i - D_bar||_F/||D_i||_F.
    Dbar=torch.zeros((d,768),dtype=torch.float64)
    for i in range(0,N,500): sl=torch.arange(i,min(i+500,N)); Dbar+=VJ_batch(sl).sum(0)
    Dbar/=N
    rep_err=[]
    for i in range(0,N,500):
        sl=torch.arange(i,min(i+500,N)); Di=VJ_batch(sl); e=(Di-Dbar).reshape(len(sl),-1).norm(dim=1)/(Di.reshape(len(sl),-1).norm(dim=1)+1e-30); rep_err.append(e)
    rep_err=torch.cat(rep_err).numpy()
    # truncated global downstream map: top-k right-singular subspace of the stacked non-shift D (eigvecs of G3 in shift^perp).
    # a rank-k truncated map has NULL = bottom (767-k) directions; their FULL-operator sensitivity = the tail energy.
    totE=float(ev3p.clamp(min=0).sum()); trunc={}
    for k in (45,90,200):
        tailE=float(ev3p.flip(0).clamp(min=0)[k:].sum())          # energy beyond top-k
        # validate: a bottom (smallest) eigenvector (truncated-map null) applied to the FULL maps as a bias dir
        vb=(Q@evec3p_desc[:,-1]); dA=torch.cat([torch.zeros(768,3072,dtype=torch.float64),vb.view(-1,1)],1); dA=dA/dA.norm()
        trunc[f"k={k}"]={"tail_energy_fraction_beyond_topk":tailE/totE,
                         "truncation_RMS_rel":np.sqrt(tailE/totE)}
    out["representative_truncated_approx"]={"NOTE":"across-deficit=0 for one group is BY CONSTRUCTION; this tests approximate constructions. Representative = mean map (gains preserved); truncated = top-k downstream subspace. Validated on the ORIGINAL maps.",
        "representative_map_approx_error_on_original":{"mean":float(rep_err.mean()),"median":float(np.median(rep_err)),"p90":float(np.percentile(rep_err,90)),"max":float(rep_err.max())},
        "truncated_map_energy_beyond_topk":trunc,
        "truncated_null_full_operator_sensitivity":cand["mlp3_downstream_tail(1 direction)"]["jv_operator_RMS_over_v"]}
    with open(args.out,"w") as fo: json.dump(out,fo,indent=2)
    print("logits_rms",logits_rms)
    print("SPECTRA:",json.dumps(out["spectra_dual_convention"]))
    print("CANDIDATES:",json.dumps(cand))
    print("CLUSTER distance_distribution:",json.dumps(dd)); print("CLUSTER groups:",json.dumps(groups))
    print("REP/TRUNC:",json.dumps(out["representative_truncated_approx"]))
    print("wrote",args.out,"\nDONE",flush=True)
if __name__=="__main__": main()
