"""ViT head-scope MULTILAYER branches: mlp.0, mlp.3, readout perturbed SIMULTANEOUSLY (2026-10-07).
Backbone through block11.ln_2 FROZEN; float64 on DEV (GPU by default, env VIT_DEV; exact LN centering). Builds on phase0_vit_geometry's exact nulls.

(A) CERTIFIED EXACT multilayer branches -- logits unchanged to machine precision at ALL radii (all orders), with the
    MECHANISM made explicit per layer:
      - feature-null-only branch (mlp0 (+)readout (+)mlp3-feature-null if r_ha<3073): each layer's perturbation leaves its
        OWN pre-activation unchanged (Delta z0 = Delta z3 = Delta logits = 0 layerwise) -> downstream sees identical input
        -> exact composition. This is LAYERWISE INPUT-PRESERVATION; it composes trivially across layers.
      - MIXED branch (mlp0 feature-null (+) mlp3 LN-shift downstream-null (+) readout feature-null): the LN-shift DOES change
        z3 (hence block-out bo) by a common additive vector, which the final LayerNorm removes exactly (Delta f = 0); mlp0 and
        readout contribute no change. Certification here is OUTPUT-preservation at the LN (Delta f = 0), not input-preservation
        at mlp3. Verified by Delta f norm + machine-precision logit drift at large radius.
(B) APPROXIMATE joint combination (mlp0 downstream-tail (+) mlp3 downstream-tail, COMPLETE FAMILY, exact nulls PROJECTED OUT
    of both in FULL [W|b] coords): each probe realizes dA0 = T0 @ randn(k0,769) and dA3 = T3 @ randn(k3,3073) (all input-augmented
    directions, not one), removes the mlp0 feature-null (769-aug nu_u) and the mlp3 LN-shift (all-ones output over 768), then
    unit-normalizes the joint. Joint finite-radius drift stores ACTUAL (full nonlinear), FROZEN_NETWORK (frozen GELU gate + frozen
    LN, FULL perturbed weights -> retains the dW0 x dW3 mixed weight-product), FIRST_ORDER (pure linearization), TRANSITION
    (|actual-frozen_network|, the gate-change beyond frozen weights) and JOINT_INTERACTION (|frozen_network-first_order|, the mixed
    weight-product). Matched controls: randsub (random orthonormal subspace of the SAME dim), generic (full random block); same
    exact-null projection + normalization applied. M=256 probes, INDEPENDENT per model seed. Reports TESTED-radius brackets at tol 1e-3.
Scope: sampled directions at ONE checkpoint vs matched controls; NOT uniform guarantees. Co-occurrence construction DISCONTINUED.
GPU: head weights + derived base tensors live on DEV (env VIT_DEV, default cuda); random directions drawn on a CPU generator then moved.
Writes results/runs/ftvit_resisc45/head_multilayer_{seed_tag}.json."""
import os, sys, json, argparse, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from ftvit_paths import cache_path
DEV = torch.device(os.environ.get("VIT_DEV","cuda") if torch.cuda.is_available() else "cpu")

def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0))) + z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r,g,b,eps):
    mu=r.mean(-1,keepdim=True); sd=torch.sqrt(r.var(-1,unbiased=False,keepdim=True)+eps); return g*(r-mu)/sd+b
def rank_tol(M,tol=1e-9):
    sv=torch.linalg.svdvals(M.double()); return int((sv>tol*sv[0]).sum()), sv
RGRID=[0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]
def bracket(drift,tol=1e-3):
    d=np.asarray(drift,float); cross=np.where(d>=tol)[0]
    if len(cross)==0: return {"last_safe":RGRID[-1],"first_cross":None,"censored":True}
    j=int(cross[0]); return {"last_safe":(RGRID[j-1] if j>0 else 0.0),"first_cross":RGRID[j],"censored":False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--seed-tag",required=True)
    ap.add_argument("--n",type=int,default=10000); ap.add_argument("--M",type=int,default=256); ap.add_argument("--out",required=True)
    args=ap.parse_args(); torch.manual_seed(0)
    ck=torch.load(args.ckpt,map_location="cpu"); ncls=ck["n_classes"]
    net=tvm.vit_b_16(weights=None); net.heads=nn.Sequential(); net.heads.add_module("head",nn.Linear(768,ncls))
    net.load_state_dict(ck["state_dict"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    blk=net.encoder.layers[11]
    cls_mid=torch.load(cache_path(f"clsmid_{args.seed_tag}_n{args.n}.pt"))["cls_mid"].double().to(DEV); N=cls_mid.shape[0]
    g2,b2=blk.ln_2.weight.double().to(DEV),blk.ln_2.bias.double().to(DEV); eps2=blk.ln_2.eps
    W0,b0=blk.mlp[0].weight.double().to(DEV),blk.mlp[0].bias.double().to(DEV)
    W3,b3=blk.mlp[3].weight.double().to(DEV),blk.mlp[3].bias.double().to(DEV)
    gf,bf=net.encoder.ln.weight.double().to(DEV),net.encoder.ln.bias.double().to(DEV); epsf=net.encoder.ln.eps
    V,bV=net.heads.head.weight.double().to(DEV),net.heads.head.bias.double().to(DEV); d=V.shape[0]

    def forward(dW0=0,db0=0,dW3=0,db3=0,dV=0,dbV=0):
        u=ln_fwd(cls_mid,g2,b2,eps2); z0=u@(W0+dW0).T+(b0+db0); a=gelu(z0); z3=a@(W3+dW3).T+(b3+db3)
        bo=cls_mid+z3; f=ln_fwd(bo,gf,bf,epsf); logits=f@(V+dV).T+(bV+dbV)
        return dict(u=u,z0=z0,a=a,z3=z3,bo=bo,f=f,logits=logits)
    S0=forward(); base=S0["logits"]; one=torch.ones(N,1,dtype=torch.float64,device=DEV)
    logits_rms=float(torch.sqrt((base**2).sum(1).mean()))
    def rel(lg): return float(((lg-base).norm(dim=1)/(base.norm(dim=1)+1e-30)).median())
    def rel_resid(a,b): return float((((a-b)).norm(dim=1)/(base.norm(dim=1)+1e-30)).median())   # per-token VECTOR residual ‖a-b‖ (NOT |‖a‖-‖b‖|), same normalization as rel

    r_hu,_=rank_tol(torch.cat([S0["u"],one],1)); r_ha,_=rank_tol(torch.cat([S0["a"],one],1)); r_hf,_=rank_tol(torch.cat([S0["f"],one],1))
    def null_vec(M): _,Sv,Vh=torch.linalg.svd(M.double(),full_matrices=False); return Vh[-1],Sv
    nu_u,_=null_vec(torch.cat([S0["u"],one],1)); nu_f,_=null_vec(torch.cat([S0["f"],one],1))
    H3=torch.cat([S0["a"],one],1); _,sv_a,Vh3=torch.linalg.svd(H3,full_matrices=False)
    nu_a=Vh3[-1] if r_ha<3073 else None
    def unit_block(*mats):
        nrm=torch.sqrt(sum((m**2).sum() for m in mats)); return [m/nrm for m in mats]

    exact={"r_hu":[r_hu,769],"r_ha":[r_ha,3073],"r_hf":[r_hf,769],
           "mlp0_feature_null":3072*(769-r_hu),"mlp3_feature_null":768*(3073-r_ha),
           "readout_feature_null":d*(769-r_hf),"mlp3_LNshift_downstream_null_genuine":r_ha}

    # ---------- (A) CERTIFIED EXACT MULTILAYER BRANCHES ----------
    torch.manual_seed(1)
    # per-layer exact-null generators (unit Frobenius within the layer)
    dA0=torch.randn(3072,dtype=torch.float64).to(DEV).view(-1,1)@nu_u.view(1,-1); dW0n,db0n=unit_block(dA0[:,:-1].contiguous(),dA0[:,-1].contiguous())
    dVn=torch.randn(d,dtype=torch.float64).to(DEV).view(-1,1)@nu_f.view(1,-1);     dV_,dbV_=unit_block(dVn[:,:-1].contiguous(),dVn[:,-1].contiguous())
    w=Vh3[:min(5,Vh3.shape[0])].sum(0); w=w/w.norm(); dA3s=torch.ones(768,1,dtype=torch.float64,device=DEV)@w.view(1,-1)
    dW3s,db3s=unit_block(dA3s[:,:-1].contiguous(),dA3s[:,-1].contiguous())  # LN-shift (exact downstream-null)
    if nu_a is not None:
        dA3f=torch.randn(768,dtype=torch.float64).to(DEV).view(-1,1)@nu_a.view(1,-1); dW3f,db3f=unit_block(dA3f[:,:-1].contiguous(),dA3f[:,-1].contiguous())

    EXT=[0.01,0.1,1.0,10.0,30.0,100.0]   # extended radii: exactness must hold at ALL orders, not just small r
    def branch(dW0=0,db0=0,dW3=0,db3=0,dV=0,dbV=0):
        rec={"radii":EXT,"rel_logit_drift":[],"layerwise":{}}
        for t in EXT:
            S=forward(t*dW0 if torch.is_tensor(dW0) else 0, t*db0 if torch.is_tensor(db0) else 0,
                      t*dW3 if torch.is_tensor(dW3) else 0, t*db3 if torch.is_tensor(db3) else 0,
                      t*dV if torch.is_tensor(dV) else 0, t*dbV if torch.is_tensor(dbV) else 0)
            rec["rel_logit_drift"].append(rel(S["logits"]))
        # layerwise preservation diagnostics at t=1 (||Δ·|| relative to the unperturbed quantity's RMS)
        S=forward(dW0 if torch.is_tensor(dW0) else 0, db0 if torch.is_tensor(db0) else 0,
                  dW3 if torch.is_tensor(dW3) else 0, db3 if torch.is_tensor(db3) else 0,
                  dV if torch.is_tensor(dV) else 0, dbV if torch.is_tensor(dbV) else 0)
        for q in ("z0","z3","f","logits"):
            num=float((S[q]-S0[q]).norm()); den=float(S0[q].norm())+1e-30; rec["layerwise"][f"rel_d{q}"]=num/den
        return rec
    exact_branches={}
    exact_branches["featurenull_mlp0+readout"]=branch(dW0=dW0n,db0=db0n,dV=dV_,dbV=dbV_)
    if nu_a is not None:
        exact_branches["featurenull_all3"]=branch(dW0=dW0n,db0=db0n,dW3=dW3f,db3=db3f,dV=dV_,dbV=dbV_)
    exact_branches["mixed_featurenull_mlp0+readout_plus_mlp3LNshift"]=branch(dW0=dW0n,db0=db0n,dW3=dW3s,db3=db3s,dV=dV_,dbV=dbV_)
    # negative control: a GENERIC multilayer perturbation (same 3 layers, random) -- must NOT be flat
    gW0,gb0=unit_block(torch.randn_like(W0),torch.randn_like(b0)); gW3,gb3=unit_block(torch.randn_like(W3),torch.randn_like(b3)); gV,gbV=unit_block(torch.randn_like(V),torch.randn_like(bV))
    exact_branches["CONTROL_generic_mlp0+mlp3+readout"]=branch(dW0=gW0/np.sqrt(3),db0=gb0/np.sqrt(3),dW3=gW3/np.sqrt(3),db3=gb3/np.sqrt(3),dV=gV/np.sqrt(3),dbV=gbV/np.sqrt(3))

    # ---------- (B) APPROXIMATE JOINT COMBINATION: mlp0 (+) mlp3 downstream-tails (COMPLETE FAMILY) ----------
    # downstream Gauss-Newton tails (reuse phase0f construction): G3 over z3->logits, G0 over z0->logits.
    u0,a0,bo0,f0=S0["u"],S0["a"],S0["bo"],S0["f"]
    rhat=(bo0-bo0.mean(-1,keepdim=True))/torch.sqrt(bo0.var(-1,unbiased=False,keepdim=True)+epsf)
    s_bo=torch.sqrt(bo0.var(-1,unbiased=False,keepdim=True)+epsf); gp=gelup(S0["z0"])
    def apply_JLN(delta):
        c=delta-delta.mean(-1,keepdim=True)-(((rhat*delta).sum(-1,keepdim=True))/768)*rhat; return (gf/s_bo)*c
    def VJ_batch(idx):
        Vg=V*gf; rsum=Vg.sum(1); rh=rhat[idx]; sb=s_bo[idx]
        Vgr=rh@Vg.T
        return ((Vg.unsqueeze(0)-rsum.view(1,-1,1)/768)-torch.einsum("bc,bm->bcm",Vgr,rh)/768)/sb.view(-1,1,1)
    G0=torch.zeros((3072,3072),dtype=torch.float64,device=DEV); G3=torch.zeros((768,768),dtype=torch.float64,device=DEV)
    for i in range(0,N,500):
        sl=torch.arange(i,min(i+500,N),device=DEV); VJ=VJ_batch(sl); G3+=VJ.reshape(-1,768).T@VJ.reshape(-1,768)
        D0=torch.einsum("bcx,xy->bcy",VJ,W3)*gp[sl].unsqueeze(1); G0+=D0.reshape(-1,3072).T@D0.reshape(-1,3072)
    ones768=torch.ones(768,dtype=torch.float64,device=DEV)                 # LN-shift OUTPUT direction (all-ones, unnormalized)
    onhat=ones768/np.sqrt(768.)
    Qsh=torch.linalg.svd(torch.eye(768,dtype=torch.float64,device=DEV)-torch.outer(onhat,onhat))[0][:,:767]  # LN-shift removed
    ev3,evec3=torch.linalg.eigh(Qsh.T@G3@Qsh); sv3=ev3.flip(0).clamp(min=0).sqrt(); V3=(Qsh@evec3.flip(1))
    ev0,evec0=torch.linalg.eigh(G0); sv0=ev0.flip(0).clamp(min=0).sqrt(); V0=evec0.flip(1)
    k0=int((sv0<1e-2*sv0[0]).sum()); k3=int((sv3<1e-2*sv3[0]).sum())
    T0=V0[:,3072-k0:]; T3=V3[:,767-k3:]              # downstream-tail z-space bases (mlp0: 3072, mlp3: 768)
    def rc(n,m,rng): return torch.randn(n,m,generator=rng,dtype=torch.float64).to(DEV)   # CPU draw (reproducible) -> DEV
    def mk_joint(mode,rng):
        # realize the COMPLETE downstream-tail family (ALL input-augmented directions), like phase0f dtail
        if mode=="tail":
            dA0=T0@rc(k0,769,rng); dA3=T3@rc(k3,3073,rng)
        elif mode=="randsub":                            # matched control: random orthonormal subspace of the SAME dim
            Q0=torch.linalg.qr(rc(3072,k0,rng))[0]; Q3=torch.linalg.qr(rc(768,k3,rng))[0]
            dA0=Q0@rc(k0,769,rng); dA3=Q3@rc(k3,3073,rng)
        else:                                            # matched control: full random block
            dA0=rc(3072,769,rng); dA3=rc(768,3073,rng)
        # project ALL exact nulls out of the FULL [W|b] coordinates so only APPROXIMATE content remains
        dA0=dA0-torch.outer(dA0@nu_u,nu_u)               # remove mlp0 feature-null (769-aug nu_u)
        dA3=dA3-torch.outer(ones768,ones768@dA3)/768.0   # remove mlp3 LN-shift (all-ones output)
        dW0,db0=dA0[:,:-1].contiguous(),dA0[:,-1].contiguous(); dW3,db3=dA3[:,:-1].contiguous(),dA3[:,-1].contiguous()
        nrm=torch.sqrt((dW0**2).sum()+(db0**2).sum()+(dW3**2).sum()+(db3**2).sum())   # global unit Frobenius
        return dW0/nrm,db0/nrm,dW3/nrm,db3/nrm
    def joint_drift(dW0,db0,dW3,db3):
        # actual (full nonlinear) | frozen_network (frozen gate+LN, FULL perturbed weights -> keeps dW0 x dW3)
        # | first_order (pure linearization) | transition = |actual-frozen_network| | joint_interaction = |frozen_network-first_order|
        actual=[]; frozen_network=[]; first_order=[]; transition=[]; joint_interaction=[]
        dz0=u0@dW0.T+db0; d3_from0_lin=(gp*dz0)@W3.T; dz3_direct=a0@dW3.T+db3   # first-order pieces (unit-dir)
        for t in RGRID:
            S=forward(t*dW0,t*db0,t*dW3,t*db3); La=S["logits"]; actual.append(rel(La))
            # frozen-network: freeze GELU gate (gp) and LN jacobian, but run the FULL perturbed weights;
            # aG depends on dW0 and is multiplied by (W3+t*dW3) -> RETAINS the dW0 x dW3 mixed weight-product
            aG=a0+t*(gp*dz0); z3F=aG@(W3+t*dW3).T+(b3+t*db3); dboF=(cls_mid+z3F)-bo0
            Lnet=(f0+apply_JLN(dboF))@V.T+bV; frozen_network.append(rel(Lnet))
            # first-order: pure linearization, ALL 2nd order dropped (old "frozenboth")
            dz3_tot=t*d3_from0_lin+t*dz3_direct
            Lfo=(f0+apply_JLN(dz3_tot))@V.T+bV; first_order.append(rel(Lfo))
            # VECTOR residuals (NOT |‖actual‖-‖frozen‖|): the logit-space distance between the models
            transition.append(rel_resid(La,Lnet))         # ‖actual - frozen_network‖ : network nonlinearity beyond frozen weights
            joint_interaction.append(rel_resid(Lnet,Lfo))  # ‖frozen_network - first_order‖ : the mixed dW0 x dW3 weight-product
        return {"actual":actual,"frozen_network":frozen_network,"first_order":first_order,
                "transition":transition,"joint_interaction":joint_interaction}
    def dist(x): x=np.asarray(x,float); return {"median":float(np.median(x)),"min":float(x.min()),"max":float(x.max())}
    approx={"tail_dims":{"mlp0_k":k0,"mlp3_k":k3}}
    MODESEED={"tail":101,"randsub":202,"generic":303}   # stable, process-independent (not hash())
    for mode in ("tail","randsub","generic"):
        rng=torch.Generator().manual_seed(2026+MODESEED[mode]+1000*int(args.seed_tag[1:]))   # INDEPENDENT per model seed
        ls=[]; per=[]
        for mi in range(args.M):
            dW0,db0,dW3,db3=mk_joint(mode,rng); cv=joint_drift(dW0,db0,dW3,db3); br=bracket(cv["actual"])
            ls.append(br["last_safe"]); per.append({"i":mi,"bracket":br,"drift":{kk:[float(x) for x in vv] for kk,vv in cv.items()}})
        approx[mode]={"last_safe_tested_tol1e-3":dist(ls),"per_direction":per}

    fam_desc=("complete downstream-tail family: dA0 = T0 @ N(0,1)[k0 x 769], dA3 = T3 @ N(0,1)[k3 x 3073]; "
              "T0 (3072 x k0), T3 (768 x k3) = small-singular (<1e-2 x top) downstream-tail bases of G0/G3; "
              "exact nulls projected out in FULL [W|b] coords (mlp0 feature-null nu_u over 769, mlp3 LN-shift all-ones output over 768); "
              "globally unit-Frobenius normalized. Controls: randsub = random orthonormal subspace of the SAME dim; generic = full random block.")
    out={"seed_tag":args.seed_tag,"N":N,"ncls":d,"M":args.M,"device":str(DEV),"logits_rms":logits_rms,"RGRID":RGRID,"EXT_radii":EXT,
         "exact_dims":exact,"exact_multilayer_branches":exact_branches,
         "approx_family":fam_desc,"approx_joint":approx,
         "val_acc":ck.get("val_acc"),"test_acc":ck.get("test_acc"),
         "CAVEATS":("EXACT branches certified by machine-precision logit drift at ALL radii + layerwise preservation diagnostics; "
                    "APPROX joint samples the COMPLETE downstream-tail family (not one input direction); "
                    "exact nulls projected out in FULL [W|b] coordinates (mlp0 feature-null over 769, mlp3 LN-shift all-ones output over 768); "
                    "FIRST-ORDER linearization vs FROZEN-NETWORK response distinguished (frozen_network retains the dW0 x dW3 mixed weight-product; first_order drops all 2nd order); "
                    "M=256 probes with INDEPENDENT per-seed draws; SAMPLED dirs at ONE checkpoint vs matched controls (randsub = same-dim random subspace, generic = full block), NOT uniform guarantees; "
                    "co-occurrence construction DISCONTINUED.")}
    with open(args.out,"w") as fo: json.dump(out,fo,indent=2)
    print("EXACT dims:",json.dumps({k:exact[k] for k in ("mlp0_feature_null","mlp3_feature_null","readout_feature_null","mlp3_LNshift_downstream_null_genuine")}))
    print("EXACT multilayer branches (rel ||Δlogits|| at",EXT,"):")
    for k,v in exact_branches.items(): print(f"  {k:48s} "+" ".join(f"{x:.1e}" for x in v["rel_logit_drift"])+f"  | layerwise {v['layerwise']}")
    print("APPROX joint last_safe (tol1e-3):",{m:approx[m]["last_safe_tested_tol1e-3"]["median"] for m in ("tail","randsub","generic")})
    print("wrote",args.out,"\nDONE",flush=True)

if __name__=="__main__": main()
