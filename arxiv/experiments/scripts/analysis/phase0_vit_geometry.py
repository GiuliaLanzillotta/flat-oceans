"""ViT Phase 0 (LIGHT): exact structural ocean + forward-only rays on the fine-tuned ViT-B/16 final MLP block + readout,
at the VALIDATION-SELECTED checkpoint. Backbone (through block11.ln_2) FROZEN; analyze mlp.0=[W0 b0], mlp.3=[W3 b3],
readout=[V bV]. All final-block linear algebra in float64 (LayerNorms applied in float64 -> exact centering).

EXACT structural invariances (loss- and nonlinearity-independent), reported SEPARATELY from first-order tangent content:
  (E1) readout feature-null       = d*(m1 - r_hf)         [head linear in its own params]
  (E2) mlp.0 feature-null         = n0*(m1u - r_hu)       [ln_2-fed input centering; preactivation-preserving]
  (E3) mlp.3 feature-null         = n3*(m1a - r_ha)
  (E4) mlp.3 LN-SHIFT downstream-null (NEW): directions ΔA3 = 1_768 ⊗ w^T give Δz3_i = (w·h_i)·1, a COMMON ADDITIVE
       SHIFT across the 768 channels of block_out; the final LayerNorm removes any common shift EXACTLY (LN(x+c1)=LN(x)
       for all c), so logits are unchanged exactly. Genuine (z3-changing) dim = r_ha; it is EXACT (all orders), distinct
       from the first-order J_LN rank-deficiency. mlp.0's analogous downstream path is NOT exact: the GELU between z0 and
       block_out breaks the shift structure -> first-order only (demonstrated by ray curvature).
First-order TANGENT content (GELU/LN): the large unresolved block Jacobian kernels are NOT computed; an unresolved large
Jacobian does NOT establish its kernel (or finite-flat subset) is small. Rays probe candidate tangent directions.
"""
import os, sys, json, argparse, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ViT_B_16_Weights
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from ftvit_paths import cache_path, data_path

def gelu(z): return 0.5*z*(1.0+torch.erf(z/np.sqrt(2.0)))
def gelup(z): return 0.5*(1+torch.erf(z/np.sqrt(2.0))) + z*torch.exp(-z**2/2)/np.sqrt(2*np.pi)
def ln_fwd(r, g, b, eps):
    mu = r.mean(-1, keepdim=True); sd = torch.sqrt(r.var(-1, unbiased=False, keepdim=True)+eps)
    return g*(r-mu)/sd + b
def ln_jac_batch(r, g, eps):
    N = r.shape[-1]; mu = r.mean(-1, keepdim=True); v = r.var(-1, unbiased=False, keepdim=True)+eps; s = torch.sqrt(v)
    rhat = (r-mu)/s
    I = torch.eye(N, dtype=r.dtype)
    # (B,N,N): g_i/s * (I - 11^T/N - rhat rhat^T /N)
    out = (g.view(1,N,1)/s.view(-1,1,1)) * (I.unsqueeze(0) - torch.ones(N,N,dtype=r.dtype)/N
            - torch.einsum("bi,bj->bij", rhat, rhat)/N)
    return out

def rank_tol(M, tol=1e-9):
    sv = torch.linalg.svdvals(M.double()); return int((sv > tol*sv[0]).sum()), sv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seed-tag", required=True)
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device); torch.manual_seed(0)
    ck = torch.load(args.ckpt, map_location="cpu"); ncls = ck["n_classes"]
    net = tvm.vit_b_16(weights=None)
    net.heads = nn.Sequential(); net.heads.add_module("head", nn.Linear(768, ncls))
    net.load_state_dict(ck["state_dict"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)

    # ---- extract class-token residual cls_mid (= x_mid[:,0]) in float32 on GPU; cache ----
    cache = cache_path(f"clsmid_{args.seed_tag}_n{args.n}.pt")
    blk = net.encoder.layers[11]
    if os.path.exists(cache):
        cls_mid = torch.load(cache)["cls_mid"]
    else:
        imgs = np.load(data_path("resisc45_train_imgs.npy"), mmap_mode="r")
        n = min(args.n, imgs.shape[0]); tfm = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        net32 = net.to(dev)
        outs = []; t0=time.time()
        with torch.no_grad():
            for i in range(0, n, 128):
                b = np.asarray(imgs[i:min(i+128,n)])
                x = torch.from_numpy(b).to(dev).float().div(255).permute(0,3,1,2).contiguous(); x = tfm(x)
                z = net32._process_input(x); m=z.shape[0]
                cls = net32.class_token.expand(m,-1,-1); z = torch.cat([cls,z],1); z = z+net32.encoder.pos_embedding
                for bl in net32.encoder.layers[:11]: z = bl(z)
                xa = blk.ln_1(z); att,_ = blk.self_attention(xa,xa,xa,need_weights=False)
                outs.append((z+att)[:,0,:].cpu())
        cls_mid = torch.cat(outs); torch.save({"cls_mid": cls_mid}, cache)
        print(f"  extracted {n} cls_mid in {time.time()-t0:.0f}s", flush=True)
    N = cls_mid.shape[0]
    # ---- everything below in float64 on CPU ----
    cls_mid = cls_mid.double().cpu()
    g2,b2 = blk.ln_2.weight.double().cpu(), blk.ln_2.bias.double().cpu(); eps2 = blk.ln_2.eps
    W0,b0 = blk.mlp[0].weight.double().cpu(), blk.mlp[0].bias.double().cpu()
    W3,b3 = blk.mlp[3].weight.double().cpu(), blk.mlp[3].bias.double().cpu()
    gf,bf = net.encoder.ln.weight.double().cpu(), net.encoder.ln.bias.double().cpu(); epsf = net.encoder.ln.eps
    V,bV = net.heads.head.weight.double().cpu(), net.heads.head.bias.double().cpu()
    d = V.shape[0]

    def forward(W0,b0,W3,b3,V,bV):
        u = ln_fwd(cls_mid, g2, b2, eps2)
        z0 = u@W0.T+b0; a = gelu(z0); z3 = a@W3.T+b3; bo = cls_mid+z3
        f = ln_fwd(bo, gf, bf, epsf); logits = f@V.T+bV
        return u,z0,a,z3,bo,f,logits
    u,z0,a,z3,bo,f,logits = forward(W0,b0,W3,b3,V,bV)

    one = torch.ones(N,1,dtype=torch.float64)
    r_hu,_ = rank_tol(torch.cat([u,one],1));  r_ha,_ = rank_tol(torch.cat([a,one],1));  r_hf,_ = rank_tol(torch.cat([f,one],1))
    exact = {
      "r_hu":[r_hu,769], "r_ha":[r_ha,3073], "r_hf":[r_hf,769],
      "mlp0_feature_null": 3072*(769-r_hu),
      "mlp3_feature_null": 768*(3073-r_ha),
      "readout_feature_null": d*(769-r_hf),
      "mlp3_LNshift_downstream_null_genuine": r_ha,     # ΔA3=1⊗w, w in rowspace(H3): changes z3, killed exactly by LN
      "mlp0_LNshift_downstream_null_exact": 0,          # GELU intervenes -> first-order only (verified by ray)
    }
    exact["mlp3_exact_total"] = exact["mlp3_feature_null"] + exact["mlp3_LNshift_downstream_null_genuine"]
    exact["mlp0_exact_total"] = exact["mlp0_feature_null"]
    exact["readout_exact_total"] = exact["readout_feature_null"]

    # input/feature null vectors (1-dim each here): smallest right singular vector of the aug feature matrix
    def null_vec(M):
        Uu,Sv,Vh = torch.linalg.svd(M.double(), full_matrices=False); return Vh[-1], Sv
    nu_u,_  = null_vec(torch.cat([u,one],1))   # 769, ⊥ rowspace(H0)
    nu_f,_  = null_vec(torch.cat([f,one],1))   # 769, ⊥ rowspace(Hf)
    # rowspace projector for H3 (to build w in rowspace): use top-r_ha right singular vectors
    H3 = torch.cat([a,one],1); _,_,Vh3 = torch.linalg.svd(H3, full_matrices=False)

    # ---------- RAYS (forward-only, matched absolute Frobenius radius) ----------
    radii = [0.01,0.03,0.1,0.3,1.0]
    base = logits
    def drift_for(dW0,db0,dW3,db3,dV,dbV):
        res = []
        for t in radii:
            _,_,_,_,_,_,lg = forward(W0+t*dW0,b0+t*db0,W3+t*dW3,b3+t*db3,V+t*dV,bV+t*dbV)
            res.append(float(((lg-base).norm(dim=1)/ (base.norm(dim=1)+1e-30)).median()))
        return res
    def unit_block(*mats):
        nrm = torch.sqrt(sum((m**2).sum() for m in mats)); return [m/nrm for m in mats]
    Z = {"W0":torch.zeros_like(W0),"b0":torch.zeros_like(b0),"W3":torch.zeros_like(W3),
         "b3":torch.zeros_like(b3),"V":torch.zeros_like(V),"bV":torch.zeros_like(bV)}
    rays = {}
    torch.manual_seed(1)
    # EXACT directions
    # (E4a) mlp.3 shift via bias: Δb3 = 1
    db3 = torch.ones(768,dtype=torch.float64); (db3,) = unit_block(db3)
    rays["EXACT_mlp3_shift_bias"] = drift_for(Z["W0"],Z["b0"],Z["W3"],db3,Z["V"],Z["bV"])
    # (E4b) mlp.3 shift via 1⊗w, w in rowspace(H3)
    w = Vh3[:min(5,Vh3.shape[0])].sum(0); w = w/ w.norm()          # a vector in rowspace(H3)
    dA3 = torch.ones(768,1,dtype=torch.float64) @ w.view(1,-1)     # 768 x 3073
    dW3s, db3s = dA3[:,:-1].contiguous(), dA3[:,-1].contiguous(); dW3s,db3s = unit_block(dW3s,db3s)
    rays["EXACT_mlp3_shift_w"] = drift_for(Z["W0"],Z["b0"],dW3s,db3s,Z["V"],Z["bV"])
    # (E2) mlp.0 feature-null: ΔA0 = a_rand ⊗ nu_u
    arand = torch.randn(3072,dtype=torch.float64); dA0 = arand.view(-1,1) @ nu_u.view(1,-1)  # 3072 x 769
    dW0n, db0n = dA0[:,:-1].contiguous(), dA0[:,-1].contiguous(); dW0n,db0n = unit_block(dW0n,db0n)
    rays["EXACT_mlp0_featurenull"] = drift_for(dW0n,db0n,Z["W3"],Z["b3"],Z["V"],Z["bV"])
    # (E1) readout feature-null: ΔV = b_rand ⊗ nu_f
    brand = torch.randn(d,dtype=torch.float64); dVn = brand.view(-1,1) @ nu_f.view(1,-1)       # d x 769
    dV_, dbV_ = dVn[:,:-1].contiguous(), dVn[:,-1].contiguous(); dV_,dbV_ = unit_block(dV_,dbV_)
    rays["EXACT_readout_featurenull"] = drift_for(Z["W0"],Z["b0"],Z["W3"],Z["b3"],dV_,dbV_)
    # GENERIC (matched radius) per block
    gW0 = torch.randn_like(W0); gb0 = torch.randn_like(b0); gW0,gb0 = unit_block(gW0,gb0)
    rays["GENERIC_mlp0"] = drift_for(gW0,gb0,Z["W3"],Z["b3"],Z["V"],Z["bV"])
    gW3 = torch.randn_like(W3); gb3 = torch.randn_like(b3); gW3,gb3 = unit_block(gW3,gb3)
    rays["GENERIC_mlp3"] = drift_for(Z["W0"],Z["b0"],gW3,gb3,Z["V"],Z["bV"])
    gV = torch.randn_like(V); gbV = torch.randn_like(bV); gV,gbV = unit_block(gV,gbV)
    rays["GENERIC_readout"] = drift_for(Z["W0"],Z["b0"],Z["W3"],Z["b3"],gV,gbV)
    # TANGENT (first-order kernel, restricted to mlp.0 bias b0; GELU curvature expected)
    # first-order dlogits_i wrt b0 = D0_i @ db0, D0_i = V J_LN(bo_i) W3 diag(gelu'(z0_i)) (d x 3072). Smallest right
    # singular vector of stacked D0 over a sample subset = approx first-order-kernel direction (bias-restricted).
    ns = min(2000, N)
    JL = ln_jac_batch(bo[:ns], gf, epsf)                 # ns x 768 x 768
    VJ = torch.einsum("ca,bax->bcx", V, JL)              # ns x d x 768   (V @ J_LN)
    D0 = torch.einsum("bcx,xy->bcy", VJ, W3)             # ns x d x 3072  (.. @ W3)
    D0 = D0 * gelup(z0[:ns]).unsqueeze(1)                # ns x d x 3072  (.. diag(gelu'))
    D0s = D0.reshape(ns*d, 3072)
    _,_,Vh0 = torch.linalg.svd(D0s, full_matrices=False)
    db0_fo = Vh0[-1]                                      # 3072, minimizes first-order output change (bias)
    (db0_fo,) = unit_block(db0_fo)
    rays["TANGENT_mlp0_firstorder_bias"] = drift_for(Z["W0"],db0_fo,Z["W3"],Z["b3"],Z["V"],Z["bV"])

    out = {"seed_tag":args.seed_tag,"N":N,"ncls":d,"radii":radii,"exact":exact,"rays":rays,
           "val_epoch":ck.get("epoch"),"val_acc":ck.get("val_acc"),"test_acc":ck.get("test_acc")}
    with open(args.out,"w") as fo: json.dump(out,fo,indent=2)
    print("EXACT:", json.dumps(exact))
    print("RAYS (median rel ||Δlogits|| at radii", radii, "):")
    for k,v in rays.items(): print(f"  {k:32s} " + " ".join(f"{x:.2e}" for x in v))
    print("wrote", args.out, "\nDONE", flush=True)

if __name__ == "__main__":
    main()
