"""COMMON ResNet FAMILY STUDY (two-day plan 2026-11-22) -- one production group per invocation:

    python3 scripts/analysis/resnet_common_family_study.py l42fc   (layer4.2 conv1/2/3 + fc readout)
    python3 scripts/analysis/resnet_common_family_study.py l41     (layer4.1 conv1/2/3)
    python3 scripts/analysis/resnet_common_family_study.py l40     (layer4.0 conv1/2/3)

Operators per selected layer (reviewer corrections memo Sec. 2, uniformly across architectures), for the
layer edit A in R^(n x m), input patches p_(i,xi) (im2col at that conv; homogeneous coordinate ONLY for
fc, which has a bias; ResNet convs have none), and downstream preactivation derivatives D_(i,xi) through
the REAL remaining network (fixed eval-BN affines, residual paths incl. the 4.0 projection shortcut,
avgpool, fc; autograd at the base point, float64):

    H  = span{p_(i,xi)},    C = span{D_(i,xi)^T e_a},
    FN = {Z U_Hperp^T},     DN_additional = {U_Cperp Z U_H^T},    N = FN (+) DN_additional.

The downstream marginal Gram is sum_(i,xi) D^T D in ACTIVATION space (n x n), weight sharing respected;
it is NOT the parameter-space Gram, and ker of the full atomic map is NOT reported as DN. At the readout
D = I so DN_additional = 0 and N = FN (structural statement). Selective families use the exact ReLU gate
structure S_(i,xi) A p_(i,xi) = 0 with gates FIXED AT BASE: per output channel c the constraint is
row_c(A) _|_ p_(i,xi) for every (i,xi) with the channel's FOLLOWING base gate active (conv1 -> post-bn1
ReLU, conv2 -> post-bn2 ReLU, conv3 -> block-output ReLU on u3+s, with the bn scale folding; a channel
with bn scale exactly 0 is unconstrained). The complete selective-space computation is per-channel gated
patch ranks (census = candidate screen only); high-activity channels are certified full by the monotone
PSD subset certificate (lambda_min of a subset Gram lower-bounds the full active Gram) and escalated to
the full active Gram when the certificate fails.

Fixed conventions (plan "Fixed conventions"): rank 1e-9 relative + gap-10 (direct re-measurement of exact
null candidates; eigh-floor ambiguity bands recorded); absolute tails sigma/sqrt(N) <= 1e-3 in the
operator's own units (PRIMARY; the ||.||_X-relative reading recorded, never primary); 256 probes per
nonempty family + dimension-matched randsub + ONE shared generic control per layer, exact components
removed from approximate families AND controls; absolute grid 0.001..30, unit-Frobenius edits; absolute
and relative dataset-RMS drift separately (brackets at 1e-3 on the relative metric; median per-example
supplementary); paired vectors actual/frozen/Jv with per-example VECTOR residuals, using the VALIDATED
frozen forward (single-layer frozen map is affine -- validated -- so frozen == F0 + r*Jv exactly and
joint_interaction = ||frozen - first_order|| STRUCTURALLY 0; verified in fp64 on the validation subset);
parameter-to-output JVP rates of the realized edits reported separately (fp64); fp64 operators, fp32
battery forwards validated against fp64 on the validation subset (recorded). Empty families are valid
results and draw no probes.

Outputs (NEW files only): results/resnet/common_family_<group>.json + _arrays.pt.
Launch (one GPU, light-job class; see the contraction pilot's projection for runtime):
    run_light resnet_fam_<group> <cores> env CUDA_VISIBLE_DEVICES=<g> \
        python3 scripts/analysis/resnet_common_family_study.py <group>
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from flatocean.paths import ROOT as PROJECT_ROOT, DATA as DATA_ROOT_PATH
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(HERE, "scripts", "analysis"))
from resnet_common_lib import (CACHE_DIR, RESULTS_DIR, RGRID, TOL_REL, RANK_REL, RANK_GAP, RATE_ABS,
                               NCLS, GROUPS, METRICS, layer_label, layer_pname, authenticate_cache,
                               open_xin, xin_chunk, load_model, FrozenLayer4Head, states_to_block,
                               block_states, base_gates_of_states, downstream_g, conv_patches,
                               im2col_consistency, tail_from_z, shapes_of, delta_from_A,
                               rank_gap_record, bracket, dist, x_norm, meta_block, stable_seed, _aff)

EPS = np.finfo(np.float64).eps


# ------------------------------------------------------------------------------ operator accumulation
def conv_operator_pass(head, xin, b, j, chunk, dev, ref_local):
    """PASS 1 for one conv layer: patch Gram Gh (m x m), downstream marginal Gram Gd (n x n), gate census
    of the layer's following gate, all float64 over the full cache."""
    n_ex = xin.shape[0]
    nW = head.theta0[f"layer4.{b}.conv{j}.weight"]
    n, m = nW.shape[0], nW.shape[1] * nW.shape[2] * nW.shape[3]
    Gh = torch.zeros(m, m, dtype=torch.float64, device=dev)
    Gd = torch.zeros(n, n, dtype=torch.float64, device=dev)
    census = torch.zeros(n, dtype=torch.int64)
    npat = 0
    t0 = time.time()
    for i0 in range(0, n_ex, chunk):
        i1 = min(i0 + chunk, n_ex)
        xb = xin_chunk(xin, i0, i1, dev)
        st = block_states(head, states_to_block(head, xb, b), b)
        if i0 == 0:
            im2col_consistency(head, st, b, j)
        P = conv_patches(head, st, b, j)
        Pm = P.permute(0, 2, 1).reshape(-1, m)
        Gh += Pm.T @ Pm
        npat += Pm.shape[0]
        g, lg0 = downstream_g(head, st, b, j)
        Gd += (gm := g.permute(0, 1, 3, 2).reshape(-1, n)).T @ gm
        if i0 == 0:
            err = float((lg0 - ref_local[i0:i1].to(dev)).abs().max() / ref_local.abs().max())
            assert err <= 1e-12, f"base logits mismatch vs cache reference: {err:.3e}"
        gate = gate_of_conv(head, st, b, j)
        census += gate.sum((0, 2, 3)).to(torch.int64).cpu()
        del P, Pm, g, gm, st, xb
    return Gh, Gd, census, npat, time.time() - t0


def gate_of_conv(head, st, b, j):
    """The base gate FOLLOWING conv j (frozen-forward convention 1{u>0})."""
    g1, g2, go = base_gates_of_states(head, st, b)
    return (g1, g2, go)[j - 1]


def eig_desc(G):
    """CPU/LAPACK eigendecomposition (basis-reproducibility rule); returns (sv desc, V desc cols, floor)."""
    ev, V = torch.linalg.eigh(G.cpu())
    sv = np.sqrt(np.clip(ev.numpy()[::-1].copy(), 0, None))
    Vd = V.flip(1)
    floor = float(np.sqrt(EPS * max(float(ev.max()), 0.0)))      # eigh sigma noise floor estimate
    return sv, Vd, floor


def tail_info(sv, floor, n_ex, logits_rms, exact_null_dim):
    """T(ii) absolute tail (PRIMARY) + recorded relX reading, EXCLUDING the exact-null block (exact-
    component removal at basis level). Returns dict + index slices into the descending spectrum."""
    sqn = np.sqrt(n_ex)
    dim = sv.size
    n_nonexact = dim - exact_null_dim
    rates = sv / sqn
    k_abs_incl = int((rates <= RATE_ABS).sum())
    k_abs = max(0, min(k_abs_incl, n_nonexact) - 0)              # tail counted among ALL, minus exact below
    k_abs_excl = max(0, k_abs_incl - exact_null_dim)
    k_rel_incl = int((rates / max(logits_rms, 1e-300) <= RATE_ABS).sum())
    return {"sigma_rate_min": float(rates[-1]), "sigma_rate_max": float(rates[0]),
            "tail_dim_abs_incl_exact": k_abs_incl, "tail_dim_abs_PRIMARY": k_abs_excl,
            "tail_dim_relX_incl_exact": k_rel_incl,
            "tail_dim_relX_recorded": max(0, k_rel_incl - exact_null_dim),
            "sigma_rate_at_tail_edge": float(rates[dim - exact_null_dim - k_abs_excl - 1]) if k_abs_excl < n_nonexact else None,
            "eigh_sigma_floor": floor, "floor_over_cutoff": floor / max(RANK_REL * sv[0], 1e-300),
            "units_note": "sigma/sqrt(N), N=10000 EXAMPLES; feature- resp. activation-to-output rate "
                          "units of THIS operator (not a parameter-to-logit rate)"}


# --------------------------------------------------------------------------------- probe construction
def make_probe_sets(layer_rec, bases, n, m, M, group, lab, rng_tag="2026-11-22"):
    """Build the ordered probe-set descriptors for one layer. Factored sets carry (L, R, Z[M]); dense
    sets carry A[M] (CPU float64). Exact-component removal (project out FN and DN_additional) is applied
    to APPROXIMATE families and ALL controls; exact families are the removed objects themselves."""
    sets = []
    U_Hperp, U_H, U_Cperp = bases.get("U_Hperp"), bases.get("U_H"), bases.get("U_Cperp")
    have_FN = U_Hperp is not None and U_Hperp.shape[1] > 0
    have_DN = U_Cperp is not None and U_Cperp.shape[1] > 0

    def removal(A):
        """Project out the exact marginal null N = FN (+) DN_additional (batched over leading dims)."""
        if have_FN:
            A = A - (A @ U_Hperp) @ U_Hperp.T
        if have_DN:
            mid = (U_Cperp.T @ A) @ U_H
            A = A - U_Cperp @ mid @ U_H.T
        return A

    def seeded(mode_key):
        return torch.Generator().manual_seed(stable_seed(20261122, "resnet_common", group, lab, mode_key))

    def add_factored(name, famtag, L_, R_, dL, dR, remove):
        g = seeded(name)
        Z = torch.randn(M, dL, dR, generator=g, dtype=torch.float64)
        sets.append({"name": name, "family": famtag, "kind": "factored", "L": L_, "R": R_, "Z": Z,
                     "remove": remove, "seed": stable_seed(20261122, "resnet_common", group, lab, name)})

    def add_dense(name, famtag, builder, remove):
        g = seeded(name)
        A = torch.stack([builder(g) for _ in range(M)])
        sets.append({"name": name, "family": famtag, "kind": "dense", "A": A,
                     "remove": remove, "seed": stable_seed(20261122, "resnet_common", group, lab, name)})

    fams = layer_rec["families"]
    # exact families (no removal on themselves)
    if fams["FN_X"]["status"] == "computed":
        add_factored("FN_X", "FN_X", None, U_Hperp, n, U_Hperp.shape[1], False)
        g = seeded("FN_X_randsub")
        Q = torch.linalg.qr(torch.randn(m, U_Hperp.shape[1], generator=g, dtype=torch.float64))[0]
        add_factored("FN_X_randsub", "FN_X", None, Q, n, Q.shape[1], True)
    if fams["DN_X_additional"]["status"] == "computed":
        add_factored("DN_X", "DN_X_additional", U_Cperp, U_H, U_Cperp.shape[1], U_H.shape[1], False)
        g = seeded("DN_X_randsub")
        Q = torch.linalg.qr(torch.randn(n, U_Cperp.shape[1], generator=g, dtype=torch.float64))[0]
        add_factored("DN_X_randsub", "DN_X_additional", Q, U_H, Q.shape[1], U_H.shape[1], True)
    # approximate families (removal on family AND controls)
    if fams["FN_A"]["status"] == "computed":
        T = bases["H_tail"]
        add_factored("FN_A", "FN_A", None, T, n, T.shape[1], True)
        g = seeded("FN_A_randsub")
        Q = torch.linalg.qr(torch.randn(m, T.shape[1], generator=g, dtype=torch.float64))[0]
        add_factored("FN_A_randsub", "FN_A", None, Q, n, Q.shape[1], True)
    if fams["DN_A"]["status"] == "computed":
        T = bases["D_tail"]
        RH = None if (U_H is not None and U_H.shape[1] == m) else U_H
        dR = m if RH is None else RH.shape[1]
        add_factored("DN_A", "DN_A", T, RH, T.shape[1], dR, True)
        g = seeded("DN_A_randsub")
        Q = torch.linalg.qr(torch.randn(n, T.shape[1], generator=g, dtype=torch.float64))[0]
        add_factored("DN_A_randsub", "DN_A", Q, RH, Q.shape[1], dR, True)
    if fams["SEL_additional"]["status"] == "computed":
        chans = bases["SEL_channels"]                            # list of (c, B_add (m x d_c))

        def build_sel(g):
            A = torch.zeros(n, m, dtype=torch.float64)
            for c, Bc in chans:
                A[c] = Bc @ torch.randn(Bc.shape[1], generator=g, dtype=torch.float64)
            return A

        add_dense("SEL", "SEL_additional", build_sel, False)
        rs_bases = []
        g = seeded("SEL_randsub_bases")
        for c, Bc in chans:
            rs_bases.append((c, torch.linalg.qr(torch.randn(m, Bc.shape[1], generator=g,
                                                            dtype=torch.float64))[0]))

        def build_sel_rs(g):
            A = torch.zeros(n, m, dtype=torch.float64)
            for c, Bc in rs_bases:
                A[c] = Bc @ torch.randn(Bc.shape[1], generator=g, dtype=torch.float64)
            return A

        add_dense("SEL_randsub", "SEL_additional", build_sel_rs, True)
    if sets:                                                     # ONE shared generic control per layer
        add_dense("generic", "generic_control",
                  lambda g: torch.randn(n, m, generator=g, dtype=torch.float64), True)

    # materialize: apply exact-component removal + unit-Frobenius normalization. Removal of the FN part
    # acts on the RIGHT factor space alone, so factored sets stay factored; removal of the DN-additional
    # part mixes left and right factors, so when it actually binds the set is materialized dense.
    for s in sets:
        if s["kind"] == "factored" and s["remove"] and (have_FN or have_DN):
            if have_DN:
                A = dense_of(s, slice(0, M), m, n, torch.device("cpu"))
                s.update(kind="dense", A=removal(A))
                s.pop("Z", None), s.pop("L", None), s.pop("R", None)
            else:                                               # FN-only removal: project the right factor
                assert s["R"] is not None, "factored set with R=None requires an empty exact null"
                s["R"] = s["R"] - U_Hperp @ (U_Hperp.T @ s["R"])
        if s["kind"] == "factored":
            # normalize per probe: ||A||_F^2 = tr(Z^T (L^T L) Z (R^T R)) via the factored Gram
            Lf, Rf, Z = s["L"], s["R"], s["Z"]
            GL = None if Lf is None else (Lf.T @ Lf)
            GR = None if Rf is None else (Rf.T @ Rf)
            for k in range(Z.shape[0]):
                Zk = Z[k]
                Mq = Zk if GL is None else GL @ Zk
                Mq = Mq if GR is None else Mq @ GR
                s["Z"][k] = Zk / max(float(torch.sqrt((Zk * Mq).sum().clamp_min(0))), 1e-300)
        if s["kind"] == "dense":
            A = s["A"]
            if s["remove"]:
                A = removal(A)
            nrm = A.reshape(A.shape[0], -1).norm(dim=1).clamp_min(1e-300)
            s["A"] = (A / nrm.view(-1, 1, 1))
    return sets


def dense_of(s, kslice, m, n, dev, dtype=torch.float64):
    """Materialize probes kslice of a set as (K, n, m) on dev."""
    if s["kind"] == "dense":
        return s["A"][kslice].to(device=dev, dtype=dtype)
    Z = s["Z"][kslice].to(dev)
    A = Z if s["L"] is None else torch.einsum("al,klr->kar", s["L"].to(dev), Z)
    A = A if s["R"] is None else torch.einsum("kar,mr->kam", A, s["R"].to(dev))
    return A.to(dtype)


# ------------------------------------------------------------------------------------- battery engine
def battery_pass(head, head32, xin, b, j, sets, chunk, kblock, dev, ref_local, logits_rms, nval,
                 null_H=None, null_D=None):
    """One full data pass per conv layer: per-chunk fp64 states + 45-VJP g; per set: fp64 Jv (factored or
    K-route contraction), fp32 actual forwards over the radius grid, per-example drift/transition norms;
    optional direct re-measurement accumulators for exact-null candidate bases (null_H (m x k), null_D
    (n x k)); fp64 Jv vectors on the first nval examples for probes 0/1 of every set."""
    n_ex = xin.shape[0]
    nW = head.theta0[f"layer4.{b}.conv{j}.weight"]
    n, m = nW.shape[0], nW.shape[1] * nW.shape[2] * nW.shape[3]
    M = sets[0]["Z"].shape[0] if sets and sets[0]["kind"] == "factored" else (sets[0]["A"].shape[0] if sets else 0)
    nR = len(RGRID)
    base_norm = ref_local.norm(dim=1).clamp_min(1e-30)           # fp64 per-example base norms
    acc = {s["name"]: {"jv_sq": torch.zeros(M, dtype=torch.float64),
                       "jv_ex": torch.zeros(M, n_ex, dtype=torch.float32),
                       "act_rel": np.zeros((M, nR, n_ex), dtype=np.float32),
                       "tr_rel": np.zeros((M, nR, n_ex), dtype=np.float32),
                       "act_abs_sq": torch.zeros(M, nR, dtype=torch.float64),
                       "tr_abs_sq": torch.zeros(M, nR, dtype=torch.float64),
                       "jv_val_vec": torch.zeros(min(4, M), min(nval, n_ex), NCLS, dtype=torch.float64)}
           for s in sets}
    direct = {"H": torch.zeros(null_H.shape[1], dtype=torch.float64) if null_H is not None else None,
              "D": torch.zeros(null_D.shape[1], dtype=torch.float64) if null_D is not None else None}
    t0 = time.time()
    t_g = t_fwd = 0.0
    for i0 in range(0, n_ex, chunk):
        i1 = min(i0 + chunk, n_ex)
        B = i1 - i0
        xb = xin_chunk(xin, i0, i1, dev)
        st = block_states(head, states_to_block(head, xb, b), b)
        P64 = conv_patches(head, st, b, j)                       # (B, m, Lp)
        Pm64 = P64.permute(0, 2, 1).reshape(-1, m)
        Lp = P64.shape[2]
        tt = time.time()
        g, lg0 = downstream_g(head, st, b, j)                    # (45, B, n, Lp) fp64
        t_g += time.time() - tt
        if null_H is not None:
            direct["H"] += ((Pm64 @ null_H.to(dev)) ** 2).sum(0).cpu()
        if null_D is not None:
            gm = g.permute(0, 1, 3, 2).reshape(-1, n)
            direct["D"] += ((gm @ null_D.to(dev)) ** 2).sum(0).cpu()
            del gm
        F0c = ref_local[i0:i1].to(dev)                           # fp64 base reference (cache)
        st32 = {k: v.float() for k, v in st.items()}
        z32 = st32[f"z{j}"]
        F0c32 = tail_from_z(head32, b, j, z32, st32["s"])        # fp32 battery-path base (common-mode)
        for s in sets:
            name = s["name"]
            a = acc[name]
            # ---------- fp64 Jv for all M probes of this set ----------
            if s["kind"] == "factored":
                Rf, Lf, Z = s["R"], s["L"], s["Z"]
                q = Pm64 if Rf is None else Pm64 @ Rf.to(dev)    # (B*Lp, dR)
                q = q.reshape(B, Lp, -1)
                gh = g if Lf is None else torch.einsum("abnp,nl->ablp", g, Lf.to(dev))
                T = torch.einsum("ablp,bpr->balr", gh, q)        # (B, 45, dL, dR)
                Jv = torch.einsum("klr,balr->kba", Z.to(dev), T) # (M, B, 45)
                del q, gh, T
            else:
                Jv = torch.zeros(M, B, NCLS, dtype=torch.float64, device=dev)
                Pp = P64.permute(0, 2, 1)                        # (B, Lp, m)
                esub = max(1, int(2e8 // (NCLS * n * m)))        # K-route example subgroup size
                for e0 in range(0, B, esub):
                    e1 = min(e0 + esub, B)
                    K = torch.einsum("aenp,epm->eanm", g[:, e0:e1], Pp[e0:e1])
                    Kf = K.reshape((e1 - e0) * NCLS, n * m)
                    for k0 in range(0, M, kblock):
                        k1 = min(k0 + kblock, M)
                        Ab = s["A"][k0:k1].to(dev).reshape(k1 - k0, -1)
                        Jv[k0:k1, e0:e1] = (Kf @ Ab.T).reshape(e1 - e0, NCLS, k1 - k0).permute(2, 0, 1)
                    del K, Kf
            a["jv_sq"] += Jv.pow(2).sum((1, 2)).cpu()
            a["jv_ex"][:, i0:i1] = Jv.norm(dim=2).float().cpu()
            if i0 < a["jv_val_vec"].shape[1]:
                v1 = min(i1, a["jv_val_vec"].shape[1])
                a["jv_val_vec"][:, i0:v1] = Jv[:a["jv_val_vec"].shape[0], :v1 - i0].cpu()
            # ---------- fp32 battery over the radius grid ----------
            tt = time.time()
            for k0 in range(0, M, kblock):
                k1 = min(k0 + kblock, M)
                K = k1 - k0
                A32 = dense_of(s, slice(k0, k1), m, n, dev, torch.float32)
                Pm32 = Pm64.float()
                dz = torch.einsum("pm,knm->kpn", Pm32, A32)
                dz = dz.reshape(K, B, Lp, n).permute(0, 1, 3, 2).reshape(K, B, n, *st[f"z{j}"].shape[2:])
                JvK = Jv[k0:k1]
                for ri, r in enumerate(RGRID):
                    zb = (z32.unsqueeze(0) + np.float32(r) * dz).reshape(K * B, *z32.shape[1:])
                    lg = tail_from_z(head32, b, j, zb, st32["s"]).reshape(K, B, NCLS)
                    d_act = lg - F0c32.unsqueeze(0)
                    a["act_abs_sq"][k0:k1, ri] += d_act.double().pow(2).sum((1, 2)).cpu()
                    a["act_rel"][k0:k1, ri, i0:i1] = (d_act.norm(dim=2).double()
                                                      / base_norm[i0:i1].to(dev)).float().cpu().numpy()
                    frz = F0c.unsqueeze(0) + r * JvK             # frozen == F0 + r*Jv (validated affine)
                    d_tr = lg.double() - frz
                    a["tr_abs_sq"][k0:k1, ri] += d_tr.pow(2).sum((1, 2)).cpu()
                    a["tr_rel"][k0:k1, ri, i0:i1] = (d_tr.norm(dim=2)
                                                     / base_norm[i0:i1].to(dev)).float().cpu().numpy()
                    del zb, lg, d_act, frz, d_tr
                del A32, dz
            t_fwd += time.time() - tt
            del Jv
        del g, st, st32, P64, Pm64, xb
    timing = {"pass_s": time.time() - t0, "g_s": t_g, "fp32_forwards_s": t_fwd}
    return acc, direct, timing


def summarize_battery(sets, acc, n_ex, logits_rms, M):
    out = {}
    arrays = {}
    for s in sets:
        a = acc[s["name"]]
        jv_abs = np.sqrt(a["jv_sq"].numpy() / n_ex)              # ||Jv||_X per unit-Frobenius edit
        jv_rel = jv_abs / logits_rms
        act_abs = np.sqrt(a["act_abs_sq"].numpy() / n_ex)        # (M, nR) absolute dataset-RMS drift
        act_rel = act_abs / logits_rms
        act_med = np.median(a["act_rel"], axis=2)                # (M, nR) median per-example rel drift
        tr_abs = np.sqrt(a["tr_abs_sq"].numpy() / n_ex)
        tr_rel = tr_abs / logits_rms
        tr_med = np.median(a["tr_rel"], axis=2)
        frz_abs = np.outer(jv_abs, np.asarray(RGRID))            # frozen = F0 + r*Jv exactly (affine)
        frz_rel = frz_abs / logits_rms
        jv_ex = a["jv_ex"].numpy()                               # (M, N) per-example |Jv_i|
        # median over examples of r*|Jv_i|/|F0_i| = r * median(|Jv_i|/|F0_i|)
        med_ratio = np.median(jv_ex / np.maximum(acc["_base_norm"], 1e-30), axis=1)
        frz_med = np.outer(med_ratio, np.asarray(RGRID))
        store = np.stack([act_abs, act_rel, act_med, frz_abs, frz_rel, frz_med,
                          tr_abs, tr_rel, tr_med], axis=1)       # (M, 9, nR) == METRICS order
        per = []
        ls, fcs, cens = [], [], 0
        for k in range(M):
            br = bracket(act_rel[k], RGRID)
            ls.append(br["last_safe"])
            cens += int(br["censored"])
            if br["first_cross"] is not None:
                fcs.append(br["first_cross"])
            per.append({"i": k, "jv_rate_abs_X": float(jv_abs[k]), "jv_rate_relX": float(jv_rel[k]),
                        "bracket_rel1e-3": br})
        agg = {mname: {f"r{r:g}": dist(store[:, mi, ri]) for ri, r in enumerate(RGRID)}
               for mi, mname in enumerate(METRICS)}
        exceed = {f"r{r:g}": int((act_rel[:, ri] > TOL_REL).sum()) for ri, r in enumerate(RGRID)}
        out[s["name"]] = {"family": s["family"], "generator_seed": s["seed"], "M": M,
                          "kind": s["kind"],
                          "jv_rate_abs_X": dist(jv_abs), "jv_rate_relX": dist(jv_rel),
                          "last_safe_tested_rel_tol1e-3": dist(ls),
                          "first_cross_tested": (dist(fcs) if fcs else None),
                          "censored_count": cens,
                          "exceedance_count_actual_rel_gt_tol_among_M": exceed,
                          "aggregates_per_radius": agg, "per_direction": per,
                          "frozen_structural": "frozen = F0 + r*Jv exactly (single-layer frozen map is "
                                               "affine; validated); joint_interaction = ||frozen - "
                                               "first_order|| STRUCTURALLY 0"}
        arrays[s["name"] + "_drift"] = torch.from_numpy(store)
        arrays[s["name"] + "_jv_ex_norm"] = a["jv_ex"]
        arrays[s["name"] + "_jv_val_vec"] = a["jv_val_vec"]
    return out, arrays


# ----------------------------------------------------------------------------------------- validation
def validate_layer(head, head32, xin, b, j, sets, acc, ref_local, nval, chunk, dev, logits_rms,
                   n_jvp_probes=8):
    """fp64 validation on the first nval examples: (i) full-path fp64 actual + frozen forwards for probes
    {0,1} of every set at radii {0.03, 3}; frozen-vs-(F0+r*Jv) affinity residual; fp32-battery agreement
    on the reported metrics; (ii) independent torch.func.jvp (batched fp64 weight-tangent route) vs the
    contraction Jv on >= 8 probes. Returns (record, chunked output vectors for the arrays file)."""
    from torch.func import jvp as fjvp
    n_ex = xin.shape[0]
    nval = min(nval, n_ex)
    name_w = f"layer4.{b}.conv{j}.weight"
    Wshape = head.theta0[name_w].shape
    n, m = Wshape[0], Wshape[1] * Wshape[2] * Wshape[3]
    base_norm = ref_local.norm(dim=1).clamp_min(1e-30)
    rad = [0.03, 3.0]
    rec = {"nval": nval, "radii": rad, "per_set": {}}
    vecs = {}
    gates_cache = []
    for i0 in range(0, nval, chunk):
        i1 = min(i0 + chunk, nval)
        xb = xin_chunk(xin, i0, i1, dev)
        gts, _ = head.base_gates(xb)
        gates_cache.append((i0, i1, gts))
    for s in sets:
        a = acc[s["name"]]
        srec = {"probes": []}
        for kp in (0, 1):
            A = dense_of(s, slice(kp, kp + 1), m, n, dev)[0]
            delta_unit = delta_from_A(head, (b, j), A)
            Jv_val = a["jv_val_vec"][kp, :nval].to(dev)
            prec = {"probe": kp, "radii": {}}
            for r in rad:
                delta = {k: r * v for k, v in delta_unit.items()}
                act64 = torch.zeros(nval, NCLS, dtype=torch.float64)
                frz64 = torch.zeros(nval, NCLS, dtype=torch.float64)
                act32 = torch.zeros(nval, NCLS, dtype=torch.float64)
                for i0, i1, gts in gates_cache:
                    xb = xin_chunk(xin, i0, i1, dev)
                    act64[i0:i1] = head.actual(xb, delta).cpu()
                    frz64[i0:i1] = head.frozen(xb, gts, delta).cpu()
                    st32 = {k: v.float() for k, v in
                            block_states(head, states_to_block(head, xb, b), b).items()}
                    P32 = conv_patches(head, {k: v for k, v in st32.items()}, b, j)
                    dz = torch.einsum("bmp,nm->bnp", P32, A.float()).reshape(st32[f"z{j}"].shape)
                    act32[i0:i1] = tail_from_z(head32, b, j, st32[f"z{j}"] + np.float32(r) * dz,
                                               st32["s"]).double().cpu()
                F0v = ref_local[:nval]
                affin = float((frz64 - (F0v + r * Jv_val.cpu())).abs().max())
                d64 = act64 - F0v
                d32 = act32 - F0v
                m64 = {"abs_X": x_norm(d64), "relX": x_norm(d64) / logits_rms,
                       "med_rel": float((d64.norm(dim=1) / base_norm[:nval]).median())}
                m32 = {"abs_X": x_norm(d32), "relX": x_norm(d32) / logits_rms,
                       "med_rel": float((d32.norm(dim=1) / base_norm[:nval]).median())}
                tr64 = x_norm(act64 - frz64) / logits_rms
                prec["radii"][f"r{r:g}"] = {
                    "frozen_vs_F0_plus_rJv_max_abs": affin,
                    "actual_fp64": m64, "actual_fp32_batterypath": m32,
                    "fp32_vs_fp64_actual_vec_max_abs": float((act64 - act32).abs().max()),
                    "fp32_vs_fp64_relX_drift_diff": abs(m64["relX"] - m32["relX"]),
                    "transition_relX_fp64": tr64}
                if kp == 0:
                    vecs[f"{s['name']}_r{r:g}_actual64"] = act64.clone()
                    vecs[f"{s['name']}_r{r:g}_frozen64"] = frz64.clone()
            srec["probes"].append(prec)
        aff_max = max(pr["radii"][k]["frozen_vs_F0_plus_rJv_max_abs"]
                      for pr in srec["probes"] for k in pr["radii"])
        srec["affinity_max_abs"] = aff_max
        rec["per_set"][s["name"]] = srec
    rec["affinity_pass"] = bool(max(v["affinity_max_abs"] for v in rec["per_set"].values())
                                <= 1e-9 * max(1.0, float(ref_local.abs().max())))
    # ---- (ii) independent weight-tangent JVP route on a 128-example block ----
    i0, i1 = 0, min(128, nval, n_ex)
    xb = xin_chunk(xin, i0, i1, dev)
    st = block_states(head, states_to_block(head, xb, b), b)
    inp = st["x"] if j == 1 else st[f"a{j - 1}"]
    cfg = head.blocks[b]["conv_cfg"][j - 1]
    W0 = head.theta0[name_w]
    jvp_rows = []
    pick = [(si, k) for k in (0, 1, 2, 3) for si in range(len(sets))][:max(n_jvp_probes, 8)]
    for si, k in pick:
        s = sets[si]
        A = dense_of(s, slice(k, k + 1), m, n, dev)[0]

        def f_of_W(W):
            return tail_from_z(head, b, j, Fn.conv2d(inp, W, **cfg), st["s"])

        _, jv_auto = fjvp(f_of_W, (W0,), (A.reshape(W0.shape),))
        jv_con = acc[s["name"]]["jv_val_vec"][k, i0:i1].to(dev)
        err = float((jv_auto - jv_con).abs().max())
        sc = float(jv_auto.abs().max())
        jvp_rows.append({"set": s["name"], "probe": k, "max_abs_err": err,
                         "rel_err_own_scale": err / max(sc, 1e-300), "jv_scale": sc})
    # normalize against the LARGEST Jv scale among the picked probes: exact-null probes have jv_scale at
    # the fp64 roundoff floor, where a per-probe relative error is meaningless (agreement there is gated
    # in absolute terms against the layer's realized Jv scale)
    jv_ref = max((r["jv_scale"] for r in jvp_rows), default=1.0)
    for r in jvp_rows:
        r["err_over_layer_jv_scale"] = r["max_abs_err"] / max(jv_ref, 1e-300)
    rec["jvp_route_check"] = {
        "n_probes": len(jvp_rows), "rows": jvp_rows, "layer_jv_scale": jv_ref,
        "max_err_over_layer_jv_scale": max((r["err_over_layer_jv_scale"] for r in jvp_rows),
                                           default=None),
        "pass": bool(jvp_rows and all(r["err_over_layer_jv_scale"] <= 1e-12 for r in jvp_rows)),
        "note": "batched float64 forward-mode weight-tangent JVP (torch.func.jvp through conv2d + real "
                "downstream) vs the g-contraction Jv, first min(128, nval) examples; error normalized by "
                "the largest Jv scale among the picked probes"}
    return rec, vecs


# ----------------------------------------------------------------------------------- SEL computation
def sel_study(head, xin, b, j, census, sv_h, U_H, U_Hperp, chunk, dev, cap_mult=3):
    """Complete selective-space computation for one conv layer (see module docstring). Returns
    (record, contributing channel bases [(c, B_add)])."""
    n_ex = xin.shape[0]
    W = head.theta0[f"layer4.{b}.conv{j}.weight"]
    n, m = W.shape[0], W.shape[1] * W.shape[2] * W.shape[3]
    r_h = int(U_H.shape[1])
    sigma0 = float(sv_h[0])
    cut = RANK_REL * sigma0
    cert_thr = RANK_GAP * cut                                    # certificate margin: 10x the rank cutoff
    a_scale = head.blocks[b]["bn"][j - 1][0].detach().cpu().double()
    cnt = census.numpy()
    K_cap = cap_mult * m
    t0 = time.time()
    # group channels by census count (similar counts together -> early exit for high-activity groups)
    order = np.argsort(cnt)
    ng = max(1, int(4e9 // (8 * m * m)))                         # group size by Gram memory (~4 GB)
    groups = [order[i:i + ng] for i in range(0, n, ng)]
    chan = {}
    for c in range(n):
        chan[c] = {"n_act": int(cnt[c]), "bn_scale": float(a_scale[c])}
    esc = []                                                     # channels needing full active Gram
    for gi, grp in enumerate(groups):
        grp = [int(c) for c in grp if cnt[c] > 0 and a_scale[c] != 0.0]
        if not grp:
            continue
        G = torch.zeros(len(grp), m, m, dtype=torch.float64, device=dev)
        got = np.zeros(len(grp), dtype=np.int64)
        seen = np.zeros(len(grp), dtype=np.int64)
        need = np.array([min(cnt[c], K_cap) for c in grp])
        # deterministic STRIDE-SPREAD subset: every floor(n_act/K)-th active occurrence in data order,
        # so the subset samples the whole cache (a first-K prefix is poorly conditioned)
        stride = np.array([max(1, cnt[c] // K_cap) for c in grp])
        for i0 in range(0, n_ex, chunk):
            if (got >= need).all():
                break
            i1 = min(i0 + chunk, n_ex)
            xb = xin_chunk(xin, i0, i1, dev)
            st = block_states(head, states_to_block(head, xb, b), b)
            P = conv_patches(head, st, b, j).permute(0, 2, 1).reshape(-1, m)
            gate = gate_of_conv(head, st, b, j)
            gm = gate.permute(0, 2, 3, 1).reshape(-1, gate.shape[1])
            for ci, c in enumerate(grp):
                if got[ci] >= need[ci]:
                    continue
                idx = torch.nonzero(gm[:, c], as_tuple=False).squeeze(1)
                k_loc = int(idx.numel())
                if k_loc == 0:
                    continue
                sel = torch.arange((-seen[ci]) % stride[ci], k_loc, int(stride[ci]), device=idx.device)
                sel = sel[: int(need[ci] - got[ci])]
                seen[ci] += k_loc
                if sel.numel() > 0:
                    rows = P[idx[sel]]
                    G[ci] += rows.T @ rows
                    got[ci] += int(sel.numel())
            del P, gm, st, xb
        # certify or solve each channel of the group
        eye = torch.eye(m, dtype=torch.float64, device=dev)
        shifted = G - (cert_thr ** 2) * eye
        _, info = torch.linalg.cholesky_ex(shifted)
        info = info.cpu().numpy()
        for ci, c in enumerate(grp):
            full_cov = (cnt[c] <= K_cap)                         # subset == entire active set -> exact
            if (not full_cov) and info[ci] == 0:
                chan[c].update(route="subset_certificate", rank=r_h, additional=0,
                               cert_rows=int(got[ci]),
                               note="lambda_min(subset Gram) > (10*cutoff)^2: full gated rank certified "
                                    "by PSD monotonicity")
            else:
                if full_cov:
                    ev = torch.linalg.eigvalsh(G[ci].cpu())
                    sv_c = np.sqrt(np.clip(ev.numpy()[::-1].copy(), 0, None))
                    # rank against the LAYER-LEVEL cutoff (declared convention): sigma > cut
                    r_c = int((sv_c > cut).sum())
                    chan[c].update(route="exact_full_active_gram", rank=r_c,
                                   additional=max(0, r_h - r_c),
                                   gap_record={"sigma_at_rank": float(sv_c[r_c - 1]) if r_c > 0 else None,
                                               "sigma_below_rank": float(sv_c[r_c]) if r_c < m else None,
                                               "gap": float(sv_c[r_c - 1] / max(sv_c[r_c], 1e-300))
                                               if 0 < r_c < m else None})
                    if chan[c]["additional"] > 0:
                        chan[c]["_gram_ci"] = (gi, ci)
                        esc.append((c, G[ci].cpu()))
                else:
                    esc.append((c, None))                        # certificate failed: escalate
                    chan[c].update(route="escalate_full_gram")
        # keep only the Grams needed for null bases; free the rest
        del G, shifted
    # escalation pass: full active Grams for failing channels, in memory-bounded groups
    esc_full = [c for c, Gc in esc if Gc is None]
    for e0 in range(0, len(esc_full), ng):
        egrp = esc_full[e0:e0 + ng]
        G = torch.zeros(len(egrp), m, m, dtype=torch.float64, device=dev)
        for i0 in range(0, n_ex, chunk):
            i1 = min(i0 + chunk, n_ex)
            xb = xin_chunk(xin, i0, i1, dev)
            st = block_states(head, states_to_block(head, xb, b), b)
            P = conv_patches(head, st, b, j).permute(0, 2, 1).reshape(-1, m)
            gate = gate_of_conv(head, st, b, j)
            gm = gate.permute(0, 2, 3, 1).reshape(-1, gate.shape[1])
            for ci, c in enumerate(egrp):
                idx = torch.nonzero(gm[:, c], as_tuple=False).squeeze(1)
                if idx.numel():
                    rows = P[idx]
                    G[ci] += rows.T @ rows
            del P, gm, st, xb
        for ci, c in enumerate(egrp):
            ev = torch.linalg.eigvalsh(G[ci].cpu())
            sv_c = np.sqrt(np.clip(ev.numpy()[::-1].copy(), 0, None))
            r_c = int((sv_c > cut).sum())
            chan[c].update(route="escalated_full_active_gram", rank=r_c, additional=max(0, r_h - r_c))
            if chan[c]["additional"] > 0:
                esc.append((c, G[ci].cpu()))
        del G
    # dead / bn-zero channels
    for c in range(n):
        if a_scale[c] == 0.0:
            chan[c].update(route="bn_scale_zero", rank=0, additional=r_h,
                           mechanism="BN scale exactly 0: edit invisible regardless of gates")
        elif cnt[c] == 0:
            chan[c].update(route="dead_gate", rank=0, additional=r_h,
                           mechanism="following base gate never active on the 10k cache")
    # null bases for contributing channels
    contributing = []
    for c in range(n):
        d_add = chan[c].get("additional", 0)
        if d_add <= 0:
            continue
        if chan[c]["route"] in ("bn_scale_zero", "dead_gate"):
            B_add = U_H                                          # whole row span within span(H) (reference)
        else:
            Gc = next(Gc for cc, Gc in esc if cc == c and Gc is not None)
            ev, V = torch.linalg.eigh(Gc)
            svs = np.sqrt(np.clip(ev.numpy()[::-1].copy(), 0, None))
            Vd = V.flip(1)
            nullmask = torch.from_numpy((svs <= cut).copy())
            Bc = Vd[:, nullmask]
            if U_Hperp is not None and U_Hperp.shape[1] > 0:     # additional = null(H_c) minus FN part
                Bc = Bc - U_Hperp @ (U_Hperp.T @ Bc)
            Q, R = torch.linalg.qr(Bc)
            keep = (R.diagonal().abs() > 1e-10)
            B_add = Q[:, keep]
        contributing.append((c, B_add))
    dims = int(sum(chan[c].get("additional", 0) for c in range(n)))
    rec = {"constraint": "S_(i,xi) A p_(i,xi) = 0 with base gates (frozen-forward convention); per "
                         "channel c: row_c _|_ gated patch span H_c; additional-beyond-FN dim per "
                         "channel = r_h - rank(H_c); bn-scale-0 channels unconstrained",
           "rank_convention": f"sigma <= 1e-9 * sigma0(H) = {cut:.3e} (layer-level declared cutoff); "
                              f"subset certificate threshold 10x cutoff",
           "K_cap": int(K_cap), "cap_mult": cap_mult,
           "n_channels": n, "r_h": r_h,
           "n_dead_gate": int(sum(1 for c in range(n) if chan[c].get("route") == "dead_gate")),
           "n_bn_scale_zero": int(sum(1 for c in range(n) if chan[c].get("route") == "bn_scale_zero")),
           "n_subset_certified_full": int(sum(1 for c in range(n)
                                              if chan[c].get("route") == "subset_certificate")),
           "n_exact_full_gram": int(sum(1 for c in range(n)
                                        if chan[c].get("route", "").endswith("full_active_gram"))),
           "n_contributing": len(contributing),
           "additional_dim_total": dims,
           "census_min_active": int(cnt.min()), "census_median_active": float(np.median(cnt)),
           "census_max_active": int(cnt.max()),
           "elapsed_s": round(time.time() - t0, 1),
           "per_channel": {str(c): {k: v for k, v in chan[c].items() if not k.startswith("_")}
                           for c in range(n) if chan[c].get("additional", 0) > 0
                           or chan[c].get("route") in ("escalated_full_active_gram",)}}
    return rec, contributing, dims


# ---------------------------------------------------------------------------------------- fc readout
def fc_study(head, xin, chunk, dev, ref_local, logits_rms, M, group, n_ex):
    """Readout layer: augmented [W|b] operator; D = I structurally; exact affine battery (fp64, analytic:
    the fc edit changes logits exactly linearly; actual == frozen == first_order, transition == 0)."""
    t0 = time.time()
    pooled = torch.zeros(n_ex, 2048, dtype=torch.float64)
    for i0 in range(0, n_ex, chunk):
        i1 = min(i0 + chunk, n_ex)
        xb = xin_chunk(xin, i0, i1, dev)
        h = states_to_block(head, xb, 3)                         # through all three blocks
        pooled[i0:i1] = h.mean((2, 3)).cpu()
    aug = torch.cat([pooled, torch.ones(n_ex, 1, dtype=torch.float64)], 1)
    Gf = aug.T @ aug
    sv, Vd, floor = eig_desc(Gf)
    rk = rank_gap_record(sv)
    ti = tail_info(sv, floor, n_ex, logits_rms, rk["null_dim"])
    n, m = NCLS, 2049
    base_norm = ref_local.norm(dim=1).clamp_min(1e-30)
    lab = "readout_fc"
    fams = {"FN_X": {"status": "computed" if rk["null_dim"] > 0 else "genuine_zero",
                     "factor_null_dim": rk["null_dim"], "param_dim": n * rk["null_dim"],
                     "rank_record": rk},
            "DN_X_additional": {"status": "structurally_empty",
                                "statement": "readout D = I: C = R^45, C-perp = 0, DN_additional = 0, "
                                             "N = FN (reviewer correction Sec. 2)"},
            "FN_A": {"status": "computed" if ti["tail_dim_abs_PRIMARY"] > 0 else "genuine_zero",
                     "factor_dim_abs_PRIMARY": ti["tail_dim_abs_PRIMARY"],
                     "param_dim": n * ti["tail_dim_abs_PRIMARY"], "tail_info": ti},
            "DN_A": {"status": "structurally_empty", "statement": "D = I has no activation tail at any "
                                                                  "cut below 1 (identity singular values)"},
            "SEL_additional": {"status": "structurally_empty",
                               "statement": "linear readout: no downstream gate acts on the fc output"}}
    # probe sets (analytic battery)
    nullmask = sv <= rk["cutoff"]
    sets = []
    U_tail_excl = Vd[:, torch.from_numpy(((sv / np.sqrt(n_ex) <= RATE_ABS) & ~nullmask).copy())]
    U_null = Vd[:, torch.from_numpy(nullmask.copy())]

    def seeded(mode_key):
        return torch.Generator().manual_seed(stable_seed(20261122, "resnet_common", group, lab, mode_key))

    def mk(name, famtag, build, remove):
        g = seeded(name)
        A = torch.stack([build(g) for _ in range(M)])
        if remove and U_null.shape[1] > 0:
            A = A - torch.einsum("kni,ij->knj", A, (U_null @ U_null.T))
        A = A / A.reshape(M, -1).norm(dim=1).clamp_min(1e-300).view(-1, 1, 1)
        sets.append({"name": name, "family": famtag, "A": A,
                     "seed": stable_seed(20261122, "resnet_common", group, lab, name)})

    if fams["FN_X"]["status"] == "computed":
        mk("FN_X", "FN_X", lambda g: torch.randn(n, U_null.shape[1], generator=g,
                                                 dtype=torch.float64) @ U_null.T, False)
        g0 = seeded("FN_X_randsub_basis")
        Q = torch.linalg.qr(torch.randn(m, U_null.shape[1], generator=g0, dtype=torch.float64))[0]
        mk("FN_X_randsub", "FN_X", lambda g: torch.randn(n, Q.shape[1], generator=g,
                                                         dtype=torch.float64) @ Q.T, True)
    if fams["FN_A"]["status"] == "computed":
        mk("FN_A", "FN_A", lambda g: torch.randn(n, U_tail_excl.shape[1], generator=g,
                                                 dtype=torch.float64) @ U_tail_excl.T, True)
        g0 = seeded("FN_A_randsub_basis")
        Q = torch.linalg.qr(torch.randn(m, U_tail_excl.shape[1], generator=g0, dtype=torch.float64))[0]
        mk("FN_A_randsub", "FN_A", lambda g: torch.randn(n, Q.shape[1], generator=g,
                                                         dtype=torch.float64) @ Q.T, True)
    if sets:
        mk("generic", "generic_control", lambda g: torch.randn(n, m, generator=g, dtype=torch.float64),
           True)
    battery = {}
    arrays = {}
    aug_dev = aug.to(dev)
    for s in sets:
        A = s["A"]
        jv_ex = torch.zeros(len(A), n_ex, dtype=torch.float64)   # (M, N) per-example |A [f;1]|
        for k0 in range(0, len(A), 16):
            k1 = min(k0 + 16, len(A))
            dl = torch.einsum("knm,im->kin", A[k0:k1].to(dev), aug_dev)
            jv_ex[k0:k1] = dl.norm(dim=2).cpu()
            del dl
        jv_abs = (jv_ex.pow(2).mean(1)).sqrt().numpy()
        act_abs = np.outer(jv_abs, np.asarray(RGRID))
        act_rel = act_abs / logits_rms
        med_ratio = np.median((jv_ex / base_norm.unsqueeze(0)).numpy(), axis=1)
        act_med = np.outer(med_ratio, np.asarray(RGRID))
        zer = np.zeros_like(act_abs)
        store = np.stack([act_abs, act_rel, act_med, act_abs, act_rel, act_med, zer, zer, zer], axis=1)
        per = []
        ls, fcs, cens = [], [], 0
        for k in range(M):
            br = bracket(act_rel[k], RGRID)
            ls.append(br["last_safe"])
            cens += int(br["censored"])
            if br["first_cross"] is not None:
                fcs.append(br["first_cross"])
            per.append({"i": k, "jv_rate_abs_X": float(jv_abs[k]),
                        "jv_rate_relX": float(jv_abs[k] / logits_rms), "bracket_rel1e-3": br})
        battery[s["name"]] = {
            "family": s["family"], "generator_seed": s["seed"], "M": M, "kind": "dense_analytic",
            "jv_rate_abs_X": dist(jv_abs), "jv_rate_relX": dist(jv_abs / logits_rms),
            "last_safe_tested_rel_tol1e-3": dist(ls),
            "first_cross_tested": (dist(fcs) if fcs else None), "censored_count": cens,
            "exceedance_count_actual_rel_gt_tol_among_M":
                {f"r{r:g}": int((act_rel[:, ri] > TOL_REL).sum()) for ri, r in enumerate(RGRID)},
            "aggregates_per_radius": {mn: {f"r{r:g}": dist(store[:, mi, ri])
                                           for ri, r in enumerate(RGRID)}
                                      for mi, mn in enumerate(METRICS)},
            "per_direction": per,
            "frozen_structural": "fc edit: logits exactly affine in the edit (no downstream "
                                 "nonlinearity): actual == frozen == first_order EXACTLY; transition "
                                 "== 0 exactly (computed analytically in fp64)"}
        arrays[s["name"] + "_drift"] = torch.from_numpy(store)
        arrays[s["name"] + "_jv_ex_norm"] = jv_ex.float()
    rec = {"layer": lab, "n_out": n, "m_in_aug": m, "has_bias": True, "param_dim": n * m,
           "counting_floor_bound": {"value": max(0, n * m - n_ex * NCLS),
                                    "note": "max(0, p - s*d), uniform counting LOWER bound; not a "
                                            "computed kernel dimension"},
           "operator_H": {"source": "augmented [pooled;1] features, full 10k cache", "sigma0": rk["sigma0"],
                          "rank_record": rk, "tail_info": ti, "n_rows": n_ex,
                          "eigh_sigma_floor": floor},
           "operator_D": {"statement": "STRUCTURALLY D = I at the readout (no contraction computed)"},
           "families": fams, "battery": battery,
           "validation": {"statement": "exact affine layer: battery computed analytically in fp64 from "
                                       "the pooled features; actual==frozen==first_order is an identity, "
                                       "not a measurement; pooled features cross-checked against the "
                                       "cache base logits",
                          "pooled_logits_check_max_rel": float(
                              (aug[:, :-1] @ head.theta0["fc.weight"].T.cpu()
                               + head.theta0["fc.bias"].cpu() - ref_local).abs().max()
                              / ref_local.abs().max())},
           "elapsed_s": round(time.time() - t0, 1)}
    arrays["sv_H"] = torch.from_numpy(sv.copy())
    arrays["G_H"] = Gf
    return rec, arrays


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="COMMON ResNet family study (one group)")
    ap.add_argument("group", choices=list(GROUPS))
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--kblock", type=int, default=32)
    ap.add_argument("--nval", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--sel-cap-mult", type=int, default=6,
                    help="SEL subset-certificate size cap = mult * m spread-sampled active patches; "
                         "channels failing the certificate escalate to the full active Gram")
    ap.add_argument("--layers", default=None,
                    help="smoke only: comma list like '2.3,fc' restricting the group's layers")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = args.out or os.path.join(RESULTS_DIR, f"common_family_{args.group}.json")
    arr_path = out_path.replace(".json", "_arrays.pt")
    t_start = time.time()

    ident, auth = authenticate_cache(args.cache_dir)
    xin, n_ex = open_xin(args.cache_dir)
    ref_local = torch.load(os.path.join(args.cache_dir, ident["files"]["ref_local"]["name"]))
    logits_rms = x_norm(ref_local)
    net, ck = load_model(dev)
    head = FrozenLayer4Head(net, dtype=torch.float64, device=dev)
    head32 = FrozenLayer4Head(net, dtype=torch.float32, device=dev)

    out = {"study": f"COMMON ResNet family study, group {args.group} (seed-0 checkpoint, 10k cache)",
           "group": args.group, "N": n_ex, "M": args.M, "RGRID": RGRID, "tol_rel": TOL_REL,
           "cache_identity": {"n": ident["n"], "index_sha256": ident["index_sha256"],
                              "checkpoint_sha256": ident["checkpoint"]["sha256"],
                              "xin_sha256": ident["files"]["xin"]["sha256"],
                              "storage": ident["storage"]},
           "logits_rms_X": logits_rms,
           "quadruple_common": {
               "operators": "H = stacked im2col patches (feature units); C/D = stacked downstream "
                            "preactivation derivatives through the real remaining network (activation-"
                            "to-output units); rates sigma/sqrt(N), N=10000 examples",
               "normalization": "unit Frobenius norm of the layer's W (conv) resp. [W|b] (fc) edit; "
                                "||.||_X = dataset RMS in absolute logit units; relX = ||.||_X / "
                                "||F(theta0)||_X (separate metric, recorded separately)",
               "radius_grid": RGRID,
               "threshold": "T(i) exact: sigma <= 1e-9*sigma0, gap-10 recorded, direct re-measurement "
                            "of null candidates; T(ii) approx: ABSOLUTE rate sigma/sqrt(N) <= 1e-3 "
                            "PRIMARY (relX reading recorded); T(iii) visibility: relX drift <= 1e-3 "
                            "tested brackets only"},
           "precision": {"operators": "float64 (Grams accumulated on GPU fp64, eigendecomposition on "
                                      "CPU/LAPACK)", "jv": "float64 contraction, validated against "
                                      "torch.func.jvp (batched fp64) per layer",
                         "battery_actual": "float32 batched forwards from the edited layer through the "
                                           "real downstream, validated in fp64 on the validation subset "
                                           "(recorded per layer); frozen = F0 + r*Jv exactly (fp64)"},
           "layers": {}, "timing": {}}
    arrays = {"RGRID": torch.tensor(RGRID), "metric_axis": list(METRICS),
              "base_norm_per_example": ref_local.norm(dim=1)}

    layers = GROUPS[args.group]
    if args.layers:
        want = set(args.layers.split(","))
        layers = [L for L in layers if ("fc" if L == "fc" else f"{L[0]}.{L[1]}") in want]
    for L in layers:
        lab = layer_label(L)
        print(f"=== {lab} ===", flush=True)
        if L == "fc":
            rec, arrs = fc_study(head, xin, 2 * args.chunk, dev, ref_local, logits_rms, args.M,
                                 args.group, n_ex)
            out["layers"][lab] = rec
            for k, v in arrs.items():
                arrays[f"{lab}/{k}"] = v
            continue
        b, j = L
        n, m, kk, _ = shapes_of(head, L)
        t0 = time.time()
        Gh, Gd, census, npat, t_op = conv_operator_pass(head, xin, b, j, args.chunk, dev, ref_local)
        sv_h, Vh, floor_h = eig_desc(Gh)
        sv_d, Vdm, floor_d = eig_desc(Gd)
        rk_h = rank_gap_record(sv_h)
        rk_d = rank_gap_record(sv_d)
        # direct re-measurement of exact-null candidates is folded into the battery pass below;
        # bases first:
        null_h_mask = sv_h <= rk_h["cutoff"]
        null_d_mask = sv_d <= rk_d["cutoff"]
        U_H = Vh[:, torch.from_numpy((~null_h_mask).copy())]
        U_Hperp = Vh[:, torch.from_numpy(null_h_mask.copy())]
        U_C = Vdm[:, torch.from_numpy((~null_d_mask).copy())]
        U_Cperp = Vdm[:, torch.from_numpy(null_d_mask.copy())]
        ti_h = tail_info(sv_h, floor_h, n_ex, logits_rms, rk_h["null_dim"])
        ti_d = tail_info(sv_d, floor_d, n_ex, logits_rms, rk_d["null_dim"])
        # tail bases EXCLUDE the exact nulls (exact-component removal at basis level)
        sqn = np.sqrt(n_ex)
        h_tail_mask = ((sv_h / sqn <= RATE_ABS) & ~null_h_mask)
        d_tail_mask = ((sv_d / sqn <= RATE_ABS) & ~null_d_mask)
        H_tail = Vh[:, torch.from_numpy(h_tail_mask.copy())]
        D_tail = Vdm[:, torch.from_numpy(d_tail_mask.copy())]
        r_h, r_d = rk_h["rank"], rk_d["rank"]
        fams = {
            "FN_X": {"status": "computed" if rk_h["null_dim"] > 0 else "genuine_zero",
                     "factor_null_dim": rk_h["null_dim"], "param_dim": n * rk_h["null_dim"],
                     "definition": "FN = {Z U_Hperp^T}", "rank_record": rk_h},
            "DN_X_additional": {"status": "computed" if (rk_d["null_dim"] > 0 and r_h > 0)
                                else "genuine_zero",
                                "factor_null_dim": rk_d["null_dim"],
                                "param_dim": rk_d["null_dim"] * r_h,
                                "definition": "DN_additional = {U_Cperp Z U_H^T}",
                                "rank_record": rk_d},
            "FN_A": {"status": "computed" if int(h_tail_mask.sum()) > 0 else "genuine_zero",
                     "factor_dim_abs_PRIMARY": int(h_tail_mask.sum()),
                     "param_dim": n * int(h_tail_mask.sum()), "tail_info": ti_h},
            "DN_A": {"status": "computed" if int(d_tail_mask.sum()) > 0 else "genuine_zero",
                     "factor_dim_abs_PRIMARY": int(d_tail_mask.sum()),
                     "param_dim": int(d_tail_mask.sum()) * r_h, "tail_info": ti_d},
        }
        rec = {"layer": lab, "n_out": n, "m_in": m, "kernel": kk, "has_bias": False, "param_dim": n * m,
               "counting_floor_bound": {"value": max(0, n * m - n_ex * NCLS),
                                        "note": "max(0, p - s*d), uniform counting LOWER bound for the "
                                                "full atomic kernel; not a computed marginal dimension"},
               "operator_H": {"n_patches": npat, "sigma0": rk_h["sigma0"], "rank_record": rk_h,
                              "tail_info": ti_h, "eigh_sigma_floor": floor_h},
               "operator_D": {"n_rows": npat * NCLS, "sigma0": rk_d["sigma0"], "rank_record": rk_d,
                              "tail_info": ti_d, "eigh_sigma_floor": floor_d,
                              "note": "activation-space downstream marginal Gram sum D^T D (n x n), "
                                      "real remaining network, 45 VJPs per chunk"},
               "families": fams}
        # ---------------- SEL ----------------
        sel_rec, sel_chans, sel_dim = sel_study(head, xin, b, j, census, sv_h, U_H, U_Hperp,
                                                args.chunk, dev, args.sel_cap_mult)
        if sel_dim > 0:
            fams["SEL_additional"] = {"status": "computed", "dim": sel_dim, "record": sel_rec}
        else:
            fams["SEL_additional"] = {"status": "structurally_coincides",
                                      "coincides_with": "FN (every per-channel gated span numerically "
                                                        "full within span(H))",
                                      "dim": 0, "record": sel_rec}
        fams["N_marginal"] = {"dim": fams["FN_X"]["param_dim"] + fams["DN_X_additional"]["param_dim"],
                              "note": "deduplicated N = FN (+) DN_additional (exact marginal null)"}
        # ---------------- probes + battery ----------------
        bases = {"U_H": U_H, "U_Hperp": U_Hperp if rk_h["null_dim"] > 0 else None,
                 "U_Cperp": U_Cperp if rk_d["null_dim"] > 0 else None,
                 "H_tail": H_tail, "D_tail": D_tail, "SEL_channels": sel_chans}
        sets = make_probe_sets(rec, bases, n, m, args.M, args.group, lab)
        rec["probe_sets"] = [{"name": s["name"], "family": s["family"], "kind": s["kind"],
                              "seed": s["seed"]} for s in sets]
        if sets:
            null_H_dir = U_Hperp if rk_h["null_dim"] > 0 else None
            null_D_dir = U_Cperp if rk_d["null_dim"] > 0 else None
            acc, direct, t_bat = battery_pass(head, head32, xin, b, j, sets, args.chunk, args.kblock,
                                              dev, ref_local, logits_rms, args.nval,
                                              null_H=null_H_dir, null_D=null_D_dir)
            acc["_base_norm"] = ref_local.norm(dim=1).clamp_min(1e-30).numpy()
            battery, bat_arrays = summarize_battery(sets, acc, n_ex, logits_rms, args.M)
            rec["battery"] = battery
            # direct re-measurement record for exact-null candidates
            if direct["H"] is not None:
                ds = np.sqrt(direct["H"].numpy())
                rec["families"]["FN_X"]["direct_sigma_of_null_basis"] = {
                    "max": float(ds.max()), "max_over_cutoff": float(ds.max() / rk_h["cutoff"]),
                    "pass": bool(ds.max() <= rk_h["cutoff"]),
                    "note": "fresh-pass contraction sigma of each eigh null-basis column (direct "
                            "measurement; eigh floor cannot certify an exact null alone)"}
            if direct["D"] is not None:
                ds = np.sqrt(direct["D"].numpy())
                rec["families"]["DN_X_additional"]["direct_sigma_of_null_basis"] = {
                    "max": float(ds.max()), "max_over_cutoff": float(ds.max() / rk_d["cutoff"]),
                    "pass": bool(ds.max() <= rk_d["cutoff"]),
                    "note": "fresh-pass contraction sigma of each eigh null-basis column"}
            val_rec, val_vecs = validate_layer(head, head32, xin, b, j, sets, acc, ref_local,
                                               args.nval, args.chunk, dev, logits_rms)
            rec["validation"] = val_rec
            rec["timing"] = {"operator_pass_s": round(t_op, 1), **{k: round(v, 1)
                                                                   for k, v in t_bat.items()}}
            for k, v in bat_arrays.items():
                arrays[f"{lab}/{k}"] = v
            for k, v in val_vecs.items():
                arrays[f"{lab}/val/{k}"] = v
        else:
            rec["battery"] = {}
            rec["validation"] = {"statement": "no nonempty family at this layer: no probes drawn "
                                              "(empty families are valid results)"}
            rec["timing"] = {"operator_pass_s": round(t_op, 1)}
        arrays[f"{lab}/sv_H"] = torch.from_numpy(sv_h.copy())
        arrays[f"{lab}/sv_D"] = torch.from_numpy(sv_d.copy())
        arrays[f"{lab}/G_H"] = Gh.cpu()
        arrays[f"{lab}/G_D"] = Gd.cpu()
        arrays[f"{lab}/gate_census"] = census
        if H_tail.shape[1]:
            arrays[f"{lab}/H_tail_basis"] = H_tail
        if D_tail.shape[1]:
            arrays[f"{lab}/D_tail_basis"] = D_tail
        if rk_h["null_dim"]:
            arrays[f"{lab}/U_Hperp"] = U_Hperp
        if rk_d["null_dim"]:
            arrays[f"{lab}/U_Cperp"] = U_Cperp
        for c, Bc in sel_chans:
            if Bc is not U_H:                                    # dead/bn-zero channels: basis = U_H (derivable)
                arrays[f"{lab}/SEL_basis_ch{c}"] = Bc
        out["layers"][lab] = rec
        del Gh, Gd
        print(f"{lab}: r_h={r_h}/{m} r_d={r_d}/{n} FN={fams['FN_X']['param_dim']} "
              f"DNadd={fams['DN_X_additional']['param_dim']} ftail={fams['FN_A']['factor_dim_abs_PRIMARY']} "
              f"dtail={fams['DN_A']['factor_dim_abs_PRIMARY']} SELadd={sel_dim} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out["statistical_wording"] = ("exceedance counts among 256 probes; zero exceedances among 256 "
                                  "controls reported as such or as the plus-one estimate 1/257; no "
                                  "p-values")
    out["meta"] = meta_block("scripts/analysis/resnet_common_family_study.py",
                             {"device": str(dev), "chunk": args.chunk, "kblock": args.kblock,
                              "nval": args.nval, "elapsed_s": round(time.time() - t_start, 1),
                              "status": "computed; automated acceptance via "
                                        "scripts/pipeline/check_resnet.py"})
    torch.save(arrays, arr_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", out_path, "and", arr_path, f"\nDONE in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
