"""
Geometry of kernel persistence, separated from later Jacobian amplification (plan amendment 2026-09-17).

For a source subspace X_t with orthonormal basis B (p x k) and the later kernel K_t' of the same block:

    Q(X_t, K_t')  = ||B_K'^T B||_F^2 / k          directed retained fraction (projected mass)
    L             = 1 - Q                          geometric loss
    d_lost        = ||(I - P_K') B||_F^2 = sum_i sin^2(theta_i)   effective lost dimension
    sin(theta_i)  = singular values of V_r'^T B    (V_r': right singular vectors of J_t' above the rank rule,
                                                    so I - P_K' = V_r' V_r'^T; sines are computed directly and
                                                    keep their accuracy at small angles)
    G             = ||J^_t' (I - P_K') B||_F / ||(I - P_K') B||_F,  J^ = J / sqrt(s)
                  = sqrt( sum_j sigma^_j^2 ||(V_r'^T B)_j||^2 / d_lost )   from the stored factor of J_t'.

d_lost is not the dimension of an intersection: two almost identical subspaces can intersect trivially.
G is reported next to the generic gain ||J^_t'||_F / sqrt(p) (RMS response of a random unit direction)
and sigma^_1(J_t'), and the lost mass is binned by the decade of sigma_j / sigma_1 of the later Jacobian.

Source subspaces per stage, all inside the kernel K_t of the stored factor:

    K_t                      full atomic kernel (declared rank rule 1e-9)
    C_t                      canonical dead-coordinate subspace (columns of always-inactive input units, rows of
                             always-inactive output units); C_t is contained in K_t
    S_t = K_t cap C_t^perp   kernel with the dead coordinates removed, P_S = P_K - P_C
    N_t = (C_l (x) H_l)^perp marginal null space (feature-null + downstream-null), contained in K_t
    D_t = K_t cap (C_l (x) H_l)   canonical co-occurrence-deficit subspace, P_D = P_K - P_N, dim = r_h r_d - rank J

Bases of S_t and D_t are eigenvectors of B^T P B (eigenvalues 0 or 1; the separation is recorded).  For a source
X in {K, S, D} the script also reports the split of its retained mass in the later kernel,
Q(S_t, K_t') = Q(S_t, C_t') + Q(S_t, S_t') and Q(D_t, K_t') = Q(D_t, N_t') + Q(D_t, D_t'), and the principal
angles to the later subspace of the same family (cosine based, floor 1e-7 from float32-stored bases).

Everything is computed on the CPU from the stored, audited factors of flatocean.analyses.persistence.

    python -m flatocean persistence-geometry run --seeds 0 1 2 --blocks 2 3 readout
    python -m flatocean persistence-geometry test
"""
# Ported unchanged from experiments/exp_persistence_geometry.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean persistence-geometry).
import argparse
import json
import os

import numpy as np
from flatocean import records, runs
import torch

from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.analyses import persistence as KP
from flatocean.runs import stage_epochs

OUT = KP.OUT                                  # geometry records (results root)
FACTOR_DIR = KP.FACTOR_DIR                    # stored factors (input root)
PROTOCOL = "EXPERIMENTS_PLAN.md amendment 2026-09-17, item 4; identifier persistence-geometry-v1"
VERSION = 3                        # 3 (2026-09-25): family splits carry a validity flag; pair records of an earlier version are recomputed on resume
LOSS_RESOLUTION = 1e-8             # effective lost dimension below this is numerically unresolved: float32-stored bases
                                   # give sin^2 errors of order 1e-14 per direction, 1e-10 over 10^4 directions; a gain or a
                                   # loss-share computed on less than 1e-8 of lost mass would describe rounding noise
RANK_TOL = KP.RANK_TOL
SIN2_EDGES = np.concatenate([[0.0], 10.0 ** np.arange(-16, 0.01, 0.25)])      # histogram of sin^2, first bin = below 1e-16
SEP_TOL = 1e-6                     # a split of K by a projector is resolved only when the eigenvalues of B^T P B separate into {0, 1} to
                                   # this tolerance (the projector then commutes with P_K on K); otherwise the "intersections" are not
                                   # subspaces of the kind claimed and the family is marked unresolved (numerical GELU kernels)
SIGMA_DECADES = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9]                     # lost mass by sigma_j/sigma_1 >= edge
PAIR_NAMES = [("high_accuracy", "mse_interpolation"), ("mse_interpolation", "epoch_100"), ("epoch_100", "epoch_250"),
              ("epoch_250", "late_endpoint"), ("high_accuracy", "late_endpoint"), ("mse_interpolation", "late_endpoint")]


# ----------------------------------------------------------------------------- subspace helpers
def span_basis(M, tol=RANK_TOL):
    """Orthonormal basis (columns) of the row space of M at the relative rank rule, and the rank."""
    _, sv, Vh = torch.linalg.svd(M, full_matrices=False)
    r = int((sv > tol * sv[0]).sum()) if sv.numel() and sv[0] > 0 else 0
    return Vh[:r].T.contiguous(), r


def split_by_projector(B, PB):
    """B: orthonormal basis of K (p x k).  PB = P B for an orthogonal projector P whose range R satisfies either
    R subset K (then P_K - P is a projector) or, more generally, P commutes with P_K.  Returns bases of
    K cap range(P) (eigenvalue 1 of B^T P B) and K cap ker(P) (eigenvalue 0), with the eigenvalue separation and the
    flag `resolved`: every eigenvalue within SEP_TOL of 0 or 1.  When the flag is false the two returned spaces are the
    eigenspaces above and below 1/2, not intersections, and the caller must report them as unresolved."""
    G = B.T @ PB; G = 0.5 * (G + G.T)
    ev, E = torch.linalg.eigh(G)
    one = ev > 0.5
    sep = {"max_zero_eigenvalue": float(ev[~one].abs().max()) if (~one).any() else 0.0,
           "min_one_eigenvalue": float(ev[one].min()) if one.any() else 1.0}
    sep["resolved"] = bool(sep["max_zero_eigenvalue"] <= SEP_TOL and sep["min_one_eigenvalue"] >= 1.0 - SEP_TOL)
    return B @ E[:, one], B @ E[:, ~one], sep


def angle_record(sines=None, cosines=None, k=None):
    """Summary of a principal-angle spectrum of a k-dimensional source.  Either the sines (exact at small angles)
    or the cosines (floor 1e-7 on sin^2 with float32-stored bases) of the min(k, k') defined angles are given;
    the remaining k - len angles are zero (sines given: the source has more dimensions than the complement of the
    target, so they lie inside the target) or 90 degrees (cosines given: the target is smaller than the source)."""
    if sines is not None:
        s2 = np.clip(np.asarray(sines, dtype=np.float64) ** 2, 0.0, 1.0)
        s2 = np.concatenate([s2, np.zeros(max(0, k - s2.size))])
    else:
        c2 = np.clip(np.asarray(cosines, dtype=np.float64) ** 2, 0.0, 1.0)
        s2 = np.concatenate([1.0 - c2, np.ones(max(0, k - c2.size))])
    s2 = np.sort(s2)[::-1][:k]
    d_lost = float(s2.sum())
    hist = np.histogram(s2, bins=np.concatenate([SIN2_EDGES, [1.0 + 1e-12]]))[0]
    big = s2 >= 0.5; resolved = d_lost >= LOSS_RESOLUTION
    return {"n_angles": int(k), "d_lost": d_lost, "L": d_lost / k if k else None, "loss_resolved": bool(resolved),
            "counts": {"sin2<1e-6": int((s2 < 1e-6).sum()), "1e-6<=sin2<1e-2": int(((s2 >= 1e-6) & (s2 < 1e-2)).sum()),
                       "1e-2<=sin2<0.5": int(((s2 >= 1e-2) & (s2 < 0.5)).sum()), "sin2>=0.5": int(big.sum())},
            "d_lost_share_from_sin2>=0.5": float(s2[big].sum() / d_lost) if resolved else None,
            "d_lost_share_from_sin2<1e-2": float(s2[s2 < 1e-2].sum() / d_lost) if resolved else None,
            "quantiles_sin2": {str(q): float(np.quantile(s2, q)) for q in (0.5, 0.9, 0.99, 0.999)} if k else None,
            "hist_sin2": hist.tolist()}, s2


def retained(Bsrc, Btgt):
    """Q(src, tgt) = ||Btgt^T Bsrc||_F^2 / dim src (0 for an empty target, None for an empty source)."""
    k = Bsrc.shape[1]
    if k == 0:
        return None
    if Btgt.shape[1] == 0:
        return 0.0
    return float((Btgt.T @ Bsrc).pow(2).sum() / k)


def coordinate_mass(Bsrc, coords):
    """Q(src, coordinate subspace)."""
    k = Bsrc.shape[1]
    if k == 0:
        return None
    return float(Bsrc[coords].pow(2).sum() / k) if len(coords) else 0.0


# ----------------------------------------------------------------------------- one stage
def run_name(a, seed):
    """<dataset>_<model>[_<tag>]_seed<seed>: the run and factor directory name (tag added 2026-09-25 for leaky / GELU)."""
    tag = getattr(a, "tag", "")
    return f"{a.dataset}_{a.model}{'_' + tag if tag else ''}_seed{seed}"


class Stage:
    """Kernel basis, later-Jacobian factor, dead coordinates and marginal spaces of one block at one checkpoint."""

    def __init__(self, a, seed, ep, blk):
        name = run_name(a, seed); rd = O.run_dir(a.dataset, a.model, seed, getattr(a, "tag", ""))
        run = O.load_run(a.dataset, a.model, seed, tag=getattr(a, "tag", ""), epoch=ep, device="cpu")
        sv, V, s, p, status = KP.load_factor(name, ep, blk, run, rd, a.allow_legacy, return_status=True)
        self.path = os.path.join(FACTOR_DIR, name, f"epoch{ep:04d}_{blk}.npz")
        self.sha = P.sha256_file(self.path); self.status = status
        self.identity_digest = KP.factor_identity(run, rd, ep, blk)["digest"]
        self.sv = sv; self.s = s; self.p = p; self.ep = ep; self.blk = str(blk)
        self.r = int((sv > RANK_TOL * sv[0]).sum())
        self.Vr = V[:, :self.r].contiguous(); self.B = V[:, self.r:].contiguous(); del V
        ctx = KP.BlockContext(run, blk, "cpu"); st = ctx.st
        self.dead_tau = float(getattr(a, "dead_tau", 0.0) or 0.0)
        dc = ctx.dead_coordinates(self.dead_tau)
        self.C = sorted(set(dc["dead_input_columns"]) | set(dc["dead_output_rows"]))
        # marginal spaces in the block's column-major coordinates (index k*n + j): vec -> (m1, n)
        if self.blk == "readout":
            H = st["H"][st["L"]]; self.n = st["d"]; self.PC = None
        else:
            l = int(self.blk) - 1; H = st["H"][l]; D = st["D"][l]; self.n = D.shape[2]
            BC, self.r_d = span_basis(D.reshape(-1, self.n)); self.PC = BC @ BC.T
        BH, self.r_h = span_basis(H); self.PH = BH @ BH.T; self.m1 = H.shape[1]
        if self.blk == "readout":
            self.r_d = self.n
        del ctx, st, run
        self._families()

    def marginal_project(self, M):
        """(P_H (x) P_C) M for the columns of M."""
        T = M.T.reshape(-1, self.m1, self.n)                  # k x m1 x n
        T = torch.einsum("ab,kbn->kan", self.PH, T)
        if self.PC is not None:
            T = torch.einsum("kan,nm->kam", T, self.PC)
        return T.reshape(M.shape[1], -1).T

    def _families(self):
        B = self.B; k = B.shape[1]
        self.fam = {"K": B}; self.sep = {}; self.resolved = {"S": True, "D": True}; self.unresolved_reason = {}
        if k == 0:
            self.fam.update({"S": B, "D": B, "Cb": B, "N": B}); return
        PB = torch.zeros_like(B); PB[self.C] = B[self.C]
        self.fam["Cb"], self.fam["S"], self.sep["S"] = split_by_projector(B, PB)      # K cap C (= C), K cap C^perp
        self.fam["D"], self.fam["N"], self.sep["D"] = split_by_projector(B, self.marginal_project(B))
        # validity of the splits: eigenvalue separation, and the dimension identities that hold for an exact kernel
        # (dim Cb = |C| since C is contained in K; dim N = p - r_h r_d and dim D = r_h r_d - rank J since the marginal
        # null space is contained in K); a failure marks the family unresolved and its pair statistics are not to be read
        # as intersections (independently thresholded numerical kernels, e.g. GELU late in training)
        checks = {"S": [("eigenvalue separation", self.sep["S"]["resolved"]), ("dim (K cap C) = |C|", self.fam["Cb"].shape[1] == len(self.C))],
                  "D": [("eigenvalue separation", self.sep["D"]["resolved"]), ("dim N = p - r_h r_d", self.fam["N"].shape[1] == self.p - self.r_h * self.r_d),
                        ("dim D = r_h r_d - rank J", self.fam["D"].shape[1] == self.r_h * self.r_d - self.r)]}
        for f, cs in checks.items():
            failed = [c for c, ok in cs if not ok]
            self.resolved[f] = not failed
            if failed:
                self.unresolved_reason[f] = failed

    def record(self):
        dims = {f: int(b.shape[1]) for f, b in self.fam.items()}
        return {"epoch": self.ep, "rank": self.r, "dims": dims, "dim_C_coordinates": len(self.C), "dead_tau": self.dead_tau, "r_h": self.r_h, "r_d": self.r_d,
                "deficit_from_ranks": self.r_h * self.r_d - self.r, "marginal_null_from_ranks": self.p - self.r_h * self.r_d,
                "eigenvalue_separation": self.sep, "split_resolved": self.resolved, "split_unresolved_reason": self.unresolved_reason,
                "source_status": self.status, "source_factor_sha256": self.sha,
                "identity_digest": self.identity_digest,
                "sigma1_hat": float(self.sv[0] / np.sqrt(self.s)),
                "generic_gain": float(np.sqrt((self.sv ** 2).sum() / self.s / self.p))}


def pair_record(A, Bt, spectra, tag):
    """A: earlier Stage, Bt: later Stage."""
    rec = {"epochs": [A.ep, Bt.ep], "families": {}}
    svh = torch.from_numpy(Bt.sv[:Bt.r] / np.sqrt(Bt.s))
    for f in ("K", "S", "D"):
        X = A.fam[f]; k = X.shape[1]
        if k == 0:
            rec["families"][f] = {"dim_source": 0, "note": "empty source, undefined"}; continue
        out = {"dim_source": k, "dim_later_kernel": int(Bt.B.shape[1]), "dim_later_same_family": int(Bt.fam[f].shape[1])}
        if f != "K" and not (getattr(A, "resolved", {}).get(f, True) and getattr(Bt, "resolved", {}).get(f, True)):
            out["unresolved"] = {"earlier": getattr(A, "unresolved_reason", {}).get(f), "later": getattr(Bt, "unresolved_reason", {}).get(f),
                                 "meaning": "the family split of at least one endpoint is not an intersection (eigenvalue separation or dimension identity failed); the numbers below are eigenspace statistics, not persistence of a subspace of this kind"}
        M = Bt.Vr.T @ X                                          # r' x k, (I - P_K') X in the basis V_r'
        sines = torch.linalg.svdvals(M).numpy()
        ang, s2 = angle_record(sines=sines, k=k); spectra[f"{tag}_{f}_to_K_sin2"] = s2
        out["to_later_kernel"] = dict(ang, Q=1.0 - ang["L"], Q_check_from_kernel_basis=retained(X, Bt.B))
        row_mass = M.pow(2).sum(1)                               # lost mass per later singular direction
        d_lost = float(row_mass.sum())
        if d_lost >= LOSS_RESOLUTION:
            G = float(torch.sqrt((svh ** 2 * row_mass).sum() / d_lost))
            rel = Bt.sv[:Bt.r] / Bt.sv[0]
            out["amplification"] = {"G": G, "G_over_sigma1": G / float(svh[0]), "G_over_generic_gain": G / Bt.record()["generic_gain"],
                                    "lost_mass_share_by_sigma_rel": {f">={e:g}": float(row_mass[torch.from_numpy(rel >= e)].sum() / d_lost) for e in SIGMA_DECADES}}
        else:
            out["amplification"] = None
            out["amplification_undefined_reason"] = f"effective lost dimension {d_lost:.3g} below the resolution {LOSS_RESOLUTION:g}"
        Y = Bt.fam[f]
        if f != "K":
            if Y.shape[1]:
                ang2, s2b = angle_record(cosines=torch.linalg.svdvals(Y.T @ X).numpy(), k=k)
            else:
                ang2, s2b = angle_record(cosines=np.zeros(0), k=k)
            spectra[f"{tag}_{f}_to_{f}_sin2"] = s2b
            out["to_later_same_family"] = dict(ang2, Q=1.0 - ang2["L"])
            other = "Cb" if f == "S" else "N"
            out["retained_mass_split"] = {"in_later_same_family": retained(X, Y), f"in_later_{'dead_coordinates' if f == 'S' else 'marginal_null'}": retained(X, Bt.fam[other])}
        rec["families"][f] = out
    return rec


def run(a):
    torch.set_num_threads(a.threads)
    for seed in a.seeds:
        rd = O.run_dir(a.dataset, a.model, seed, getattr(a, "tag", "")); stages = stage_epochs(rd)
        name = run_name(a, seed); tau = float(getattr(a, "dead_tau", 0.0) or 0.0); suffix = f"_tau{tau:g}" if tau else ""
        out_path = os.path.join(OUT, f"{name}_geometry{suffix}.json"); npz_path = os.path.join(OUT, f"{name}_geometry{suffix}_spectra.npz")
        res = json.load(open(out_path)) if (a.resume and os.path.exists(out_path)) else {
            "protocol": PROTOCOL, "version": VERSION, "code_revision": KP.git_rev(), "run": rd, "stages": stages, "dead_tau": tau,
            "rank_tolerance": RANK_TOL, "sin2_hist_edges": SIN2_EDGES.tolist(), "blocks": {}}
        if res.get("dead_tau", 0.0) != tau:
            raise RuntimeError(f"{out_path} was made with dead_tau {res.get('dead_tau', 0.0)}, requested {tau}")
        if res.get("version") != VERSION:                                        # a resumed file is upgraded: every pair is recomputed below
            res["upgraded_from"] = {"version": res.get("version"), "code_revision": res.get("code_revision")}
            res["version"] = VERSION; res["code_revision"] = KP.git_rev()
        spectra = dict(np.load(npz_path)) if (a.resume and os.path.exists(npz_path)) else {}
        pairs = [(stages[x]["checkpoint_epoch"], stages[y]["checkpoint_epoch"], f"{x}->{y}") for x, y in PAIR_NAMES]
        for blk in a.blocks:
            bres = res["blocks"].setdefault(str(blk), {"stages": {}, "pairs": {}})
            cache = {}
            def stage(ep):
                if ep not in cache:
                    print(f"== {name} block {blk} epoch {ep}: loading", flush=True)
                    cache[ep] = Stage(a, seed, ep, blk); bres["stages"][str(ep)] = cache[ep].record()
                    print(f"   dims {bres['stages'][str(ep)]['dims']}  deficit(ranks) {bres['stages'][str(ep)]['deficit_from_ranks']}  sep {cache[ep].sep}", flush=True)
                return cache[ep]
            for (e1, e2, label) in pairs:
                if label in bres["pairs"] and bres["pairs"][label].get("source_factor_sha256") == [stage(e1).sha, stage(e2).sha] and bres["pairs"][label].get("validation_version") == VERSION:
                    continue
                A, Bt = stage(e1), stage(e2)
                rec = pair_record(A, Bt, spectra, f"b{blk}_{e1}_{e2}")
                rec["source_factor_sha256"] = [A.sha, Bt.sha]; rec["identity_digests"] = [A.identity_digest, Bt.identity_digest]; rec["validation_version"] = VERSION
                bres["pairs"][label] = rec
                kk = rec["families"]["K"]
                if "to_later_kernel" in kk:
                    print(f"   {label}: Q {kk['to_later_kernel']['Q']:.4f} d_lost {kk['to_later_kernel']['d_lost']:.1f} "
                          f"G/generic {kk['amplification']['G_over_generic_gain'] if kk['amplification'] else None}", flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)
                np.savez_compressed(npz_path, **spectra)
                for ep in list(cache):                              # keep only stages still needed
                    if not any(ep in (x, y) for (x, y, lab) in pairs if lab not in bres["pairs"]):
                        del cache[ep]
        print("saved", out_path, flush=True)


# ----------------------------------------------------------------------------- tests
def _tests():
    g = torch.Generator().manual_seed(0); p = 60
    Qm, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=torch.float64))
    # identical, nested, rotated by a known angle, orthogonal
    K1 = Qm[:, :10]
    # a projector that does not commute with P_K: the eigenvalues of B^T P B are strictly between 0 and 1 and the split
    # is flagged unresolved; a coordinate projector onto columns inside K commutes and is resolved
    g2 = torch.Generator().manual_seed(1); Bk = torch.linalg.qr(torch.randn(12, 4, generator=g2, dtype=torch.float64)).Q
    u = torch.randn(12, 1, generator=g2, dtype=torch.float64); u /= u.norm(); Pu = u @ u.T                    # rank-one projector, generic direction
    _, _, sep_bad = split_by_projector(Bk, Pu @ Bk); assert not sep_bad["resolved"], sep_bad
    Bc = torch.zeros(12, 4, dtype=torch.float64); Bc[[0, 3, 5, 9], range(4)] = 1.0; PB = torch.zeros_like(Bc); PB[[0, 3]] = Bc[[0, 3]]
    inside, outside, sep_ok = split_by_projector(Bc, PB); assert sep_ok["resolved"] and inside.shape[1] == 2 and outside.shape[1] == 2, sep_ok
    for name, K2, exp in [("identical", Qm[:, :10], 0.0), ("nested", Qm[:, :25], 0.0), ("orthogonal", Qm[:, 30:45], 10.0)]:
        Vr = torch.linalg.svd(K2.T, full_matrices=True).Vh[K2.shape[1]:].T       # complement basis
        ang, _ = angle_record(sines=torch.linalg.svdvals(Vr.T @ K1).numpy(), k=10)
        assert abs(ang["d_lost"] - exp) < 1e-10, (name, ang["d_lost"])
        assert abs((1 - ang["L"]) - retained(K1, K2)) < 1e-12
    th = 0.3
    K2 = torch.cat([np.cos(th) * Qm[:, :1] + np.sin(th) * Qm[:, 50:51], Qm[:, 1:10]], 1)
    Vr = torch.linalg.svd(K2.T, full_matrices=True).Vh[10:].T
    ang, s2 = angle_record(sines=torch.linalg.svdvals(Vr.T @ K1).numpy(), k=10)
    assert abs(ang["d_lost"] - np.sin(th) ** 2) < 1e-12 and abs(s2[0] - np.sin(th) ** 2) < 1e-12
    # a slightly rotated subspace: trivial exact intersection, d_lost tiny (the reviewer's caution)
    R = torch.linalg.matrix_exp(1e-4 * (lambda A: A - A.T)(torch.randn(p, p, generator=g, dtype=torch.float64)))
    K2 = R @ K1; Vr = torch.linalg.svd(K2.T, full_matrices=True).Vh[10:].T
    ang, _ = angle_record(sines=torch.linalg.svdvals(Vr.T @ K1).numpy(), k=10)
    assert 0 < ang["d_lost"] < 1e-4 and ang["counts"]["sin2>=0.5"] == 0
    # split by a coordinate projector contained in the kernel
    coords = [0, 1, 2]; E = torch.eye(p, dtype=torch.float64)[:, coords]
    Z, _ = torch.linalg.qr(torch.cat([E, (torch.eye(p, dtype=torch.float64) - E @ E.T) @ torch.randn(p, 5, generator=g, dtype=torch.float64)], 1))
    Bk, _ = torch.linalg.qr(Z @ torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=torch.float64))[0])
    PB = torch.zeros_like(Bk); PB[coords] = Bk[coords]
    Cb, S, sep = split_by_projector(Bk, PB)
    assert Cb.shape[1] == 3 and S.shape[1] == 5 and S[coords].abs().max() < 1e-12 and sep["max_zero_eigenvalue"] < 1e-12
    # amplification: J with known singular values; source = one kernel direction rotated toward singular direction j
    sv = torch.tensor([5.0, 2.0, 0.5]); V = Qm[:, :3]
    src = (np.cos(th) * Qm[:, 10:11] + np.sin(th) * V[:, 1:2])                   # later kernel = span(V)^perp
    M = V.T @ src; G = float(torch.sqrt((sv ** 2 * M.pow(2).sum(1)).sum() / M.pow(2).sum()))
    assert abs(G - 2.0) < 1e-12
    # pair_record on identical and nested kernels: loss unresolved, amplification undefined (not a gain of rounding noise)
    class _S:
        def __init__(self, Vfull, r, ep):
            self.Vr = Vfull[:, :r].contiguous(); self.B = Vfull[:, r:].contiguous(); self.r = r; self.ep = ep; self.s = 100; self.p = Vfull.shape[0]
            self.sv = np.concatenate([np.linspace(5.0, 1.0, r), np.zeros(self.p - r)]); self.fam = {"K": self.B, "S": self.B, "D": self.B, "Cb": self.B[:, :0], "N": self.B[:, :0]}
        def record(self):
            return {"generic_gain": float(np.sqrt((self.sv ** 2).sum() / self.s / self.p))}
    noise = torch.linalg.matrix_exp(1e-9 * (lambda A: A - A.T)(torch.randn(p, p, generator=g, dtype=torch.float64)))
    for later_r in (40, 30):                                          # identical kernel (rank 40) and a larger, nesting kernel (rank 30)
        rec = pair_record(_S(noise @ Qm, 40, 1), _S(Qm, later_r, 2), {}, "t")
        kk = rec["families"]["K"]
        assert kk["amplification"] is None and "amplification_undefined_reason" in kk and not kk["to_later_kernel"]["loss_resolved"], kk
        assert kk["to_later_kernel"]["d_lost_share_from_sin2>=0.5"] is None and abs(kk["to_later_kernel"]["Q"] - 1) < 1e-12
    rec = pair_record(_S(Qm, 40, 1), _S(Qm, 45, 2), {}, "t")            # the later kernel lost five directions: resolved
    assert rec["families"]["K"]["amplification"] is not None and abs(rec["families"]["K"]["to_later_kernel"]["d_lost"] - 5) < 1e-9
    print("persistence-geometry tests passed")


def repair(a):
    """Apply the loss-resolution rule to stored version-1 records without recomputation: where the recorded effective
    lost dimension is below LOSS_RESOLUTION, the amplification and the loss shares are set to undefined with the
    reason; every other number is left untouched.  The repair is logged in the file."""
    for seed in a.seeds:
        path = os.path.join(OUT, f"{run_name(a, seed)}_geometry.json"); res = json.load(open(path)); log = []
        for blk, b in res["blocks"].items():
            for label, pr in b["pairs"].items():
                for fam, f in pr["families"].items():
                    for key in ("to_later_kernel", "to_later_same_family"):
                        t = f.get(key)
                        if not t or "d_lost" not in t:
                            continue
                        t["loss_resolved"] = bool(t["d_lost"] >= LOSS_RESOLUTION)
                        if not t["loss_resolved"]:
                            t["d_lost_share_from_sin2>=0.5"] = None; t["d_lost_share_from_sin2<1e-2"] = None
                    t = f.get("to_later_kernel")
                    if t and not t["loss_resolved"] and f.get("amplification") is not None:
                        log.append({"block": blk, "pair": label, "family": fam, "d_lost": t["d_lost"], "removed_G_over_generic": f["amplification"]["G_over_generic_gain"]})
                        f["amplification"] = None
                        f["amplification_undefined_reason"] = f"effective lost dimension {t['d_lost']:.3g} below the resolution {LOSS_RESOLUTION:g}"
        res["version"] = VERSION; res.setdefault("repairs", []).append({"date": "2026-09-18", "rule": f"loss resolution {LOSS_RESOLUTION:g}", "removed": log})
        records.dump_json(path + ".tmp", res); os.replace(path + ".tmp", path)
        print(f"{path}: {len(log)} unresolved amplification records set to undefined: {[(x['block'], x['pair'], x['family']) for x in log]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "test", "repair"])
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="", help="run tag ('' for ReLU; leaky, gelu, ...)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--blocks", nargs="+", default=["2", "3", "readout"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-legacy", action="store_true")
    ap.add_argument("--dead-tau", type=float, default=0.0, help="tau-dead coordinates: a unit is dead when |sigma'| <= tau on every analysis sample (0 = exact; GELU line item (b)); output files get the suffix _tau<value>")
    a = ap.parse_args()
    {"test": lambda a: _tests(), "run": run, "repair": repair}[a.phase](a)


