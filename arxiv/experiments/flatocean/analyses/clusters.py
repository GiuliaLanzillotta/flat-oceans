"""
Clusters of features h_i and downstream maps D_i, the cluster decomposition of the co-occurrence deficit, purity and stability,
the D_i clusters against gate patterns, the matched random-partition control, the shuffled-pairing validation, the greedy
zero-within merge, and the clustered approximation of the GELU exposure (plan Section 0.6, items 3 and 5; revised after the
reviews of 2026-09-24 and 2026-09-25).

For hidden layer l, a partition pi of the analysis set into groups g gives P(pi) = sum_g H_g (x) C_g (H_g: span of the h_i of the
group, C_g: span of the D_i^T e_a of the group) and
    delta_joint = r_h r_d - rank J_l = [r_h r_d - dim P(pi)]  (pairings absent across groups)  +  [dim P(pi) - rank J_l]  (within groups).
dim P(pi) is the numerical rank of the stacked orthonormal tensor-product bases relative to their own largest singular value
(ladder 1e-6/1e-9/1e-12, gap recorded); the exact inequalities rank J_l <= dim P <= r_h r_d are checked and a violation is
flagged as unresolved.  rank J_l and sigma_1(J_l) come from the authenticated stored factor (identity manifest checked) or are
computed here.

Controls.  `random_<kind>` is the matched random partition of `<kind>` (a random permutation of its assignments: the same
group sizes, no relation to the samples), which gives the chance level of purity, ARI and of the across/within deficit split
(for a random partition the across term is expected to vanish).  The shuffle phase answers a different question: it keeps
every h_i and every D_i and permutes their pairing (globally or within the groups of a partition) and measures the rank of
the permuted Jacobian.

Identities (2026-09-25).  Every layer measurement is bound to its own identity: checkpoint, preprocessing, ordered samples,
the numerical configuration, the layer index and the hash of that layer's stored factor (`layer_identity`); the requested
layer list, seeds and stages of a launch are not part of it, so a single-layer run consumes what a multi-layer run produced.
An assignment file (`..._layer<l>_G<G>_assignments.npz`, one array per partition kind, k-means seeds stacked) carries the
digest of the partitions identity of its layer; a greedy file carries the digest of its greedy identity (knn, max_group and
the merge threshold stored beside it, so that a consumer can re-derive and check it).  Every result that consumes an
assignment array records that array's SHA-256 (`assignment_sha256`) and is reused only while the array is unchanged; results
of other kinds, other G, other knn and other layers are never touched.  Records written under the earlier scheme (one
identity per stage carrying the launch's layer list) are converted once by `clusters migrate` after their authentication.

    python -m flatocean clusters partitions --tag "" --seeds 0 --stages high_accuracy late_endpoint --layers 2 3 --G 10
    python -m flatocean clusters partitions --tag "" --seeds 0 --stages high_accuracy late_endpoint --layers 2 3 --G 5 10 20 40 80 --kinds kmeans_D joint
    python -m flatocean clusters greedy --tag "" --seeds 0 --stages high_accuracy --layers 2 --knn 10
    python -m flatocean clusters shuffle --tag "" --seeds 0 --stages high_accuracy late_endpoint --layers 2 3 --kinds labels kmeans_D joint --greedy-knn 10 25 50
    python -m flatocean clusters approx --tag gelu --seeds 0 --stages high_accuracy late_endpoint --layers 2 3 --G 10
    python -m flatocean clusters migrate --tag "" --seeds 0
"""
# Ported unchanged from experiments/exp_clusters.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean clusters).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import hashlib
import json
import os
import time

import numpy as np
from flatocean import records, runs
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

from flatocean.analyses import persistence as KP
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs

OUT = runs.out_dir("clusters")
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.6 items 3 and 5 (2026-09-24); identifier clusters-v2"
RTOL = 1e-9; LAD = (1e-6, 1e-9, 1e-12); KM_SEEDS = (0, 1, 2); ATOL_MERGE = 1e-7; RANDOM_SEED = 2026
BASE_KINDS = ("labels", "kmeans_h", "kmeans_D", "kmeans_Dn", "joint")
DEFAULT_KINDS = ["labels", "kmeans_h", "kmeans_D", "kmeans_Dn", "joint", "random_labels", "random_kmeans_D", "random_joint",
                 "gates_L2", "gates_L3", "gates_suffix", "patterns_L2", "patterns_L3", "patterns_suffix"]


# ----------------------------------------------------------------------------- linear algebra
def span_basis(M, rtol=RTOL):
    if M.shape[1] == 0:
        return M[:, :0]
    U, sv, _ = torch.linalg.svd(M, full_matrices=False)
    r = int((sv > rtol * sv[0]).sum()) if sv.numel() and sv[0] > 0 else 0
    return U[:, :r]


def block_columns(H, D, idx):
    """Orthonormal columns of H_g (x) C_g in the row-major vec(A) coordinates of ocean_lib (index k * n + j): p x (a b)."""
    n = D.shape[2]; Hg = span_basis(H[idx].T); Cg = span_basis(D[idx].reshape(-1, n).T)
    if Hg.shape[1] == 0 or Cg.shape[1] == 0:
        return None
    return torch.einsum("ka,jb->kjab", Hg, Cg).reshape(Hg.shape[0] * n, Hg.shape[1] * Cg.shape[1])


CHUNK_ROWS = 8192                 # group bases are stacked until this many columns are pending, then one QR update (2026-09-25)


def dim_P(H, D, groups, device, rank_J, rhrd, chunk_rows=CHUNK_ROWS):
    """dim P(pi): ladder of the stacked block bases relative to their own sigma_1, gap at the rule, and the exact-inequality check.
    The stacked matrix [B_1 ... B_G] is triangularized by QR updates on the CPU; group bases are accumulated in chunks of at
    least `chunk_rows` columns before each update (the spectrum of the stack is the same as with one update per group, only
    the number of O(p^3) factorizations changes: with tens of thousands of singleton groups the per-group version takes days)."""
    R = None; ncols = 0; pending = []; npend = 0; p = None
    def flush():
        nonlocal R, pending, npend
        if not pending:
            return
        M = torch.cat(([R] if R is not None else []) + pending, 0); R = torch.linalg.qr(M, mode="r").R[:p]; pending = []; npend = 0
    for g in np.unique(groups):
        B = block_columns(H, D, torch.as_tensor(np.where(groups == g)[0], device=device))
        if B is None:
            continue
        p = B.shape[0]; ncols += B.shape[1]; pending.append(B.T.cpu()); npend += B.shape[1]
        if npend >= chunk_rows:
            flush()
    flush()
    if R is None:
        return {"dim_P": 0, "ladder": {f"{t:g}": 0 for t in LAD}, "gap": None, "columns": 0, "resolved": rank_J == 0}
    sv = torch.linalg.svdvals(R.cpu()).numpy(); lad = {f"{t:g}": int((sv > t * sv[0]).sum()) for t in LAD}; r = lad["1e-09"]    # SVD on the CPU (cusolver float64 is ~30x slower)
    gap = float(sv[r - 1] / max(sv[r], 1e-300)) if 0 < r < sv.size else None
    return {"dim_P": r, "ladder": lad, "gap": gap, "columns": ncols, "sigma1_stack": float(sv[0]),
            "resolved": bool(rank_J <= r <= rhrd and lad["1e-06"] == lad["1e-12"]), "inequalities_hold": bool(rank_J <= r <= rhrd)}


def jacobian_spectrum(H, D, perm, device, s):
    """Singular values of the Jacobian with rows h_i (x) D_{perm(i)}^T e_a (perm = identity: J_l itself)."""
    Dp = D[perm]; p = H.shape[1] * D.shape[2]
    def row_fn(idx):
        h = H[idx]; Dl = Dp[idx]; b, d, n = Dl.shape
        return torch.einsum("ik,iaj->iakj", h, Dl).reshape(b * d, h.shape[1] * n)
    R = O.full_R_cpu(row_fn, s, p, device, chunk=500); sv = O.svdvals(R); del R
    return np.concatenate([sv, np.zeros(max(0, p - sv.size))])


def block_reference(name, ep, l, H, D, run, rd, device):
    """Authenticated rank ladder, gap and sigma_1 of J_l (stored factor), else computed; plus r_h, r_d."""
    s = H.shape[0]; out = KP.factor_spectrum(name, ep, str(l), run, rd) if s == 10000 else None
    if out is not None:
        sv, p, s_, sha, status = out; src = {"rank_source": "stored factor (identity manifest checked)", "factor_sha256": sha, "factor_status": status}
    else:
        sv = jacobian_spectrum(H, D, torch.arange(s, device=device), device, s); src = {"rank_source": "computed here"}
    lad = O.rank_ladder(sv); rJ = lad["1e-09"]
    r_h = O.numrank_matrix(H)[0]["1e-09"]; r_d = O.numrank_matrix(D.reshape(-1, D.shape[2]))[0]["1e-09"]
    return dict(rank_J=rJ, rank_J_ladder=lad, gap_J=O.spectral_gap(sv, rJ), sigma1=float(sv[0]), r_h=r_h, r_d=r_d, delta_joint=r_h * r_d - rJ, **src)


# ----------------------------------------------------------------------------- k-means and partition statistics
def kmeans(Xf, G, seed, iters=60):
    """Lloyd's k-means with k-means++ seeding (float64).  Distinct rows bound the number of groups; empty groups are dropped and
    the assignments are relabelled 0..G_eff-1."""
    g = torch.Generator(device="cpu").manual_seed(seed); s = Xf.shape[0]
    distinct = np.unique(Xf.cpu().numpy(), axis=0).shape[0]; G_eff = min(G, distinct)
    idx = [int(torch.randint(s, (1,), generator=g))]
    d2 = ((Xf - Xf[idx[0]]) ** 2).sum(1)
    for _ in range(1, G_eff):
        tot = float(d2.sum())
        if tot <= 0:
            break
        nxt = int(torch.multinomial((d2 / tot).cpu(), 1, generator=g)); idx.append(nxt); d2 = torch.minimum(d2, ((Xf - Xf[nxt]) ** 2).sum(1))
    C = Xf[idx].clone()
    for _ in range(iters):
        a = torch.cdist(Xf, C).argmin(1)
        newC = torch.stack([Xf[a == k].mean(0) if (a == k).any() else C[k] for k in range(C.shape[0])])
        if torch.allclose(newC, C):
            break
        C = newC
    a = torch.cdist(Xf, C).argmin(1).cpu().numpy(); _, a = np.unique(a, return_inverse=True)
    return a


def pca_scaled(Xf, q):
    Xc = Xf - Xf.mean(0); U, sv, _ = torch.linalg.svd(Xc, full_matrices=False); E = U[:, :q] * sv[:q]
    return E / (E.norm() / np.sqrt(E.shape[0])).clamp_min(1e-300)


def purity(groups, y):
    return sum(np.bincount(y[groups == g]).max() for g in np.unique(groups)) / len(y)


def purity_baseline(groups, y, draws=5, seed=0):
    rng = np.random.default_rng(seed); return float(np.mean([purity(groups, rng.permutation(y)) for _ in range(draws)]))


def partition_stats(groups, y):
    sizes = np.bincount(groups); sizes = sizes[sizes > 0]
    return {"n_groups": int(len(sizes)), "sizes_min_median_max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
            "purity": float(purity(groups, y)), "purity_random_baseline": purity_baseline(groups, y),
            "ARI_labels": float(adjusted_rand_score(y, groups)), "NMI_labels": float(normalized_mutual_info_score(y, groups)), "AMI_labels": float(adjusted_mutual_info_score(y, groups))}


def gate_features(st, l, kind):
    if st.get("activation") == "identity":
        return None
    return torch.cat([st["S"][k] for k in range(l, st["L"])], 1) if kind == "suffix" else st["S"][kind]


def sign_partition(gf):
    """Partition by identical sign/slope pattern (rows of the 0/1 or {0.1, 1} matrix)."""
    _, inv = np.unique((gf > 0.5).cpu().numpy(), axis=0, return_inverse=True); return inv.reshape(-1)


def applicable_kinds(st, l, kinds):
    """Requested kinds that exist for this activation and layer (gate kinds need a downstream gated layer; sign patterns need a
    hard gate; identity has no gate kinds; a matched random kind needs its base kind in the request)."""
    out = []
    for kind in kinds:
        if kind in BASE_KINDS:
            out.append(kind)
        elif kind.startswith("random_"):
            base = kind[7:]
            if base not in BASE_KINDS:
                raise ValueError(f"{kind}: matched random partitions exist for {BASE_KINDS} only")
            if base not in kinds:
                raise ValueError(f"{kind} requested without its base kind {base}")
            out.append(kind)
        elif st.get("activation") == "identity":
            continue
        elif kind == "gates_suffix":
            out.append(kind)
        elif kind == "patterns_suffix" and st.get("activation") in ("relu", "leaky"):
            out.append(kind)
        elif kind.startswith("gates_L") and int(kind[7:]) - 1 >= l:
            out.append(kind)
        elif kind.startswith("patterns_L") and int(kind[10:]) - 1 >= l and st.get("activation") in ("relu", "leaky"):
            out.append(kind)
    return out


def matched_random(base_assignment, l, G, kind):
    """Random permutation of an assignment vector: the same group sizes, no relation to the samples (deterministic seed)."""
    rng = np.random.default_rng([RANDOM_SEED, int(l), int(G), int(hashlib.sha256(kind.encode()).hexdigest()[:8], 16)])
    return np.asarray(base_assignment)[rng.permutation(len(base_assignment))]


def make_partitions(st, l, H, D, y, G, kinds, known=None, seeds=KM_SEEDS):
    """kind -> list of assignments over k-means seeds (one for deterministic kinds) for the requested `kinds`; `known` holds
    the assignments already on file (a matched random kind is built from its base kind, on file or computed here)."""
    s = H.shape[0]; d, n = D.shape[1], D.shape[2]; parts = {}; known = dict(known or {})
    vecD = D.reshape(s, d * n); nrm = vecD.norm(dim=1, keepdim=True).clamp_min(1e-300)
    feats = {"kmeans_h": H, "kmeans_D": vecD, "kmeans_Dn": vecD / nrm, "joint": torch.cat([pca_scaled(H, 50), pca_scaled(vecD, 50)], 1)}
    hard = st.get("activation") in ("relu", "leaky")
    for k in range(l, st["L"]):
        gf = gate_features(st, l, k)
        if gf is not None:
            feats[f"gates_L{k + 1}"] = gf[:s]
            if hard:
                feats[f"patterns_L{k + 1}"] = gf[:s]
    gs = gate_features(st, l, "suffix")
    if gs is not None:
        feats["gates_suffix"] = gs[:s]
        if hard:
            feats["patterns_suffix"] = gs[:s]
    for kind in kinds:
        if kind.startswith("random_"):
            continue
        if kind == "labels":
            parts[kind] = [y]
        elif kind.startswith("patterns_") and kind in feats:
            parts[kind] = [sign_partition(feats[kind])]
        elif kind in feats:
            parts[kind] = [kmeans(feats[kind], G, sd) for sd in seeds]
    for kind in kinds:
        if kind.startswith("random_"):
            base = kind[7:]; src = parts.get(base) or known.get(base)
            if src is None:
                raise RuntimeError(f"{kind}: base kind {base} neither on file nor computed")
            parts[kind] = [matched_random(np.asarray(src)[0], l + 1, G, kind)]
    return parts


# ----------------------------------------------------------------------------- identities and assignment files
def base_config():
    return dict(protocol=PROTOCOL, rtol=RTOL, ladder=list(LAD), km_seeds=list(KM_SEEDS))


_identity = P.analysis_identity                   # (run, rd, ep, config) -> identity; the tests substitute a file-free version


def _load_stage(a, seed, stage):
    """Checkpoint, analysis inputs and frozen structure of one stage of one run (None if the stage has no checkpoint)."""
    rd = O.run_dir("mnist", "mlp", seed, a.tag); stages = stage_epochs(rd); name = f"mnist_mlp{'_' + a.tag if a.tag else ''}_seed{seed}"
    ep = stages.get(stage, {}).get("checkpoint_epoch")
    if ep is None:
        return None
    run_ = O.load_run("mnist", "mlp", seed, tag=a.tag, epoch=ep, device=a.device); X, y, _ = run_["data"]["analysis"]; y = y.cpu().numpy()
    return dict(rd=rd, name=name, ep=ep, run=run_, X=X, y=y, st=O.mlp_structure(run_["net"], X))


def _factor_sha(name, ep, l, run_, rd):
    """SHA-256 of the authenticated stored factor of layer l (None when the block is computed here)."""
    fs = KP.factor_spectrum(name, ep, str(l), run_, rd); return fs[3] if fs is not None else None


def stage_setup(a, seed, stage):
    """Stage data plus the per-layer factor hashes (factors authenticated here, before any cache decision) and the stage
    identity (checkpoint, samples, numerical configuration; no layer list)."""
    S = _load_stage(a, seed, stage)
    if S is None:
        return None
    S["factor_sha"] = {str(l): _factor_sha(S["name"], S["ep"], l, S["run"], S["rd"]) for l in a.layers}
    S["stage_identity"] = _identity(S["run"], S["rd"], S["ep"], base_config())
    return S


def layer_identity(S, l, **extra):
    """Identity of one layer measurement: stage identity + layer index + that layer's factor hash + the phase settings."""
    if str(l) not in S["factor_sha"]:
        S["factor_sha"][str(l)] = _factor_sha(S["name"], S["ep"], l, S["run"], S["rd"])
    return _identity(S["run"], S["rd"], S["ep"], dict(base_config(), layer=int(l), factor_sha256=S["factor_sha"][str(l)], **extra))


def array_sha256(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype=np.int64)).tobytes()).hexdigest()


def assignments_path(name, ep, l, G):
    return os.path.join(OUT, f"{name}_epoch{ep:04d}_layer{l}_G{G}_assignments.npz")


def greedy_path(name, ep, l, knn):
    return os.path.join(OUT, f"{name}_epoch{ep:04d}_layer{l}_greedy_knn{knn}_assignments.npz")


def load_assignments(path, digest):
    """Arrays of an assignment file bound to `digest` (ProvenanceError under another identity); None if absent."""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["identity_digest"]) != digest:
        raise P.ProvenanceError(f"{path}: assignments under another identity (run `clusters migrate` for files of the earlier scheme)")
    return {k: z[k] for k in z.files if k != "identity_digest"}


def save_assignments(path, digest, arrays, **scalars):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, identity_digest=digest, **arrays, **scalars); os.replace(tmp, path)


def legacy_stage(srec):
    """True for a stage record of the earlier scheme (one identity per stage carrying the launch's layer list)."""
    return isinstance(srec, dict) and "layers" in (srec.get("identity") or {}).get("config", {})


def layer_slot(srec, key, ident):
    """The layer record under `key` when it carries this identity, else a fresh one; a record under another identity is
    moved in full to the stage's `superseded` list, never silently reused or dropped."""
    if legacy_stage(srec):
        raise RuntimeError("record of the earlier identity scheme; run `python -m flatocean clusters migrate` first")
    L = srec["layers"].get(key)
    if L is None or (L.get("identity") or {}).get("digest") != ident["digest"]:
        if L is not None:
            srec.setdefault("superseded", []).append({"key": key, "record": L})
        L = srec["layers"][key] = {"identity": ident}
    return L


def open_record(out_path, S, stage):
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "results": {}}
    srec = res["results"].setdefault(stage, {"epoch": S["ep"], "layers": {}})
    if not legacy_stage(srec):
        srec["epoch"] = S["ep"]; srec["stage_identity"] = S["stage_identity"]; srec.setdefault("layers", {})
    return res, srec


def save_record(out_path, res):
    records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def greedy_arrays(S, l, knn):
    """Assignments of a stored greedy partition, authenticated by re-deriving its identity from the settings stored beside it:
    (arrays, greedy identity digest) or (None, None)."""
    gp = greedy_path(S["name"], S["ep"], l, knn)
    if not os.path.exists(gp):
        return None, None
    z = np.load(gp, allow_pickle=False)
    gid = layer_identity(S, l, phase="greedy", knn=int(z["knn"]), max_group=int(z["max_group"]), merge_threshold=float(z["merge_threshold"]))
    if str(z["identity_digest"]) != gid["digest"]:
        raise P.ProvenanceError(f"{gp}: greedy assignments under another identity")
    return {k: z[k] for k in z.files if k.startswith("order_")}, gid["digest"]


# ----------------------------------------------------------------------------- phases
def partitions_phase(a):
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        name = f"mnist_mlp{'_' + a.tag if a.tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_partitions.json")
        for stage in a.stages:
            S = stage_setup(a, seed, stage)
            if S is None:
                continue
            res, srec = open_record(out_path, S, stage); ep = S["ep"]; st = S["st"]; y = S["y"]; s = S["X"].shape[0]
            prev = {}                                                          # G -> {kind: assignments} of the previous layer
            for l in a.layers:
                H = st["H"][l - 1][:s]; D = st["D"][l - 1][:s]; t0 = time.time()
                lrec = layer_slot(srec, str(l), layer_identity(S, l, phase="partitions")); digest = lrec["identity"]["digest"]
                if "rank_J" not in lrec or a.overwrite:
                    lrec.update(block_reference(name, ep, l, H, D, S["run"], S["rd"], a.device))
                rJ, r_h, r_d, delta = lrec["rank_J"], lrec["r_h"], lrec["r_d"], lrec["delta_joint"]
                for G in a.G:
                    needed = applicable_kinds(st, l - 1, a.kinds); apath = assignments_path(name, ep, l, G)
                    saved = load_assignments(apath, digest) or {}
                    parts = {} if a.overwrite else {k: list(saved[k]) for k in needed if k in saved}
                    missing = [k for k in needed if k not in parts]
                    if missing:                                                # only the kinds not on file are built; the others keep their arrays and results
                        new = make_partitions(st, l - 1, H, D, y, G, missing, known=parts)
                        still = [k for k in missing if k not in new]
                        if still:
                            raise RuntimeError(f"{name} {stage} layer {l}: applicable kinds not produced: {still}")
                        parts.update(new); keep = {k: v for k, v in saved.items() if k not in parts}
                        save_assignments(apath, digest, {**{k: np.stack(v) for k, v in parts.items()}, **keep})
                    grec = lrec.setdefault(str(G), {}); grec["applicable_kinds"] = needed; grec["inapplicable_kinds"] = [k for k in a.kinds if k not in needed]
                    for kind in needed:
                        sha = array_sha256(np.stack(parts[kind])); old = grec.get(kind)
                        if isinstance(old, dict) and old.get("assignment_sha256") == sha and "dim_P" in old and not a.overwrite:
                            continue
                        tk = time.time(); rec = partition_stats(parts[kind][0], y); pp = dim_P(H, D, parts[kind][0], a.device, rJ, r_h * r_d); rec["assignment_sha256"] = sha
                        print(f"   {name} {stage} layer {l} G={G} {kind}: {rec['n_groups']} groups, dim P {pp['dim_P']} (rank J {rJ}, r_h r_d {r_h * r_d}) ({time.time() - tk:.0f}s)", flush=True)
                        rec.update({"dim_P": pp["dim_P"], "dim_P_ladder": pp["ladder"], "dim_P_gap": pp["gap"], "dim_P_resolved": pp["resolved"], "inequalities_hold": pp["inequalities_hold"],
                                    "within": pp["dim_P"] - rJ, "across": r_h * r_d - pp["dim_P"], "across_share_of_delta": (r_h * r_d - pp["dim_P"]) / delta if delta > 0 else None})
                        if kind.startswith("random_"):
                            rec["control"] = f"matched random partition of {kind[7:]} (same group sizes)"
                        if len(parts[kind]) > 1:
                            plist = parts[kind]; rec["stability_ARI_over_seeds"] = float(np.mean([adjusted_rand_score(plist[i], plist[j]) for i in range(len(plist)) for j in range(i + 1, len(plist))]))
                        grec[kind] = rec; save_record(out_path, res)                   # per-kind checkpoint of the record
                    if "kmeans_D" in parts:
                        base = parts["kmeans_D"][0]; cmp = {k: {"ARI": float(adjusted_rand_score(base, parts[k][0])), "AMI": float(adjusted_mutual_info_score(base, parts[k][0]))} for k in parts if k != "kmeans_D"}
                        cmp["assignment_sha256"] = {k: array_sha256(np.stack(parts[k])) for k in parts}
                        if "patterns_suffix" in parts:
                            per = [len(np.unique(parts["patterns_suffix"][0][base == g])) for g in np.unique(base)]
                            cmp["distinct_suffix_patterns_per_D_cluster"] = {"min": int(min(per)), "median": float(np.median(per)), "max": int(max(per)), "total_distinct": int(len(np.unique(parts["patterns_suffix"][0])))}
                        grec["D_cluster_comparisons"] = cmp
                    # agreement with the previous requested layer (same kind and G), from this run or from its authenticated file
                    if l > min(a.layers):
                        lp = a.layers[a.layers.index(l) - 1]
                        pv = prev.get(G) or load_assignments(assignments_path(name, ep, lp, G), layer_identity(S, lp, phase="partitions")["digest"]) or {}
                        for kind in parts:
                            if kind in pv:
                                grec[kind]["ARI_with_previous_layer"] = {"layer": lp, "ARI": float(adjusted_rand_score(np.asarray(pv[kind])[0], parts[kind][0]))}
                    prev[G] = {k: np.stack(v) for k, v in parts.items()}
                    print(f"== {name} {stage} layer {l} G={G}: delta {delta}; " + "; ".join(f"{k}: G_eff {v['n_groups']} across {v['across']} ({(v['across_share_of_delta'] or 0):.2f}{'' if v['dim_P_resolved'] else ', unresolved'}) purity {v['purity']:.2f}/{v['purity_random_baseline']:.2f} ARI {v['ARI_labels']:.2f}"
                                                                                for k, v in grec.items() if isinstance(v, dict) and "dim_P" in v) + f" ({time.time() - t0:.0f}s)", flush=True)
                    save_record(out_path, res)


def shuffle_phase(a):
    """Within-group and global permutations of the D_i for the stored partitions (G = a.G[0]) and the stored greedy partitions
    (every knn in a.greedy_knn that exists); rank of the permuted pairs.  Each kind's result is bound to the assignment
    array it consumed."""
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        name = f"mnist_mlp{'_' + a.tag if a.tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_shuffle.json")
        for stage in a.stages:
            S = stage_setup(a, seed, stage)
            if S is None:
                continue
            res, srec = open_record(out_path, S, stage); ep = S["ep"]; st = S["st"]; y = S["y"]; s = S["X"].shape[0]
            for l in a.layers:
                H = st["H"][l - 1][:s]; D = st["D"][l - 1][:s]
                lrec = layer_slot(srec, str(l), layer_identity(S, l, phase="shuffle", G=a.G[0]))
                if "rank_J" not in lrec:
                    lrec.update(block_reference(name, ep, l, H, D, S["run"], S["rd"], a.device))
                rJ, r_h, r_d, delta = lrec["rank_J"], lrec["r_h"], lrec["r_d"], lrec["delta_joint"]
                parts = load_assignments(assignments_path(name, ep, l, a.G[0]), layer_identity(S, l, phase="partitions")["digest"]) or {}
                sources = {"labels": (y, None)}
                for kind in a.kinds:
                    if kind != "labels" and kind in parts:
                        sources[kind] = (np.asarray(parts[kind])[0], None)
                for k in a.greedy_knn:
                    arrays, gdig = greedy_arrays(S, l, k)
                    if arrays is not None:
                        sources[f"greedy_knn{k}"] = (np.asarray(arrays["order_distance"]), gdig)
                def permuted(perm):
                    sv = jacobian_spectrum(H, D, torch.as_tensor(perm, device=a.device), a.device, s); lad = O.rank_ladder(sv)
                    return {"rank_J_permuted": lad, "gap": O.spectral_gap(sv, lad["1e-09"]), "delta_permuted": r_h * r_d - lad["1e-09"], "delta_preserved_fraction": (r_h * r_d - lad["1e-09"]) / delta if delta else None}
                if "global" not in lrec or a.overwrite:
                    t0 = time.time(); lrec["global"] = permuted(np.random.default_rng([seed, l, 0]).permutation(s)); lrec["global"]["seconds"] = time.time() - t0
                    print(f"== {name} {stage} layer {l} global permutation: delta {delta} -> {lrec['global']['delta_permuted']} ({lrec['global']['seconds']:.0f}s)", flush=True); save_record(out_path, res)
                for kind in a.kinds + [f"greedy_knn{k}" for k in a.greedy_knn]:
                    if kind not in sources:
                        lrec[kind] = {"unavailable": "no stored assignments for this kind"}; continue
                    groups, gdig = sources[kind]; sha = array_sha256(groups); old = lrec.get(kind)
                    if isinstance(old, dict) and old.get("assignment_sha256") == sha and "delta_permuted" in old and not a.overwrite:
                        continue
                    t0 = time.time(); perm = np.arange(s); rng = np.random.default_rng([seed, l, int(hashlib.sha256(kind.encode()).hexdigest()[:8], 16)])
                    for g in np.unique(groups):
                        idx = np.where(groups == g)[0]; perm[idx] = idx[rng.permutation(len(idx))]
                    lrec[kind] = dict(permuted(perm), n_groups=int(len(np.unique(groups))), assignment_sha256=sha, seconds=time.time() - t0)
                    if gdig is not None:
                        lrec[kind]["greedy_identity_digest"] = gdig
                    print(f"== {name} {stage} layer {l} within-{kind} permutation ({lrec[kind]['n_groups']} groups): delta {delta} -> {lrec[kind]['delta_permuted']} ({lrec[kind]['seconds']:.0f}s)", flush=True)
                    save_record(out_path, res)
                save_record(out_path, res)


def greedy_phase(a):
    """Locally merge-irreducible zero-within partition relative to the gate-neighbour graph (exploratory, decision of 2026-09-24):
    start from the partition by identical suffix gate patterns; candidate pairs are the union of the k nearest groups (Hamming
    distance between mean patterns); a merge is accepted when the merged block H (x) C adds no direction outside the row space
    of J_l (orthonormal residual threshold ATOL_MERGE), so W_l stays zero and no merge is ever forced.  `max_group` is a search
    restriction only: a candidate merge whose union would exceed that many samples is not tested.  Several merge orders
    (increasing distance, two noisy distances, one random); one record per (layer, knn), each under its own identity."""
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        name = f"mnist_mlp{'_' + a.tag if a.tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_greedy.json")
        for stage in a.stages:
            S = stage_setup(a, seed, stage)
            if S is None:
                continue
            res, srec = open_record(out_path, S, stage); ep = S["ep"]; st = S["st"]; y = S["y"]; s = S["X"].shape[0]
            if st.get("activation") == "identity":
                continue
            for l in a.layers:
                gid = layer_identity(S, l, phase="greedy", knn=a.knn, max_group=a.max_group, merge_threshold=ATOL_MERGE)
                lrec = layer_slot(srec, f"{l}:knn{a.knn}", gid)
                if "orders" in lrec and not a.overwrite and os.path.exists(greedy_path(name, ep, l, a.knn)):
                    continue
                H = st["H"][l - 1][:s]; D = st["D"][l - 1][:s]; n = D.shape[2]; p = H.shape[1] * n; t0 = time.time()
                ref = block_reference(name, ep, l, H, D, S["run"], S["rd"], a.device); rJ = ref["rank_J"]
                # orthonormal basis of the row space of J_l (P at the finest partition), float64
                def row_fn(idx):
                    h = H[idx]; Dl = D[idx]; b, d, _ = Dl.shape; return torch.einsum("ik,iaj->iakj", h, Dl).reshape(b * d, p)
                R = O.full_R_cpu(row_fn, s, p, a.device, chunk=500); U_, sv, Vh = torch.linalg.svd(R, full_matrices=False); del R, U_
                r = int((sv > RTOL * sv[0]).sum()); Q = Vh[:r].T.to(a.device); del Vh; t_basis = time.time() - t0
                gs = gate_features(st, l - 1, "suffix")[:s]; init = sign_partition(gs); n0 = int(init.max()) + 1
                means = torch.stack([gs[init == g].mean(0) for g in range(n0)])
                k = min(a.knn, n0 - 1); edges = set()
                for i in range(0, n0, 500):
                    dist = torch.cdist(means[i:i + 500], means, p=1); dist[torch.arange(dist.shape[0]), torch.arange(i, i + dist.shape[0])] = float("inf")
                    nn = dist.topk(k, largest=False)
                    for ii in range(dist.shape[0]):
                        for jj, dd in zip(nn.indices[ii].tolist(), nn.values[ii].tolist()):
                            edges.add((min(i + ii, jj), max(i + ii, jj), dd))
                edges = sorted(edges, key=lambda e: (e[0], e[1])); rng = np.random.default_rng(seed)
                orders = {"distance": sorted(edges, key=lambda e: e[2]),
                          "noisy_1": sorted(edges, key=lambda e: e[2] * float(np.exp(0.7 * rng.standard_normal()))),
                          "noisy_2": sorted(edges, key=lambda e: e[2] * float(np.exp(0.7 * rng.standard_normal()))),
                          "random": [edges[i] for i in rng.permutation(len(edges))]}
                out = {"rank_J": rJ, "initial_groups_identical_suffix_patterns": n0, "candidate_edges": len(edges), "knn": k,
                       "max_group": a.max_group, "max_group_meaning": "search restriction: merges whose union would exceed this many samples are not tested; no merge is forced",
                       "merge_threshold": ATOL_MERGE, "name": "locally merge-irreducible zero-within partition relative to the gate-neighbour graph",
                       "seconds_row_space_basis": t_basis, "orders": {}}; assign = {}
                for oname, order in orders.items():
                    to = time.time(); parent = np.arange(n0); members = {g: list(np.where(init == g)[0]) for g in range(n0)}; accepted = tested = 0
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]; x = parent[x]
                        return x
                    for gi_, gj_, dd in order:
                        gi, gj = find(gi_), find(gj_)
                        if gi == gj or len(members[gi]) + len(members[gj]) > a.max_group:
                            continue
                        B = block_columns(H, D, torch.as_tensor(members[gi] + members[gj], device=a.device)); tested += 1
                        resid = torch.linalg.svdvals((B - Q @ (Q.T @ B)).cpu()).numpy()
                        if int((resid > ATOL_MERGE).sum()) == 0:
                            parent[gj] = gi; members[gi] += members.pop(gj); accepted += 1
                    groups = np.array([find(init[i]) for i in range(s)]); _, groups = np.unique(groups, return_inverse=True); assign[f"order_{oname}"] = groups
                    pp = dim_P(H, D, groups, a.device, rJ, ref["r_h"] * ref["r_d"]); sizes = np.bincount(groups)
                    out["orders"][oname] = {"n_groups": int(len(sizes)), "singletons": int((sizes == 1).sum()), "largest_group": int(sizes.max()), "tested": tested, "accepted": accepted,
                                            "within_audit": pp["dim_P"] - rJ, "dim_P_ladder": pp["ladder"], "dim_P_resolved": pp["resolved"], "stats": partition_stats(groups, y), "seconds": time.time() - to}
                    print(f"== {name} {stage} layer {l} greedy knn {k} [{oname}]: {len(sizes)} groups from {n0} ({accepted} merges of {tested} tests), largest {sizes.max()}, within audit {pp['dim_P'] - rJ}, purity {out['orders'][oname]['stats']['purity']:.2f} ({time.time() - to:.0f}s)", flush=True)
                names = list(assign); out["stability_ARI_between_orders"] = float(np.mean([adjusted_rand_score(assign[x], assign[z]) for i, x in enumerate(names) for z in names[i + 1:]])) if len(names) > 1 else None
                save_assignments(greedy_path(name, ep, l, a.knn), gid["digest"], assign, knn=int(a.knn), max_group=int(a.max_group), merge_threshold=float(ATOL_MERGE),
                                 partitions_identity_digest=layer_identity(S, l, phase="partitions")["digest"])
                out["seconds"] = time.time() - t0; out["assignment_sha256"] = {k_: array_sha256(v) for k_, v in assign.items()}; lrec.update(out)
                save_record(out_path, res)


def approx_phase(a):
    """Clustered approximation of the exposure (GELU, and any activation): replace D_i by the mean of its k-means_D cluster,
    J~ with rows h_i (x) Dbar_g(i)^T e_a; spectra of J~ and of the residual J - J~ against the spectrum of J."""
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        name = f"mnist_mlp{'_' + a.tag if a.tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_approx.json")
        for stage in a.stages:
            S = stage_setup(a, seed, stage)
            if S is None:
                continue
            res, srec = open_record(out_path, S, stage); ep = S["ep"]; st = S["st"]; s = S["X"].shape[0]
            for l in a.layers:
                H = st["H"][l - 1][:s]; D = st["D"][l - 1][:s]; lrec = layer_slot(srec, str(l), layer_identity(S, l, phase="approx"))
                ref = None; sv_J = None
                for G in a.G:
                    parts = load_assignments(assignments_path(name, ep, l, G), layer_identity(S, l, phase="partitions")["digest"])
                    if parts is None or "kmeans_D" not in parts:
                        print(f"   {name} {stage} layer {l} G={G}: no kmeans_D assignments; run partitions first", flush=True); continue
                    sha = array_sha256(parts["kmeans_D"]); old = lrec.get(str(G))
                    if isinstance(old, dict) and old.get("assignment_sha256") == sha and not a.overwrite:
                        continue
                    t0 = time.time()
                    if ref is None:
                        ref = block_reference(name, ep, l, H, D, S["run"], S["rd"], a.device); fs = KP.factor_spectrum(name, ep, str(l), S["run"], S["rd"])
                        sv_J = fs[0] if fs is not None else jacobian_spectrum(H, D, torch.arange(s, device=a.device), a.device, s)
                    groups = np.asarray(parts["kmeans_D"])[0]; Dbar = torch.stack([D[torch.as_tensor(np.where(groups == g)[0], device=a.device)].mean(0) for g in range(groups.max() + 1)])
                    Dt = Dbar[torch.as_tensor(groups, device=a.device)]; p = H.shape[1] * D.shape[2]
                    def rows_of(DD):
                        return lambda idx: torch.einsum("ik,iaj->iakj", H[idx], DD[idx]).reshape(-1, p)
                    sv_t = O.svdvals(O.full_R_cpu(rows_of(Dt), s, p, a.device, chunk=500)); sv_r = O.svdvals(O.full_R_cpu(rows_of(D - Dt), s, p, a.device, chunk=500))
                    sv_t = np.concatenate([sv_t, np.zeros(p - sv_t.size)]); sv_r = np.concatenate([sv_r, np.zeros(p - sv_r.size)])
                    lad_t = O.rank_ladder(sv_t); k = lambda sv, e: int((sv / np.sqrt(s) <= e).sum())
                    lrec[str(G)] = {"rank_Jtilde_ladder": lad_t, "gap_Jtilde": O.spectral_gap(sv_t, lad_t["1e-09"]), "residual_frobenius_over_J": float(np.linalg.norm(sv_r) / np.linalg.norm(sv_J)),
                                    "residual_sigma1_over_J_sigma1": float(sv_r[0] / sv_J[0]), "k_counts": {e: {"J": k(sv_J, float(e)), "Jtilde": k(sv_t, float(e)), "residual": k(sv_r, float(e))} for e in ("0.01", "0.001", "0.0001", "1e-06")},
                                    "rank_J": ref["rank_J"], "assignment_sha256": sha, "seconds": time.time() - t0}
                    np.savez_compressed(os.path.join(OUT, f"{name}_epoch{ep:04d}_layer{l}_G{G}_approx_spectra.npz"), sv_J=sv_J, sv_Jtilde=sv_t, sv_residual=sv_r, assignment_sha256=sha)
                    q = lrec[str(G)]; print(f"== {name} {stage} layer {l} G={G}: rank J~ {lad_t['1e-09']} (J {ref['rank_J']}), residual Frobenius {q['residual_frobenius_over_J']:.3f}, k(1e-3): J {q['k_counts']['0.001']['J']} J~ {q['k_counts']['0.001']['Jtilde']} residual {q['k_counts']['0.001']['residual']} ({time.time() - t0:.0f}s)", flush=True)
                    save_record(out_path, res)
                save_record(out_path, res)


def migrate_phase(a):
    """Convert partition records and assignment files of the earlier scheme (one identity per stage carrying the launch's
    layer list; results bound to the assignment file's hash) to per-layer identities and per-array hashes.  Nothing is
    recomputed.  Order, so that an interruption can be resumed: (1) the legacy identity is re-derived from the current
    checkpoint, samples and the factors on disk (a replaced factor changes the hash and the digest, and the stage is
    refused); (2) every assignment file is authenticated against the legacy digest and every result against the file's
    hash; (3) the record is rewritten with per-layer identities and per-array hashes and saved with a `migrated_from`
    note; (4) the assignment files are rewritten with the per-layer digests.  A stage whose record is already migrated
    but whose files still carry the legacy digest (an interrupted step 4) is completed: its files are accepted when
    their digest is the recorded legacy digest and their arrays hash to the recorded `assignment_sha256`."""
    import shutil
    for seed in a.seeds:
        name = f"mnist_mlp{'_' + a.tag if a.tag else ''}_seed{seed}"; out_path = os.path.join(OUT, f"{name}_partitions.json")
        if not os.path.exists(out_path):
            continue
        res = json.load(open(out_path)); bak = os.path.join(OUT, f"legacy_{time.strftime('%Y-%m-%d')}")
        for stage, srec in res["results"].items():
            if legacy_stage(srec):
                old = srec["identity"]; layers = [int(l) for l in old["config"]["layers"]]; a.layers = layers
                S = stage_setup(a, seed, stage)
                if S is None:
                    raise RuntimeError(f"{name} {stage}: no checkpoint for a stage with a legacy record")
                # (1) the legacy identity from the CURRENT inputs: same checkpoint, samples, configuration and factor bytes
                cfg = dict(old["config"]); cfg["factor_sha256"] = {str(l): S["factor_sha"][str(l)] for l in layers}
                cur = _identity(S["run"], S["rd"], S["ep"], cfg); P.require_match(old, cur, f"{name} {stage} legacy identity")
                # (2) authenticate every file and result before touching anything
                files = {}
                for lkey, lrec in srec["layers"].items():
                    for gkey, grec in lrec.items():
                        if not (isinstance(grec, dict) and gkey.isdigit()):
                            continue
                        apath = assignments_path(name, S["ep"], int(lkey), int(gkey))
                        if not os.path.exists(apath):
                            raise RuntimeError(f"{apath}: missing assignments for a legacy record")
                        z = np.load(apath, allow_pickle=False)
                        if str(z["identity_digest"]) != old["digest"]:
                            raise P.ProvenanceError(f"{apath}: assignments under an identity other than the legacy stage identity")
                        arrays = {k: z[k] for k in z.files if k != "identity_digest"}; fsha = P.sha256_file(apath)
                        for kind, krec in grec.items():
                            if isinstance(krec, dict) and "dim_P" in krec and (krec.get("assignments_sha256") != fsha or kind not in arrays):
                                raise P.ProvenanceError(f"{name} {stage} layer {lkey} G={gkey} {kind}: legacy result not bound to the file on disk")
                        files[(lkey, gkey)] = (apath, arrays)
                os.makedirs(bak, exist_ok=True)
                if not os.path.exists(os.path.join(bak, os.path.basename(out_path))):
                    shutil.copy2(out_path, os.path.join(bak, os.path.basename(out_path)))
                for apath, _ in files.values():
                    if not os.path.exists(os.path.join(bak, os.path.basename(apath))):
                        shutil.copy2(apath, os.path.join(bak, os.path.basename(apath)))
                # (3) the record: per-layer identities, per-array hashes
                for lkey, lrec in srec["layers"].items():
                    lrec["identity"] = layer_identity(S, int(lkey), phase="partitions")
                    for gkey, grec in lrec.items():
                        if not (isinstance(grec, dict) and gkey.isdigit()):
                            continue
                        arrays = files[(lkey, gkey)][1]
                        for kind, krec in grec.items():
                            if isinstance(krec, dict) and "dim_P" in krec:
                                krec["assignment_sha256"] = array_sha256(arrays[kind]); krec.pop("assignments_sha256", None)
                            elif kind == "D_cluster_comparisons" and isinstance(krec, dict):
                                krec["assignment_sha256"] = {k: array_sha256(arrays[k]) for k in arrays}; krec.pop("assignments_sha256", None)
                srec["stage_identity"] = S["stage_identity"]
                srec["migrated_from"] = {"identity": old, "date": time.strftime("%Y-%m-%d"), "scheme": "one identity per stage with the launch's layer list", "files_rewritten": False}
                del srec["identity"]; save_record(out_path, res)
                print(f"== {name} {stage}: record migrated (layers {layers})", flush=True)
            if not (isinstance(srec.get("migrated_from"), dict) and not srec["migrated_from"].get("files_rewritten", True)):
                continue
            # (4) the files: legacy digest -> per-layer digest (resumable: files already rewritten are accepted as they are)
            legacy_digest = srec["migrated_from"]["identity"]["digest"]
            for lkey, lrec in srec["layers"].items():
                for gkey, grec in lrec.items():
                    if not (isinstance(grec, dict) and gkey.isdigit()):
                        continue
                    apath = assignments_path(name, srec["epoch"], int(lkey), int(gkey)); z = np.load(apath, allow_pickle=False)
                    arrays = {k: z[k] for k in z.files if k != "identity_digest"}; dig = str(z["identity_digest"])
                    if dig == lrec["identity"]["digest"]:
                        continue
                    if dig != legacy_digest:
                        raise P.ProvenanceError(f"{apath}: neither the legacy nor the migrated digest")
                    for kind, krec in grec.items():
                        if isinstance(krec, dict) and "dim_P" in krec and krec.get("assignment_sha256") != array_sha256(arrays[kind]):
                            raise P.ProvenanceError(f"{apath}: array {kind} differs from the migrated record")
                    save_assignments(apath, lrec["identity"]["digest"], arrays)
            srec["migrated_from"]["files_rewritten"] = True; save_record(out_path, res)
            print(f"== {name} {stage}: assignment files rewritten", flush=True)


def _tests():
    from flatocean import tier1_train as T
    torch.manual_seed(0); net = T.MLP(6, [8, 8, 7], 3, init_seed=3, dtype=torch.float64)
    with torch.no_grad():
        net.layers[1].bias[:2] = -50.0
    X = torch.randn(24, 6, dtype=torch.float64); st = O.mlp_structure(net, X); H, D = st["H"][1], st["D"][1]; s = 24
    sv = jacobian_spectrum(H, D, torch.arange(s), "cpu", s); rJ = O.rank_ladder(sv)["1e-09"]
    r_h = O.numrank_matrix(H)[0]["1e-09"]; r_d = O.numrank_matrix(D.reshape(-1, D.shape[2]))[0]["1e-09"]
    ps = dim_P(H, D, np.arange(s), "cpu", rJ, r_h * r_d); po = dim_P(H, D, np.zeros(s, dtype=int), "cpu", rJ, r_h * r_d)
    assert ps["dim_P"] == rJ and po["dim_P"] == r_h * r_d and ps["resolved"] and po["resolved"], (ps, po, rJ, r_h * r_d)
    # scale invariance: multiplying every D_i by 1e12 changes neither dim P nor the merge decision
    Ds = D * 1e12; ps2 = dim_P(H, Ds, np.arange(s), "cpu", rJ, r_h * r_d); assert ps2["dim_P"] == rJ, ps2
    Xf = torch.cat([torch.randn(30, 4, dtype=torch.float64) + 8 * torch.eye(4, dtype=torch.float64)[k] for k in range(4)]); y = np.repeat(np.arange(4), 30)
    g = kmeans(Xf, 4, 0); assert adjusted_rand_score(y, g) == 1.0
    g1 = kmeans(torch.ones(20, 3, dtype=torch.float64), 10, 0); assert len(np.unique(g1)) == 1               # constant responses: one group
    st_ = partition_stats(g, y); assert st_["purity"] == 1.0 and st_["purity_random_baseline"] < 0.6
    sp = sign_partition(torch.tensor([[1.0, 0], [1, 0], [0, 1]])); assert list(sp) == [0, 0, 1] or list(sp) == [1, 1, 0]
    print("cluster tests passed:", {"rank_J": rJ, "r_h r_d": r_h * r_d, "dim P singletons": ps["dim_P"], "dim P one group": po["dim_P"]})


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["partitions", "shuffle", "greedy", "approx", "migrate", "test"])
    ap.add_argument("--tag", default=""); ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--stages", nargs="+", default=["high_accuracy", "mse_interpolation", "epoch_250", "late_endpoint"])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3]); ap.add_argument("--G", type=int, nargs="+", default=[10])
    ap.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    ap.add_argument("--knn", type=int, default=10, help="greedy: neighbours of the candidate graph"); ap.add_argument("--greedy-knn", type=int, nargs="*", default=[10, 25, 50], help="shuffle: stored greedy partitions to permute within")
    ap.add_argument("--max-group", type=int, default=200)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    {"partitions": partitions_phase, "shuffle": shuffle_phase, "greedy": greedy_phase, "approx": approx_phase, "migrate": migrate_phase, "test": lambda a_: _tests()}[a.phase](a)
