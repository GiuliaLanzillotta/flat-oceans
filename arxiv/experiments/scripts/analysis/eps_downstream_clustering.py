"""Epsilon-threshold, DIAMETER-CONTROLLED clustering of layer-l downstream maps D_i (coordinator item 4, 2026-10-02;
extended 2026-10-03 to GELU/leaky + within/between distance distributions and authenticated metadata).

Distinct from the k-means / rank-share cluster line (`flatocean clusters`): a clustering-PURITY measurement by an explicit
DISTANCE THRESHOLD epsilon, with diameter-controlled groups so chaining cannot merge dissimilar maps. It does NOT touch
the >10k-column dim_P rank-share (the expensive separate analysis); purity is kept strictly separate from rank shares.

Object. For gated layer l, D_i in R^{d x n_l} is the per-sample downstream map dF/dz_l^pre (flatocean.blocks.mlp_structure,
original affine coords; for a smooth activation D_i uses the activation derivative). Flattened to a vector of dim d*n_l.

NORM and COMMON normalization. dist(i,j) = ||D_i - D_j||_F / c,  c = mean_i ||D_i||_F (ONE common scalar over the sample
set): every map is rescaled by the SAME factor, PRESERVING relative gain differences (we do NOT normalize each map by its
own norm). c and the raw-norm spread are recorded.

Groups. Complete-linkage agglomerative clustering; a cut at epsilon yields groups of DIAMETER <= epsilon (no single-linkage
chaining). A COMMON epsilon grid is applied across stages so stages are directly comparable at the same epsilon.

Per (seed, stage, layer) we record: authenticated metadata (activation, epoch, checkpoint/preprocessing/analysis-index
sha256), c and raw-norm spread, the WITHIN-class and BETWEEN-class distance distributions (quantiles + 20-bin histogram on
a common edge set) so bimodality can be judged from the data, and per epsilon: n_groups, group-size quantiles
(distribution), singleton-GROUP fraction and singleton-SAMPLE fraction, purity, matched-SIZE random control (mean/std),
purity gap, ARI/NMI/AMI vs labels.

    python scripts/analysis/eps_downstream_clustering.py --tags "" gelu leaky --seeds 0 --stages high_accuracy epoch_250 late_endpoint --layers 3 2
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # experiments/
sys.path.insert(0, HERE)
import torch
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import run_dir, stage_epochs
from flatocean.analyses.clusters import purity, purity_baseline

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

OUT = os.path.join(HERE, "results", "clusters")
PROTOCOL = "eps-threshold diameter-controlled downstream-map clustering (2026-10-02, ext 2026-10-03); complete linkage; common-factor Frobenius normalization; common epsilon grid across stages"
QS = [1, 5, 25, 50, 75, 95, 99]


def dist_distribution(cond, y, iu_cache):
    same = (y[iu_cache[0]] == y[iu_cache[1]])
    within = cond[same]; between = cond[~same]
    hi = float(max(cond.max(), 1e-9)); edges = np.linspace(0.0, hi, 21)
    out = {"hist_edges": edges.tolist(),
           "within_class": {"count": int(within.size), "quantiles_pct": {str(q): float(np.percentile(within, q)) for q in QS},
                            "mean": float(within.mean()) if within.size else None, "hist": np.histogram(within, bins=edges)[0].tolist()},
           "between_class": {"count": int(between.size), "quantiles_pct": {str(q): float(np.percentile(between, q)) for q in QS},
                             "mean": float(between.mean()) if between.size else None, "hist": np.histogram(between, bins=edges)[0].tolist()}}
    return out


def cluster_layer(D, y, eps_grid, rng_seed=0):
    s, d, n = D.shape
    vecD = D.reshape(s, d * n).to(torch.float64)
    raw = vecD.norm(dim=1)
    c = float(raw.mean())
    Dn = (vecD / c).cpu().numpy().astype(np.float64)
    cond = pdist(Dn, metric="euclidean")
    Z = linkage(cond, method="complete")
    iu = np.triu_indices(s, k=1)
    dists = dist_distribution(cond, y, iu); del iu
    rows = []
    for eps in eps_grid:
        g = fcluster(Z, t=float(eps), criterion="distance")
        _, g = np.unique(g, return_inverse=True)
        sizes = np.bincount(g); sing = int((sizes == 1).sum())
        base = [purity(g, np.random.default_rng([rng_seed, i]).permutation(y)) for i in range(10)]
        rows.append({
            "epsilon": float(eps), "n_groups": int(len(sizes)),
            "size_quantiles_pct": {str(q): float(np.percentile(sizes, q)) for q in [50, 90, 99]},
            "size_min_median_max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
            "singleton_groups": sing, "singleton_group_fraction": float(sing / len(sizes)),
            "singleton_sample_fraction": float(sing / s),              # singleton groups hold 1 sample each
            "largest_group_fraction": float(sizes.max() / s),
            "purity": float(purity(g, y)),
            "purity_matched_size_random_mean": float(np.mean(base)),
            "purity_matched_size_random_std": float(np.std(base)),
            "purity_gap_over_random": float(purity(g, y) - np.mean(base)),
            "ARI_labels": float(adjusted_rand_score(y, g)),
            "NMI_labels": float(normalized_mutual_info_score(y, g)),
            "AMI_labels": float(adjusted_mutual_info_score(y, g)),
        })
    return {"samples": s, "d": d, "n": n, "flat_dim": d * n, "norm": "Frobenius (Euclidean on flattened d*n vector)",
            "common_factor_c_mean_frob_norm": c, "raw_frob_norm_min_median_max": [float(raw.min()), float(raw.median()), float(raw.max())],
            "distance_units": "||D_i-D_j||_F / c", "pairwise_distance_summary": {"min": float(cond.min()), "median": float(np.median(cond)), "max": float(cond.max())},
            "distance_distributions_within_between": dists, "n_classes": int(len(np.unique(y))), "eps_grid": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["", "gelu", "leaky"], help="activation run tags: '' = ReLU, gelu, leaky")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=["high_accuracy", "epoch_250", "late_endpoint"])
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 2])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], help="COMMON epsilon grid across stages")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for tag in a.tags:
        for seed in a.seeds:
            rd = run_dir("mnist", "mlp", seed, tag); stages = stage_epochs(rd)
            name = f"mnist_mlp{'_' + tag if tag else ''}_seed{seed}"
            out_path = os.path.join(OUT, f"{name}_dmap_eps.json")
            res = {"protocol": PROTOCOL, "activation": ("relu" if tag == "" else tag), "common_eps_grid": list(a.eps), "stages": {}}
            for stage in a.stages:
                ep = stages.get(stage, {}).get("checkpoint_epoch")
                if ep is None:
                    print(f"== {name} {stage}: no checkpoint; skip", flush=True); continue
                run_ = O.load_run("mnist", "mlp", seed, tag=tag, epoch=ep, device=a.device)
                X, y, idx = run_["data"]["analysis"]; y = y.cpu().numpy()
                if a.max_samples and a.max_samples < X.shape[0]:
                    X = X[:a.max_samples]; y = y[:a.max_samples]
                st = O.mlp_structure(run_["net"], X)
                meta = {"epoch": int(ep), "activation": st.get("activation", "relu"),
                        "checkpoint_sha256": P.sha256_file(os.path.join(rd, "checkpoints", f"epoch_{ep:04d}.pt")),
                        "preprocessing_sha256": (P.sha256_file(os.path.join(rd, "preprocessing.pt")) if os.path.exists(os.path.join(rd, "preprocessing.pt")) else None),
                        "analysis_index_sha256": P.sha256_index(np.asarray(idx)), "samples": int(X.shape[0])}
                sd = {"meta": meta}
                for l in a.layers:
                    t0 = time.time()
                    rec = cluster_layer(st["D"][l - 1][:X.shape[0]], y, a.eps); rec["seconds"] = time.time() - t0
                    sd[f"layer{l}"] = rec
                    dd = rec["distance_distributions_within_between"]
                    sweet = max(rec["eps_grid"], key=lambda r: r["purity_gap_over_random"])
                    print(f"== {name} {stage} ep{ep} L{l} ({meta['activation']}): c={rec['common_factor_c_mean_frob_norm']:.3g} "
                          f"within[25/50/75]={[round(dd['within_class']['quantiles_pct'][q],2) for q in ('25','50','75')]} "
                          f"between[25/50/75]={[round(dd['between_class']['quantiles_pct'][q],2) for q in ('25','50','75')]} | "
                          f"best-gap eps={sweet['epsilon']} G={sweet['n_groups']} purity {sweet['purity']:.3f} rand {sweet['purity_matched_size_random_mean']:.3f} "
                          f"singG {sweet['singleton_group_fraction']:.2f} singS {sweet['singleton_sample_fraction']:.2f} ARI {sweet['ARI_labels']:.3f} ({rec['seconds']:.0f}s)", flush=True)
                res["stages"][stage] = sd
                if a.max_samples == 0:
                    tmp = out_path + ".tmp"; json.dump(res, open(tmp, "w"), indent=0); os.replace(tmp, out_path)
            print(f"saved {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
