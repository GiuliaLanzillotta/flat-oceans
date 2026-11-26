"""
ε-flat radius and the smooth transition bound (GELU line, revised 2026-09-26, plan Section 0.8 item 10 (c)).

Reanalysis of stored records plus one forward pass per checkpoint:

  * survival curves S(r; delta) = fraction of the 256 structured-ray probes of a category (kernel_<block>, tail_<block>_<eps>)
    whose true output change ||F(theta_0 + r u) - F(theta_0)||_X is at most delta at radius r, for delta in a list, with the
    medians of the true and frozen changes; from results/transition_v2/<run>_structured.json (every category and stage);
  * the factors of the propagated activation-transition bound (theory, Theorem "smooth-radius", (7.11)):
        sup ||A_tr||_X <= 1/2 sum_l G_l beta_l d_l^2,
    with G_l the downstream gain (operator norm from a residual after activation l to the output, computed here at the
    base point as the largest per-sample spectral norm of D^post_{l,i}), beta the activation's curvature bound (the global
    maximum of |sigma''|, 0.798 for GELU), and d_l the preactivation displacement of the probe (the stored root-mean-square
    displacement; the theorem's d_l is the maximum over samples, which is not stored, so the value computed here is the
    bound evaluated with the mean-square displacement and is reported as such); from the all-block transition records
    results/transition_v3/<run>_{absolute,relative}_all.json (256 probes per radius, curvature fields), per probe: the
    measured transition norm, the bound with the RMS displacement, their ratio, and the measured curvature coefficient
    ||rho_l||_X / d_l^2 against beta / 2.

Records results/transition_v2/<run>_epsilon_flat.json (identity: checkpoint, preprocessing, analysis samples, configuration;
the consumed records bound by hash).  Smooth activations only for the bound part; ReLU and leaky get the survival part.

    python -m flatocean epsilon-flat --tag gelu --seeds 0 1 2
    python -m flatocean epsilon-flat --seeds 0 1 2                      # ReLU reference (survival curves only)
"""
import argparse
import json
import os

import numpy as np
import torch

from flatocean import provenance as P
from flatocean import blocks, records, runs
from flatocean.runs import stage_epochs

PROTOCOL = "epsilon-flat-v1.1 (2026-09-26; curvature coefficient read from records without the validity flag)"
DELTAS = (1e-4, 1e-3, 1e-2)
SMOOTH_BETA = {"gelu": 0.7978845608028654}                 # max |GELU''| = 2 phi(0) at u = 0


def key(x):
    return f"{x:g}"


def survival(probes, deltas=DELTAS):
    """probes: {radius: [ {true, frozen, ...}, ... ]} -> per radius the survival fractions and medians."""
    out = {}
    for r, lst in probes.items():
        t = np.array([q["true"] for q in lst], dtype=np.float64); f = np.array([q["frozen"] for q in lst], dtype=np.float64)
        out[r] = {"n": int(len(lst)), "survival": {key(d): float((t <= d).mean()) for d in deltas}, "true_median": float(np.median(t)), "true_q90": float(np.quantile(t, 0.9)),
                  "frozen_median": float(np.median(f))}
    return out


def flat_radius(surv, delta, level=0.5):
    """The largest radius at which at least `level` of the probes stay within delta (None when none does at the smallest radius)."""
    rs = sorted(surv, key=float); best = None
    for r in rs:
        if surv[r]["survival"][key(delta)] >= level:
            best = float(r)
        else:
            break
    return best


@torch.no_grad()
def downstream_gains(net, X):
    """G_l for every hidden layer l (1-based): the largest over samples of the spectral norm of the map from a residual after
    activation l to the output, with the base slopes downstream; for the last hidden layer it is ||V||_2 for every sample."""
    st = blocks.mlp_structure(net, X); L = st["L"]; Ws = [lin.weight for lin in net.layers]; V = net.readout.weight
    out = []
    for l in range(L):
        if l == L - 1:
            Dp = V.unsqueeze(0).expand(1, -1, -1)
        else:
            Dp = st["D"][l + 1] @ Ws[l + 1]                                     # s x d x n_l: V S_L W_L ... S_{l+1} W_{l+1}
        sv = torch.linalg.matrix_norm(Dp.double(), ord=2)
        out.append({"layer": l + 1, "G_max": float(sv.max()), "G_median": float(sv.median()), "n_samples": int(Dp.shape[0])})
    return out


def bound_table(probes, gains, beta):
    """Per radius: medians over probes of the measured transition, the bound with the RMS displacement, their ratio, and the
    per-layer measured curvature coefficient against beta / 2."""
    G = [g["G_max"] for g in gains]; out = {}
    for r, lst in probes.items():
        tr = []; bd = []; ratio = []; coef = {l: [] for l in range(len(G))}
        for q in lst:
            d = q.get("preactivation_displacement"); c = q.get("curvature")
            if d is None or len(d) != len(G):
                continue
            # records made before the validity flag (the all-block runs) carry the coefficient wherever the layer moved and it is a number
            cd = q.get("curvature_defined") or [bool(c is not None and c[l] is not None and d[l] > 0) for l in range(len(G))]
            b = 0.5 * sum(G[l] * beta * d[l] ** 2 for l in range(len(G)))
            tr.append(q["transition"]); bd.append(b); ratio.append(b / q["transition"] if q["transition"] > 0 else np.inf)
            for l in range(len(G)):
                if c is not None and cd[l] and c[l] is not None:
                    coef[l].append(c[l])
        if not tr:
            continue
        out[r] = {"n": len(tr), "transition_median": float(np.median(tr)), "bound_rms_median": float(np.median(bd)),
                  "ratio_bound_over_transition": {"median": float(np.median(ratio)), "min": float(np.min(ratio)), "q10": float(np.quantile(ratio, 0.1))},
                  "fraction_bound_holds": float(np.mean([x >= 1 for x in ratio])),
                  "curvature_coefficient_over_half_beta": {str(l + 1): ({"median": float(np.median(v) / (beta / 2)), "max": float(np.max(v) / (beta / 2))} if v else None) for l, v in coef.items()}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="epsilon-flat survival curves and the smooth transition bound from stored records")
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv); torch.set_num_threads(a.threads)
    v2 = runs.out_dir("transition_v2"); v3 = runs.out_dir("transition_v3")
    for seed in a.seeds:
        name = runs.run_name(a.dataset, a.model, seed, a.tag); rd = runs.run_dir(a.dataset, a.model, seed, a.tag)
        sp = os.path.join(v2, f"{name}_structured.json")
        if not os.path.exists(sp):
            print(f"== {name}: no structured-ray record; skipped", flush=True); continue
        S = json.load(open(sp)); s_sha = P.sha256_file(sp)
        tr = {}
        for mode in ("absolute", "relative"):
            tp = os.path.join(v3, f"{name}_{mode}_all.json")
            if os.path.exists(tp):
                tr[mode] = (json.load(open(tp)), P.sha256_file(tp))
        path = os.path.join(v2, f"{name}_epsilon_flat.json")
        rec = json.load(open(path)) if os.path.exists(path) and not a.overwrite else {"protocol": PROTOCOL, "run": name, "stages": {}}
        rec["protocol"] = PROTOCOL
        for stage, sres in S["results"].items():
            ep = int(sres["epoch"])
            run = runs.load_run(a.dataset, a.model, seed, tag=a.tag, epoch=ep, device="cpu", sets=("analysis",)); X = run["data"]["analysis"][0]
            act = getattr(run["net"], "activation", "relu")
            config = {"protocol": PROTOCOL, "deltas": [key(d) for d in a.deltas], "structured_sha256": s_sha, "transition_sha256": {m: v[1] for m, v in tr.items()}, "beta": SMOOTH_BETA.get(act)}
            ident = P.analysis_identity(run, rd, ep, config)
            old = rec["stages"].get(stage)
            if old and (old.get("identity") or {}).get("digest") == ident["digest"]:
                print(f"== {name} {stage}: complete under the current identity", flush=True); continue
            srec = {"epoch": ep, "identity": ident, "activation": act, "categories": {}, "bound": None}
            for cat, c in sres["categories"].items():
                if not c.get("probes") or c.get("dim", 0) == 0:
                    continue
                surv = survival(c["probes"], a.deltas)
                srec["categories"][cat] = {"dim": c["dim"], "eps": c.get("eps"), "block": c.get("block"), "survival": surv,
                                           "flat_radius_half": {key(d): flat_radius(surv, d) for d in a.deltas}, "identity_digest": c.get("identity_digest")}
            if act in SMOOTH_BETA and tr:
                gains = downstream_gains(run["net"], X); beta = SMOOTH_BETA[act]; srec["bound"] = {"beta": beta, "gains": gains, "modes": {}}
                for mode, (T, sha) in tr.items():
                    ts = (T.get("results") or {}).get(stage)
                    if ts and int(ts.get("epoch", ep)) == ep:
                        srec["bound"]["modes"][mode] = {"table": bound_table(ts["probes"], gains, beta), "checkpoint_sha256": ts.get("checkpoint_sha256")}
                bt = (srec["bound"]["modes"].get("absolute") or {}).get("table") or {}
                print(f"== {name} {stage} (ep {ep}): G {[round(g['G_max'], 3) for g in gains]}; bound/transition medians " + ", ".join(f"r={r}: {v['ratio_bound_over_transition']['median']:.2g}" for r, v in bt.items()), flush=True)
            fr = {cat: v["flat_radius_half"][key(1e-3)] for cat, v in srec["categories"].items() if cat.startswith(("kernel", "tail"))}
            print(f"== {name} {stage}: half-survival radius at delta 1e-3 {fr}", flush=True)
            rec["stages"][stage] = srec; records.dump_json(path, rec)


if __name__ == "__main__":
    main()
