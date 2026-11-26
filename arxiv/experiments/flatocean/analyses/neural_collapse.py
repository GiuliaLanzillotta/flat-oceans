"""
Classic neural-collapse metrics per hidden layer (plan Section 0.6, item 4), on the analysis set.

Features: the post-activation h_l,i of every hidden layer (and the input for layer 0).  With class means mu_y, global mean mu_G,
within-class covariance Sigma_W and between-class covariance Sigma_B (G classes):
    NC1 = tr(Sigma_W Sigma_B^+) / G;
    NC2 = coefficient of variation of ||mu_y - mu_G|| and ||M^T M / ||M^T M||_F - ETF||_F with M the normalized centred means
          and ETF = (I - 11^T/G) / sqrt(G - 1) the simplex Gram matrix;
    NC3 = ||V^T / ||V||_F - M / ||M||_F||_F at the penultimate layer (V: readout weights, columns matched to classes);
    NC4 = disagreement between the nearest-class-centre rule on h_l and (a) the network prediction, (b) the label.

    python -m flatocean neural-collapse --tags "" leaky gelu identity --seeds 0 1 2
"""
# Ported unchanged from experiments/exp_neural_collapse.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean neural-collapse).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os

import numpy as np
from flatocean import records, runs
import torch

from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.runs import stage_epochs

OUT = runs.out_dir("neural_collapse")
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.6 item 4 (2026-09-24); identifier nc-v1"


@torch.no_grad()
def nc_metrics(Hf, y, pred, V=None):
    """Hf: s x q features, y: labels (0..G-1), pred: network predictions; V: d x q readout weights for NC3."""
    Hf = Hf.double(); G = int(y.max()) + 1; muG = Hf.mean(0); mus = torch.stack([Hf[y == c].mean(0) for c in range(G)])
    Xc = torch.cat([Hf[y == c] - mus[c] for c in range(G)]); SW = Xc.T @ Xc / Hf.shape[0]
    Mc = mus - muG; SB = Mc.T @ Mc / G
    nc1 = float(torch.trace(SW @ torch.linalg.pinv(SB, rtol=1e-9)) / G)
    norms = Mc.norm(dim=1); cv = float(norms.std() / norms.mean())
    Mn = Mc / norms[:, None].clamp_min(1e-300); gram = Mn @ Mn.T; etf = (torch.eye(G, dtype=Hf.dtype, device=Hf.device) - 1.0 / G) / np.sqrt(G - 1)
    nc2_gram = float((gram / gram.norm() - etf / etf.norm()).norm())
    out = {"NC1": nc1, "NC2_norm_cv": cv, "NC2_gram_etf": nc2_gram}
    if V is not None:
        out["NC3"] = float((V.double() / V.norm() - Mc / Mc.norm()).norm())
    d2 = torch.cdist(Hf, mus); ncc = d2.argmin(1)
    out["NC4_vs_prediction"] = float((ncc != pred).double().mean()); out["NC4_vs_label"] = float((ncc != y).double().mean())
    out["train_error_on_set"] = float((pred != y).double().mean())
    return out


def run(a):
    os.makedirs(OUT, exist_ok=True)
    for tag in a.tags:
        for seed in a.seeds:
            rd = O.run_dir("mnist", "mlp", seed, tag); name = f"mnist_mlp{'_' + tag if tag else ''}_seed{seed}"
            stages = stage_epochs(rd) if not a.match_stages_of_relu or tag == "" else stage_epochs(O.run_dir("mnist", "mlp", seed, ""))
            out_path = os.path.join(OUT, f"{name}.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "results": {}}
            saved = sorted(int(f[6:10]) for f in os.listdir(os.path.join(rd, "checkpoints")) if f.startswith("epoch_"))
            matched = a.match_stages_of_relu and tag != ""
            for st in a.stages:
                ep_req = stages.get(st, {}).get("checkpoint_epoch")
                if ep_req is None:
                    res["results"].setdefault("omitted_stages", {})[st] = "stage not reached by the source run"; continue
                ep = min(saved, key=lambda e: (abs(e - ep_req), e)) if matched else ep_req      # matched control: nearest saved checkpoint, offset recorded
                if not os.path.exists(os.path.join(rd, "checkpoints", f"epoch_{ep:04d}.pt")):
                    res["results"].setdefault("omitted_stages", {})[st] = f"checkpoint epoch {ep} absent"; continue
                run_ = O.load_run("mnist", "mlp", seed, tag=tag, epoch=ep, device=a.device); net = run_["net"]; X, y, _ = run_["data"]["analysis"]
                stage_label = {"high_accuracy": "ReLU-matched high-accuracy epoch", "mse_interpolation": "ReLU-matched interpolation epoch"}.get(st, st) if matched else st
                ident = P.analysis_identity(run_, rd, ep, {"protocol": PROTOCOL, "requested_epoch": ep_req, "stage_label": stage_label})
                if res["results"].get(st, {}).get("identity", {}).get("digest") == ident["digest"] and not a.overwrite:
                    continue
                with torch.no_grad():
                    out, Z, Hin = net.forward(X, return_states=True); pred = out.argmax(1)
                    feats = {"0": Hin[0]}
                    for l in range(len(Z)):
                        feats[str(l + 1)] = net.act(Z[l])
                rec = {"epoch": ep, "requested_epoch": ep_req, "epoch_offset": ep - ep_req, "stage_label": stage_label, "identity": ident, "layers": {}}
                for l, Hf in feats.items():
                    rec["layers"][l] = nc_metrics(Hf, y, pred, V=net.readout.weight if int(l) == len(Z) else None)
                res["results"][st] = rec
                print(f"== {name} {st} (ep {ep}): " + "; ".join(f"L{l}: NC1 {m['NC1']:.3g} NC2 {m['NC2_gram_etf']:.3f} NC4 {m['NC4_vs_label']:.3f}" + (f" NC3 {m['NC3']:.3f}" if "NC3" in m else "") for l, m in rec["layers"].items()), flush=True)
                records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


def _tests():
    torch.manual_seed(0); G = 4; q = 6
    mus = torch.eye(G, q, dtype=torch.float64) * 3; y = torch.arange(G).repeat_interleave(50)
    Hf = mus[y] + 0.01 * torch.randn(len(y), q, dtype=torch.float64)
    m = nc_metrics(Hf, y, y, V=torch.cat([torch.eye(G, dtype=torch.float64), torch.zeros(G, q - G, dtype=torch.float64)], 1))
    assert m["NC1"] < 1e-3 and m["NC2_gram_etf"] < 1e-2 and m["NC4_vs_label"] == 0.0, m
    print("nc tests passed", {k: round(v, 4) for k, v in m.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("phase", nargs="?", default="run", choices=["run", "test"])
    ap.add_argument("--tags", nargs="+", default=["", "leaky", "gelu", "identity"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stages", nargs="+", default=["initialization", "high_accuracy", "mse_interpolation", "epoch_100", "epoch_250", "late_endpoint"])
    ap.add_argument("--match-stages-of-relu", action="store_true", help="identity: use the ReLU stage epochs of the same seed"); ap.add_argument("--device", default="cuda"); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(); _tests() if a.phase == "test" else run(a)


