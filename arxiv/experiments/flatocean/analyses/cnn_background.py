"""
conv1 background against digit positions in the transition decomposition of the MNIST CNN
(EXPERIMENTS_PLAN.md Section 0.4, item 9, last bullet).

A conv1 position whose 3 x 3 input patch is all background (the constant standardized value of a zero pixel,
including the zero padding at the border which differs from the background) sees the same patch for every sample,
so the channel's preactivation there is one constant shared by every sample and position; when that constant is
close to the activation boundary of a channel, a small perturbation flips all its background gates at once.  For
each of 32 probes (block-relative, all blocks, alpha = 0.01 and 0.1; matched seeds 100-131) this script splits the
conv1 activation residual rho_1 into its background-position part and its remaining part, and reports for each
part the departure ||rho||_X and the propagated contribution ||D^post_1 rho||_X, next to the full transition
residual and the contributions of the other layers.  The split is repeated with the exceptional channel (the one
with the smallest normalized background margin at the stage) removed from rho_1.

    python -m flatocean cnn-background --seeds 0 1 2 --stages high_accuracy mse_interpolation epoch_250 late_endpoint
"""
# Ported unchanged from experiments/exp_cnn_background.py on 2026-09-24; that script was removed on 2026-09-25 (command: python -m flatocean cnn-background).
from flatocean.runs import HERE                                   # the experiments directory (paths, git revision)
import argparse
import json
import os

import numpy as np
from flatocean import records, runs
import torch
import torch.nn.functional as F

from flatocean import frozen_lib as FL
from flatocean import ocean_lib as O
from flatocean import provenance as P
from flatocean.analyses import transitions as TR
from flatocean.runs import stage_epochs

OUT = runs.out_dir("transition_v3")
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.4 item 9, background split (2026-09-22); identifier cnn-background-v2"
# Position classes of conv1 (3 x 3 patches of the zero-padded input): "background" = every entry of the patch is the
# standardized background value (interior of the image; one constant preactivation per channel, shared by all
# samples); "boundary" = the patch contains padding (the padded zeros differ from the standardized background, so
# these preactivations are not the constant); "digit" = the patch contains a non-background pixel.  The
# "*_no_exc" quantities remove the exceptional channel's conv1 residual from the propagated sum: a contribution
# exclusion diagnostic on the unchanged network, not a model intervention.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--stages", nargs="+", default=["high_accuracy", "mse_interpolation", "epoch_250", "late_endpoint"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--nprobes", type=int, default=128); ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1])
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        rd = O.run_dir("mnist", "cnn", seed); stages = stage_epochs(rd); name = f"mnist_cnn_seed{seed}"
        out_path = os.path.join(OUT, f"{name}_background_split.json"); res = json.load(open(out_path)) if os.path.exists(out_path) else {"protocol": PROTOCOL, "alphas": a.alphas, "stages": {}}
        for stage in a.stages:
            ep = stages[stage]["checkpoint_epoch"]
            if ep is None:
                continue                                                          # completeness is decided below, after the identity check
            run = O.load_run("mnist", "cnn", seed, epoch=ep, device=a.device); net = run["net"]; X = run["data"]["analysis"][0]
            theta0 = FL.flat_params(net).clone(); masks = FL.base_masks(net, X, block=a.block); s = X.shape[0]
            bg = float(X[:, :, 0, 0].median())
            Xp = F.pad(X, (1, 1, 1, 1), value=0.0) - bg                                                          # the convolution pads X with zeros
            patch_bg = (F.conv2d(Xp.abs(), torch.ones(1, 1, 3, 3, dtype=X.dtype, device=a.device)) == 0)          # s x 1 x 32 x 32, all nine entries equal bg
            boundary = torch.zeros(1, 1, 32, 32, dtype=torch.bool, device=a.device); boundary[..., [0, -1], :] = True; boundary[..., :, [0, -1]] = True
            patch_digit = ~patch_bg & ~boundary
            conv = net.convs[0]; w = conv.weight; b = conv.bias
            zbg = bg * w.sum(dim=(1, 2, 3)) + b; wn = torch.sqrt((w ** 2).sum(dim=(1, 2, 3)) + b ** 2); m_bg = (zbg / (wn * float(np.sqrt(9 * bg * bg + 1)))).abs()
            exceptional = int(m_bg.argmin())
            slices = TR.block_slices(net)
            ident = P.analysis_identity(run, rd, ep, {"protocol": PROTOCOL, "alphas": a.alphas, "nprobes": a.nprobes, "classes": "background (interior constant patch), boundary (padding), digit"})
            old = res["stages"].get(stage)
            if old and old.get("identity", {}).get("digest") == ident["digest"] and all(len(old["probes"][str(al)]) >= a.nprobes for al in a.alphas):
                print(f"== {name} {stage}: complete under the current identity", flush=True); continue
            rec = {"epoch": ep, "background_positions_share": float(patch_bg.double().mean()), "boundary_positions_share": float(boundary.double().mean()),
                   "channel_background_margins": m_bg.tolist(), "exceptional_channel": exceptional, "identity": ident, "probes": {str(al): [] for al in a.alphas}}
            for k, ds in enumerate(TR.DIRECTION_SEEDS[: a.nprobes]):
                g = torch.Generator().manual_seed(ds); u = torch.randn(theta0.numel(), generator=g, dtype=torch.float64).to(a.device)
                u = TR.relative_direction(u, theta0, slices, "all")
                for al in a.alphas:
                    FL.set_flat(net, theta0 + al * u)
                    keys = ("bg", "boundary", "digit", "bg_no_exc", "boundary_no_exc", "digit_no_exc", "conv1_all", "others")
                    parts = {key: torch.zeros(s, net.readout.out_features, dtype=X.dtype, device=a.device) for key in keys}
                    dep = {key: 0.0 for key in keys}; flips_bg = 0; flips_bd = 0; flips_fg = 0; flips_exc = 0
                    for i in range(0, s, a.block):
                        idx = torch.arange(i, min(i + a.block, s), device=a.device)
                        _, Z = FL.true_states(net, X, idx)
                        rho1 = FL.local_residual(net, masks, 0, idx, Z[0])                                   # b x 16 x 32 x 32
                        mb = patch_bg[idx]; md = patch_digit[idx]                                             # b x 1 x 32 x 32
                        fl = ((Z[0] > 0) != masks[0][idx]); flips_bg += int((fl & mb).sum()); flips_bd += int((fl & boundary).sum()); flips_fg += int((fl & md).sum()); flips_exc += int(fl[:, exceptional].sum())
                        keep = torch.ones(16, dtype=X.dtype, device=a.device); keep[exceptional] = 0.0; kv = keep.view(1, -1, 1, 1)
                        variants = {"bg": rho1 * mb, "boundary": rho1 * boundary, "digit": rho1 * md, "bg_no_exc": rho1 * mb * kv, "boundary_no_exc": rho1 * boundary * kv, "digit_no_exc": rho1 * md * kv, "conv1_all": rho1}
                        for key, r_ in variants.items():
                            dep[key] += float(r_.pow(2).sum()); parts[key][idx] = FL.frozen_tail_linear(net, masks, idx, 0, r_)
                        oth = torch.zeros(len(idx), net.readout.out_features, dtype=X.dtype, device=a.device)
                        for l in range(1, len(Z)):
                            r_ = FL.local_residual(net, masks, l, idx, Z[l]); dep["others"] += float(r_.pow(2).sum())
                            if float(r_.abs().max()) > 0:
                                oth += FL.frozen_tail_linear(net, masks, idx, l, r_)
                        parts["others"][idx] = oth
                    FL.set_flat(net, theta0)
                    q = {"direction_seed": ds, "flips_background": flips_bg, "flips_boundary": flips_bd, "flips_digit": flips_fg, "flips_exceptional_channel": flips_exc}
                    for key in parts:
                        q[f"departure_{key}"] = float(np.sqrt(dep[key] / s)); q[f"propagated_{key}"] = FL.x_norm(parts[key])
                    q["propagated_total"] = FL.x_norm(parts["conv1_all"] + parts["others"]); q["propagated_total_no_exc"] = FL.x_norm(parts["bg_no_exc"] + parts["boundary_no_exc"] + parts["digit_no_exc"] + parts["others"])
                    rec["probes"][str(al)].append(q)
                if k % 8 == 7:
                    print(f"   {name} {stage}: {k + 1} probes", flush=True)
            res["stages"][stage] = rec
            for al in a.alphas:
                pr = rec["probes"][str(al)]; med = lambda key: float(np.median([q[key] for q in pr]))
                print(f"== {name} {stage} (epoch {ep}) alpha {al}: background {100 * rec['background_positions_share']:.1f}%, boundary {100 * rec['boundary_positions_share']:.1f}% of positions, exceptional channel {exceptional} (margin {m_bg[exceptional]:.2e}); "
                      f"flips bg {med('flips_background'):.0f} boundary {med('flips_boundary'):.0f} digit {med('flips_digit'):.0f} (exc. channel {med('flips_exceptional_channel'):.0f}); "
                      f"propagated bg {med('propagated_bg'):.2e} boundary {med('propagated_boundary'):.2e} digit {med('propagated_digit'):.2e} others {med('propagated_others'):.2e} total {med('propagated_total'):.2e}; excluding the exc. channel: total {med('propagated_total_no_exc'):.2e}", flush=True)
            records.dump_json(out_path + ".tmp", res); os.replace(out_path + ".tmp", out_path)


