"""Channel-wise against cross-channel kernel directions of the MNIST CNN (plan Section 0.6 item 6).

dim K_l = sum_c dim ker J_lc + (sum_c rank J_lc - rank J_l), the first term split into the selective patch space (stored,
exp_cnn_patch_spaces, rank at the same rung) and a within-channel residual.  Ranks relative to sigma_1 of the whole block
(Reference.BLOCK_SIGMA1); the channel's own reference as a diagnostic.  Inputs (factor files, patch-space record) are
authenticated before any cache decision and their hashes are part of the identity.
"""
import json
import os
import time

import numpy as np
import torch

from flatocean import provenance as P
from flatocean import blocks, cli, factors, linalg, records, runs
from flatocean.linalg import Reference, key

OUT = lambda: runs.out_dir("branches")            # noqa: E731
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.6 item 6 (2026-09-24); identifier cnn-channels-v2 (blocks as independent measurements)"
LAD = linalg.TOL_LADDER


@torch.no_grad()
def channel_spectra(blk, chunk=100):
    """Singular values of every channel block J_lc (C x K) by QR accumulation of the channel's columns."""
    Rs = [None] * blk.C; groups = blk.unit_groups()
    for i in range(0, blk.s, chunk):
        rows = blk.rows(torch.arange(i, min(i + chunk, blk.s), device=blk.device))
        for c in range(blk.C):
            Jc = rows[:, groups[c]]; M = Jc if Rs[c] is None else torch.cat([Rs[c], Jc], 0); Rs[c] = torch.linalg.qr(M, mode="r").R[: blk.K]
    return np.stack([torch.linalg.svdvals(R).cpu().numpy() for R in Rs])


def decomposition(usv, sv_block, K, p, patch_block):
    """Per rung: channel-wise, cross-channel, selective patch (same rung), within-channel residual, consistency."""
    sigma1 = float(sv_block[0]); C = usv.shape[0]; out = {}
    rankJ = linalg.counts_only(sv_block, Reference.OWN)
    for t in LAD:
        k = key(t); rc = (usv > t * sigma1).sum(1); rc_own = np.array([(usv[c] > t * usv[c, 0]).sum() if usv[c, 0] > 0 else 0 for c in range(C)]); ker = K - rc
        if patch_block.get("channels") is not None:
            sel = np.array([K - q["rank_ladder"][k] for q in patch_block["channels"]], dtype=int)
        else:
            sel = np.full(C, K - patch_block["feature_rank_ladder"][k], dtype=int)
        within = ker - sel
        out[k] = {"dim_K": p - rankJ[k], "channelwise": int(ker.sum()), "cross_channel": int(rc.sum() - rankJ[k]), "selective_patch": int(sel.sum()),
                  "within_channel_residual": int(within.sum()), "consistent": bool((rc.sum() - rankJ[k]) >= 0 and (within >= 0).all()),
                  "channels_violating_patch_containment": int((within < 0).sum()), "channels_with_kernel": int((ker > 0).sum()),
                  "channelwise_own_reference": int((K - rc_own).sum()), "channel_kernels": ker.tolist(), "channel_selective": sel.tolist(),
                  "note": "selective patch rank at the same rung as the channel rank; channel ranks relative to sigma_1 of J_l"}
    return rankJ, out


def spectra_path(out, name, ep, blk):
    return os.path.join(out, f"{name}_epoch{ep:04d}_{blk}_channel_spectra.npy")


def patch_block_sha256(ps_block):
    """Hash of the patch-rank data of one block that the decomposition consumes (the per-channel selective ranks)."""
    import hashlib
    return hashlib.sha256(json.dumps(ps_block, sort_keys=True, default=str).encode()).hexdigest()


def block_complete(rec, blk, path, name, ep, run, rd, ps_block=None):
    """A block measurement is complete when its record and spectra exist, the factor it consumed is the same bytes and
    still authenticates against the current run and configuration, and the patch-rank data it consumed hash to the
    recorded value (2026-09-25; records without that field are recomputed)."""
    B = (rec.get("blocks") or {}).get(blk)
    if B is None or not os.path.exists(path):
        return False
    if ps_block is not None and B.get("patch_block_sha256") != patch_block_sha256(ps_block):
        return False
    fp = factors.factor_path(name, ep, blk)
    if not os.path.exists(fp) or P.sha256_file(fp) != B.get("factor_sha256"):
        return False
    try:
        factors.factor_status(fp, run, rd, blk, ep=ep)
    except P.ProvenanceError:
        return False
    return True


def run(a):
    from flatocean.analyses import cnn_patch_spaces as CP
    from flatocean import frozen_lib as FL
    assert (a.dataset, a.model) == ("mnist", "cnn"), "the channel decomposition is defined for the MNIST CNN"
    cli.configure(a); OUT_ = OUT()
    for seed in a.seeds:
        rd = runs.run_dir("mnist", "cnn", seed); name = f"mnist_cnn_seed{seed}"
        ps_all = json.load(open(os.path.join(OUT_, f"{name}_patch_spaces.json")))["stages"]
        store = records.ResultStore(os.path.join(OUT_, f"{name}_channels.json"), {"protocol": PROTOCOL})
        for stage, ep, _, _ in cli.stages_of(a, seed):
            run_ = runs.load_run("mnist", "cnn", seed, epoch=ep, device=a.device); net = run_["net"]; X = run_["data"]["analysis"][0]
            ps = ps_all.get(stage)
            if not ps or ps.get("identity", {}).get("digest") != CP.stage_identity(run_, rd, ep)["digest"] or not ps.get("complete"):
                raise P.ProvenanceError(f"{name} {stage}: patch-space record missing, incomplete or under another identity")
            # stage identity: protocol, ladder and the patch-space record; blocks are independent measurements (v2 layout)
            ident = records.identity(run_, rd, ep, {"protocol": PROTOCOL, "ladder": list(LAD), "patch_record_digest": ps["identity"]["digest"]})
            rec = store.stage(stage, ident)
            todo = [b for b in a.blocks if a.overwrite or not block_complete(rec, b, spectra_path(OUT_, name, ep, b), name, ep, run_, rd, ps["blocks"].get(b))]
            if not todo:
                print(f"== {name} {stage}: blocks {a.blocks} complete", flush=True); continue
            rec["epoch"] = ep; rec.setdefault("blocks", {}); masks = FL.base_masks(net, X)
            for blk in todo:
                t0 = time.time(); fs = factors.factor_spectrum(name, ep, blk, run_, rd)     # authenticated; raises on a mismatch
                if fs is None:
                    raise P.ProvenanceError(f"{name} {stage}: missing factor for {blk}")
                sv, p, _, sha, status = fs; B = blocks.CNNBlock(net, masks, X, blk)
                usv = channel_spectra(B); rankJ, dec = decomposition(usv, sv, B.K, p, ps["blocks"][blk])
                path = spectra_path(OUT_, name, ep, blk); tmp = path + ".tmp.npy"; np.save(tmp, usv); os.replace(tmp, path)   # artifact first
                rec["blocks"][blk] = {"p": p, "C": B.C, "K": B.K, "rank_J": rankJ, "gap_J": linalg.spectral_gap(sv, rankJ[linalg.DECLARED]), "sigma1": float(sv[0]),
                                      "decomposition": dec, "channel_sigma1": usv[:, 0].tolist(), "factor_sha256": sha, "factor_status": status,
                                      "patch_block_sha256": patch_block_sha256(ps["blocks"][blk]), "seconds": time.time() - t0}
                store.save(); c9 = dec[linalg.DECLARED]
                print(f"== {name} {stage} (ep {ep}) {blk}: dim K {c9['dim_K']} = channelwise {c9['channelwise']} (selective patch {c9['selective_patch']} + within-channel {c9['within_channel_residual']}) "
                      f"+ cross-channel {c9['cross_channel']}; consistent {c9['consistent']} ({time.time() - t0:.0f}s)", flush=True)


def main(argv=None):
    """Command line (python -m flatocean cnn-channels ...; formerly exp_cnn_channels.py, removed 2026-09-25):
        --seeds 0 1 2 --stages high_accuracy epoch_250 mse_interpolation late_endpoint --blocks conv5 readout
        test
    """
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["test"]:
        import pytest; return pytest.main(["-q", os.path.join(runs.HERE, "tests", "test_cnn_channels.py")])
    ap = cli.parser("CNN channel-wise decomposition", stages=["high_accuracy", "epoch_250", "mse_interpolation", "late_endpoint"])
    ap.set_defaults(dataset="mnist", model="cnn"); ap.add_argument("--blocks", nargs="+", default=["conv5", "readout"])
    run(ap.parse_args(argv))
