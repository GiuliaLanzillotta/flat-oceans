"""Channel decomposition arithmetic on synthetic spectra: channel-wise + cross-channel = dim K, patch containment flagged."""
import numpy as np

from flatocean.analyses import cnn_channels as CC
from flatocean.linalg import key


def test_decomposition_arithmetic():
    K, p = 4, 12; usv = np.array([[1.0, 0.5, 1e-11, 0], [0.8, 0.3, 0.1, 0], [1e-11, 0, 0, 0]])
    sv_block = np.array([1.2, 0.9, 0.6, 0.4, 0.1, 1e-11, 0, 0, 0, 0, 0, 0])
    patch = {"channels": [{"rank_ladder": {key(t): r for t in (1e-6, 1e-9, 1e-12)}} for r in (2, 3, 0)]}
    rankJ, dec = CC.decomposition(usv, sv_block, K, p, patch); d = dec[key(1e-9)]
    assert rankJ[key(1e-9)] == 5 and d["dim_K"] == 7
    assert d["channelwise"] + d["cross_channel"] == d["dim_K"]
    assert d["channelwise"] == (4 - 2) + (4 - 3) + (4 - 0)
    assert d["selective_patch"] + d["within_channel_residual"] == d["channelwise"]
    assert d["consistent"] and d["channels_violating_patch_containment"] == 0


def test_patch_containment_violation_flagged():
    K, p = 4, 8; usv = np.array([[1.0, 0.5, 0.2, 0.1], [1.0, 0.5, 0.2, 0.1]]); sv_block = np.array([1.5, 1, 1, 0.5, 0.2, 0.1, 0.1, 0.05])
    patch = {"channels": [{"rank_ladder": {key(t): 1 for t in (1e-6, 1e-9, 1e-12)}}, {"rank_ladder": {key(t): 4 for t in (1e-6, 1e-9, 1e-12)}}]}
    _, dec = CC.decomposition(usv, sv_block, K, p, patch); d = dec[key(1e-9)]
    assert d["channels_violating_patch_containment"] == 1 and not d["consistent"]


def test_block_complete_binds_the_consumed_patch_ranks(tmp_path, monkeypatch):
    """A stored block is reused only while the patch-rank data it consumed hash to the recorded value; a record without the
    field (before 2026-09-25) is recomputed."""
    from flatocean.analyses import cnn_channels as CC
    from flatocean import factors
    from flatocean import provenance as P
    spectra = tmp_path / "s.npy"; spectra.write_bytes(b"x"); fp = tmp_path / "f.npz"; fp.write_bytes(b"factor")
    monkeypatch.setattr(factors, "factor_path", lambda name, ep, blk: str(fp)); monkeypatch.setattr(factors, "factor_status", lambda *a, **k: ("ok", None))
    ps_block = {"dim": 3, "channels": [{"free": 1}, {"free": 2}]}
    rec = {"blocks": {"conv5": {"factor_sha256": P.sha256_file(str(fp)), "patch_block_sha256": CC.patch_block_sha256(ps_block)}}}
    assert CC.block_complete(rec, "conv5", str(spectra), "n", 1, None, None, ps_block)
    assert not CC.block_complete(rec, "conv5", str(spectra), "n", 1, None, None, dict(ps_block, channels=[{"free": 1}, {"free": 3}]))
    del rec["blocks"]["conv5"]["patch_block_sha256"]
    assert not CC.block_complete(rec, "conv5", str(spectra), "n", 1, None, None, ps_block)
