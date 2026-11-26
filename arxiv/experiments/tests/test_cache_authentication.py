"""A cached measurement is accepted only while the stored factor it consumed still authenticates against the current
configuration: same bytes is not enough (review of 2026-09-24, item 1)."""
import json

import numpy as np
import pytest

from flatocean import provenance as P
from flatocean import factors
from flatocean.analyses import cnn_channels as CC
from flatocean.analyses import neuronwise as NW


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setattr(factors, "FACTOR_DIR", str(tmp_path / "factors"))
    d = tmp_path / "factors" / "run"; d.mkdir(parents=True)
    ident = {"digest": "abc", "config": factors.factor_config("2"), "identity_version": 1}
    fp = d / "epoch0001_2.npz"; np.savez(fp, singular_values=np.array([1.0]), p=1, s=3, epoch=1, identity=json.dumps(ident))
    sha = P.sha256_file(str(fp)); art = tmp_path / "art.npz"; np.savez(art, unit_sv_layer2=np.ones((2, 2))); art_c = tmp_path / "c.npy"; np.save(art_c, np.ones((2, 2)))
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: ident)
    return dict(ident=ident, sha=sha, art=str(art), art_c=str(art_c))


def test_neuronwise_cached_layer_requires_authentication(cached, monkeypatch):
    rec = {"layers": {"2": {"factor_sha256": cached["sha"], "rank_source": "stored factor"}}}
    assert NW.layer_complete(rec, 2, cached["art"], "run", 1, None, None)
    changed = dict(cached["ident"], digest="other", config=dict(cached["ident"]["config"], rank_tolerance=1e-8))
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: changed)          # configuration changed, file unchanged
    assert not NW.layer_complete(rec, 2, cached["art"], "run", 1, None, None)


def test_cnn_cached_block_requires_authentication_and_artifact(cached, monkeypatch, tmp_path):
    rec = {"blocks": {"2": {"factor_sha256": cached["sha"]}}}
    assert CC.block_complete(rec, "2", cached["art_c"], "run", 1, None, None)
    assert not CC.block_complete(rec, "2", str(tmp_path / "missing.npy"), "run", 1, None, None)
    assert not CC.block_complete(rec, "3", cached["art_c"], "run", 1, None, None)
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: dict(cached["ident"], digest="other"))
    assert not CC.block_complete(rec, "2", cached["art_c"], "run", 1, None, None)
