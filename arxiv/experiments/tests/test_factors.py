"""Stored factors are authenticated before use: a factor whose identity does not match the current run raises, a factor
without a manifest and without a passing audit raises, and a manifest written under an older configuration is refused."""
import json, os

import numpy as np
import pytest

from flatocean import provenance as P
from flatocean import factors


def _write(tmp_path, name, ep, blk, ident, sv):
    d = tmp_path / "factors" / name; d.mkdir(parents=True, exist_ok=True)
    path = d / f"epoch{ep:04d}_{blk}.npz"
    extra = {"identity": json.dumps(ident)} if ident is not None else {}
    np.savez(path, singular_values=sv, p=sv.size, s=3, epoch=ep, **extra)
    return str(path)


def _current(digest, config):
    return {"digest": digest, "config": config, "checkpoint": "c", "identity_version": 1}


@pytest.fixture
def factor_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(factors, "FACTOR_DIR", str(tmp_path / "factors")); return tmp_path


def test_current_factor_is_accepted(factor_dir, monkeypatch):
    good = _current("abc", factors.factor_config("2")); _write(factor_dir, "run", 1, "2", good, np.array([1.0, 0.5]))
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: good)
    sv, p, s, sha, status = factors.factor_spectrum("run", 1, "2", None, None)
    assert status == "identity verified" and p == 2 and len(sha) == 64 and sv[0] == 1.0


def test_stale_identity_is_refused(factor_dir, monkeypatch):
    stored = _current("abc", factors.factor_config("2")); _write(factor_dir, "run", 1, "2", stored, np.array([1.0]))
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: _current("other", factors.factor_config("2")))
    with pytest.raises(P.ProvenanceError):
        factors.factor_spectrum("run", 1, "2", None, None)
    with pytest.raises(P.ProvenanceError):
        factors.kernel_basis("run", 1, "2", None, None)


def test_added_configuration_key_is_refused(factor_dir, monkeypatch):
    old_cfg = {k: v for k, v in factors.factor_config("2").items() if k != "storage"}
    _write(factor_dir, "run", 1, "2", _current("old", old_cfg), np.array([1.0]))
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: _current("new", factors.factor_config("2")))
    with pytest.raises(P.ProvenanceError, match="config"):
        factors.factor_spectrum("run", 1, "2", None, None)


def test_no_manifest_no_audit_is_refused(factor_dir, monkeypatch):
    _write(factor_dir, "run", 1, "2", None, np.array([1.0]))
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: _current("abc", factors.factor_config("2")))
    with pytest.raises(P.ProvenanceError):
        factors.factor_spectrum("run", 1, "2", None, None)
    assert factors.factor_spectrum("run", 2, "2", None, None) is None      # no file at all: None, not an error


def test_factor_at_wrong_epoch_path_is_refused(factor_dir, monkeypatch):
    good = _current("abc", factors.factor_config("2")); sv = np.array([1.0])
    d = factor_dir / "factors" / "run"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "epoch0002_2.npz", singular_values=sv, p=1, s=3, epoch=1, identity=json.dumps(good))    # epoch-1 factor at the epoch-2 path
    monkeypatch.setattr(factors, "factor_identity", lambda run, rd, ep, blk: good)
    with pytest.raises(P.ProvenanceError, match="epoch"):
        factors.factor_spectrum("run", 2, "2", None, None)
