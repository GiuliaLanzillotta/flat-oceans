"""Shared analysis identity and reuse validation for cached measurements (factors, audits, ray stages,
structured categories, overlap pairs).

An analysis identity binds a measurement to its inputs: the actual checkpoint bytes, the preprocessing
file, the ordered analysis-sample identifiers, the analysis size and dtype, and the configuration that
determines the numbers (probe seeds, radii, rank tolerance, tolerance grid, coordinate convention,
protocol identifier).  Its digest is the SHA-256 of the canonical JSON of these fields.  Code revisions are
recorded alongside but are not part of the identity; a code change that alters numbers must bump the
relevant *_VERSION constant, which is part of the identity.

Reuse rules (enforced by `require_match`): a cached record is reused only when its stored identity
equals the current identity field by field.  Records written before identities existed carry only a
checkpoint hash (or nothing); they are accepted only with an explicit legacy flag, and then their status
string is propagated into every record derived from them.  Artifacts (factor files) are additionally
bound by the SHA-256 of their bytes, so an audit entry certifies one specific file.
"""
import hashlib
import json
import os

import numpy as np

IDENTITY_VERSION = 1


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_index(idx):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes()).hexdigest()


def analysis_identity(run, rd, ep, config):
    """Identity of one analysis of checkpoint `ep` of run directory `rd` under configuration `config`
    (a JSON-serializable dict of every setting that determines the numbers)."""
    X, y, idx = run["data"]["analysis"]
    ident = {"identity_version": IDENTITY_VERSION,
             "checkpoint": f"checkpoints/epoch_{ep:04d}.pt",
             "checkpoint_sha256": sha256_file(os.path.join(rd, "checkpoints", f"epoch_{ep:04d}.pt")),
             "preprocessing_sha256": sha256_file(os.path.join(run["dir"], "preprocessing.pt")),
             "analysis_index_sha256": sha256_index(idx), "analysis_size": int(len(idx)),
             "analysis_dtype": "float64", "config": config}
    ident["digest"] = identity_digest(ident)
    return ident


def identity_digest(ident):
    core = {k: v for k, v in ident.items() if k != "digest"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=_jsonable).encode()).hexdigest()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (tuple, list, np.ndarray)):
        return [_jsonable(x) for x in o]
    raise TypeError(str(type(o)))


class ProvenanceError(RuntimeError):
    pass


def require_match(stored, current, label):
    """Raise unless the stored identity equals the current one.  Returns a status string."""
    if stored is None:
        raise ProvenanceError(f"{label}: no stored identity")
    if stored.get("digest") == current["digest"]:
        return "identity verified"
    diffs = [k for k in current if k != "digest" and json.dumps(stored.get(k), sort_keys=True, default=_jsonable) != json.dumps(current[k], sort_keys=True, default=_jsonable)]
    raise ProvenanceError(f"{label}: stored identity differs from the current analysis in {diffs}")


def legacy_status(stored_partial, current, label, allow_legacy):
    """For a record written before identities existed: compare whatever it stored (a checkpoint hash, probe
    seeds) with the current identity.  Any recorded field that differs is a refusal; fields it never recorded
    are reported as unverified.  Accepted only with allow_legacy; the returned status must be propagated."""
    checks = {"checkpoint_sha256": stored_partial.get("checkpoint_sha256")}
    bad = [k for k, v in checks.items() if v is not None and v != current[k]]
    if bad:
        raise ProvenanceError(f"{label}: recorded {bad} differ from the current analysis")
    if not allow_legacy:
        raise ProvenanceError(f"{label}: legacy record without a full analysis identity (preprocessing and sample order unrecorded); pass --allow-legacy to reuse it with unverified status")
    known = [k for k, v in checks.items() if v is not None]
    return f"legacy record, UNVERIFIED: {', '.join(known) or 'nothing'} matched; preprocessing and ordered samples not recorded at compute time"


# ----------------------------------------------------------------------------- audits of factor files
AUDIT_VERSION = 2
AUDIT_PASS = {"kernel_columns_max_over_eta": 2.0,       # every stored kernel column has exposure <= 2 x (1e-9 sigma_1 / sqrt s)
              "spectral_consistency_max_rel": 1e-4,     # stored (sigma_j, v_j) with sigma_j >= 1e-3 sigma_1 reproduce ||J v_j||/sqrt s to 1e-4
              "orthonormality_max_abs": 1e-6}


def audit_passes(entry):
    """An audit entry certifies its factor only when it is bound to the file bytes and to the analysis
    identity, was produced by the current audit procedure, and meets every pass criterion, including the
    spectral-consistency check against the current Jacobian (which is what ties the stored spectrum to the
    current checkpoint; an empty kernel alone proves nothing)."""
    if entry.get("audit_version") != AUDIT_VERSION or "factor_sha256" not in entry or "identity_digest" not in entry:
        return False, "audit entry lacks binding or is from an older audit version"
    sc = entry.get("spectral_consistency")
    if not sc or sc.get("max_rel_deviation") is None or sc["max_rel_deviation"] > AUDIT_PASS["spectral_consistency_max_rel"]:
        return False, "spectral consistency check missing or failed"
    if entry.get("dim_K_1e-9", 0) > 0:
        k = entry.get("kernel_columns_exposure", {})
        if k.get("max_over_eta") is None or k["max_over_eta"] > AUDIT_PASS["kernel_columns_max_over_eta"]:
            return False, "kernel column exposure exceeds the allowance"
        if entry.get("kernel_basis_orthonormality_max_abs", 1.0) > AUDIT_PASS["orthonormality_max_abs"]:
            return False, "kernel basis not orthonormal to tolerance"
    return True, "audited"


def legacy_factor_status(path, audit_entry, current_identity, allow_legacy):
    """Status of a factor file without a manifest: accepted as 'legacy, audited' only if a passing audit entry
    is bound to this file's bytes and to the current identity digest."""
    if audit_entry is not None:
        ok, why = audit_passes(audit_entry)
        if ok and audit_entry["factor_sha256"] == sha256_file(path) and audit_entry["identity_digest"] == current_identity["digest"]:
            return f"legacy, audited (audit v{AUDIT_VERSION}: kernel columns within {AUDIT_PASS['kernel_columns_max_over_eta']} x rank scale, spectral consistency {audit_entry['spectral_consistency']['max_rel_deviation']:.1e})"
        why = why if ok is False else "audit entry bound to different file bytes or analysis identity"
    else:
        why = "no audit entry"
    if allow_legacy:
        return f"legacy, UNVERIFIED (allow_legacy; {why})"
    raise ProvenanceError(f"factor file {path}: {why}; run the audit phase or pass --allow-legacy")


# ----------------------------------------------------------------------------- summaries over seeds
def per_seed_ratio(values_by_seed):
    """values_by_seed: {seed: (ratio_median_or_None, n_valid, n_undefined)}.  Returns a string that keeps
    seed, ratio and denominator counts together, and the aggregate over seeds with a valid ratio."""
    parts = []
    for s, (v, nv, nu) in values_by_seed.items():
        parts.append(f"seed {s}: {('n/a' if v is None else f'{v:.3g}')} ({nv} valid; {nu} undefined)")
    valid = [v for (v, nv, nu) in values_by_seed.values() if v is not None]
    agg = f"median over seeds {np.median(valid):.3g} [{min(valid):.3g}, {max(valid):.3g}]" if valid else "no seed with a valid ratio"
    return "; ".join(parts) + f"; {agg}"


# ----------------------------------------------------------------------------- tests
def _tests():
    import tempfile
    # identity digest depends on every field
    base = {"identity_version": 1, "checkpoint_sha256": "a", "preprocessing_sha256": "b", "analysis_index_sha256": "c", "analysis_size": 10, "analysis_dtype": "float64", "config": {"seeds": [1, 2]}}
    base["digest"] = identity_digest(base)
    assert require_match(base, base, "x") == "identity verified"
    for k, v in (("preprocessing_sha256", "B"), ("analysis_index_sha256", "C"), ("config", {"seeds": [1, 3]})):
        other = dict(base); other[k] = v; other["digest"] = identity_digest(other)
        try:
            require_match(base, other, "x"); raise AssertionError(f"mismatch in {k} accepted")
        except ProvenanceError as e:
            assert k in str(e)
    # legacy record: recorded checkpoint must match; unrecorded fields only with allow_legacy, status propagated
    try:
        legacy_status({"checkpoint_sha256": "zzz"}, base, "old stage", True); raise AssertionError("stale checkpoint accepted")
    except ProvenanceError:
        pass
    try:
        legacy_status({"checkpoint_sha256": "a"}, base, "old stage", False); raise AssertionError("legacy accepted without flag")
    except ProvenanceError:
        pass
    assert "UNVERIFIED" in legacy_status({"checkpoint_sha256": "a"}, base, "old stage", True)
    # audited legacy factor: bound to bytes and identity; empty kernel needs the spectral check; altered bytes refused
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "epoch0000_2.npz"); np.savez(fp, singular_values=np.array([1.0, 0.5]), V=np.eye(2, dtype=np.float32))
        good = {"audit_version": AUDIT_VERSION, "factor_sha256": sha256_file(fp), "identity_digest": base["digest"], "dim_K_1e-9": 0,
                "spectral_consistency": {"max_rel_deviation": 1e-6}}
        assert legacy_factor_status(fp, good, base, False).startswith("legacy, audited")
        try:
            legacy_factor_status(fp, {"dim_K_1e-9": 0}, base, False); raise AssertionError("unbound empty-kernel entry accepted")
        except ProvenanceError:
            pass
        np.savez(fp, singular_values=np.array([1.0, 0.4]), V=np.eye(2, dtype=np.float32))            # contents changed
        try:
            legacy_factor_status(fp, good, base, False); raise AssertionError("changed bytes accepted")
        except ProvenanceError:
            pass
        other = dict(base); other["analysis_index_sha256"] = "C"; other["digest"] = identity_digest(other)
        np.savez(fp, singular_values=np.array([1.0, 0.5]), V=np.eye(2, dtype=np.float32))
        try:
            legacy_factor_status(fp, good, other, False); raise AssertionError("different identity accepted")
        except ProvenanceError:
            pass
        bad = dict(good); bad["dim_K_1e-9"] = 5; bad["kernel_columns_exposure"] = {"max_over_eta": 3.0}
        try:
            legacy_factor_status(fp, bad, base, False); raise AssertionError("exposure failure accepted")
        except ProvenanceError:
            pass
        assert "UNVERIFIED" in legacy_factor_status(fp, bad, base, True)
    # per-seed ratio reporting keeps seed, value and counts together
    s = per_seed_ratio({0: (None, 0, 32), 1: (2.0, 32, 0)})
    assert "seed 0: n/a (0 valid; 32 undefined)" in s and "seed 1: 2 (32 valid; 0 undefined)" in s and "median over seeds 2" in s, s
    print("provenance: identity, legacy, audit-binding and ratio tests passed")


if __name__ == "__main__":
    _tests()
