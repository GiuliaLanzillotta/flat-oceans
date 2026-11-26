"""layer_record on tiny synthetic layers in the production coordinates (column index a*n + j <-> A[j, a]).
Cases with independently known answers: a dead unit (delta 0), a selective unit (delta = m1 - |active|, all selective),
a genuine cross-neuron kernel (s < 2 m1, delta = 2 m1 - s, no neuron-wise part), a zero Jacobian, GELU without selective term."""
import numpy as np
import pytest
import torch

from flatocean import linalg
from flatocean.analyses import neuronwise as NW
from flatocean.linalg import key

DEC = key(1e-9)


class TinyBlock:
    """Hidden layer with n units, augmented inputs h_i (s x m1), downstream maps D_i (d x n); J_i[c, a*n + j] = D_i[c, j] h_i[a]."""
    def __init__(self, H, D, Z, name="2"):
        self.H, self.D, self.Z, self.name = H, D, Z, name; self.s, self.m1 = H.shape; self.d, self.n = D.shape[1:]; self.p = self.m1 * self.n
    def rows(self, idx):
        return torch.einsum("ia,icj->icaj", self.H[idx], self.D[idx]).reshape(len(idx) * self.d, self.p)
    def unit_rows(self, j, Hd):
        return self.D[:, :, j].reshape(-1, 1) * Hd


def spectrum(blk):
    J = blk.rows(torch.arange(blk.s)); sv, Vh, _ = linalg.svd_factor(torch.linalg.qr(J, mode="r").R if blk.s * blk.d > blk.p else J, vectors=True)
    sv = np.concatenate([sv, np.zeros(blk.p - sv.size)]); lad = linalg.rank_ladder(sv, linalg.Reference.OWN); r = lad["counts"][DEC]
    return dict(rank_J=lad["counts"], gap_J=NW.gaps(sv, lad), sigma1_J=float(sv[0]), rank_source="computed here", rank_verified=True), Vh[r:].T


def test_rows_match_production_convention():
    """The tiny block flattens like flatocean.blocks: unit j occupies the columns a*n + j."""
    g = torch.Generator().manual_seed(0); H = torch.randn(3, 4, generator=g, dtype=torch.float64); D = torch.randn(3, 2, 5, generator=g, dtype=torch.float64)
    blk = TinyBlock(H, D, torch.ones(3, 5, dtype=torch.float64)); J = blk.rows(torch.arange(3))
    for j in range(5):
        assert torch.allclose(J[:, j::5], blk.unit_rows(j, H.repeat_interleave(2, 0)))
    assert torch.allclose(J[1 * 2 + 1, 2 * 5 + 3], D[1, 1, 3] * H[1, 2])      # row i*d + c, column a*n + j


def relu_block(seed, s, m1, n, d, active):
    """active: list of boolean masks per unit; D_i restricted to the active units, generic otherwise."""
    g = torch.Generator().manual_seed(seed); H = torch.randn(s, m1, generator=g, dtype=torch.float64); H[:, -1] = 1
    Z = torch.stack([torch.where(torch.as_tensor(a), 1.0, -1.0).double() for a in active], 1)
    D = torch.randn(s, d, n, generator=g, dtype=torch.float64) * (Z > 0).double()[:, None, :]
    return TinyBlock(H, D, Z)


def test_dead_unit_delta_zero():
    s, m1 = 40, 5; blk = relu_block(1, s, m1, 2, 1, [np.ones(s, bool), np.zeros(s, bool)])
    rec, usv = NW.layer_record(blk, *spectrum(blk), "relu"); c = rec["decomposition"][DEC]; pd = rec["deficit_decomposition"]
    assert rec["r_h"][DEC] == m1 and rec["r_d"][DEC] == 1 and rec["rank_J"][DEC] == m1
    assert (c["dim_K"], c["neuronwise"], c["cross_neuron"]) == (m1, m1, 0) and c["consistent"]
    assert pd["delta_joint"] == 0 and pd["kernel_projected"]["counts"][DEC] == 0 and pd["kernel_projected_equals_delta"]
    assert pd["contributions@1e-09"] == {"selective": 0, "other_neuronwise": 0, "neuronwise_total": 0, "genuine_cross_neuron": 0, "consistent": True}
    assert rec["units"]["active_samples"][1] == 0 and rec["units"]["selective_dim@1e-09"][1] == m1 and usv.shape == (2, m1)


def test_selective_unit_gives_the_whole_deficit():
    """unit 0 active on 3 samples, unit 1 on the other 10 (m1 = 5): rank J = 3 + 5, r_h r_d = 10, delta = 2 = selective."""
    s, m1, k = 13, 5, 3; act0 = np.arange(s) < k; blk = relu_block(2, s, m1, 2, 1, [act0, ~act0])
    rec, _ = NW.layer_record(blk, *spectrum(blk), "relu"); c = rec["decomposition"][DEC]; pd = rec["deficit_decomposition"]
    assert rec["rank_J"][DEC] == k + m1 and rec["r_h"][DEC] * rec["r_d"][DEC] == 2 * m1
    assert (c["dim_K"], c["neuronwise"], c["cross_neuron"]) == (m1 - k, m1 - k, 0)
    assert pd["delta_joint"] == m1 - k and pd["kernel_projected_equals_delta"]
    assert pd["contributions@1e-09"] == {"selective": m1 - k, "other_neuronwise": 0, "neuronwise_total": m1 - k, "genuine_cross_neuron": 0, "consistent": True}


def test_genuine_cross_neuron_kernel():
    """both units active everywhere, generic D, s = 8 < 2 m1 = 10: unit blocks have full rank m1, rank J = s,
    delta = r_h r_d - s = 2, all of it cross-neuron."""
    s, m1 = 8, 5; blk = relu_block(3, s, m1, 2, 1, [np.ones(s, bool), np.ones(s, bool)])
    rec, _ = NW.layer_record(blk, *spectrum(blk), "relu"); c = rec["decomposition"][DEC]; pd = rec["deficit_decomposition"]
    assert rec["rank_J"][DEC] == s and rec["r_h"][DEC] == m1 and rec["r_d"][DEC] == 2
    assert (c["dim_K"], c["neuronwise"], c["cross_neuron"]) == (2 * m1 - s, 0, 2 * m1 - s)
    assert pd["delta_joint"] == 2 * m1 - s and pd["kernel_projected_equals_delta"]
    assert pd["contributions@1e-09"] == {"selective": 0, "other_neuronwise": 0, "neuronwise_total": 0, "genuine_cross_neuron": 2 * m1 - s, "consistent": True}


def test_zero_jacobian():
    s, m1, n = 10, 4, 3; blk = relu_block(4, s, m1, n, 2, [np.zeros(s, bool)] * n)
    rec, usv = NW.layer_record(blk, *spectrum(blk), "relu"); c = rec["decomposition"][DEC]
    assert (c["dim_K"], c["neuronwise"], c["cross_neuron"]) == (m1 * n, m1 * n, 0) and c["consistent"]
    assert c["closest_unit_singular_value_to_cutoff_log10"] is None and c["min_unit_gap_at_cutoff"] is None
    assert rec["deficit_decomposition"]["delta_joint"] == 0 and rec["deficit_decomposition"]["kernel_projected_equals_delta"]
    assert (usv == 0).all()


def test_gelu_reports_no_selective_contribution():
    s = 40; blk = relu_block(5, s, 5, 3, 2, [np.ones(s, bool)] * 3)
    rec, _ = NW.layer_record(blk, *spectrum(blk), "gelu")
    assert rec["deficit_decomposition"]["selective_projected"] is None
    assert rec["decomposition"][DEC]["extra_neuronwise_beyond_selective"] is None


def test_resume_keeps_other_layers_spectra(tmp_path):
    p = str(tmp_path / "x_unit_spectra.npz")
    NW.save_spectra(p, {"unit_sv_layer2": np.ones((3, 2))}); NW.save_spectra(p, {"unit_sv_layer3": np.zeros((3, 2))})
    with np.load(p) as z:
        assert set(z.files) == {"unit_sv_layer2", "unit_sv_layer3"} and z["unit_sv_layer2"].sum() == 6
    rec = {"layers": {"2": {"rank_source": "computed here"}, "3": {"rank_source": "computed here"}}}
    assert NW.layer_complete(rec, 2, p, "run", 1, None, None) and NW.layer_complete(rec, 3, p, "run", 1, None, None)
    assert not NW.layer_complete(rec, 1, p, "run", 1, None, None)
    assert not NW.layer_complete(rec, 2, str(tmp_path / "missing.npz"), "run", 1, None, None)
    rec["layers"]["2"]["factor_sha256"] = "0" * 64                      # consumed a stored factor that no longer exists
    assert not NW.layer_complete(rec, 2, p, "run", 1, None, None)


def test_kronecker_spectrum_matches_dense_and_projected_count():
    """Constant downstream map (identity activation): the product spectrum gives the dense ranks at every rung and the
    Kronecker count of dim Pi(K) equals the dense projected kernel rank."""
    g = torch.Generator().manual_seed(7); s, m1, n, d = 30, 6, 4, 2
    H = torch.randn(s, m1, generator=g, dtype=torch.float64); H[:, -1] = 1; H[:, 2] = H[:, 0] * 2          # r_h = 5
    D0 = torch.randn(d, n, generator=g, dtype=torch.float64); D = D0.expand(s, d, n).clone()             # r_d = 2
    blk = TinyBlock(H, D, torch.ones(s, n, dtype=torch.float64))
    assert NW.constant_downstream(D)
    sv_k, _, _ = NW.kronecker_spectrum(H, D0); dense_spec, dense_kernel = spectrum(blk)
    assert linalg.rank_ladder(sv_k, linalg.Reference.OWN)["counts"] == dense_spec["rank_J"]
    spec_k = dict(dense_spec, kronecker={"sv_H": linalg.svdvals(H), "sv_D": linalg.svdvals(D0)})
    rec_k, _ = NW.layer_record(blk, spec_k, None, "identity"); rec_d, _ = NW.layer_record(blk, dense_spec, dense_kernel, "identity")
    assert rec_k["deficit_decomposition"]["kernel_projected"]["counts"][DEC] == rec_d["deficit_decomposition"]["kernel_projected"]["counts"][DEC]
    assert rec_k["deficit_decomposition"]["delta"] == 0 and rec_k["deficit_decomposition"]["identities"]["dim_Pi_K_equals_delta"]
    assert rec_k["decomposition"][DEC]["cross_neuron"] == rec_d["decomposition"][DEC]["cross_neuron"] > 0     # raw cross term = downstream-null freedoms


def test_values_only_record_flags_missing_kernel_check(tmp_path):
    s, m1 = 8, 5; blk = relu_block(3, s, m1, 2, 1, [np.ones(s, bool), np.ones(s, bool)])
    spec, _ = spectrum(blk); rec, _ = NW.layer_record(blk, spec, None, "relu"); pd = rec["deficit_decomposition"]
    assert pd["kernel_projected"] is None and pd["identities"]["dim_Pi_K_equals_delta"] is None and pd["delta"] == 2 * m1 - s
    assert pd["delta_neuron"] + pd["delta_cross"] == pd["delta"]
    p = str(tmp_path / "x.npz"); NW.save_spectra(p, {"unit_sv_layer1": np.ones((2, 2))})
    rec_v = {"layers": {"1": dict(rec, pass_="values-only")}}; rec_v["layers"]["1"]["pass"] = "values-only"
    assert NW.layer_complete(rec_v, 1, p, "run", 1, None, None, values_only=True)          # complete for the sweep
    assert not NW.layer_complete(rec_v, 1, p, "run", 1, None, None, values_only=False)     # a full pass must redo it (delta > 0)
    rec_v["layers"]["1"]["deficit_decomposition"]["delta"] = 0
    assert NW.layer_complete(rec_v, 1, p, "run", 1, None, None, values_only=False)         # zero deficit: nothing to project


def test_historical_spectrum_requires_matching_checkpoint_hash(tmp_path, monkeypatch):
    import json, hashlib
    monkeypatch.setattr(NW.runs, "RESULTS_ROOT", str(tmp_path))       # the runs module NW actually uses (other tests reload the package)
    (tmp_path / "atomic").mkdir(); rd = tmp_path / "run"; (rd / "checkpoints").mkdir(parents=True)
    ck = rd / "checkpoints" / "epoch_0005.pt"; ck.write_bytes(b"weights"); sha = hashlib.sha256(b"weights").hexdigest()
    json.dump({"checkpoints/epoch_0005.pt": sha}, open(rd / "checksums.json", "w"))
    json.dump({"results": {"late_endpoint": {"epoch": 5, "layers": [{"p": 4, "rank_J": {"1e-06": 3, "1e-09": 3, "1e-12": 3}}]}}}, open(tmp_path / "atomic" / "r.json", "w"))
    np.savez(tmp_path / "atomic" / "r_epoch0005_spectra.npz", J1=np.array([2.0, 1.0, 0.5]))
    sv, prov = NW.atomic_spectrum("r", "late_endpoint", 5, str(rd), 0)
    assert sv.tolist() == [2.0, 1.0, 0.5, 0.0] and prov["checkpoint_hash_matches_checksums"] and not prov["record_has_identity_manifest"]
    assert NW.atomic_spectrum("r", "late_endpoint", 6, str(rd), 0) is None                 # another epoch: not usable
    ck.write_bytes(b"tampered")
    from flatocean import provenance as P
    with pytest.raises(P.ProvenanceError):
        NW.atomic_spectrum("r", "late_endpoint", 5, str(rd), 0)


def test_kronecker_ladder_matches_dense_at_every_rung():
    """Both paths fix the kernel and the marginal spaces at the declared rung; the projected counts agree on all rungs."""
    g = torch.Generator().manual_seed(11); s, m1, n, d = 40, 6, 5, 3
    H = torch.randn(s, m1, generator=g, dtype=torch.float64); H[:, -1] = 1; H[:, 3] = H[:, 1] + 1e-7 * H[:, 0]     # a 1e-7-scale direction
    D0 = torch.randn(d, n, generator=g, dtype=torch.float64); D0[:, 4] = 1e-8 * D0[:, 0]; D = D0.expand(s, d, n).clone()
    blk = TinyBlock(H, D, torch.ones(s, n, dtype=torch.float64)); dense_spec, dense_kernel = spectrum(blk)
    spec_k = dict(dense_spec, kronecker={"sv_H": linalg.svdvals(H), "sv_D": linalg.svdvals(D0)})
    rec_k, _ = NW.layer_record(blk, spec_k, None, "identity"); rec_d, _ = NW.layer_record(blk, dense_spec, dense_kernel, "identity")
    assert rec_k["deficit_decomposition"]["kernel_projected"]["counts"] == rec_d["deficit_decomposition"]["kernel_projected"]["counts"]


def test_historical_record_is_revalidated_and_replaced_by_fresh_requests(tmp_path, monkeypatch):
    import json, hashlib
    monkeypatch.setattr(NW.runs, "RESULTS_ROOT", str(tmp_path)); (tmp_path / "atomic").mkdir(); rd = tmp_path / "run"; (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints" / "epoch_0005.pt").write_bytes(b"w"); json.dump({"checkpoints/epoch_0005.pt": hashlib.sha256(b"w").hexdigest()}, open(rd / "checksums.json", "w"))
    json.dump({"results": {"late_endpoint": {"epoch": 5, "layers": [{"p": 3, "rank_J": {"1e-06": 3, "1e-09": 3, "1e-12": 3}}]}}}, open(tmp_path / "atomic" / "r.json", "w"))
    npz = tmp_path / "atomic" / "r_epoch0005_spectra.npz"; np.savez(npz, J1=np.array([2.0, 1.0, 0.5]))
    sv, prov = NW.atomic_spectrum("r", "late_endpoint", 5, str(rd), 0)
    art = str(tmp_path / "u.npz"); NW.save_spectra(art, {"unit_sv_layer1": np.ones((2, 2))})
    rec = {"layers": {"1": {"pass": "historical", "provisional": True, "historical_provenance": prov, "deficit_decomposition": {"delta": 0}}}}
    kw = dict(values_only=True, historical=True, stage="late_endpoint")
    assert NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), **kw)
    assert not NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), values_only=True, historical=False, stage="late_endpoint")   # fresh values-only pass replaces it
    assert not NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), values_only=False, historical=False, stage="late_endpoint")  # so does a full pass
    np.savez(npz, J1=np.array([2.0, 1.0, 0.4]))                                                                             # source spectrum changed
    assert not NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), **kw)
    npz.unlink()                                                                                                             # source spectrum missing
    assert not NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), **kw)


def test_historical_spectrum_must_reproduce_the_recorded_ladder(tmp_path, monkeypatch):
    import json, hashlib
    from flatocean import provenance as P
    monkeypatch.setattr(NW.runs, "RESULTS_ROOT", str(tmp_path)); (tmp_path / "atomic").mkdir(); rd = tmp_path / "run"; (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints" / "epoch_0005.pt").write_bytes(b"w"); json.dump({"checkpoints/epoch_0005.pt": hashlib.sha256(b"w").hexdigest()}, open(rd / "checksums.json", "w"))
    json.dump({"results": {"late_endpoint": {"epoch": 5, "layers": [{"p": 3, "rank_J": {"1e-06": 2, "1e-09": 2, "1e-12": 2}}]}}}, open(tmp_path / "atomic" / "r.json", "w"))
    np.savez(tmp_path / "atomic" / "r_epoch0005_spectra.npz", J1=np.array([2.0, 1.0, 0.5]))       # rank 3, record says 2
    with pytest.raises(P.ProvenanceError, match="rank ladder"):
        NW.atomic_spectrum("r", "late_endpoint", 5, str(rd), 0)
    rec = {"layers": {"1": {"pass": "historical", "historical_provenance": {"spectrum_sha256": "x", "recorded_rank_ladder": {}}, "deficit_decomposition": {"delta": 0}}}}
    art = str(tmp_path / "u.npz"); NW.save_spectra(art, {"unit_sv_layer1": np.ones((2, 2))})
    assert not NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), values_only=True, historical=True, stage="late_endpoint")
    rec["layers"]["1"]["deficit_decomposition"]["delta"] = -1                                       # a negative "deficit" is never accepted either
    assert not NW.layer_complete(rec, 1, art, "r", 5, None, str(rd), values_only=True, historical=True, stage="late_endpoint")
