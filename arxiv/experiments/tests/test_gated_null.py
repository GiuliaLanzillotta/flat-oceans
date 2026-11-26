"""gated_null: the selective-space splits (by unit type and by mechanism) add up, the rare-unit type carries rank deficiency
beyond counting, and the identity binds checkpoint, preprocessing and sample indices."""
import torch

from flatocean.analyses.gated_null import gated_null


def test_splits_add_up_and_rare_units_can_exceed_counting():
    torch.manual_seed(0); s, m1, n = 60, 8, 4
    H = torch.randn(s, m1, dtype=torch.float64); H[:, -1] = 1.0                 # augmented features
    S = torch.zeros(s, n, dtype=torch.float64)
    S[:, 1] = 1.0                                                             # unit 1: active everywhere (full rank, no nullity)
    S[10:13, 2] = 1.0                                                         # unit 2: rare, 3 active samples of full rank -> nullity m1 - 3 = counting
    S[:5, 3] = 1.0; H[1:5] = H[0]                                             # unit 3: rare, 5 active samples spanning rank 1 -> nullity m1 - 1 > m1 - 5
    g = gated_null(H, S)                                                      # unit 0: dead
    sp, mech = g["split"], g["split_mechanism"]
    assert sp["dead_units"] == m1 and sp["rank_deficient_on_large_active_set"] == 0
    assert sp["rarely_active_units"] == (m1 - 3) + (m1 - 1)                     # whole nullity of each rare unit, counting or not
    assert g["dim_gated_null"] == sum(sp.values()) == sum(mech.values())
    assert g["unit_level_counting_nullity"] == m1 + (m1 - 3) + (m1 - 5)         # dead row + counting parts of the two rare units
    assert mech == {"dead_rows": m1, "counting_beyond_dead": (m1 - 3) + (m1 - 5), "beyond_counting": (m1 - 1) - (m1 - 5)}
    assert g["unit_counts"] == {"dead_units": 1, "rarely_active_units": 2, "units_with_deficient_large_set": 0}


def test_identity_binds_samples(tmp_path):
    import os
    from flatocean.analyses.gated_null import sample_identity
    from flatocean import provenance as P
    ck = tmp_path / "checkpoints"; ck.mkdir(); (ck / "epoch_0007.pt").write_bytes(b"ckpt"); (tmp_path / "preprocessing.pt").write_bytes(b"pre")
    run = {"dir": str(tmp_path), "data": {"train": (None, None, [0, 1, 2]), "analysis": (None, None, [5, 6])}}
    a = sample_identity(run, str(tmp_path), 7, ("train",), {"set": "train"}); b = sample_identity(run, str(tmp_path), 7, ("analysis",), {"set": "analysis"})
    assert a["sample_size"] == 3 and b["sample_size"] == 2 and a["digest"] != b["digest"]
    assert a["checkpoint_sha256"] == P.sha256_file(str(ck / "epoch_0007.pt")) and a["preprocessing_sha256"] == P.sha256_file(str(tmp_path / "preprocessing.pt"))
    assert a["sample_index_sha256"] == P.sha256_index([0, 1, 2]) and P.identity_digest(a) == a["digest"]
