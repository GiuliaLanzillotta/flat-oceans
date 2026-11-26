"""Rank ladders, references and the projected rank: the rank-scale bug of exp_clusters (thresholding against a foreign sigma_1)
and the cross-neuron cancellation fixture J = [H, 0] are the regression cases."""
import numpy as np
import torch

from flatocean import linalg
from flatocean.linalg import Reference, key


def test_ladder_reference_matters():
    sv = np.array([1.0, 1e-7, 1e-10, 1e-13])
    own = linalg.rank_ladder(sv, Reference.OWN)["counts"]
    foreign = linalg.rank_ladder(sv, Reference.BLOCK_SIGMA1, ref_value=1e7)["counts"]
    assert own == {key(1e-6): 1, key(1e-9): 2, key(1e-12): 3}
    assert foreign[key(1e-6)] == 0 and foreign[key(1e-9)] == 1 < own[key(1e-9)], "a foreign, larger reference lowers the counts (the exp_clusters scale bug)"
    absolute = linalg.rank_ladder(sv, Reference.ABSOLUTE)["counts"]
    assert absolute == own and linalg.rank_ladder(sv, Reference.OWN)["gap"] > 100


def test_ladder_empty_and_zero():
    assert linalg.rank_ladder(np.zeros(3), Reference.OWN)["counts"][key(1e-9)] == 0
    assert linalg.rank_ladder(np.array([]), Reference.OWN)["counts"][key(1e-9)] == 0


def test_projected_rank_removes_marginal_null():
    # kernel directions supported on rows outside the row space are invisible after P_C A P_H
    g = torch.Generator().manual_seed(0); B = torch.randn(6, 3, generator=g, dtype=torch.float64)
    B[3:] = 0                     # rows 3..5 are marginally null
    assert linalg.projected_rank(B)["counts"][key(1e-9)] == 3
    assert linalg.projected_rank(torch.zeros(6, 2, dtype=torch.float64))["counts"][key(1e-9)] == 0


def test_cross_neuron_cancellation_fixture():
    """J = [H, 0]: unit 2 is entirely null (neuron-wise), the joint rank equals rank H, so cross-neuron must be zero."""
    g = torch.Generator().manual_seed(1); H = torch.randn(20, 4, generator=g, dtype=torch.float64)
    J = torch.cat([H, torch.zeros(20, 4, dtype=torch.float64)], 1); sv = torch.linalg.svdvals(J).numpy(); sigma1 = sv[0]
    rank_J = linalg.rank_ladder(sv, Reference.OWN)["counts"][key(1e-9)]
    units = [linalg.counts_only(torch.linalg.svdvals(J[:, i * 4:(i + 1) * 4]).numpy(), Reference.BLOCK_SIGMA1, sigma1)[key(1e-9)] for i in range(2)]
    assert units == [4, 0] and sum(units) - rank_J == 0


def test_svd_factor_small():
    g = torch.Generator().manual_seed(2); R = torch.randn(5, 5, generator=g, dtype=torch.float64)
    sv, Vh, driver = linalg.svd_factor(R, vectors=True)
    assert np.allclose(sv, torch.linalg.svdvals(R).numpy()) and Vh.shape == (5, 5)
    assert np.allclose(np.abs(Vh.numpy() @ Vh.numpy().T), np.eye(5))
