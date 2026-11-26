"""soft_selectivity: the per-unit profiles, the tau-activity classes and the layer profile lookup on synthetic inputs."""
import numpy as np

from flatocean.analyses.soft_selectivity import layer_profile, tau_activity, unit_profiles


def test_unit_profiles_count_singular_values_at_or_below_eps_on_the_scaled_block():
    s = 100; sv = np.array([[10.0, 1.0, 0.05, 0.0], [10.0, 10.0, 10.0, 0.0]])        # raw singular values; scaled by 1/sqrt(s) = 0.1
    pr = unit_profiles(sv, s, eps=(1e-2, 1e-3))
    assert pr["per_unit"]["0.01"] == [2, 1] and pr["per_unit"]["0.001"] == [1, 1] and pr["sum_over_units"] == {"0.01": 3, "0.001": 2}
    assert pr["sigma1_unit_over_sqrt_s"] == [1.0, 1.0]


def test_tau_activity_classes_and_counting_nullity():
    S = np.array([[0.5, 0.0, 1e-4], [0.5, 0.0, 1e-4], [0.5, 0.0, 0.0]]); m1 = 3            # unit 0 active everywhere, unit 1 dead, unit 2 weakly active on two samples
    ta = tau_activity(S, m1, taus=(1e-3, 1e-6))
    assert ta["0.001"]["active_samples"] == [3, 0, 0] and ta["0.001"]["dead_units"] == 2 and ta["0.001"]["rare_units"] == 0 and ta["0.001"]["counting_nullity"] == 6
    assert ta["1e-06"]["active_samples"] == [3, 0, 2] and ta["1e-06"]["dead_units"] == 1 and ta["1e-06"]["rare_units"] == 1 and ta["1e-06"]["counting_nullity"] == 3 + 1


def test_layer_profile_reads_k_abs_by_value():
    assert layer_profile({"k_abs": {"0.01": 5, "0.001": 3, "1e-06": 1}}, eps=(1e-2, 1e-3, 1e-4, 1e-6)) == {"0.01": 5, "0.001": 3, "0.0001": None, "1e-06": 1}
