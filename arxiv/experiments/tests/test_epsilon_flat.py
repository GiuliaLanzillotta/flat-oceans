"""epsilon_flat: survival curves, the half-survival radius, the downstream gains and the bound table on synthetic inputs."""
import numpy as np
import torch

from flatocean.analyses.epsilon_flat import bound_table, downstream_gains, flat_radius, survival


def test_survival_and_flat_radius():
    probes = {"0.1": [{"true": 1e-5, "frozen": 0.0}, {"true": 2e-3, "frozen": 0.0}], "1.0": [{"true": 5e-3, "frozen": 0.0}, {"true": 2e-2, "frozen": 0.0}], "10.0": [{"true": 1.0, "frozen": 0.0}] * 2}
    s = survival(probes, deltas=(1e-3, 1e-2))
    assert s["0.1"]["survival"] == {"0.001": 0.5, "0.01": 1.0} and s["1.0"]["survival"] == {"0.001": 0.0, "0.01": 0.5} and s["10.0"]["survival"]["0.01"] == 0.0
    assert flat_radius(s, 1e-2) == 1.0 and flat_radius(s, 1e-3) == 0.1 and flat_radius(s, 1e-3, level=0.9) is None


def test_downstream_gains_match_a_direct_product():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flatocean import tier1_train as T
    torch.manual_seed(0); net = T.MLP(5, [6, 4], 3, init_seed=1, dtype=torch.float64, activation="gelu"); X = torch.randn(7, 5, dtype=torch.float64)
    g = downstream_gains(net, X)
    out, Z, _ = net.forward(X, return_states=True); V = net.readout.weight; W2 = net.layers[1].weight
    S2 = net.act_prime(Z[1]).double()
    direct = torch.stack([torch.linalg.matrix_norm(V @ torch.diag(S2[i]) @ W2, ord=2) for i in range(7)])
    assert abs(g[0]["G_max"] - float(direct.max())) < 1e-10 and abs(g[1]["G_max"] - float(torch.linalg.matrix_norm(V, ord=2))) < 1e-12


def test_bound_table_uses_rms_displacement_and_beta():
    gains = [{"G_max": 2.0}, {"G_max": 1.0}]; beta = 0.8
    probes = {"1.0": [{"transition": 0.1, "preactivation_displacement": [0.5, 1.0], "curvature": [0.2, 0.3], "curvature_defined": [True, True]}]}
    t = bound_table(probes, gains, beta)["1.0"]
    assert abs(t["bound_rms_median"] - 0.5 * (2.0 * 0.8 * 0.25 + 1.0 * 0.8 * 1.0)) < 1e-12 and t["fraction_bound_holds"] == 1.0
    assert abs(t["curvature_coefficient_over_half_beta"]["1"]["median"] - 0.2 / 0.4) < 1e-12
