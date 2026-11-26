"""persistence: the tau-dead coordinate rule (exact gates at tau = 0, thresholded derivative values otherwise)."""
import torch

from flatocean.analyses.persistence import dead_coordinates, dead_units


def test_dead_units_threshold():
    S = torch.tensor([[1.0, 0.0, 1e-7], [0.5, 0.0, 0.0], [1.0, 0.0, 1e-8]], dtype=torch.float64)
    assert dead_units(S) == [1] and dead_units(S, 1e-9) == [1] and dead_units(S, 1e-6) == [1, 2] and dead_units(S, 2.0) == [0, 1, 2]


def test_dead_coordinates_of_a_hidden_block_and_the_readout():
    s, m1, n, d = 3, 4, 3, 2
    st = {"L": 2, "d": d, "S": [torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),                        # layer 1: unit 1 dead
                              torch.tensor([[1.0, 1e-5, 0.0]] * s, dtype=torch.float64)],                                    # layer 2: unit 2 dead, unit 1 tau-dead at 1e-4
          "H": [None, torch.zeros(s, m1), torch.zeros(s, n + 1)], "D": [None, torch.zeros(s, d, n)]}
    dc = dead_coordinates(st, 2, s)
    assert dc["dead_input_columns"] == [1 * n + j for j in range(n)] and dc["dead_output_rows"] == [k * n + 2 for k in range(m1)]
    dc4 = dead_coordinates(st, 2, s, tau=1e-4)
    assert dc4["dead_output_rows"] == [k * n + j for k in range(m1) for j in (1, 2)]
    ro = dead_coordinates(st, "readout", s, tau=1e-4)
    assert ro["dead_input_columns"] == [k * d + c for k in (1, 2) for c in range(d)] and ro["dead_output_rows"] == []
