"""dim P of a partition: the chunked QR accumulation gives the same stacked-basis spectrum and ladder as one update per group."""
import numpy as np
import torch

from flatocean.analyses import clusters as CL


def test_chunked_accumulation_matches_per_group():
    g = torch.Generator().manual_seed(3); s, m1, n, d = 60, 5, 4, 2
    H = torch.randn(s, m1, generator=g, dtype=torch.float64); D = torch.randn(s, d, n, generator=g, dtype=torch.float64)
    groups = np.arange(s) % 7                                                       # 7 groups of 8-9 samples
    per_group = CL.dim_P(H, D, groups, "cpu", 0, 0, chunk_rows=1)                    # one update per group
    chunked = CL.dim_P(H, D, groups, "cpu", 0, 0, chunk_rows=10 ** 9)                # one update for everything
    mixed = CL.dim_P(H, D, groups, "cpu", 0, 0, chunk_rows=25)
    for r in (chunked, mixed):
        assert r["dim_P"] == per_group["dim_P"] and r["ladder"] == per_group["ladder"] and r["columns"] == per_group["columns"]
        assert abs(r["sigma1_stack"] - per_group["sigma1_stack"]) <= 1e-10 * per_group["sigma1_stack"]
    singletons = CL.dim_P(H, D, np.arange(s), "cpu", 0, 0)
    assert singletons["dim_P"] <= per_group["dim_P"]                                 # finer partition: no larger stacked space than... (monotone in the join)
