"""Embed layerwise bases in the network's actual parameter ordering."""
import torch
from flatocean import frozen_lib as FL
from flatocean.analyses import branch as B

def subspace_basis(net, nt, M, device):
    """Orthonormal flat basis (total_params x dim) of the layerwise space M, in the EXACT coordinate convention of
    branch.embed: each block's basis columns are written to the true flat [W;b] positions of that block (branch.embed),
    every other coordinate zero.  Blocks are disjoint in the flat vector and each block basis is orthonormal, so the
    columns are globally orthonormal.  Verified against branch.embed on a random member."""
    slices = {name: (a, b) for name, a, b in FL.block_slices(net)}
    total = sum(p.numel() for p in net.parameters())
    dim = sum(M[blk].shape[1] for blk in nt.blocks)
    Bf = torch.zeros(total, dim, dtype=torch.float64)
    c0 = 0
    for blk in nt.blocks:
        Mb = M[blk]; d = Mb.shape[1]
        if d == 0:
            continue
        n, m1 = nt.shape[blk]; m = m1 - 1
        A3 = Mb.reshape(n, m1, d)
        a, b = slices[blk]
        Bf[a:a + n * m, c0:c0 + d] = A3[:, :m, :].reshape(n * m, d)       # weight, row-major (n,m)
        Bf[a + n * m:b, c0:c0 + d] = A3[:, m, :].reshape(n, d)            # bias
        c0 += d
    if dim:
        gen = torch.Generator().manual_seed(0)
        cc = torch.randn(dim, generator=gen, dtype=torch.float64)
        mats = {}
        j = 0
        for blk in nt.blocks:
            dk = M[blk].shape[1]
            mats[blk] = (M[blk] @ cc[j:j + dk]).reshape(nt.shape[blk]) if dk else torch.zeros(nt.shape[blk], dtype=torch.float64)
            j += dk
        assert float((Bf @ cc - B.embed(net, nt, mats)).abs().max()) < 1e-9, "basis coordinate convention disagrees with branch.embed"
        ortho = float((Bf.T @ Bf - torch.eye(dim, dtype=torch.float64)).abs().max())
        assert ortho < 1e-8, f"basis not orthonormal (err {ortho:.1e})"
    return Bf.to(device)
