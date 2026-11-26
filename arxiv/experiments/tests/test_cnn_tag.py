"""Child-tag support: parent (tag='') and spawn-child (tag=...) records must not collide anywhere.

The patch-space/branch/factor/sibling records are all named `mnist_cnn[_<tag>]_seed<seed>`; with a non-empty tag
the name -- and therefore every output path built from it -- differs from the parent's, so a child run can never
overwrite or be read in place of the parent's factorization."""
from flatocean.analyses import cnn_patch_spaces as PS
from flatocean.analyses import persistence_siblings as SIB


def _name(seed, tag):
    return f"mnist_cnn{'_' + tag if tag else ''}_seed{seed}"


def test_parent_child_names_do_not_collide():
    tag = "lmc_cnn_s0_e250_c0"
    assert _name(0, "") != _name(0, tag)
    assert SIB.run_name(0, "") != SIB.run_name(0, tag)
    # every patch-space / selective / atomic basis path built from the two names is distinct per block and kind
    for kind in ("patch", "selective", "atomic"):
        for blk in ("conv5", "readout"):
            p_parent = PS.basis_path(_name(0, ""), 500, kind, blk)
            p_child = PS.basis_path(_name(0, tag), 500, kind, blk)
            assert p_parent != p_child, (kind, blk)
    # the factor / sibling record paths are likewise distinct
    assert SIB.factor_path(SIB.run_name(0, ""), 250, "conv5") != SIB.factor_path(SIB.run_name(0, tag), 500, "conv5")
