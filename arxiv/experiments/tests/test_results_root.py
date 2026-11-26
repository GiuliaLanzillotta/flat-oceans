"""FLATOCEAN_RESULTS_ROOT redirects every output path of every analysis module; FLATOCEAN_INPUT_ROOT the stored factors."""
import importlib, os, pkgutil, sys

import pytest


def _reload_all():
    for m in [m for m in list(sys.modules) if m == "flatocean" or m.startswith("flatocean.")]:
        del sys.modules[m]
    import flatocean, flatocean.runs, flatocean.factors, flatocean.directions, flatocean.analyses
    mods = [importlib.import_module(f"flatocean.analyses.{i.name}") for i in pkgutil.iter_modules(flatocean.analyses.__path__)]
    return flatocean, mods


def _path_attrs(mod):
    return {k: v for k, v in vars(mod).items() if isinstance(v, str) and k.isupper() and ("OUT" in k or k.endswith("DIR")) and os.sep in v}


def test_results_root_redirects_every_output(tmp_path, monkeypatch):
    root = str(tmp_path / "scratch"); prod = str(tmp_path / "prod")
    monkeypatch.setenv("FLATOCEAN_RESULTS_ROOT", root); monkeypatch.setenv("FLATOCEAN_INPUT_ROOT", prod)
    fo, mods = _reload_all()
    assert fo.runs.RESULTS_ROOT == root and fo.runs.INPUT_ROOT == prod
    assert fo.factors.FACTOR_DIR.startswith(prod)
    for k, v in _path_attrs(fo.directions).items():
        assert v.startswith(root), (k, v)
    seen = 0
    for m in mods:
        for k, v in _path_attrs(m).items():
            if k == "FACTOR_DIR":
                assert v.startswith(prod), (m.__name__, k, v)                 # the only input-root path
            else:
                assert v.startswith(root), (m.__name__, k, v); seen += 1     # every output path is under the results root
    assert seen >= 15
    # the persistence writers (factor phase, overlaps, geometry) point at the results root, their factor reads at the input root
    from flatocean.analyses import persistence, persistence_geometry
    assert persistence.OUT.startswith(root) and persistence.FACTOR_DIR.startswith(prod)
    assert persistence_geometry.OUT.startswith(root) and persistence_geometry.FACTOR_DIR.startswith(prod)
    # every consumer of stored factors looks for them under the input root (structured directions checks for missing ones by path)
    from flatocean.analyses import structured_directions, cnn_branch, cnn_patch_spaces
    assert structured_directions.PERSIST.startswith(prod) and structured_directions.OUT.startswith(root)
    assert cnn_branch.KP.FACTOR_DIR.startswith(prod) and cnn_patch_spaces.KP.FACTOR_DIR.startswith(prod)
    assert not os.path.exists(os.path.join(os.path.dirname(fo.runs.HERE), "results", "__never__"))


def test_default_root_is_production(monkeypatch):
    monkeypatch.delenv("FLATOCEAN_RESULTS_ROOT", raising=False); monkeypatch.delenv("FLATOCEAN_INPUT_ROOT", raising=False)
    fo, mods = _reload_all()
    assert fo.runs.RESULTS_ROOT == os.path.join(fo.runs.HERE, "results") and fo.runs.INPUT_ROOT == fo.runs.RESULTS_ROOT
    for m in mods:
        for k, v in _path_attrs(m).items():
            assert v.startswith(fo.runs.RESULTS_ROOT), (m.__name__, k, v)
