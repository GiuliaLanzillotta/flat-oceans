"""Reporting modules: importing them draws nothing and writes nothing; every registered command's module exposes main()."""
import importlib, os, pkgutil, sys


def test_importing_reporting_writes_nothing(tmp_path, monkeypatch):
    fig, tab = tmp_path / "fig", tmp_path / "tab"
    monkeypatch.setenv("FLATOCEAN_FIGURES_DIR", str(fig)); monkeypatch.setenv("FLATOCEAN_TABLES_DIR", str(tab))
    for m in [m for m in list(sys.modules) if m.startswith("flatocean")]:
        del sys.modules[m]
    import flatocean.reporting as R
    mods = [importlib.import_module(f"flatocean.reporting.{i.name}") for i in pkgutil.iter_modules(R.__path__)]
    assert len(mods) >= 17
    assert sorted(os.listdir(fig)) == [] and sorted(os.listdir(tab)) == []           # created, but empty: nothing drawn or written


def test_every_command_module_has_main():
    import flatocean.__main__ as M
    for cmd, (mod, old) in M.COMMANDS.items():
        module = importlib.import_module(f"flatocean.{mod}" if "." in mod else f"flatocean.analyses.{mod}")
        assert callable(getattr(module, "main", None)), cmd
