"""Importing the library modules has no process-wide side effects (thread counts are set only by flatocean.cli.configure)."""
import importlib, os, sys

import torch


def test_ocean_lib_import_keeps_thread_count(monkeypatch):
    monkeypatch.setenv("OCEAN_THREADS", "3"); torch.set_num_threads(2)
    for m in ("flatocean.ocean_lib", "flatocean.runs", "flatocean.linalg", "flatocean.analyses.neuronwise"):
        sys.modules.pop(m, None); importlib.import_module(m)
    assert torch.get_num_threads() == 2
