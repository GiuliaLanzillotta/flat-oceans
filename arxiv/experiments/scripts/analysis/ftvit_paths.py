"""Portable RESISC45 cache paths and deterministic direction seeds."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import os
import zlib

from flatocean.paths import RESULTS, dataset
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = str(RESULTS / "runs" / "ftvit_resisc45")
CACHE_DIR = os.environ.get("FTVIT_CACHE_DIR", os.path.join(RUN_DIR, "cache"))
DATA_ROOT = os.environ.get("FTVIT_DATA_ROOT", str(dataset("RESISC45")))

def _resolve(filename):
    cand = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cand):
        return cand
    return cand                                              # canonical path for a fresh write / clear error


def cache_path(filename):
    """Resolve a cache file (e.g. clsmid_s0_n10000.pt); ensure CACHE_DIR exists for writes."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return _resolve(filename)


def data_path(filename):
    """Resolve a decoded-data file (e.g. resisc45_train_imgs.npy)."""
    return _resolve(filename)


def stable_seed(base, *keys):
    """Process-independent seed from (base, *keys) -- replaces Python's randomized hash()."""
    key = "|".join(str(k) for k in keys)
    return (int(base) + zlib.crc32(key.encode("utf-8"))) % (2 ** 31)
