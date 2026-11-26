"""Applications figures/tables from completed saved records; no new measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from . import style as S

ROOT = Path(S.HERE)
OUT = Path(S.F) / 'early_draft/applications'
PAPER = Path(S.TABLES)

def read(path):
    return json.loads((ROOT / 'results' / path).read_text())

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    S.save(fig, str(OUT / name))






