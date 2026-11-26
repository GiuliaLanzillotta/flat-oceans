"""Repository paths; override data and result locations with environment variables."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FLATOCEAN_DATA_ROOT", ROOT / "data")).expanduser()
RESULTS = Path(os.environ.get("FLATOCEAN_RESULTS_ROOT", ROOT / "results")).expanduser()
PAPER = ROOT.parent

def dataset(name):
    """Allow a dataset-specific override without embedding machine paths in source."""
    return Path(os.environ.get("FLATOCEAN_" + name.upper() + "_ROOT", DATA / name)).expanduser()


def recorded_result_path(value):
    """Resolve a saved result path after moving the repository, without editing its provenance."""
    path = Path(value).expanduser()
    if path.exists():
        return path
    parts = path.parts
    if "results" in parts:
        candidate = RESULTS.joinpath(*parts[parts.index("results") + 1:])
        if candidate.exists():
            return candidate
    return path
