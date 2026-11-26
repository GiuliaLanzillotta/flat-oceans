"""Stage matching: the requested epoch of another run is mapped to the nearest saved checkpoint and the offset is recorded."""
import os

from flatocean import runs


def test_nearest_checkpoint(tmp_path):
    for e in (0, 5, 60, 500):
        (tmp_path / "checkpoints").mkdir(exist_ok=True); open(tmp_path / "checkpoints" / f"epoch_{e:04d}.pt", "w").close()
    assert runs.saved_checkpoints(str(tmp_path)) == [0, 5, 60, 500]
    assert runs.nearest_checkpoint(str(tmp_path), 61) == 60 and runs.nearest_checkpoint(str(tmp_path), 300) == 500 and runs.nearest_checkpoint(str(tmp_path), 280) == 60


def test_stage_order_and_labels():
    assert runs.STAGE_ORDER[0] == "initialization" and runs.STAGE_ORDER[-1] == "late_endpoint"
    assert set(runs.MATCHED_LABELS) >= {"high_accuracy", "mse_interpolation"}
