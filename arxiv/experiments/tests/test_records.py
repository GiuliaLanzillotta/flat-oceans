"""Result store: stale records under another identity are superseded, interrupted saves never corrupt the file, missing keys."""
import json, os

from flatocean import records


def test_stage_supersedes_other_identity(tmp_path):
    p = tmp_path / "r.json"; store = records.ResultStore(str(p), {"protocol": "t"})
    rec = store.stage("high_accuracy", {"digest": "a"}); rec["x"] = 1; store.save()
    store2 = records.ResultStore(str(p), {"protocol": "t"}); rec2 = store2.stage("high_accuracy", {"digest": "a"})
    assert rec2["x"] == 1
    rec3 = store2.stage("high_accuracy", {"digest": "b"})
    assert "x" not in rec3 and rec3["identity"]["digest"] == "b" and rec3["superseded_digest"] == "a"
    store2.save(); saved = json.load(open(p))
    assert saved["protocol"] == "t" and saved["results"]["high_accuracy"]["identity"]["digest"] == "b"


def test_atomic_save_keeps_previous_file(tmp_path, monkeypatch):
    p = tmp_path / "r.json"; store = records.ResultStore(str(p), {"protocol": "t"}); store.stage("s", {"digest": "a"})["x"] = 1; store.save()
    def boom(*a, **k): raise RuntimeError("disk full")
    monkeypatch.setattr(records.json, "dump", boom)
    try:
        store.stage("s", {"digest": "a"})["x"] = 2; store.save()
    except RuntimeError:
        pass
    assert json.load(open(p))["results"]["s"]["x"] == 1


def test_missing():
    assert records.missing({"a": {"b": 1}}, ["a.b"]) == []
    assert records.missing({"a": {}}, ["a.b", "c"]) == ["a.b", "c"]
    assert records.missing(None, ["a"]) == ["a"]


def test_superseded_record_is_archived_in_full(tmp_path):
    p = tmp_path / "r.json"; store = records.ResultStore(str(p), {"protocol": "t"})
    store.stage("s", {"digest": "a"})["layers"] = {"2": {"value": 1}}; store.save()
    store2 = records.ResultStore(str(p), {"protocol": "t"}); rec = store2.stage("s", {"digest": "b"}); store2.save()
    saved = json.load(open(p)); arch = saved["superseded"]
    assert rec.get("layers") is None and arch[0]["stage"] == "s" and arch[0]["record"]["layers"]["2"]["value"] == 1
    for d in "cdef":
        records.ResultStore(str(p), {}).stage("s", {"digest": d})
    store3 = records.ResultStore(str(p), {}); [store3.stage("s", {"digest": d}) for d in "cdef"]; store3.save()
    assert len(json.load(open(p))["superseded"]) == records.KEEP_SUPERSEDED


def test_artifacts_present(tmp_path):
    f = tmp_path / "a.npz"; f.write_bytes(b"x")
    assert records.artifacts_present([str(f)]) and not records.artifacts_present([str(f), str(tmp_path / "b")])
