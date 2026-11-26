"""Resume logic of the transition driver: a stale sole record for an epoch (missing curvature diagnostics) is dropped and
its epoch leaves the recorded set, so the stage is recomputed rather than aliased to itself."""
from flatocean.analyses import transitions as TR


def test_stale_sole_record_is_recomputed_not_aliased():
    stale = {"epoch": 500, "probes": {"1.0": [{"true": 1.0}], "0.1": [{"true": 0.5, "curvature_defined": True}]}}
    res = {"results": {"late_endpoint": stale, "epoch_250": {"epoch": 250, "probes": {"1.0": [{"curvature_defined": True}]}}}}
    assert TR.recorded_epochs(res) == {500, 250}
    assert TR.radii_without_curvature(stale) == ["1.0"] and TR.radii_without_curvature(res["results"]["epoch_250"]) == []
    done_epochs = TR.drop_stage(res, "late_endpoint")
    assert 500 not in done_epochs and done_epochs == {250} and "late_endpoint" not in res["results"]
    # the driver's alias rule would now not fire for epoch 500
    assert not (500 in done_epochs)


def test_alias_records_do_not_count_as_recorded():
    res = {"results": {"late_endpoint": {"same_as_epoch": 500}, "epoch_100": {"epoch": 100, "probes": {}}}}
    assert TR.recorded_epochs(res) == {100}


def test_stage_defects_require_exact_probes_seeds_and_the_full_schema():
    seeds = [100, 101]; radii = [0.1, 1.0]
    base = {k: 1 for k in TR.BASE_FIELDS + TR.V3_FIELDS}
    curv = {"curvature": [1.0], "curvature_defined": [True], "preactivation_displacement": [1.0]}
    good = {"probes": {str(r): [dict(base, direction_seed=s_, **curv) for s_ in seeds] for r in radii}}
    assert TR.stage_defects(good, radii, seeds, smooth=True) == []
    assert TR.stage_defects({}, radii, seeds, smooth=True) == ["no probe collection"]
    assert TR.stage_defects({"probes": {"0.1": [], "1.0": good["probes"]["1.0"]}}, radii, seeds, True)[0].startswith("radius 0.1: 0/2")
    one = {"probes": {str(r): [dict(base, direction_seed=100, **curv)] for r in radii}}
    d = TR.stage_defects(one, radii, seeds, True); assert len(d) == 2 and all("1/2 probes" in x for x in d)
    extra = {"probes": {str(r): good["probes"][str(r)] + [dict(base, direction_seed=100, **curv)] for r in radii}}      # a duplicate extra probe
    assert all("3/2 probes" in x for x in TR.stage_defects(extra, radii, seeds, True))
    wrong_seed = {"probes": {str(r): [dict(q, direction_seed=q["direction_seed"] + 5) for q in good["probes"][str(r)]] for r in radii}}
    assert all("direction seeds differ" in x for x in TR.stage_defects(wrong_seed, radii, seeds, True))
    no_curv = {"probes": {str(r): [dict(base, direction_seed=s_) for s_ in seeds] for r in radii}}
    assert TR.stage_defects(no_curv, radii, seeds, smooth=False) == [] and all("curvature" in x for x in TR.stage_defects(no_curv, radii, seeds, smooth=True))
    no_decomp = {"probes": {str(r): [{k: v for k, v in q.items() if k != "cancellation_ratio"} for q in good["probes"][str(r)]] for r in radii}}
    assert all("cancellation_ratio" in x for x in TR.stage_defects(no_decomp, radii, seeds, True))
    stray = dict(good, probes=dict(good["probes"], **{"7.0": []}))
    assert any("not in the configuration" in x for x in TR.stage_defects(stray, radii, seeds, True))
