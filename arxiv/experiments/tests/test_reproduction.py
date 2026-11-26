"""Reproduction plans must be inspectable and stop on a failed prerequisite."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location('reproduction_runner', ROOT/'reproduce.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_plan_is_read_only_and_failure_blocks_following_jobs(tmp_path, monkeypatch):
    mod = runner(); monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path/'configs').mkdir()
    first = tmp_path/'first.py';first.write_text('raise SystemExit(7)\n')
    second = tmp_path/'second.py';second.write_text("from pathlib import Path\nPath('ran').write_text('bad')\n")
    jobs = [{'id': p.stem, 'group': 'test', 'command': [str(p)], 'outputs': []} for p in [first, second]]
    (tmp_path/'configs/experiments.json').write_text(json.dumps({'group_order': ['test'], 'jobs': jobs}))
    assert mod.main(['all']) == 0
    assert not (tmp_path/'build').exists()
    assert mod.main(['all', '--execute']) == 7
    assert not (tmp_path/'ran').exists()
    assert not (tmp_path/'build/reproduction/completed.json').exists()


def test_resume_checks_the_command_not_just_the_job_name(tmp_path, monkeypatch):
    mod = runner(); monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path/'configs').mkdir()
    path = tmp_path/'configs/experiments.json'
    def catalog(value):
        return {'group_order': ['test'], 'jobs': [{'id':'one', 'group':'test', 'command': ['-c',
            f"from pathlib import Path; Path('ran').write_text('{value}')"], 'outputs': []}]}
    path.write_text(json.dumps(catalog('a')))
    assert mod.main(['all','--execute']) == 0
    path.write_text(json.dumps(catalog('b')))
    assert mod.main(['all','--execute','--resume']) == 0
    assert (tmp_path/'ran').read_text() == 'b'


def test_catalog_has_unique_jobs_and_existing_entry_points():
    d = json.loads((ROOT/'configs/experiments.json').read_text())
    assert len(d['jobs']) == len({j['id'] for j in d['jobs']})
    for job in d['jobs']:
        assert job['group'] in d['group_order']
        command=job['command']
        assert command
        if command[0].endswith('.py'):
            assert (ROOT/command[0]).is_file(), job['id']
    figs=json.loads((ROOT/'configs/figures.json').read_text())
    assert len(figs)==71
    assert all((ROOT/v['paper_path']).is_file() for v in figs.values())


def test_saved_result_paths_survive_repository_relocation(tmp_path, monkeypatch):
    from flatocean import paths
    results = tmp_path / "results"
    parent = results / "tier1_models/mnist_mlp/seed_0"
    parent.mkdir(parents=True)
    monkeypatch.setattr(paths, "RESULTS", results)
    assert paths.recorded_result_path("/old/checkout/experiments/results/tier1_models/mnist_mlp/seed_0") == parent
    assert paths.recorded_result_path(parent) == parent
    missing = "/old/results/does-not-exist"
    assert str(paths.recorded_result_path(missing)) == missing
