"""Every registered command answers --help with status 0 without doing any work (no figure, table or result written), and
rejects an unknown argument with status 2."""
import os, subprocess, sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import flatocean.__main__ as M                                                        # noqa: E402


def run(args, env):
    return subprocess.run([sys.executable, "-m", "flatocean", *args], cwd=HERE, capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.parametrize("cmd", sorted(M.COMMANDS))
def test_help_is_harmless_and_unknown_arguments_are_rejected(cmd, tmp_path):
    fig, tab, res = tmp_path / "fig", tmp_path / "tab", tmp_path / "res"
    env = dict(os.environ, FLATOCEAN_FIGURES_DIR=str(fig), FLATOCEAN_TABLES_DIR=str(tab), FLATOCEAN_RESULTS_ROOT=str(res), FLATOCEAN_INPUT_ROOT=os.path.join(HERE, "results"))
    r = run([cmd, "--help"], env)
    assert r.returncode == 0, f"{cmd} --help: rc {r.returncode}\n{r.stderr[-800:]}"
    assert "usage:" in r.stdout
    for d in (fig, tab):
        assert not d.exists() or os.listdir(d) == [], f"{cmd} --help wrote into {d}"
    written = [os.path.join(dp, f) for dp, _, fs in os.walk(res) for f in fs] if res.exists() else []
    assert written == [], f"{cmd} --help wrote results {written[:3]}"
    r = run([cmd, "--no-such-argument"], env)
    assert r.returncode == 2, f"{cmd} accepted an unknown argument (rc {r.returncode})"
