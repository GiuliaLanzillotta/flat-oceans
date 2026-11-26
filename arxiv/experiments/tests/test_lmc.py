"""LMC of spawned pairs: the segment decomposition on a tiny MLP (linear part against polynomial interpolation of the
multi-affine frozen map, the transition identity, the deviation identity, the endpoints), and the spawn-state recovery
end to end on a tiny MNIST run (replay verified against the stored checkpoints and the recovery moments; a child with
the parent's own order reproduces the parent bitwise, a child with another order diverges; the child passes the
trainer's verification)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from flatocean import frozen_lib as FL # noqa: E402
from flatocean import tier1_train as T # noqa: E402
from flatocean.analyses import lmc as LM     # noqa: E402


def tiny():
    torch.manual_seed(0); net = T.MLP(6, [8, 8, 7], 3, init_seed=3, dtype=torch.float64)
    X = torch.randn(40, 6, dtype=torch.float64); y = torch.randint(0, 3, (40,)); Y = torch.nn.functional.one_hot(y, 3).double()
    thA = FL.flat_params(net).detach().clone(); g = torch.Generator().manual_seed(5); thB = thA + 0.3 * torch.randn(thA.numel(), generator=g, dtype=torch.float64)
    return net, X, Y, y, thA, thB


def test_segment_decomposition_identities():
    net, X, Y, y, thA, thB = tiny()
    rec = LM.segment_record(net, thA, thB, {"analysis": (X, Y, y)}, alphas=[0.0, 0.25, 0.5, 1.0])["sets"]["analysis"]
    pts = rec["points"]; scale = rec["endpoint_difference"]
    assert pts[0]["change_from_A"] <= 1e-14 * max(1, scale) and pts[-1]["change_from_B"] <= 1e-14 * max(1, scale)
    assert rec["max_identity_residual"] <= 1e-11 * max(1, scale), rec["max_identity_residual"]
    assert rec["max_deviation_identity_residual"] <= 1e-11 * max(1, scale), rec["max_deviation_identity_residual"]
    p = pts[2]; assert abs(p["alpha"] - 0.5) < 1e-12 and p["interpolation_deviation"] > 0
    assert rec["barriers"]["at_half"]["mse_excess"] == pytest.approx(pts[2]["loss"]["mse"] - 0.5 * (pts[0]["loss"]["mse"] + pts[-1]["loss"]["mse"]))
    assert all(g["agree_A"] == 1.0 for g in pts[0]["gates"]) and all(g["agree_B"] == 1.0 for g in pts[-1]["gates"])


def test_linear_part_is_the_degree_one_coefficient_of_the_frozen_map():
    """F_frz(theta_A + t w) is a polynomial of degree at most 4 in t (four multi-affine blocks): its degree-one coefficient,
    recovered by interpolation on five points, equals the sum of the single-block frozen changes."""
    net, X, Y, y, thA, thB = tiny(); w = thB - thA; slices = FL.block_slices(net)
    FL.set_flat(net, thA); F_A = LM._forward(net, X); masks = FL.base_masks(net, X)
    ts = np.array([-2.0, -1.0, 1.0, 2.0, 3.0]); vals = []
    for t in ts:
        FL.set_flat(net, thA + t * w); vals.append((LM._frozen(net, masks, X) - F_A).reshape(-1).numpy())
    FL.set_flat(net, thA)
    V = np.vander(ts, 5, increasing=True); coef = np.linalg.solve(V, np.stack(vals))           # rows: t^0 .. t^4
    lin_interp = torch.from_numpy(coef[1]).reshape(F_A.shape)
    d = LM.decompose(net, masks, X, thA, w, F_A, slices)
    assert float((d["linear"] - lin_interp).norm() / lin_interp.norm()) < 1e-9
    assert float(np.abs(coef[0]).max()) < 1e-10 * max(1.0, float(np.abs(coef).max()))     # no constant term: F_frz(theta_A) = F_A
    assert torch.allclose(d["true"] - F_A, d["linear"] + d["cross"] + d["transition"], atol=1e-13)


def test_spawn_state_recovery_and_children(tmp_path):
    """Tiny MNIST parent (width 8, PCA-16, four epochs, CPU); replay to epoch 2 verified against the stored checkpoint and
    continued to epoch 4 against the stored weights and recovery moments; a child with the parent's order reproduces the
    parent's epoch-4 weights bitwise, a child with another order does not; both pass the trainer's verification."""
    env = dict(os.environ, OMP_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    base = [sys.executable, "-m", "flatocean.tier1_train", "--dataset", "mnist", "--model", "mlp", "--seed", "0", "--width", "8", "--depth", "3", "--q", "16",
            "--epochs", "4", "--device", "cpu", "--out", str(tmp_path), "--log-every", "1"]
    r = subprocess.run(base + ["--tag", "parent"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900); assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    parent = os.path.join(str(tmp_path), "mnist_mlp_parent", "seed_0")
    assert os.path.exists(os.path.join(parent, "checkpoints", "recovery_epoch_0004.pt")) and not os.path.exists(os.path.join(parent, "checkpoints", "recovery_epoch_0002.pt"))
    torch.set_num_threads(2)                                                  # the replay must use the trainer's thread count: MKL's reduction order depends on it (and MKL_DYNAMIC adapts to load)
    state = os.path.join(str(tmp_path), "spawn_e2.pt"); ver = LM.recover_spawn_state(parent, 2, state, device="cpu", log=lambda *a: None)
    assert ver["passed"] and ver["weights_vs_checkpoint"]["max_abs"] == 0.0 and ver["moments_vs_recovery_checkpoint"] is None
    cont = ver["same_order_continuation"]; assert cont["epoch"] == 4 and cont["weights_vs_checkpoint"]["max_abs"] == 0.0 and cont["moments_vs_recovery_checkpoint"]["steps_equal"]
    assert cont["moments_vs_recovery_checkpoint"]["max_abs_exp_avg"] == 0.0 and cont["moments_vs_recovery_checkpoint"]["max_abs_exp_avg_sq"] == 0.0
    for tag, order in (("same_order", "2000"), ("other_order", "9")):
        r = subprocess.run(base + ["--tag", tag, "--spawn-state", state, "--order-seed", order, "--high-accuracy-milestone"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        child = os.path.join(str(tmp_path), f"mnist_mlp_{tag}", "seed_0"); cfg = json.load(open(os.path.join(child, "config.json")))
        assert cfg["spawn"]["epoch"] == 2 and cfg["order_seed"] == int(order) and cfg["init_seed"] == 1000 and cfg["spawn"]["adam_policy"].startswith("inherited")
        hist = json.load(open(os.path.join(child, "history.json")))["epochs"]; assert [e["epoch"] for e in hist] == [0, 1, 2, 3, 4] and hist[2]["inherited_from_parent"] and "inherited_from_parent" not in hist[3]
        assert sorted(f for f in os.listdir(os.path.join(child, "checkpoints")) if f.startswith("epoch_")) == ["epoch_0002.pt", "epoch_0004.pt"]
        r = subprocess.run(base + ["--tag", tag, "--verify"], capture_output=True, text=True, env=env, cwd=HERE, timeout=600); assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        p4 = torch.load(os.path.join(parent, "checkpoints", "epoch_0004.pt"), map_location="cpu", weights_only=False)["model"]
        c4 = torch.load(os.path.join(child, "checkpoints", "epoch_0004.pt"), map_location="cpu", weights_only=False)["model"]
        same = all(torch.equal(p4[k], c4[k]) for k in p4)
        assert same == (tag == "same_order"), f"{tag}: bitwise equal to the parent = {same}"


def test_reset_policy_state_and_child(tmp_path):
    """The uniform reset policy: the state holds the stored weights and a fresh Adam state; a child trains from it and its
    configuration carries the reset label; a same-order child no longer reproduces the parent (moments differ)."""
    env = dict(os.environ, OMP_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    base = [sys.executable, "-m", "flatocean.tier1_train", "--dataset", "mnist", "--model", "mlp", "--seed", "0", "--width", "8", "--depth", "3", "--q", "16",
            "--epochs", "4", "--device", "cpu", "--out", str(tmp_path), "--log-every", "1"]
    r = subprocess.run(base + ["--tag", "parent"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900); assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    parent = os.path.join(str(tmp_path), "mnist_mlp_parent", "seed_0"); state = os.path.join(str(tmp_path), "reset_e2.pt")
    ver = LM.reset_spawn_state(parent, 2, state, replay_evidence={"note": "test"}, log=lambda *a: None); assert ver["usable"] and ver["policy"] == "reset"
    obj = torch.load(state, map_location="cpu", weights_only=False); assert obj["optimizer"]["state"] == {} and obj["adam_policy"].startswith("reset")
    r = subprocess.run(base + ["--tag", "reset_same_order", "--spawn-state", state, "--order-seed", "2000"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    child = os.path.join(str(tmp_path), "mnist_mlp_reset_same_order", "seed_0"); cfg = json.load(open(os.path.join(child, "config.json")))
    assert cfg["spawn"]["adam_policy"].startswith("reset") and cfg["spawn"]["verification"]["replay_attempt"] == {"note": "test"}
    p4 = torch.load(os.path.join(parent, "checkpoints", "epoch_0004.pt"), map_location="cpu", weights_only=False)["model"]
    c4 = torch.load(os.path.join(child, "checkpoints", "epoch_0004.pt"), map_location="cpu", weights_only=False)["model"]
    assert not all(torch.equal(p4[k], c4[k]) for k in p4)
    r = subprocess.run(base + ["--tag", "reset_same_order", "--verify"], capture_output=True, text=True, env=env, cwd=HERE, timeout=600); assert r.returncode == 0, r.stdout[-1500:]


def test_policy_kind_is_fixed_and_failed_states_are_refused(tmp_path):
    """Two verified replays with different residuals share the policy kind 'inherited'; a failed verification gives 'failed'
    and the spawn-state phase exits nonzero; the trainer refuses a failed state and a state whose parent checkpoint changed."""
    ok1 = {"passed": True, "weights_vs_checkpoint": {"max_abs": 9e-15, "n_bitwise_equal": 0, "n_tensors": 8}, "same_order_continuation": None, "moments_vs_recovery_checkpoint": None}
    ok2 = dict(ok1, weights_vs_checkpoint={"max_abs": 3e-14, "n_bitwise_equal": 0, "n_tensors": 8})
    bad = dict(ok1, passed=False)
    assert LM.policy_kind(ok1) == LM.policy_kind(ok2) == "inherited" and LM.policy_label(ok1) != LM.policy_label(ok2)
    assert LM.policy_kind(bad) == "failed" and LM.policy_kind({"policy": "reset", "passed": True}) == "reset"
    env = dict(os.environ, OMP_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    base = [sys.executable, "-m", "flatocean.tier1_train", "--dataset", "mnist", "--model", "mlp", "--seed", "0", "--width", "8", "--depth", "3", "--q", "16",
            "--epochs", "4", "--device", "cpu", "--out", str(tmp_path), "--log-every", "1"]
    r = subprocess.run(base + ["--tag", "parent"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900); assert r.returncode == 0, r.stderr[-1500:]
    parent = os.path.join(str(tmp_path), "mnist_mlp_parent", "seed_0"); state = os.path.join(str(tmp_path), "reset_e2.pt")
    LM.reset_spawn_state(parent, 2, state, log=lambda *a: None)
    obj = torch.load(state, map_location="cpu", weights_only=False); obj["verification"]["passed"] = False; obj["verification"]["usable"] = False; obj["adam_policy_kind"] = "failed"
    failed = os.path.join(str(tmp_path), "failed.pt"); torch.save(obj, failed)
    r = subprocess.run(base + ["--tag", "child_failed", "--spawn-state", failed, "--order-seed", "9"], capture_output=True, text=True, env=env, cwd=HERE, timeout=300)
    assert r.returncode != 0 and "verification failed" in r.stdout + r.stderr
    obj = torch.load(state, map_location="cpu", weights_only=False); obj["parent_checkpoint_sha256"] = "0" * 64; wrong = os.path.join(str(tmp_path), "wrong.pt"); torch.save(obj, wrong)
    r = subprocess.run(base + ["--tag", "child_wrong", "--spawn-state", wrong, "--order-seed", "9"], capture_output=True, text=True, env=env, cwd=HERE, timeout=300)
    assert r.returncode != 0 and "differs from the one the state was built from" in r.stdout + r.stderr


def test_segment_refuses_incompatible_endpoints():
    net, X, Y, y, thA, thB = tiny()
    other = T.MLP(6, [8, 9, 7], 3, init_seed=3, dtype=torch.float64)
    rA = {"config": {"model": "mlp", "widths": [8, 8, 7], "activation": "relu", "q": 6, "d_out": 3, "dataset": "x"}, "net": net}
    rB = {"config": dict(rA["config"], widths=[8, 9, 7]), "net": other}
    with pytest.raises(RuntimeError):
        LM.check_compatible(rA, rB, "a", "b")


def test_endpoint_compatibility_covers_preprocessing_and_sample_order(tmp_path):
    """Compatible endpoints pass and return B's preprocessing hash; a different stored preprocessing, a different analysis
    set and a different layout are refused."""
    net, X, Y, y, thA, thB = tiny(); cfg = {"model": "mlp", "widths": [8, 8, 7], "activation": "relu", "q": 6, "d_out": 3, "dataset": "x"}
    rA = {"config": cfg, "net": net}; rB = {"config": dict(cfg), "net": T.MLP(6, [8, 8, 7], 3, init_seed=4, dtype=torch.float64)}
    def run_dir(name, pre, idx):
        d = tmp_path / name; d.mkdir(); (d / "preprocessing.pt").write_bytes(pre); json.dump({"analysis_indices": idx, "raw_file_sha256": "h"}, open(d / "data_manifest.json", "w")); return str(d)
    a = run_dir("A", b"pre-1", [1, 2, 3]); b_same = run_dir("B", b"pre-1", [1, 2, 3]); b_pre = run_dir("Bp", b"pre-2", [1, 2, 3]); b_idx = run_dir("Bi", b"pre-1", [3, 2, 1])
    from flatocean import provenance as P
    assert LM.check_compatible(rA, rB, a, b_same) == P.sha256_file(os.path.join(b_same, "preprocessing.pt"))
    with pytest.raises(RuntimeError, match="preprocessing"):
        LM.check_compatible(rA, rB, a, b_pre)
    with pytest.raises(RuntimeError, match="analysis set"):
        LM.check_compatible(rA, rB, a, b_idx)
    with pytest.raises(RuntimeError, match="layout"):
        LM.check_compatible(rA, {"config": dict(cfg), "net": T.MLP(6, [8, 9, 7], 3, init_seed=4, dtype=torch.float64)}, a, b_same)      # same declared configuration, different tensor shapes


def test_spawn_state_command_exits_nonzero_on_a_failed_replay(tmp_path, monkeypatch):
    """The spawn-state command ends with status 1 when a replay fails, so a queue waiting on it stops."""
    monkeypatch.setattr(LM, "OUT", lambda: str(tmp_path))
    monkeypatch.setattr(LM, "stage_epochs", lambda rd: {"high_accuracy": {"checkpoint_epoch": 40}})
    monkeypatch.setattr(LM.P, "sha256_file", lambda p: "x")
    bad = {"passed": False, "weights_vs_checkpoint": {"max_abs": 0.1, "n_bitwise_equal": 0, "n_tensors": 8}, "same_order_continuation": None, "moments_vs_recovery_checkpoint": None}
    def fake_recover(parent, ep, path, device, log=print):
        os.makedirs(os.path.dirname(path), exist_ok=True); torch.save({"adam_policy_kind": "failed", "verification": bad, "parent_checkpoint_sha256": "x"}, path); return dict(bad)
    monkeypatch.setattr(LM, "recover_spawn_state", fake_recover)
    with pytest.raises(SystemExit) as e:
        LM.main(["spawn-state", "--seeds", "0", "--stages", "high_accuracy", "--parent-dir", str(tmp_path), "--device", "cpu"])
    assert e.value.code == 1


def test_resume_accepts_a_state_file_rewritten_after_the_launch():
    """The trainer's resume rule: the child's launch-time state hash is accepted when the current file lists it in its chain."""
    saved = {"parent_run": "p", "epoch": 2, "parent_order_seed": 2000, "adam_policy_kind": "reset", "state_sha256": "old"}
    new = dict(saved, state_sha256="new")
    assert T._spawn_resume_compatible(saved, new, {"previous_sha256": ["old", "new"]})
    assert not T._spawn_resume_compatible(saved, new, {"previous_sha256": ["other"]})
    assert not T._spawn_resume_compatible(saved, dict(new, epoch=3), {"previous_sha256": ["old"]})
    assert T._spawn_resume_compatible(saved, dict(saved), {})


def test_resume_derives_the_policy_of_a_legacy_configuration():
    """A child configuration written before adam_policy_kind existed carries its policy in the verification; a resume against a
    state of another policy is refused even when the old hash is in the chain."""
    legacy_reset = {"parent_run": "p", "epoch": 2, "parent_order_seed": 2000, "state_sha256": "old", "adam_policy": "reset (uniform for every spawn; ...)",
                    "verification": {"policy": "reset", "passed": True, "usable": True}}
    new_inherited = {"parent_run": "p", "epoch": 2, "parent_order_seed": 2000, "state_sha256": "new", "adam_policy_kind": "inherited", "verification": {"passed": True}}
    new_reset = dict(new_inherited, adam_policy_kind="reset", verification={"policy": "reset", "passed": True})
    chain = {"previous_sha256": ["old", "new"]}
    assert T._policy_kind_of(legacy_reset) == "reset" and T._policy_kind_of({"verification": {"passed": True}}) == "inherited" and T._policy_kind_of({"verification": {"passed": False}}) == "failed"
    assert not T._spawn_resume_compatible(legacy_reset, new_inherited, chain)
    assert T._spawn_resume_compatible(legacy_reset, new_reset, chain)
    import json, glob, os
    for d in glob.glob(os.path.join(HERE, "results", "tier1_models", "mnist_mlp_lmc_s*_c*", "seed_*"))[:2]:       # the actual children, if present
        c = json.load(open(os.path.join(d, "config.json")))["spawn"]; assert T._policy_kind_of(c) == "reset"


def test_recovery_policy_inherits_the_moments_exactly(tmp_path):
    """The inherited policy from a recovery file: the state carries the parent's moments and step counters; a same-order
    child reproduces the parent's later checkpoint bitwise; a spawn epoch without a recovery file is refused; the
    verification summary keeps the states of earlier invocations."""
    env = dict(os.environ, OMP_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    base = [sys.executable, "-m", "flatocean.tier1_train", "--dataset", "mnist", "--model", "mlp", "--seed", "0", "--width", "8", "--depth", "3", "--q", "16",
            "--epochs", "10", "--recovery-every", "5", "--device", "cpu", "--out", str(tmp_path), "--log-every", "1"]
    r = subprocess.run(base + ["--tag", "parent"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900); assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    parent = os.path.join(str(tmp_path), "mnist_mlp_parent", "seed_0")
    recs = sorted(f for f in os.listdir(os.path.join(parent, "checkpoints")) if f.startswith("recovery_")); e = int(recs[0][15:19]); assert 0 < e < 10, recs
    state = os.path.join(str(tmp_path), "rec.pt"); ver = LM.recovery_spawn_state(parent, e, state, log=lambda *a: None)
    assert ver["passed"] and ver["policy"] == "recovery" and ver["weights_vs_checkpoint"]["max_abs"] == 0.0 and ver["steps_per_epoch"] and LM.policy_kind(ver) == "inherited"
    obj = torch.load(state, map_location="cpu", weights_only=False); assert obj["adam_policy_kind"] == "inherited" and obj["optimizer"]["state"] and obj["adam_policy"].startswith("inherited (Adam moments")
    with pytest.raises(FileNotFoundError):
        LM.recovery_spawn_state(parent, e + 1, os.path.join(str(tmp_path), "none.pt"), log=lambda *a: None)
    r = subprocess.run(base + ["--tag", "rec_same_order", "--spawn-state", state, "--order-seed", "2000"], capture_output=True, text=True, env=env, cwd=HERE, timeout=900)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    child = os.path.join(str(tmp_path), "mnist_mlp_rec_same_order", "seed_0"); cfg = json.load(open(os.path.join(child, "config.json")))
    assert cfg["spawn"]["adam_policy_kind"] == "inherited" and cfg["spawn"]["epoch"] == e
    p8 = torch.load(os.path.join(parent, "checkpoints", "epoch_0010.pt"), map_location="cpu", weights_only=False)["model"]
    c8 = torch.load(os.path.join(child, "checkpoints", "epoch_0010.pt"), map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(p8[k], c8[k]) for k in p8)
    # the phase merges its summary into an existing verification file
    import argparse
    out = os.path.join(str(tmp_path), "lmc"); os.makedirs(os.path.join(out, "spawn_states"))
    json.dump({"states": {"old_state": {"passed": True, "policy": "reset"}}, "invocations": [{"policy": "reset"}]}, open(os.path.join(out, "spawn_states", "verification.json"), "w"))
    a = argparse.Namespace(threads=2, seeds=[0], parent_dir=parent, dataset="mnist", model="mlp", tag="parent", spawn_epochs=[e], stages=None, policy="recovery", overwrite=False, device="cpu")
    orig = LM.OUT; LM.OUT = lambda: out
    try:
        LM.spawn_state_phase(a)
    finally:
        LM.OUT = orig
    v = json.load(open(os.path.join(out, "spawn_states", "verification.json")))
    assert "old_state" in v["states"] and len(v["invocations"]) == 2 and v["invocations"][-1]["policy"] == "recovery" and v["all_usable_and_uniform"]


def test_displacement_shares_connected_rule_and_orbit_projection():
    """Block displacement shares sum to one; the loss-connected flag follows the stated rule; the orbit projection share is
    one for a displacement along a rescaling generator and below one for a random one."""
    from flatocean.analyses import symmetries as SY
    net, X, Y, y, thA, thB = tiny()
    rec = LM.segment_record(net, thA, thB, {"analysis": (X, Y, y)}, alphas=[0.0, 0.5, 1.0])
    assert abs(sum(rec["block_displacement_shares"].values()) - 1) < 1e-12 and rec["displacement_norm"] == pytest.approx(float((thB - thA).norm()))
    b = rec["sets"]["analysis"]["barriers"]; assert rec["loss_connected"] == (b["entezari_mse"]["value"] < LM.CONNECTED_FACTOR * b["loss_connected"]["mean_endpoint_mse"])
    pr = rec["projections"]["rescaling_orbit_at_A"]; assert 0 <= pr["norm_share"] <= 1 + 1e-12 and pr["dim"] == pr["n_generators"] == 8 + 8 + 7
    FL.set_flat(net, thA); G, _, _ = SY.unit_generators(net); thG = thA + 0.1 * G[:, 3]
    rec2 = LM.segment_record(net, thA, thG, {"analysis": (X, Y, y)}, alphas=[0.0, 0.5, 1.0])
    assert rec2["projections"]["rescaling_orbit_at_A"]["norm_share"] == pytest.approx(1.0, abs=1e-10)


def test_find_pairs_parses_mlp_and_cnn_child_tags(tmp_path, monkeypatch):
    import argparse
    base = tmp_path / "models"; base.mkdir()
    for d in ("mnist_mlp_lmc_s0_e100_c0", "mnist_mlp_lmc_s0_e100_c1", "mnist_mlp_lmc_s2_e1_c0", "mnist_cnn_lmc_cnn_s1_e250_c0", "mnist_cnn_lmc_cnn_s1_e250_c1", "mnist_cnn_lmc_cnn_s2_e100_c0"):
        (base / d).mkdir()
    monkeypatch.setattr(T, "RESULTS", str(base))
    mlp = LM.find_pairs(argparse.Namespace(pairs=["siblings"], dataset="mnist", model="mlp", tag=""))
    cnn = LM.find_pairs(argparse.Namespace(pairs=["siblings"], dataset="mnist", model="cnn", tag=""))
    assert [(n, a, b) for n, a, b, _ in mlp] == [("siblings_lmc_s0_e100", (0, "lmc_s0_e100_c0"), (0, "lmc_s0_e100_c1"))]          # the s2_e1 pair lacks its c1 sibling
    assert [(n, a, b) for n, a, b, _ in cnn] == [("siblings_lmc_cnn_s1_e250", (1, "lmc_cnn_s1_e250_c0"), (1, "lmc_cnn_s1_e250_c1"))]


def test_control_pair_names_are_model_specific_and_do_not_collide():
    """The parent-pair (independent-run) controls of the MLP and the CNN must have distinct names so they do not overwrite
    each other in the shared segments directory (regression for the 2026-09-28 collision)."""
    import argparse
    def names(model, tag=""):
        return [n for n, *_ in LM.find_pairs(argparse.Namespace(pairs=["controls"], dataset="mnist", model=model, tag=tag))]
    mlp, cnn = names("mlp"), names("cnn")
    assert mlp == ["control_parents_mnist_mlp_s0_s1", "control_parents_mnist_mlp_s0_s2", "control_parents_mnist_mlp_s1_s2"]
    assert cnn == ["control_parents_mnist_cnn_s0_s1", "control_parents_mnist_cnn_s0_s2", "control_parents_mnist_cnn_s1_s2"]
    assert set(mlp).isdisjoint(cnn)                                              # no shared filename between the two models
    assert names("mlp", "w64")[0] == "control_parents_mnist_mlp_w64_s0_s1"       # a tag is carried into the name too
