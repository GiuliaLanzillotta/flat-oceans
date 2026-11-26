"""The identity scheme of the cluster analyses on a tiny frozen structure: partitions -> greedy at knn 10/25/50 -> shuffles,
single-layer runs consuming what a two-layer run produced, every result preserved, reuse invalidated only for the results
that consumed an assignment array that changed, and the migration of a record of the earlier per-stage scheme."""
import argparse
import json
import os

import numpy as np
import pytest
import torch

from flatocean.analyses import clusters as CL
from flatocean import provenance as P
def tiny_structure(seed=0, s=8, m=5, n=4, d=2, L=3):
    """H[l], D[l], S[l], Z[l] for l = 0..L-1 with a hard 0/1 gate on every layer and generic downstream maps; s d = 16 rows against
    r_h r_d = 20, so every layer has a co-occurrence deficit of at least 4."""
    g = torch.Generator().manual_seed(seed); H = []; D = []; S = []; Z = []
    for l in range(L):
        Hl = torch.randn(s, m, generator=g, dtype=torch.float64); Hl[:, -1] = 1.0
        Zl = torch.randn(s, n, generator=g, dtype=torch.float64); Sl = (Zl > 0).double()
        Dl = torch.randn(s, d, n, generator=g, dtype=torch.float64) * Sl[:, None, :]
        H.append(Hl); D.append(Dl); S.append(Sl); Z.append(Zl)
    return dict(H=H, D=D, S=S, Z=Z, L=L, d=d, activation="relu")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A results root in tmp_path, a stage loader that returns the tiny structure, no stored factors, and a file-free identity."""
    out = tmp_path / "clusters"; out.mkdir(); monkeypatch.setattr(CL, "OUT", str(out))
    st = tiny_structure(); s = st["H"][0].shape[0]; y = np.arange(s) % 3
    calls = {"dim_P": 0, "spectrum": 0}
    real_dim_P = CL.dim_P; real_spec = CL.jacobian_spectrum
    monkeypatch.setattr(CL, "dim_P", lambda *a_, **k: (calls.__setitem__("dim_P", calls["dim_P"] + 1), real_dim_P(*a_, **k))[1])
    monkeypatch.setattr(CL, "jacobian_spectrum", lambda *a_, **k: (calls.__setitem__("spectrum", calls["spectrum"] + 1), real_spec(*a_, **k))[1])
    monkeypatch.setattr(CL, "_load_stage", lambda a, seed, stage: dict(rd="rd", name=f"mnist_mlp_seed{seed}", ep=7, run={"dir": "rd"}, X=torch.zeros(s, 1), y=y, st=st) if stage == "high_accuracy" else None)
    monkeypatch.setattr(CL, "_factor_sha", lambda name, ep, l, run, rd: None)
    def identity(run, rd, ep, config):
        ident = {"identity_version": 1, "checkpoint": f"epoch_{ep}", "checkpoint_sha256": "ck", "preprocessing_sha256": "pre", "analysis_index_sha256": "idx",
                 "analysis_size": s, "analysis_dtype": "float64", "config": config}; ident["digest"] = P.identity_digest(ident); return ident
    monkeypatch.setattr(CL, "_identity", identity)
    return dict(out=str(out), st=st, y=y, calls=calls, s=s)


def args(phase, **kw):
    base = dict(tag="", seeds=[0], stages=["high_accuracy"], layers=[2, 3], G=[3], kinds=["labels", "kmeans_D", "random_labels", "random_kmeans_D", "patterns_suffix"],
                knn=10, greedy_knn=[10, 25, 50], max_group=200, device="cpu", threads=1, overwrite=False)
    base.update(kw); return argparse.Namespace(phase=phase, **base)


def rec(path):
    return json.load(open(path))


def test_sequence_preserves_results_and_binds_consumed_arrays(world):
    out = world["out"]; calls = world["calls"]; pj = os.path.join(out, "mnist_mlp_seed0_partitions.json")
    # partitions for two layers, then the same request again: nothing recomputed
    CL.partitions_phase(args("partitions")); r = rec(pj); L = r["results"]["high_accuracy"]["layers"]
    assert set(L) == {"2", "3"} and all("identity" in L[k] and L[k]["identity"]["config"]["layer"] == int(k) for k in L)
    assert "layers" not in L["2"]["identity"]["config"] and "layers" not in r["results"]["high_accuracy"]["stage_identity"]["config"]
    for k in ("labels", "kmeans_D", "random_labels", "random_kmeans_D", "patterns_suffix"):
        assert "dim_P" in L["2"]["3"][k] and "assignment_sha256" in L["2"]["3"][k]
    assert L["2"]["3"]["random_labels"]["n_groups"] == L["2"]["3"]["labels"]["n_groups"]              # matched sizes
    assert L["3"]["3"]["labels"]["ARI_with_previous_layer"]["layer"] == 2
    n = calls["dim_P"]; CL.partitions_phase(args("partitions")); assert calls["dim_P"] == n
    assert rec(pj)["results"]["high_accuracy"]["layers"] == L
    # a single-layer run consumes the two-layer assignments and adds one kind only (its base kind already on file)
    CL.partitions_phase(args("partitions", layers=[2], kinds=["labels", "kmeans_D", "random_labels", "random_kmeans_D", "patterns_suffix", "joint", "random_joint"]))
    assert calls["dim_P"] == n + 2; L2 = rec(pj)["results"]["high_accuracy"]["layers"]
    assert L2["3"] == L["3"] and all(L2["2"]["3"][k] == L["2"]["3"][k] for k in ("labels", "kmeans_D", "patterns_suffix")) and "dim_P" in L2["2"]["3"]["joint"]
    z = np.load(CL.assignments_path("mnist_mlp_seed0", 7, 2, 3)); assert {"labels", "kmeans_D", "joint", "random_joint"} <= set(z.files)
    # greedy at three neighbourhoods on one layer: three records, each under its own identity, none overwriting another
    for k in (10, 25, 50):
        CL.greedy_phase(args("greedy", layers=[2], knn=k))
    gj = rec(os.path.join(out, "mnist_mlp_seed0_greedy.json"))["results"]["high_accuracy"]["layers"]
    assert set(gj) == {"2:knn10", "2:knn25", "2:knn50"} and all(gj[k]["identity"]["config"]["knn"] == int(k[5:]) and "orders" in gj[k] for k in gj)
    for k in (10, 25, 50):
        z = np.load(CL.greedy_path("mnist_mlp_seed0", 7, 2, k)); assert int(z["knn"]) == k and "order_distance" in z.files
    n_g = calls["dim_P"]; CL.greedy_phase(args("greedy", layers=[2], knn=25)); assert calls["dim_P"] == n_g          # complete: skipped
    # shuffles on the single layer consume the partition and greedy arrays; a repeat recomputes nothing
    CL.shuffle_phase(args("shuffle", layers=[2], kinds=["labels", "kmeans_D"])); sj = os.path.join(out, "mnist_mlp_seed0_shuffle.json")
    S2 = rec(sj)["results"]["high_accuracy"]["layers"]["2"]
    for k in ("global", "labels", "kmeans_D", "greedy_knn10", "greedy_knn25", "greedy_knn50"):
        assert "delta_permuted" in S2[k]
    assert S2["greedy_knn25"]["greedy_identity_digest"] == gj["2:knn25"]["identity"]["digest"]
    n_s = calls["spectrum"]; CL.shuffle_phase(args("shuffle", layers=[2], kinds=["labels", "kmeans_D"])); assert calls["spectrum"] == n_s
    # change the kmeans_D array on file (same identity): only the kmeans_D shuffle is recomputed; the partition result of kmeans_D too
    ap = CL.assignments_path("mnist_mlp_seed0", 7, 2, 3); z = dict(np.load(ap)); dig = str(z.pop("identity_digest"))
    z["kmeans_D"] = z["kmeans_D"][:, ::-1].copy(); CL.save_assignments(ap, dig, z)
    CL.shuffle_phase(args("shuffle", layers=[2], kinds=["labels", "kmeans_D"])); assert calls["spectrum"] == n_s + 1
    S2b = rec(sj)["results"]["high_accuracy"]["layers"]["2"]; assert S2b["labels"] == S2["labels"] and S2b["kmeans_D"]["assignment_sha256"] != S2["kmeans_D"]["assignment_sha256"]
    n = calls["dim_P"]; CL.partitions_phase(args("partitions", layers=[2], kinds=["labels", "kmeans_D"])); assert calls["dim_P"] == n + 1
    # change the greedy knn10 file (same identity): only that shuffle entry is recomputed
    gp = CL.greedy_path("mnist_mlp_seed0", 7, 2, 10); z = dict(np.load(gp)); dig = str(z.pop("identity_digest")); sc = {k: z.pop(k) for k in ("knn", "max_group", "merge_threshold", "partitions_identity_digest")}
    z["order_distance"] = z["order_distance"][::-1].copy(); CL.save_assignments(gp, dig, z, **{k: v.item() if hasattr(v, "item") else v for k, v in sc.items()})
    CL.shuffle_phase(args("shuffle", layers=[2], kinds=["labels", "kmeans_D"])); assert calls["spectrum"] == n_s + 2
    # a greedy file whose settings do not match its digest is refused
    CL.save_assignments(gp, dig, z, knn=11, max_group=200, merge_threshold=CL.ATOL_MERGE, partitions_identity_digest="x")
    with pytest.raises(P.ProvenanceError):
        CL.shuffle_phase(args("shuffle", layers=[2], kinds=["labels"]))


def test_random_kind_requires_its_base_and_is_a_permutation(world):
    st = world["st"]
    with pytest.raises(ValueError):
        CL.applicable_kinds(st, 1, ["random_kmeans_D"])
    with pytest.raises(ValueError):
        CL.applicable_kinds(st, 1, ["gates_L3", "random_gates_L3", "labels"])
    base = np.arange(12) % 4; r = CL.matched_random(base, 2, 3, "random_labels")
    assert sorted(r) == sorted(base) and (r == CL.matched_random(base, 2, 3, "random_labels")).all() and not (r == CL.matched_random(base, 3, 3, "random_labels")).all()


def test_migration_of_a_legacy_record(world):
    """A record of the earlier scheme (stage identity with the layer list, results bound to the file hash) is converted after
    its identity has been re-derived and matched; the arrays and every result value are unchanged."""
    out = world["out"]; st = world["st"]; y = world["y"]; s = world["s"]; a = args("partitions"); name = "mnist_mlp_seed0"; ep = 7
    S = CL.stage_setup(a, 0, "high_accuracy"); old = CL._identity(S["run"], S["rd"], ep, dict(CL.base_config(), layers=[2, 3], factor_sha256={"2": None, "3": None}, phase="partitions"))
    srec = {"epoch": ep, "identity": old, "layers": {}}
    for l in (2, 3):
        H = st["H"][l - 1]; D = st["D"][l - 1]; lrec = CL.block_reference(name, ep, l, H, D, S["run"], S["rd"], "cpu")
        parts = CL.make_partitions(st, l - 1, H, D, y, 3, ["labels", "kmeans_D"]); ap = CL.assignments_path(name, ep, l, 3)
        np.savez_compressed(ap, identity_digest=old["digest"], **{k: np.stack(v) for k, v in parts.items()}); fsha = P.sha256_file(ap)
        grec = {"applicable_kinds": ["labels", "kmeans_D"], "inapplicable_kinds": []}
        for kind in ("labels", "kmeans_D"):
            pp = CL.dim_P(H, D, parts[kind][0], "cpu", lrec["rank_J"], lrec["r_h"] * lrec["r_d"]); rhrd = lrec["r_h"] * lrec["r_d"]
            grec[kind] = dict(CL.partition_stats(parts[kind][0], y), dim_P=pp["dim_P"], dim_P_ladder=pp["ladder"], dim_P_gap=pp["gap"], dim_P_resolved=pp["resolved"], inequalities_hold=pp["inequalities_hold"],
                              within=pp["dim_P"] - lrec["rank_J"], across=rhrd - pp["dim_P"], across_share_of_delta=(rhrd - pp["dim_P"]) / lrec["delta_joint"], assignments_sha256=fsha)   # the fields the earlier code wrote
        grec["D_cluster_comparisons"] = {"labels": {"ARI": 0.0, "AMI": 0.0}, "assignments_sha256": fsha}
        lrec["3"] = grec; srec["layers"][str(l)] = lrec
    pj = os.path.join(out, f"{name}_partitions.json"); json.dump({"protocol": CL.PROTOCOL, "results": {"high_accuracy": srec}}, open(pj, "w"))
    with pytest.raises(RuntimeError):                                              # the new code refuses to touch a legacy record
        CL.partitions_phase(args("partitions", layers=[2], kinds=["labels"]))
    CL.migrate_phase(args("migrate")); r = rec(pj)["results"]["high_accuracy"]
    assert "identity" not in r and "migrated_from" in r and r["stage_identity"]["config"] == CL.base_config()
    for l in (2, 3):
        L = r["layers"][str(l)]; assert L["identity"]["config"]["layer"] == l
        z = np.load(CL.assignments_path(name, ep, l, 3)); assert str(z["identity_digest"]) == L["identity"]["digest"]
        for kind in ("labels", "kmeans_D"):
            assert L["3"][kind]["assignment_sha256"] == CL.array_sha256(z[kind]) and "assignments_sha256" not in L["3"][kind]
            assert L["3"][kind]["dim_P"] == srec["layers"][str(l)]["3"][kind]["dim_P"]
    assert os.path.exists(os.path.join(out, [d for d in os.listdir(out) if d.startswith("legacy_")][0], f"{name}_partitions.json"))
    # after the migration a single-layer run reuses everything
    n = world["calls"]["dim_P"]; CL.partitions_phase(args("partitions", layers=[3], kinds=["labels", "kmeans_D"])); assert world["calls"]["dim_P"] == n
    CL.migrate_phase(args("migrate"))                                              # idempotent
    assert rec(pj)["results"]["high_accuracy"]["layers"]["3"]["3"]["labels"]["dim_P"] == srec["layers"]["3"]["3"]["labels"]["dim_P"]


def legacy_record(world, factor_sha=None):
    """Write a legacy partitions record and its assignment files (labels, kmeans_D; G = 3; layers 2, 3); returns the paths."""
    out = world["out"]; st = world["st"]; y = world["y"]; a = args("partitions"); name = "mnist_mlp_seed0"; ep = 7
    S = CL.stage_setup(a, 0, "high_accuracy"); fs = {"2": factor_sha, "3": factor_sha}
    old = CL._identity(S["run"], S["rd"], ep, dict(CL.base_config(), layers=[2, 3], factor_sha256=fs, phase="partitions"))
    srec = {"epoch": ep, "identity": old, "layers": {}}
    for l in (2, 3):
        H = st["H"][l - 1]; D = st["D"][l - 1]; lrec = CL.block_reference(name, ep, l, H, D, S["run"], S["rd"], "cpu")
        parts = CL.make_partitions(st, l - 1, H, D, y, 3, ["labels", "kmeans_D"]); ap = CL.assignments_path(name, ep, l, 3)
        np.savez_compressed(ap, identity_digest=old["digest"], **{k: np.stack(v) for k, v in parts.items()}); fsha = P.sha256_file(ap)
        grec = {"applicable_kinds": ["labels", "kmeans_D"], "inapplicable_kinds": []}
        for kind in ("labels", "kmeans_D"):
            pp = CL.dim_P(H, D, parts[kind][0], "cpu", lrec["rank_J"], lrec["r_h"] * lrec["r_d"]); rhrd = lrec["r_h"] * lrec["r_d"]
            grec[kind] = dict(CL.partition_stats(parts[kind][0], y), dim_P=pp["dim_P"], dim_P_ladder=pp["ladder"], dim_P_gap=pp["gap"], dim_P_resolved=pp["resolved"], inequalities_hold=pp["inequalities_hold"],
                              within=pp["dim_P"] - lrec["rank_J"], across=rhrd - pp["dim_P"], across_share_of_delta=(rhrd - pp["dim_P"]) / lrec["delta_joint"], assignments_sha256=fsha)
        lrec["3"] = grec; srec["layers"][str(l)] = lrec
    pj = os.path.join(out, f"{name}_partitions.json"); json.dump({"protocol": CL.PROTOCOL, "results": {"high_accuracy": srec}}, open(pj, "w"))
    return pj, old


def test_migration_refuses_a_replaced_factor(world, monkeypatch):
    """The legacy record was produced with factor hash 'f1'; the factor on disk now hashes to 'f2': the stage is refused."""
    pj, old = legacy_record(world, factor_sha="f1")
    monkeypatch.setattr(CL, "_factor_sha", lambda name, ep, l, run, rd: "f2")
    with pytest.raises(P.ProvenanceError):
        CL.migrate_phase(args("migrate"))
    r = rec(pj)["results"]["high_accuracy"]; assert "identity" in r and r["identity"]["digest"] == old["digest"]     # untouched
    z = np.load(CL.assignments_path("mnist_mlp_seed0", 7, 2, 3)); assert str(z["identity_digest"]) == old["digest"]


def test_migration_resumes_after_an_interruption(world, monkeypatch):
    """An interruption between the record rewrite and the file rewrite leaves files under the legacy digest; the next run
    completes them, and a run in between refuses the files with a clear error rather than reusing them."""
    pj, old = legacy_record(world)
    real = CL.save_assignments; calls = {"n": 0}
    def interrupt(*a_, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*a_, **k)
    monkeypatch.setattr(CL, "save_assignments", interrupt)
    with pytest.raises(KeyboardInterrupt):
        CL.migrate_phase(args("migrate"))
    r = rec(pj)["results"]["high_accuracy"]; assert "migrated_from" in r and r["migrated_from"]["files_rewritten"] is False
    digs = {l: str(np.load(CL.assignments_path("mnist_mlp_seed0", 7, l, 3))["identity_digest"]) for l in (2, 3)}
    assert sorted(digs.values()) != sorted(r["layers"][l]["identity"]["digest"] for l in ("2", "3"))            # one file still legacy
    with pytest.raises(P.ProvenanceError):
        CL.partitions_phase(args("partitions", layers=[3], kinds=["labels", "kmeans_D"]))                          # not silently reused
    monkeypatch.setattr(CL, "save_assignments", real)
    CL.migrate_phase(args("migrate")); r = rec(pj)["results"]["high_accuracy"]; assert r["migrated_from"]["files_rewritten"] is True
    for l in ("2", "3"):
        assert str(np.load(CL.assignments_path("mnist_mlp_seed0", 7, int(l), 3))["identity_digest"]) == r["layers"][l]["identity"]["digest"]
    n = world["calls"]["dim_P"]; CL.partitions_phase(args("partitions", layers=[2, 3], kinds=["labels", "kmeans_D"])); assert world["calls"]["dim_P"] == n
