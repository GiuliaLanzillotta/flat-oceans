"""
Linear mode connectivity of spawned pairs (plan Section 0.8 item 6, bounded version approved on 2026-09-25):
recovery of a parent's optimizer state at a spawn checkpoint, and the analysis of the segment between two endpoints.

Phase `spawn-state`.  A parent run has weight checkpoints on its schedule but Adam moments only at the recovery
checkpoints (every 50 epochs).  The state at a spawn epoch E is recovered by an exact replay: the parent's stored
initialization (init seed), stored preprocessing and sample sets, the parent's order seed and the same update loop as
the trainer, from epoch 1 to E, in float64 with deterministic kernels (on a GPU this reproduces the parent to rounding, about 1e-14, not bitwise).
Verification, all recorded: the replayed weights
against the stored checkpoint of epoch E (largest absolute difference, bitwise-equal tensors); the replayed moments and
step counters against the recovery checkpoint of epoch E when one exists; and a same-order continuation to the next
stored checkpoint E' > E, whose weights (and moments, when E' has a recovery file) must again match.  The state at E is
saved once per (parent, E) and cloned by every child.  The Adam policy recorded in the state (and in every child's
configuration) is "inherited" with the measured largest difference only when every check passes the tolerance; otherwise the state is marked failed and the experiment must use a uniform reset for every spawn.

Phase `segment`.  For a pair of endpoints theta_A, theta_B (two children of one spawn, or two independently
initialized parents as a control) and 31 points alpha in [0, 1] (30 evenly spaced plus 1/2), on the analysis set and on
the test set: the function change ||F(theta_alpha) - F(theta_A)||_X, ||F(theta_alpha) - F(theta_B)||_X and the
deviation from the interpolated outputs D(alpha) = F(theta_alpha) - [(1 - alpha) F_A + alpha F_B], absolute and
relative to ||F_B - F_A||_X and to ||F_A||_X, with per-sample quantiles; the exact split of F(theta_alpha) - F_A at
w = alpha (theta_B - theta_A) into the linear part (sum of the single-block frozen changes, exact by multi-affinity),
the cross-layer part C (frozen minus linear) and the transition part T (true minus frozen; also by propagation, with
the identity residual), from which D(alpha) = C(alpha) - alpha C(1) + T(alpha) - alpha T(1) (the linear part cancels;
the residual of this identity is recorded); gate agreement with both endpoints and crossings per layer; MSE and error
with the sampled barrier estimates (Entezari: largest excess of the MSE above the chord between the endpoint losses;
Frankle: largest error minus the mean endpoint error).  Unaligned weights; no permutation search.

    python -m flatocean lmc spawn-state --seeds 0 1 2 --stages early_fitting high_accuracy
    python -m flatocean lmc segment --pairs siblings controls
    python -m flatocean lmc test
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fnn

from flatocean import frozen_lib as FL
from flatocean import provenance as P
from flatocean import tier1_train as T
from flatocean import records, runs
from flatocean.runs import stage_epochs

OUT = lambda: runs.out_dir("lmc")                      # noqa: E731
PROTOCOL = "EXPERIMENTS_PLAN.md Section 0.8 item 6, revised 2026-09-26 (inherited Adam from recovery files, later spawns, displacement shares, loss-connected rule, orbit projection); identifier lmc-v2"
CONNECTED_FACTOR = 10.0          # a pair is loss-connected when its MSE barrier on the analysis set is below CONNECTED_FACTOR times the mean endpoint loss
REPLAY_TOL = 1e-12                                     # largest absolute weight or moment difference accepted as an exact replay
ALPHAS = sorted(set(np.linspace(0.0, 1.0, 30).tolist()) | {0.5})


# ----------------------------------------------------------------------------- spawn state
def _tensor_diff(a, b):
    """Largest absolute difference and the number of bitwise-equal tensors between two state dicts."""
    keys = list(a.keys()); assert keys == list(b.keys()), "state dicts differ in their keys"
    diffs = [float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0.0 for k in keys]
    return {"max_abs": max(diffs) if diffs else 0.0, "n_tensors": len(keys), "n_bitwise_equal": int(sum(torch.equal(a[k], b[k]) for k in keys))}


def _opt_diff(state_a, state_b):
    """Largest absolute difference of the moments and the agreement of the step counters between two Adam state dicts."""
    sa, sb = state_a["state"], state_b["state"]; assert set(sa) == set(sb)
    out = {"max_abs_exp_avg": 0.0, "max_abs_exp_avg_sq": 0.0, "steps_equal": True}
    for k in sa:
        for m in ("exp_avg", "exp_avg_sq"):
            out[f"max_abs_{m}"] = max(out[f"max_abs_{m}"], float((sa[k][m].double() - sb[k][m].double()).abs().max()))
        out["steps_equal"] = out["steps_equal"] and float(sa[k]["step"]) == float(sb[k]["step"])
    return out


def _cpu_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _cpu_opt(opt):
    sd = copy.deepcopy(opt.state_dict())
    for st in sd["state"].values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.cpu()
    return sd


def replay(parent, epoch_to, device, log=print):
    """Replay the parent's training from its initialization to `epoch_to` with the trainer's own loop; returns
    (model, optimizer, data) with the data kept for a continuation."""
    T.set_deterministic()
    config = json.load(open(os.path.join(parent, "config.json"))); dtype = {"float64": torch.float64, "float32": torch.float32}[config["dtype"]]
    pre = torch.load(os.path.join(parent, "preprocessing.pt"), weights_only=False); manifest = json.load(open(os.path.join(parent, "data_manifest.json")))
    Xtr_raw, ytr_np, _, _, hashes = T.load_raw(config["dataset"])
    if hashes != manifest["raw_file_sha256"]:
        raise RuntimeError("raw data hashes differ from the parent's manifest")
    Xtr = T.apply_preprocessing(Xtr_raw, pre, dtype).to(device); ytr = torch.from_numpy(ytr_np).to(device)
    Ytr = Fnn.one_hot(ytr, config["d_out"]).to(dtype); s = Xtr.shape[0]
    model = T.build_model(config, dtype).to(device); opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    if config.get("schedule", "none") != "none" or config.get("augment"):
        raise RuntimeError("replay supports the constant-rate, unaugmented recipe only")
    data = dict(config=config, Xtr=Xtr, Ytr=Ytr, s=s)
    advance(model, opt, data, 1, epoch_to, device, log)
    return model, opt, data


def advance(model, opt, data, epoch_from, epoch_to, device, log=print):
    """Epochs epoch_from..epoch_to of the trainer's loop (order seed of the parent's config, batch size, MSE, Adam)."""
    cfg = data["config"]; Xtr, Ytr, s = data["Xtr"], data["Ytr"], data["s"]; t0 = time.time()
    for epoch in range(epoch_from, epoch_to + 1):
        model.train(); perm = T.epoch_permutation(cfg["order_seed"], epoch, s).to(device)
        for i in range(0, s, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            loss = T.mse_loss(model(Xtr[idx]), Ytr[idx]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
    log(f"   replayed epochs {epoch_from}..{epoch_to} in {time.time() - t0:.0f}s")


def next_saved_epoch(parent, epoch):
    eps = sorted(int(f[6:10]) for f in os.listdir(os.path.join(parent, "checkpoints")) if f.startswith("epoch_"))
    later = [e for e in eps if e > epoch]
    return later[0] if later else None


def recover_spawn_state(parent, epoch, out_path, device="cuda", log=print):
    """Recover and verify the parent's state at `epoch`; write the state file; return the verification record."""
    ck = os.path.join(parent, "checkpoints"); stored = torch.load(os.path.join(ck, f"epoch_{epoch:04d}.pt"), map_location="cpu", weights_only=False)
    model, opt, data = replay(parent, epoch, device, log)
    ver = {"epoch": epoch, "tolerance": REPLAY_TOL, "weights_vs_checkpoint": _tensor_diff(_cpu_state(model.state_dict()), stored["model"])}
    rec_path = os.path.join(ck, f"recovery_epoch_{epoch:04d}.pt")
    if os.path.exists(rec_path):
        r = torch.load(rec_path, map_location="cpu", weights_only=False); ver["moments_vs_recovery_checkpoint"] = _opt_diff(_cpu_opt(opt), r["optimizer"])
    else:
        ver["moments_vs_recovery_checkpoint"] = None
    state_model, state_opt = _cpu_state(model.state_dict()), _cpu_opt(opt)
    e2 = next_saved_epoch(parent, epoch)
    if e2 is not None:                                                            # same-order continuation: the recovered moments must reproduce the parent
        advance(model, opt, data, epoch + 1, e2, device, log)
        stored2 = torch.load(os.path.join(ck, f"epoch_{e2:04d}.pt"), map_location="cpu", weights_only=False)
        cont = {"epoch": e2, "weights_vs_checkpoint": _tensor_diff(_cpu_state(model.state_dict()), stored2["model"])}
        rec2 = os.path.join(ck, f"recovery_epoch_{e2:04d}.pt")
        cont["moments_vs_recovery_checkpoint"] = _opt_diff(_cpu_opt(opt), torch.load(rec2, map_location="cpu", weights_only=False)["optimizer"]) if os.path.exists(rec2) else None
        ver["same_order_continuation"] = cont
    else:
        ver["same_order_continuation"] = None
    checks = [ver["weights_vs_checkpoint"]["max_abs"] <= REPLAY_TOL]
    for m in (ver["moments_vs_recovery_checkpoint"], (ver["same_order_continuation"] or {}).get("moments_vs_recovery_checkpoint")):
        if m is not None:
            checks.append(m["max_abs_exp_avg"] <= REPLAY_TOL and m["max_abs_exp_avg_sq"] <= REPLAY_TOL and m["steps_equal"])
    if ver["same_order_continuation"] is not None:
        checks.append(ver["same_order_continuation"]["weights_vs_checkpoint"]["max_abs"] <= REPLAY_TOL)
    ver["passed"] = bool(all(checks)); ver["n_checks"] = len(checks)
    hist = json.load(open(os.path.join(parent, "history.json")))
    obj = {"protocol": PROTOCOL, "parent_run": os.path.abspath(parent), "epoch": epoch, "config": data["config"],
           "parent_checkpoint_sha256": P.sha256_file(os.path.join(ck, f"epoch_{epoch:04d}.pt")), "model": state_model, "optimizer": state_opt,
           "path_length": stored["path_length"], "milestones": stored["milestones"], "parent_history": hist["epochs"],
           "verification": ver, "adam_policy": policy_label(ver), "adam_policy_kind": policy_kind(ver)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True); tmp = out_path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, out_path)
    log(f"== {os.path.basename(os.path.dirname(parent))}/{os.path.basename(parent)} epoch {epoch}: weights max|diff| {ver['weights_vs_checkpoint']['max_abs']:.1e} "
        f"({ver['weights_vs_checkpoint']['n_bitwise_equal']}/{ver['weights_vs_checkpoint']['n_tensors']} bitwise), moments {ver['moments_vs_recovery_checkpoint']}, "
        f"continuation {ver['same_order_continuation']}, passed {ver['passed']}")
    return ver


def recovery_spawn_state(parent, epoch, out_path, log=print):
    """The inherited policy from a recovery file (2026-09-26 revision): the parent's Adam moments and step counters at
    `epoch` are read from checkpoints/recovery_epoch_<epoch>.pt, which the trainer writes every RECOVERY_EVERY epochs
    together with the weights.  Verified: the recovery file's weights equal the stored epoch checkpoint (tolerance
    REPLAY_TOL, bitwise count recorded), every parameter's step counter is the same and a multiple of the epoch.  Spawns
    are only possible at epochs with a recovery file; anything else is refused."""
    ck = os.path.join(parent, "checkpoints"); rec_path = os.path.join(ck, f"recovery_epoch_{epoch:04d}.pt")
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"{parent}: no recovery file at epoch {epoch}; the inherited policy spawns only where the parent's optimizer state is stored")
    stored = torch.load(os.path.join(ck, f"epoch_{epoch:04d}.pt"), map_location="cpu", weights_only=False)
    r = torch.load(rec_path, map_location="cpu", weights_only=False)
    if int(r.get("epoch", epoch)) != epoch:
        raise ValueError(f"{rec_path}: records epoch {r.get('epoch')}, expected {epoch}")
    wdiff = _tensor_diff(_cpu_state(r["model"]), stored["model"])
    steps = sorted({int(v["step"]) for v in r["optimizer"]["state"].values()})
    ver = {"epoch": epoch, "policy": "recovery", "tolerance": REPLAY_TOL, "recovery_file": os.path.abspath(rec_path), "recovery_sha256": P.sha256_file(rec_path),
           "weights_vs_checkpoint": wdiff, "steps": steps, "steps_per_epoch": (steps[0] // epoch if len(steps) == 1 and epoch > 0 and steps[0] % epoch == 0 else None),
           "moments": "exp_avg, exp_avg_sq and step of every parameter from the recovery file"}
    ver["passed"] = bool(wdiff["max_abs"] <= REPLAY_TOL and len(steps) == 1 and ver["steps_per_epoch"] is not None and len(r["optimizer"]["state"]) == len(stored["model"]))
    ver["usable"] = ver["passed"]
    hist = json.load(open(os.path.join(parent, "history.json"))); config = json.load(open(os.path.join(parent, "config.json")))
    obj = {"protocol": PROTOCOL, "parent_run": os.path.abspath(parent), "epoch": epoch, "config": config,
           "parent_checkpoint_sha256": P.sha256_file(os.path.join(ck, f"epoch_{epoch:04d}.pt")), "model": _cpu_state(stored["model"]), "optimizer": r["optimizer"],
           "path_length": stored["path_length"], "milestones": stored["milestones"], "parent_history": hist["epochs"], "verification": ver,
           "adam_policy": policy_label(ver), "adam_policy_kind": policy_kind(ver)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True); tmp = out_path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, out_path)
    log(f"== {os.path.basename(os.path.dirname(parent))}/{os.path.basename(parent)} epoch {epoch}: inherited from the recovery file, weights max|diff| {wdiff['max_abs']:.1e} "
        f"({wdiff['n_bitwise_equal']}/{wdiff['n_tensors']} bitwise), steps {steps}, passed {ver['passed']}")
    return ver


def policy_kind(ver):
    """The fixed policy identifier of a state: 'inherited' (moments recovered by a verified replay, or read from the parent's
    recovery file and verified), 'reset', or 'failed' (a verification outside the tolerance; the state must not be used).
    The measured residuals stay in the verification record."""
    if ver.get("policy") == "reset":
        return "reset"
    return "inherited" if ver["passed"] else "failed"


def policy_label(ver):
    """A human-readable description of the policy; the machine-readable identity is `policy_kind`."""
    kind = policy_kind(ver)
    if kind == "failed":
        return "FAILED replay: do not use; reset policy required"
    if kind == "reset":
        return "reset (uniform for every spawn; see the replay attempt in the verification)"
    if ver.get("policy") == "recovery":
        bit = ver["weights_vs_checkpoint"]["n_bitwise_equal"] == ver["weights_vs_checkpoint"]["n_tensors"]
        return f"inherited (Adam moments and step counters from the parent's recovery file {os.path.basename(ver['recovery_file'])}, sha256 {ver['recovery_sha256'][:12]}; weights {'bitwise' if bit else 'not bitwise'} equal to the checkpoint)"
    worst = max([ver["weights_vs_checkpoint"]["max_abs"]] + ([ver["same_order_continuation"]["weights_vs_checkpoint"]["max_abs"]] if ver["same_order_continuation"] else []))
    bit = ver["weights_vs_checkpoint"]["n_bitwise_equal"] == ver["weights_vs_checkpoint"]["n_tensors"]
    return f"inherited (replay verified to tolerance {REPLAY_TOL:g}: largest weight difference {worst:.1e}, {'bitwise' if bit else 'not bitwise'})"


def reset_spawn_state(parent, epoch, out_path, replay_evidence=None, log=print):
    """The uniform reset policy: the stored weights of the spawn checkpoint and a fresh Adam state (zero moments, step 0)
    for every spawn alike; the replay attempt that motivated the policy is kept as evidence in the state file."""
    ck = os.path.join(parent, "checkpoints"); stored = torch.load(os.path.join(ck, f"epoch_{epoch:04d}.pt"), map_location="cpu", weights_only=False)
    config = json.load(open(os.path.join(parent, "config.json"))); dtype = {"float64": torch.float64, "float32": torch.float32}[config["dtype"]]
    model = T.build_model(config, dtype); model.load_state_dict(stored["model"]); opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    ver = {"epoch": epoch, "policy": "reset", "passed": True, "usable": True, "weights": "stored checkpoint of the parent", "moments": "reset to zero, step counter 0, for every spawn alike",
           "replay_attempt": replay_evidence, "caveat": "the first updates after the spawn are bias-corrected Adam steps from zero moments; identical for every child and stage"}
    hist = json.load(open(os.path.join(parent, "history.json")))
    obj = {"protocol": PROTOCOL, "parent_run": os.path.abspath(parent), "epoch": epoch, "config": config,
           "parent_checkpoint_sha256": P.sha256_file(os.path.join(ck, f"epoch_{epoch:04d}.pt")), "model": _cpu_state(model.state_dict()), "optimizer": _cpu_opt(opt),
           "path_length": stored["path_length"], "milestones": stored["milestones"], "parent_history": hist["epochs"], "verification": ver,
           "adam_policy": policy_label(ver), "adam_policy_kind": "reset"}
    os.makedirs(os.path.dirname(out_path), exist_ok=True); tmp = out_path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, out_path)
    log(f"== {os.path.basename(os.path.dirname(parent))}/{os.path.basename(parent)} epoch {epoch}: reset policy, weights from the stored checkpoint, fresh Adam")
    return ver


def spawn_state_phase(a):
    torch.set_num_threads(a.threads); out = os.path.join(OUT(), "spawn_states"); summary = {}; policies = set()
    for seed in a.seeds:
        parent = a.parent_dir or runs.run_dir(a.dataset, a.model, seed, a.tag); st = stage_epochs(parent)
        epochs = [int(e) for e in a.spawn_epochs] if a.spawn_epochs else [st[s_]["checkpoint_epoch"] for s_ in a.stages]
        for ep in epochs:
            name = f"{runs.run_name(a.dataset, a.model, seed, a.tag)}_epoch{ep:04d}"; path = os.path.join(out, name + ".pt"); sha = P.sha256_file(os.path.join(parent, "checkpoints", f"epoch_{ep:04d}.pt"))
            old = torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None
            if old is not None and old.get("parent_checkpoint_sha256") != sha:
                old = None
            if a.policy == "replay":
                if old is not None and old["verification"].get("policy", "replay") == "replay" and not a.overwrite:
                    print(f"== {name}: replay state exists (passed {old['verification']['passed']})", flush=True); summary[name] = old["verification"]
                else:
                    summary[name] = recover_spawn_state(parent, ep, path, a.device)
                summary[name]["usable"] = bool(summary[name]["passed"])
            elif a.policy == "recovery":
                if old is not None and old["verification"].get("policy") == "recovery" and not a.overwrite:
                    print(f"== {name}: recovery state exists (passed {old['verification']['passed']})", flush=True); summary[name] = old["verification"]
                else:
                    summary[name] = recovery_spawn_state(parent, ep, path)
            else:
                evidence = old["verification"] if (old is not None and old["verification"].get("policy", "replay") == "replay") else (old["verification"].get("replay_attempt") if old is not None else None)
                if old is not None and old["verification"].get("policy") == "reset" and not a.overwrite:
                    print(f"== {name}: reset state exists", flush=True); summary[name] = old["verification"]
                else:
                    summary[name] = reset_spawn_state(parent, ep, path, evidence)
            policies.add(torch.load(path, map_location="cpu", weights_only=False)["adam_policy_kind"])
    ok = all(v.get("usable", v["passed"]) for v in summary.values()) and len(policies) == 1 and "failed" not in policies
    vpath = os.path.join(out, "verification.json"); prev = json.load(open(vpath)) if os.path.exists(vpath) else {}
    states = dict(prev.get("states") or {}); states.update(summary)                # states of earlier invocations (other policies, other epochs) stay on file
    invocations = list(prev.get("invocations") or []) + [{"policy": a.policy, "states": sorted(summary), "all_usable_and_uniform": ok, "policies_recorded": sorted(policies)}]
    records.dump_json(vpath, {"protocol": PROTOCOL, "policy": a.policy, "policies_recorded": sorted(policies), "all_usable_and_uniform": ok, "states": states, "invocations": invocations})
    print("all usable and one policy:", ok, sorted(policies), flush=True)
    if not ok:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- segment analysis
def _forward(net, X, block=2000):
    with torch.no_grad():
        return torch.cat([net(X[i:i + block]) for i in range(0, X.shape[0], block)])


def _frozen(net, masks, X, block=1000):
    with torch.no_grad():
        return torch.cat([FL.frozen_forward(net, masks, X, torch.arange(i, min(i + block, X.shape[0]), device=X.device)) for i in range(0, X.shape[0], block)])


def _propagated(net, masks, X, block=1000):
    tot = None; flips = None
    with torch.no_grad():
        for i in range(0, X.shape[0], block):
            idx = torch.arange(i, min(i + block, X.shape[0]), device=X.device); _, p, fl = FL.residual_propagation(net, masks, X, idx)
            tot = p if tot is None else torch.cat([tot, p]); flips = fl if flips is None else [x + y for x, y in zip(flips, fl)]
    return tot, flips


def masks_agreement(net, X, masks_A, masks_B):
    """Per layer: fraction of gates equal to A's and to B's, and the number differing from A (crossings)."""
    m = FL.base_masks(net, X); out = []
    for l in range(len(m)):
        eqA = (m[l] == masks_A[l]).double().mean().item(); eqB = (m[l] == masks_B[l]).double().mean().item()
        out.append({"agree_A": eqA, "agree_B": eqB, "crossings_from_A": int((m[l] != masks_A[l]).sum())})
    return out


def decompose(net, masks_A, X, theta_A, w, F_A, slices):
    """At theta_A + w with A's masks: (true, frozen, linear, cross, transition, propagated, flips) as stacked outputs."""
    FL.set_flat(net, theta_A + w); true = _forward(net, X); frz = _frozen(net, masks_A, X); prop, flips = _propagated(net, masks_A, X)
    lin = torch.zeros_like(F_A)
    for _, s0, s1 in slices:                                                      # single-block changes are exactly linear in the frozen model
        wl = torch.zeros_like(w); wl[s0:s1] = w[s0:s1]; FL.set_flat(net, theta_A + wl); lin += _frozen(net, masks_A, X) - F_A
    FL.set_flat(net, theta_A)
    return {"true": true, "frozen": frz, "linear": lin, "cross": frz - F_A - lin, "transition": true - frz, "propagated": prop, "flips": flips}


def xn(V):
    return float(FL.x_norm(V))


def quantiles(v):
    q = np.quantile(v, [0.5, 0.9, 0.99, 1.0]); return {"median": float(q[0]), "q90": float(q[1]), "q99": float(q[2]), "max": float(q[3])}


def loss_error(F, Y, y):
    return {"mse": float(((F - Y) ** 2).mean()), "errors": int((F.argmax(1) != y).sum()), "error_rate": float((F.argmax(1) != y).double().mean())}


def orbit_projection(net, theta_A, w):
    """Share of the displacement w that lies in the tangent space of the rescaling orbit at theta_A (span of the per-unit
    generators, orthonormalized): the norm share ||P w|| / ||w|| and the squared share, with the span's dimension."""
    from flatocean.analyses import symmetries as SY
    FL.set_flat(net, theta_A); G, labels, _ = SY.unit_generators(net); FL.set_flat(net, theta_A)
    Q, R = torch.linalg.qr(G.to(w.dtype)); keep = R.diagonal().abs() > 1e-12 * R.diagonal().abs().max(); Q = Q[:, keep]
    c = Q.T @ w; share = float(c.norm() / w.norm()) if float(w.norm()) > 0 else None
    return {"dim": int(Q.shape[1]), "n_generators": len(labels), "norm_share": share, "squared_share": None if share is None else share ** 2,
            "definition": "||P_orbit w|| / ||w|| with P_orbit the orthogonal projector onto the span of the rescaling generators at endpoint A"}


def segment_record(net, theta_A, theta_B, sets, alphas=ALPHAS, projections=True):
    """The full record of one segment; `sets` = {name: (X, Y_onehot, y)}."""
    slices = FL.block_slices(net); w1 = theta_B - theta_A; w2 = float(w1.norm() ** 2)
    out = {"alphas": list(alphas), "sets": {}, "displacement_norm": float(w1.norm()),
           "block_displacement_fractions": {n: float(w1[s0:s1].norm() / theta_A[s0:s1].norm()) for n, s0, s1 in slices},
           "block_displacement_norms": {n: float(w1[s0:s1].norm()) for n, s0, s1 in slices},
           "block_displacement_shares": {n: (float(w1[s0:s1].norm() ** 2) / w2 if w2 > 0 else None) for n, s0, s1 in slices},
           "endpoint_A_norm": float(theta_A.norm()), "endpoint_B_norm": float(theta_B.norm()),
           "projections": {"rescaling_orbit_at_A": orbit_projection(net, theta_A, w1) if projections else None,
                           "certified_branch_at_A": None, "tangent_kernel_at_A": None,
                           "note": "branch and tangent-kernel projections need the width-128 certified branch and tangent kernel (plan 0.7 task E4), not built"}}
    for sname, (X, Y, y) in sets.items():
        FL.set_flat(net, theta_A); F_A = _forward(net, X); masks_A = FL.base_masks(net, X)
        FL.set_flat(net, theta_B); F_B = _forward(net, X); masks_B = FL.base_masks(net, X); FL.set_flat(net, theta_A)
        dAB = xn(F_B - F_A); nA = xn(F_A); base = {"F_A_norm": nA, "F_B_norm": xn(F_B), "endpoint_difference": dAB, "loss_A": loss_error(F_A, Y, y), "loss_B": loss_error(F_B, Y, y)}
        d1 = decompose(net, masks_A, X, theta_A, w1, F_A, slices); C1, T1 = d1["cross"], d1["transition"]
        pts = []
        for al in alphas:
            w = al * w1; FL.set_flat(net, theta_A + w); gates = masks_agreement(net, X, masks_A, masks_B)      # gates at theta_alpha, before decompose resets to A
            d = decompose(net, masks_A, X, theta_A, w, F_A, slices); Fa = d["true"]
            D = Fa - ((1 - al) * F_A + al * F_B); D_pred = d["cross"] - al * C1 + d["transition"] - al * T1
            per = (Fa - F_A).norm(dim=1).cpu().numpy() / max(dAB, 1e-300)
            pts.append({"alpha": al, "change_from_A": xn(Fa - F_A), "change_from_B": xn(Fa - F_B), "interpolation_deviation": xn(D),
                        "change_from_A_rel_endpoint": xn(Fa - F_A) / dAB if dAB > 0 else None, "interpolation_deviation_rel_endpoint": xn(D) / dAB if dAB > 0 else None,
                        "interpolation_deviation_rel_base": xn(D) / nA if nA > 0 else None, "per_sample_change_from_A_over_endpoint": quantiles(per),
                        "linear": xn(d["linear"]), "cross_layer": xn(d["cross"]), "transition": xn(d["transition"]), "transition_propagated": xn(d["propagated"]),
                        "identity_residual": xn(d["transition"] - d["propagated"]), "deviation_identity_residual": xn(D - D_pred),
                        "gates": gates, "flips_from_A": d["flips"], "loss": loss_error(Fa, Y, y)})
            FL.set_flat(net, theta_A)
        L = np.array([p["loss"]["mse"] for p in pts]); E = np.array([p["loss"]["error_rate"] for p in pts]); A = np.array(alphas)
        chord = (1 - A) * L[0] + A * L[-1]
        out["sets"][sname] = {**base, "points": pts,
                              "barriers": {"note": "sampled estimates on the named set over the alpha grid; not full-training-set barriers",
                                           "entezari_mse": {"value": float((L - chord).max()), "alpha": float(A[(L - chord).argmax()]), "definition": "max_alpha L(alpha) - [(1-alpha) L(0) + alpha L(1)]"},
                                           "frankle_error": {"value": float((E - 0.5 * (E[0] + E[-1])).max()), "alpha": float(A[(E - 0.5 * (E[0] + E[-1])).argmax()]), "definition": "max_alpha E(alpha) - (E(0) + E(1)) / 2"},
                                           "at_half": {"mse_excess": float(L[A == 0.5][0] - chord[A == 0.5][0]), "error_excess": float(E[A == 0.5][0] - 0.5 * (E[0] + E[-1]))}},
                              "max_interpolation_deviation_rel_endpoint": max((p["interpolation_deviation_rel_endpoint"] or 0) for p in pts),
                              "max_identity_residual": max(p["identity_residual"] for p in pts), "max_deviation_identity_residual": max(p["deviation_identity_residual"] for p in pts)}
        b = out["sets"][sname]["barriers"]; mean_end = 0.5 * (L[0] + L[-1])
        b["loss_connected"] = {"value": bool(b["entezari_mse"]["value"] < CONNECTED_FACTOR * mean_end), "mean_endpoint_mse": float(mean_end), "factor": CONNECTED_FACTOR,
                               "definition": "MSE barrier (largest excess above the chord) below factor x mean endpoint MSE on this set"}
    if "analysis" in out["sets"]:
        out["loss_connected"] = out["sets"]["analysis"]["barriers"]["loss_connected"]["value"]      # the rule is decided on the analysis set
    return out


def endpoint(dataset, model, seed, tag, epoch, device):
    r = runs.load_run(dataset, model, seed, tag=tag, epoch=epoch, device=device, sets=("analysis", "test")); rd = runs.run_dir(dataset, model, seed, tag)
    return r, rd, P.sha256_file(os.path.join(rd, "checkpoints", f"epoch_{epoch:04d}.pt"))


def check_compatible(rA, rB, rdA, rdB):
    """Two endpoints share a parameter space and an input map only when their architecture, activation, input dimension,
    parameter layout and stored preprocessing agree; anything else makes the segment meaningless.  Returns B's
    preprocessing hash for the identity."""
    ca, cb = rA["config"], rB["config"]
    for k in ("model", "widths", "activation", "q", "d_out", "dataset", "arch", "input"):
        if ca.get(k) != cb.get(k):
            raise RuntimeError(f"endpoints differ in {k}: {ca.get(k)} vs {cb.get(k)}")
    sa = [(n, tuple(p.shape)) for n, p in rA["net"].named_parameters()]; sb = [(n, tuple(p.shape)) for n, p in rB["net"].named_parameters()]
    if sa != sb:
        raise RuntimeError("endpoints differ in their parameter layout")
    ha, hb = P.sha256_file(os.path.join(rdA, "preprocessing.pt")), P.sha256_file(os.path.join(rdB, "preprocessing.pt"))
    if ha != hb:
        raise RuntimeError(f"endpoints differ in their stored preprocessing ({ha[:12]} vs {hb[:12]})")
    ma, mb = json.load(open(os.path.join(rdA, "data_manifest.json"))), json.load(open(os.path.join(rdB, "data_manifest.json")))
    if ma["analysis_indices"] != mb["analysis_indices"] or ma["raw_file_sha256"] != mb["raw_file_sha256"]:
        raise RuntimeError("endpoints differ in their analysis set or raw data")
    return hb


def find_pairs(a):
    """(name, endpoint A, endpoint B, kind): siblings = the two children of one spawn; controls = the pairs of parents."""
    pairs = []
    if "siblings" in a.pairs:
        base = os.path.join(T.RESULTS)
        import re
        for d in sorted(os.listdir(base)):
            m = re.match(rf"{a.dataset}_{a.model}_(lmc\w*?_s(\d+)_e(\d+))_c0$", d)       # lmc_s<seed>_e<epoch>_c0 (MLP) or lmc_cnn_s<seed>_e<epoch>_c0
            if m:
                tag0 = m.group(1) + "_c0"; tag1 = m.group(1) + "_c1"; seed = int(m.group(2))
                if os.path.exists(os.path.join(base, f"{a.dataset}_{a.model}_{tag1}")):
                    pairs.append((f"siblings_{m.group(1)}", (seed, tag0), (seed, tag1), "siblings: two children of one spawn, different order seeds"))
    if "controls" in a.pairs:
        mt = f"{a.dataset}_{a.model}{'_' + a.tag if a.tag else ''}"                # MLP and CNN controls must not collide in the shared segments directory
        for i, j in ((0, 1), (0, 2), (1, 2)):
            pairs.append((f"control_parents_{mt}_s{i}_s{j}", (i, a.tag), (j, a.tag), "control: independently initialized parents at the endpoint (three dependent pairwise comparisons of three endpoints)"))
    return pairs


def segment_phase(a):
    torch.set_num_threads(a.threads); out = os.path.join(OUT(), "segments"); os.makedirs(out, exist_ok=True)
    for name, (sa, ta), (sb, tb), kind in find_pairs(a):
        path = os.path.join(out, f"{name}.json")
        rA, rdA, hA = endpoint(a.dataset, a.model, sa, ta, a.epoch, a.device); rB, rdB, hB = endpoint(a.dataset, a.model, sb, tb, a.epoch, a.device)
        preB = check_compatible(rA, rB, rdA, rdB)
        ident = records.identity(rA, rdA, a.epoch, {"protocol": PROTOCOL, "endpoint_B": {"run": rdB, "epoch": a.epoch, "checkpoint_sha256": hB, "preprocessing_sha256": preB}, "alphas": list(ALPHAS), "sets": ["analysis", "test"]})
        if os.path.exists(path) and not a.overwrite:
            old = json.load(open(path))
            if old.get("identity", {}).get("digest") == ident["digest"]:
                print(f"== {name}: complete under the current identity", flush=True); continue
        net = rA["net"]; theta_A = FL.flat_params(net).detach().clone(); theta_B = FL.flat_params(rB["net"]).detach().clone(); t0 = time.time()
        d_out = rA["config"]["d_out"]; sets = {}
        for sname in ("analysis", "test"):
            X, y, _ = rA["data"][sname]; sets[sname] = (X, Fnn.one_hot(y, d_out).to(X.dtype), y)
        spawn = {}
        for lab, rd in (("A", rdA), ("B", rdB)):
            sp = (json.load(open(os.path.join(rd, "config.json"))).get("spawn") or {})
            spawn[lab] = {"epoch": sp.get("epoch"), "adam_policy_kind": T._policy_kind_of(sp) if sp else None, "state_sha256": sp.get("state_sha256"), "parent_run": sp.get("parent_run")} if sp else None
        rec = segment_record(net, theta_A, theta_B, sets); rec.update({"protocol": PROTOCOL, "pair": name, "kind": kind, "identity": ident, "spawn": spawn,
                                                                        "endpoints": {"A": {"run": rdA, "epoch": a.epoch, "checkpoint_sha256": hA}, "B": {"run": rdB, "epoch": a.epoch, "checkpoint_sha256": hB}},
                                                                        "weights": "unaligned (no permutation search)", "seconds": time.time() - t0})
        records.dump_json(path, rec)
        an = rec["sets"]["analysis"]; te = rec["sets"]["test"]
        print(f"== {name}: |F_B-F_A|_X {an['endpoint_difference']:.3e}; max deviation/endpoint diff {an['max_interpolation_deviation_rel_endpoint']:.3f} (test {te['max_interpolation_deviation_rel_endpoint']:.3f}); "
              f"MSE barrier {an['barriers']['entezari_mse']['value']:.2e} (test {te['barriers']['entezari_mse']['value']:.2e}); error barrier {an['barriers']['frankle_error']['value']:.4f}; "
              f"loss-connected {rec['loss_connected']}; displacement {rec['displacement_norm']:.3f}, orbit share {(rec['projections']['rescaling_orbit_at_A'] or {}).get('norm_share')}; "
              f"identity residual {an['max_identity_residual']:.1e}, deviation identity {an['max_deviation_identity_residual']:.1e} ({time.time() - t0:.0f}s)", flush=True)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["test"]:
        import pytest; return pytest.main(["-q", os.path.join(runs.HERE, "tests", "test_lmc.py")])
    ap = argparse.ArgumentParser(description="LMC of spawned pairs: spawn-state recovery and segment analysis")
    ap.add_argument("phase", choices=["spawn-state", "segment"])
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="", help="parents' tag ('' = the width-128 ReLU runs)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--stages", nargs="+", default=["early_fitting", "high_accuracy"])
    ap.add_argument("--spawn-epochs", type=int, nargs="*", default=None, help="explicit spawn epochs instead of stages"); ap.add_argument("--parent-dir", default=None, help="explicit parent run directory (tests)")
    ap.add_argument("--policy", choices=["replay", "reset", "recovery"], default="replay", help="Adam state at the spawn: recovered by exact replay (verified), reset for every spawn alike, or read from the parent's recovery file (inherited; 2026-09-26 revision)")
    ap.add_argument("--pairs", nargs="+", default=["siblings", "controls"]); ap.add_argument("--epoch", type=int, default=500, help="endpoint epoch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    {"spawn-state": spawn_state_phase, "segment": segment_phase}[a.phase](a)
