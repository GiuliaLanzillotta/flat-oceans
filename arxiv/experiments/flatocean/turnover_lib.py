"""Aligned endpoint-turnover statistic (protocol Section 4, "Temporal turnover").

For a saved history with per-epoch records e = 0, 1, ..., let f_{l,e} be the
fraction of layer-l gates that disagree between epoch endpoints e-1 and e
(the history's flip_fraction at record e) and d_e = P_e - P_{e-1} the
full-network optimizer path increment over the same epoch.  For a window W
of epoch indices (each e >= 1 with a predecessor record),

    T_l(W) = mean_{e in W} f_{l,e} / d_e          (primary: mean of per-epoch ratios)
    R_l(W) = sum_{e in W} f_{l,e} / sum_{e in W} d_e   (path-weighted ratio of sums)

Intervals with a missing predecessor, a missing flip value, or a nonpositive
path increment are omitted and reported, never filled with a fabricated
value.  Stage windows are W_t = {max(1, t-9), ..., t}; the production report's
windows are {50, ..., 100} and {450, ..., 500} (51 intervals).

This replaces the earlier `turnover_from_history`, which averaged the flip
records of ten epochs but the path increments of nine, divided the means,
and returned nothing at epoch one.
"""
import numpy as np


def turnover(history_epochs, window, layer=None):
    """history_epochs: the list under history["epochs"] (record e at index e).
    window: iterable of epoch indices.  Returns a dict with T (mean of ratios),
    R (ratio of sums), the per-epoch pairs, the valid count, and the omitted
    intervals with reasons; per layer (list) unless `layer` is given."""
    E = {rec["epoch"]: rec for rec in history_epochs}
    L = len(next(iter(E.values()))["gates"])
    layers = [layer] if layer is not None else list(range(L))
    pairs = {l: [] for l in layers}; omitted = []
    for e in window:
        if e not in E or (e - 1) not in E:
            omitted.append((e, "missing record or predecessor")); continue
        d = E[e]["path_length"]["total"] - E[e - 1]["path_length"]["total"]
        if not (d > 0):
            omitted.append((e, "nonpositive path increment")); continue
        fs = [E[e]["gates"][l]["flip_fraction"] for l in layers]
        if any(f is None for f in fs):
            omitted.append((e, "missing flip fraction")); continue
        for l, f in zip(layers, fs):
            pairs[l].append((e, f, d))
    out = {"window": list(window), "valid_count": len(pairs[layers[0]]), "omitted": omitted,
           "T": [], "R": [], "pairs": {}}
    for l in layers:
        P = pairs[l]
        if not P:
            out["T"].append(None); out["R"].append(None); out["pairs"][l] = []; continue
        f = np.array([q[1] for q in P]); d = np.array([q[2] for q in P])
        out["T"].append(float(np.mean(f / d))); out["R"].append(float(f.sum() / d.sum()))
        out["pairs"][l] = [(int(q[0]), float(q[1]), float(q[2])) for q in P]
    if layer is not None:
        out["T"], out["R"] = out["T"][0], out["R"][0]
    return out


def stage_window(t):
    return list(range(max(1, t - 9), t + 1))


# ----------------------------------------------------------------------------- tests (hand-checkable)
def _hist(flips_by_epoch, path_by_epoch):
    """flips_by_epoch[e] = list per layer (None at e=0); path_by_epoch[e] = cumulative path."""
    return [{"epoch": e, "path_length": {"total": path_by_epoch[e]},
             "gates": [{"flip_fraction": (None if e == 0 else flips_by_epoch[e][l])} for l in range(2)]}
            for e in range(len(path_by_epoch))]


def _tests():
    # varying increments: epochs 1..4 with d = 1, 2, 4, 8 and f = 0.1, 0.2, 0.4, 0.8 in layer 0 (ratio 0.1 each)
    # and f = 0.1, 0.1, 0.1, 0.1 in layer 1 (ratios 0.1, 0.05, 0.025, 0.0125)
    path = [0.0, 1.0, 3.0, 7.0, 15.0]
    flips = [None, [0.1, 0.1], [0.2, 0.1], [0.4, 0.1], [0.8, 0.1]]
    h = _hist(flips, path)
    r = turnover(h, [1, 2, 3, 4])
    assert abs(r["T"][0] - 0.1) < 1e-15 and abs(r["R"][0] - 0.1) < 1e-15, r
    assert abs(r["T"][1] - np.mean([0.1, 0.05, 0.025, 0.0125])) < 1e-15, r
    assert abs(r["R"][1] - 0.4 / 15.0) < 1e-15, r
    assert r["valid_count"] == 4 and r["omitted"] == []
    # epoch one alone: one valid interval (predecessor epoch 0)
    r1 = turnover(h, stage_window(1)); assert r1["valid_count"] == 1 and abs(r1["T"][0] - 0.1) < 1e-15, r1
    # initialization: undefined (no predecessor for epoch 0; window empty of valid intervals)
    r0 = turnover(h, [0]); assert r0["valid_count"] == 0 and r0["T"][0] is None and r0["omitted"] == [(0, "missing record or predecessor")], r0
    # short window and missing predecessor: window {3, 4, 5} with no record 5
    r2 = turnover(h, [3, 4, 5]); assert r2["valid_count"] == 2 and r2["omitted"] == [(5, "missing record or predecessor")], r2
    # nonpositive path increment is omitted, not zero-filled
    path2 = [0.0, 1.0, 1.0, 2.0]; h2 = _hist([None, [0.1, 0.1], [0.3, 0.3], [0.1, 0.1]], path2)
    r3 = turnover(h2, [1, 2, 3]); assert r3["valid_count"] == 2 and r3["omitted"] == [(2, "nonpositive path increment")] and abs(r3["T"][0] - 0.1) < 1e-15, r3
    # mean of ratios differs from ratio of sums when increments vary (layer 1 above)
    assert abs(r["T"][1] - r["R"][1]) > 1e-3
    print("turnover_lib: all tests passed")


if __name__ == "__main__":
    _tests()
