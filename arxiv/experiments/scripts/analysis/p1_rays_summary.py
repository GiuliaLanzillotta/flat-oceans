"""P1 rays summary (plan 2026-10-02, revised after Giulia review 2026-10-02).

Combines the canonical <name>_rays.json (absolute grid 1e-3..100) with the extended <name>_ext_rays.json (6 sub-1e-3
radii) and, where present, <name>_atomic_rays.json (single-layer + generic_w1frozen) and <name>_exposure_rays.json
(held-out rows).  It reports TWO SEPARATE radii per set (NEVER conflated), plus the actual output-drift magnitude so
that "no gate crossing" is NOT mistaken for "no output change":

  r_gate               first gate-crossing radius: smallest absolute radius with any flip (sum of [W1,W2,W3] > 0).
  r_vis(tau)           output-visibility radius: smallest absolute radius where the ACTUAL output drift
                       (|F(theta0+r w) - F0|_X, the 'true' field) exceeds tau * base_output_norm_X (base ~ 1).
  drift magnitudes     median 'true' (actual) AND 'frozen' (frozen-gate) drift at the smallest probed radius and at
                       r=1e-3, so the frozen response is visible (on held-out rows it is NONZERO, ~5e-8, even where no
                       gate has crossed and true == frozen).

Each radius is reported as a GRID BRACKET [last-below, first-above] (median over the 128 matched directions) and
CENSORED (first-above = None, '> grid max') when the threshold is not reached within the grid.  Reads only; writes
results/analysis/p1_rays_summary.json.

Guardrails (Giulia 2026-10-02): "no crossing"/"true==frozen" does NOT establish zero output change; the frozen-flatness
is SAMPLE-SPECIFIC and does NOT transfer to held-out rows; keep fitted and (fixed-checkpoint) exposure separate.
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BR = os.path.join(HERE, "results", "branches")
TAUS = (1e-4, 1e-3)

CASES = []
for cond, tagpat in (("retrain", "mnist_mlp_p1_sub{N}_seed{s}"), ("subsample", "mnist_mlp_sub{N}_seed{s}")):
    for s in (0, 1, 2):
        for N in (1000, 2000, 5000):
            CASES.append((cond, N, s, tagpat.format(N=N, s=s)))


def load(name, suffix=""):
    p = os.path.join(BR, f"{name}{suffix}_rays.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]
    return r.get("late_endpoint")


def per_dir(setrec_list):
    """{direction_seed: sorted [(radius, record)]} merged over several records of ONE set."""
    merged = {}
    for sr in setrec_list:
        if not sr:
            continue
        for rstr, lst in sr.get("absolute", {}).items():
            for q in lst:
                merged.setdefault(q["direction_seed"], {})[float(rstr)] = q
    return {ds: sorted(byr.items()) for ds, byr in merged.items()}


def _bracket(pairs, predicate):
    """pairs = sorted [(radius, rec)]; first radius satisfying predicate and the grid radius just below it.
    Returns (lower_or_None, first_or_None); first=None means censored (never satisfied in the grid)."""
    lo = None
    for r, rec in pairs:
        if predicate(rec):
            return lo, r
        lo = r
    return lo, None


def _agg(firsts, los, n):
    # CENSORED-INCLUSIVE median (fix 2026-10-03): directions that never cross within the grid are first=None; they are
    # NOT dropped -- they enter the median as +inf (i.e. > grid-max), so the median is over ALL n directions and is not
    # biased toward the crossing ones.  first_above_median is None ("> grid max") when the median direction is censored.
    fin = [f for f in firsts if f is not None]
    withinf = [f if f is not None else float("inf") for f in firsts]
    med = float(np.median(withinf)) if withinf else None
    return {"first_above_median": (med if (med is not None and np.isfinite(med)) else None),
            "first_above_median_is_censored": bool(med is not None and not np.isfinite(med)),
            "first_above_min": float(min(fin)) if fin else None,       # over crossing directions only
            "first_above_max": float(max(fin)) if fin else None,
            "lower_median": float(np.median([l for l in los if l is not None])) if any(l is not None for l in los) else None,
            "n_crossing": len(fin), "n_total": int(n),
            "censored_frac": float((n - len(fin)) / max(n, 1))}


def analyze(setrec_list):
    pd = per_dir(setrec_list)
    n = len(pd)
    if n == 0:
        return None
    # r_gate
    g = [_bracket(p, lambda rec: sum(rec["flips"]) > 0) for p in pd.values()]
    out = {"n_dir": n, "r_gate": _agg([x[1] for x in g], [x[0] for x in g], n)}
    # r_vis(tau) on the ACTUAL output drift
    for tau in TAUS:
        v = [_bracket(p, lambda rec: rec["true"] > tau * rec.get("base_output_norm_X", 1.0)) for p in pd.values()]
        out[f"r_vis_{tau:g}"] = _agg([x[1] for x in v], [x[0] for x in v], n)
    # drift magnitudes
    rmin = min(min(r for r, _ in p) for p in pd.values())
    at = lambda r, k: [byr[r] for p in pd.values() for (rr, byr) in p if rr == r]
    def medkey(r, k):
        vals = [rec[k] for p in pd.values() for (rr, rec) in p if rr == r]
        return float(np.median(vals)) if vals else None
    out["smallest_probed_radius"] = rmin
    out["median_true_drift_at_smallest"] = medkey(rmin, "true")
    out["median_frozen_drift_at_smallest"] = medkey(rmin, "frozen")
    out["median_true_drift_at_1e-3"] = medkey(0.001, "true")
    out["median_frozen_drift_at_1e-3"] = medkey(0.001, "frozen")
    out["frac_no_gate_crossing_at_smallest"] = float(np.mean([sum(next(rec for rr, rec in p if rr == rmin)["flips"]) == 0 for p in pd.values()]))
    out["median_flips_at_1e-3"] = medflips(pd, 0.001)
    out["median_flips_at_1.0"] = medflips(pd, 1.0)
    # per-radius CROSSING FRACTION (fix 2026-10-03): the median flip count hides real crossings (e.g. a few/128 probes
    # cross while the median is 0); report the fraction of probes with ANY gate flip at each radius, and the count.
    radii = sorted({r for p in pd.values() for (r, _) in p})
    cross = {}; ncross = {}
    for r in radii:
        vals = [1 if sum(rec["flips"]) > 0 else 0 for p in pd.values() for (rr, rec) in p if rr == r]
        if vals:
            cross[f"{r:g}"] = float(sum(vals) / len(vals)); ncross[f"{r:g}"] = f"{sum(vals)}/{len(vals)}"
    out["crossing_fraction_by_radius"] = cross
    out["n_crossing_by_radius"] = ncross
    return out


def medflips(pd, r):
    v = [rec["flips"] for p in pd.values() for (rr, rec) in p if rr == r]
    return np.median(v, axis=0).tolist() if v else None


def summarize_sets(reclist_by_name):
    """reclist_by_name: {display: [records...]} e.g. fitted = [ext, canonical]."""
    names = set()
    for recs in reclist_by_name.values():
        for r in recs:
            if r:
                names |= set(r.get("sets", {}))
    out = {}
    for sn in sorted(names):
        srs = []
        for recs in reclist_by_name.values():
            for r in recs:
                if r and sn in r.get("sets", {}) and "absolute" in r["sets"][sn]:
                    srs.append(r["sets"][sn])
        a = analyze(srs)
        if a:
            a["dim"] = next((s.get("dim") for s in srs if s.get("dim") is not None), None)
            out[sn] = a
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "results", "analysis", "p1_rays_summary.json"))
    a = ap.parse_args()
    summary = {"meta": {"taus": list(TAUS), "radii_reported": "grid brackets [last-below, first-above]; first_above_median=None is CENSORED (> grid max)",
                        "note": "r_gate (first flip) and r_vis(tau) (actual-output-drift threshold) are DISTINCT INDEPENDENT measured brackets, no assumed ordering; frozen/true drift magnitudes reported so 'no gate crossing' is not read as 'no output change'; frozen-flatness is sample-specific (fitted), NOT claimed to transfer to held-out rows.",
                        "median_is_censored_inclusive": "first_above_median is over ALL n directions with censored (never-crossing) directions counted as > grid-max; it is None when the MEDIAN direction never crosses. censored_frac and n_crossing/n_total are reported.",
                        "crossing_fractions": "crossing_fraction_by_radius / n_crossing_by_radius give the FRACTION of probes with any gate flip at each radius (the median flip count can be 0 while a nonzero fraction crosses).",
                        "single_layer_source": "retrain-N1000 single-layer bases are the FULL standalone atomic kernels (from naive_atomic_sum); every other cell's single-layer bases are the per-block components of the bottom_up branch (NOT the standalone atomic kernel), labelled per entry."}}
    for cond, N, s, name in CASES:
        canon = load(name); ext = load(name, "_ext"); atom = load(name, "_atomic"); expo = load(name, "_exposure")
        if not any((canon, ext, atom, expo)):
            continue
        entry = {"name": name, "available": {"canonical": canon is not None, "ext": ext is not None,
                                             "atomic": atom is not None, "exposure": expo is not None}}
        if canon or ext:
            entry["fitted_sets"] = summarize_sets({"fitted": [ext, canon]})
        if atom:
            entry["single_layer_sets"] = summarize_sets({"atomic": [atom]})
            entry["atomic_kernel_dims"] = atom.get("atomic_kernel_dims"); entry["bottom_up_dims"] = atom.get("bottom_up_dims")
            src = atom.get("atomic_construction")
            entry["single_layer_source"] = src
            entry["single_layer_source_note"] = ("FULL standalone atomic kernel (direct sum, no compatibility filtering)"
                                                 if src == "naive_atomic_sum" else
                                                 "per-block component of the bottom_up branch (W2 = full atomic since it is first; W3 = compatibility-reduced subset, NOT the standalone atomic kernel)")
        if expo:
            entry["exposure_sets"] = summarize_sets({"exposure": [expo]})
            entry["exposure_index_sha256"] = expo.get("exposure_index_sha256")
        summary[f"{cond}_N{N}_seed{s}"] = entry
    json.dump(summary, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}\n")
    for k, e in summary.items():
        if k == "meta":
            continue
        print(f"==== {k}  available={[kk for kk,v in e['available'].items() if v]}")
        for group in ("fitted_sets", "single_layer_sets", "exposure_sets"):
            if group not in e:
                continue
            print(f"  -- {group}:")
            if group == "single_layer_sets":
                print(f"     [single-layer source: {e.get('single_layer_source')} -- {e.get('single_layer_source_note')}]")
            for sn, s in e[group].items():
                g = s["r_gate"]; v4 = s["r_vis_0.0001"]; v3 = s["r_vis_0.001"]
                ncr = s.get("n_crossing_by_radius", {})
                print(f"     {sn:20s} dim={str(s['dim']):>6s}  r_gate~{_b(g)}  r_vis(1e-4)~{_b(v4)}  r_vis(1e-3)~{_b(v3)}")
                print(f"        {'':20s} @r={s['smallest_probed_radius']:g}: true={_e(s['median_true_drift_at_smallest'])} frozen={_e(s['median_frozen_drift_at_smallest'])}; "
                      f"@1e-3: true={_e(s['median_true_drift_at_1e-3'])} frozen={_e(s['median_frozen_drift_at_1e-3'])} flips={s['median_flips_at_1e-3']} cross@1e-3={ncr.get('0.001','-')} cross@1={ncr.get('1','-')}")


def _e(x):
    return "na" if x is None else f"{x:.2e}"


def _b(a):
    lo = a["lower_median"]; hi = a["first_above_median"]
    los = "na" if lo is None else f"{lo:g}"
    his = ">max" if (hi is None) else f"{hi:g}"          # None => the MEDIAN direction is censored (>grid max)
    cens = a["censored_frac"]
    return f"[{los},{his}]" + (f"(cens{cens*100:.0f}%,{a.get('n_crossing')}/{a.get('n_total')}cross)" if cens > 0 else "")


if __name__ == "__main__":
    main()
