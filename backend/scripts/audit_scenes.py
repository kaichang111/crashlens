"""Audit every case's scene.json for positional sanity (what the 3D replay will draw).

Checks, per case:
  * off_road   : share of samples where a car's centre is outside the template's paved area
  * wrong_side : share of approach samples (outside the junction) driving against their lane
  * hdg_err    : median heading error vs the road direction on approach (deg)
  * pre_overlap: earliest time the outlines overlap before the impact (should be none)
  * impact     : inside the junction box? outlines touching at impact time (gap in m)?
  * truth_err  : for scripted demos, max position error vs the scenario ground truth before impact (m)

    .venv\\Scripts\\python.exe scripts\\audit_scenes.py [case ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import reconstruct as rc, scenarios, store  # noqa: E402


def rect_gap(a, b):
    if rc.overlap(a, b):
        return 0.0
    return min(min(abs(rc._dist_to_poly(p, b)) for p in a), min(abs(rc._dist_to_poly(p, a)) for p in b))


def audit(cid: str) -> dict:
    sc = store.read_json(cid, "scene.json")
    road = sc["road"]; half = road["lanes_per_dir"] * road["lane_width_m"]
    inter = road["template"].startswith("intersection")
    tr = rc.tracks_from_scene(sc)
    imp = sc.get("impact")
    t_imp = imp["t"] if imp else max(T.t[-1] for T in tr.values())
    out = {"case": cid, "template": road["template"], "cached": bool(store.read_json(cid, "case.json").get("cached"))}
    for vid, T in tr.items():
        pre = T.t <= t_imp
        x, y = T.s[pre, 0], T.s[pre, 1]
        oneway = road["template"] == "straight_oneway"  # lanes at x ∈ [0, half], all heading north
        on_x = ((x >= -0.3) & (x <= half + 0.3)) if oneway else (np.abs(x) <= half + 0.3)
        paved = on_x | ((np.abs(y) <= half + 0.3) if inter else False)
        out[f"{vid}_off_road"] = round(float(1 - paved.mean()), 2) if len(x) else None
        appr = (np.abs(y) > half + 3) if inter else np.ones_like(x, bool)
        appr &= on_x
        moving = T.speed()[pre] > 1.0
        sel = appr & moving
        if sel.sum() >= 5:
            h = T.heading[pre][sel]
            want = np.where(oneway | (x[sel] > 0), math.pi / 2, -math.pi / 2)   # right-hand traffic
            err = np.degrees(np.abs((h - want + math.pi) % (2 * math.pi) - math.pi))
            out[f"{vid}_hdg_err"] = round(float(np.median(err)), 0)
            out[f"{vid}_wrong_side"] = round(float((err > 90).mean()), 2)
    if "A" in tr and "B" in tr:
        A, B = rc._align(tr["A"], tr["B"])
        first = None
        for i in range(len(A.t)):
            if A.t[i] >= t_imp - 0.05:
                break
            ra = rc.rect(*A.s[i, :2], A.heading[i], *A.size); rb = rc.rect(*B.s[i, :2], B.heading[i], *B.size)
            if rc.overlap(ra, rb):
                first = round(float(A.t[i]), 2)
                break
        out["pre_overlap_t"] = first
        if imp and len(A.t):
            i = int(np.argmin(np.abs(A.t - imp["t"])))
            ra = rc.rect(*A.s[i, :2], A.heading[i], *A.size); rb = rc.rect(*B.s[i, :2], B.heading[i], *B.size)
            out["impact_gap_m"] = round(rect_gap(ra, rb), 2)
            out["impact_in_junction"] = (abs(imp["pos"][0]) <= half + 1 and abs(imp["pos"][1]) <= half + 1) if inter else None
            out["impact_inferred"] = bool(imp.get("inferred"))
    name = sc.get("meta", {}).get("scenario")
    if name in scenarios.SCENARIOS:
        truth = rc.tracks_from_scene(scenarios.SCENARIOS[name]())
        errs = []
        for vid in ("A", "B"):
            T, G = tr[vid], truth[vid]
            for ti in np.arange(0.2, t_imp, 0.1):
                errs.append(float(np.linalg.norm(T.s[T.at(ti), :2] - G.s[G.at(ti), :2])))
        out["truth_err_max_m"] = round(max(errs), 2)
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(p.name for p in store.CASES.iterdir() if (p / "scene.json").exists())
    for cid in ids:
        try:
            print(json.dumps(audit(cid)))
        except Exception as e:  # report and keep going
            print(json.dumps({"case": cid, "error": repr(e)}))
