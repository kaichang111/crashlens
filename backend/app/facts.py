"""Step 4.1 — turn smoothed trajectories into measured facts with confidences."""
from __future__ import annotations

import math
import re

import numpy as np

from . import reconstruct as rc

MU = {"dry": 0.7, "wet": 0.4}
T_REACT = 1.5
SOURCE_FPS = 10


def _conf_from_sigma(sig: float, base: float = 0.97) -> float:
    """Tighter track → higher confidence. σ 0.15 m → ~0.92, σ 1 m → ~0.67."""
    return round(float(np.clip(base - 0.3 * sig, 0.4, 0.95)), 2)


def _frames(t: float) -> list[int]:
    f = int(round(t * SOURCE_FPS))
    return [max(0, f - 1), f, f + 1]


class _Builder:
    def __init__(self):
        self.facts: list[dict] = []

    def add(self, actor, ftype, value, t, conf, label, unit=None, source="video", extra=None):
        fid = f"F{len(self.facts) + 1}"
        f = {"id": fid, "actor": actor, "type": ftype, "t": None if t is None else round(float(t), 2),
             "value": value, "unit": unit, "confidence": conf, "source": source, "label": label,
             "evidence": {"frames": _frames(t) if t is not None else []}}
        if extra:
            f.update(extra)
        self.facts.append(f)
        return f


def _signal_state(signal: dict, t: float) -> str:
    s = "unknown"
    for st in signal["states"]:
        if st["t"] <= t + 1e-6:
            s = st["s"]
    if s == "green" and signal.get("protected_left"):
        return "green_arrow"
    return s


def _stop_line_cross(tr: rc.Track, approach_heading: float, stop_line: float):
    """First time the vehicle crosses the stop line along its approach axis."""
    ax = np.array([math.cos(approach_heading), math.sin(approach_heading)])
    front = tr.s[:, :2] + np.c_[np.cos(tr.heading), np.sin(tr.heading)] * tr.size[0] / 2
    proj = front @ ax
    line = -abs(stop_line)  # stop line is behind the centre along the approach direction
    idx = np.where((proj[:-1] < line) & (proj[1:] >= line))[0]
    return int(idx[0] + 1) if len(idx) else None


def _left_across_centreline(T: rc.Track, t_imp: float, half: float, min_lat: float = 1.5):
    """Did this vehicle turn left across the oncoming lanes, judged by position rather than heading?

    The scene puts the intersection centre at the origin with the N–S road's centre line at x = 0 and
    right-hand traffic (A northbound at x > 0). A vehicle approaching on the N–S road is, by its lane,
    southbound if it starts at x < 0 and northbound if x > 0. A left turn = moving at least min_lat
    metres toward its own left and ending on the far side of the centre line, inside the junction.
    Returns (turn_start_t, lateral_m, approach_heading) or None.
    """
    pre = T.t <= t_imp - 0.1
    if pre.sum() < 10:
        return None
    p = T.s[pre, :2]; t = T.t[pre]
    x0, y0 = p[0]
    if abs(x0) > half + 1.0 or abs(y0) <= abs(x0):   # only vehicles first seen on the N–S road
        return None
    h_app = -math.pi / 2 if x0 < 0 else math.pi / 2  # its lane says which way it was going
    left = np.array([math.cos(h_app + math.pi / 2), math.sin(h_app + math.pi / 2)])
    lat = np.convolve(np.pad(p @ left, 7, mode="edge"), np.ones(15) / 15, mode="valid")   # 0.5 s smoothing
    end_in_box = abs(p[-1, 0]) < half + 1.0 and abs(p[-1, 1]) < half + 3.0
    crossed = (p[0] @ left) < 0 < (p[-1] @ left)      # from its own side of the centre line to the other
    recent = t >= t[-1] - 2.5
    i_min = int(np.flatnonzero(recent)[np.argmin(lat[recent])])
    gain = float(lat[-1] - lat[i_min])
    if not (end_in_box and crossed and gain >= min_lat):
        return None
    k = i_min + int(np.argmax(lat[i_min:] >= lat[i_min] + 0.3))   # where the leftward drift began
    return float(t[k]), gain, h_app


def extract(scene: dict, form: dict | None = None) -> list[dict]:
    form = form or scene.get("meta", {})
    tr = rc.tracks_from_scene(scene)
    imp = scene.get("impact")
    road = scene["road"]
    limit = form.get("speed_limit_mph") or road.get("speed_limit_mph")
    mu = MU.get(form.get("surface", "dry"), 0.7)
    fb = _Builder()
    if not imp:
        return fb.facts
    t_imp = imp["t"]
    ids = ("A", "B")

    # --- impact type and contact points (Step 3.6)
    fb.add("both", "impact_type", imp["type"], t_imp, 0.9,
           f"Impact type: {imp['type'].replace('_', ' ')} at {imp['angle_deg']:.0f}° between headings, "
           f"closing speed {imp['closing_speed'] * rc.MPS_TO_MPH:.0f} mph", extra={"closing_speed_mph": round(imp['closing_speed'] * rc.MPS_TO_MPH, 1)})
    for v in ids:
        fb.add(v, "contact_point", imp["contact"][v], t_imp, 0.85,
               f"{v} was struck on its {imp['contact'][v].replace('_', ' ')}")

    # --- speeds (1 s before impact, so the crash itself doesn't pull the number down)
    for v in ids:
        T = tr[v]
        i = T.at(t_imp - 1.0)
        sp = float(T.speed()[i]) * rc.MPS_TO_MPH
        conf = _conf_from_sigma(float(T.sigma[i]))
        fb.add(v, "speed_mph", round(sp, 1), T.t[i], conf, f"{v} was travelling {sp:.0f} mph 1 s before impact", "mph")
        if limit:
            # typical speed over the approach (median of impact −3 s … −0.5 s), not a single peak:
            # the last half-second before a crash is the least reliable part of any dashcam track
            w = (T.t >= t_imp - 3.0) & (T.t <= t_imp - 0.5)
            if w.any():
                vtyp = float(np.median(T.speed()[w])) * rc.MPS_TO_MPH
                over = round(vtyp - limit, 1)
                if over >= 1.0:
                    fb.add(v, "speed_over_limit_mph", over, float(t_imp - 1.75), conf,
                           f"{v} was travelling about {vtyp:.0f} mph in a {limit:g} mph zone ({over:.0f} over) on the approach",
                           "mph")

    # --- braking onset: first deceleration over 3 m/s² in the 3 s before impact
    brake_t = {}
    for v in ids:
        T = tr[v]
        acc = np.gradient(T.speed(), T.t)
        # stop 0.2 s short of impact: the smoother's split at the crash is not braking
        w = np.where((T.t >= t_imp - 3.0) & (T.t <= t_imp - 0.2) & (acc < -3.0))[0]
        brake_t[v] = float(T.t[w[0]]) if len(w) else None
        if brake_t[v] is not None:
            fb.add(v, "braking_onset_s", round(t_imp - brake_t[v], 2), brake_t[v], _conf_from_sigma(float(T.sigma[w[0]])),
                   f"{v} began braking firmly {t_imp - brake_t[v]:.1f} s before impact", "s before impact")
        else:
            fb.add(v, "braked_before_impact", False, t_imp - 0.5, 0.75, f"{v} shows no firm braking before impact")

    tmpl = road["template"]
    # --- intersections: signals, stop line crossings, left turns
    if tmpl.startswith("intersection"):
        half = road["lanes_per_dir"] * road["lane_width_m"]
        for s in scene.get("signals", []):
            v = s["faces"]
            T = tr[v]
            i = _stop_line_cross(T, s["approach_heading"], s["stop_line"])
            if i is None:
                continue
            st = _signal_state(s, float(T.t[i]))
            fb.add(v, "signal_state", st, T.t[i], 0.86,
                   f"{v}'s light was {st.replace('_', ' ')} when {v} crossed the stop line"
                   + (" (circular green, not a protected arrow)" if st == "green" else ""))
        for s in scene.get("signs", []):
            v = s["faces"]
            T = tr[v]
            i = _stop_line_cross(T, s["approach_heading"], s["stop_line"])
            if i is None:
                continue
            fb.add(v, "faced_stop_sign", True, T.t[i], 0.9, f"{v} approached a stop sign")
            w = (T.t >= T.t[i] - 3.0) & (T.t <= T.t[i])
            slow = T.speed()[w] < 0.5
            run = max((len(x) for x in "".join("1" if b else "0" for b in slow).split("0")), default=0)
            stopped = run / 30.0 >= 0.5
            fb.add(v, "stopped_at_stop_sign", stopped, T.t[i], _conf_from_sigma(float(T.sigma[i])),
                   f"{v} {'came to a full stop' if stopped else 'did not come to a full stop'} before the line")
        for v in ids:
            o = "B" if v == "A" else "A"
            T = tr[v]
            inside = (np.abs(T.s[:, 0]) < half + 2) & (np.abs(T.s[:, 1]) < half + 2) & (T.t <= t_imp - 0.1)
            if not inside.any():
                continue
            h0 = T.heading[np.argmax(inside)]
            dh = T.heading[inside] - h0
            left = bool(np.max(dh) > math.radians(45)) or (
                np.max(dh) > math.radians(30) and imp["type"] == "angle")
            crossing = None if left else _left_across_centreline(T, t_imp, half)
            if crossing:
                # Slow turner hit early in its turn: heading barely changed (and is noisy near standstill),
                # but it moved leftward across the centre line into the oncoming lanes inside the junction.
                t_turn, lat_m, h_app = crossing
                fb.add(v, "turning_left", True, t_turn, 0.7,
                       f"{v} moved {lat_m:.1f} m to its left across the centre line into {o}'s lanes inside the "
                       f"intersection (left turn in progress)", extra={"method": "lane_crossing"})
                ttc = t_imp - t_turn
                fb.add(v, "oncoming_ttc_s", round(ttc, 2), t_turn,
                       _conf_from_sigma(float(tr[o].sigma[tr[o].at(t_turn)]), 0.99),
                       f"When {v} began turning, {o} was {ttc:.1f} s from the collision point", "s")
                O = tr[o]
                j = O.at(t_turn)
                vo = float(O.speed()[j])
                gap_m = float(np.sum(O.speed()[j:O.at(t_imp)]) / 30.0)
                d_stop = rc.stopping_distance(vo, mu, T_REACT)
                fb.add(o, "could_avoid", bool(d_stop < gap_m), t_turn, 0.75,
                       f"{o} needed {d_stop:.0f} m to stop (1.5 s reaction, μ={mu}) and had {gap_m:.0f} m",
                       extra={"stopping_distance_m": round(d_stop, 1), "gap_m": round(gap_m, 1)})
                continue
            if left:
                # turn start: walk back from the impact while the vehicle is still turning left
                # (> 8°/s). Robust to a noisy early heading, unlike "10° off the first sample".
                rate = np.gradient(T.heading, T.t)
                # 0.7 s smoothing + 10°/s: tuned so the live (YOLO) and ground-truth tracks of the
                # rendered left-turn clip agree within 0.2 s; slow turners have noisy headings
                rate = np.convolve(np.pad(rate, 10, mode="edge"), np.ones(21) / 21, mode="valid")
                thr = math.radians(10)
                # start at the last clearly-turning sample ≥0.2 s before impact (the smoother's
                # split at the crash distorts the final samples), then walk back to where it began
                cand = np.where((T.t <= t_imp - 0.2) & (T.t >= t_imp - 1.5) & (rate > thr))[0]
                k = int(cand[-1]) if len(cand) else int(np.searchsorted(T.t, t_imp - 0.2))
                # walk back through the turn, tolerating brief dips (≤0.3 s) in the noisy turn rate
                gap_tol = int(round(0.3 * 30))
                j = k
                while j > 0:
                    back = rate[max(0, j - 1 - gap_tol):j]
                    hits = np.where(back > thr)[0]
                    if not len(hits):
                        break
                    j = max(0, j - 1 - gap_tol) + int(hits[-1])
                    if rate[j] > thr:
                        k = j
                t_turn = float(T.t[k])
                fb.add(v, "turning_left", True, t_turn, 0.9,
                       f"{v} turned left inside the intersection (heading changed {math.degrees(np.max(dh)):.0f}°)")
                ttc = t_imp - t_turn
                fb.add(v, "oncoming_ttc_s", round(ttc, 2), t_turn, _conf_from_sigma(float(tr[o].sigma[tr[o].at(t_turn)]), 0.99),
                       f"When {v} began turning, {o} was {ttc:.1f} s from the collision point", "s")
                # could the oncoming driver have stopped?
                O = tr[o]
                j = O.at(t_turn)
                vo = float(O.speed()[j])
                gap_m = float(np.sum(O.speed()[j:O.at(t_imp)]) / 30.0)
                d_stop = rc.stopping_distance(vo, mu, T_REACT)
                fb.add(o, "could_avoid", bool(d_stop < gap_m), t_turn, 0.75,
                       f"{o} needed {d_stop:.0f} m to stop (1.5 s reaction, μ={mu}) and had {gap_m:.0f} m",
                       extra={"stopping_distance_m": round(d_stop, 1), "gap_m": round(gap_m, 1)})

    # --- rear-end: follower, time gap, could they stop
    if imp["type"] == "rear_end":
        fronts = ("front", "front_left", "front_right")
        follower = "A" if imp["contact"]["A"] in fronts else "B"
        lead = "B" if follower == "A" else "A"
        F, L = tr[follower], tr[lead]
        i = F.at(t_imp - 3.0)
        gap = float(np.linalg.norm(L.s[i, :2] - F.s[i, :2])) - (F.size[0] + L.size[0]) / 2
        v = float(F.speed()[i])
        fb.add(follower, "is_follower", True, t_imp, 0.9, f"{follower} was following {lead} in the same lane")
        fb.add(follower, "following_time_gap_s", round(gap / max(v, 0.1), 2), F.t[i],
               _conf_from_sigma(float(L.sigma[i]), 0.99),
               f"3 s before impact {follower} was {gap:.0f} m behind {lead}: a {gap / max(v, 0.1):.1f} s gap", "s")
        t_h = brake_t.get(lead) or (t_imp - 2.5)
        j = F.at(t_h)
        vf = float(F.speed()[j])
        gap_h = float(np.linalg.norm(L.s[j, :2] - F.s[j, :2])) - (F.size[0] + L.size[0]) / 2
        d_lead = float(np.sum(L.speed()[j:]) / 30.0)  # lead keeps moving while braking
        d_stop = rc.stopping_distance(vf, mu, T_REACT)
        fb.add(follower, "could_avoid", bool(d_stop < gap_h + d_lead), t_h, 0.75,
               f"When {lead} braked, {follower} needed {d_stop:.0f} m to stop and had {gap_h + d_lead:.0f} m",
               extra={"stopping_distance_m": round(d_stop, 1), "gap_m": round(gap_h + d_lead, 1)})

    # --- lane change (straight roads)
    if tmpl.startswith("straight"):
        for v in ids:
            T = tr[v]
            w = (T.t >= t_imp - 3.0) & (T.t <= t_imp)
            # lateral axis = perpendicular to the road (road runs along y)
            lat = T.s[w, 0]
            if lat.max() - lat.min() > 0.6 * road["lane_width_m"]:
                o = "B" if v == "A" else "A"
                k = np.where(w)[0][0]
                g = float(np.linalg.norm(tr[o].s[k, :2] - T.s[k, :2]))
                tg = g / max(float(tr[o].speed()[k]), 0.1)
                fb.add(v, "lane_change", True, T.t[k], 0.85, f"{v} moved {lat.max() - lat.min():.1f} m sideways across a lane line")
                fb.add(v, "lane_change_gap_s", round(tg, 2), T.t[k], 0.7, f"{v} moved into a {tg:.1f} s gap", "s")
    return fb.facts


# ------------------------------------------------------------ user claims
_COLORS = ("green arrow", "green", "yellow", "red")


def claims_from_description(text: str, your: str = "A") -> list[dict]:
    """Keyword fallback for §4.1 claim extraction (the LLM path lives in narrative.py)."""
    t = (text or "").lower()
    out = []
    other = "B" if your == "A" else "A"
    m = re.search(r"\b(?:i|my light)\b[^.]*?\b(green arrow|green|yellow|red)\b", t)
    if m:
        out.append({"actor": your, "type": "signal_state", "value": m.group(1).replace(" ", "_")})
    if re.search(r"\b(they|other (?:car|driver))\b[^.]*\bturned left\b", t):
        out.append({"actor": other, "type": "turning_left", "value": True})
    if re.search(r"\b(stopped suddenly|slammed|braked hard)\b", t):
        out.append({"actor": other, "type": "braking_onset_s", "value": "claimed"})
    return out


def reconcile_claims(facts: list[dict], claims: list[dict]) -> list[dict]:
    """Agreeing claims add a little confidence; contradictions are flagged, never used silently."""
    idx = {(f["type"], f["actor"]): f for f in facts}
    notes = []
    for c in claims:
        f = idx.get((c["type"], c["actor"]))
        if f is None:
            notes.append({**c, "status": "unverified"})
        elif c["value"] == "claimed" or f["value"] == c["value"]:
            f["confidence"] = round(min(0.97, f["confidence"] + 0.03), 2)
            f.setdefault("corroborated_by", []).append("user_claim")
            notes.append({**c, "status": "agrees", "fact_id": f["id"]})
        else:
            notes.append({**c, "status": "conflict", "fact_id": f["id"], "video_value": f["value"]})
    return notes
