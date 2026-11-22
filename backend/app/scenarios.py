"""Scripted crash scenarios — the sample / cached cases (spec "Backups" + §6.5).

Each scenario simulates ground-truth vehicle motion at 100 Hz, finds the
contact, applies a simple momentum exchange afterwards, crops impact-5 s to
impact+2 s, samples at 10 fps, adds measurement noise that grows with
distance from the dashcam (ΔZ ∝ Z²), and then runs the normal Step-3
reconstruction (Kalman + RTS, 30 Hz resample, rectangle impact test).
"""
from __future__ import annotations

import math

import numpy as np

from . import reconstruct as rc

DT = 0.01
LANE = 3.66
MASS = {"car": 1500.0, "pickup": 2300.0}


def _post_impact(pa, va, pb, vb, ta, tb, n, mu=0.55):
    """Momentum exchange + separation + skid to rest. Returns (n,2) positions for A and B."""
    ma, mb = MASS[ta], MASS[tb]
    vc = (ma * va + mb * vb) / (ma + mb)
    normal = pb - pa
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    va2 = vc + 0.25 * (va - vc) - 1.2 * normal
    vb2 = vc + 0.25 * (vb - vc) + 1.8 * normal
    out_a = np.zeros((n, 2)); out_b = np.zeros((n, 2))
    for i in range(n):
        for v in (va2, vb2):
            sp = np.linalg.norm(v)
            if sp > 0:
                v *= max(0.0, sp - mu * 9.81 * DT) / sp
        pa = pa + va2 * DT; pb = pb + vb2 * DT
        out_a[i], out_b[i] = pa, pb
    return out_a, out_b


def _heading_of(path):
    v = np.gradient(path, DT, axis=0)
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def _first_contact(pa, pb, ta, tb, start=0):
    ha, hb = _heading_of(pa), _heading_of(pb)
    sa, sb = rc.VEHICLE_SIZES[ta], rc.VEHICLE_SIZES[tb]
    for i in range(start, len(pa)):
        if rc.overlap(rc.rect(*pa[i], ha[i], *sa), rc.rect(*pb[i], hb[i], *sb)):
            return i
    return None


def _observe(t, path, cam_path, rng, ego=False):
    """10 fps samples with noise. Ego (camera car) comes from optical flow: small, constant noise."""
    idx = np.round(np.arange(0, t[-1] - t[0] + 1e-9, 0.1) / DT).astype(int)
    idx = idx[idx < len(path)]
    z = path[idx].copy()
    if ego:
        sig = np.full(len(idx), 0.15)
    else:
        f = 298.0  # 1280 px wide, 130° HFOV
        Z = np.linalg.norm(path[idx] - cam_path[idx], axis=1)
        sig = np.clip(0.12 + 0.5 * np.array([rc.depth_sigma(max(zz, 2.0), f) for zz in Z]), 0.12, 3.0)
    z += rng.normal(0, 1, z.shape) * sig[:, None]
    return t[idx] - t[0], z, sig


def _finish(name, ta, tb, pa, pb, t, i_imp, road, signals, signs, meta, seed):
    """Post-impact, crop, observe, reconstruct."""
    n_post = int(2.2 / DT)
    va = (pa[i_imp] - pa[i_imp - 1]) / DT
    vb = (pb[i_imp] - pb[i_imp - 1]) / DT
    post_a, post_b = _post_impact(pa[i_imp].copy(), va.copy(), pb[i_imp].copy(), vb.copy(), ta, tb, n_post)
    pa = np.r_[pa[: i_imp + 1], post_a]; pb = np.r_[pb[: i_imp + 1], post_b]
    t = np.r_[t[: i_imp + 1], t[i_imp] + DT * np.arange(1, n_post + 1)]
    i0 = i_imp - int(5.0 / DT)
    i1 = i_imp + int(2.0 / DT)
    pa, pb, t = pa[i0:i1 + 1], pb[i0:i1 + 1], t[i0:i1 + 1]
    t_imp = 5.0
    rng = np.random.default_rng(seed)
    t10, za, sa = _observe(t, pa, pa, rng, ego=True)
    _, zb, sb = _observe(t, pb, pa, rng)
    A = rc.build_track("A", ta, "#3b82f6", t10, za, sa, t_imp)
    B = rc.build_track("B", tb, "#ef4444", t10, zb, sb, t_imp)
    impact = rc.find_impact(A, B)
    for s in signals:  # shift signal timelines into clip time
        for st in s["states"]:
            st["t"] = round(max(0.0, st["t"] - t[0]), 2)
    meta = {**meta, "scenario": name, "clip_start_abs": round(float(t[0]), 2),
            "true_impact_t": t_imp, "ground_truth": {
                "A_speed_mph": round(float(np.linalg.norm(va)) * rc.MPS_TO_MPH, 1),
                "B_speed_mph": round(float(np.linalg.norm(vb)) * rc.MPS_TO_MPH, 1)}}
    return rc.scene_json("", road, signals, signs, [A, B], impact, meta)


# ------------------------------------------------------------------ left turn
def left_turn(seed: int = 7) -> dict:
    """A (car, camera) northbound at 41 mph in a 35; B (pickup) southbound turns left across A
    on a permissive green with A ~2.6 s away. Spec worked example: B ≈ 91 %, A ≈ 9 %."""
    t = np.arange(0, 20, DT)
    T0 = 10.0                       # B starts its turn
    R = 7.83; cx, cy = 6.0, 6.0     # arc from (-1.83, 6) heading south to heading east at y = -1.83
    vturn = 3.0
    # B speed profile: 11 m/s, decelerating to 3 m/s over the 4 s before T0
    vb = np.where(t < T0 - 4, 11.0, np.where(t < T0, 11.0 - 8.0 * (t - (T0 - 4)) / 4, vturn))
    pb = np.zeros((len(t), 2))
    i0 = int(round(T0 / DT))
    s_back = np.r_[np.cumsum((vb[:i0][::-1]) * DT)[::-1][1:], 0.0] if i0 > 0 else np.zeros(0)
    pb[:i0, 0] = -1.83; pb[:i0, 1] = 6.0 + s_back
    th = math.pi + vturn * (t[i0:] - T0) / R
    th = np.minimum(th, 1.5 * math.pi)
    pb[i0:, 0] = cx + R * np.cos(th); pb[i0:, 1] = cy + R * np.sin(th)
    extra = np.maximum(0, vturn * (t[i0:] - T0) / R - 0.5 * math.pi) * R
    pb[i0:, 0] += extra
    vA = 18.33  # 41 mph

    def a_path(y0):
        return np.c_[np.full(len(t), 1.83), y0 + vA * (t - T0)]

    # Place A so its front meets B's right side 3.1 s after B starts steering (measured gap
    # ≈ 2.8–2.9 s once the turn-rate filter registers the turn). This timing is what the
    # Blender-rendered samples/left_turn.mp4 was made from — re-render if you change it.
    # Coarse-to-fine scan of A's start (contact time is not monotonic in it).
    target = int(round((T0 + 3.1) / DT))

    def err(y0):
        i = _first_contact(a_path(y0)[: target + 120], pb[: target + 120], "car", "pickup", start=target - 200)
        return 1e9 if i is None else abs(i - target)

    best = min(np.arange(-70.0, -30.0, 0.5), key=err)
    best = min(np.arange(best - 0.5, best + 0.5, 0.02), key=err)
    pa = a_path(best)
    i_imp = _first_contact(pa, pb, "car", "pickup", start=target - 200)
    road = {"template": "intersection_signal_4way", "lanes_per_dir": 2, "lane_width_m": LANE,
            "speed_limit_mph": 35}
    half = 2 * LANE
    signals = [
        {"id": "L1", "pos": [half + 1.2, half + 1.5], "faces": "A", "approach_heading": math.pi / 2,
         "stop_line": -half - 1.0, "protected_left": False,
         "states": [{"t": 0.0, "s": "green"}]},
        {"id": "L2", "pos": [-half - 1.2, -half - 1.5], "faces": "B", "approach_heading": -math.pi / 2,
         "stop_line": half + 1.0, "protected_left": False,
         "states": [{"t": 0.0, "s": "green"}]},
    ]
    meta = {"location_type": "intersection_signal", "speed_limit_mph": 35, "surface": "dry",
            "your_vehicle": "A", "description": "I was going straight on green, they turned left in front of me."}
    return _finish("left_turn", "car", "pickup", pa, pb, t, i_imp, road, signals, [], meta, seed)


# ------------------------------------------------------------------ rear-end
def rear_end(seed: int = 11) -> dict:
    """A (car, camera) follows B (car) at 0.9 s; B brakes hard, A reacts after 1.5 s and rear-ends B."""
    t = np.arange(0, 20, DT)
    v0 = 20.0; TB = 10.0
    gap = 0.9 * v0  # bumper to bumper
    def profile(t_brake, decel):
        v = np.where(t < t_brake, v0, np.maximum(0.0, v0 - decel * (t - t_brake)))
        return np.cumsum(v) * DT
    sb = profile(TB, 6.5); sa = profile(TB + 1.5, 7.0)
    pb = np.c_[np.full(len(t), 1.83), sb]
    pa = np.c_[np.full(len(t), 1.83), sa - (gap + 4.5)]
    i_imp = _first_contact(pa, pb, "car", "car")
    road = {"template": "straight_multilane", "lanes_per_dir": 2, "lane_width_m": LANE, "speed_limit_mph": 45}
    meta = {"location_type": "multi_lane_road", "speed_limit_mph": 45, "surface": "dry", "your_vehicle": "A",
            "description": "Traffic stopped suddenly and I couldn't stop in time."}
    # recentre the road on the impact so the replay frames it
    shift = pa[i_imp, 1]
    pa = pa - [0, shift]; pb = pb - [0, shift]
    return _finish("rear_end", "car", "car", pa, pb, t, i_imp, road, [], [], meta, seed)


SCENARIOS = {"left_turn": left_turn, "rear_end": rear_end}
