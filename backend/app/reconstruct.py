"""Step 3 — geometry, smoothing and impact detection (numpy only).

Inputs are world-frame positions in meters at the source frame rate (10 fps),
with a per-sample measurement sigma. Outputs are smoothed 30 Hz tracks, the
impact, and the scene.json structure from spec §3.7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
MPS_TO_MPH = 2.237
VEHICLE_SIZES = {"car": (4.5, 1.8), "pickup": (5.8, 2.0), "suv": (5.0, 1.9), "bus": (12.0, 2.6),
                 "truck": (12.0, 2.6), "motorcycle": (2.2, 0.8)}
HALF_LENGTH_PUSH = {"car": 2.25, "pickup": 3.0, "suv": 3.0, "bus": 6.0, "truck": 6.0, "motorcycle": 1.1}


# ------------------------------------------------------------ homography
def homography(img_pts: np.ndarray, road_pts: np.ndarray) -> np.ndarray:
    """4-point DLT, identical to cv2.getPerspectiveTransform."""
    A = []
    for (u, v), (x, y) in zip(img_pts, road_pts):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def calibration_check(img_pts: np.ndarray, lane_w: float, length_m: float, W: int, hfov_deg: float,
                      h_range: tuple[float, float] = (0.8, 2.2)) -> dict:
    """Cross-check the 4 clicks against a flat road seen by a level camera.

    Width and length each imply a camera height:
      * lateral:      px per metre at row v = (v − v_h)/h          → h_lat  (no focal length needed)
      * longitudinal: Z = f·h/(v − v_h), so far − near = length    → h_long (needs f, from the HFOV)
    A dashcam sits ~1.2–1.5 m up. If h_long is implausible but h_lat isn't, the rectangle isn't really
    one dash + gap long (common when there are no dashes near the car); return the length that makes
    the two agree, so the caller can rescale instead of stretching every distance and speed.
    """
    p = np.asarray(img_pts, float)  # near-left, near-right, far-right, far-left
    def line(a, b):
        return np.cross(np.r_[a, 1.0], np.r_[b, 1.0])
    vp = np.cross(line(p[0], p[3]), line(p[1], p[2]))
    if abs(vp[2]) < 1e-9:
        return {"ok": False, "reason": "lane edges are parallel in the image"}
    vp = vp[:2] / vp[2]
    v_h = float(vp[1])
    v_near = (p[0, 1] + p[1, 1]) / 2; v_far = (p[2, 1] + p[3, 1]) / 2
    w_near = abs(p[1, 0] - p[0, 0]); w_far = abs(p[2, 0] - p[3, 0])
    if v_near - v_h < 2 or v_far - v_h < 1 or w_near < 2:
        return {"ok": False, "reason": "points are at or above the horizon"}
    h_lat = float(np.mean([lane_w * (v_near - v_h) / w_near, lane_w * (v_far - v_h) / max(w_far, 1e-6)]))
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    inv = 1 / (v_far - v_h) - 1 / (v_near - v_h)
    h_long = float(length_m / (f * inv)) if inv > 0 else float("inf")
    lat_ok = h_range[0] <= h_lat <= h_range[1]
    ratio = h_long / h_lat if h_lat > 0 else float("inf")
    # Focal length the clicks themselves imply (only meaningful if the rectangle really is one dash cycle)
    f_clicks = length_m / (h_lat * inv) if inv > 0 and h_lat > 0 else float("inf")
    hfov_clicks = math.degrees(2 * math.atan((W / 2) / f_clicks)) if math.isfinite(f_clicks) else 0.0
    out = {"ok": True, "horizon_row": round(v_h, 1), "h_lat_m": round(h_lat, 2), "h_long_m": round(h_long, 2),
           "ratio": round(ratio, 2), "hfov_from_clicks": round(hfov_clicks, 1), "hfov_assumed": hfov_deg,
           "consistent": 1 / 1.4 <= ratio <= 1.4}
    if out["consistent"] or not lat_ok:
        out["f_px"] = round(float(f), 1)
        return out
    # The length claim is off. A focal length is still needed to turn rows into distances: use the
    # assumed HFOV unless it puts the first visible road row (just above the hood) implausibly close.
    # Windscreen dashcams typically first see road 2.5–6 m ahead; if outside that, assume 3.5 m.
    near_m = f * h_lat / (v_near - v_h)
    if not 2.5 <= near_m <= 6.0:
        f = 3.5 * (v_near - v_h) / h_lat
        out["hfov_estimated_from_hood"] = round(math.degrees(2 * math.atan((W / 2) / f)), 1)
    out["near_edge_m"] = round(float(f * h_lat / (v_near - v_h)), 2)
    out["f_px"] = round(float(f), 1)
    out["suggested_length_m"] = round(float(f * h_lat * inv), 2)
    return out


def bev_extent(H: np.ndarray, W: int, Himg: int, v_h: float, hood_row: int, far_frac: float = 0.05,
               x_mode: str = "wide") -> dict:
    """Bird's-eye window (road metres) that starts at the first road actually visible above the hood
    (a fixed window can start behind the camera when the clicks sit near the hood) and runs out to
    ~95 % of the way to the horizon, at most 65 m: the lane-dash correlation needs a long strip."""
    near_v = hood_row - 2
    far_v = v_h + far_frac * (near_v - v_h)
    pts = project(H, np.array([[W / 2, near_v], [W / 2, far_v], [0, near_v], [W, near_v]], float))
    y0 = float(pts[0, 1])
    y1 = min(float(pts[1, 1]), y0 + 65.0)
    if not (np.isfinite(y0) and np.isfinite(y1)) or y1 - y0 < 5.0:
        return {"x": (-15.0, 20.0), "y": (-5.0, 60.0), "ppm": 10.0}
    x = (-15.0, 20.0)
    if x_mode == "image":   # only the road width visible across the bottom of the frame
        x = (max(float(min(pts[2, 0], pts[3, 0])), -15.0), min(float(max(pts[2, 0], pts[3, 0])), 20.0))
    ppm = float(np.clip(650.0 / (y1 - y0), 4.0, 40.0))
    return {"x": x, "y": (y0, y1), "ppm": ppm}


def pinhole_ground(u, v, W, hfov_deg, v_h, cam_h=1.3, cx=None):
    """Fallback when there are no lane lines (spec §3.1)."""
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    cx = W / 2 if cx is None else cx
    Z = f * cam_h / max(v - v_h, 1e-3)
    X = (u - cx) * Z / f
    return X, Z, f


def depth_sigma(Z: float, f: float, cam_h: float = 1.3, px: float = 1.0) -> float:
    """ΔZ ≈ Z² / (f h) per pixel of error — error grows with distance squared."""
    return px * Z * Z / (f * cam_h)


# ------------------------------------------------------------ Kalman + RTS
def _FQ(dt: float, q: float):
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Q = q * np.array([[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2],
                      [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]])
    return F, Q


def kalman_rts(z: np.ndarray, t: np.ndarray | float, meas_sigma: np.ndarray, accel_sigma: float = 3.0):
    """Constant-velocity Kalman filter + Rauch–Tung–Striebel smoother.

    z: (N,2) positions; t: (N,) sample times (uneven spacing is fine) or a constant dt;
    meas_sigma: (N,) per-sample std in meters (∝ Z²).
    Returns smoothed states (N,4) [x,y,vx,vy] and position std (N,).
    """
    n = len(z)
    t = np.arange(n) * float(t) if np.isscalar(t) else np.asarray(t, float)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    q = accel_sigma ** 2
    x = np.zeros((n, 4)); P = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4)); Fs = [np.eye(4)] * n
    k3 = min(3, n - 1)
    v0 = (z[k3] - z[0]) / max(t[k3] - t[0], 1e-3)
    xk = np.r_[z[0], v0]; Pk = np.diag([1, 1, 25, 25.0])
    for k in range(n):
        if k > 0:
            F, Q = _FQ(max(t[k] - t[k - 1], 1e-3), q)
            Fs[k] = F
            xk = F @ xk; Pk = F @ Pk @ F.T + Q
        xp[k], Pp[k] = xk, Pk
        R = np.eye(2) * meas_sigma[k] ** 2
        S = Hm @ Pk @ Hm.T + R
        K = Pk @ Hm.T @ np.linalg.inv(S)
        xk = xk + K @ (z[k] - Hm @ xk)
        Pk = (np.eye(4) - K @ Hm) @ Pk
        x[k], P[k] = xk, Pk
    xs = x.copy(); Ps = P.copy()
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        C = P[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = x[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = P[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    sig = np.sqrt(np.maximum(Ps[:, 0, 0], Ps[:, 1, 1]))
    return xs, sig


def smooth_track(t: np.ndarray, z: np.ndarray, sigma: np.ndarray, split_t: float | None = None):
    """Smooth, optionally in two segments split at the impact (a crash is not constant-velocity)."""
    if split_t is None:
        return kalman_rts(z, t, sigma)
    i = int(np.searchsorted(t, split_t + 1e-6, side="right"))  # impact sample stays pre-crash
    i = min(max(i, 4), len(t) - 4)
    a, sa = kalman_rts(z[:i], t[:i], sigma[:i])
    b, sb = kalman_rts(z[i:], t[i:], sigma[i:], accel_sigma=8.0)
    return np.r_[a, b], np.r_[sa, sb]


def resample(t: np.ndarray, states: np.ndarray, sig: np.ndarray, hz: float = 30.0):
    """Cubic Hermite resample using smoothed velocities (C1-smooth replay)."""
    tt = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    out = np.zeros((len(tt), 4)); so = np.interp(tt, t, sig)
    for j, ti in enumerate(tt):
        k = min(int(np.searchsorted(t, ti, side="right")) - 1, len(t) - 2)
        h = t[k + 1] - t[k]; s = (ti - t[k]) / h
        p0, p1 = states[k, :2], states[k + 1, :2]
        m0, m1 = states[k, 2:] * h, states[k + 1, 2:] * h
        h00 = 2 * s**3 - 3 * s**2 + 1; h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2; h11 = s**3 - s**2
        out[j, :2] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        out[j, 2:] = (1 - s) * states[k, 2:] + s * states[k + 1, 2:]
    return tt, out, so


def headings(states: np.ndarray, min_speed: float = 0.5) -> np.ndarray:
    h = np.zeros(len(states)); last = None
    for i, (vx, vy) in enumerate(states[:, 2:]):
        if math.hypot(vx, vy) >= min_speed or last is None:
            last = math.atan2(vy, vx)
        h[i] = last
    return np.unwrap(h)


# ------------------------------------------------------------ rectangles
def rect(cx, cy, heading, L, W) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    f = np.array([c, s]) * L / 2; r = np.array([s, -c]) * W / 2
    p = np.array([cx, cy])
    return np.array([p + f + r, p + f - r, p - f - r, p - f + r])  # FR, FL, RL, RR


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quads (shapely.Polygon.intersects equivalent)."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _dist_to_poly(p: np.ndarray, poly: np.ndarray) -> float:
    """Signed-ish distance from a point to a convex quad: negative when inside."""
    best = 1e9; inside = True
    for i in range(4):
        a, b = poly[i], poly[(i + 1) % 4]
        e = b - a
        tt = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + tt * e))))
        if e[0] * (p - a)[1] - e[1] * (p - a)[0] < 0:  # rect() corners run counter-clockwise → inside is left
            inside = False
    return -best if inside else best


def contact_zone(car_c, heading, point) -> str:
    """Name the part of a car closest to a world point: front, rear, left_side, right_side, front_left …"""
    d = np.asarray(point) - np.asarray(car_c)
    c, s = math.cos(heading), math.sin(heading)
    lon = d @ np.array([c, s]); lat = d @ np.array([-s, c])  # +lat = left
    ang = math.degrees(math.atan2(lat, lon))
    zones = [(-22.5, 22.5, "front"), (22.5, 67.5, "front_left"), (67.5, 112.5, "left_side"),
             (112.5, 157.5, "rear_left"), (-67.5, -22.5, "front_right"), (-112.5, -67.5, "right_side"),
             (-157.5, -112.5, "rear_right")]
    for lo, hi, name in zones:
        if lo <= ang < hi:
            return name
    return "rear"


def classify(angle_deg: float, contact_a: str, contact_b: str) -> str:
    a = abs((angle_deg + 180) % 360 - 180)
    fronts = ("front", "front_left", "front_right")
    rears = ("rear", "rear_left", "rear_right")
    if a < 20:
        if (contact_a in fronts and contact_b in rears) or (contact_b in fronts and contact_a in rears):
            return "rear_end"
        return "sideswipe"
    if a > 160:
        return "head_on"
    return "angle"


@dataclass
class Track:
    id: str
    type: str
    color: str
    t: np.ndarray       # 30 Hz
    s: np.ndarray       # (N,4)
    sigma: np.ndarray
    heading: np.ndarray

    @property
    def size(self):
        return VEHICLE_SIZES.get(self.type, VEHICLE_SIZES["car"])

    def speed(self):
        return np.hypot(self.s[:, 2], self.s[:, 3])

    def at(self, ti: float) -> int:
        return int(np.clip(np.searchsorted(self.t, ti), 0, len(self.t) - 1))


def _align(a: Track, b: Track) -> tuple[Track, Track]:
    """Restrict both tracks to their common time span on a shared clock (B may be seen only part of the clip)."""
    t0, t1 = max(a.t[0], b.t[0]), min(a.t[-1], b.t[-1])
    tt = a.t[(a.t >= t0 - 1e-6) & (a.t <= t1 + 1e-6)]

    def on(tr: Track) -> Track:
        s = np.c_[[np.interp(tt, tr.t, tr.s[:, k]) for k in range(4)]].T
        return Track(tr.id, tr.type, tr.color, tt, s, np.interp(tt, tr.t, tr.sigma), np.interp(tt, tr.t, tr.heading))
    return on(a), on(b)


def closest_approach(a: Track, b: Track, t_imp: float, window: float = 0.5) -> dict | None:
    """Fallback impact: when the reconstructed outlines just miss at the camera-confirmed impact time,
    take the moment of closest approach within ±window of it and classify from pre-crash headings."""
    a, b = _align(a, b)
    if not len(a.t):
        return None
    sel = np.where(np.abs(a.t - t_imp) <= window)[0]
    if not len(sel):
        return None
    i = int(sel[np.argmin(np.linalg.norm(a.s[sel, :2] - b.s[sel, :2], axis=1))])
    gap = float(np.linalg.norm(a.s[i, :2] - b.s[i, :2]))
    k = max(0, i - 3)
    ha, hb = a.heading[k], b.heading[k]
    ra = rect(*a.s[i, :2], ha, *a.size); rb = rect(*b.s[i, :2], hb, *b.size)
    pa = min(ra, key=lambda p: _dist_to_poly(p, rb)); pb = min(rb, key=lambda p: _dist_to_poly(p, ra))
    pos = (pa + pb) / 2
    ca = contact_zone(a.s[i, :2], ha, pos); cb = contact_zone(b.s[i, :2], hb, pos)
    ang = math.degrees(ha - hb)
    return {"t": round(float(a.t[i]), 3), "index": i, "pos": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
            "type": classify(ang, ca, cb), "closing_speed": round(float(np.linalg.norm(a.s[k, 2:] - b.s[k, 2:])), 2),
            "angle_deg": round(abs((ang + 180) % 360 - 180), 1), "contact": {"A": ca, "B": cb},
            "inferred": True, "centre_gap_m": round(gap, 2)}


def coast(tr: Track, t_end: float, hz: float = 30.0, sigma_rate: float = 1.0,
          velocity: np.ndarray | None = None) -> Track:
    """Extend a lost track at constant velocity (tracker 'coasting'); uncertainty grows with time.
    velocity: override (m/s) when a better estimate than the smoother's last state is available."""
    if t_end <= tr.t[-1]:
        return tr
    extra = np.arange(tr.t[-1] + 1 / hz, t_end + 1e-9, 1 / hz)
    dt = extra - tr.t[-1]
    last = tr.s[-1].copy()
    if velocity is not None:
        last[2:] = velocity
    hdg = math.atan2(last[3], last[2]) if np.hypot(last[2], last[3]) > 0.5 else tr.heading[-1]
    s = np.c_[last[0] + last[2] * dt, last[1] + last[3] * dt, np.full_like(dt, last[2]), np.full_like(dt, last[3])]
    return Track(tr.id, tr.type, tr.color, np.r_[tr.t, extra], np.r_[tr.s, s],
                 np.r_[tr.sigma, tr.sigma[-1] + sigma_rate * dt], np.r_[tr.heading, np.full_like(dt, hdg)])


def find_impact(a: Track, b: Track) -> dict | None:
    a, b = _align(a, b)
    for i in range(len(a.t)):
        ra = rect(*a.s[i, :2], a.heading[i], *a.size)
        rb = rect(*b.s[i, :2], b.heading[i], *b.size)
        if overlap(ra, rb):
            # Headings/velocities from 0.1 s before contact: the pre-crash state is what
            # classifies the crash (the smoother's post-impact segment already shows the rebound).
            k = max(0, i - 3)
            ha, hb = a.heading[k], b.heading[k]
            ra = rect(*a.s[i, :2], ha, *a.size); rb = rect(*b.s[i, :2], hb, *b.size)
            # contact point: the corner of one car that sits deepest inside / nearest the other
            pa = min(ra, key=lambda p: _dist_to_poly(p, rb))
            pb = min(rb, key=lambda p: _dist_to_poly(p, ra))
            pos = pa if _dist_to_poly(pa, rb) <= _dist_to_poly(pb, ra) else pb
            ca = contact_zone(a.s[i, :2], ha, pos)
            cb = contact_zone(b.s[i, :2], hb, pos)
            ang = math.degrees(ha - hb)
            closing = float(np.linalg.norm(a.s[k, 2:] - b.s[k, 2:]))
            return {"t": round(float(a.t[i]), 3), "index": i, "pos": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                    "type": classify(ang, ca, cb), "closing_speed": round(closing, 2),
                    "angle_deg": round(abs((ang + 180) % 360 - 180), 1),
                    "contact": {"A": ca, "B": cb}}
    return None


def stopping_distance(v: float, mu: float = 0.7, t_r: float = 1.5) -> float:
    """d = v t_r + v² / (2 μ g)."""
    return v * t_r + v * v / (2 * mu * G)


def build_track(vid, vtype, color, t10, z, sigma, split_t) -> Track:
    st, sg = smooth_track(t10, z, sigma, split_t)
    tt, s30, sg30 = resample(t10, st, sg)
    return Track(vid, vtype, color, tt, s30, sg30, headings(s30))


def scene_json(sha: str, road: dict, signals: list, signs: list, tracks: list[Track], impact: dict | None,
               meta: dict | None = None) -> dict:
    return {
        "evidence_sha256": sha,
        "road": road,
        "signals": signals,
        "signs": signs,
        "vehicles": [{
            "id": tr.id, "type": tr.type, "size": list(tr.size), "color": tr.color,
            "track": [{"t": round(float(tr.t[i]), 3), "x": round(float(tr.s[i, 0]), 3), "y": round(float(tr.s[i, 1]), 3),
                       "heading": round(float(tr.heading[i]), 4), "speed": round(float(np.hypot(*tr.s[i, 2:])), 3),
                       "sigma": round(float(tr.sigma[i]), 3)} for i in range(len(tr.t))],
        } for tr in tracks],
        "impact": {k: v for k, v in (impact or {}).items() if k != "index"} or None,
        "duration": round(float(max(tr.t[-1] for tr in tracks)), 3),
        "meta": meta or {},
    }


def tracks_from_scene(scene: dict) -> dict[str, Track]:
    out = {}
    for v in scene["vehicles"]:
        tr = v["track"]
        t = np.array([p["t"] for p in tr])
        h = np.array([p["heading"] for p in tr])
        sp = np.array([p["speed"] for p in tr])
        s = np.c_[[p["x"] for p in tr], [p["y"] for p in tr], sp * np.cos(h), sp * np.sin(h)]
        out[v["id"]] = Track(v["id"], v["type"], v["color"], t, s, np.array([p["sigma"] for p in tr]), h)
    return out
