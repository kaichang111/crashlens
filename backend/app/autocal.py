"""Calibrate the road plane from the video itself instead of trusting four hand-placed clicks.

The reconstruction needs the four corners of the dashcam car's own lane, exactly one dash + gap long.
Users click something else surprisingly often (both lanes plus the shoulder, or a rectangle up to the
horizon), and a few pixels on the far corners rescale every distance and speed. Here:

  1. lane lines: Hough segments on the per-pixel median of the first frames (lane paint barely moves
     in the image while the car drives straight; traffic does, and blurs out);
  2. vanishing point: RANSAC intersection of those segments → horizon row;
  3. own lane: the two lines either side of the image centre at a near row — the dashcam car is in it
     by construction, whatever was clicked;
  4. camera height from the lane width (no focal length needed), focal length from the dash spacing
     along the dashed line (successive dash ends are one cycle apart: 1/(v−v_h) steps by C/(f·h));
  5. the four corners of a one-cycle rectangle in that lane, so the rest of the pipeline is unchanged.

Falls back (ok=False) when the road has no usable lines; the caller then keeps the user's clicks.
"""
from __future__ import annotations

import math

import numpy as np

from . import store

N_FRAMES = 15          # frames used for the median image / dash spacing
MIN_SEG_FRAC = 0.035   # Hough segment min length, as a fraction of the frame width
MIN_ANGLE_DEG = 12     # ignore near-horizontal segments (hood edge, stop lines, horizon)
H_RANGE = (0.8, 2.4)   # plausible dashcam heights, m
HFOV_RANGE = (45, 150)  # plausible horizontal fields of view, deg


def _vanishing_point(segs: np.ndarray, W: int, H: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """RANSAC: the point most (length-weighted) segments point at. Returns (vp, inlier mask)."""
    if len(segs) < 4:
        return None
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]; p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lens = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    best, best_score = None, 0.0
    for _ in range(600):
        i, j = rng.choice(len(segs), 2, replace=False)
        x = np.cross(lines[i], lines[j])
        if abs(x[2]) < 1e-9:
            continue
        vp = x[:2] / x[2]
        if not (0.15 * W < vp[0] < 0.85 * W and 0.1 * H < vp[1] < 0.75 * H):
            continue
        d = np.abs(lines[:, :2] @ vp + lines[:, 2])
        score = lens[d < 3.0].sum()
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None
    inl = np.abs(lines[:, :2] @ best + lines[:, 2]) < 3.0
    # refine: least-squares point closest to all inlier lines (length-weighted)
    A = lines[inl, :2] * lens[inl, None]; b = -lines[inl, 2] * lens[inl]
    vp = np.linalg.lstsq(A, b, rcond=None)[0]
    return vp, inl


def _line_clusters(segs: np.ndarray, vp: np.ndarray, W: int) -> list[tuple[float, float]]:
    """Distinct lines through the vanishing point: (angle from vp, total segment length)."""
    mids = (segs[:, :2] + segs[:, 2:]) / 2
    ang = np.degrees(np.arctan2(mids[:, 1] - vp[1], mids[:, 0] - vp[0]))
    lens = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    order = np.argsort(ang)
    clusters, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if ang[b] - ang[a] <= 1.2:
            cur.append(b)
        else:
            clusters.append(cur); cur = [b]
    clusters.append(cur)
    out = []
    for c in clusters:
        w = lens[c].sum()
        if w >= 0.05 * W:
            out.append((float(np.average(ang[c], weights=lens[c])), float(w)))
    return out


def _u_at(vp: np.ndarray, ang_deg: float, v: float) -> float:
    a = math.radians(ang_deg)
    return float(vp[0] + (v - vp[1]) * math.cos(a) / math.sin(a))


def _refine_line(img: np.ndarray, vp: np.ndarray, ang_deg: float, v_lo: float, v_hi: float,
                 half_px_at_hi: float) -> float:
    """Re-centre a line through the vanishing point on the paint's brightness ridge.

    Hough clusters land on paint edges (or a mix of both edges and nearby cracks), which put the lane
    0.3–0.5 m off; the dash tracker samples a 0.7 m strip on each line and then misses the dashes.
    Row by row, take the brightest column near the line (search window shrinks toward the horizon),
    then least-squares fit a line through the vanishing point to those ridge points."""
    H, W = img.shape
    sm = img.astype(np.float32)
    du, dv = [], []
    for v in range(int(v_lo), int(v_hi), 2):
        u0 = _u_at(vp, ang_deg, v)
        half = max(3, int(half_px_at_hi * (v - vp[1]) / max(v_hi - vp[1], 1)))
        lo, hi = int(max(0, u0 - half)), int(min(W - 1, u0 + half))
        if hi - lo < 3:
            continue
        row = np.convolve(sm[v, lo:hi + 1], np.ones(3) / 3, mode="same")
        j = int(np.argmax(row))
        if row[j] - np.median(row) < 12:  # no paint here (gap between dashes, occlusion)
            continue
        du.append(lo + j - vp[0]); dv.append(v - vp[1])
    if len(du) < 8:
        return ang_deg
    du, dv = np.array(du), np.array(dv)
    c = float((du * dv).sum() / (dv * dv).sum())  # u − u_vp = c · (v − v_vp)
    return math.degrees(math.atan2(1.0, c))


def _dash_focal(grays: list[np.ndarray], vp: np.ndarray, ang_deg: float, v_lo: float, v_hi: float,
                cam_h: float, cycle_m: float) -> list[float]:
    """Focal-length estimates from successive dash near-ends along one line, over several frames."""
    est = []
    vs = np.arange(int(v_lo), int(v_hi))
    if len(vs) < 20:
        return est
    us = np.array([_u_at(vp, ang_deg, v) for v in vs])
    for g in grays:
        H, W = g.shape
        ok = (us >= 2) & (us < W - 2)
        if ok.sum() < 20:
            continue
        ui = np.round(us[ok]).astype(int); vi = vs[ok]
        prof = np.max(np.stack([g[vi, ui + d] for d in (-2, -1, 0, 1, 2)]), axis=0).astype(np.float32)
        lo, hi = np.percentile(prof, 20), np.percentile(prof, 95)
        if hi - lo < 25:
            continue
        bright = prof > (lo + hi) / 2
        # near end of each dash = last bright row of a bright run (rows grow toward the camera)
        ends = [vi[i] for i in range(len(bright) - 1) if bright[i] and not bright[i + 1]]
        ends = [e for e in ends if e - vp[1] > 3]
        for a, b in zip(ends[:-1], ends[1:]):
            ds = 1 / (a - vp[1]) - 1 / (b - vp[1])  # a is farther (smaller row) than b
            if ds > 0:
                est.append(cycle_m / (cam_h * ds))
    return est


def calibrate(cid: str, fm: dict, hint: dict, lane_w: float, cycle_m: float, rng_seed: int = 0) -> dict:
    import cv2
    frames = fm.get("frames") or []
    t_use = fm.get("impact_t", 5.0) - 1.5  # before the other car fills the view
    idx = [f["i"] for f in frames if f["t"] <= t_use][:N_FRAMES] or [f["i"] for f in frames[:N_FRAMES]]
    grays = [g for g in (cv2.imread(str(store.case_dir(cid) / "frames" / f"{i:03d}.jpg"), cv2.IMREAD_GRAYSCALE)
                         for i in idx) if g is not None]
    if len(grays) < 3:
        return {"ok": False, "reason": "too few frames"}
    H, W = grays[0].shape
    # 80th percentile, not median: a dash covers a given pixel only ~25 % of the time while driving,
    # so the median erases dashed lines (and then two lanes look like one); the 80th percentile turns
    # them into continuous lines while traffic, which passes each pixel briefly, still blurs out
    med = np.percentile(np.stack(grays), 80, axis=0).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(med, (5, 5), 0), 40, 120)
    edges[: int(0.2 * H)] = 0
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=int(MIN_SEG_FRAC * W), maxLineGap=int(0.02 * W))
    if raw is None:
        return {"ok": False, "reason": "no line segments"}
    segs = np.asarray(raw).reshape(-1, 4).astype(float)  # (N,1,4) or (N,4) depending on the OpenCV build
    ang = np.degrees(np.abs(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])))
    ang = np.minimum(ang, 180 - ang)
    segs = segs[ang >= MIN_ANGLE_DEG]
    found = _vanishing_point(segs, W, H, np.random.default_rng(rng_seed))
    if found is None:
        return {"ok": False, "reason": "no vanishing point"}
    vp, inl = found
    segs_in = segs[inl]
    if len(segs_in) < 3:
        return {"ok": False, "reason": "too few lines through the vanishing point"}
    lines = _line_clusters(segs_in, vp, W)
    # near row: where the lane paint is still visible above the hood (lowest inlier endpoints)
    v_near = float(np.percentile(np.r_[segs_in[:, 1], segs_in[:, 3]], 90))
    if hint.get("points"):
        v_near = min(v_near, float(np.mean([p[1] for p in hint["points"][:2]])) + 0.05 * H)
    v_near = min(v_near, 0.95 * H)
    if v_near - vp[1] < 0.08 * H:
        return {"ok": False, "reason": "lane lines end too close to the horizon", "vp": vp.round(1).tolist()}
    v_top0 = vp[1] + 0.06 * (v_near - vp[1]) + 3
    lines = [(_refine_line(med, vp, a, v_top0, v_near, 0.012 * W), w) for a, w in lines]
    us = sorted(((_u_at(vp, a, v_near), a) for a, _ in lines), key=lambda x: x[0])
    left = [x for x in us if x[0] < W / 2]; right = [x for x in us if x[0] >= W / 2]
    if not left or not right:
        return {"ok": False, "reason": "no lane line on one side of the car", "vp": vp.round(1).tolist()}
    (uL, aL), (uR, aR) = left[-1], right[0]
    w_px = uR - uL
    cam_h = lane_w * (v_near - vp[1]) / w_px
    n_lanes = 1
    if cam_h < H_RANGE[0]:
        # The pair is wider than one lane: a line between them (usually a faint dashed one) was missed.
        # If it spans 2 or 3 lanes, the height comes out at 1/2 or 1/3 of a normal dashcam's; take the
        # lane the camera is in.
        for n in (2, 3):
            if H_RANGE[0] <= cam_h * n <= H_RANGE[1]:
                n_lanes = n
                break
        if n_lanes > 1:
            i = min(n_lanes - 1, int((W / 2 - uL) / (w_px / n_lanes)))
            def sub(frac):  # a line through the vp, frac of the way from the left line to the right one
                u = uL + frac * w_px
                return u, math.degrees(math.atan2(v_near - vp[1], u - vp[0]))
            (uL, aL), (uR, aR) = sub(i / n_lanes), sub((i + 1) / n_lanes)
            w_px, cam_h = uR - uL, cam_h * n_lanes
    if not (H_RANGE[0] <= cam_h <= H_RANGE[1]):
        return {"ok": False, "reason": f"own-lane width implies a {cam_h:.2f} m camera height",
                "vp": vp.round(1).tolist()}
    v_top = vp[1] + 0.06 * (v_near - vp[1]) + 3
    est = (_dash_focal(grays, vp, aL, v_top, v_near, cam_h, cycle_m)
           + _dash_focal(grays, vp, aR, v_top, v_near, cam_h, cycle_m))
    f_lo = (W / 2) / math.tan(math.radians(HFOV_RANGE[1] / 2)); f_hi = (W / 2) / math.tan(math.radians(HFOV_RANGE[0] / 2))
    est = [e for e in est if f_lo <= e <= f_hi]
    method, f = "lines+dashes", None
    if len(est) >= 4:
        # Densest cluster, not the overall spread: pairing a dash with the one after next (a missed or
        # merged dash) gives ½× or 2× the true value, which scatters the set without moving its mode.
        lg = np.log(np.array(est))
        counts = [(np.abs(lg - c) <= 0.12).sum() for c in lg]
        c = lg[int(np.argmax(counts))]
        near = np.exp(lg[np.abs(lg - c) <= 0.12])
        if len(near) >= max(4, 0.3 * len(est)):
            f = float(np.median(near))
    if f is None:  # no clean dash spacing: keep the hint's (assumed-HFOV) focal length
        method = "lines+hfov"
        f = (W / 2) / math.tan(math.radians(float(hint.get("hfov_deg", 130.0)) / 2))
    z_near = f * cam_h / (v_near - vp[1])
    v_far = vp[1] + f * cam_h / (z_near + cycle_m)
    pts = [[uL, v_near], [uR, v_near], [_u_at(vp, aR, v_far), v_far], [_u_at(vp, aL, v_far), v_far]]
    return {"ok": True, "method": method, "points": [[round(x, 1), round(y, 1)] for x, y in pts],
            "vp": vp.round(1).tolist(), "cam_h_m": round(cam_h, 2), "f_px": round(f, 1),
            "hfov_deg": round(math.degrees(2 * math.atan((W / 2) / f)), 1), "lane_px": round(w_px, 1),
            "dash_estimates": len(est), "near_row": round(v_near, 1), "lines_found": len(lines),
            "lanes_spanned": n_lanes}
