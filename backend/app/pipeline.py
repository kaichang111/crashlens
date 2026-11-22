"""Steps 1–3 on a real dashcam clip.

Heavy dependencies are optional: OpenCV for video I/O and geometry,
Ultralytics YOLO11 (+ built-in ByteTrack) for detection/tracking, librosa for
the crash-sound cue. If they are missing, the API reports that and the demo
uses cached cases (?cached=1), as the spec's backup plan describes.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path

import numpy as np

from . import facts as factmod
from . import reconstruct as rc
from . import store

KEEP = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic_light", 11: "stop_sign"}
VEHICLE_CLS = {"car", "motorcycle", "bus", "truck"}
OUT_FPS = 10
PRE, POST = 5.0, 2.0
DASH_CYCLE_M = 12.19  # 10 ft dash + 30 ft gap


def capabilities() -> dict:
    caps = {}
    for mod in ("cv2", "ultralytics", "librosa"):
        try:
            __import__(mod)
            caps[mod] = True
        except Exception:
            caps[mod] = False
    caps["live_pipeline"] = caps["cv2"] and caps["ultralytics"]
    return caps


# ------------------------------------------------------------------ Step 1
def _camera_matrix(w, h, hfov=130.0):
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float), f


def _undistort(fr: np.ndarray, k1: float) -> np.ndarray:
    """Radial (barrel) correction, with k1 measured at the image corner (r = 1 at the half-diagonal).

    It used to be normalised by a 130° focal length, which puts the corners at r ≈ 2.4: the default
    k1 = −0.25 then folded the mapping back on itself beyond ~340 px from the centre and mirrored
    the scene into a ring, so detection and calibration clicks ran on garbage. The map stays
    monotonic while 1 + 3·k1·r² > 0, i.e. k1 > −1/3 with r ≤ 1, so k1 is clamped just above that.
    """
    import cv2
    if abs(k1) < 1e-6:
        return fr
    h, w = fr.shape[:2]
    r = math.hypot(w / 2, h / 2)
    K = np.array([[r, 0, w / 2], [0, r, h / 2], [0, 0, 1]], float)
    return cv2.undistort(fr, K, np.array([max(k1, K1_MIN), 0, 0, 0], float))


K1_MIN = -0.3


def visual_jolt(path: Path) -> tuple[np.ndarray, float]:
    """Mean absolute difference of consecutive grayscale frames (camera shake at impact)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None; diffs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return np.array(diffs), fps


def crash_sound_t(path: Path) -> float | None:
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        if len(y) == 0:
            return None
        env = librosa.onset.onset_strength(y=y, sr=sr)
        return float(librosa.frames_to_time(int(np.argmax(env)), sr=sr))
    except Exception:
        return None


def guess_impact(path: Path) -> dict:
    d, fps = visual_jolt(path)
    t_jolt = None
    if len(d) > 5:
        # Robust baseline: the median frame difference. (mean + k·σ fails because the
        # post-crash chaos inflates σ and the first shake falls under the bar.)
        base = float(np.median(d[1:]))
        thr = max(15.0, 5.0 * base)
        idx = np.where(d > thr)[0]
        t_jolt = float((idx[0] if len(idx) else int(np.argmax(d))) / fps)
    t_sound = crash_sound_t(path)
    cands = [t for t in (t_jolt, t_sound) if t is not None]
    # take the earliest signal the cues agree on (within 0.5 s); else the jolt
    t = min(cands) if len(cands) == 2 and abs(cands[0] - cands[1]) < 0.5 else (t_jolt or t_sound or 0.0)
    return {"t": t, "jolt_t": t_jolt, "sound_t": t_sound, "fps": fps, "duration": len(d) / fps}


def extract_frames(cid: str, path: Path, t_impact: float, k1: float = -0.25) -> dict:
    """impact − 5 s … impact + 2 s, resampled to 10 fps, mildly undistorted."""
    import cv2
    out = store.case_dir(cid) / "frames"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = max(0.0, t_impact - PRE); t1 = min(n / fps, t_impact + POST)
    # which source frames we want, in order (10 fps from a 15/25/29.97 fps clip is unevenly spaced)
    wanted = sorted({int(round(t * fps)) for t in np.arange(t0, t1, 1.0 / OUT_FPS)})
    if not wanted:
        cap.release()
        raise ValueError(f"no frames to extract: impact at {t_impact:.2f} s is outside the clip "
                         f"({n} frames at {fps:.1f} fps)")
    frames = []
    h = w = 0
    # Decode sequentially. Seeking with CAP_PROP_POS_FRAMES on H.264 can land a frame off depending
    # on the keyframe interval, which silently corrupts every time step (and every speed) downstream.
    src, i, want = 0, 0, set(wanted)
    while src <= wanted[-1]:
        ok, fr = cap.read()
        if not ok:
            break
        if src in want:
            h, w = fr.shape[:2]
            fr = _undistort(fr, k1)
            cv2.imwrite(str(out / f"{i:03d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            frames.append({"i": i, "t": round(src / fps - t0, 4), "src_t": round(src / fps, 4)})
            i += 1
        src += 1
    cap.release()
    meta = {"fps": OUT_FPS, "clip_offset": round(float(t0), 3), "impact_t": round(float(t_impact - t0), 3),
            "width": int(w), "height": int(h), "k1": k1, "frames": frames}
    store.write_json(cid, "frames.json", meta)
    return meta


def preview_frame(cid: str, i: int, k1: float) -> bytes:
    """Frame i of the extracted window, re-undistorted with k1 (spec §1.5 slider preview)."""
    import cv2
    case = store.read_json(cid, "case.json")
    fm = store.read_json(cid, "frames.json")
    cap = cv2.VideoCapture(str(store.case_dir(cid) / case["video_file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((fm["clip_offset"] + i / OUT_FPS) * fps)))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(i)
    fr = _undistort(fr, k1)
    return cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def ingest(cid: str) -> None:
    """Runs right after upload so the Calibrate screen has frames to click on."""
    try:
        store.set_status(cid, "ingesting", 5)
        case = store.read_json(cid, "case.json")
        path = store.case_dir(cid) / case["video_file"]
        imp = guess_impact(path)
        store.write_json(cid, "impact_guess.json", imp)
        extract_frames(cid, path, imp["t"])
        store.set_status(cid, "awaiting_calibration", 15, status="awaiting_calibration")
    except Exception as e:
        traceback.print_exc()
        store.set_status(cid, "error", 0, status="error", error=f"ingest failed: {e}")


# ------------------------------------------------------------------ Step 2
def light_state(crop_bgr) -> str:
    import cv2
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (s > 90) & (v > 150)
    counts = {"red": int(np.sum(bright & ((h < 10) | (h > 170)))),
              "yellow": int(np.sum(bright & (h >= 15) & (h <= 35))),
              "green": int(np.sum(bright & (h >= 45) & (h <= 90)))}
    best = max(counts, key=counts.get)
    return best if counts[best] >= max(6, 0.02 * h.size) else "unknown"


def perceive(cid: str, device: str | None = None, imgsz: int = 1280, conf: float = 0.25) -> dict:
    import cv2
    from ultralytics import YOLO
    fdir = store.case_dir(cid) / "frames"
    files = sorted(fdir.glob("*.jpg"))
    model = YOLO("yolo11s.pt")
    frames = []
    kw = {"device": device} if device else {}
    for i, f in enumerate(files):
        # imgsz 1280: a car 50 m out is ~25 px wide and vanishes at the default 640.
        # agnostic_nms: one box per object, so a pickup isn't both "car" and "truck" (which splits tracks).
        r = model.track(source=str(f), tracker="bytetrack.yaml", classes=list(KEEP), conf=conf,
                        imgsz=imgsz, agnostic_nms=True, persist=True, verbose=False, **kw)[0]
        img = cv2.imread(str(f))
        objs = []
        if r.boxes is not None and r.boxes.id is not None:
            for box, tid, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist(),
                                            r.boxes.conf.tolist()):
                o = {"track": int(tid), "cls": KEEP[int(cls)], "bbox": [round(x, 1) for x in box],
                     "conf": round(float(score), 3)}
                if o["cls"] == "traffic_light":
                    x1, y1, x2, y2 = map(int, box)
                    o["state_raw"] = light_state(img[max(0, y1):y2, max(0, x1):x2])
                objs.append(o)
        frames.append({"t": round(i / OUT_FPS, 2), "objects": objs})  # nominal; true times live in frames.json
    det = {"fps": OUT_FPS, "frames": frames}
    _clean_tracks(det)
    _vote_lights(det)
    store.write_json(cid, "detections.json", det)
    return det


def _clean_tracks(det: dict) -> None:
    """Drop flicker (<5 frames); merge re-IDs that end/start within 3 frames and 40 px."""
    spans: dict[int, list] = {}
    for fi, fr in enumerate(det["frames"]):
        for o in fr["objects"]:
            spans.setdefault(o["track"], []).append((fi, o))
    remap = {}
    order = sorted(spans, key=lambda k: spans[k][0][0])
    for a in order:
        for b in order:
            if a == b or b in remap or a in remap:
                continue
            ea, oa = spans[a][-1]; sb, ob = spans[b][0]
            same_kind = oa["cls"] == ob["cls"] or (oa["cls"] in VEHICLE_CLS and ob["cls"] in VEHICLE_CLS)
            if 0 < sb - ea <= 3 and same_kind:
                ca = np.array([(oa["bbox"][0] + oa["bbox"][2]) / 2, oa["bbox"][3]])
                cb = np.array([(ob["bbox"][0] + ob["bbox"][2]) / 2, ob["bbox"][3]])
                if np.linalg.norm(ca - cb) < 40:
                    remap[b] = a
    for fr in det["frames"]:
        for o in fr["objects"]:
            o["track"] = remap.get(o["track"], o["track"])
    counts: dict[int, int] = {}
    for fr in det["frames"]:
        for o in fr["objects"]:
            counts[o["track"]] = counts.get(o["track"], 0) + 1
    for fr in det["frames"]:  # drop flicker; small far-off lights get a lower bar
        fr["objects"] = [o for o in fr["objects"]
                         if counts[o["track"]] >= (3 if o["cls"] == "traffic_light" else 5)]
    # one class per track (majority vote): a low-poly pickup flickers between car/truck/bus
    votes: dict[int, dict[str, int]] = {}
    for fr in det["frames"]:
        for o in fr["objects"]:
            if o["cls"] in VEHICLE_CLS:
                v = votes.setdefault(o["track"], {})
                v[o["cls"]] = v.get(o["cls"], 0) + 1
    for fr in det["frames"]:
        for o in fr["objects"]:
            if o["track"] in votes:
                o["cls"] = max(votes[o["track"]], key=votes[o["track"]].get)


def _vote_lights(det: dict, win: int = 5) -> None:
    hist: dict[int, list] = {}
    for fr in det["frames"]:
        for o in fr["objects"]:
            if o["cls"] == "traffic_light":
                h = hist.setdefault(o["track"], [])
                h.append(o.get("state_raw", "unknown"))
                recent = [s for s in h[-win:] if s != "unknown"]
                o["state"] = max(set(recent), key=recent.count) if recent else "unknown"


def pick_other(det: dict, impact_i: int, tap: list[float] | None) -> int | None:
    """Vehicle B = the track the user tapped on the impact frame (fallback: biggest box).

    At the impact frame the other car often fills the windscreen and isn't detected, so look
    up to 1 s around it, earlier frames first.
    """
    order = [0] + [d for k in range(1, 11) for d in (-k, k)]
    for di in order:
        k = impact_i + di
        if not 0 <= k < len(det["frames"]):
            continue
        vs = [o for o in det["frames"][k]["objects"] if o["cls"] in VEHICLE_CLS]
        if not vs:
            continue
        if tap:
            inside = [o for o in vs if o["bbox"][0] <= tap[0] <= o["bbox"][2] and o["bbox"][1] <= tap[1] <= o["bbox"][3]]
            if inside:
                return inside[0]["track"]
            return min(vs, key=lambda o: math.hypot((o["bbox"][0] + o["bbox"][2]) / 2 - tap[0],
                                                     (o["bbox"][1] + o["bbox"][3]) / 2 - tap[1]))["track"]
        return max(vs, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))["track"]
    return None


# ------------------------------------------------------------------ Step 3
# Ego-motion options (module-level so the regression harness can compare variants on real clips)
# Defaults chosen by scripts on the verified Blender clip (ground truth) and a real dash-less clip:
# bev_auto picks the long/dash window or the near/flow window per clip; the static-overlay mask hurt the
# Blender turn timing (TTC 0.7 s vs 2.6 s) and isn't needed once the near window handles dash-less roads.
EGO_OPTS = {"bev_auto": True, "bev_far_frac": 0.05, "bev_x": "wide", "yaw_far_scene": True,
            "image_ground_fallback": True, "static_mask": False, "measured_hood": True}

BEV_PPM = 10.0  # bird's-eye pixels per metre
DASH_Q_OVERRIDE = 0.6  # dash-correlation quality needed to use it over (or without) optical flow
RELOCK_N, RELOCK_TOL, RELOCK_MAX_GAP = 5, 0.4, 3  # re-lock: 4 of the last 5 rejected locks within ±40 %
MAX_LAT_ACC = 4.0  # m/s²: yaw rate × speed beyond a hard swerve (~0.4 g) isn't the camera turning
LANE_LOCK = True  # sideways position/heading from the clicked lane lines when they're visible (_lane_lock)
AUTOCAL = os.environ.get("CRASHLENS_AUTOCAL", "1") != "0"  # lane rectangle + focal length from the video
LANE_PEAK_MIN = 6.0  # grey levels: a lane line's ridge above the local road brightness
BEV_X = (-15.0, 20.0)
BEV_Y = (-5.0, 60.0)


DEFAULT_EXT = {"x": BEV_X, "y": BEV_Y, "ppm": BEV_PPM}


def _bev_matrix(H, ext: dict = DEFAULT_EXT):
    ppm, (x0, x1), (y0, y1) = ext["ppm"], ext["x"], ext["y"]
    S = np.array([[ppm, 0, -x0 * ppm], [0, -ppm, y1 * ppm], [0, 0, 1]], float)
    size = (int((x1 - x0) * ppm), int((y1 - y0) * ppm))
    return S @ H, size


def _bev_to_m(pts, ext: dict = DEFAULT_EXT):
    ppm, (x0, _), (_, y1) = ext["ppm"], ext["x"], ext["y"]
    return np.c_[pts[:, 0] / ppm + x0, y1 - pts[:, 1] / ppm]


def _strip_profile(bev: np.ndarray, x_m: float, y0: float | None = None, y1: float | None = None,
                   half_w_m: float = 0.35, peak: bool = False, ext: dict = DEFAULT_EXT):
    """Brightness along a lane line in the bird's-eye view, as a function of distance ahead.

    Mean across the strip by default. peak=True takes the max instead: a painted line is ~0.15 m
    wide (1–2 BEV px) inside a 0.7 m strip that absorbs click error, so on real, low-contrast
    footage a mean dilutes the dashes ~5× below the energy threshold. The max is noisier on crisp
    footage, so it is only the fallback.
    """
    ppm, (bx0, _), (by0, by1) = ext["ppm"], ext["x"], ext["y"]
    span = by1 - by0
    y0 = by0 + 0.12 * span if y0 is None else y0   # skip the stretched rows right above the hood
    y1 = by0 + 0.55 * span if y1 is None else y1   # and the smeared rows near the horizon
    u = int(round((x_m - bx0) * ppm)); hw = max(1, int(round(half_w_m * ppm)))
    v0 = int(round((by1 - y1) * ppm)); v1 = int(round((by1 - y0) * ppm))
    strip = bev[max(0, v0):v1, max(0, u - hw):u + hw + 1].astype(np.float32)
    if strip.size == 0:
        return np.zeros(1, np.float32)
    p = strip.max(axis=1) if peak else strip.mean(axis=1)
    # high-pass (subtract a 3 m moving average): keep dash edges, drop the static brightness
    # gradient and solid lines, which otherwise correlate best at zero shift
    k = int(3 * ppm) | 1
    smooth = np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return p - smooth


def _strip_shift(prev: np.ndarray, nxt: np.ndarray, max_px: int = 75, ppm: float = BEV_PPM) -> tuple[float, float]:
    """Forward motion (m) between frames from 1-D cross-correlation of a dashed lane-line profile.

    A dash at distance y in one frame is at y − dy in the next, i.e. dy·PPM rows further down the
    bird's-eye image. One dash + gap = 12.19 m = 122 px, so shifts up to ~60 px (≈60 m/s at 10 fps)
    are unambiguous. Returns (dy_m, quality 0–1).
    """
    n = len(prev)
    if n < 3 * max_px or prev.std() < 4 or nxt.std() < 4:
        return 0.0, 0.0
    a = prev[: n - max_px]
    best, best_s = -1.0, 0
    for s in range(0, max_px):
        b = nxt[s: s + len(a)]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        if c > best:
            best, best_s = c, s
    return best_s / ppm, max(0.0, best)


def yaw_steps(grays: list[np.ndarray], static: np.ndarray | None, v_h: float, f: float) -> list[float | None]:
    """Per-frame yaw change (rad, + = left) from the sideways shift of distant scenery.

    Rows just above the horizon (trees, poles, buildings) are far away, so forward motion barely moves
    them, while turning shifts them by f·tan(Δψ). Phase correlation on that band gives the shift; the
    band ignores the road, so it works where road-texture flow fails. Camera-fixed overlays are blanked.
    """
    import cv2
    h, w = grays[0].shape[:2]
    # Band from well above the horizon to just below it: trees, poles, buildings. (Phase correlation over
    # this band locked onto the plain sky and the windscreen reflection and read ~0° through a real turn.)
    top = int(max(0, v_h - 0.6 * v_h)); bot = int(min(h, v_h + 0.05 * (h - v_h)))
    if bot - top < 8:
        return [None] * len(grays)
    mask = np.zeros((h, w), np.uint8)
    mask[top:bot] = 255
    if static is not None:
        mask[static > 0] = 0
    out: list[float | None] = [0.0]
    for k in range(1, len(grays)):
        p0 = cv2.goodFeaturesToTrack(grays[k - 1], 200, 0.01, 5, mask=mask)
        if p0 is None or len(p0) < 8:
            out.append(None)
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[k - 1], grays[k], p0, None, winSize=(31, 31), maxLevel=5)
        dx = (p1 - p0).reshape(-1, 2)[st.ravel() == 1, 0]
        if len(dx) < 8:
            out.append(None)
            continue
        # Most common sideways shift: forward motion spreads near features left and right symmetrically,
        # moving cars are a minority, a turn shifts the whole far scene together.
        hist, edges = np.histogram(dx, bins=np.arange(-0.3 * w, 0.3 * w + 5, 5.0))
        i = int(np.argmax(hist))
        sel = dx[np.abs(dx - (edges[i] + 2.5)) <= 5.0]
        if len(sel) < 0.25 * len(dx):
            out.append(None)
            continue
        # scene moving right in the image (dx > 0) = camera yawing left (+)
        out.append(math.atan2(float(np.median(sel)), f))
    return out


def _hood_row_from_video(cid: str, v_h: float | None) -> int | None:
    """Top row of the camera-fixed band at the bottom of the frame (hood, dashboard, its reflection).

    Per-row temporal std over the clip: the road changes as the car moves, the hood doesn't. Scan up
    from the bottom while rows stay still (std below 5, or below a third of the road's typical std).
    """
    import cv2
    files = sorted((store.case_dir(cid) / "frames").glob("*.jpg"))
    if len(files) < 10:
        return None
    use = files[: max(10, int(len(files) * 0.65))]   # before the crash shakes everything
    g = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in use]).astype(np.float32)
    rs = g.std(axis=0).mean(axis=1)
    h = len(rs)
    lo = int(v_h + 0.2 * (h - v_h)) if v_h is not None else h // 2
    road = float(np.median(rs[lo: int(0.75 * h)])) if int(0.75 * h) > lo else float(np.median(rs[h // 2:]))
    thr = max(5.0, min(0.35 * road, 8.0))
    r = h - 1
    while r > lo and rs[r] < thr:
        r -= 1
    if r >= h - 3:
        return None          # no still band at the bottom: no visible hood
    return int(r - 3)


def _image_ground_step(prev: np.ndarray, cur: np.ndarray, mask_img: np.ndarray, H: np.ndarray,
                       v_h: float, hood_row: int | None) -> tuple[np.ndarray | None, int]:
    """Camera motion between two frames from ground features tracked in image space.

    Corners on the road (between just below the horizon and the hood, with vehicles and camera-fixed
    overlays already masked out in mask_img) are tracked with LK, both ends are projected to road
    metres through H, and a rigid motion is fitted with RANSAC. Returns (step, inliers).
    """
    import cv2
    h = prev.shape[0]
    bot = hood_row if hood_row is not None else h
    top = int(v_h + 0.12 * (bot - v_h))          # rows nearer the horizon are metres per pixel
    m = np.zeros_like(mask_img)
    m[top:bot] = mask_img[top:bot]
    p0 = cv2.goodFeaturesToTrack(prev, 300, 0.005, 5, mask=m)
    if p0 is None or len(p0) < 10:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=4)
    good = (st.ravel() == 1) & (p1.reshape(-1, 2)[:, 1] > v_h + 2)
    if good.sum() < 10:
        return None, 0
    a = rc.project(H, p0.reshape(-1, 2)[good]); b = rc.project(H, p1.reshape(-1, 2)[good])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=0.25)
    n = int(inl.sum()) if inl is not None else 0
    if M is None or n < 12:
        return None, n
    # keep it rigid: rebuild the rotation from the angle so RANSAC's scale can't leak into the pose
    ang = math.atan2(M[1, 0], M[0, 0])
    Mr = np.array([[math.cos(ang), -math.sin(ang), M[0, 2]], [math.sin(ang), math.cos(ang), M[1, 2]], [0, 0, 1]])
    return np.linalg.inv(Mr), n


def ego_motion(cid: str, H: np.ndarray, det: dict, lane_lines_x: tuple[float, ...] = (0.0, 3.66),
               hood_row: int | None = None, times: list[float] | None = None,
               ext: dict = DEFAULT_EXT, v_h: float | None = None, f: float | None = None
               ) -> tuple[np.ndarray, list[dict]]:
    """Camera pose per frame (3×3 homogeneous, road metres).

    Two estimators per frame pair:
      * optical flow on road texture in the bird's-eye view → rigid motion (rotation + translation);
        vehicles and the car's own hood are masked out (the hood never moves relative to the camera);
      * lane-dash correlation along each lane line → forward distance (spec §3.3 dash check).
    The dash estimate wins for forward motion when flow is missing or disagrees by > 15 %.
    """
    import cv2
    files = sorted((store.case_dir(cid) / "frames").glob("*.jpg"))
    grays = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    if any(g is None for g in grays):
        raise RuntimeError("a frame image could not be read; re-extract the frames")
    if len(grays) != len(det["frames"]) or (times is not None and len(times) != len(grays)):
        raise RuntimeError(f"frame count mismatch (images {len(grays)}, detections {len(det['frames'])}, "
                           f"timestamps {len(times) if times is not None else '-'}): stale frames folder?")
    # Camera-fixed overlays (windscreen reflections, the dashcam's date stamp, the hood) keep sharp
    # edges in the per-pixel median of the whole clip, while the moving scene blurs out. Corners on
    # them track with zero motion, and RANSAC then "proves" the car is parked. Mask them out.
    static = np.zeros(grays[0].shape[:2], np.uint8) if grays else None
    if len(grays) >= 10 and EGO_OPTS["static_mask"]:
        med = np.median(np.stack(grays[:: max(1, len(grays) // 25)]), axis=0).astype(np.uint8)
        static = cv2.dilate(cv2.Canny(med, 60, 140), np.ones((9, 9), np.uint8))
        # A stopped or very slow camera leaves the whole road sharp in the median: then the mask would
        # erase the road itself. Drop it if it covers more than 40 % of the lower half.
        if float((static[static.shape[0] // 2:] > 0).mean()) > 0.4:
            static = np.zeros_like(static)
    yaws = (yaw_steps(grays, static, v_h, f) if (EGO_OPTS["yaw_far_scene"] and v_h is not None and f and grays)
            else [None] * len(grays))
    Hb, size = _bev_matrix(H, ext)
    ppm = ext["ppm"]
    bevs = [cv2.warpPerspective(g, Hb, size) for g in grays]
    # Strip mean vs peak (see _strip_profile), chosen once per clip: mixing them frame by frame feeds
    # the noisier peak readings into clips the mean already handles.
    use_peak = bool(len(bevs) and np.median([max((_strip_profile(b, x, ext=ext).std() for x in lane_lines_x),
                                                 default=0.0) for b in bevs]) < 3.0)
    times = times or [k / OUT_FPS for k in range(len(files))]
    poses = [np.eye(3)]
    diag = []
    prev = None
    last_v = np.zeros(2)  # m/s in the camera's road frame, for holding through gaps
    recent: list[float] = []  # accepted per-step speeds
    rejected: list[tuple[int, float]] = []  # (frame, speed) of consecutive good dash locks the gates threw out
    steps: list[np.ndarray] = []  # per-frame camera motion, kept so a re-lock can backfill earlier steps
    for k, g in enumerate(grays):
        mask_img = np.full(g.shape[:2], 255, np.uint8)
        if hood_row is not None:
            mask_img[hood_row:, :] = 0
        if static is not None:
            mask_img[static > 0] = 0
        for o in det["frames"][k]["objects"]:  # vehicles move on their own
            x1, y1, x2, y2 = map(int, o["bbox"])
            mask_img[max(0, y1):y2, max(0, x1):x2] = 0
        bev = bevs[k]
        bmask = cv2.warpPerspective(mask_img, Hb, size)
        # keep features well away from mask borders (hood cut line, vehicle boxes, warp edges):
        # those edges are strong corners that move with the camera or the other car, not the road
        er = max(3, int(round(1.5 * ppm))) | 1
        bmask = cv2.erode(bmask, np.ones((er, er), np.uint8))
        profiles = [_strip_profile(bev, x, peak=use_peak, ext=ext) for x in lane_lines_x]
        road_vis = float((bmask > 0).mean())
        if k == 0:
            road_vis0 = max(road_vis, 1e-3)
        if prev is not None:
            pb, pm, pprof = prev
            # --- optical flow
            M = None; n_in = 0
            p0 = cv2.goodFeaturesToTrack(pb, 300, 0.01, 6, mask=pm)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(pb, bev, p0, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 6:
                    a = _bev_to_m(p0.reshape(-1, 2)[good], ext); b = _bev_to_m(p1.reshape(-1, 2)[good], ext)
                    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=0.3)
                    n_in = int(inl.sum()) if inl is not None else 0
            flow_step = None
            if M is not None and n_in >= 15:
                # rigid only: rebuild the rotation from its angle so RANSAC's scale can't leak into poses
                ang = math.atan2(M[1, 0], M[0, 0])
                flow_step = np.linalg.inv(np.array([[math.cos(ang), -math.sin(ang), M[0, 2]],
                                                    [math.sin(ang), math.cos(ang), M[1, 2]], [0, 0, 1]]))
            if flow_step is None and v_h is not None and EGO_OPTS["image_ground_fallback"]:
                # Bird's-eye flow found too little texture: track ground corners in the original image
                # (sharper, no warp blur, the full width of the road) and map the matches to metres.
                flow_step, n_img = _image_ground_step(grays[k - 1], g, mask_img, H, v_h, hood_row)
                n_in = max(n_in, n_img)
            # --- lane-dash correlation
            dt = max(times[k] - times[k - 1], 1e-3)
            # dashed line = the strip with the most dash-edge energy in both frames (a solid line has ~none)
            cand = [(min(a_.std(), b_.std()), a_, b_) for a_, b_ in zip(pprof, profiles)]
            energy, pa_, pb_ = max(cand, key=lambda c: c[0]) if cand else (0.0, None, None)
            # search at most half a dash cycle: beyond that a shift is ambiguous with the next dash
            dy_dash, q = (_strip_shift(pa_, pb_, max_px=max(8, int(0.5 * DASH_CYCLE_M * ppm)), ppm=ppm)
                          if energy >= 3.0 else (0.0, 0.0))
            # a step implying an impossible jump in speed (>8 m/s change in one frame) is rejected
            if road_vis < 0.6 * road_vis0:
                # another vehicle fills most of the road view (typical in the last moments before a
                # crash): neither estimator sees enough road, so hold the previous velocity
                flow_step, q = None, 0.0
            raw_q, dash_v = q, dy_dash / dt
            relocked_now = False
            if k > 2:  # a step implying an impossible jump in speed (>8 m/s change in one frame)
                if flow_step is not None and np.linalg.norm(flow_step[:2, 2] / dt - last_v) > 8.0:
                    flow_step = None
                if q > 0 and abs(dash_v - np.linalg.norm(last_v)) > 8.0:
                    q = 0.0
            if len(recent) >= 5:  # >15 % off the recent typical speed: not physically plausible in 0.1 s
                med = float(np.median(recent[-10:]))
                if flow_step is not None and abs(np.linalg.norm(flow_step[:2, 2]) / dt - med) > 0.15 * max(med, 2.0):
                    flow_step = None
                if q > 0 and abs(dash_v - med) > 0.15 * max(med, 2.0):
                    q = 0.0
            # Re-lock. The gates above trust last_v/recent, so one bad early estimate (flow reading
            # ~0 m/s off a camera-fixed windscreen reflection) used to reject every correct dash reading
            # for the rest of the clip and keep a 60 mph car "parked". If clear dash locks keep getting
            # rejected while agreeing with each other, the reference is what's wrong: adopt the dash.
            if raw_q >= DASH_Q_OVERRIDE and q == 0.0 and road_vis >= 0.6 * road_vis0:
                # A run of rejected clear locks. One or two weak frames in between don't break it (real
                # dashes drop below the quality bar now and then); a longer gap does.
                if rejected and k - rejected[-1][0] > RELOCK_MAX_GAP:
                    rejected.clear()
                rejected.append((k, dash_v))
                win = [v for _, v in rejected[-RELOCK_N:]]
                rm = float(np.median(win))
                # Real dashes on real footage are noisy (±30 %), so don't demand they agree tightly; a
                # run that is consistently far from the reference is enough. On a re-uploaded highway clip
                # the tight 3-in-a-row rule never fired and the car stayed "parked" for 2.5 s, which made
                # the car ahead drive a semicircle in the replay.
                if len(win) == RELOCK_N and sum(abs(v - rm) <= RELOCK_TOL * max(rm, 2.0) for v in win) >= RELOCK_N - 1:
                    q, recent, last_v = raw_q, [], np.array([0.0, rm])
                    relocked_now = True
                    dy_dash = rm * dt  # adopt the run's median, not this one noisy reading
                    rejected.clear()
                    # the flow reading that passed the (wrong) gates this frame is the bad reference the
                    # re-lock is correcting; don't let it overwrite last_v again in the same frame
                    flow_step = None
            elif q > 0:  # a dash reading the gates accepted: the reference agrees with the dashes
                rejected.clear()
            # Peak-profile dashes are noisier: they fill gaps and can force the re-lock above, but don't
            # override a flow reading that passed the gates.
            if q >= DASH_Q_OVERRIDE and (flow_step is None or (
                    not use_peak and abs(flow_step[1, 2] - dy_dash) > 0.15 * max(dy_dash, 0.5))):
                step = np.eye(3); step[1, 2] = dy_dash
                if flow_step is not None:  # keep flow's small heading change, not its translation
                    step[:2, :2] = flow_step[:2, :2]
                src = "dash"
            elif flow_step is not None:
                step, src = flow_step, "flow"
            else:
                # hold the previous speed, forward only: a car doesn't slide sideways, and carrying a
                # stale lateral flow component through a long hold walked the car a lane over
                step = np.eye(3); step[1, 2] = last_v[1] * dt
                src = "hold"
            if src != "hold":
                last_v = step[:2, 2] / dt
                recent.append(float(np.linalg.norm(last_v)))
            # Rotation: the far-scenery yaw (independent of road texture) wins whenever it's available;
            # otherwise keep whatever rotation the step already has (flow's, or none for hold).
            if yaws[k] is not None:
                a_ = yaws[k]
                step = step.copy()
                step[:2, :2] = [[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]]
            # A turn needs sideways grip: yaw rate × speed is the lateral acceleration. Beyond a hard
            # swerve it's the other car filling the view, not the camera turning (34°/s at 50 mph was read
            # on a straight highway and curved both cars' paths 7 m off the road).
            yaw_ = math.atan2(step[1, 0], step[0, 0])
            if abs(yaw_) / dt * max(float(np.linalg.norm(step[:2, 2])) / dt, 1.0) > MAX_LAT_ACC:
                step = step.copy(); step[:2, :2] = np.eye(2)
            if relocked_now:
                # Backfill: the steps just before the re-lock carried the same wrong (near-zero) speed
                # the re-lock corrected; give them the re-locked speed so the car doesn't sit still
                # for the first half-second and make the other car look like it's accelerating.
                v_new = float(last_v[1])
                j = len(steps) - 1
                while j >= 0 and abs(steps[j][1, 2]) / diag[j]["dt"] < 0.5 * v_new:
                    steps[j] = steps[j].copy(); steps[j][:2, 2] = [0.0, v_new * diag[j]["dt"]]
                    diag[j]["dy"] = round(float(steps[j][1, 2]), 3); diag[j]["dx"] = 0.0
                    diag[j]["backfilled"] = True
                    j -= 1
            steps.append(step)
            poses.append(poses[-1] @ step)
            diag.append({"k": k, "dt": round(dt, 4), "road_vis": round(road_vis / road_vis0, 2),
                         "dash_raw_q": round(raw_q, 2), "dx": round(float(step[0, 2]), 3), "dy": round(float(step[1, 2]), 3),
                         "src": src, "inliers": n_in, "dash_dy": round(dy_dash, 2), "dash_q": round(q, 2),
                         "flow_dy": None if flow_step is None else round(float(flow_step[1, 2]), 3),
                         "yaw_deg": None if yaws[k] is None else round(math.degrees(yaws[k]), 2)})
        prev = (bev, bmask, profiles)
    if any(d.get("backfilled") for d in diag):  # rebuild the path from the corrected steps
        poses = [np.eye(3)]
        for s_ in steps:
            poses.append(poses[-1] @ s_)
    poses = _despike(poses, diag)
    if LANE_LOCK:
        poses = _lane_lock(poses, bevs, lane_lines_x, ext, diag)
    return poses, diag


def _lane_lines_at(bev: np.ndarray, lines_x: tuple[float, ...], ext: dict, ya: float, yb: float,
                   prev_off: float) -> float | None:
    """Mean sideways shift (m) of the clicked lane lines in one bird's-eye frame, over rows ya..yb m ahead.

    A painted line is a bright ridge across x; we look for it within ±1 m of where it was last frame.
    None if neither line stands out (occluded by the other car, worn paint, glare)."""
    ppm, (x0, _), (_, y1) = ext["ppm"], ext["x"], ext["y"]
    r0, r1 = int((y1 - yb) * ppm), int((y1 - ya) * ppm)
    if r1 - r0 < 3 or r0 < 0:
        return None
    p = bev[r0:r1].astype(np.float32).mean(axis=0)
    k = int(1.5 * ppm) | 1
    p = p - np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    found = []
    for lx in lines_x:
        exp = lx + prev_off
        lo, hi = int((exp - 1.0 - x0) * ppm), int((exp + 1.0 - x0) * ppm)
        seg = p[max(0, lo):max(0, hi)]
        if seg.size < 3:
            continue
        j = int(np.argmax(seg))
        if seg[j] >= LANE_PEAK_MIN:
            found.append((max(0, lo) + j) / ppm + x0 - lx)
    return float(np.mean(found)) if found else None


def _lane_lock(poses: np.ndarray, bevs: list[np.ndarray], lines_x: tuple[float, ...], ext: dict,
               diag: list[dict]) -> np.ndarray:
    """Replace the integrated sideways position and heading of the camera with what the lane lines say.

    Integrating small yaw errors walks the car (and the other car, placed relative to it) out of its
    lane: a 2–3° bias moved both across the centre line on a straight highway. The clicked lane lines
    are visible in most frames, so measure the camera's offset and heading relative to them directly.
    Frames where they're hidden keep the last measured offset (no invented lane change); forward
    distance still comes from the speed estimate. A real lane change shows as the lines sliding."""
    _, (_, _), (y0, y1) = ext["ppm"], ext["x"], ext["y"]
    near = (max(y0 + 1.0, 3.0), max(y0 + 1.0, 3.0) + 8.0)
    far = (near[1], min(y1 - 1.0, near[1] + 12.0))
    if far[1] - far[0] < 4.0:
        return poses
    yn, yf = sum(near) / 2, sum(far) / 2
    xc, th = [None] * len(bevs), [None] * len(bevs)
    off = 0.0
    for k, b in enumerate(bevs):
        on, of_ = _lane_lines_at(b, lines_x, ext, *near, off), _lane_lines_at(b, lines_x, ext, *far, off)
        if on is None or of_ is None:
            continue
        theta = -math.atan2(of_ - on, yf - yn)  # lines lean right in the image when the camera yaws left
        if abs(theta) > math.radians(10):  # not a lane line pair (or a hard swerve we can't follow)
            continue
        off = on
        xc[k], th[k] = -on - theta * yn, theta
    ok = [k for k in range(len(bevs)) if xc[k] is not None]
    if len(ok) < max(6, len(bevs) // 3):
        return poses
    # smooth (median of 5 measured frames), then fill gaps by holding/interpolating — no extrapolated drift
    xs = np.array([xc[k] for k in ok]); ts = np.array([th[k] for k in ok])
    xs_s = np.array([np.median(xs[max(0, i - 2):i + 3]) for i in range(len(xs))])
    ts_s = np.array([np.median(ts[max(0, i - 2):i + 3]) for i in range(len(ts))])
    ks = np.arange(len(bevs))
    xa, ta = np.interp(ks, ok, xs_s), np.interp(ks, ok, ts_s)
    out = poses.copy()
    for k in ks:
        c, s = math.cos(ta[k]), math.sin(ta[k])
        out[k] = np.array([[c, -s, xa[k]], [s, c, poses[k][1, 2]], [0, 0, 1]])
        if k >= 1 and k - 1 < len(diag):
            diag[k - 1]["lane_x"] = round(float(xa[k]), 2)
            diag[k - 1]["lane_deg"] = round(math.degrees(ta[k]), 2)
            diag[k - 1]["lane_measured"] = xc[k] is not None
    for d in diag:
        d["lane_locked"] = True
    return out


def _despike(poses: list[np.ndarray], diag: list[dict], win: int = 5, tol: float = 0.25) -> np.ndarray:
    """Replace per-step speeds/turn rates that are far from their rolling median, then re-integrate.

    One mis-locked dash correlation or a flow fit on the other car shows up as a single-frame
    speed spike; a car can't change speed by 25 % in 0.1 s.
    """
    if len(diag) < win:
        return np.array(poses)
    steps = [np.linalg.inv(poses[k]) @ poses[k + 1] for k in range(len(poses) - 1)]
    dts = np.array([d["dt"] for d in diag])
    v = np.array([s[:2, 2] for s in steps]) / dts[:, None]
    w = np.array([np.arctan2(s[1, 0], s[0, 0]) for s in steps]) / dts
    sp = np.linalg.norm(v, axis=1)
    h = win // 2
    out = [np.eye(3)]
    for k, s in enumerate(steps):
        lo, hi = max(0, k - h), min(len(steps), k + h + 1)
        med_sp, med_w = np.median(sp[lo:hi]), np.median(w[lo:hi])
        vk, wk = v[k], w[k]
        if abs(sp[k] - med_sp) > tol * max(med_sp, 1.0):
            vk = v[k] / max(sp[k], 1e-6) * med_sp if sp[k] > 1e-6 else np.median(v[lo:hi], axis=0)
            diag[k]["despiked"] = True
        if abs(wk - med_w) > 0.1:  # >0.1 rad/s off the local turn rate
            wk = med_w
        if abs(med_w) < 0.05:  # < 3°/s is flow noise, not steering; integrating it drifts the heading
            wk = 0.0
        a = wk * dts[k]
        st = np.array([[np.cos(a), -np.sin(a), vk[0] * dts[k]], [np.sin(a), np.cos(a), vk[1] * dts[k]], [0, 0, 1]])
        out.append(out[-1] @ st)
    return np.array(out)


# Typical roof height (m) by vehicle type, and its spread within the type, for box-height ranging.
VEH_HEIGHT_M = {"car": (1.6, 0.12), "pickup": (1.85, 0.07), "bus": (3.0, 0.12), "motorcycle": (1.3, 0.12)}
GROUND_PX_ERR = 2.2  # px: ±1 px box bottom plus ~±2 px camera pitch (bumps, braking) on the horizon
BOX_CAL_MAX_Z = 25.0  # m: sightings closer than this calibrate the box-height constant
# Off by default: on the one clip with ground truth (scripts/live_check.py) it made the other car's
# far ranges and speed worse (split A 19.5 % vs 12.2 % without; truth 9.7 %). On real footage with
# camera pitch it looked better (bcdfe860ce) but that's unverified. Turn on with CRASHLENS_BOX_RANGING=1.
BOX_RANGING = os.environ.get("CRASHLENS_BOX_RANGING", "0") == "1"
# px: YOLO boxes on tiny, far vehicles run a pixel or two loose. With ±1 px the box cue pulled a
# Blender car 60 m out to read 7 % short (the ground cue, exact on a render, was 2 % long).
BOX_PX_ERR = 2.0


def _box_ok(bbox: list[float], W: int, Himg: int) -> bool:
    x1, y1, x2, y2 = bbox
    return y2 - y1 >= 4 and y1 > 1 and y2 < Himg - 2 and x1 > 1 and x2 < W - 2  # big enough, not cut off


def _box_const(samples: list[tuple[float, list[float]]], vtype: str, f: float, W: int, Himg: int
               ) -> tuple[float, float, str]:
    """The clip's own range·height constant k (Z = k / h) for the other vehicle, and its relative std.

    Measured from the close sightings, where ground-contact range is accurate, so it absorbs the
    vehicle's real height and any focal-length error (a fixed height prior made a Blender clip worse).
    Falls back to f × typical height for the vehicle type when there are too few close sightings.
    """
    ks = []
    for Zg, bb in samples:
        if Zg <= BOX_CAL_MAX_Z and _box_ok(bb, W, Himg) and bb[3] - bb[1] >= 15:
            off = math.atan(((bb[0] + bb[2]) / 2 - W / 2) / f)
            ks.append(Zg * (bb[3] - bb[1]) * math.cos(off))
    if len(ks) >= 3:
        k = float(np.median(ks))
        return k, max(0.05, float(np.std(ks)) / k / math.sqrt(len(ks))), "measured"
    Hm, rel = VEH_HEIGHT_M.get(vtype, VEH_HEIGHT_M["car"])
    return f * Hm, rel, "prior"


def _fuse_box_range(Zg: float, bbox: list[float], kconst: tuple[float, float, str], f: float, W: int,
                    Himg: int) -> tuple[float, float]:
    """Range to the other vehicle from two independent cues, inverse-variance weighted.

    Ground contact (box bottom through the homography): error ∝ Z², and a couple of pixels of camera
    pitch near the horizon moves a 60 m car by tens of metres. Box height (Z = k/h, k from
    _box_const): error ∝ Z and independent of pitch. Far away the box cue dominates; close in, the
    ground cue does. On a real clip the ground cue alone read a car approaching the junction as
    parked in it, because its far ranges were ~40 % short.
    """
    if not BOX_RANGING:
        return Zg, rc.depth_sigma(max(Zg, 2.0), f)
    sg = rc.depth_sigma(max(Zg, 2.0), f, px=GROUND_PX_ERR)
    if not _box_ok(bbox, W, Himg):
        return Zg, sg
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    k, rel, _ = kconst
    off = math.atan(((x1 + x2) / 2 - W / 2) / f)  # k/h is depth along the axis; range is longer off-axis
    Zh = k / h / max(math.cos(off), 0.3)
    sh = Zh * math.hypot(BOX_PX_ERR / h, rel)
    wg, wh = 1 / sg ** 2, 1 / sh ** 2
    return (Zg * wg + Zh * wh) / (wg + wh), (wg + wh) ** -0.5


def reconstruct_case(cid: str, calib: dict, det: dict, frames_meta: dict) -> dict:
    W, Himg = frames_meta["width"], frames_meta["height"]
    lane_w = float(calib.get("lane_width_m", 3.66))
    img_pts = np.array(calib["points"], float)  # near-left, near-right, far-right, far-left
    hfov = float(calib.get("hfov_deg", 130.0))
    # The rectangle is assumed to be one dash + gap long. Cross-check that against the width: if the
    # length implies an impossible camera height, trust the width and rescale the length (see
    # rc.calibration_check) — otherwise every distance and speed is stretched by the same factor.
    check = rc.calibration_check(img_pts, lane_w, DASH_CYCLE_M, W, hfov)
    length_m = check.get("suggested_length_m", DASH_CYCLE_M)
    check["length_used_m"] = length_m
    road_pts = np.array([[0, 0], [lane_w, 0], [lane_w, length_m], [0, length_m]], float)
    H = rc.homography(img_pts, road_pts)
    f = float(check.get("f_px") or _camera_matrix(W, Himg, hfov)[1])
    cam_pt = rc.project(H, np.array([[W / 2, Himg - 1]]))[0] - [0, 1.0]  # hood edge → camera
    ftimes = [f["t"] for f in frames_meta["frames"]]  # true per-frame timestamps
    impact_i = int(np.argmin(np.abs(np.array(ftimes) - frames_meta["impact_t"])))
    other = pick_other(det, impact_i, calib.get("other_tap"))
    if other is None:
        raise RuntimeError("could not find the other vehicle — tap it on the impact frame")
    # Far away the other car is too small for full-frame YOLO, so its track starts late. Look for it
    # in earlier frames around where it first appeared (marked "backtracked"; stops when it's lost).
    from . import backtrack
    if backtrack.extend_backward(cid, det, other, VEHICLE_CLS):
        store.write_json(cid, "detections.json", det)
    ego_diag: list[dict] = []
    if calib.get("static_camera"):
        poses = np.array([np.eye(3)] * len(det["frames"]))
    else:
        # the hood sits below the road the user clicked: ignore rows past ~1/3 of the way down from it
        v_near = float(img_pts[:, 1].max())
        hood_row = int(v_near + 0.35 * (Himg - v_near))
        # The hood/dashboard barely changes over the clip; the road does. Measure where that starts —
        # clicks placed right at the hood edge otherwise let the dashboard into the motion estimate,
        # and its zero-motion corners out-vote the road.
        measured = _hood_row_from_video(cid, check.get("horizon_row")) if EGO_OPTS["measured_hood"] else None
        if measured is not None:
            hood_row = min(hood_row, measured)
        check["hood_row"] = hood_row
        # Two bird's-eye windows suit different roads: a long, wide one lets the lane-dash correlation
        # work (crisp dashes, little texture), a short one over just the visible lane lets ground flow
        # work (real asphalt texture, no dashes near the car). Use the long one when the road really has
        # trackable dashes (clean correlation locks in a good share of frames); otherwise the near one.
        # ("Which measured more" doesn't work: the long window happily measures ~0 m/s off static
        # structure on a dash-less road.)
        if check.get("ok") and EGO_OPTS.get("bev_auto", True):
            variants = [("long", rc.bev_extent(H, W, Himg, check["horizon_row"], hood_row, 0.05, "wide")),
                        ("near", rc.bev_extent(H, W, Himg, check["horizon_row"], hood_row, 0.15, "image"))]
        elif check.get("ok"):
            variants = [("fixed", rc.bev_extent(H, W, Himg, check["horizon_row"], hood_row,
                                                EGO_OPTS["bev_far_frac"], EGO_OPTS["bev_x"]))]
        else:
            variants = [("default", DEFAULT_EXT)]
        runs = []
        for name, ext in variants:
            p_, d_ = ego_motion(cid, H, det, lane_lines_x=(0.0, lane_w), hood_row=hood_row, times=ftimes,
                                ext=ext, v_h=check.get("horizon_row") if check.get("ok") else None, f=f)
            pre = [d for d in d_ if ftimes[d["k"]] <= frames_meta["impact_t"] - 0.3]
            locks = np.array([d["dash_dy"] / d["dt"] for d in pre if d.get("dash_raw_q", 0) >= DASH_Q_OVERRIDE])
            dash_frac = len(locks) / max(1, len(pre))
            med = float(np.median(locks)) if len(locks) else 0.0
            spread = (float(np.subtract(*np.percentile(locks, [75, 25]))) / med) if len(locks) >= 5 and med > 0.5 \
                else float("inf")
            runs.append((name, ext, p_, d_, dash_frac, spread))
        # Real dashes: frequent clean locks AND steady speeds (spread ≈ 0.4 on the verified Blender clip).
        # Cracks / texture: locks that swing between 0 and 25 m/s (spread ≈ 2.5 on a real dash-less clip).
        pick_i = 0
        if len(runs) == 2 and not (runs[0][4] >= 0.3 and runs[0][5] <= 1.0):
            pick_i = 1
        name, ext, poses, ego_diag = runs[pick_i][:4]
        check["bev"] = {"variant": name, "dash_frac_long": round(runs[0][4], 2),
                        "dash_spread_long": None if not math.isfinite(runs[0][5]) else round(runs[0][5], 2),
                        "x": [round(v, 1) for v in ext["x"]], "y": [round(v, 1) for v in ext["y"]],
                        "ppm": round(ext["ppm"], 1)}
        store.write_json(cid, "ego.json", ego_diag)  # per-frame estimator log (audit trail)
    t10, za, sa, btype = [], [], [], "car"
    obs = []  # (time, near contact point in world, unit viewing ray in world, Z)
    obs_sig = []  # range std (m) per observation
    raw = []  # (k, P, ground range Zg, unit ray, bbox) — fused after the box constant is known
    for k, fr in enumerate(det["frames"]):
        P = poses[min(k, len(poses) - 1)]
        cam_w = (P @ np.r_[cam_pt, 1])[:2]
        za.append(cam_w - (P[:2, :2] @ np.array([0, 0.75]))); sa.append(0.15)
        ob = [o for o in fr["objects"] if o["track"] == other]
        if not ob:
            continue
        o = ob[0]
        btype = {"truck": "pickup", "bus": "bus", "motorcycle": "motorcycle"}.get(o["cls"], "car")
        u = (o["bbox"][0] + o["bbox"][2]) / 2; v = o["bbox"][3]
        near = rc.project(H, np.array([[u, v]]))[0]
        ray = near - cam_pt; Zg = float(np.linalg.norm(ray))
        raw.append((k, P, Zg, ray / max(Zg, 1e-6), o["bbox"]))
    kconst = _box_const([(Zg, bb) for _, _, Zg, _, bb in raw], btype, f, W, Himg)
    check["box_range"] = {"k": round(kconst[0], 1), "rel_sd": round(kconst[1], 3), "source": kconst[2]}
    for k, P, Zg, ray_u, bb in raw:
        Z, sZ = _fuse_box_range(Zg, bb, kconst, f, W, Himg)
        near = cam_pt + ray_u * Z
        obs.append((ftimes[k], (P @ np.r_[near, 1])[:2], P[:2, :2] @ ray_u, Z))
        obs_sig.append(sZ)
    if len(obs) < 8:
        raise RuntimeError("the other vehicle is visible in too few frames")
    tb = np.array([o[0] for o in obs])
    sb = np.array([min(3.0, 0.12 + s) for s in obs_sig])
    L, Wd = rc.VEHICLE_SIZES.get(btype, rc.VEHICLE_SIZES["car"])
    t_imp = frames_meta["impact_t"]

    def centres(heading_at):
        # The bottom of the box is the vehicle's nearest edge. Push back along the ray by the
        # rectangle's half-extent in that direction: L/2 head-on, W/2 broadside.
        out = []
        for (ti, near_w, ray_w, _), h in zip(obs, heading_at):
            rel = h - np.arctan2(ray_w[1], ray_w[0])
            push = L / 2 * abs(np.cos(rel)) + Wd / 2 * abs(np.sin(rel))
            out.append(near_w + ray_w * push)
        return np.array(out)

    B = rc.build_track("B", btype, "#ef4444", tb, centres([np.arctan2(r[1], r[0]) for _, _, r, _ in obs]), sb, t_imp)
    zb = centres(np.interp(tb, B.t, B.heading))
    B = rc.build_track("B", btype, "#ef4444", tb, zb, sb, t_imp)
    # Velocity at the last sighting, straight from the last ~0.5 s of measured positions. These are the
    # closest (most accurate) readings; the smoother leans on the noisy far ones and under-reads it.
    recent_b = tb >= tb[-1] - 0.5
    b_vel = None
    if recent_b.sum() >= 3 and np.ptp(tb[recent_b]) > 0.15:
        b_vel = np.array([np.polyfit(tb[recent_b], zb[recent_b, j], 1)[0] for j in (0, 1)])
    tA = np.array(ftimes[: len(za)])
    A = rc.build_track("A", "car", "#3b82f6", tA, np.array(za), np.array(sa), t_imp)
    coasted = 0.0
    b_last_seen = float(B.t[-1])
    if B.t[-1] < t_imp + 0.6:  # lost at close range (fills the frame): coast through the impact window
        coasted = round(t_imp + 0.6 - B.t[-1], 2)
        if b_vel is not None and ego_diag and all(d.get("lane_locked") for d in ego_diag):
            # unseen from here on: carry it along the lane at its measured speed, with no sideways
            # motion it wasn't seen making (close-range box jitter read as a drift toward the next lane)
            b_vel = np.array([0.0, b_vel[1]])
        B = rc.coast(B, t_imp + 0.6, velocity=b_vel)
    impact = rc.find_impact(A, B)
    if impact is None:
        # The camera says a crash happened around t_imp (jolt / user-confirmed), but the tracks just miss:
        # small depth errors near the camera do that. Use the closest approach around t_imp, marked inferred.
        impact = rc.closest_approach(A, B, t_imp, window=0.5)
        if impact:
            # Where A was hit is better told by where the camera last saw B than by extrapolated outlines
            ia, ib = A.at(b_last_seen), B.at(b_last_seen)
            impact["contact"]["A"] = rc.contact_zone(A.s[ia, :2], A.heading[ia], B.s[ib, :2])
    if impact and coasted:
        # coasted motion after the crash is pure guesswork: end B at the impact
        keep = B.t <= max(impact["t"], b_last_seen) + 0.1
        B = rc.Track(id=B.id, type=B.type, color=B.color, t=B.t[keep], s=B.s[keep],
                     sigma=B.sigma[keep], heading=B.heading[keep])
    # After the impact the camera usually sees only the other car or chaos, so ego-motion is mostly
    # "hold": pre-crash speed carried forward, which drove A straight through B in the replay.
    # If that's what the post-impact segment is, end A where the measurements end (like B).
    if impact and ego_diag:
        post = [d for d in ego_diag if ftimes[d["k"]] > impact["t"]]
        hold_frac = sum(d["src"] == "hold" for d in post) / len(post) if post else 0.0
        keep = A.t <= impact["t"] + 0.1
        a_truncated = len(post) >= 3 and hold_frac > 0.5 and keep.sum() >= 3  # never leave A empty
        if a_truncated:
            A = rc.Track(id=A.id, type=A.type, color=A.color, t=A.t[keep], s=A.s[keep],
                         sigma=A.sigma[keep], heading=A.heading[keep])
        check["post_impact"] = {"hold_frac": round(hold_frac, 2), "a_truncated": bool(a_truncated)}
    # World frame: rotate so A's approach runs along +y (north), like the road templates, then put the
    # template's origin at the impact — the crash happens in the intersection, not wherever A started.
    lane_locked = bool(ego_diag) and all(d.get("lane_locked") for d in ego_diag)
    check["lane_locked"] = lane_locked
    early = (A.t >= max(A.t[0], t_imp - 3.0)) & (A.t <= t_imp - 1.0)
    if lane_locked:  # the lane lines already define "north": rotating by A's path would undo that
        rot = 0.0
    elif early.sum() >= 3:
        d = A.s[early, :2][-1] - A.s[early, :2][0]
        rot = math.pi / 2 - math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1.0 else 0.0
    else:
        rot = 0.0
    c, s_ = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s_], [s_, c]])
    for T in (A, B):
        T.s[:, :2] = T.s[:, :2] @ R.T
        T.s[:, 2:] = T.s[:, 2:] @ R.T
        T.heading = T.heading + rot
    if impact:
        impact["pos"] = list((R @ np.array(impact["pos"])).round(2))
    origin = np.array(impact["pos"]) if impact else A.s[A.at(t_imp), :2]
    # A drives in the inner northbound lane (x = lane_w/2); the template's centre sits at the impact
    # (lane-locked: the clicked lane is world x ∈ [0, lane_w] already, so keep A's measured place in it)
    shift = np.array([0.0 if lane_locked else A.s[A.at(max(0, t_imp - 2)), 0] - lane_w / 2, origin[1]])
    for T in (A, B):
        T.s[:, :2] -= shift
    if impact:
        impact["pos"] = [round(float(impact["pos"][0] - shift[0]), 2), round(float(impact["pos"][1] - shift[1]), 2)]
    loc = calib.get("location_type", "multi_lane_road")
    tmpl = {"intersection_signal": "intersection_signal_4way", "intersection_stop": "intersection_stop_4way",
            "parking_lot": "parking_aisle",
            # all lanes one direction (divided highway): the clicked lane is the leftmost, x ∈ [0, lane_w]
            "divided_highway": "straight_oneway"}.get(loc, "straight_multilane")
    road = {"template": tmpl, "lanes_per_dir": int(calib.get("lanes_per_dir", 2)), "lane_width_m": lane_w,
            "speed_limit_mph": calib.get("speed_limit_mph")}
    # Signals and signs come from what the camera actually saw, never from the location dropdown alone:
    # a stop sign assumed from the form became a "fact" that A ran it, and decided the case.
    seen = {o["cls"] for fr in det["frames"] for o in fr["objects"]}
    signals = _signals_from_detections(det, road) if tmpl.startswith("intersection") else []
    if tmpl.startswith("intersection") and "traffic_light" in seen and tmpl == "intersection_stop_4way":
        road["template"] = tmpl = "intersection_signal_4way"   # lights in view: it's a signalised junction
    signs = []
    if tmpl == "intersection_stop_4way" and "stop_sign" in seen:
        half = road["lanes_per_dir"] * lane_w
        signs = [{"id": "S1", "pos": [half + 1.0, -half - 1.0], "faces": "A", "approach_heading": math.pi / 2,
                  "stop_line": -half - 1.0}]
    case = store.read_json(cid, "case.json")
    return rc.scene_json(case["evidence_sha256"], road, signals, signs, [A, B], impact,
                         {"location_type": loc, "clip_offset": frames_meta["clip_offset"], "other_track": other,
                          "calibration": "homography" if calib.get("points") else "pinhole",
                          "ego": "static" if calib.get("static_camera") else "flow+dash",
                          "b_coasted_s": coasted,
                          "calibration_check": check,
                          "ego_sources": {s: sum(d["src"] == s for d in ego_diag) for s in ("flow", "dash", "hold")},
                          "dash_check": _dash_check(ego_diag)})


def _dash_check(diag: list[dict]) -> dict:
    """Spec §3.3 sanity check: lane-dash speed vs optical-flow speed; >15 % apart → low confidence."""
    pairs = [(d["dash_dy"] / d["dt"], d["flow_dy"] / d["dt"]) for d in diag
             if d["dash_q"] >= 0.6 and d["flow_dy"] is not None]
    if not pairs:
        return {"available": False}
    dash = float(np.median([p[0] for p in pairs]))
    flow = float(np.median([p[1] for p in pairs]))
    diff = abs(dash - flow) / max(dash, 0.1)
    return {"available": True, "dash_mps": round(dash, 2), "flow_mps": round(flow, 2),
            "diff_pct": round(diff * 100, 1), "low_confidence": diff > 0.15}


def _signals_from_detections(det: dict, road: dict) -> list[dict]:
    states = []
    for fr in det["frames"]:
        ls = [o for o in fr["objects"] if o["cls"] == "traffic_light" and o.get("state", "unknown") != "unknown"]
        if ls:
            biggest = max(ls, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
            s = biggest["state"]
            if not states or states[-1]["s"] != s:
                states.append({"t": fr["t"], "s": s})
    if not states:
        return []
    half = road["lanes_per_dir"] * road["lane_width_m"]
    return [{"id": "L1", "pos": [half + 1.2, half + 1.5], "faces": "A", "approach_heading": math.pi / 2,
             "stop_line": -half - 1.0, "protected_left": False, "states": states}]


def _choose_calibration(cid: str, fm: dict, calib: dict) -> dict:
    """Lane rectangle and focal length to reconstruct with: measured from the video where possible.

    The four clicks are easy to get wrong (both lanes plus the shoulder, a rectangle up to the horizon)
    and the HFOV is a guess; either rescales every distance and speed. Priority:
      lane lines + dash spacing measured from the video (autocal)  >  clicks that pass the width/length
      consistency check (and the HFOV they imply)  >  auto-detected lane with the assumed HFOV
      >  the clicks as given.
    The user's clicks are kept in "points_user"; autocal.json records what was found and why.
    """
    from . import autocal
    if not calib.get("points") or not AUTOCAL:
        return calib
    W = fm["width"]
    lane_w = float(calib.get("lane_width_m", 3.66))
    ac = autocal.calibrate(cid, fm, calib, lane_w, DASH_CYCLE_M)
    f_dash = ac["f_px"] if ac.get("ok") and ac.get("method") == "lines+dashes" else None
    hfov_eval = ac["hfov_deg"] if f_dash else float(calib.get("hfov_deg", 130.0))
    pts = np.array(calib["points"], float)
    chk = rc.calibration_check(pts, lane_w, DASH_CYCLE_M, W, hfov_eval)
    # The clicked side lines bound ONE lane (camera height from their width is plausible): clicks on the
    # paint are more precise than detected lines, so keep them. Lines spanning two lanes plus the
    # shoulder fail this (the height comes out at half a dashcam's).
    lines_ok = bool(chk.get("ok") and rc_h_ok(chk.get("h_lat_m")))
    if lines_ok and ac.get("vp"):
        # Judge the clicks against the horizon measured in the video, not the one they imply: lines
        # clicked on the guardrail and the far edge meet at their own (wrong) vanishing point, and
        # against that a 2½-lane width can pass as one plausible lane.
        vp = np.array(ac["vp"], float); H_img = fm["height"]
        def off_vp(a, b):
            d = b - a
            return abs(d[0] * (vp[1] - a[1]) - d[1] * (vp[0] - a[0])) / max(np.hypot(*d), 1e-6)
        near_v = float((pts[0, 1] + pts[1, 1]) / 2)
        h_video = lane_w * (near_v - vp[1]) / max(abs(pts[1, 0] - pts[0, 0]), 1e-6)
        tol = max(6.0, 0.025 * H_img)
        lines_ok = off_vp(pts[0], pts[3]) <= tol and off_vp(pts[1], pts[2]) <= tol and rc_h_ok(h_video)
        chk["clicks_vs_video"] = {"left_line_off_vp_px": round(off_vp(pts[0], pts[3]), 1),
                                  "right_line_off_vp_px": round(off_vp(pts[1], pts[2]), 1),
                                  "cam_h_from_video_horizon_m": round(h_video, 2), "lines_ok": bool(lines_ok)}
    hfov_clicks = chk.get("hfov_from_clicks") or 0.0
    out = {**calib, "points_user": calib["points"]}
    if lines_ok and f_dash and chk.get("consistent"):
        out.update(hfov_deg=hfov_eval, calibration_source="clicks, confirmed by the dash spacing in the video")
    elif lines_ok and f_dash:
        # right lane lines, wrong far corners: keep the lines, put the far edge one dash cycle out
        out.update(points=_rebuild_far_edge(pts, chk["horizon_row"], f_dash, chk["h_lat_m"], DASH_CYCLE_M),
                   hfov_deg=hfov_eval, calibration_source="clicked lane lines; length from the dash spacing")
    elif lines_ok and 45 <= hfov_clicks <= 150:
        out.update(hfov_deg=hfov_clicks, calibration_source="clicks (no dashes to check against; lens from clicks)")
    elif ac.get("ok"):
        out.update(points=ac["points"], calibration_source="lane detected in the video"
                   + ("" if f_dash else "; assumed lens (low confidence)"))
        if f_dash:
            out["hfov_deg"] = ac["hfov_deg"]
    else:
        out["calibration_source"] = "clicks as given (could not be verified)"
    store.write_json(cid, "autocal.json", {**ac, "clicks_check": chk, "chosen": out["calibration_source"]})
    return out


def _rebuild_far_edge(pts: np.ndarray, v_h: float, f: float, cam_h: float, cycle_m: float) -> list:
    """Same two lane lines, far edge moved to exactly one dash cycle beyond the near edge."""
    v_near = float((pts[0, 1] + pts[1, 1]) / 2)
    z_near = f * cam_h / (v_near - v_h)
    v_far = v_h + f * cam_h / (z_near + cycle_m)
    def u_on(a, b, v):  # point on the line a→b at row v
        return float(a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
    return [[u_on(pts[0], pts[3], v_near), v_near], [u_on(pts[1], pts[2], v_near), v_near],
            [u_on(pts[1], pts[2], v_far), v_far], [u_on(pts[0], pts[3], v_far), v_far]]


def rc_h_ok(h) -> bool:
    return h is not None and 0.8 <= float(h) <= 2.4


def run(cid: str, calib: dict) -> None:
    """Background job started by POST /cases/{id}/calibration."""
    try:
        case = store.read_json(cid, "case.json")
        path = store.case_dir(cid) / case["video_file"]
        fm = store.read_json(cid, "frames.json")
        k1 = float(calib.get("k1", fm.get("k1", -0.25)))
        new_imp = calib.get("impact_t")
        moved = new_imp is not None and abs(new_imp - fm["impact_t"]) > 0.05
        if moved or abs(k1 - fm.get("k1", -0.25)) > 1e-3:
            store.set_status(cid, "ingesting", 18)
            t_abs = fm["clip_offset"] + float(new_imp if new_imp is not None else fm["impact_t"])
            fm = extract_frames(cid, path, t_abs, k1)
            # impact_t was relative to the old window; re-express it in the new one and save, so a
            # re-run doesn't treat it as a new choice and slide the window again
            calib = {**calib, "impact_t": fm["impact_t"], "k1": k1}
            store.write_json(cid, "calibration.json", calib)
        calib = _choose_calibration(cid, fm, calib)
        store.set_status(cid, "detecting", 25)
        det = perceive(cid)
        store.set_status(cid, "reconstructing", 65)
        scene = reconstruct_case(cid, {**case["form"], **calib}, det, fm)
        store.write_json(cid, "scene.json", scene)
        store.set_status(cid, "judging", 90)
        fs = factmod.extract(scene, case["form"])
        store.write_json(cid, "facts.json", fs)
        store.set_status(cid, "cross-checking", 95)
        from . import crosscheck  # informational only: never changes facts or the split
        crosscheck.run(cid)
        store.set_status(cid, "done", 100, status="done")
    except Exception as e:
        traceback.print_exc()
        store.set_status(cid, "error", 0, status="error", error=str(e))
