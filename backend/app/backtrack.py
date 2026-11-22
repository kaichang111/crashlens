"""Extend the other vehicle's track backwards in time, before the tracker first picked it up.

Far away, the other car is a few pixels tall and YOLO on the full frame misses it, so the track
starts late and the replay had it pop into existence mid-intersection. Walking back frame by frame
from its first detection, we crop a small window around where it was, upscale the crop, and look
for a vehicle there. Each hit becomes a real (marked) detection; we stop at the first frames where
nothing plausible is found, rather than inventing an approach path.
"""
from __future__ import annotations

import numpy as np

from . import store

CROP_MIN = 48      # px, smallest search window (source-frame pixels)
CROP_SCALE = 5.0   # window = this × the expected box size
UPSCALE_TO = 640   # crops are resized to this before detection
CONF = 0.15        # low: the vehicle is tiny, and we only accept boxes near the predicted spot
MAX_MISSES = 2     # consecutive frames without a match before giving up


def _first_seen(det: dict, track: int) -> tuple[int, dict] | None:
    for k, fr in enumerate(det["frames"]):
        for o in fr["objects"]:
            if o["track"] == track:
                return k, o
    return None


def extend_backward(cid: str, det: dict, track: int, vehicle_cls: set[str]) -> int:
    """Add backtracked detections for `track` to det in place. Returns how many frames were added."""
    import cv2
    from ultralytics import YOLO

    first = _first_seen(det, track)
    if first is None or first[0] == 0:
        return 0
    k0, o0 = first
    files = sorted((store.case_dir(cid) / "frames").glob("*.jpg"))
    model = YOLO("yolo11s.pt")
    box = np.array(o0["bbox"], float)
    # recent per-frame motion of the box (centre and size), from its first few real detections
    later = [o["bbox"] for fr in det["frames"][k0:k0 + 4] for o in fr["objects"] if o["track"] == track]
    step = (np.array(later[0]) - np.array(later[-1])) / max(1, len(later) - 1) if len(later) >= 2 else np.zeros(4)
    added, misses = 0, 0
    for k in range(k0 - 1, -1, -1):
        img = cv2.imread(str(files[k])) if k < len(files) else None
        if img is None:
            break
        H, W = img.shape[:2]
        pred = box + step  # going back in time: farther away, so usually a little smaller
        bw, bh = max(2.0, pred[2] - pred[0]), max(2.0, pred[3] - pred[1])
        cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        half = max(CROP_MIN, CROP_SCALE * max(bw, bh)) / 2
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            break
        s = UPSCALE_TO / max(crop.shape[:2])
        r = model.predict(cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC),
                          conf=CONF, verbose=False, imgsz=UPSCALE_TO)[0]
        best, best_d = None, None
        names = r.names
        for b, c, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if names[int(c)] not in vehicle_cls:
                continue
            bb = np.array(b) / s + [x0, y0, x0, y0]
            ww, hh = bb[2] - bb[0], bb[3] - bb[1]
            # plausible: near the predicted spot, and not suddenly much bigger/smaller
            d = np.hypot((bb[0] + bb[2]) / 2 - cx, (bb[1] + bb[3]) / 2 - cy)
            if d > max(6.0, 1.5 * max(bw, bh)) or not (0.5 < hh / bh < 1.6) or not (0.4 < ww / bw < 2.0):
                continue
            if best_d is None or d < best_d:
                best, best_d = (bb, names[int(c)], sc), d
        if best is None:
            misses += 1
            if misses >= MAX_MISSES:
                break
            box = pred  # carry the prediction through one miss
            continue
        misses = 0
        bb, cls, sc = best
        step = 0.6 * step + 0.4 * (bb - box)  # box change per frame, going backwards
        box = bb
        det["frames"][k]["objects"].append({"track": track, "cls": o0["cls"], "bbox": [round(float(x), 1) for x in bb],
                                            "conf": round(float(sc), 3), "backtracked": True})
        added += 1
    return added
