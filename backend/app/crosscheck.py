"""Independent visual sanity check of the reconstruction (Nemotron VL looks at the frames).

The fault engine never uses this. A vision-language model is shown a few frames around the impact
and asked what kind of crash it sees and whether the dashcam car hit or was hit. If that contradicts
the reconstruction, the case is flagged for review: a wrong ego speed once turned an obvious
rear-end into an "angle" crash with the dashcam car parked, and a glance at the frames catches that.
"""
from __future__ import annotations

import base64
import json
import os
import re

from . import store
from .narrative import BASE_URL, _api_key

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
OFFSETS_S = (-3.0, -1.5, -0.5, 0.4)  # frames sent, relative to the impact
MIN_CONF = 0.6  # below this the model's view is shown but doesn't raise a flag
PRE_HOLD_MAX = 0.3  # pre-impact share of ego steps that were guessed ("hold") before we flag it

CRASH_TYPES = ("rear_end", "angle", "head_on", "sideswipe", "none", "unclear")
ROLES = ("striking", "struck", "unclear")
DIRECTIONS = ("same", "opposite", "crossing", "unclear")
LIGHTS = ("red", "yellow", "green", "none_visible", "unclear")
# Frames (plates, faces) go to a third-party API. CRASHLENS_CROSSCHECK=0 turns that off entirely.
ENABLED = os.environ.get("CRASHLENS_CROSSCHECK", "1") != "0"

PROMPT = (
    "These are dashcam frames in time order; the last one is just after the moment of impact. "
    "The camera is mounted in the dashcam car. Answer with JSON only, no prose, exactly these keys:\n"
    '{"crash_type": one of ["rear_end","angle","head_on","sideswipe","none","unclear"], '
    '"dashcam_vehicle": one of ["striking","struck","unclear"], '
    '"other_vehicle_direction": one of ["same","opposite","crossing","unclear"], '
    '"traffic_light_for_dashcam": one of ["red","yellow","green","none_visible","unclear"], '
    '"confidence": number 0-1, "reason": "one short sentence"}\n'
    "rear_end = one vehicle drives into the back of another going the same way. A left turn across "
    "the dashcam car's path is \"angle\". Say \"unclear\" rather than guess."
)


def _frames(cid: str) -> list[tuple[float, str]]:
    """(time rel. to impact, base64 jpeg) for the frames nearest OFFSETS_S, downscaled to 640 px."""
    import cv2
    fm = store.read_json(cid, "frames.json")
    if not fm or not fm.get("frames"):
        return []
    t_imp = fm["impact_t"]
    out, used = [], set()
    for off in OFFSETS_S:
        fr = min(fm["frames"], key=lambda f: abs(f["t"] - (t_imp + off)))
        if fr["i"] in used:
            continue
        used.add(fr["i"])
        img = cv2.imread(str(store.case_dir(cid) / "frames" / f"{fr['i']:03d}.jpg"))
        if img is None:
            continue
        img = cv2.resize(img, (640, max(1, int(img.shape[0] * 640 / img.shape[1]))))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append((round(fr["t"] - t_imp, 2), base64.b64encode(buf.tobytes()).decode()))
    return out


def _parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or d.get("crash_type") not in CRASH_TYPES or d.get("dashcam_vehicle") not in ROLES:
        return None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    # Untrusted model output: keep only known keys, enum-checked, and a clipped reason.
    direction, light = d.get("other_vehicle_direction"), d.get("traffic_light_for_dashcam")
    return {"crash_type": d["crash_type"], "dashcam_vehicle": d["dashcam_vehicle"],
            "other_vehicle_direction": direction if direction in DIRECTIONS else "unclear",
            "traffic_light_for_dashcam": light if light in LIGHTS else "unclear",
            "confidence": conf, "reason": str(d.get("reason", ""))[:200]}


def pipeline_view(scene: dict) -> dict:
    """The reconstruction's answer to the same questions. The dashcam car is always A."""
    imp = scene.get("impact")
    if not imp:
        return {"crash_type": "none", "dashcam_vehicle": "unclear"}
    ca = (imp.get("contact") or {}).get("A", "")
    role = "striking" if ca.startswith("front") else "struck" if ca else "unclear"
    return {"crash_type": imp.get("type", "unclear"), "dashcam_vehicle": role}


def compare(vlm: dict, pipe: dict) -> list[str]:
    """Disagreements worth a human look. Only confident, definite VLM answers count."""
    if vlm["confidence"] < MIN_CONF:
        return []
    out = []
    if vlm["crash_type"] != "unclear" and vlm["crash_type"] != pipe["crash_type"]:
        out.append(f"the frames look like {vlm['crash_type'].replace('_', ' ')}, "
                   f"but the reconstruction says {pipe['crash_type'].replace('_', ' ')}")
    if "unclear" not in (vlm["dashcam_vehicle"], pipe["dashcam_vehicle"]) and vlm["dashcam_vehicle"] != pipe["dashcam_vehicle"]:
        out.append(f"the frames suggest the dashcam car was {vlm['dashcam_vehicle']}, "
                   f"but the reconstruction has it {pipe['dashcam_vehicle']}")
    return out


def run(cid: str) -> dict:
    """Compute, store and return crosscheck.json. Never raises; failures are recorded, not hidden."""
    scene = store.read_json(cid, "scene.json") or {}
    pipe = pipeline_view(scene)
    out = {"model": MODEL, "pipeline": pipe, "vlm": None, "disagreements": [], "needs_review": False}
    key = _api_key() if ENABLED else None
    frames = _frames(cid) if key else []
    if not key or not frames:
        out["status"] = "unavailable"
        out["error"] = ("disabled (CRASHLENS_CROSSCHECK=0)" if not ENABLED
                        else "no NVIDIA_API_KEY" if not key else "no frames")
    else:
        out["frames_t"] = [t for t, _ in frames]
        try:
            from openai import OpenAI
            parts = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for _, b in frames]
            parts.append({"type": "text", "text": PROMPT})
            client = OpenAI(base_url=BASE_URL, api_key=key, timeout=90)
            raw, finish = "", None
            for _ in range(2):  # a reasoning model sometimes spends the whole budget thinking: retry once
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.1, max_tokens=4096,
                    messages=[{"role": "system", "content": "detailed thinking off"},
                              {"role": "user", "content": parts}])
                raw, finish = r.choices[0].message.content or "", r.choices[0].finish_reason
                if raw.strip():
                    break
            vlm = _parse(raw)
            if vlm is None:  # raw text not stored: it can echo plates/scene details into a synced folder
                out.update(status="error", error=f"unparseable model answer (finish_reason={finish})")
            else:
                out.update(status="ok", vlm=vlm, disagreements=compare(vlm, pipe))
                out["needs_review"] = bool(out["disagreements"])
        except Exception as e:  # network / auth / model errors: recorded and shown, never block the verdict
            # type + HTTP status only: the message body could echo request details to the browser
            code = getattr(e, "status_code", None)
            out.update(status="error", error=type(e).__name__ + (f" (HTTP {code})" if code else ""))
            print(f"[crosscheck] {cid}: {type(e).__name__} {code or ''}".rstrip())
    store.write_json(cid, "crosscheck.json", out)
    return out


def review(cid: str) -> dict:
    """Everything that should stop a user trusting the split as-is: the VLM disagreement plus the
    reconstruction's own quality signals, which used to be written to scene.json and never read."""
    reasons: list[str] = []  # blocking: the result may be wrong
    quality: list[str] = []  # measurement weaknesses; blocking only if the visual check didn't confirm
    cc = store.read_json(cid, "crosscheck.json")
    if cc and cc.get("disagreements"):
        reasons += [f"Visual check ({cc['model'].split('/')[-1]}): {d}." for d in cc["disagreements"]]
    vlm = (cc or {}).get("vlm") or {}
    confirmed = bool(cc and cc.get("status") == "ok" and not cc.get("disagreements")
                     and vlm.get("confidence", 0) >= MIN_CONF and vlm.get("crash_type") != "unclear")
    scene = store.read_json(cid, "scene.json") or {}
    meta = scene.get("meta") or {}
    imp = scene.get("impact")
    if scene and not imp:
        reasons.append("No impact could be reconstructed from the video.")
    elif imp and imp.get("inferred"):
        reasons.append("The impact is inferred from the closest approach, not an observed overlap.")
    ego = store.read_json(cid, "ego.json")
    fm = store.read_json(cid, "frames.json")
    if ego and fm and imp:
        ts = [f["t"] for f in fm["frames"]]
        pre = [d for d in ego if d["k"] < len(ts) and ts[d["k"]] <= imp["t"]]
        hold = sum(d["src"] == "hold" for d in pre) / max(len(pre), 1)
        if pre and hold > PRE_HOLD_MAX:
            quality.append(f"The dashcam car's own speed was measured in only {100 - round(hold * 100)}% of "
                           f"the frames before impact; the rest were carried forward.")
    if (meta.get("dash_check") or {}).get("low_confidence"):
        quality.append("The two speed estimators (road texture and lane dashes) disagree, so speeds are less certain.")
    if not confirmed:
        reasons += quality
    return {"needs_review": bool(reasons), "reasons": reasons, "notes": quality if confirmed else [],
            "visually_confirmed": confirmed,
            "crosscheck": {k: cc.get(k) for k in ("status", "model", "vlm", "error")} if cc else None}
