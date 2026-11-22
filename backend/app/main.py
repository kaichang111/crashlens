"""CrashLens API (FastAPI). Run:  uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import calibrate as calib_mod
from . import crosscheck, engine, facts as factmod, narrative, pipeline, scenarios, store

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
WEIGHTS = store.DATA / "weights.json"
MAX_UPLOAD = 300 * 1024 * 1024

@asynccontextmanager
async def lifespan(_app):
    seed_demo_cases()
    yield


app = FastAPI(title="CrashLens API", version="0.1.0", lifespan=lifespan,
              description="Dashcam clip → 3D replay → Texas-statute fault split you can dispute.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------ helpers
def _need(cid: str) -> dict:
    c = store.get_case(cid)
    if not c:
        raise HTTPException(404, "case not found")
    return c


def _weights() -> dict | None:
    return json.loads(WEIGHTS.read_text()) if WEIGHTS.exists() else None


def _verdict(cid: str, with_narrative: bool = True) -> dict:
    case = store.read_json(cid, "case.json")
    fs = store.read_json(cid, "facts.json")
    if fs is None:
        raise HTTPException(409, "case not processed yet")
    ds = store.disputes(cid)
    your = case["form"].get("your_vehicle", "A")
    v = engine.score(fs, ds, _weights(), claimant=your)
    v["case_id"] = cid
    v["your_vehicle"] = your
    v["evidence_sha256"] = case["evidence_sha256"]
    v["received_at"] = case["received_at"]
    v["location_type"] = case["form"].get("location_type")
    v["disputes"] = ds
    v["claims"] = factmod.reconcile_claims(
        [dict(f) for f in fs], factmod.claims_from_description(case["form"].get("description", ""), your))
    v["calibrated_weights"] = bool(_weights())
    v["review"] = crosscheck.review(cid)
    if with_narrative:
        v["narrative"] = narrative.narrate(v)
    return v


def _materialize_scenario(cid: str, name: str, sha: str, received: str, form: dict) -> None:
    sc = scenarios.SCENARIOS[name]()
    sc["evidence_sha256"] = sha
    form = {**sc["meta"], **{k: v for k, v in form.items() if v not in (None, "")}}
    store.create_case(cid, sha, received, form, cached=True)
    case = store.read_json(cid, "case.json")
    vid = SAMPLES / f"{name}.mp4"
    if vid.exists():
        case["sample_video"] = name
    store.write_json(cid, "case.json", case)
    store.write_json(cid, "scene.json", sc)
    store.write_json(cid, "facts.json", factmod.extract(sc, form))
    store.set_status(cid, "done", 100, status="done")


def _video_path(cid: str) -> Path | None:
    case = store.read_json(cid, "case.json") or {}
    if case.get("sample_video"):
        p = SAMPLES / f"{case['sample_video']}.mp4"
    elif case.get("video_file"):
        p = store.case_dir(cid) / case["video_file"]
    else:
        return None
    return p if p.exists() else None


def seed_demo_cases() -> None:
    """Pre-built demo cases: the spec's 'cached results for every demo clip' backup."""
    for name in scenarios.SCENARIOS:
        cid = f"demo-{name.replace('_', '-')}"
        if store.get_case(cid) and (store.CASES / cid / "facts.json").exists():
            continue
        vid = SAMPLES / f"{name}.mp4"
        sha = hashlib.sha256(vid.read_bytes()).hexdigest() if vid.exists() else hashlib.sha256(name.encode()).hexdigest()
        _materialize_scenario(cid, name, sha, store.now(), {})


# ------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"ok": True, "capabilities": pipeline.capabilities(), "samples": sorted(scenarios.SCENARIOS)}


@app.get("/cases")
def list_cases():
    return store.list_cases()


@app.post("/cases")
async def create_case(
    background: BackgroundTasks,
    video: UploadFile | None = File(None),
    location_type: str = Form("intersection_signal"),
    your_vehicle: str = Form("A"),
    description: str = Form(""),
    speed_limit_mph: float | None = Form(None),
    weather: str = Form(""),
    surface: str = Form("dry"),
    cached: str | None = Query(None, description="hidden demo switch: left_turn | rear_end | 1"),
):
    """Upload video + form. Hash first, before anything touches the bytes (spec §1.2)."""
    raw = await video.read() if video is not None else b""
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "video too large")
    received = store.now()
    form = {"location_type": location_type, "your_vehicle": your_vehicle, "description": description,
            "speed_limit_mph": speed_limit_mph, "weather": weather, "surface": surface}
    cid = uuid.uuid4().hex[:10]
    if cached:
        name = cached if cached in scenarios.SCENARIOS else (
            "rear_end" if location_type == "multi_lane_road" else "left_turn")
        sample = SAMPLES / f"{name}.mp4"
        sha = hashlib.sha256(raw or (sample.read_bytes() if sample.exists() else name.encode())).hexdigest()
        _materialize_scenario(cid, name, sha, received, {k: v for k, v in form.items() if k != "description" or v})
        return {"id": cid, "evidence_sha256": sha, "received_at": received, "cached": True}
    if not raw:
        raise HTTPException(400, "video file required")
    if not pipeline.capabilities()["cv2"]:
        raise HTTPException(503, "OpenCV not installed on the server; install requirements-cv.txt or use ?cached=1")
    sha = hashlib.sha256(raw).hexdigest()
    ext = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    store.create_case(cid, sha, received, form, cached=False)
    (store.case_dir(cid) / f"original{ext}").write_bytes(raw)
    case = store.read_json(cid, "case.json")
    case["video_file"] = f"original{ext}"
    case["video_name"] = video.filename
    store.write_json(cid, "case.json", case)
    background.add_task(pipeline.ingest, cid)
    return {"id": cid, "evidence_sha256": sha, "received_at": received, "cached": False}


@app.get("/cases/{cid}")
def get_case(cid: str):
    c = _need(cid)
    return {**c, **(store.read_json(cid, "case.json") or {}), "frames": store.read_json(cid, "frames.json")}


class Calibration(BaseModel):
    points: list[list[float]] = Field(..., min_length=4, max_length=4,
                                      description="near-left, near-right, far-right, far-left (frame-0 pixels)")
    lane_width_m: float = 3.66
    other_tap: list[float] | None = None
    impact_t: float | None = None
    static_camera: bool = False
    hfov_deg: float = 130.0
    lanes_per_dir: int = 2
    k1: float = Field(-0.25, ge=-0.6, le=0.2, description="barrel-distortion correction (spec §1.5)")


@app.post("/cases/{cid}/calibration")
def post_calibration(cid: str, body: Calibration):
    c = _need(cid)
    if c["cached"]:
        return {"ok": True, "cached": True}
    if not pipeline.capabilities()["live_pipeline"]:
        raise HTTPException(503, "ultralytics/opencv not installed; install requirements-cv.txt or use ?cached=1")
    store.write_json(cid, "calibration.json", body.model_dump())
    store.set_status(cid, "queued", 20)
    threading.Thread(target=pipeline.run, args=(cid, body.model_dump()), daemon=True).start()
    return {"ok": True}


@app.get("/cases/{cid}/status")
def status(cid: str):
    c = _need(cid)
    return {"status": c["status"], "stage": c["stage"], "percent": c["percent"], "error": c["error"]}


@app.get("/cases/{cid}/frames")
def frames_meta(cid: str):
    _need(cid)
    return store.read_json(cid, "frames.json") or {"frames": []}


@app.get("/cases/{cid}/frames/{i}.jpg")
def frame(cid: str, i: int):
    _need(cid)  # known case ids only: a raw cid like ".." (percent-encoded) would escape CASES
    p = store.CASES / cid / "frames" / f"{i:03d}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/cases/{cid}/preview/{i}.jpg")
def preview(cid: str, i: int, k1: float = Query(-0.25, ge=-0.6, le=0.2)):
    _need(cid)
    from fastapi.responses import Response
    try:
        return Response(pipeline.preview_frame(cid, i, k1), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(404)


@app.api_route("/cases/{cid}/video", methods=["GET", "HEAD"])
def video(cid: str):
    p = _video_path(cid)
    if p is None:
        raise HTTPException(404, "no video for this case")
    return FileResponse(p)


@app.get("/cases/{cid}/scene")
def scene(cid: str):
    _need(cid)
    s = store.read_json(cid, "scene.json")
    if s is None:
        raise HTTPException(409, "scene not ready")
    return s


@app.get("/cases/{cid}/verdict")
def verdict(cid: str):
    _need(cid)
    return _verdict(cid)


class DisputeIn(BaseModel):
    fact_id: str
    new_value: str | float | bool
    evidence_type: Literal["second_video", "photo", "witness", "user_word"] = "user_word"
    evidence_sha256: str | None = None
    note: str = ""


@app.post("/cases/{cid}/disputes")
def dispute(cid: str, body: DisputeIn):
    """Disputes change facts, never percentages. Rules rerun; nothing is overwritten."""
    _need(cid)
    before = _verdict(cid, with_narrative=False)
    fact = next((f for f in before["facts"] if f["id"] == body.fact_id), None)
    if fact is None:
        raise HTTPException(404, "unknown fact")
    old_a = before["split"]["A"] if before["split"] else None
    did = store.add_dispute(cid, {"fact_id": body.fact_id, "actor": fact["actor"], "old_value": fact["value"],
                                  "new_value": body.new_value, "evidence_type": body.evidence_type,
                                  "evidence_sha256": body.evidence_sha256, "old_split": old_a, "note": body.note})
    after = _verdict(cid)
    store.update_dispute_split(did, after["split"]["A"] if after["split"] else None)
    after["disputes"] = store.disputes(cid)
    old_rules = {(r["rule_id"], r["charged"]): r["contribution"] for r in before["fired_rules"]}
    new_rules = {(r["rule_id"], r["charged"]): r["contribution"] for r in after["fired_rules"]}
    changed = sorted({f"{k[0]}:{k[1]}" for k in old_rules.keys() | new_rules.keys()
                      if old_rules.get(k) != new_rules.get(k)})
    return {"verdict": after, "old_split": before["split"], "changed_rules": changed, "dispute_id": did}


@app.post("/cases/{cid}/evidence")
async def upload_evidence(cid: str, file: UploadFile = File(...)):
    """Hash a supporting photo / second video so the dispute record can reference it."""
    _need(cid)
    raw = await file.read()
    sha = hashlib.sha256(raw).hexdigest()
    d = store.case_dir(cid) / "evidence"
    d.mkdir(exist_ok=True)
    (d / f"{sha[:16]}{Path(file.filename or '').suffix}").write_bytes(raw)
    return {"evidence_sha256": sha, "name": file.filename}


@app.get("/cases/{cid}/disputes")
def list_disputes(cid: str):
    _need(cid)
    return store.disputes(cid)


@app.delete("/cases/{cid}/disputes")
def reset_disputes(cid: str):
    """Demo rehearsal reset — only allowed on cached demo cases."""
    c = _need(cid)
    if not c["cached"]:
        raise HTTPException(403, "disputes on real cases are permanent")
    store.clear_disputes(cid)
    return {"ok": True}


class CalibrateIn(BaseModel):
    apply: bool = False


@app.post("/calibrate")
def calibrate(body: CalibrateIn):
    res = calib_mod.fit()
    if body.apply:
        store.DATA.mkdir(parents=True, exist_ok=True)
        WEIGHTS.write_text(json.dumps({w["rule_id"]: w["after"] for w in res["weights"]}))
    res["applied"] = body.apply
    return res


@app.delete("/calibrate")
def reset_calibration():
    if WEIGHTS.exists():
        WEIGHTS.unlink()
    return {"ok": True}
