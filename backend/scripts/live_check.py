"""End-to-end check of the live pipeline (Steps 1-5) on a rendered demo clip.

Runs ingest -> YOLO11/ByteTrack -> homography + ego-motion -> Kalman -> facts -> verdict
on samples/left_turn.mp4 with calibration clicks projected from the Blender camera,
then compares against the scripted ground truth (scenarios.left_turn).

    .venv\\Scripts\\python.exe scripts\\live_check.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import engine, facts, pipeline, scenarios, store  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "left_turn.mp4"
# Ground-truth lane rectangle (inner northbound lane, one dash + gap = 12.19 m), frame 0, 854x480,
# projected from the Blender dashcam camera.
CALIB = {
    "points": [[363.7, 291.6], [492.6, 291.6], [454.4, 262.8], [401.5, 262.8]],
    "lane_width_m": 3.66,
    "other_tap": [537.4, 286.5],   # pickup on the frame ~0.1 s before impact
    "k1": 0.0,                     # rendered with a pinhole camera: no lens distortion
    "hfov_deg": 110.0,
    "lanes_per_dir": 2,
}


def main():
    raw = SAMPLE.read_bytes()
    cid = "live-check"
    form = {"location_type": "intersection_signal", "your_vehicle": "A", "speed_limit_mph": 35,
            "surface": "dry", "description": "I was going straight on green, they turned left in front of me"}
    store.create_case(cid, hashlib.sha256(raw).hexdigest(), store.now(), form, cached=False)
    (store.case_dir(cid) / "original.mp4").write_bytes(raw)
    case = store.read_json(cid, "case.json"); case["video_file"] = "original.mp4"
    store.write_json(cid, "case.json", case)

    t0 = time.time()
    pipeline.ingest(cid)
    imp = store.read_json(cid, "impact_guess.json")
    print(f"[ingest] {time.time() - t0:.1f}s impact guess t={imp['t']:.2f} (jolt {imp['jolt_t']}, sound {imp['sound_t']}) truth 5.00")
    st = store.get_case(cid)
    if st["status"] == "error":
        sys.exit(st["error"])

    t0 = time.time()
    pipeline.run(cid, {**CALIB, "impact_t": 5.0 - store.read_json(cid, "frames.json")["clip_offset"]})
    st = store.get_case(cid)
    print(f"[pipeline] {time.time() - t0:.1f}s status={st['status']} stage={st['stage']} err={st['error']}")
    if st["status"] != "done":
        sys.exit(1)

    det = store.read_json(cid, "detections.json")
    tracks = {}
    for fr in det["frames"]:
        for o in fr["objects"]:
            tracks.setdefault((o["track"], o["cls"]), 0)
            tracks[(o["track"], o["cls"])] += 1
    print("[detect] tracks:", sorted(tracks.items(), key=lambda kv: -kv[1])[:8])
    lights = [o.get("state") for fr in det["frames"] for o in fr["objects"] if o["cls"] == "traffic_light"]
    print("[detect] light states:", {s: lights.count(s) for s in set(lights)})

    scene = store.read_json(cid, "scene.json")
    print("[scene] impact:", json.dumps(scene["impact"]), "| signals:", [s["states"] for s in scene["signals"]])
    live = facts.extract(scene, form)
    truth = facts.extract(scenarios.left_turn(), form)
    print("\n[facts] type/actor            live            truth")
    tv = {(f["type"], f["actor"]): f["value"] for f in truth}
    for f in live:
        print(f"  {f['type'] + '/' + f['actor']:<28}{str(f['value']):<16}{tv.get((f['type'], f['actor']), '—')}")
    for k, v in tv.items():
        if k not in {(f['type'], f['actor']) for f in live}:
            print(f"  {k[0] + '/' + k[1]:<28}{'MISSING':<16}{v}")
    vl, vt = engine.score(live), engine.score(truth)
    print("\n[verdict] live split", vl["split"], [r["rule_id"] + ":" + r["charged"] for r in vl["fired_rules"]])
    print("[verdict] truth split", vt["split"], [r["rule_id"] + ":" + r["charged"] for r in vt["fired_rules"]])


if __name__ == "__main__":
    main()
