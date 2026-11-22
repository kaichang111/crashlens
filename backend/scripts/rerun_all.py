"""Re-run the live pipeline on every real-video case with its saved calibration, then audit positions.

    .venv\\Scripts\\python.exe scripts\\rerun_all.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import pipeline, store  # noqa: E402
from audit_scenes import audit  # noqa: E402

cases = []
for d in sorted(store.CASES.iterdir()):
    case = store.read_json(d.name, "case.json") or {}
    if not case.get("cached") and case.get("video_file") and (d / "calibration.json").exists():
        cases.append(d.name)
print("real-video cases:", cases, flush=True)
for cid in cases:
    t0 = time.time()
    pipeline.run(cid, store.read_json(cid, "calibration.json"))
    st = store.get_case(cid)
    print(f"{cid}: {st['status']} {st['error'] or ''} ({time.time() - t0:.0f}s)", flush=True)
    if st["status"] == "done":
        print("   ", json.dumps(audit(cid)), flush=True)
