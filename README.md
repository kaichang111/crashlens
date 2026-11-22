# CrashLens

Dashcam clip → 3D replay → a fault split where every percentage point traces to a Texas statute and a
measured fact. Dispute any fact and the split recalculates live. Built from *CrashLens — Build Spec (Tidal Hackathon)*.

```
crashlens/
  blender/crashlens_models.blend   source models + the two dashcam render scenes
  backend/                         FastAPI + the 6-stage Python pipeline
    app/rules.yaml                 the Texas statute rules judges can read
    app/engine.py                  Step 4  deterministic fault engine, disputes, uncertainty range
    app/facts.py                   Step 4.1 trajectories → facts with confidences
    app/reconstruct.py             Step 3  homography, Kalman + RTS, impact detection/classification
    app/pipeline.py                Steps 1–3 on real video (OpenCV, YOLO11 + ByteTrack, librosa)
    app/narrative.py               Step 5.2 Nemotron summary + citation/number guard
    app/calibrate.py               Step 6.4 weight fitting on (simulated) decided outcomes
    app/scenarios.py               scripted demo crashes (the cached backup cases)
    samples/*.mp4                  dashcam clips rendered in Blender for the demo cases
  frontend/                        React + Vite + TS + Tailwind + react-three-fiber
    public/models/*.glb            car, pickup, traffic signal, stop sign (exported from Blender)
```

## Run it

Backend (Python 3.11+; tested on 3.14):

```powershell
cd crashlens\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Frontend:

```powershell
cd crashlens\frontend
npm install
npm run dev
```

Open http://localhost:5173 and click **Left turn across traffic** (or go straight to `#/results/demo-left-turn`).
API docs: http://localhost:8000/docs. Tests: `.\.venv\Scripts\python.exe -m pytest -q tests`.

Live video pipeline (optional, heavy: pulls PyTorch): `pip install -r requirements-cv.txt`. Without it, uploads
need the hidden `?cached=1` switch (`http://localhost:5173/?cached=1`), which is the spec's demo backup.

Narrative: set `NVIDIA_API_KEY` (same key `nemotron.py` uses). Without it, or if the model cites a fact that
doesn't exist or uses a number not in the facts, the summary falls back to the rule-card sentences.

## Demo script (spec, 3 minutes)

1. Upload → **Left turn across traffic**. Results show the dashcam clip next to the 3D replay on one scrubber.
2. Play top-down at 0.5×, then **Driver B** view: "this is what the turning driver could see."
3. Open the **Left turn failed to yield** card: § 545.152, the 2.6 s gap, Jump to moment. Split ≈ B 89 / A 11
   (spec worked example: 91 / 9), with the 50% bar.
4. **Dispute** F10 (B's light) → "green arrow", *My word only*: fact becomes contested, split moves 11 → 14%.
   Dispute again with *Photo*: it moves much further. "Facts move the verdict, not opinions."
5. **Calibrate** tab: fit weights on 10 simulated outcomes and watch them move. **Export report** → print to PDF.
   **Reset demo disputes** before the next rehearsal.

## Honest notes

- Rule weights are reasoned estimates; the calibration cases are simulated and labelled as such.
- The demo clips are Blender renders of scripted crashes. Their trajectories go through the real Step 3 code
  (noise growing with Z², Kalman + RTS, 30 Hz Hermite resample, rectangle-overlap impact), but not through YOLO.
- The live path (`pipeline.py`) has been run end to end on `samples/left_turn.mp4`, both through
  `scripts/live_check.py` and through the UI (upload → calibrate → processing → results). With the exact
  lane-corner pixels (projected from the Blender camera) the live split is A 10–13 % vs 9.7 % ground
  truth, the same two rules fire, A's speed is 41.5–43 mph vs 41.1, and B's turn gap is 2.2–2.5 s vs 2.6 s.
- Accuracy is dominated by the four calibration clicks: half-pixel errors on the far corners (~20 m out)
  scale every speed by ~8 % (ΔZ ≈ Z²/fh). With raw clicks the split was A 17–20 % and the turn gap ~1.1 s.
  The Calibrate screen therefore has a 6× loupe, Shift-drag for fine movement, and arrow-key nudging
  (0.1 px). With corners refined that way, the UI run gave A 12 % / B 88 % (truth 9.7 / 90.3), a 2.4 s turn
  gap (truth 2.6), and A at 43 mph (truth 41).
- If the repo lives in OneDrive, set `CRASHLENS_DATA` to a folder outside it. OneDrive briefly locks new
  files; writes retry, but case data doesn't belong in a synced folder anyway.
- YOLO11 detects the low-poly Blender traffic signal but can't read its lamp colour, so the demo clip has no
  signal-state facts on the live path. The left-turn rule still fires, because only evidence of a protected
  arrow excuses the turn.
- Ego-motion uses bird's-eye optical flow plus a lane-dash correlation check (spec §3.3), masks the hood
  and other vehicles, and holds velocity when the other car fills the view. The per-frame log is saved as
  `data/cases/<id>/ego.json`.
- An evidence summary, not legal advice or an official fault determination.
