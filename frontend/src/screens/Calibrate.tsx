import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { go } from "../router";
import type { CaseStatus, FramesMeta } from "../types";

const LANES = [
  [3.66, "12 ft, interstate or highway"],
  [3.35, "11 ft, arterial road"],
  [3.05, "10 ft, city street"],
] as const;
const CORNER = ["near-left", "near-right", "far-right", "far-left"];

/** Screen 2: 4-point ground-plane calibration, other-vehicle tap, optional impact frame. */
const ZOOM = 6;
const LOUPE = 180; // px

/** Magnifier: the frame at ZOOM× around a point, with a crosshair and the placed corners. */
function Loupe({ src, at, natW, disp, pts, sel }: {
  src: string; at: [number, number]; natW: number; disp: DOMRect;
  pts: [number, number][]; sel: number | null;
}) {
  const s = disp.width / natW;                       // natural → displayed scale
  const cx = at[0] * s, cy = at[1] * s;              // point in displayed px
  const bw = disp.width * ZOOM, bh = disp.height * ZOOM;
  // keep the loupe in the corner away from the cursor so it never covers what you're aiming at
  const left = cx > disp.width / 2 ? 12 : disp.width - LOUPE - 12;
  const top = cy > disp.height / 2 ? 12 : disp.height - LOUPE - 12;
  const toLoupe = (p: [number, number]) => [(p[0] * s - cx) * ZOOM + LOUPE / 2, (p[1] * s - cy) * ZOOM + LOUPE / 2];
  return (
    <div className="pointer-events-none absolute overflow-hidden rounded-full border-2 border-white"
         style={{ left, top, width: LOUPE, height: LOUPE, backgroundColor: "#000",
                  backgroundImage: `url(${src})`, backgroundRepeat: "no-repeat",
                  backgroundSize: `${bw}px ${bh}px`,
                  backgroundPosition: `${LOUPE / 2 - cx * ZOOM}px ${LOUPE / 2 - cy * ZOOM}px`,
                  imageRendering: "pixelated" }}>
      <svg className="absolute inset-0" width={LOUPE} height={LOUPE}>
        <line x1={LOUPE / 2} y1={0} x2={LOUPE / 2} y2={LOUPE / 2 - 6} stroke="#38bdf8" strokeWidth={1} />
        <line x1={LOUPE / 2} y1={LOUPE / 2 + 6} x2={LOUPE / 2} y2={LOUPE} stroke="#38bdf8" strokeWidth={1} />
        <line x1={0} y1={LOUPE / 2} x2={LOUPE / 2 - 6} y2={LOUPE / 2} stroke="#38bdf8" strokeWidth={1} />
        <line x1={LOUPE / 2 + 6} y1={LOUPE / 2} x2={LOUPE} y2={LOUPE / 2} stroke="#38bdf8" strokeWidth={1} />
        {pts.map((p, i) => {
          const [x, y] = toLoupe(p);
          return <circle key={i} cx={x} cy={y} r={5} fill="none" stroke={sel === i ? "#38bdf8" : "#fbbf24"} strokeWidth={1.5} />;
        })}
      </svg>
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded bg-black/70 px-1.5 text-[10px] text-white">
        {at[0].toFixed(1)}, {at[1].toFixed(1)} at {ZOOM}×
      </div>
    </div>
  );
}

export default function Calibrate({ id }: { id: string }) {
  const [st, setSt] = useState<CaseStatus | null>(null);
  const [fm, setFm] = useState<FramesMeta | null>(null);
  const [pts, setPts] = useState<[number, number][]>([]);
  const [drag, setDrag] = useState<number | null>(null);
  const [lane, setLane] = useState(3.66);
  const [impactI, setImpactI] = useState(0);
  const [tap, setTap] = useState<[number, number] | null>(null);
  const [mode, setMode] = useState<"lane" | "vehicle">("lane");
  const [stat, setStat] = useState(false);
  const [k1, setK1] = useState(-0.25);
  const [err, setErr] = useState<string | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null); // natural px under the cursor
  const [sel, setSel] = useState<number | null>(null);                // handle the arrow keys nudge
  const img = useRef<HTMLImageElement>(null);
  const dragRef = useRef<{ i: number; start: [number, number]; mouse: [number, number] } | null>(null);

  // Arrow keys nudge the selected corner by 0.1 px (Shift: 1 px). Far corners sit ~20 m out where one
  // pixel is about a metre of depth (ΔZ ≈ Z²/fh), so sub-pixel placement matters.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode !== "lane" || sel == null || !pts[sel]) return;
      const d = e.shiftKey ? 1 : 0.1;
      const mv: Record<string, [number, number]> = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] };
      const m = mv[e.key];
      if (!m) return;
      e.preventDefault();
      setPts((p) => p.map((q, i) => (i === sel ? [q[0] + m[0], q[1] + m[1]] : q)));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, sel, pts]);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await api.status(id);
        if (!alive) return;
        setSt(s);
        if (s.status === "awaiting_calibration") {
          const f = await api.frames(id);
          setFm(f); setImpactI(Math.round(f.impact_t * f.fps));
          return;
        }
        if (s.status !== "error") setTimeout(poll, 1000);
      } catch (e) { setErr((e as Error).message); }
    };
    poll();
    return () => { alive = false; };
  }, [id]);

  // convert between displayed and natural image pixels
  const toNat = (e: { clientX: number; clientY: number }): [number, number] => {
    const r = img.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * fm!.width, ((e.clientY - r.top) / r.height) * fm!.height];
  };
  const pctPos = (p: [number, number]) => ({ left: `${(p[0] / fm!.width) * 100}%`, top: `${(p[1] / fm!.height) * 100}%` });

  async function submit() {
    setErr(null);
    try {
      await api.calibrate(id, {
        points: pts, lane_width_m: lane, other_tap: tap, impact_t: fm ? impactI / fm.fps : null, static_camera: stat, k1,
      });
      go(`/processing/${id}`);
    } catch (e) { setErr((e as Error).message); }
  }

  if (!fm) {
    return (
      <div className="mx-auto max-w-xl px-4 py-20 text-center">
        <div className="text-lg font-semibold">Preparing frames…</div>
        <p className="mt-2 text-sm text-ink2">Finding the impact moment (visual jolt + crash sound) and extracting 5 s before / 2 s after.</p>
        {st?.error && <p className="mt-4 text-sm text-bad">{st.error}</p>}
        {err && <p className="mt-4 text-sm text-bad">{err}</p>}
      </div>
    );
  }
  const frameIdx = mode === "lane" ? 0 : impactI;
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <div className="flex gap-2">
        <button onClick={() => setMode("lane")} className={`tab ${mode === "lane" ? "tab-on" : ""}`}>1. Mark one lane</button>
        <button onClick={() => setMode("vehicle")} className={`tab ${mode === "vehicle" ? "tab-on" : ""}`}>2. Point to the other vehicle</button>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_300px]">
        <div className="relative select-none overflow-hidden bg-black"
             onMouseMove={(e) => {
               const n = toNat(e);
               setHover(n);
               const d = dragRef.current;
               if (drag != null && d) {
                 // Shift-drag moves the handle at 1/5 speed for fine placement
                 const k = e.shiftKey ? 0.2 : 1;
                 const p: [number, number] = [d.start[0] + (n[0] - d.mouse[0]) * k, d.start[1] + (n[1] - d.mouse[1]) * k];
                 setPts((ps) => ps.map((q, i) => (i === drag ? p : q)));
               }
             }}
             onMouseLeave={() => { setHover(null); setDrag(null); }}
             onMouseUp={() => { setDrag(null); dragRef.current = null; }}>
          <img ref={img} src={api.previewUrl(id, frameIdx, k1)} className="block w-full" draggable={false}
               onClick={(e) => {
                 if (mode === "lane") { if (pts.length < 4) { setPts([...pts, toNat(e)]); setSel(pts.length); } }
                 else setTap(toNat(e));
               }} />
          {mode === "lane" && pts.length > 1 && (
            <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${fm.width} ${fm.height}`} preserveAspectRatio="none">
              <polygon points={pts.map((p) => p.join(",")).join(" ")} fill="rgba(251,191,36,0.2)" stroke="#fbbf24" strokeWidth={3} />
            </svg>
          )}
          {mode === "lane" && pts.map((p, i) => (
            // hollow ring with a 1 px centre dot, so the handle doesn't hide the line it sits on
            <div key={i} onMouseDown={(e) => { e.stopPropagation(); setDrag(i); setSel(i); dragRef.current = { i, start: p, mouse: toNat(e) }; }}
                 className={`absolute -ml-2.5 -mt-2.5 h-5 w-5 cursor-move rounded-full border-2 ${sel === i ? "border-veha" : "border-sign"}`}
                 style={pctPos(p)}>
              <div className="absolute left-1/2 top-1/2 h-0.5 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-sign" />
              <span className="absolute left-6 top-0 whitespace-nowrap rounded bg-black/70 px-1 text-[10px] text-signink">{CORNER[i]}</span>
            </div>
          ))}
          {mode === "lane" && hover && img.current && (
            <Loupe src={api.previewUrl(id, frameIdx, k1)} at={drag != null && pts[drag] ? pts[drag] : hover}
                   natW={fm.width} disp={img.current.getBoundingClientRect()} pts={pts} sel={sel} />
          )}
          {mode === "vehicle" && tap && (
            <div className="absolute -ml-3 -mt-3 h-6 w-6 rounded-full border-4 border-red-500" style={pctPos(tap)} />
          )}
        </div>
        <div className="space-y-4">
          {mode === "lane" ? (
            <>
              <p className="text-sm text-ink2">
                Click four corners of one lane: two points on each lane line, from the start of one dash to the start of the next
                (one dash + gap = 40 ft / 12.19 m). Order: {CORNER.join(" → ")}.
              </p>
              <p className="text-xs text-ink2">
                Use the magnifier to land exactly on the line edge. Drag a corner to move it (hold <b>Shift</b> for fine
                movement), or select it and use the <b>arrow keys</b> (0.1 px; Shift = 1 px). The far corners matter most:
                at 20 m, one pixel is about a metre.
              </p>
              {pts.length > 0 && (
                <div className="space-y-0.5 text-[11px] text-ink2">
                  {pts.map((p, i) => (
                    <button key={i} onClick={() => setSel(i)} className={`block ${sel === i ? "text-veha" : ""}`}>
                      {CORNER[i].padEnd(10)} ({p[0].toFixed(1)}, {p[1].toFixed(1)})
                    </button>
                  ))}
                </div>
              )}
              <div>
                <label className="lbl">Lane width</label>
                <select className="field" value={lane} onChange={(e) => setLane(Number(e.target.value))}>
                  {LANES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </div>
              <div>
                <label className="lbl">Lens correction (nudge until lane lines look straight)</label>
                <input type="range" min={-0.5} max={0} step={0.01} value={k1}
                       onChange={(e) => setK1(Number(e.target.value))} className="w-full accent-[#1e2226]" />
                <div className="text-[11px] text-muted">k1 = {k1.toFixed(2)}{k1 === 0 ? " (no correction)" : ""}</div>
              </div>
              <label className="flex items-center gap-2 text-sm text-ink2">
                <input type="checkbox" checked={stat} onChange={(e) => setStat(e.target.checked)} /> Camera was parked (skip ego-motion)
              </label>
              <button onClick={() => setPts([])} className="text-xs text-ink2 underline">Reset points</button>
              <button disabled={pts.length !== 4} onClick={() => setMode("vehicle")} className="btn w-full">Next</button>
            </>
          ) : (
            <>
              <p className="text-sm text-ink2">Scrub to the impact frame if the auto-detected one is off, then tap the other vehicle.</p>
              <input type="range" min={0} max={fm.frames.length - 1} value={impactI} onChange={(e) => setImpactI(Number(e.target.value))} className="w-full accent-[#1e2226]" />
              <div className="text-xs text-ink2">Frame {impactI}, {(impactI / fm.fps).toFixed(1)} s into the clip</div>
              <button disabled={pts.length !== 4 || !tap} onClick={submit} className="btn w-full">Reconstruct crash</button>
            </>
          )}
          {err && <p className="text-sm text-bad">{err}</p>}
        </div>
      </div>
    </div>
  );
}
