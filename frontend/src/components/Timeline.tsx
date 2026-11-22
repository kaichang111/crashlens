import { useEffect, useRef } from "react";
import { useClock } from "../store";
import type { View } from "../types";

const VIEWS: [View, string][] = [["top", "From above"], ["driverA", "A's seat"], ["driverB", "B's seat"], ["orbit", "Free camera"]];

/** One shared scrubber for the video and the replay; while playing, requestAnimationFrame advances t. */
export default function Timeline({ impactT }: { impactT?: number | null }) {
  const { t, duration, playing, speed, view, setT, setPlaying, setSpeed, setView } = useClock();
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) { last.current = null; return; }
    let id = 0;
    const loop = (now: number) => {
      const s = useClock.getState();
      if (last.current != null) {
        const nt = s.t + ((now - last.current) / 1000) * s.speed;
        if (nt >= s.duration) { s.setT(s.duration); s.setPlaying(false); return; }
        s.setT(nt);
      }
      last.current = now;
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [playing]);

  return (
    <div className="border border-rule bg-panel px-4 py-3">
      <div className="flex items-center gap-4">
        <button onClick={() => { if (t >= duration) setT(0); setPlaying(!playing); }} className="btn w-[84px]">
          {playing ? "Pause" : "Play"}
        </button>
        <div className="relative flex-1">
          <input type="range" min={0} max={duration} step={0.01} value={t} aria-label="Time in clip"
                 onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} className="w-full accent-[#1e2226]" />
          {impactT != null && (
            <div className="pointer-events-none absolute -top-2 h-2.5 w-[3px] bg-sign" style={{ left: `${(impactT / duration) * 100}%` }} />
          )}
        </div>
        <div className="w-20 text-right text-[15px] font-semibold">{t.toFixed(2)} s</div>
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-[13px]">
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-ink2">Speed</span>
          {[0.25, 0.5, 1].map((s) => (
            <button key={s} onClick={() => setSpeed(s)} aria-pressed={speed === s}
                    className={`rounded-[3px] border px-2 py-0.5 font-semibold ${speed === s ? "border-ink bg-ink text-white" : "border-rule text-ink2 hover:border-ink"}`}>{s}×</button>
          ))}
          {impactT != null && (
            <button onClick={() => { setPlaying(false); setT(impactT); }} className="link ml-3">Go to impact</button>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-ink2">Camera</span>
          {VIEWS.map(([v, l]) => (
            <button key={v} onClick={() => setView(v)} aria-pressed={view === v}
                    className={`rounded-[3px] border px-2 py-0.5 font-semibold ${view === v ? "border-ink bg-ink text-white" : "border-rule text-ink2 hover:border-ink"}`}>{l}</button>
          ))}
        </div>
      </div>
    </div>
  );
}
