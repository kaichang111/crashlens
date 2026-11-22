import { useEffect, useRef, useState } from "react";
import type { Verdict } from "../types";

/**
 * The fault split drawn as a lane: A's share in A's colour from the left, B's from the right, the Texas
 * 50% bar as a yellow stop line across it, and the uncertainty range as a hatched band. After a dispute
 * the boundary slides from the old split to the new one (motion that answers the user's action).
 */
export default function FaultSplit({ verdict, from, light = false }: { verdict: Verdict; from?: number | null; light?: boolean }) {
  const target = verdict.split?.A ?? null;
  const [a, setA] = useState<number | null>(from ?? target);
  const cur = useRef<number | null>(a);
  cur.current = a;
  useEffect(() => {
    if (target == null) { setA(null); return; }
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const start = from ?? cur.current ?? target, t0 = performance.now(), dur = reduce ? 0 : 1100;
    let id = 0;
    const step = (now: number) => {
      const u = dur ? Math.min(1, (now - t0) / dur) : 1, e = 1 - Math.pow(1 - u, 3);
      setA(start + (target - start) * e);
      if (u < 1) id = requestAnimationFrame(step);
    };
    id = requestAnimationFrame(step);
    return () => cancelAnimationFrame(id);
  }, [target, from]);

  if (verdict.status !== "ok" || a == null) {
    return (
      <section className={`border border-rule p-5 ${light ? "bg-white" : "bg-panel"}`}>
        <h2 className="text-[17px] font-bold">No fault split yet</h2>
        <p className="mt-1 text-[14px] text-ink2">
          None of the traffic-law rules applied to either driver on the facts measured so far, so CrashLens doesn't
          guess a 50/50. Check the facts below; disputing one with evidence can change this.
        </p>
      </section>
    );
  }
  const you = verdict.your_vehicle;
  const yours = you === "A" ? a : 1 - a;
  const rA = verdict.range?.A;
  const pa = Math.round(a * 100), pb = 100 - pa;
  return (
    <section className={`border border-rule p-5 ${light ? "bg-white" : "bg-panel"}`} aria-label="Fault split">
      <div className="flex items-end justify-between">
        <div>
          <div className="text-[13px] font-semibold text-ink2"><span className="veh veh-A mr-1.5 align-[-3px]">A</span>{you === "A" ? "You" : "Other driver"}</div>
          <div className="text-[44px] font-extrabold leading-none tracking-tight text-veha">{pa}%</div>
        </div>
        <div className="text-right">
          <div className="text-[13px] font-semibold text-ink2">{you === "B" ? "You" : "Other driver"}<span className="veh veh-B ml-1.5 align-[-3px]">B</span></div>
          <div className="text-[44px] font-extrabold leading-none tracking-tight text-vehb">{pb}%</div>
        </div>
      </div>

      {/* the lane */}
      <div className="relative mt-4 h-9 bg-asphalt" role="img"
           aria-label={`A ${pa} percent, B ${pb} percent at fault. The 50 percent bar is in the middle.`}>
        <div className="absolute inset-y-[5px] left-0 bg-veha" style={{ width: `${a * 100}%` }} />
        <div className="absolute inset-y-[5px] right-0 bg-vehb" style={{ width: `${(1 - a) * 100}%` }} />
        {rA && rA[1] - rA[0] > 0.004 && (
          <div className="absolute inset-y-[5px]" title="Uncertainty range"
               style={{ left: `${rA[0] * 100}%`, width: `${(rA[1] - rA[0]) * 100}%`,
                        backgroundImage: "repeating-linear-gradient(135deg, rgba(255,255,255,.55) 0 2px, transparent 2px 6px)" }} />
        )}
        {/* lane edge lines */}
        <div className="absolute inset-x-0 top-0 h-[2px] bg-white/80" />
        <div className="absolute inset-x-0 bottom-0 h-[2px] bg-white/80" />
        {/* the 50% bar, as a stop line */}
        <div className="absolute -inset-y-1.5 left-1/2 w-[5px] -translate-x-1/2 bg-sign" />
      </div>
      <div className="mt-1.5 flex justify-between text-[12px] text-muted">
        <span>0%</span>
        <span className="font-semibold text-signink">50% bar</span>
        <span>100%</span>
      </div>

      {verdict.recovery && (
        <p className={`mt-4 border-l-0 p-3 text-[14px] leading-snug ${yours > 0.5 ? "bg-badbg text-bad" : "bg-okbg text-ok"}`}>
          {verdict.recovery.text}
        </p>
      )}
      {rA && (
        <p className="mt-3 text-[13px] text-muted">
          With every measured confidence moved up or down by 10 points, A's share stays between {Math.round(rA[0] * 100)}% and {Math.round(rA[1] * 100)}%.
          Texas proportionate responsibility, Civ. Prac. &amp; Rem. Code § 33.001.
        </p>
      )}
    </section>
  );
}
