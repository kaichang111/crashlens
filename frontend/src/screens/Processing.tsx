import { useEffect, useState } from "react";
import { api } from "../api";
import { go } from "../router";
import type { CaseStatus } from "../types";

const STAGES = ["detecting", "reconstructing", "judging", "done"];

/** Screen 3: poll GET /cases/{id}/status every second; no websockets needed. */
export default function Processing({ id }: { id: string }) {
  const [s, setS] = useState<CaseStatus | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const st = await api.status(id).catch(() => null);
      if (!alive) return;
      if (st) setS(st);
      if (st?.status === "done") return go(`/results/${id}`);
      if (st?.status !== "error") setTimeout(tick, 1000);
    };
    tick();
    return () => { alive = false; };
  }, [id]);
  return (
    <div className="mx-auto max-w-xl px-4 py-24">
      <div className="text-lg font-semibold">Reconstructing the crash…</div>
      <div className="mt-4 h-3 overflow-hidden bg-rule2">
        <div className="h-full bg-ink" style={{ width: `${s?.percent ?? 0}%` }} />
      </div>
      <div className="mt-3 flex justify-between text-xs">
        {STAGES.map((x) => <span key={x} className={s?.stage === x ? "font-semibold text-signink" : "text-muted"}>{x}</span>)}
      </div>
      {s?.status === "error" && (
        <div className="mt-6 rounded-[3px] bg-badbg p-3 text-sm text-bad">
          {s.error}
          <button className="ml-2 underline" onClick={() => go(`/calibrate/${id}`)}>Back to calibration</button>
        </div>
      )}
    </div>
  );
}
