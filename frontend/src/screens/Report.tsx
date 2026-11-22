import { useEffect, useState } from "react";
import { api } from "../api";
import FaultSplit from "../components/FaultSplit";
import { fmtVal } from "../lib";
import { go } from "../router";
import { useClock } from "../store";
import Replay from "../three/Replay";
import type { Scene, Verdict } from "../types";

const DISCLAIMER = "An evidence summary, not legal advice or an official fault determination.";

/** Key frames grabbed from the original video at given times. */
function useKeyFrames(src: string | null, times: number[]) {
  const [shots, setShots] = useState<{ t: number; url: string }[]>([]);
  useEffect(() => {
    if (!src) return;
    let cancelled = false;
    const v = document.createElement("video");
    v.src = src; v.muted = true; v.preload = "auto"; v.crossOrigin = "anonymous";
    const out: { t: number; url: string }[] = [];
    const grab = (i: number) => {
      if (i >= times.length) { if (!cancelled) setShots(out); return; }
      v.currentTime = Math.max(0, times[i]);
      v.onseeked = () => {
        const c = document.createElement("canvas");
        c.width = v.videoWidth; c.height = v.videoHeight;
        c.getContext("2d")!.drawImage(v, 0, 0);
        out.push({ t: times[i], url: c.toDataURL("image/jpeg", 0.85) });
        grab(i + 1);
      };
    };
    v.onloadeddata = () => grab(0);
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, times.join(",")]);
  return shots;
}

/** Dedicated print route → browser print-to-PDF (spec §5.3). */
export default function Report({ id }: { id: string }) {
  const [scene, setScene] = useState<Scene | null>(null);
  const [v, setV] = useState<Verdict | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  useEffect(() => {
    Promise.all([api.scene(id), api.verdict(id)]).then(([s, vv]) => {
      setScene(s); setV(vv);
      useClock.getState().setDuration(s.duration);
      useClock.getState().setPlaying(false);
      useClock.getState().setT(s.impact?.t ?? s.duration);
      useClock.getState().setView("top");
    });
    fetch(api.videoUrl(id), { method: "HEAD" }).then((r) => r.ok && setVideo(api.videoUrl(id)));
  }, [id]);
  const ti = scene?.impact?.t ?? 0;
  const off = scene?.meta.clip_offset ?? 0;
  const shots = useKeyFrames(video, scene ? [ti - 2 + off, ti - 0.7 + off, ti + 0.1 + off] : []);
  if (!scene || !v) return <div className="p-10 text-muted">Preparing report…</div>;
  const contested = v.facts.filter((f) => f.status && f.status !== "measured");

  return (
    <div className="report min-h-screen bg-paper py-6 text-ink print:bg-white print:py-0">
      <div className="no-print mx-auto mb-4 flex max-w-[820px] justify-between">
        <button onClick={() => go(`/results/${id}`)} className="text-sm text-muted underline">← Back to results</button>
        <button onClick={() => window.print()} className="btn">Print or save as PDF</button>
      </div>
      <article className="mx-auto max-w-[820px] border border-rule bg-white p-10 print:border-0 print:p-0">
        <header className="border-b-2 border-ink pb-3">
          <div className="flex items-baseline justify-between">
            <h1 className="text-2xl font-bold">CrashLens Evidence Report</h1>
            <div className="text-sm text-muted">Case {id}</div>
          </div>
          <table className="mt-3 w-full text-xs">
            <tbody>
              <tr><td className="w-40 py-0.5 text-muted">Report generated</td><td>{new Date().toISOString()}</td></tr>
              <tr><td className="py-0.5 text-muted">Location type</td><td>{v.location_type?.replace(/_/g, " ")}, {scene.road.lanes_per_dir} lanes each way, {scene.road.lane_width_m} m lanes</td></tr>
              <tr><td className="py-0.5 text-muted">Evidence SHA-256</td><td className="break-all">{v.evidence_sha256}</td></tr>
              <tr><td className="py-0.5 text-muted">Original received (UTC)</td><td className="font-mono">{v.received_at}</td></tr>
              <tr><td className="py-0.5 text-muted">Your vehicle</td><td>{v.your_vehicle === "A" ? "A (blue, camera car)" : "B (red)"}</td></tr>
            </tbody>
          </table>
        </header>

        <h2 className="sec">1. Reconstruction at impact</h2>
        <div className="grid grid-cols-2 gap-3">
          <div className="aspect-square overflow-hidden rounded border border-rule">
            <Replay scene={scene} view="top" preserve labels />
          </div>
          <div className="text-xs">
            {scene.impact && (
              <table className="w-full">
                <tbody>
                  <tr><td className="py-0.5 text-muted">Impact time</td><td>{scene.impact.t.toFixed(2)} s into the clip</td></tr>
                  <tr><td className="py-0.5 text-muted">Crash type</td><td>{scene.impact.type.replace("_", " ")} ({scene.impact.angle_deg}° between headings)</td></tr>
                  <tr><td className="py-0.5 text-muted">Closing speed</td><td>{(scene.impact.closing_speed * 2.237).toFixed(0)} mph</td></tr>
                  <tr><td className="py-0.5 text-muted">Contact</td><td>A on its {scene.impact.contact.A.replace("_", " ")}, B on its {scene.impact.contact.B.replace("_", " ")}</td></tr>
                </tbody>
              </table>
            )}
            <p className="mt-2 text-muted">Top-down still from the 3D replay. Positions come from a ground-plane homography, ego-motion removal and Kalman/RTS smoothing; shaded boxes show positional uncertainty.</p>
          </div>
        </div>
        {shots.length > 0 && (
          <div className="mt-3 grid grid-cols-3 gap-2">
            {shots.map((s) => (
              <figure key={s.t}><img src={s.url} className="w-full rounded border border-rule" />
                <figcaption className="mt-0.5 text-center text-[10px] text-muted">video t = {(s.t - off).toFixed(1)} s</figcaption></figure>
            ))}
          </div>
        )}

        <h2 className="sec">2. Fault split</h2>
        <FaultSplit verdict={v} light />
        <p className="mt-2 text-xs text-muted">
          Texas proportionate responsibility (Civ. Prac. &amp; Rem. Code § 33.001): a claimant more than 50% responsible recovers nothing; below that,
          damages are reduced by the claimant's share. The range shows the split with every fact confidence moved ±10 points.
        </p>

        <h2 className="sec">3. Rules applied</h2>
        {v.fired_rules.map((r) => (
          <div key={r.rule_id + r.charged} className="mb-2 break-inside-avoid rounded border border-rule p-2 text-xs">
            <div className="flex justify-between font-semibold"><span>{r.title}, {r.statute}</span><span>+{r.contribution.toFixed(2)} to {r.charged}</span></div>
            <div className="mt-0.5">{r.explanation}</div>
            <div className="mt-0.5 text-[10px] text-muted">{r.weight.toFixed(2)} × {r.confidence.toFixed(2)} × {r.causation.toFixed(1)}, from facts {r.fact_ids.join(", ")}</div>
          </div>
        ))}

        <h2 className="sec">4. Measured facts</h2>
        <table className="w-full text-[11px]">
          <thead><tr className="border-b border-rule text-left"><th className="py-1">ID</th><th>Fact</th><th>Value</th><th>Conf.</th><th>Frames</th></tr></thead>
          <tbody>
            {v.facts.map((f) => (
              <tr key={f.id} className="break-inside-avoid border-b border-rule2 align-top">
                <td className="py-1">{f.id}</td>
                <td className="pr-2">{f.label}{f.status && f.status !== "measured" && <span className="ml-1 font-semibold text-warn">[{f.status}]</span>}</td>
                <td>{fmtVal(f.value)}{f.unit ? ` ${f.unit}` : ""}</td>
                <td>{Math.round(f.confidence * 100)}%</td>
                <td className="font-mono">{f.evidence.frames.join(",")}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2 className="sec">5. Disputes</h2>
        {v.disputes.length === 0 ? <p className="text-xs text-muted">No facts disputed.</p> : (
          <table className="w-full text-[11px]">
            <thead><tr className="border-b border-rule text-left"><th className="py-1">#</th><th>Fact</th><th>Change</th><th>Evidence</th><th>Split A</th><th>When (UTC)</th></tr></thead>
            <tbody>
              {v.disputes.map((d) => (
                <tr key={d.id} className="border-b border-rule2">
                  <td className="py-1">{d.id}</td><td>{d.fact_id}</td><td>{fmtVal(d.old_value)} → {fmtVal(d.new_value)}</td>
                  <td>{d.evidence_type.replace("_", " ")}{d.evidence_sha256 ? ` (${d.evidence_sha256.slice(0, 10)}…)` : ""}</td>
                  <td>{d.old_split != null ? Math.round(d.old_split * 100) : "–"}% → {d.new_split != null ? Math.round(d.new_split * 100) : "–"}%</td>
                  <td className="font-mono">{d.created_at.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {contested.length > 0 && (
          <p className="mt-2 text-xs text-muted">
            Contested facts appear with both versions: {contested.map((f) => `${f.id} video "${fmtVal(f.original?.value)}" vs claimed "${fmtVal(f.contested_value ?? f.value)}"`).join("; ")}.
          </p>
        )}

        {v.narrative && (
          <>
            <h2 className="sec">6. Plain-English summary</h2>
            {v.narrative.text.split("\n\n").map((p, i) => <p key={i} className="mb-2 text-xs leading-relaxed">{p}</p>)}
          </>
        )}

        <footer className="mt-8 border-t border-rule pt-2 text-[10px] text-muted">
          {DISCLAIMER} Evidence hash {v.evidence_sha256.slice(0, 16)}… carried through every artifact.
        </footer>
      </article>
      <div className="print-footer">{DISCLAIMER}</div>
    </div>
  );
}
