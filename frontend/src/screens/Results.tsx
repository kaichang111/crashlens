import { useEffect, useMemo, useState } from "react";
import { api, type CalibrationResult } from "../api";
import DisputeDialog from "../components/DisputeDialog";
import FaultSplit from "../components/FaultSplit";
import RuleCard, { ConfidenceBar } from "../components/RuleCard";
import Timeline from "../components/Timeline";
import VideoPane from "../components/VideoPane";
import { fmtVal, sceneChecks } from "../lib";
import { go } from "../router";
import { useClock } from "../store";
import Replay from "../three/Replay";
import type { Fact, Review, Scene, Verdict } from "../types";

const LOCATION: Record<string, string> = {
  intersection_signal: "Intersection with a traffic signal", intersection_stop: "Intersection with a stop sign",
  multi_lane_road: "Multi-lane road, two-way", divided_highway: "Highway, all lanes one direction",
  parking_lot: "Parking lot",
};

function received(iso: string) {
  const d = new Date(iso);
  return isNaN(+d) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }) + " UTC";
}

/** Loading state shaped like the page it becomes. */
function ResultsSkeleton() {
  return (
    <div className="mx-auto max-w-[1440px] px-5 py-6" aria-busy="true" aria-label="Loading case">
      <div className="skeleton h-7 w-72" />
      <div className="skeleton mt-2 h-4 w-[28rem] max-w-full" />
      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_400px]">
        <div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="skeleton aspect-video" /><div className="skeleton aspect-video" />
          </div>
          <div className="skeleton mt-3 h-24" />
        </div>
        <div className="skeleton h-72" />
      </div>
    </div>
  );
}

/** Screen 4: video and 3D replay on one clock, the split, and the findings behind it. */
export default function Results({ id }: { id: string }) {
  const [scene, setScene] = useState<Scene | null>(null);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [hasVideo, setHasVideo] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [disputing, setDisputing] = useState<Fact | null>(null);
  const [fromSplit, setFromSplit] = useState<number | null>(null);
  const [changed, setChanged] = useState<string[]>([]);
  const [tab, setTab] = useState<"rules" | "facts" | "story" | "learn">("rules");
  const view = useClock((s) => s.view);

  useEffect(() => {
    Promise.all([api.scene(id), api.verdict(id)])
      .then(([s, v]) => {
        setScene(s); setVerdict(v);
        useClock.getState().setDuration(s.duration);
        useClock.getState().setT(0);
      })
      .catch((e) => setErr(e.message));
    fetch(api.videoUrl(id), { method: "HEAD" }).then((r) => setHasVideo(r.ok)).catch(() => setHasVideo(false));
  }, [id]);

  const jump = (t: number) => { useClock.getState().setPlaying(false); useClock.getState().setT(t); };
  const isCached = useMemo(() => id.startsWith("demo-") || !!scene?.meta?.scenario, [id, scene]);

  if (err) {
    return (
      <div className="mx-auto max-w-[68ch] px-5 py-16">
        <h1 className="text-[24px] font-extrabold">This case couldn't be opened</h1>
        <p className="mt-2 text-ink2">The server said: {err}. Check that the CrashLens server is running, then reload.</p>
        <button className="btn mt-5" onClick={() => go("/")}>Start a new case</button>
      </div>
    );
  }
  if (!scene || !verdict) return <ResultsSkeleton />;

  const tabs = [["rules", "Rules that applied"], ["facts", "All measured facts"], ["story", "Summary"], ["learn", "Calibration"]] as const;
  return (
    <div className="mx-auto max-w-[1440px] px-5 py-6">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-rule pb-4">
        <div>
          <h1 className="text-[26px] font-extrabold leading-tight tracking-tight">
            Case {id}
          </h1>
          <p className="mt-0.5 text-[14px] text-ink2">
            {LOCATION[verdict.location_type] ?? verdict.location_type?.replace(/_/g, " ")}. Video received {received(verdict.received_at)}.
          </p>
          <p className="mt-0.5 break-all text-[12px] text-muted" title="SHA-256 fingerprint of the original video">
            Video fingerprint (SHA-256) {verdict.evidence_sha256}
          </p>
        </div>
        <div className="flex gap-2">
          {isCached && verdict.disputes.length > 0 && (
            <button onClick={async () => { await api.resetDisputes(id); setFromSplit(null); setVerdict(await api.verdict(id)); }}
                    className="btn-line">Clear demo disputes</button>
          )}
          <button onClick={() => go(`/report/${id}`)} className="btn">Open printable report</button>
        </div>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_400px]">
        <div className="min-w-0 space-y-3">
          <div className={`grid gap-3 ${hasVideo ? "md:grid-cols-2" : ""}`}>
            {hasVideo && (
              <figure>
                <div className="aspect-video bg-black"><VideoPane src={api.videoUrl(id)} clipOffset={scene.meta.clip_offset ?? 0} /></div>
                <figcaption className="mt-1 text-[13px] text-ink2">Original dashcam video</figcaption>
              </figure>
            )}
            <figure>
              <div className="aspect-video overflow-hidden bg-asphalt"><Replay scene={scene} view={view} /></div>
              <figcaption className="mt-1 text-[13px] text-ink2">Reconstruction, to scale. Shaded boxes show position uncertainty.</figcaption>
            </figure>
          </div>
          <Timeline impactT={scene.impact?.t} />
          <ReconstructionCheck scene={scene} />
          <ReviewBanner review={verdict.review} />
          {verdict.claims.some((c) => c.status === "conflict") && (
            <div className="border border-sign bg-warnbg p-4 text-[14px] text-warn">
              <b>Your description doesn't match the video.</b>{" "}
              {verdict.claims.filter((c) => c.status === "conflict").map((c) =>
                `You said ${c.actor}'s ${c.type.replace(/_/g, " ")} was "${fmtVal(c.value)}"; the video measured "${fmtVal(c.video_value)}" (${c.fact_id})`).join(". ")}.
              Both are listed in the report; the split uses the measurement.
            </div>
          )}
        </div>

        <aside className="space-y-5">
          <FaultSplit verdict={verdict} from={fromSplit} />
          <div role="tablist" className="flex flex-wrap gap-x-5 gap-y-2 border-b border-rule">
            {tabs.map(([k, l]) => (
              <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)} className={`tab ${tab === k ? "tab-on" : ""}`}>{l}</button>
            ))}
          </div>
          {tab === "rules" && (
            <div>
              {verdict.fired_rules.length === 0 && (
                <p className="text-[14px] text-ink2">No traffic-law rule applied to either driver on the facts measured so far.</p>
              )}
              {verdict.fired_rules.map((r) => (
                <RuleCard key={`${r.rule_id}${r.charged}`} rule={r} facts={verdict.facts}
                          changed={changed.includes(`${r.rule_id}:${r.charged}`)} onJump={jump} onDispute={setDisputing} />
              ))}
              {changed.filter((c) => !verdict.fired_rules.some((r) => `${r.rule_id}:${r.charged}` === c)).map((c) => (
                <p key={c} className="border-b border-rule py-4 text-[14px] text-ink2">
                  After the dispute, <b>{c.split(":")[0].replace(/_/g, " ")}</b> no longer applies to {c.split(":")[1]}.
                </p>
              ))}
              <Disputes verdict={verdict} />
            </div>
          )}
          {tab === "facts" && <FactsTable facts={verdict.facts} onJump={jump} onDispute={setDisputing} />}
          {tab === "story" && verdict.narrative && (
            <div>
              <div className="space-y-3 font-serif text-[16px] leading-[1.7]">
                {verdict.narrative.text.split("\n\n").map((p, i) => <p key={i}>{p}</p>)}
              </div>
              <p className="mt-4 text-[13px] text-muted">
                {verdict.narrative.source === "llm"
                  ? `Written by ${verdict.narrative.model} from the measured facts only. Every fact reference and number was checked against the facts before display.`
                  : `Put together from the rule explanations${verdict.narrative.problems.length ? `, because the AI summary was unavailable or failed the check (${verdict.narrative.problems.slice(0, 2).join("; ")})` : ""}.`}
              </p>
            </div>
          )}
          {tab === "learn" && <CalibratePanel onApplied={async () => setVerdict(await api.verdict(id))} applied={verdict.calibrated_weights} />}
        </aside>
      </div>

      {disputing && (
        <DisputeDialog caseId={id} fact={disputing} onClose={() => setDisputing(null)}
                       onDone={(r) => {
                         setFromSplit(r.old_split?.A ?? null);
                         setVerdict(r.verdict);
                         setChanged(r.changed_rules);
                         setDisputing(null);
                         setTab("rules");
                         setTimeout(() => setChanged([]), 5000);
                       }} />
      )}
    </div>
  );
}

function FactsTable({ facts, onJump, onDispute }: { facts: Fact[]; onJump: (t: number) => void; onDispute: (f: Fact) => void }) {
  return (
    <table className="w-full text-left text-[14px]">
      <tbody>
        {facts.map((f) => (
          <tr key={f.id} className="border-b border-rule2 align-top">
            <td className="py-2.5 pr-2 text-[12px] font-bold text-muted">{f.id}</td>
            <td className="py-2.5 pr-3">
              <div className="leading-snug">{f.label}</div>
              {f.status && f.status !== "measured" && <div className="mt-0.5 text-[13px] font-semibold text-warn">{f.status === "contested" ? "Contested" : "Overturned"}</div>}
              <div className="mt-1 flex gap-4 text-[13px]">
                {f.t != null && <button onClick={() => onJump(f.t!)} className="link">Show at {f.t.toFixed(1)} s</button>}
                <button onClick={() => onDispute(f)} className="link">Dispute</button>
              </div>
            </td>
            <td className="py-2.5"><ConfidenceBar c={f.confidence} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Position checks on the replay itself (lib.sceneChecks). */
function ReconstructionCheck({ scene }: { scene: Scene }) {
  const issues = sceneChecks(scene);
  if (!issues.length) {
    return (
      <p className="text-[13px] text-ink2">
        Position check passed: both vehicles stay on the road, travel the right way in their lanes, and their outlines
        meet at the impact.
      </p>
    );
  }
  return (
    <div className="border border-sign bg-warnbg p-4 text-[14px] text-warn">
      <b>Check the replay against the video.</b>
      <ul className="mt-1 list-disc pl-5">{issues.map((m) => <li key={m}>{m}</li>)}</ul>
    </div>
  );
}

/** Nemotron VL's independent look at the frames, plus the reconstruction's own quality warnings. */
function ReviewBanner({ review }: { review?: Review }) {
  if (!review) return null;
  const cc = review.crosscheck;
  const vlm = cc?.vlm;
  const seen = vlm ? `${vlm.crash_type.replace(/_/g, " ")}, dashcam car ${vlm.dashcam_vehicle} (${Math.round(vlm.confidence * 100)}%)` : null;
  const model = cc?.model?.split("/").pop();
  if (review.needs_review) {
    return (
      <div className="border border-bad bg-badbg p-4 text-[14px] text-bad">
        <b>Needs review: don't rely on this split yet.</b>
        <ul className="mt-1 list-disc pl-5">{review.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
        {seen && <p className="mt-2 text-[13px]">{model} sees: {seen}.{vlm?.reason ? ` "${vlm.reason}"` : ""}</p>}
      </div>
    );
  }
  if (review.visually_confirmed && seen) {
    return (
      <div className="border border-ok bg-okbg p-4 text-[14px] text-ok">
        <b>Visual check agrees.</b> {model} sees {seen}.{vlm?.reason ? ` "${vlm.reason}"` : ""}
        {review.notes.length > 0 && <p className="mt-1">Note: {review.notes.join(" ")}</p>}
        <p className="mt-1 text-[13px] opacity-80">Informational only; the split comes from the measured facts and rules.</p>
      </div>
    );
  }
  if (cc && cc.status !== "ok") {
    return <p className="text-[13px] text-muted">Visual check unavailable{cc.error ? ` (${cc.error})` : ""}.</p>;
  }
  return null;
}

function Disputes({ verdict }: { verdict: Verdict }) {
  if (!verdict.disputes.length) return null;
  return (
    <section className="mt-6">
      <h3 className="text-[15px] font-bold">Disputes on record</h3>
      <p className="text-[13px] text-muted">Each dispute is added to the record. Nothing is overwritten.</p>
      <ol className="mt-2 divide-y divide-rule2 border-y border-rule2 text-[14px]">
        {verdict.disputes.map((d) => (
          <li key={d.id} className="py-2">
            {d.fact_id}: "{fmtVal(d.old_value)}" disputed as "{fmtVal(d.new_value)}", backed by {d.evidence_type.replace("_", " ")}.
            {d.old_split != null && d.new_split != null && (
              <span className="ml-1 font-semibold">A moved from {Math.round(d.old_split * 100)}% to {Math.round(d.new_split * 100)}%.</span>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}

function CalibratePanel({ onApplied, applied }: { onApplied: () => void; applied: boolean }) {
  const [res, setRes] = useState<CalibrationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (apply: boolean) => {
    setBusy(true);
    try { setRes(await api.fitWeights(apply)); if (apply) onApplied(); } finally { setBusy(false); }
  };
  return (
    <div className="space-y-4 text-[14px]">
      <p>
        The rule weights are reasoned estimates today. Each resolved case, whether decided by an adjuster, an
        arbitrator or a court, can become a training example; fitting the weights to those decisions is how they get
        calibrated.
      </p>
      <p className="bg-warnbg p-3 text-warn">This demonstration fits the weights to 10 <b>simulated</b> outcomes, not real decisions.</p>
      <div className="flex flex-wrap gap-2">
        <button disabled={busy} onClick={() => run(false)} className="btn">Fit weights</button>
        {res && <button disabled={busy} onClick={() => run(true)} className="btn-line">Use fitted weights for this case</button>}
        {applied && <button onClick={async () => { await api.resetWeights(); onApplied(); }} className="link">Go back to hand-set weights</button>}
      </div>
      {res && (
        <>
          <p className="text-ink2">Average error against the decided splits went from <b>{res.mae_before_pct}%</b> to <b>{res.mae_after_pct}%</b> across {res.n_cases} cases.</p>
          <table className="w-full text-[13px]">
            <thead><tr className="border-b border-rule text-left text-ink2"><th className="py-1 font-semibold">Rule</th><th className="font-semibold">Before</th><th className="font-semibold">After</th></tr></thead>
            <tbody>
              {res.weights.map((w) => (
                <tr key={w.rule_id} className="border-b border-rule2">
                  <td className="py-1.5 pr-2">{w.title}</td><td>{w.before.toFixed(2)}</td><td className="font-bold">{w.after.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
