import { fmtVal } from "../lib";
import type { Fact, FiredRule } from "../types";

export function ConfidenceBar({ c }: { c: number }) {
  const pct = Math.round(c * 100);
  const col = c >= 0.8 ? "bg-ok" : c >= 0.6 ? "bg-sign" : "bg-bad";
  return (
    <div className="flex items-center gap-2" title={`${pct}% confidence`}>
      <div className="h-[6px] w-14 bg-rule2"><div className={`h-full ${col}`} style={{ width: `${pct}%` }} /></div>
      <span className="w-9 text-right text-[12px] text-ink2">{pct}%</span>
    </div>
  );
}

/** One rule that applied: the statute, the plain sentence, what it adds, and the facts behind it. */
export default function RuleCard({ rule, facts, changed, onJump, onDispute }: {
  rule: FiredRule; facts: Fact[]; changed?: boolean;
  onJump: (t: number) => void; onDispute: (f: Fact) => void;
}) {
  const used = rule.fact_ids.map((id) => facts.find((f) => f.id === id)).filter(Boolean) as Fact[];
  return (
    <article className={`border-b border-rule py-5 ${changed ? "bg-warnbg" : ""}`}>
      <div className="flex items-start gap-3">
        <span className={`veh veh-${rule.charged} mt-0.5`} title={`Applies to driver ${rule.charged}`}>{rule.charged}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4">
            <h3 className="text-[16px] font-bold leading-snug">{rule.title}</h3>
            <span className="text-[14px] font-bold">+{rule.contribution.toFixed(2)} to {rule.charged}</span>
          </div>
          {rule.url
            ? <a href={rule.url} target="_blank" rel="noreferrer" className="link text-[13px] font-semibold">{rule.statute}</a>
            : <span className="text-[13px] text-ink2">{rule.statute}</span>}
          <p className="mt-2 text-[15px] leading-relaxed">{rule.explanation}</p>
          <p className="mt-1 text-[12px] text-muted">
            Weight {rule.weight.toFixed(2)}, times confidence {rule.confidence.toFixed(2)}, times causation {rule.causation.toFixed(1)}
            {rule.contested && <span className="ml-2 font-semibold text-warn">Relies on a contested fact</span>}
          </p>
          <ul className="mt-3 divide-y divide-rule2 border-y border-rule2">
            {used.map((f) => (
              <li key={f.id} className="py-2.5">
                <div className="flex items-start justify-between gap-3">
                  <span className="text-[14px] leading-snug"><span className="mr-1.5 text-[12px] font-bold text-muted">{f.id}</span>{f.label}</span>
                  <ConfidenceBar c={f.confidence} />
                </div>
                {f.status && f.status !== "measured" && (
                  <div className="mt-1 text-[13px] font-semibold text-warn">
                    {f.status === "contested" ? `Contested: the other version says "${fmtVal(f.contested_value)}"` : `Overturned: the video measured "${fmtVal(f.original?.value)}"`}
                  </div>
                )}
                <div className="mt-1.5 flex gap-4 text-[13px]">
                  {f.t != null && <button onClick={() => onJump(f.t!)} className="link">Show at {f.t.toFixed(1)} s</button>}
                  <button onClick={() => onDispute(f)} className="link">Dispute this fact</button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </article>
  );
}
