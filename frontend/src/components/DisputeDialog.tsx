import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { DISPUTE_OPTIONS, EVIDENCE, fmtVal } from "../lib";
import type { Fact } from "../types";

/** Dispute a fact (never a percentage): pick the correction and the evidence; evidence type sets its weight. */
export default function DisputeDialog({ caseId, fact, onClose, onDone }: {
  caseId: string; fact: Fact; onClose: () => void;
  onDone: (r: Awaited<ReturnType<typeof api.dispute>>) => void;
}) {
  const isBool = typeof fact.value === "boolean";
  const isNum = typeof fact.value === "number";
  const opts = DISPUTE_OPTIONS[fact.type]?.filter((o) => o !== fact.value);
  const [value, setValue] = useState<string>(isBool ? String(!fact.value) : opts?.[0] ?? String(fact.value));
  const [ev, setEv] = useState("user_word");
  const [file, setFile] = useState<File | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const first = useRef<HTMLSelectElement & HTMLInputElement>(null);
  useEffect(() => {
    first.current?.focus();
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  async function submit() {
    setBusy(true); setErr(null);
    try {
      let sha: string | null = null;
      if (file && ev !== "user_word") sha = (await api.uploadEvidence(caseId, file)).evidence_sha256;
      const nv = isBool ? value === "true" : isNum ? Number(value) : value;
      onDone(await api.dispute(caseId, { fact_id: fact.id, new_value: nv, evidence_type: ev, evidence_sha256: sha, note }));
    } catch (e) { setErr(`The dispute wasn't saved: ${(e as Error).message}`); } finally { setBusy(false); }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-labelledby="dispute-title"
           className="w-full max-w-lg border border-ink bg-panel p-6" onClick={(e) => e.stopPropagation()}>
        <h2 id="dispute-title" className="text-[20px] font-extrabold">Dispute fact {fact.id}</h2>
        <p className="mt-1 text-[15px]">{fact.label}</p>
        <p className="mt-1 text-[13px] text-ink2">The video measured <b>{fmtVal(fact.value)}</b> with {Math.round(fact.confidence * 100)}% confidence.</p>

        <label className="lbl mt-5 block" htmlFor="dispute-value">What it should be</label>
        {isBool ? (
          <select id="dispute-value" ref={first} className="field" value={value} onChange={(e) => setValue(e.target.value)}>
            <option value="true">Yes</option><option value="false">No</option>
          </select>
        ) : opts ? (
          <select id="dispute-value" ref={first} className="field" value={value} onChange={(e) => setValue(e.target.value)}>
            {opts.map((o) => <option key={o} value={o}>{fmtVal(o)}</option>)}
          </select>
        ) : (
          <input id="dispute-value" ref={first} className="field" value={value} onChange={(e) => setValue(e.target.value)}
                 type={isNum ? "number" : "text"} step="0.1" />
        )}

        <fieldset className="mt-4">
          <legend className="lbl">Your evidence</legend>
          <div className="mt-1 grid grid-cols-2 gap-2">
            {EVIDENCE.map((e) => (
              <label key={e.id} className={`flex cursor-pointer items-center justify-between rounded-[3px] border px-3 py-2 text-[14px] ${ev === e.id ? "border-ink bg-rule2" : "border-rule"}`}>
                <span><input type="radio" name="evidence" className="mr-2 accent-[#1e2226]" checked={ev === e.id} onChange={() => setEv(e.id)} />{e.label}</span>
                <span className="text-[12px] text-muted">weight {e.weight}</span>
              </label>
            ))}
          </div>
        </fieldset>
        {ev !== "user_word" && (
          <input type="file" aria-label="Attach evidence" className="mt-2 block w-full text-[13px] text-ink2" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        )}
        <label className="lbl mt-4 block" htmlFor="dispute-note">Note (optional)</label>
        <textarea id="dispute-note" className="field h-16" value={note} onChange={(e) => setNote(e.target.value)} />
        <p className="mt-3 text-[13px] text-ink2">
          If your evidence outweighs the video's confidence, the fact changes. Otherwise it's marked contested and both
          versions go in the report. The original measurement is always kept.
        </p>
        {err && <p className="mt-3 bg-badbg p-2 text-[13px] text-bad">{err}</p>}
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="btn-line">Cancel</button>
          <button disabled={busy} onClick={submit} className="btn">{busy ? "Recalculating…" : "Submit dispute"}</button>
        </div>
      </div>
    </div>
  );
}
