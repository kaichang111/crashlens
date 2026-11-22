import type { CaseStatus, FramesMeta, Scene, Verdict } from "./types";

export const API = "/api";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${API}/health`).then(j<{ ok: boolean; capabilities: Record<string, boolean> }>),
  createCase: (form: FormData, cached?: string | null) =>
    fetch(`${API}/cases${cached ? `?cached=${encodeURIComponent(cached)}` : ""}`, { method: "POST", body: form })
      .then(j<{ id: string; evidence_sha256: string; received_at: string; cached: boolean }>),
  getCase: (id: string) => fetch(`${API}/cases/${id}`).then(j<Record<string, any>>),
  status: (id: string) => fetch(`${API}/cases/${id}/status`).then(j<CaseStatus>),
  frames: (id: string) => fetch(`${API}/cases/${id}/frames`).then(j<FramesMeta>),
  frameUrl: (id: string, i: number) => `${API}/cases/${id}/frames/${i}.jpg`,
  previewUrl: (id: string, i: number, k1: number) => `${API}/cases/${id}/preview/${i}.jpg?k1=${k1.toFixed(2)}`,
  videoUrl: (id: string) => `${API}/cases/${id}/video`,
  calibrate: (id: string, body: unknown) =>
    fetch(`${API}/cases/${id}/calibration`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j<{ ok: boolean }>),
  scene: (id: string) => fetch(`${API}/cases/${id}/scene`).then(j<Scene>),
  verdict: (id: string) => fetch(`${API}/cases/${id}/verdict`).then(j<Verdict>),
  dispute: (id: string, body: { fact_id: string; new_value: string | number | boolean; evidence_type: string; evidence_sha256?: string | null; note?: string }) =>
    fetch(`${API}/cases/${id}/disputes`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(j<{ verdict: Verdict; old_split: Record<string, number> | null; changed_rules: string[] }>),
  resetDisputes: (id: string) => fetch(`${API}/cases/${id}/disputes`, { method: "DELETE" }).then(j<{ ok: boolean }>),
  uploadEvidence: (id: string, file: File) => {
    const fd = new FormData(); fd.append("file", file);
    return fetch(`${API}/cases/${id}/evidence`, { method: "POST", body: fd }).then(j<{ evidence_sha256: string }>);
  },
  fitWeights: (apply: boolean) =>
    fetch(`${API}/calibrate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ apply }) }).then(j<CalibrationResult>),
  resetWeights: () => fetch(`${API}/calibrate`, { method: "DELETE" }).then(j<{ ok: boolean }>),
};

export interface CalibrationResult {
  n_cases: number; simulated: boolean; applied: boolean;
  weights: { rule_id: string; title: string; before: number; after: number }[];
  mae_before_pct: number; mae_after_pct: number;
  cases: { id: string; decided_by: string; decided_A: number; predicted_before: number; predicted_after: number }[];
}
