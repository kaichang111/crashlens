export type DriverId = "A" | "B";

export interface TrackPoint { t: number; x: number; y: number; heading: number; speed: number; sigma: number }
export interface Vehicle { id: DriverId; type: string; size: [number, number]; color: string; track: TrackPoint[] }
export interface SignalState { t: number; s: string }
export interface Signal {
  id: string; pos: [number, number]; faces: DriverId; approach_heading: number;
  stop_line: number; protected_left?: boolean; states: SignalState[];
}
export interface Sign { id: string; pos: [number, number]; faces: DriverId; approach_heading: number; stop_line: number }
export interface Impact {
  t: number; pos: [number, number]; type: string; closing_speed: number; angle_deg: number;
  contact: Record<DriverId, string>;
}
export interface Scene {
  evidence_sha256: string;
  road: { template: string; lanes_per_dir: number; lane_width_m: number; speed_limit_mph?: number };
  signals: Signal[]; signs: Sign[]; vehicles: Vehicle[]; impact: Impact | null; duration: number;
  meta: Record<string, unknown> & { clip_offset?: number };
}

export interface Fact {
  id: string; actor: DriverId | "both"; type: string; t: number | null; value: string | number | boolean;
  unit?: string | null; confidence: number; source: string; label: string;
  evidence: { frames: number[] }; status?: "measured" | "contested" | "overturned";
  original?: { value: unknown; confidence: number }; contested_value?: unknown; dispute_strength?: number;
  stopping_distance_m?: number; gap_m?: number;
}
export interface FiredRule {
  rule_id: string; title: string; statute: string; url: string; charged: DriverId; primary: boolean;
  weight: number; confidence: number; causation: number; contribution: number; explanation: string;
  fact_ids: string[]; contested: boolean;
}
export interface Dispute {
  id: number; fact_id: string; actor: string; old_value: unknown; new_value: unknown; evidence_type: string;
  evidence_sha256: string | null; old_split: number | null; new_split: number | null; note: string; created_at: string;
}
export interface Review {
  needs_review: boolean; reasons: string[]; notes: string[]; visually_confirmed: boolean;
  crosscheck: {
    status: "ok" | "unavailable" | "error"; model: string; error?: string;
    vlm: { crash_type: string; dashcam_vehicle: string; other_vehicle_direction?: string;
           traffic_light_for_dashcam?: string; confidence: number; reason?: string } | null;
  } | null;
}
export interface Verdict {
  status: "ok" | "insufficient_evidence"; case_id: string; your_vehicle: DriverId;
  evidence_sha256: string; received_at: string; location_type: string;
  facts: Fact[]; fired_rules: FiredRule[];
  scores: Record<DriverId, number>;
  split: Record<DriverId, number> | null; range: Record<DriverId, [number, number]> | null;
  recovery: { claimant_share: number; recoverable_pct: number; barred: boolean; text: string } | null;
  narrative?: { text: string; source: "llm" | "rule_cards"; problems: string[]; model?: string };
  review?: Review;
  disputes: Dispute[];
  claims: { actor: string; type: string; value: unknown; status: string; fact_id?: string; video_value?: unknown }[];
  calibrated_weights: boolean;
}
export interface CaseStatus { status: string; stage: string; percent: number; error: string | null }
export interface FramesMeta { fps: number; clip_offset: number; impact_t: number; width: number; height: number; frames: { i: number; t: number }[] }
export type View = "top" | "driverA" | "driverB" | "orbit";
