import type { Signal, TrackPoint } from "./types";

export const MPH = 2.237;

/** Linear interpolation of position and heading between the two samples around t. */
export function sample(track: TrackPoint[], t: number): TrackPoint {
  if (!track.length) return { t, x: 0, y: 0, heading: 0, speed: 0, sigma: 0 };
  if (t <= track[0].t) return track[0];
  if (t >= track[track.length - 1].t) return track[track.length - 1];
  let lo = 0, hi = track.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (track[m].t <= t) lo = m; else hi = m; }
  const a = track[lo], b = track[hi], u = (t - a.t) / (b.t - a.t);
  let dh = b.heading - a.heading;
  dh = Math.atan2(Math.sin(dh), Math.cos(dh));
  return {
    t, x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, heading: a.heading + dh * u,
    speed: a.speed + (b.speed - a.speed) * u, sigma: a.sigma + (b.sigma - a.sigma) * u,
  };
}

/**
 * Position sanity checks on a reconstruction (same tests as backend/scripts/audit_scenes.py): the replay
 * is only as good as these, so problems are shown to the user instead of drawn as if measured.
 */
export function sceneChecks(scene: import("./types").Scene): string[] {
  const out: string[] = [];
  const half = scene.road.lanes_per_dir * scene.road.lane_width_m;
  const inter = scene.road.template.startsWith("intersection");
  const oneway = scene.road.template === "straight_oneway"; // lanes at x ∈ [0, half], all heading north
  const onRoad = (x: number) => (oneway ? x >= -0.3 && x <= half + 0.3 : Math.abs(x) <= half + 0.3);
  const tImp = scene.impact?.t ?? Infinity;
  for (const v of scene.vehicles) {
    const pre = v.track.filter((p) => p.t <= tImp);
    if (!pre.length) continue;
    const off = pre.filter((p) => !(onRoad(p.x) || (inter && Math.abs(p.y) <= half + 0.3))).length / pre.length;
    if (off > 0.1) out.push(`${v.id} is drawn off the road for ${Math.round(off * 100)}% of the time before impact. Its far-away positions are the least certain; compare with the video.`);
    const appr = pre.filter((p) => onRoad(p.x) && (!inter || Math.abs(p.y) > half + 3) && p.speed > 1);
    if (appr.length >= 5) {
      const wrong = appr.filter((p) => {
        const want = oneway || p.x > 0 ? Math.PI / 2 : -Math.PI / 2; // right-hand traffic
        const d = Math.abs(Math.atan2(Math.sin(p.heading - want), Math.cos(p.heading - want)));
        return d > Math.PI / 2;
      }).length / appr.length;
      if (wrong > 0.3) out.push(`${v.id} is drawn driving against its lane for part of the approach. The reconstruction may have its position or direction wrong.`);
    }
  }
  if ((scene.impact as { inferred?: boolean } | null)?.inferred) {
    out.push("The two outlines don't quite meet at the moment the camera shows the crash, so the impact point is estimated from where they came closest.");
  }
  return out;
}

export function signalAt(s: Signal, t: number): string {
  let st = "unknown";
  for (const x of s.states) if (x.t <= t + 1e-6) st = x.s;
  return st;
}

/** Scene metres (x east, y north) → three.js (x, y-up, z south). */
export const w3 = (x: number, y: number, h = 0): [number, number, number] => [x, h, -y];

export const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
export const fmtVal = (v: unknown) =>
  typeof v === "boolean" ? (v ? "yes" : "no") : typeof v === "number" ? (Math.round(v * 100) / 100).toString() : String(v).replace(/_/g, " ");

export const DISPUTE_OPTIONS: Record<string, string[]> = {
  signal_state: ["red", "yellow", "green", "green_arrow"],
  impact_type: ["rear_end", "angle", "sideswipe", "head_on"],
  contact_point: ["front", "front_left", "front_right", "left_side", "right_side", "rear", "rear_left", "rear_right"],
};

export const EVIDENCE = [
  { id: "user_word", label: "My word only", weight: 0.3 },
  { id: "witness", label: "Witness statement", weight: 0.6 },
  { id: "photo", label: "Photo", weight: 0.7 },
  { id: "second_video", label: "Second video", weight: 0.9 },
];
