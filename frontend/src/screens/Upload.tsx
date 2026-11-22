import { useEffect, useState } from "react";
import { api } from "../api";
import { go } from "../router";
import { useClock } from "../store";
import Replay from "../three/Replay";
import type { Scene } from "../types";

const LOCATIONS = [
  ["intersection_signal", "Intersection with a traffic signal"],
  ["intersection_stop", "Intersection with a stop sign"],
  ["multi_lane_road", "Multi-lane road, two-way"],
  ["divided_highway", "Highway, all lanes one direction"],
  ["parking_lot", "Parking lot"],
] as const;

/** The real left-turn demo, replaying from above: the product, not an illustration of it. */
function LiveDemo() {
  const [scene, setScene] = useState<Scene | null>(null);
  useEffect(() => {
    let alive = true;
    api.scene("demo-left-turn").then((s) => {
      if (!alive) return;
      setScene(s);
      const c = useClock.getState();
      c.setDuration(s.duration); c.setView("top");
      c.setT(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? (s.impact?.t ?? 0) : 0);
    }).catch(() => {});
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    if (!scene || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    let id = 0, last: number | null = null;
    const loop = (now: number) => {
      const c = useClock.getState();
      if (last != null) {
        let nt = c.t + ((now - last) / 1000) * 0.6;
        if (nt > scene.duration) nt = 0;
        c.setT(nt);
      }
      last = now;
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [scene]);
  return (
    <figure>
      <div className="aspect-[4/3] bg-asphalt">{scene ? <Replay scene={scene} view="top" /> : <div className="skeleton h-full w-full" />}</div>
      <figcaption className="mt-2 text-[13px] text-ink2">
        The left-turn demo, rebuilt from its dashcam clip. A (blue) goes straight; B (red) turns across its path.
        The rules put 90% on B. <button className="link" onClick={() => go("/results/demo-left-turn")}>Open the full case</button>
      </figcaption>
    </figure>
  );
}

/** Screen 1: the clip and the four questions from the spec (§1.1). */
export default function Upload() {
  const hiddenCached = new URLSearchParams(window.location.search).get("cached"); // backup switch
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [loc, setLoc] = useState<string>("intersection_signal");
  const [yours, setYours] = useState("A");
  const [desc, setDesc] = useState("");
  const [limit, setLimit] = useState("");
  const [weather, setWeather] = useState("clear");
  const [surface, setSurface] = useState("dry");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(cached?: string) {
    setBusy(true); setErr(null);
    try {
      const fd = new FormData();
      if (file && !cached) fd.append("video", file);
      fd.append("location_type", cached === "rear_end" ? "multi_lane_road" : cached === "left_turn" ? "intersection_signal" : loc);
      fd.append("your_vehicle", yours);
      fd.append("description", desc);
      if (limit) fd.append("speed_limit_mph", limit);
      fd.append("weather", weather);
      fd.append("surface", surface);
      const r = await api.createCase(fd, cached ?? hiddenCached);
      go(r.cached ? `/results/${r.id}` : `/calibrate/${r.id}`);
    } catch (e) {
      setErr(`The upload didn't go through: ${(e as Error).message}. Check the file is an mp4 or mov under 300 MB and try again.`);
    } finally { setBusy(false); }
  }

  return (
    <div className="mx-auto max-w-[1440px] px-5">
      <section className="grid items-center gap-10 py-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className="max-w-[34rem]">
          <h1 className="text-[44px] font-extrabold leading-[1.05] tracking-tight sm:text-[54px]">
            Show the adjuster what happened.
          </h1>
          <p className="mt-5 text-[18px] leading-relaxed text-ink2">
            Upload your dashcam clip. CrashLens rebuilds the crash in 3D, measures speeds, gaps and light states, and
            applies the Texas traffic laws to give a fault split you can check line by line. If a fact is wrong,
            dispute it and watch the split recalculate.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <a href="#upload" className="btn">Upload a clip</a>
            <button className="btn-line" onClick={() => go("/results/demo-left-turn")}>Explore the demo case</button>
          </div>
        </div>
        <LiveDemo />
      </section>

      <section id="upload" className="border-t border-rule py-12">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <div className="max-w-[34rem]">
            <h2 className="text-[30px] font-extrabold leading-tight tracking-tight">Start a case</h2>
            <p className="mt-3 text-[16px] text-ink2">
              The original file is fingerprinted with SHA-256 the moment it arrives, so anyone can later check it hasn't
              been edited. Your answers below help pick the right road layout and rules.
            </p>
            <ol className="mt-8 space-y-5">
              {[
                ["Upload the clip", "A few seconds before the crash through a couple after is enough."],
                ["Mark one lane", "Click four corners of one lane on the first frame so distances can be measured."],
                ["Point to the other vehicle", "Tap it on the impact frame. Reconstruction takes about a minute."],
                ["Review the result", "Scrub the replay, check each fact, dispute anything wrong, then print the report."],
              ].map(([h, d], i) => (
                <li key={h} className="flex gap-4">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center border-2 border-ink text-[15px] font-extrabold">{i + 1}</span>
                  <div><div className="text-[16px] font-bold">{h}</div><div className="text-[14px] text-ink2">{d}</div></div>
                </li>
              ))}
            </ol>
          </div>

          <form className="border border-rule bg-panel p-6" onSubmit={(e) => { e.preventDefault(); submit(); }}>
            <label onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
                   onDrop={(e) => { e.preventDefault(); setDrag(false); setFile(e.dataTransfer.files?.[0] ?? null); }}
                   className={`flex cursor-pointer flex-col items-center justify-center border-2 border-dashed px-6 py-10 text-center ${drag ? "border-ink bg-rule2" : "border-rule"}`}>
              <input type="file" accept="video/mp4,video/quicktime" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <span className="text-[17px] font-bold">{file ? file.name : "Choose or drop your dashcam clip"}</span>
              <span className="mt-1 text-[14px] text-ink2">{file ? `${(file.size / 1e6).toFixed(1)} MB` : "mp4 or mov, up to about a minute"}</span>
            </label>

            <div className="mt-6 grid gap-x-5 gap-y-4 sm:grid-cols-2">
              <label className="block"><span className="lbl">Where did it happen?</span>
                <select className="field" value={loc} onChange={(e) => setLoc(e.target.value)}>
                  {LOCATIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </label>
              <label className="block"><span className="lbl">Which vehicle were you in?</span>
                <select className="field" value={yours} onChange={(e) => setYours(e.target.value)}>
                  <option value="A">The car with the dashcam</option>
                  <option value="B">The other vehicle</option>
                </select>
              </label>
              <label className="block sm:col-span-2"><span className="lbl">What happened, in your words</span>
                <textarea className="field h-24" value={desc} onChange={(e) => setDesc(e.target.value)}
                          placeholder="I was going straight on green and they turned left in front of me." />
              </label>
              <label className="block"><span className="lbl">Posted speed limit, if you know it</span>
                <input className="field" type="number" inputMode="numeric" value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="mph" />
              </label>
              <div className="grid grid-cols-2 gap-3">
                <label className="block"><span className="lbl">Weather</span>
                  <select className="field" value={weather} onChange={(e) => setWeather(e.target.value)}>
                    <option value="clear">Clear</option><option value="cloudy">Cloudy</option><option value="rain">Rain</option>
                  </select>
                </label>
                <label className="block"><span className="lbl">Road surface</span>
                  <select className="field" value={surface} onChange={(e) => setSurface(e.target.value)}>
                    <option value="dry">Dry</option><option value="wet">Wet</option>
                  </select>
                </label>
              </div>
            </div>

            {err && <p className="mt-5 bg-badbg p-3 text-[14px] text-bad">{err}</p>}
            <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-3">
              <button type="submit" disabled={busy || (!file && !hiddenCached)} className="btn">
                {busy ? "Uploading…" : "Upload and continue"}
              </button>
              <span className="text-[14px] text-ink2">
                No clip handy? Try a demo:{" "}
                <button type="button" disabled={busy} onClick={() => submit("left_turn")} className="link">left turn</button>{" or "}
                <button type="button" disabled={busy} onClick={() => submit("rear_end")} className="link">rear-end</button>.
              </span>
            </div>
          </form>
        </div>
      </section>
    </div>
  );
}
