import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, useGLTF } from "@react-three/drei";
import { Suspense, useLayoutEffect, useMemo, useRef, type ReactElement } from "react";
import * as THREE from "three";
import { sample, signalAt, w3, MPH } from "../lib";
import { useClock } from "../store";
import type { Impact, Scene, Sign, Signal, Vehicle, View } from "../types";

/** Low-poly models built in Blender (crashlens/blender/crashlens_models.blend), exported as glTF. */
const MODELS = {
  car: "/models/car.glb",
  pickup: "/models/pickup.glb",
  signal: "/models/traffic_signal.glb",
  stop: "/models/stop_sign.glb",
};
Object.values(MODELS).forEach((m) => useGLTF.preload(m));

// Same surface colours as the Blender scenes the demo clips were rendered from, so the replay and the
// video read as the same place.
const C = { asphalt: "#2f3236", walk: "#a6a8a4", grass: "#557134", white: "#ecece6", yellow: "#d9a514", sky: "#b9cfe6" };

/* ---------------------------------------------------------------- road (procedural, spec §3.7) */
function Strip({ x, y, w, l, color, h = 0.02 }: { x: number; y: number; w: number; l: number; color: string; h?: number }) {
  return (
    <mesh position={w3(x, y, h)} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[w, l]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

function dashes(axis: "ns" | "ew", offset: number, from: number, to: number) {
  const out: ReactElement[] = [];
  for (let d = from; d < to; d += 12.19) { // 10 ft dash + 30 ft gap
    const c = d + 1.52;
    out.push(axis === "ns"
      ? <Strip key={`${axis}${offset}${d}`} x={offset} y={c} w={0.12} l={3.05} color={C.white} />
      : <Strip key={`${axis}${offset}${d}`} x={c} y={offset} w={3.05} l={0.12} color={C.white} />);
  }
  return out;
}

/**
 * The paved geometry, built from the scene's own numbers: lane width and count, and the stop-line
 * distance the facts use (signal.stop_line / sign.stop_line). Drawing the stop line anywhere else would
 * put the "crossed on red" moment in a different place from where it is measured.
 */
function Road({ scene }: { scene: Scene }) {
  const road = scene.road;
  const lw = road.lane_width_m, n = road.lanes_per_dir, half = n * lw, L = 160;
  if (road.template.startsWith("intersection")) {
    const control = [...scene.signals, ...scene.signs];
    const stop = control.length ? Math.max(...control.map((s) => Math.abs(s.stop_line))) : half + 1.0;
    const arms: ReactElement[] = [];
    for (const s of [1, -1]) {
      for (const o of [-0.12, 0.12]) {
        arms.push(<Strip key={`yn${s}${o}`} x={o} y={s * (half + 1.5 + L / 2)} w={0.1} l={L} color={C.yellow} />);
        arms.push(<Strip key={`ye${s}${o}`} x={s * (half + 1.5 + L / 2)} y={o} w={L} l={0.1} color={C.yellow} />);
      }
      for (let k = 1; k < n; k++) for (const side of [1, -1]) {
        const [a, b] = s > 0 ? [half + 3, half + 3 + L] : [-half - 3 - L, -half - 3];
        arms.push(...dashes("ns", side * k * lw, a, b));
        arms.push(...dashes("ew", side * k * lw, a, b));
      }
    }
    // A crosswalk only fits between the junction and the stop line when the stop line is set back.
    const cw: ReactElement[] = [];
    if (stop >= half + 3.2) {
      const mid = half + (stop - half) / 2 - 0.2, len = Math.min(3.0, stop - half - 0.8);
      for (let k = -half + 0.6; k < half; k += 1.2) {
        cw.push(<Strip key={`cn${k}`} x={k} y={mid} w={0.6} l={len} color={C.white} />);
        cw.push(<Strip key={`cs${k}`} x={k} y={-mid} w={0.6} l={len} color={C.white} />);
        cw.push(<Strip key={`ce${k}`} x={mid} y={k} w={len} l={0.6} color={C.white} />);
        cw.push(<Strip key={`cw${k}`} x={-mid} y={k} w={len} l={0.6} color={C.white} />);
      }
    }
    return (
      <group>
        <Strip x={0} y={0} w={2 * half} l={2 * L} color={C.asphalt} h={0.005} />
        <Strip x={0} y={0} w={2 * L} l={2 * half} color={C.asphalt} h={0.006} />
        {[[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([sx, sy]) => (
          <Strip key={`walk${sx}${sy}`} x={sx * (half + L / 2 + 0.01)} y={sy * (half + L / 2 + 0.01)} w={L} l={L} color={C.walk} h={0.004} />
        ))}
        {/* stop lines on the approach (right-hand) half of each arm, at the measured distance */}
        <Strip x={half / 2} y={-stop} w={half} l={0.4} color={C.white} />
        <Strip x={-half / 2} y={stop} w={half} l={0.4} color={C.white} />
        <Strip x={stop} y={half / 2} w={0.4} l={half} color={C.white} />
        <Strip x={-stop} y={-half / 2} w={0.4} l={half} color={C.white} />
        {arms}
        {cw}
      </group>
    );
  }
  if (road.template === "straight_oneway") {
    // divided highway: every lane runs north (+y), lanes at x ∈ [0, n·lw]; median and guardrail on the left
    return (
      <group>
        <Strip x={half / 2} y={0} w={half + 0.6} l={2 * L} color={C.asphalt} h={0.005} />
        <Strip x={-2.6} y={0} w={4.4} l={2 * L} color={C.grass} h={0.004} />
        <Strip x={-0.7} y={0} w={0.25} l={2 * L} color={C.walk} h={0.6} />
        <Strip x={half + 3} y={0} w={5} l={2 * L} color={C.walk} h={0.004} />
        {[0, half].map((x) => <Strip key={`edge${x}`} x={x} y={0} w={0.15} l={2 * L} color={C.white} />)}
        {Array.from({ length: n - 1 }, (_, k) => dashes("ns", (k + 1) * lw, -L, L))}
      </group>
    );
  }
  return (
    <group>
      <Strip x={0} y={0} w={2 * half + 0.6} l={2 * L} color={C.asphalt} h={0.005} />
      {[1, -1].map((s) => <Strip key={`sw${s}`} x={s * (half + 3)} y={0} w={5} l={2 * L} color={C.walk} h={0.004} />)}
      {[1, -1].map((s) => <Strip key={`edge${s}`} x={s * (half + 0.1)} y={0} w={0.15} l={2 * L} color={C.white} />)}
      {[-0.12, 0.12].map((o) => <Strip key={`y${o}`} x={o} y={0} w={0.1} l={2 * L} color={C.yellow} />)}
      {Array.from({ length: n - 1 }, (_, k) => [1, -1].flatMap((side) => dashes("ns", side * (k + 1) * lw, -L, L)))}
    </group>
  );
}

/* ---------------------------------------------------------------- models */
function Car({ v, label = true }: { v: Vehicle; label?: boolean }) {
  const { scene: src } = useGLTF(v.type === "pickup" ? MODELS.pickup : MODELS.car);
  const [L, W] = v.size;
  const { obj, mats, scale } = useMemo(() => {
    const o = src.clone(true);
    const mats: THREE.MeshStandardMaterial[] = [];
    o.traverse((c) => {
      const m = c as THREE.Mesh;
      if (!m.isMesh) return;
      m.castShadow = true;
      const mat = (m.material as THREE.MeshStandardMaterial).clone();
      if (mat.name === "Body") mat.color = new THREE.Color(v.color); // A blue, B red, as in the findings
      m.material = mat;
      mats.push(mat);
    });
    // Draw the vehicle at its measured footprint: the model is built along +X (length) and Z (width).
    const box = new THREE.Box3().setFromObject(o);
    const size = box.getSize(new THREE.Vector3());
    const sx = size.x > 0 ? L / size.x : 1, sz = size.z > 0 ? W / size.z : 1;
    return { obj: o, mats, scale: [sx, (sx + sz) / 2, sz] as [number, number, number] };
  }, [src, v.color, L, W]);
  const ghost = useRef(false);
  const group = useRef<THREE.Group>(null!);
  const band = useRef<THREE.Mesh>(null!);
  const speedRef = useRef<HTMLDivElement>(null);
  useFrame(() => {
    const t = useClock.getState().t;
    const p = sample(v.track, t);
    group.current.position.set(...w3(p.x, p.y));
    group.current.rotation.y = p.heading;
    // honest uncertainty: translucent rectangle grown by the track's sigma (wide far away, tight at impact)
    const s = Math.max(0.05, p.sigma);
    band.current.scale.set(L + 2 * s, W + 2 * s, 1);
    // Before its first observation the vehicle isn't measured: don't pretend it was parked there.
    // Show a faded ghost at the first measured spot instead of inventing an approach path.
    const unseen = v.track.length > 0 && t < v.track[0].t;
    if (unseen !== ghost.current) {
      ghost.current = unseen;
      for (const m of mats) { m.transparent = unseen; m.opacity = unseen ? 0.18 : 1; m.depthWrite = !unseen; m.needsUpdate = true; }
      band.current.visible = !unseen;
    }
    if (speedRef.current) {
      speedRef.current.textContent = unseen ? `${v.id}, not visible yet` : `${v.id}  ${Math.round(p.speed * MPH)} mph`;
      speedRef.current.style.opacity = unseen ? "0.6" : "1";
    }
  });
  return (
    <group ref={group}>
      <primitive object={obj} scale={scale} />
      <mesh ref={band} position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color={v.color} transparent opacity={0.22} depthWrite={false} />
      </mesh>
      {label && (
        <Html position={[0, 2.6, 0]} center zIndexRange={[10, 0]}>
          <div ref={speedRef} className="-translate-y-7 whitespace-pre rounded-[2px] px-1.5 pt-[3px] pb-[1px] font-sans text-[12px] font-bold text-white"
               style={{ background: v.color }} />
        </Html>
      )}
    </group>
  );
}

const LAMP_COLOR: Record<string, string> = { Lamp_Red: "red", Lamp_Yellow: "yellow", Lamp_Green: "green" };

function SignalHead({ s }: { s: Signal }) {
  const { scene: src } = useGLTF(MODELS.signal);
  const { obj, lamps } = useMemo(() => {
    const o = src.clone(true);
    const lamps: { mat: THREE.MeshStandardMaterial; state: string }[] = [];
    o.traverse((c) => {
      const m = c as THREE.Mesh;
      if (!m.isMesh) return;
      m.castShadow = true;
      const mat = (m.material as THREE.MeshStandardMaterial).clone();
      m.material = mat;
      if (LAMP_COLOR[mat.name]) lamps.push({ mat, state: LAMP_COLOR[mat.name] });
    });
    return { obj: o, lamps };
  }, [src]);
  useFrame(() => {
    const st = signalAt(s, useClock.getState().t); // light colour follows t
    for (const l of lamps) {
      const on = st === l.state || (st === "green_arrow" && l.state === "green");
      l.mat.emissiveIntensity = on ? 4 : 0;
    }
  });
  // model lamps face local −X; rotating by the approach heading points them at oncoming traffic
  return <primitive object={obj} position={w3(s.pos[0], s.pos[1])} rotation={[0, s.approach_heading, 0]} />;
}

function StopSign({ s }: { s: Sign }) {
  const { scene: src } = useGLTF(MODELS.stop);
  const obj = useMemo(() => src.clone(true), [src]);
  return <primitive object={obj} position={w3(s.pos[0], s.pos[1])} rotation={[0, s.approach_heading, 0]} />;
}

/* ---------------------------------------------------------------- overlays */
function Trail({ v }: { v: Vehicle }) {
  const ref = useRef<any>(null);
  const pts = useMemo(() => v.track.map((p) => new THREE.Vector3(...w3(p.x, p.y, 0.06))), [v.track]);
  useFrame(() => {
    const t = useClock.getState().t;
    const i = v.track.findIndex((p) => p.t > t);
    const count = i < 0 ? v.track.length : Math.max(2, i);
    const g = ref.current?.geometry;
    if (g) g.instanceCount = count - 1; // path so far
  });
  return <Line ref={ref} points={pts} color={v.color} lineWidth={3} transparent opacity={0.85} />;
}

function ImpactMarker({ impact }: { impact: Impact & { inferred?: boolean } }) {
  const ref = useRef<THREE.Group>(null!);
  const label = useRef<HTMLDivElement>(null);
  useFrame(() => {
    const visible = useClock.getState().t >= impact.t;
    ref.current.visible = visible;
    if (label.current) label.current.style.display = visible ? "block" : "none"; // DOM overlay ignores .visible
  });
  return (
    <group ref={ref} position={w3(impact.pos[0], impact.pos[1], 0.08)}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.5, 1.85, 48]} />
        <meshBasicMaterial color="#f1b812" />
      </mesh>
      <Html position={[0, 1.5, 0]} center zIndexRange={[10, 0]}>
        <div ref={label} className="whitespace-nowrap rounded-[2px] bg-[#f1b812] px-1.5 pt-[3px] pb-[1px] font-sans text-[12px] font-bold text-[#1e2226]">
          {impact.inferred ? "Impact (estimated)" : "Impact"}
        </div>
      </Html>
    </group>
  );
}

/* ---------------------------------------------------------------- cameras */
function CameraRig({ view, scene }: { view: View; scene: Scene }) {
  const { camera } = useThree();
  const fx = scene.impact?.pos[0] ?? 0, fy = scene.impact?.pos[1] ?? 0;
  const look = useRef(new THREE.Vector3());
  useLayoutEffect(() => {
    if (view === "top" || view === "orbit") {
      const pc = camera as THREE.PerspectiveCamera;
      pc.position.set(fx, view === "top" ? 46 : 22, -fy + (view === "top" ? 0.01 : 26));
      pc.lookAt(fx, 0, -fy);
      pc.fov = view === "top" ? 45 : 50;
      pc.updateProjectionMatrix();
    }
  }, [view, camera, fx, fy]);
  useFrame(() => {
    if (view !== "driverA" && view !== "driverB") return;
    const v = scene.vehicles.find((x) => x.id === (view === "driverA" ? "A" : "B"));
    if (!v) return;
    const p = sample(v.track, useClock.getState().t);
    const hx = Math.cos(p.heading), hy = Math.sin(p.heading);
    camera.position.set(...w3(p.x + hx * 0.3, p.y + hy * 0.3, 1.2)); // driver's eye, 1.2 m up
    look.current.set(...w3(p.x + hx * 20, p.y + hy * 20, 1.0));
    camera.lookAt(look.current);
    const pc = camera as THREE.PerspectiveCamera;
    if (pc.fov !== 75) { pc.fov = 75; pc.updateProjectionMatrix(); }
  });
  return null;
}

/* ---------------------------------------------------------------- canvas */
export default function Replay({ scene, view, preserve = false, labels = true }: {
  scene: Scene; view: View; preserve?: boolean; labels?: boolean;
}) {
  const hideSelf = view === "driverA" ? "A" : view === "driverB" ? "B" : null;
  const target = scene.impact ? w3(scene.impact.pos[0], scene.impact.pos[1]) : ([0, 0, 0] as [number, number, number]);
  return (
    <Canvas shadows camera={{ position: [0, 60, 0.01], fov: 45, near: 0.1, far: 800 }}
            gl={{ preserveDrawingBuffer: preserve, antialias: true }} dpr={[1, 2]}>
      <color attach="background" args={[C.sky]} />
      <fog attach="fog" args={[C.sky, 120, 320]} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[30, 50, 20]} intensity={1.5} castShadow
                        shadow-mapSize={[2048, 2048]} shadow-camera-left={-60} shadow-camera-right={60}
                        shadow-camera-top={60} shadow-camera-bottom={-60} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[800, 800]} />
        <meshStandardMaterial color={C.grass} />
      </mesh>
      <Suspense fallback={null}>
        <Road scene={scene} />
        {scene.signals.map((s) => <SignalHead key={s.id} s={s} />)}
        {scene.signs.map((s) => <StopSign key={s.id} s={s} />)}
        {scene.vehicles.filter((v) => v.id !== hideSelf).map((v) => <Car key={v.id} v={v} label={labels} />)}
      </Suspense>
      {scene.vehicles.map((v) => <Trail key={`trail${v.id}`} v={v} />)}
      {scene.impact && <ImpactMarker impact={scene.impact} />}
      <CameraRig view={view} scene={scene} />
      {view === "orbit" && <OrbitControls target={target} />}
    </Canvas>
  );
}
