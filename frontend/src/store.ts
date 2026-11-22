import { create } from "zustand";
import type { View } from "./types";

/** One shared clock for the video, the 3D replay and the scrubber (spec: "a single t"). */
interface Clock {
  t: number; duration: number; playing: boolean; speed: number; view: View;
  setT: (t: number) => void; setDuration: (d: number) => void; setPlaying: (p: boolean) => void;
  setSpeed: (s: number) => void; setView: (v: View) => void;
}

export const useClock = create<Clock>((set) => ({
  t: 0, duration: 7, playing: false, speed: 0.5, view: "top",
  setT: (t) => set((s) => ({ t: Math.max(0, Math.min(s.duration, t)) })),
  setDuration: (duration) => set({ duration }),
  setPlaying: (playing) => set({ playing }),
  setSpeed: (speed) => set({ speed }),
  setView: (view) => set({ view }),
}));
