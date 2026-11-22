import { useEffect, useRef } from "react";
import { useClock } from "../store";

/** The original clip follows the shared clock: video.currentTime = t + clipOffset. */
export default function VideoPane({ src, clipOffset = 0 }: { src: string; clipOffset?: number }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => useClock.subscribe((s) => {
    const v = ref.current;
    if (!v || !Number.isFinite(v.duration)) return;
    const want = s.t + clipOffset;
    v.playbackRate = s.speed;
    if (s.playing) {
      if (v.paused) v.play().catch(() => {});
      if (Math.abs(v.currentTime - want) > 0.12) v.currentTime = want;
    } else {
      if (!v.paused) v.pause();
      if (Math.abs(v.currentTime - want) > 0.01) v.currentTime = want;
    }
  }), [clipOffset]);
  return (
    <video ref={ref} src={src} muted playsInline preload="auto"
           className="h-full w-full bg-black object-contain" />
  );
}
