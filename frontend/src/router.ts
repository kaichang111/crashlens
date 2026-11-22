import { useEffect, useState } from "react";

/** Tiny hash router: #/, #/calibrate/:id, #/processing/:id, #/results/:id, #/report/:id */
export function go(path: string) {
  window.location.hash = path;
}

export function useRoute(): string[] {
  const parse = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [r, setR] = useState(parse);
  useEffect(() => {
    const on = () => setR(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return r;
}
