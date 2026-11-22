import { useRoute, go } from "./router";
import Calibrate from "./screens/Calibrate";
import { Privacy, Terms } from "./screens/Legal";
import Processing from "./screens/Processing";
import Report from "./screens/Report";
import Results from "./screens/Results";
import Upload from "./screens/Upload";

export default function App() {
  const [page, id] = useRoute();
  if (page === "report" && id) return <Report id={id} />;
  return (
    <div className="flex min-h-screen flex-col bg-paper text-ink">
      <header className="no-print border-b border-rule bg-panel">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between px-5 py-3">
          <button onClick={() => go("/")} className="flex items-center gap-2.5 text-[19px] font-extrabold tracking-tight">
            {/* the mark: a stop line across a lane */}
            <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
              <rect x="1" y="1" width="20" height="20" fill="#1e2226" />
              <rect x="10" y="3" width="2" height="5" fill="#f7f8f6" />
              <rect x="10" y="14" width="2" height="5" fill="#f7f8f6" />
              <rect x="3" y="10" width="16" height="2.5" fill="#f1b812" />
            </svg>
            CrashLens
          </button>
          <nav className="flex items-center gap-5 text-[14px] font-semibold text-ink2">
            <button onClick={() => go("/results/demo-left-turn")} className="hover:text-ink">Left-turn demo</button>
            <button onClick={() => go("/results/demo-rear-end")} className="hover:text-ink">Rear-end demo</button>
            <button onClick={() => go("/")} className="btn btn-sm">New case</button>
          </nav>
        </div>
      </header>
      <main className="flex-1">
        {page === "calibrate" && id ? <Calibrate id={id} />
          : page === "processing" && id ? <Processing id={id} />
          : page === "results" && id ? <Results key={id} id={id} />
          : page === "terms" ? <Terms />
          : page === "privacy" ? <Privacy />
          : <Upload />}
      </main>
      <footer className="no-print border-t border-rule">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-5 py-4 text-[13px] text-muted">
          <span>CrashLens gives an evidence summary. It is not legal advice or an official fault determination.</span>
          <span className="flex gap-5">
            <button onClick={() => go("/terms")} className="hover:text-ink">Terms of use</button>
            <button onClick={() => go("/privacy")} className="hover:text-ink">Privacy</button>
          </span>
        </div>
      </footer>
    </div>
  );
}
