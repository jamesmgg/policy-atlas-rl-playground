import ExperimentBrief from "./ExperimentBrief";
import EvaluationPanel from "./EvaluationPanel";
import Leaderboard from "./Leaderboard";
import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import LearningLens from "./LearningLens";
import ScenarioSwitcher from "./ScenarioSwitcher";
import SceneCanvas from "./SceneCanvas";
import TrainingControls from "./TrainingControls";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import { focusExperimentHeading, mobileViewFromHash } from "../../api/types";
import type { MobileView } from "../../api/types";

const LearningCurve = lazy(() => import("./LearningCurve"));

export default function SimulatorPage() {
  const { connectionState, currentScenario, scenarios, status, lastError, clearError,
    stopTraining, readOnly } = useTrainingSocket();
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [mobileView, setMobileView] = useState(() => mobileViewFromHash(window.location.hash));

  const navigate = useCallback((view: MobileView) => {
    if (!window.matchMedia("(max-width: 900px)").matches) return;
    window.location.hash = view;
    setMobileView(view);
  }, []);
  const watch = useCallback(() => navigate("watch"), [navigate]);

  useEffect(() => {
    const onHashChange = () => setMobileView(mobileViewFromHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (!window.matchMedia("(max-width: 900px)").matches) return;
    const headings = { watch: "experiment-title", projects: "library-title",
      train: "run-title", results: "results-title" };
    const frame = requestAnimationFrame(() => {
      window.scrollTo({ top: 0, behavior: "instant" });
      focusExperimentHeading(document.getElementById(headings[mobileView]), true);
    });
    return () => cancelAnimationFrame(frame);
  }, [mobileView]);

  return (
    <div className="app-shell" data-mobile-view={mobileView}>
      <a className="skip-link" href="#experiment-stage">Skip to active experiment</a>
      <header className="topbar">
        <div className="brand-lockup" aria-label="Policy Atlas RL Playground">
          <span className="brand-mark" aria-hidden="true">π</span>
          <span>
            <strong>Policy Atlas</strong>
            <small>RL playground</small>
          </span>
        </div>
        <div className="topbar-context">
          <span>{scenarios.length ? `${scenarios.length} experiments` : "Loading experiments"}</span>
          <span aria-hidden="true">·</span>
          <span>{readOnly ? "Replay verified policies" : "Watch policies learn"}</span>
        </div>
        <div className={`connection-state connection-${connectionState}`} aria-live="polite">
          <span className="connection-dot" aria-hidden="true" />
          {connectionState === "connected" ? (readOnly ? "Public library" : status?.training ? "Training live" : "Lab ready") : connectionState}
        </div>
        <a className="mobile-project-switch" href="#projects" aria-label={`Switch project: ${currentScenario?.name ?? "Loading"}`}>
          <span><small>Current project</small><strong>{currentScenario?.name ?? "Loading experiments…"}</strong></span>
          <span className="project-switch-label">Switch <span aria-hidden="true">⌄</span></span>
        </a>
      </header>

      {lastError && mobileView !== "train" && <div className="mobile-error error-banner" role="alert">
        <span>{lastError}</span><button onClick={clearError} aria-label="Dismiss error">×</button>
      </div>}

      <div className="workspace">
        <main className="experiment-stage" id="experiment-stage">
          <div className="watch-screen">
            <section className="experiment-heading" aria-labelledby="experiment-title">
              <div>
                <div className="eyebrow">
                  <span>{currentScenario?.group ?? "Experiment"}</span>
                  <span>{currentScenario?.difficulty ?? "Loading"}</span>
                </div>
                <h1 id="experiment-title" tabIndex={-1}>{currentScenario?.name ?? "Opening experiment…"}</h1>
                <p>{currentScenario?.description ?? "Loading the environment and its experiment contract."}</p>
              </div>
            </section>

            <SceneCanvas />
            <div className="mobile-watch-actions">
              {status?.training
                ? <button className="primary-action action-pause" onClick={stopTraining}>Pause training</button>
                : readOnly
                  ? <a className="primary-action" href="#results">Choose verified policy</a>
                  : <a className="primary-action" href="#train">Train this policy</a>}
              <a className="secondary-action" href="#results">Saved runs & results</a>
              <p>{status?.training ? "Training continues while you browse. Pausing finishes the current rollout."
                : readOnly ? "Switch projects or choose either qualified training seed in Results."
                  : "Choose a training budget, or watch a saved policy from Results."}</p>
            </div>
          </div>
          <div className="results-screen">
            <h2 className="mobile-screen-title" id="results-title" tabIndex={-1}>Results</h2>
            <Leaderboard onWatch={watch} />
            <EvaluationPanel />
            <details className="technical-drawer"
              onToggle={(event) => setShowDiagnostics(event.currentTarget.open)}>
              <summary>Show learning diagnostics</summary>
              <div className="technical-drawer-content">
                <LearningLens />
                {showDiagnostics && <Suspense fallback={<p>Loading learning charts…</p>}>
                  <LearningCurve />
                </Suspense>}
              </div>
            </details>
          </div>
        </main>

        <ScenarioSwitcher onSelected={watch} />

        <aside className="experiment-inspector" aria-label="Experiment setup">
          <TrainingControls onRun={watch} />
          <ExperimentBrief />
        </aside>
      </div>

      <nav className="mobile-setup-dock" aria-label="Mobile navigation">
        {([
          ["projects", "Projects", "▦"], ["watch", "Watch", "◉"],
          ["train", readOnly ? "Run locally" : "Train", "▷"], ["results", "Results", "▥"],
        ] as const).map(([view, label, icon]) => (
          <a key={view} href={`#${view}`} aria-current={mobileView === view ? "page" : undefined}>
            <span aria-hidden="true">{icon}</span><strong>{label}</strong>
            {view === "train" && status?.training && <span className="nav-training-dot" aria-label="Training in progress" />}
          </a>
        ))}
      </nav>

      <footer className="app-footer">
        <span>Policy Atlas · {readOnly ? "public verified replay library" : "local experiment workspace"}</span>
        <span>{readOnly ? "46 policies passed 2,300/2,300 independent holdout starts."
          : `Scores use ${status?.eval_episodes ?? 10} fixed test starts with 95% success intervals.`}</span>
      </footer>
    </div>
  );
}
