import ExperimentBrief from "./ExperimentBrief";
import Leaderboard from "./Leaderboard";
import LearningCurve from "./LearningCurve";
import LearningLens from "./LearningLens";
import ScenarioSwitcher from "./ScenarioSwitcher";
import SceneCanvas from "./SceneCanvas";
import TrainingControls from "./TrainingControls";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

export default function SimulatorPage() {
  const { connectionState, currentScenario, scenarios, status } = useTrainingSocket();

  return (
    <div className="app-shell">
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
          <span>{scenarios.length || 16} experiments</span>
          <span aria-hidden="true">·</span>
          <span>Watch policies learn</span>
        </div>
        <div className={`connection-state connection-${connectionState}`} aria-live="polite">
          <span className="connection-dot" aria-hidden="true" />
          {connectionState === "connected" ? (status?.training ? "Training live" : "Lab ready") : connectionState}
        </div>
      </header>

      <div className="workspace">
        <main className="experiment-stage" id="experiment-stage">
          <section className="experiment-heading" aria-labelledby="experiment-title">
            <div>
              <div className="eyebrow">
                <span>{currentScenario?.group ?? "Experiment"}</span>
                <span>{currentScenario?.difficulty ?? "Loading"}</span>
              </div>
              <h1 id="experiment-title">{currentScenario?.name ?? "Opening experiment…"}</h1>
              <p>{currentScenario?.description ?? "Loading the environment and its experiment contract."}</p>
            </div>
          </section>

          <SceneCanvas />
          <Leaderboard />
          <details className="technical-drawer">
            <summary>Show learning diagnostics</summary>
            <div className="technical-drawer-content">
              <LearningLens />
              <LearningCurve />
            </div>
          </details>
        </main>

        <ScenarioSwitcher />

        <aside className="experiment-inspector" aria-label="Experiment setup">
          <ExperimentBrief />
          <TrainingControls />
        </aside>
      </div>

      <nav className="mobile-setup-dock" aria-label="Quick setup navigation">
        <a href="#experiment-stage">
          <span className="dock-glyph dock-glyph-watch" aria-hidden="true">◉</span>
          <span><small>Live</small><strong>Watch</strong></span>
        </a>
        <a href="#experiment-library">
          <span className="dock-glyph" aria-hidden="true">∿</span>
          <span><small>Browse</small><strong>Experiment</strong></span>
        </a>
        <a href="#run-setup">
          <span className="dock-glyph dock-glyph-run" aria-hidden="true">▶</span>
          <span><small>Choose a budget</small><strong>Run setup</strong></span>
        </a>
      </nav>

      <footer className="app-footer">
        <span>Policy Atlas · local experiment workspace</span>
        <span>Scores use {status?.eval_episodes ?? 10} fixed test starts with 95% success intervals.</span>
      </footer>
    </div>
  );
}
