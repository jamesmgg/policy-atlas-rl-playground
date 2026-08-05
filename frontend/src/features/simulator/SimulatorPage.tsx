import ExperimentBrief from "./ExperimentBrief";
import Leaderboard from "./Leaderboard";
import LearningCurve from "./LearningCurve";
import LearningLens from "./LearningLens";
import ScenarioSwitcher from "./ScenarioSwitcher";
import SceneCanvas from "./SceneCanvas";
import TrainingControls from "./TrainingControls";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import { formatHorizon } from "../../api/types";

export default function SimulatorPage() {
  const { connectionState, currentScenario, scenarios, status } = useTrainingSocket();
  const direction = currentScenario?.metric_mode === "min" ? "Lower is better" : "Higher is better";

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
          <span>PPO learning live</span>
        </div>
        <div className={`connection-state connection-${connectionState}`} aria-live="polite">
          <span className="connection-dot" aria-hidden="true" />
          {connectionState === "connected" ? (status?.training ? "Training live" : "Lab ready") : connectionState}
        </div>
      </header>

      <div className="workspace">
        <ScenarioSwitcher />

        <aside className="experiment-inspector" aria-label="Experiment setup">
          <ExperimentBrief />
          <TrainingControls />
        </aside>

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
            {currentScenario && (
              <dl className="heading-facts">
                <div><dt>Primary measure</dt><dd>{currentScenario.metric_label}</dd></div>
                <div><dt>Direction</dt><dd>{direction}</dd></div>
                <div><dt>Horizon</dt><dd>{formatHorizon(currentScenario.horizon_steps, currentScenario.horizon_seconds)}</dd></div>
              </dl>
            )}
          </section>

          <SceneCanvas />
          <LearningLens />
          <LearningCurve />
          <Leaderboard />
        </main>
      </div>

      <footer className="app-footer">
        <span>Policy Atlas · local experiment workspace</span>
        <span>Scores use {status?.eval_episodes ?? 10} fixed test starts with 95% success intervals.</span>
      </footer>
    </div>
  );
}
