import { useTrainingSocket } from "../../hooks/useTrainingSocket";

export default function ExperimentBrief() {
  const { currentScenario, metricMode, status } = useTrainingSocket();
  if (!currentScenario) {
    return <section className="panel brief-panel panel-loading" aria-label="Loading experiment brief" />;
  }

  return (
    <section className="panel brief-panel" aria-labelledby="brief-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Experiment brief</span>
          <h2 id="brief-title">What the agent is learning</h2>
        </div>
        <span className="difficulty-tag">{currentScenario.difficulty}</span>
      </div>

      <div className="brief-callout">
        <span className="brief-icon" aria-hidden="true">◎</span>
        <div><strong>Objective</strong><p>{currentScenario.objective}</p></div>
      </div>
      <div className="brief-success">
        <span aria-hidden="true">✓</span>
        <p><strong>Success:</strong> {currentScenario.success}</p>
      </div>

      <details className="contract-details">
        <summary>Agent–environment contract</summary>
        <div className="contract-grid">
          <div>
            <h3>Agent sees</h3>
            <ul>{currentScenario.observations.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
          <div>
            <h3>Agent controls</h3>
            <ul>{currentScenario.actions.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
        </div>
      </details>

      <details className="contract-details reward-contract">
        <summary>Reward and episode endings</summary>
        <div className="contract-grid">
          <div>
            <h3>Return changes</h3>
            <ul>{currentScenario.reward_terms.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
          <div>
            <h3>Episode ends on</h3>
            <ul>{currentScenario.termination_conditions.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
        </div>
      </details>

      <details className="method-details">
        <summary>Method details</summary>
        <div className="method-strip">
          <span><strong>PPO</strong> clipped policy updates</span>
          <span><strong>Seed {status?.seed ?? 42}</strong> reproducible reset</span>
          <span><strong>{metricMode === "min" ? "↓" : "↑"}</strong> {currentScenario.metric_label}</span>
        </div>
      </details>
    </section>
  );
}
