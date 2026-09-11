import { useEffect, useRef, useState } from "react";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

interface Comparison {
  scenario_id: string;
  policy_episode: number;
  seed_base: number;
  episodes: number;
  metric_label: string;
  results: {
    controller: string;
    successes: number;
    success_ci_low: number;
    success_ci_high: number;
    reward_mean: number;
    reward_std: number;
    metric: number | null;
  }[];
}

const LABELS: Record<string, string> = {
  policy: "Current policy", zero: "Zero action", random: "Random actions", reference: "Analytic reference",
};

export default function EvaluationPanel() {
  const { scenarioId, status } = useTrainingSocket();
  const [result, setResult] = useState<Comparison | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef(0);

  useEffect(() => {
    requestRef.current += 1;
    setResult(null); setError(null); setBusy(false);
    return () => { requestRef.current += 1; };
  }, [scenarioId]);

  const compare = async () => {
    const request = ++requestRef.current;
    setBusy(true); setError(null);
    try {
      const response = await fetch("/api/evaluation", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: scenarioId, episodes: 10 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Comparison failed.");
      if (request === requestRef.current) setResult(data);
    } catch (cause) {
      if (request === requestRef.current) setError(cause instanceof Error ? cause.message : "Comparison failed.");
    } finally {
      if (request === requestRef.current) setBusy(false);
    }
  };

  return <section className="panel evaluation-panel" aria-labelledby="evaluation-title" aria-busy={busy}>
    <div className="panel-heading">
      <div><span className="section-kicker">Test the claim</span><h2 id="evaluation-title">Controller comparison</h2></div>
      <button type="button" className="fullscreen-button" disabled={busy || status?.training || !scenarioId}
        onClick={() => void compare()}>{busy ? "Testing 10 starts…" : "Compare controllers"}</button>
    </div>
    <p className="evaluation-context">Same 10 initial states. Frozen policy, zero action, random actions, and an analytic reference where available.</p>
    {status?.training && <p role="status" className="evaluation-context">Pause training to take a policy snapshot.</p>}
    {error && <p role="alert">{error}</p>}
    {result && <>
      <div className="evaluation-table-wrap" tabIndex={0} role="region" aria-label="Controller comparison results">
        <table className="evaluation-table">
          <thead><tr><th>Controller</th><th>Successes</th><th>95% interval</th><th>Return ± SD</th><th>{result.metric_label}</th></tr></thead>
          <tbody>{result.results.map((row) => <tr key={row.controller} data-controller={row.controller}>
            <th scope="row">{LABELS[row.controller] ?? row.controller}</th>
            <td>{row.successes} / {result.episodes}</td>
            <td>{Math.round(row.success_ci_low * 100)}–{Math.round(row.success_ci_high * 100)}%</td>
            <td>{row.reward_mean.toFixed(1)} ± {row.reward_std.toFixed(1)}</td>
            <td>{row.metric == null ? "—" : row.metric.toFixed(3)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <p className="evaluation-context">Policy snapshot: episode {result.policy_episode}. Seeds {result.seed_base}–{result.seed_base + result.episodes - 1}. These repeatable diagnostics are not an untouched holdout.</p>
    </>}
    {!result && <p className="evaluation-context">A good reference result establishes task feasibility. A good policy result requires learning evidence of its own.</p>}
  </section>;
}
