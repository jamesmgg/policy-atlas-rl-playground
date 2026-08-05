import { useMemo, useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { PpoUpdateRecord } from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

const CHART_GRID = "#d8e1de";
const CHART_TEXT = "#52645f";
const TOOLTIP_STYLE = {
  background: "#ffffff", border: "1px solid #c7d5d1", borderRadius: 10,
  boxShadow: "0 10px 30px rgba(23,39,37,.12)", fontSize: 12,
};

function DiagnosticChart({
  title, description, data, dataKey, color, reference,
}: {
  title: string;
  description: string;
  data: PpoUpdateRecord[];
  dataKey: "entropy" | "approx_kl" | "policy_loss";
  color: string;
  reference?: number;
}) {
  return (
    <article className="diagnostic-chart">
      <div><h3>{title}</h3><p>{description}</p></div>
      <ResponsiveContainer width="100%" height={145}>
        <ComposedChart data={data} margin={{ top: 12, right: 8, bottom: 4, left: -12 }}>
          <CartesianGrid stroke={CHART_GRID} vertical={false} />
          <XAxis dataKey="update" stroke={CHART_TEXT} fontSize={10} tickLine={false}
            label={{ value: "PPO update", position: "insideBottomRight", offset: -2, fill: CHART_TEXT, fontSize: 10 }} />
          <YAxis stroke={CHART_TEXT} fontSize={10} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(value) => `Update ${value}`} />
          {reference != null && <ReferenceLine y={reference} stroke="#c95663" strokeDasharray="4 4" />}
          <Line dataKey={dataKey} stroke={color} dot={false} strokeWidth={2}
            isAnimationActive={false} name={title} />
        </ComposedChart>
      </ResponsiveContainer>
    </article>
  );
}

export default function LearningCurve() {
  const { history, ppo, currentScenario } = useTrainingSocket();
  const [view, setView] = useState<"performance" | "diagnostics">("performance");

  const performance = useMemo(() => {
    const recentSuccess: number[] = [];
    return history.map((episode) => {
      if (typeof episode.success === "boolean") {
        recentSuccess.push(episode.success ? 1 : 0);
        if (recentSuccess.length > 20) recentSuccess.shift();
      }
      return {
        ...episode,
        successMean: recentSuccess.length
          ? Math.round(recentSuccess.reduce((sum, value) => sum + value, 0) / recentSuccess.length * 100)
          : undefined,
      };
    });
  }, [history]);

  const latest = performance.at(-1);
  const recentReturns = performance.slice(-20).map((item) => item.reward);
  const range = recentReturns.length
    ? `${Math.min(...recentReturns).toFixed(0)} to ${Math.max(...recentReturns).toFixed(0)}` : "—";

  return (
    <section className="panel evidence-panel" aria-labelledby="evidence-title">
      <div className="panel-heading evidence-heading">
        <div><span className="section-kicker">Evidence</span><h2 id="evidence-title">Is the policy learning?</h2></div>
        <div className="view-tabs" role="group" aria-label="Chart view">
          <button type="button" aria-pressed={view === "performance"}
            onClick={() => setView("performance")}>Performance</button>
          <button type="button" aria-pressed={view === "diagnostics"}
            onClick={() => setView("diagnostics")}>PPO diagnostics</button>
        </div>
      </div>

      {view === "performance" ? (
        history.length === 0 ? (
          <div className="evidence-empty">
            <span className="empty-trace" aria-hidden="true">⌁⌁⌁</span>
            <h3>No evidence yet</h3>
            <p>Run a starter budget. The raw return, 20-episode mean, and success rate will appear here.</p>
          </div>
        ) : (
          <>
            <div className="chart-summary">
              <span><small>Current 20-episode mean</small><strong>{latest?.rollingMean?.toFixed(1) ?? "—"}</strong></span>
              <span><small>Recent return range</small><strong>{range}</strong></span>
              <span><small>Rolling success</small><strong>{latest?.successMean != null ? `${latest.successMean}%` : "—"}</strong></span>
            </div>
            <div className="chart-legend" aria-label="Chart legend">
              <span><i className="legend-raw" />Episode return</span>
              <span><i className="legend-mean" />20-episode mean</span>
              <span><i className="legend-success" />Success rate</span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={performance} margin={{ top: 12, right: 16, bottom: 22, left: 6 }}>
                <CartesianGrid stroke={CHART_GRID} vertical={false} />
                <XAxis dataKey="episode" stroke={CHART_TEXT} fontSize={11} tickLine={false}
                  label={{ value: "Episode", position: "insideBottom", offset: -12, fill: CHART_TEXT, fontSize: 11 }} />
                <YAxis yAxisId="return" stroke={CHART_TEXT} fontSize={11} tickLine={false} axisLine={false}
                  label={{ value: "Return", angle: -90, position: "insideLeft", fill: CHART_TEXT, fontSize: 11 }} />
                <YAxis yAxisId="success" orientation="right" domain={[0, 100]} hide />
                <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(value) => `Episode ${value}`} />
                <Area yAxisId="return" dataKey="reward" stroke="none" fill="rgba(53,103,200,.10)"
                  isAnimationActive={false} name="Episode return" />
                <Line yAxisId="return" dataKey="reward" stroke="rgba(53,103,200,.3)" dot={false}
                  strokeWidth={1} isAnimationActive={false} name="Episode return" />
                <Line yAxisId="return" dataKey="rollingMean" stroke="#3567c8" dot={false}
                  strokeWidth={2.5} isAnimationActive={false} name="20-episode mean" />
                <Line yAxisId="success" dataKey="successMean" stroke="#236b61" dot={false}
                  strokeWidth={1.8} strokeDasharray="5 4" isAnimationActive={false} name="Success rate (%)" />
              </ComposedChart>
            </ResponsiveContainer>
            <p className="chart-caption">
              Return is the sum of rewards in one episode. Compare trends within this experiment;
              reward scales are not comparable across environments. Primary measure: {currentScenario?.metric_label ?? "task metric"}.
            </p>
            <details className="chart-data-table">
              <summary>Read the latest chart data as a table</summary>
              <div>
                <table>
                  <caption className="sr-only">Latest twenty episode performance records</caption>
                  <thead><tr><th scope="col">Episode</th><th scope="col">Return</th>
                    <th scope="col">20-episode mean</th><th scope="col">Success</th></tr></thead>
                  <tbody>{performance.slice(-20).map((item) => (
                    <tr key={item.episode}><th scope="row">{item.episode}</th>
                      <td>{item.reward.toFixed(2)}</td><td>{item.rollingMean?.toFixed(2) ?? "—"}</td>
                      <td>{item.success == null ? "Not recorded" : item.success ? "Yes" : "No"}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </details>
          </>
        )
      ) : ppo.length === 0 ? (
        <div className="evidence-empty"><h3>No optimizer updates yet</h3><p>Diagnostics appear after PPO has optimized its first rollout.</p></div>
      ) : (
        <div className="diagnostic-grid">
          <DiagnosticChart title="Squashed entropy" description="Executed-action exploration; collapse trends downward."
            data={ppo} dataKey="entropy" color="#8b5bb1" />
          <DiagnosticChart title="Approximate KL" description="How far the policy moved; red line is the early-stop target."
            data={ppo} dataKey="approx_kl" color="#a85b12" reference={0.03} />
          <DiagnosticChart title="Policy loss" description="Optimization signal; scale is meaningful only within this run."
            data={ppo} dataKey="policy_loss" color="#3567c8" />
        </div>
      )}
    </section>
  );
}
