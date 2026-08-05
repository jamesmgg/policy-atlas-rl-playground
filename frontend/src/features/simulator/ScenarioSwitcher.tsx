import { useMemo, useState } from "react";
import { filterScenarios, formatMetric } from "../../api/types";
import type { ScenarioInfo } from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

const GROUP_ORDER = ["All", "Foundations", "Classic", "Circuits", "Weather", "Vehicles", "Objectives"];

function experimentGlyph(scenario: ScenarioInfo): string {
  if (scenario.id.includes("cartpole") || scenario.id.includes("pendulum")) return "ϕ";
  if (scenario.id.includes("mountain")) return "∿";
  if (scenario.id.includes("lander")) return "△";
  if (scenario.id.includes("drone")) return "✣";
  if (scenario.group === "Weather") return "◌";
  if (scenario.group === "Objectives") return "◎";
  return "⌁";
}

export default function ScenarioSwitcher() {
  const { scenarioId, status, scenarios, selectScenario, lastError } = useTrainingSocket();
  const [query, setQuery] = useState("");
  const [group, setGroup] = useState("All");
  const [switching, setSwitching] = useState(false);

  const groups = GROUP_ORDER.filter((name) => name === "All" || scenarios.some((s) => s.group === name));
  const filtered = useMemo(
    () => filterScenarios(scenarios, query, group),
    [scenarios, query, group],
  );

  const onSelect = async (scenario: ScenarioInfo) => {
    if (scenario.id === scenarioId || switching) return;
    if (status?.training && !window.confirm(
      `Open “${scenario.name}”? The current run will pause after its current PPO rollout.`,
    )) return;
    setSwitching(true);
    try {
      await selectScenario(scenario.id);
    } finally {
      setSwitching(false);
    }
  };

  return (
    <aside className="experiment-library" aria-labelledby="library-title" aria-busy={switching}>
      <div className="library-heading">
        <span className="section-kicker">Explore</span>
        <div><h2 id="library-title">Experiment library</h2><span>{scenarios.length} available</span></div>
      </div>

      <label className="library-search">
        <span className="sr-only">Search experiments</span>
        <span aria-hidden="true">⌕</span>
        <input value={query} onChange={(event) => setQuery(event.target.value)}
          placeholder="Search goal, domain, level…" type="search" />
      </label>

      <div className="library-filters" aria-label="Experiment categories">
        {groups.map((name) => (
          <button key={name} type="button" aria-pressed={group === name}
            onClick={() => setGroup(name)}>{name}</button>
        ))}
      </div>

      <div className="experiment-list" aria-label="Available experiments">
        {filtered.map((scenario) => {
          const active = scenario.id === scenarioId;
          return (
            <button type="button" key={scenario.id}
              className={`experiment-item ${active ? "experiment-item-active" : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => onSelect(scenario)} disabled={switching}>
              <span className="experiment-glyph" aria-hidden="true">{experimentGlyph(scenario)}</span>
              <span className="experiment-item-copy">
                <span className="experiment-item-title">{scenario.name}</span>
                <span className="experiment-item-meta">
                  {scenario.difficulty} · {scenario.metric_mode === "min" ? "minimize" : "maximize"} {scenario.metric_label}
                </span>
                <span className="experiment-item-progress">
                  {scenario.progress
                    ? `Episode ${scenario.progress.episode} · ${formatMetric(scenario.progress.best_metric, scenario.metric_label)}`
                    : "Ready for a first run"}
                </span>
              </span>
              <span className="experiment-open" aria-hidden="true">›</span>
            </button>
          );
        })}
        {scenarios.length === 0 && <div className="library-empty" role="status">
          {lastError ?? "Connecting to the experiment catalog…"}
        </div>}
        {scenarios.length > 0 && filtered.length === 0 && (
          <div className="library-empty">No experiments match that search.</div>
        )}
      </div>
    </aside>
  );
}
