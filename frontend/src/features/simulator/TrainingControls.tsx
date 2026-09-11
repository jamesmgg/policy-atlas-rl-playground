import { useEffect, useMemo, useState } from "react";
import {
  formatMetric, getRunWindow, normalizeRunConfig, recentCheckpoints,
  scrollExperimentStage,
} from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

const PRESETS = [
  { name: "Quick look", episodes: 100, checkpoint: 10, note: "See the loop" },
  { name: "Study", episodes: 500, checkpoint: 25, note: "Track a trend" },
  { name: "Extended", episodes: 2000, checkpoint: 50, note: "Train seriously" },
];

function durationLabel(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "Estimating after the first episode";
  if (seconds < 60) return `About ${Math.max(1, Math.round(seconds))} sec remaining`;
  if (seconds < 3600) return `About ${Math.round(seconds / 60)} min remaining`;
  return `About ${(seconds / 3600).toFixed(1)} hr remaining`;
}

export default function TrainingControls({ onRun }: { onRun?: () => void }) {
  const {
    status, connected, connectionState, metricLabel, history, checkpoints,
    startTraining, stopTraining, resetTraining, lastError, clearError,
  } = useTrainingSocket();
  const [episodes, setEpisodes] = useState(500);
  const [checkpointEvery, setCheckpointEvery] = useState(25);
  const [seed, setSeed] = useState(42);
  const training = status?.training ?? false;

  useEffect(() => {
    if (status?.seed != null) setSeed(status.seed);
  }, [status?.seed]);

  const validSeed = Number.isInteger(seed) && seed >= 0 && seed <= 2 ** 32 - 1;
  const valid = Number.isFinite(episodes) && episodes >= 1 && episodes <= 1_000_000
    && Number.isFinite(checkpointEvery) && checkpointEvery >= 1 && checkpointEvery <= 100_000
    && validSeed;
  const activePreset = PRESETS.find((preset) =>
    preset.episodes === episodes && preset.checkpoint === checkpointEvery)?.name;

  const runWindow = status ? getRunWindow(status) : {
    remaining: 0, progress: 0, resumable: false,
  };
  const eta = useMemo(() => {
    const recent = history.slice(-20);
    const meanSteps = recent.length
      ? recent.reduce((sum, item) => sum + item.steps, 0) / recent.length : 0;
    return durationLabel(status?.sps ? runWindow.remaining * meanSteps / status.sps : 0);
  }, [history, runWindow.remaining, status?.sps]);
  const latestEvaluation = recentCheckpoints(checkpoints)[0];

  const applyPreset = (preset: typeof PRESETS[number]) => {
    setEpisodes(preset.episodes);
    setCheckpointEvery(preset.checkpoint);
  };

  const watchTraining = () => {
    onRun?.();
    scrollExperimentStage(
      document.getElementById("experiment-stage"),
      window.matchMedia("(max-width: 900px)").matches,
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    );
  };

  const run = () => {
    const config = normalizeRunConfig(episodes, checkpointEvery);
    setEpisodes(config.episodes);
    setCheckpointEvery(config.checkpointEvery);
    const target = runWindow.resumable
      ? status!.run_target_episode
      : (status?.episode ?? 0) + config.episodes;
    if (startTraining(target, config.checkpointEvery)) watchTraining();
  };

  const replaceBudget = () => {
    const config = normalizeRunConfig(episodes, checkpointEvery);
    if (startTraining((status?.episode ?? 0) + config.episodes, config.checkpointEvery)) watchTraining();
  };

  const newRun = () => {
    const confirmed = window.confirm(
      `Start a new run with seed ${seed}? Current checkpoints will move to a recoverable archive.`,
    );
    if (confirmed) resetTraining(seed);
  };

  return (
    <section className="panel controls-panel" id="run-setup" aria-labelledby="run-title">
      <div className="panel-heading">
        <div><span className="section-kicker">Run setup</span><h2 id="run-title" tabIndex={-1}>Train the policy</h2></div>
        <span className={`run-badge run-${training ? "live" : connectionState}`} aria-live="polite">
          {training ? "Running" : connectionState === "connected" ? "Ready" : connectionState}
        </span>
      </div>

      <fieldset className="preset-fieldset" disabled={training}>
        <legend>Choose a training budget</legend>
        <div className="preset-grid">
          {PRESETS.map((preset) => (
            <button type="button" key={preset.name}
              className={activePreset === preset.name ? "preset-active" : ""}
              aria-pressed={activePreset === preset.name}
              onClick={() => applyPreset(preset)}>
              <strong>{preset.name}</strong><span>{preset.episodes.toLocaleString()} episodes</span><small>{preset.note}</small>
            </button>
          ))}
        </div>
      </fieldset>

      <details className="advanced-controls">
        <summary>Reproducibility and saving</summary>
        <div className="field-grid">
          <label>
            <span>Episodes to add</span>
            <input type="number" min={1} max={1_000_000} value={episodes} disabled={training}
              aria-invalid={!valid}
              onChange={(event) => setEpisodes(Number(event.target.value))} />
            <small>Each episode ends on success, failure, or the time limit.</small>
          </label>
          <label>
            <span>Save every</span>
            <input type="number" min={1} max={100_000} value={checkpointEvery} disabled={training}
              aria-invalid={!valid}
              onChange={(event) => setCheckpointEvery(Number(event.target.value))} />
            <small>Evaluates the policy on {status?.eval_episodes ?? 10} fixed, versioned test starts.</small>
          </label>
          <label>
            <span>Random seed</span>
            <input type="number" min={0} max={2 ** 32 - 1} value={seed} disabled={training}
              aria-invalid={!validSeed}
              onChange={(event) => setSeed(Number(event.target.value))} />
            <small>Used when you start a new run.</small>
          </label>
        </div>
      </details>

      {status && (
        <div className="run-progress">
          <div className="progress-copy">
            <span>Episode <strong>{status.episode.toLocaleString()}</strong>{(training || runWindow.resumable) && ` of ${status.run_target_episode.toLocaleString()}`}</span>
            <span>{training ? eta : runWindow.resumable ? `${runWindow.remaining.toLocaleString()} episodes paused` : `${status.total_steps.toLocaleString()} environment steps`}</span>
          </div>
          <progress max={100} value={runWindow.progress} aria-label="Training budget progress" />
        </div>
      )}

      <div className={`run-actions ${runWindow.resumable ? "has-paused-budget" : ""}`}>
        {!training ? (
          <button type="button" className="primary-action" disabled={!connected || !valid} onClick={run}>
            <span aria-hidden="true">▶</span> {runWindow.resumable
              ? `Resume ${runWindow.remaining.toLocaleString()} remaining`
              : `Run ${episodes.toLocaleString()} episodes`}
          </button>
        ) : (
          <button type="button" className="primary-action action-pause" onClick={stopTraining}>
            <span aria-hidden="true">Ⅱ</span> Pause after rollout
          </button>
        )}
        {runWindow.resumable && (
          <button type="button" className="secondary-action" disabled={!connected || !valid}
            onClick={replaceBudget}>Start selected budget instead</button>
        )}
        <button type="button" className="secondary-action" disabled={!connected || training} onClick={newRun}>
          New seeded run
        </button>
      </div>

      <details className="run-statistics">
        <summary>Show run statistics</summary>
        <dl className="run-stats">
          <div><dt>Best training return</dt><dd>{status?.best_reward?.toFixed(1) ?? "—"}</dd></div>
          <div><dt>{metricLabel}</dt><dd>{formatMetric(status?.best_metric, metricLabel)}</dd></div>
          <div><dt>Latest success rate</dt><dd>{latestEvaluation?.success_rate != null ? `${Math.round(latestEvaluation.success_rate * 100)}%` : "Not evaluated"}</dd></div>
          <div><dt>Compute</dt><dd>{status ? `${status.device.toUpperCase()} · ${status.sps.toLocaleString()} steps/s` : "Connecting"}</dd></div>
        </dl>
      </details>

      {lastError && (
        <div className="error-banner" role="alert">
          <span>{lastError}</span>
          <button type="button" onClick={clearError} aria-label="Dismiss error">×</button>
        </div>
      )}
    </section>
  );
}
