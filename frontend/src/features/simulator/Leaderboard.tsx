import { useMemo, useState } from "react";
import type { CheckpointMeta } from "../../api/types";
import {
  formatMetric, isComparableCheckpoint, rankCheckpoints, recentCheckpoints,
} from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

function savedAt(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export default function Leaderboard() {
  const {
    checkpoints, archivedRuns, ghostEpisode, status, metricLabel, metricMode,
    setGhost, clearGhost, loadCheckpoint, restoreArchivedRun,
  } = useTrainingSocket();
  const [view, setView] = useState<"best" | "recent">("best");
  const training = status?.training ?? false;
  const comparableCheckpoints = useMemo(
    () => status
      ? checkpoints.filter((checkpoint) => isComparableCheckpoint(checkpoint, status))
      : [],
    [checkpoints, status],
  );

  const ranked = useMemo(
    () => rankCheckpoints(comparableCheckpoints, metricMode),
    [comparableCheckpoints, metricMode],
  );
  const rows = view === "best" ? ranked : recentCheckpoints(checkpoints);

  const resume = (checkpoint: CheckpointMeta) => {
    const compatibilityNote = status && !isComparableCheckpoint(checkpoint, status)
      ? " Its saved score uses a different experiment protocol and will not be ranked with current results."
      : "";
    if (window.confirm(
      `Resume from episode ${checkpoint.episode}? This replaces the policy currently loaded in memory.${compatibilityNote}`,
    )) loadCheckpoint(checkpoint.episode);
  };

  const restore = (id: string, episode: number) => {
    if (window.confirm(
      `Restore the archived branch ending at episode ${episode}? The active branch will be archived first.`,
    )) void restoreArchivedRun(id);
  };

  return (
    <section className="panel checkpoints-panel" aria-labelledby="checkpoints-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Reproducibility</span>
          <h2 id="checkpoints-title">Saved checkpoints</h2>
        </div>
        <div className="view-tabs" role="group" aria-label="Checkpoint order">
          <button type="button" aria-pressed={view === "best"} onClick={() => setView("best")}>Best evaluated</button>
          <button type="button" aria-pressed={view === "recent"} onClick={() => setView("recent")}>Most recent</button>
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="checkpoint-empty">
          <span aria-hidden="true">◇</span>
          {checkpoints.length > 0 && view === "best"
            ? <div><h3>No comparable checkpoints</h3><p>{checkpoints.length} older-protocol checkpoint{checkpoints.length === 1 ? " is" : "s are"} available under Most recent.</p></div>
            : <div><h3>No checkpoints saved</h3><p>A checkpoint stores policy weights and evaluates them on the fixed test suite.</p></div>}
        </div>
      ) : (
        <div className="checkpoint-table-wrap">
          <table className="checkpoint-table">
            <caption className="sr-only">Evaluated policy checkpoints for the active experiment</caption>
            <thead><tr>
              <th scope="col">Rank</th><th scope="col">Checkpoint</th><th scope="col">Success</th>
              <th scope="col">{metricLabel}</th><th scope="col">Evaluation return</th>
              <th scope="col">Protocol</th><th scope="col"><span className="sr-only">Actions</span></th>
            </tr></thead>
            <tbody>
              {rows.map((checkpoint, index) => {
                const ghostActive = ghostEpisode === checkpoint.episode;
                const sourceHash = typeof checkpoint.protocol?.engine_source_sha256 === "string"
                  ? checkpoint.protocol.engine_source_sha256
                  : null;
                const comparable = status
                  ? isComparableCheckpoint(checkpoint, status)
                  : false;
                return (
                  <tr key={checkpoint.episode} className={ghostActive ? "checkpoint-comparing" : ""}>
                    <td data-label="Rank"><span className="rank-number">{view === "best" ? index + 1 : "—"}</span></td>
                    <th scope="row" data-label="Checkpoint">
                      <strong>Episode {checkpoint.episode.toLocaleString()}</strong>
                      <small>{savedAt(checkpoint.timestamp)}</small>
                      {!comparable && <small className="protocol-warning">Different protocol · not ranked</small>}
                    </th>
                    <td data-label="Success">
                      <strong>{checkpoint.success_rate != null ? `${Math.round(checkpoint.success_rate * 100)}%` : "Legacy"}</strong>
                      <small>{checkpoint.success_ci_low != null && checkpoint.success_ci_high != null
                        ? `95% CI ${Math.round(checkpoint.success_ci_low * 100)}–${Math.round(checkpoint.success_ci_high * 100)}%`
                        : "No confidence interval"}</small>
                      <small>{checkpoint.evaluation_suite
                        ? `${checkpoint.eval_episodes} versioned test starts`
                        : `${checkpoint.eval_episodes} legacy evaluation${checkpoint.eval_episodes === 1 ? "" : "s"}`}</small>
                    </td>
                    <td data-label={metricLabel}>
                      <strong>{formatMetric(checkpoint.eval_metric, metricLabel)}</strong>
                      {checkpoint.eval_metric_std != null && <small>σ {checkpoint.eval_metric_std.toFixed(2)}</small>}
                      {checkpoint.eval_failure_progress != null && <small>
                        Failed-start peak {checkpoint.eval_failure_progress.toFixed(3)}
                      </small>}
                    </td>
                    <td data-label="Evaluation return">
                      <strong>{checkpoint.eval_reward.toFixed(1)}</strong>
                      <small>± {checkpoint.eval_reward_std.toFixed(1)}</small>
                    </td>
                    <td data-label="Protocol">
                      <span>Training seed {checkpoint.seed ?? "unknown"}</span>
                      <small>Update {checkpoint.update_count.toLocaleString()} · {(checkpoint.total_steps ?? 0).toLocaleString()} steps</small>
                      <small>{checkpoint.protocol
                        ? `${checkpoint.protocol.algorithm} v${checkpoint.protocol.version} · γ ${checkpoint.protocol.gamma}`
                        : "Legacy protocol"}</small>
                      {checkpoint.protocol && <small title={sourceHash ?? undefined}>
                        {checkpoint.evaluation_suite ?? "unversioned evaluation"}
                        {sourceHash ? ` · source ${sourceHash.slice(0, 8)}` : ""}
                      </small>}
                      <details className="checkpoint-protocol-details">
                        <summary>Reproducibility details</summary>
                        <small>Evaluation seeds {checkpoint.protocol?.evaluation_seed_base ?? "unknown"}–{typeof checkpoint.protocol?.evaluation_seed_base === "number"
                          ? checkpoint.protocol.evaluation_seed_base + checkpoint.eval_episodes - 1
                          : "unknown"}</small>
                        <small>Schema {checkpoint.schema_version} · observations {checkpoint.obs_dim ?? "?"} · actions {(checkpoint.n_continuous ?? 0) + (checkpoint.n_binary ?? 0)}</small>
                        <small title={checkpoint.checkpoint_sha256 ?? undefined}>Tensor {checkpoint.checkpoint_sha256?.slice(0, 8) ?? "unverified"} · metadata {checkpoint.metadata_sha256?.slice(0, 8) ?? "legacy"}</small>
                        <small>Ranking: success rate, task metric, failed-start progress, then return.</small>
                      </details>
                    </td>
                    <td className="checkpoint-actions">
                      <button type="button" className={ghostActive ? "compare-active" : ""}
                        aria-pressed={ghostActive}
                        onClick={() => ghostActive ? clearGhost() : setGhost(checkpoint.episode)}>
                        {ghostActive ? "Hide replay" : "Compare replay"}
                      </button>
                      <button type="button" disabled={training}
                        title={training ? "Pause training before loading a checkpoint" : "Load weights and continue from here"}
                        onClick={() => resume(checkpoint)}>Resume</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {archivedRuns.length > 0 && (
        <details className="archived-runs">
          <summary>Archived branches ({archivedRuns.length})</summary>
          <div className="archived-run-list">
            {archivedRuns.map((run) => (
              <article key={run.id}>
                <div><strong>Through episode {run.latest_episode.toLocaleString()}</strong>
                  <small>{run.checkpoints} checkpoint{run.checkpoints === 1 ? "" : "s"} · seed {run.seed ?? "unknown"}</small></div>
                <button type="button" disabled={training}
                  onClick={() => restore(run.id, run.latest_episode)}>Restore branch</button>
              </article>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
