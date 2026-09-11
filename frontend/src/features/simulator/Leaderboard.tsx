import { useMemo, useState } from "react";
import type { CheckpointMeta } from "../../api/types";
import {
  formatMetric, isComparableCheckpoint, rankCheckpoints, recentCheckpoints,
} from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

const MEDALS = [
  { tone: "gold", label: "Gold", accessible: "Gold medal, first place" },
  { tone: "silver", label: "Silver", accessible: "Silver medal, second place" },
  { tone: "bronze", label: "Bronze", accessible: "Bronze medal, third place" },
] as const;

function savedAt(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export default function Leaderboard({ onWatch }: { onWatch?: () => void }) {
  const {
    checkpoints, archivedRuns, ghostEpisode, ghostRef, status, metricLabel, metricMode,
    setGhost, setArchiveGhost, clearGhost, loadCheckpoint, restoreArchivedRun,
  } = useTrainingSocket();
  const [view, setView] = useState<"best" | "recent">("best");
  const training = status?.training ?? false;
  const verifiedRuns = archivedRuns.filter((run) => run.compatible && run.requalification);
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
  const podiumRuns = ranked.slice(0, 3);
  const hasSolvedRun = podiumRuns.some((checkpoint) => (checkpoint.success_rate ?? 0) > 0);
  const fullMedalPodium = podiumRuns.length === MEDALS.length
    && podiumRuns.every((checkpoint) => (checkpoint.success_rate ?? 0) > 0);
  const rows = view === "best" ? ranked : recentCheckpoints(checkpoints);
  const watchReplay = (episode: number) => {
    if (ghostEpisode === episode && !ghostRef.current?.lap.archive_id) clearGhost();
    else { setGhost(episode); onWatch?.(); }
  };

  const resume = (checkpoint: CheckpointMeta) => {
    if (!status || !isComparableCheckpoint(checkpoint, status)) return;
    if (window.confirm(
      `Load episode ${checkpoint.episode} for further training? This replaces the current policy and leaves training paused. To see its saved performance, choose Watch replay instead.`,
    )) loadCheckpoint(checkpoint.episode);
  };

  const restore = (id: string, episode: number) => {
    if (window.confirm(
      `Restore the archived branch ending at episode ${episode}? The active branch will be archived first.`,
    )) void restoreArchivedRun(id);
  };

  return (
    <section className="panel checkpoints-panel" aria-labelledby="checkpoints-title">
      <div className="panel-heading checkpoint-heading">
        <div>
          <span className="section-kicker">{hasSolvedRun ? "Policy podium" : "Best attempts"}</span>
          <h2 id="checkpoints-title">Top runs</h2>
          <p>{hasSolvedRun
            ? "The strongest policies on the current fixed test starts."
            : ranked.length ? "No policy has passed a test start yet. These are the closest attempts so far."
              : "Train a policy to evaluate its first test starts."}</p>
        </div>
        <span className="ranking-note">Success first · task score breaks ties</span>
      </div>

      {verifiedRuns.length > 0 && <div className="verified-policies">
        <h3>Ready to watch</h3>
        <p>Saved neural policies tested on this version. Watching leaves your training policy in place.</p>
        <div className="verified-policy-list">{verifiedRuns.map((run) => (
          <article key={run.id}>
            <div><strong>Verified policy · seed {run.seed}</strong>
              <small>{run.requalification!.holdout_successes}/{run.requalification!.holdout_episodes} separate test starts passed · successful replay</small>
              <small>Originally trained through episode {run.requalification!.origin.episode}</small>
            </div>
            <button type="button" onClick={() => { setArchiveGhost(run.id, run.latest_episode); onWatch?.(); }}>Watch verified policy</button>
          </article>
        ))}</div>
      </div>}

      {ranked.length === 0 ? (
        <div className="checkpoint-empty">
          <span aria-hidden="true">◇</span>
          {checkpoints.length > 0
            ? <div><h3>No comparable runs yet</h3><p>Older-protocol runs remain available in the saved-run details below.</p></div>
            : <div><h3>The podium is waiting</h3><p>Train a policy to place its first evaluated run.</p></div>}
        </div>
      ) : (
        <>
          <div className="checkpoint-podium" aria-label="Top three evaluated runs">
            {podiumRuns.map((checkpoint, index) => {
              const medal = MEDALS[index];
              const ghostActive = ghostEpisode === checkpoint.episode && !ghostRef.current?.lap.archive_id;
              const solved = (checkpoint.success_rate ?? 0) > 0;
              const comparable = status
                ? isComparableCheckpoint(checkpoint, status)
                : false;
              return (
                <article className={`podium-card ${
                  solved ? `podium-${medal.tone}` : "podium-attempt"
                }`} key={checkpoint.episode}>
                  <div className="podium-card-topline">
                    {solved ? (
                      <>
                        <span className={`podium-medal medal-${medal.tone}`} role="img" aria-label={medal.accessible}>
                          <span aria-hidden="true">{index + 1}</span>
                        </span>
                        <span className="medal-name" aria-hidden="true">{medal.label}</span>
                      </>
                    ) : (
                      <>
                        <span className="podium-attempt-rank" role="img"
                          aria-label={`Best attempt, rank ${index + 1}`}>{index + 1}</span>
                        <span className="attempt-name" aria-hidden="true">Best attempt</span>
                      </>
                    )}
                  </div>
                  <div className="podium-run-copy">
                    <strong>Episode {checkpoint.episode.toLocaleString()}</strong>
                    <small>{savedAt(checkpoint.timestamp)}</small>
                  </div>
                  <dl className="podium-score">
                    <div><dt>Success</dt><dd>{checkpoint.success_rate != null
                      ? `${Math.round(checkpoint.success_rate * 100)}%` : "Legacy"}</dd></div>
                    <div><dt>{metricLabel}</dt><dd>{formatMetric(checkpoint.eval_metric, metricLabel)}</dd></div>
                  </dl>
                  <div className="podium-actions">
                    <button type="button" className={ghostActive ? "compare-active" : ""}
                      aria-pressed={ghostActive}
                      onClick={() => watchReplay(checkpoint.episode)}>
                      {ghostActive ? "Hide replay" : "Watch replay"}
                    </button>
                    <button type="button" disabled={training || !comparable}
                      title={!comparable
                        ? "Different experiment protocol: replay is available, but resume is disabled"
                        : training ? "Pause training before loading a checkpoint" : "Load this policy for further training; playback uses Watch replay"}
                      onClick={() => resume(checkpoint)}>Load for training</button>
                  </div>
                </article>
              );
            })}
          </div>
          {ranked.length > 1 && (
            <p className="podium-scroll-cue"><span aria-hidden="true">↔</span> {
              fullMedalPodium ? "Swipe for Silver and Bronze" : "Swipe for more ranked runs"
            }</p>
          )}
        </>
      )}

      <details className="checkpoint-details">
        <summary>
          <span>All saved runs</span>
          <small>Scores, confidence intervals, and reproducibility details</small>
        </summary>
        <div className="checkpoint-details-body">
          <div className="view-tabs" role="group" aria-label="Checkpoint order">
            <button type="button" aria-pressed={view === "best"} onClick={() => setView("best")}>Best evaluated</button>
            <button type="button" aria-pressed={view === "recent"} onClick={() => setView("recent")}>Most recent</button>
          </div>

          {rows.length === 0 ? (
            <div className="checkpoint-empty compact-empty">
              <div><h3>No runs in this view</h3><p>Saved checkpoints will appear here after evaluation.</p></div>
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
                    const ghostActive = ghostEpisode === checkpoint.episode && !ghostRef.current?.lap.archive_id;
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
                            onClick={() => watchReplay(checkpoint.episode)}>
                            {ghostActive ? "Hide replay" : "Watch replay"}
                          </button>
                          <button type="button" disabled={training || !comparable}
                            title={!comparable
                              ? "Different experiment protocol: replay is available, but resume is disabled"
                              : training ? "Pause training before loading a checkpoint" : "Load this policy for further training; playback uses Watch replay"}
                            onClick={() => resume(checkpoint)}>Load for training</button>
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
                  <article key={run.id} className={!run.compatible ? "archived-run-incompatible" : undefined}>
                    <div>
                      <strong>Through episode {run.latest_episode.toLocaleString()}</strong>
                      <small>{run.checkpoints} checkpoint{run.checkpoints === 1 ? "" : "s"} · seed {run.seed ?? "unknown"} · schema {run.schema_version}</small>
                      {!run.compatible && <small>Older scientific protocol · kept for audit only</small>}
                    </div>
                    <button type="button" disabled={training || !run.compatible}
                      title={!run.compatible
                        ? "Older scientific protocol: this branch does not match the current engine, environment schema, or test suite"
                        : training ? "Pause training before restoring a branch" : "Restore this saved branch"}
                      onClick={() => restore(run.id, run.latest_episode)}>
                      {run.compatible ? "Restore branch" : "Older scientific protocol"}
                    </button>
                  </article>
                ))}
              </div>
            </details>
          )}
        </div>
      </details>
    </section>
  );
}
