import { useEffect, useState } from "react";
import type { FrameMsg } from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

function signed(value: number): string {
  if (value > 0) return `+${value.toFixed(3)}`;
  return value.toFixed(3).replace("-", "−");
}

export default function LearningLens() {
  const { currentScenario, frameRef, history, ppo, status } = useTrainingSocket();
  const [frame, setFrame] = useState<FrameMsg | null>(null);

  useEffect(() => {
    const update = () => setFrame(frameRef.current);
    update();
    const timer = window.setInterval(update, 250);
    return () => window.clearInterval(timer);
  }, [frameRef]);

  const transition = frame?.learning;
  const lastEpisode = history.at(-1);
  const lastUpdate = ppo.at(-1);

  return (
    <section className="learning-lens" aria-labelledby="lens-title">
      <div className="lens-heading">
        <span className="section-kicker">Learning lens</span>
        <h2 id="lens-title">One step through the policy loop</h2>
      </div>

      <div className="lens-flow">
        <article className="lens-step lens-observe">
          <span className="lens-index">01</span>
          <div><h3>Agent sees</h3><p>First values in the normalized state vector</p></div>
          <div className="vector-values" aria-label="Current observation values">
            {(transition?.observation ?? []).map((value, index) => (
              <span key={index}><small title={currentScenario?.observation_dimensions[index]}>
                {currentScenario?.observation_dimensions[index] ?? `state ${index + 1}`}
              </small>{signed(value)}</span>
            ))}
            {!transition && <em>Waiting for the first transition</em>}
            {transition && currentScenario
              && currentScenario.observation_dimensions.length > transition.observation.length && (
              <em>+ {currentScenario.observation_dimensions.length - transition.observation.length} more state variables</em>
            )}
          </div>
        </article>

        <span className="flow-arrow" aria-hidden="true">→</span>

        <article className="lens-step lens-act">
          <span className="lens-index">02</span>
          <div><h3>Policy chooses</h3><p>Bounded actions sampled by PPO</p></div>
          <div className="action-values">
            {(transition?.action ?? []).map((value, index) => (
              <div key={index}>
                <span>{currentScenario?.actions[index] ?? `action ${index + 1}`}</span>
                <strong>{signed(value)}</strong>
              </div>
            ))}
            {!transition && <em>No action yet</em>}
          </div>
        </article>

        <span className="flow-arrow" aria-hidden="true">→</span>

        <article className="lens-step lens-reward">
          <span className="lens-index">03</span>
          <div><h3>Environment responds</h3><p>Immediate signal and episode result</p></div>
          <div className="reward-readout">
            <span className={(transition?.reward ?? 0) >= 0 ? "reward-positive" : "reward-negative"}>
              {transition ? signed(transition.reward) : "—"}<small> reward now</small>
            </span>
            <span>{frame ? frame.episode_reward.toFixed(1) : "—"}<small> episode return</small></span>
          </div>
        </article>
      </div>

      <div className="lens-footer">
        <span className="lens-status" aria-live="polite">
          {lastEpisode
            ? `Last episode: ${lastEpisode.success ? "success" : lastEpisode.cause.replaceAll("_", " ")} · return ${lastEpisode.reward}`
            : status?.training ? "Collecting the first episode…" : "Start a run to inspect the learning loop."}
        </span>
        <span>Policy entropy <strong>{lastUpdate?.entropy.toFixed(3) ?? "—"}</strong></span>
        <span>Update <strong>{status?.update_count ?? 0}</strong></span>
      </div>
    </section>
  );
}
