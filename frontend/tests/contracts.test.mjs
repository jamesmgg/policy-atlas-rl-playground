import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import * as api from "../src/api/types.ts";

test("metric formatting is readable across experiment families", () => {
  assert.equal(api.formatMetric(null, "best lap"), "Not measured");
  assert.equal(api.formatMetric(72.345, "best lap"), "72.34 s");
  assert.equal(api.formatMetric(0, "landing error"), "Landed");
  assert.equal(api.formatMetric(8.5, "balance time"), "8.50 s");
  assert.equal(api.formatMetric(3.251, "control effort"), "3.25 effort");
});

test("horizon formatting distinguishes physical time from benchmark steps", () => {
  assert.equal(api.formatHorizon(500, 10), "10 s · 500 steps");
  assert.equal(api.formatHorizon(999, null), "999 control steps");
});

test("experiment library filtering searches plain-language metadata", () => {
  assert.equal(typeof api.filterScenarios, "function");
  const scenarios = [
    { id: "cartpole", name: "Cart-Pole", group: "Foundations",
      description: "Balance an unstable pole", difficulty: "Introductory",
      objective: "Apply horizontal force", success: "Remain upright",
      observations: ["pole angle"], actions: ["signed force"], metric_label: "time" },
    { id: "wet", name: "Wet Circuit", group: "Driving",
      description: "Low-grip racing", difficulty: "Advanced" },
  ];
  assert.deepEqual(api.filterScenarios(scenarios, "balance", "All"), [scenarios[0]]);
  assert.deepEqual(api.filterScenarios(scenarios, "", "Foundations"), [scenarios[0]]);
  assert.deepEqual(api.filterScenarios(scenarios, "advanced", "All"), [scenarios[1]]);
  assert.deepEqual(api.filterScenarios(scenarios, "horizontal force", "All"), [scenarios[0]]);
});

test("run configuration clamps invalid values to safe experiment bounds", () => {
  assert.equal(typeof api.normalizeRunConfig, "function");
  assert.deepEqual(api.normalizeRunConfig(Number.NaN, 0), {
    episodes: 100,
    checkpointEvery: 10,
  });
  assert.deepEqual(api.normalizeRunConfig(2_000_000, 200_000), {
    episodes: 1_000_000,
    checkpointEvery: 100_000,
  });
});

test("paused run progress preserves its original budget", () => {
  const state = api.getRunWindow({
    training: false,
    episode: 150,
    run_start_episode: 100,
    run_target_episode: 500,
  });
  assert.deepEqual(state, { remaining: 350, progress: 12.5, resumable: true });
});

test("checkpoint ranking prioritizes success before a raw task metric", () => {
  assert.equal(typeof api.rankCheckpoints, "function");
  const failedButClose = { episode: 10, success_rate: 0, eval_metric: 0 };
  const reliable = { episode: 20, success_rate: 0.8, eval_metric: 12 };
  const lessReliable = { episode: 30, success_rate: 0.4, eval_metric: 5 };
  assert.deepEqual(
    api.rankCheckpoints([failedButClose, lessReliable, reliable], "min"),
    [reliable, lessReliable, failedButClose],
  );

  const lowerReturn = { episode: 40, success_rate: 1, eval_metric: 0, eval_reward: 20 };
  const higherReturn = { episode: 35, success_rate: 1, eval_metric: 0, eval_reward: 30 };
  assert.deepEqual(
    api.rankCheckpoints([lowerReturn, higherReturn], "min"),
    [higherReturn, lowerReturn],
  );

  const idle = { episode: 50, success_rate: 0, eval_metric: null,
    eval_failure_progress: -0.5, eval_reward: 0 };
  const nearSummit = { episode: 60, success_rate: 0, eval_metric: null,
    eval_failure_progress: 0.4, eval_reward: -12 };
  assert.deepEqual(
    api.rankCheckpoints([idle, nearSummit], "min"),
    [nearSummit, idle],
  );

  const efficientPartial = { episode: 70, success_rate: 0.5, eval_metric: 2,
    eval_failure_progress: 0.1, eval_reward: 20 };
  const costlyPartial = { episode: 80, success_rate: 0.5, eval_metric: 5,
    eval_failure_progress: 0.4, eval_reward: 25 };
  assert.deepEqual(
    api.rankCheckpoints([costlyPartial, efficientPartial], "min"),
    [efficientPartial, costlyPartial],
  );
});

test("checkpoint comparison requires the active evaluation suite and engine", () => {
  const current = {
    evaluation_suite: "policy-atlas-eval-v1-n10",
    engine_source_sha256: "a".repeat(64),
  };
  const checkpoint = {
    evaluation_suite: "policy-atlas-eval-v1-n10",
    protocol: { engine_source_sha256: "a".repeat(64) },
  };

  assert.equal(api.isComparableCheckpoint(checkpoint, current), true);
  assert.equal(api.isComparableCheckpoint(
    { ...checkpoint, evaluation_suite: null }, current,
  ), false);
  assert.equal(api.isComparableCheckpoint(
    { ...checkpoint, protocol: { engine_source_sha256: "b".repeat(64) } }, current,
  ), false);
});

test("recent checkpoints are newest first regardless of API order", () => {
  assert.deepEqual(
    api.recentCheckpoints([{ episode: 2 }, { episode: 10 }, { episode: 5 }])
      .map((checkpoint) => checkpoint.episode),
    [10, 5, 2],
  );
});

test("terminal frames are held briefly and explain why the episode reset", () => {
  const live = { type: "frame", scenario_id: "rally-ridge", episode: 8,
    episode_reward: 0 };
  const terminal = { type: "frame", scenario_id: "rally-ridge", episode: 7,
    episode_reward: -15, terminal: true, cause: "stall", terminal_steps: 300 };
  const held = { frame: terminal, receivedAt: 1_000 };

  assert.equal(api.selectVisibleFrame(live, held, 1_349), terminal);
  assert.equal(api.selectVisibleFrame(live, held, 1_350), live);
  assert.equal(api.selectVisibleFrame(null, held, 1_350), null);
  assert.equal(api.shouldCaptureTerminal(null, 1_000), true);
  assert.equal(api.shouldCaptureTerminal(held, 1_999), false);
  assert.equal(api.shouldCaptureTerminal(held, 2_000), true);
  const finalTerminal = { ...terminal, episode: 8, cause: "collision" };
  const latest = api.recordTerminalFrame(held, finalTerminal, 1_500);
  assert.equal(latest.frame, finalTerminal);
  assert.equal(latest.receivedAt, 1_000);
  assert.equal(api.selectVisibleFrame(live, latest, 1_500), live);
  assert.equal(api.selectVisibleFrame(live, latest, 1_500, undefined, true), finalTerminal);
  assert.equal(api.displayedEpisodeNumber(live, 99), 9);
  assert.equal(api.displayedEpisodeNumber(finalTerminal, 99), 9);
  assert.equal(api.displayedEpisodeNumber(null, 99), 99);
  assert.equal(api.formatTerminationCause("stall"), "No forward progress");
  assert.equal(api.formatTerminationCause("wrong_way"), "Wrong-way travel");
  assert.equal(api.formatEpisodeDuration(300, 1_500, 60), "12.0 simulated s");
  assert.equal(api.formatSimulationRate(508, 1_500, 60), "20x real time");
});

test("terminal reset explanations are announced to assistive technology", () => {
  const source = readFileSync(
    new URL("../src/features/simulator/SceneCanvas.tsx", import.meta.url),
    "utf8",
  );
  const learningLens = readFileSync(
    new URL("../src/features/simulator/LearningLens.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /className="termination-notice"[^>]*role="status"[^>]*aria-live="polite"[^>]*aria-atomic="true"/s,
  );
  assert.match(
    source,
    /ctx\.fillText\(`EP \$\{displayedEpisodeNumber\(frame, 0\)\}`/,
    "the canvas HUD must use the same one-based episode number",
  );
  assert.doesNotMatch(
    learningLens,
    /className="lens-status"[^>]*aria-live/,
    "rapid Learning Lens updates must not duplicate terminal announcements",
  );
});

test("the active experiment is visualization first with diagnostics collapsed", () => {
  const source = readFileSync(
    new URL("../src/features/simulator/SimulatorPage.tsx", import.meta.url),
    "utf8",
  );
  const simulator = source.indexOf("<SceneCanvas />");
  const topRuns = source.indexOf("<Leaderboard />");
  const diagnostics = source.indexOf('<details className="technical-drawer">');

  assert.ok(simulator >= 0, "the live simulator must remain on the experiment page");
  assert.ok(topRuns > simulator, "top runs must immediately follow the visualization");
  assert.ok(diagnostics > topRuns, "technical diagnostics belong after the top runs");
  assert.match(source, /<summary>Show learning diagnostics<\/summary>/);
  assert.ok(source.indexOf("<LearningLens />", diagnostics) > diagnostics);
  assert.ok(source.indexOf("<LearningCurve />", diagnostics) > diagnostics);
});

test("the top three evaluated runs receive accessible medal treatments", () => {
  const source = readFileSync(
    new URL("../src/features/simulator/Leaderboard.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /className="checkpoint-podium"/);
  assert.match(source, /Gold medal, first place/);
  assert.match(source, /Silver medal, second place/);
  assert.match(source, /Bronze medal, third place/);
  assert.match(source, /<details className="checkpoint-details">/);
});

test("Policy Atlas declares a dark instrument theme", () => {
  const styles = readFileSync(
    new URL("../src/styles/app.css", import.meta.url),
    "utf8",
  );

  assert.match(styles, /color-scheme:\s*dark/);
  assert.match(styles, /--lab-fog:\s*#07131c/i);
  assert.match(
    styles,
    /background-size:\s*auto,\s*32px 32px,\s*32px 32px/,
    "the ambient glow must not tile with the instrument grid",
  );
});

test("optimizer telemetry types include critic calibration and exploration", () => {
  const source = readFileSync(
    new URL("../src/api/types.ts", import.meta.url),
    "utf8",
  );

  assert.match(source, /explained_variance:\s*number/);
  assert.match(source, /value_bias:\s*number/);
  assert.match(source, /value_clip_frac:\s*number/);
  assert.match(source, /action_std_mean:\s*number/);

  const diagnostics = readFileSync(
    new URL("../src/features/simulator/LearningCurve.tsx", import.meta.url),
    "utf8",
  );
  assert.match(diagnostics, /title="Critic explained variance"/);
  assert.match(diagnostics, /title="Critic value bias"/);
  assert.match(diagnostics, /title="Action spread"/);
});

test("older checkpoint schemas remain visible but cannot be restored", () => {
  const types = readFileSync(
    new URL("../src/api/types.ts", import.meta.url),
    "utf8",
  );
  const leaderboard = readFileSync(
    new URL("../src/features/simulator/Leaderboard.tsx", import.meta.url),
    "utf8",
  );

  assert.match(types, /compatible:\s*boolean/);
  assert.match(types, /schema_version:\s*number/);
  assert.match(leaderboard, /training \|\| !run\.compatible/);
  assert.match(leaderboard, /Older scientific protocol/);
});

test("saved optimizer diagnostics survive reconnects and restored branches", () => {
  const diagnostics = {
    policy_loss: -0.02,
    value_loss: 0.4,
    entropy: 1.1,
    approx_kl: 0.01,
    clip_frac: 0.08,
    value_scale: 12,
    value_clip_frac: 0.03,
    explained_variance: 0.72,
    value_bias: -0.4,
    action_std_mean: 0.5,
    action_std_min: 0.4,
    action_std_max: 0.6,
  };
  const record = api.ppoRecordFromStatus({
    scenario_id: "rally-ridge",
    episode: 250,
    total_steps: 75_000,
    update_count: 31,
    sps: 2_500,
    ppo_diagnostics: diagnostics,
  });

  assert.deepEqual(record, {
    scenario_id: "rally-ridge",
    episode: 250,
    total_steps: 75_000,
    update: 31,
    sps: 2_500,
    ...diagnostics,
  });
  assert.equal(api.ppoRecordFromStatus({
    scenario_id: "rally-ridge", episode: 0, total_steps: 0,
    update_count: 0, sps: 0, ppo_diagnostics: null,
  }), null);
  assert.equal(api.ppoRecordFromStatus({
    scenario_id: "rally-ridge", episode: 20, total_steps: 100,
    update_count: 1, sps: 1, ppo_diagnostics: { policy_loss: 0.1 },
  }), null, "a partial legacy payload must not fabricate a chart point");

  const hook = readFileSync(
    new URL("../src/hooks/useTrainingSocket.tsx", import.meta.url),
    "utf8",
  );
  assert.match(hook, /ppoRecordFromStatus\(msg\)/);
});
