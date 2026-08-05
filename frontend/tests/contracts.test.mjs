import assert from "node:assert/strict";
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
