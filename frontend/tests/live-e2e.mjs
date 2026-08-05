import assert from "node:assert/strict";

const ORIGIN = process.env.RL_PLAYGROUND_URL ?? "http://localhost:8900";

async function json(path, init) {
  const response = await fetch(`${ORIGIN}${path}`, init);
  assert.equal(response.ok, true, `${path} returned ${response.status}`);
  return response.json();
}

async function waitFor(check, timeoutMs = 45_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const result = await check();
    if (result) return result;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`condition not met within ${timeoutMs} ms`);
}

const catalog = await json("/api/scenarios");
assert.equal(catalog.scenarios.length, 16);
assert.ok(catalog.scenarios.some((item) => item.id === "cartpole-balance"));
assert.ok(catalog.scenarios.every((item) => item.objective && item.actions.length));

await json("/api/scenario", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ id: "cartpole-balance" }),
});
const scene = await json("/api/scene");
assert.equal(scene.primary_shape, "cartpole");

const wsUrl = ORIGIN.replace(/^http/, "ws") + "/ws/training";
const socket = new WebSocket(wsUrl);
const events = [];
socket.addEventListener("message", (event) => events.push(JSON.parse(event.data)));
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

await json("/api/training/start", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ max_episodes: 2, checkpoint_every_n: 1 }),
});

await waitFor(async () => {
  const status = await json("/api/training/status");
  return !status.training && status.episode >= 2 ? status : null;
});
await waitFor(() => events.some((event) => event.type === "frame" && event.learning));

const checkpoints = await json("/api/checkpoints");
assert.equal(checkpoints.checkpoints.length, 1);
assert.equal(checkpoints.checkpoints.at(-1).episode, 2);
assert.equal(checkpoints.checkpoints.at(-1).eval_episodes, 10);
assert.equal(typeof checkpoints.checkpoints.at(-1).eval_reward_std, "number");
assert.equal(typeof checkpoints.checkpoints.at(-1).success_rate, "number");
assert.equal(checkpoints.checkpoints.at(-1).evaluation_suite, "policy-atlas-eval-v1-n10");
assert.equal(checkpoints.checkpoints.at(-1).protocol.algorithm, "PPO");
assert.match(checkpoints.checkpoints.at(-1).protocol.engine_source_sha256, /^[a-f0-9]{64}$/);
assert.match(checkpoints.checkpoints.at(-1).metadata_sha256, /^[a-f0-9]{64}$/);
assert.equal(checkpoints.checkpoints.at(-1).schema_version, 1);
assert.ok(events.some((event) => event.type === "ppo_update"));

const reset = await json("/api/training/reset", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ seed: 2026 }),
});
assert.equal(reset.episode, 0);
assert.equal(reset.seed, 2026);
assert.deepEqual((await json("/api/checkpoints")).checkpoints, []);

const archived = await json("/api/runs");
assert.equal(archived.archives.length, 1);
const restored = await json(`/api/runs/${encodeURIComponent(archived.archives[0].id)}/restore`, {
  method: "POST",
});
assert.equal(restored.episode, 2);
assert.equal((await json("/api/checkpoints")).checkpoints.length, 1);

await json("/api/training/reset", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ seed: 2026 }),
});

socket.close();
console.log("LIVE E2E PASSED: catalog → scenario → frames → PPO update → evaluation → reset → branch restore");
