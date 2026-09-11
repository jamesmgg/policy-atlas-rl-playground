// Mirrors the backend JSON contracts (app/main.py, app/trainer.py, app/envs/*).

export interface TrackGeometry {
  centerline: [number, number][];
  normals: [number, number][];
  half_widths: number[];
  checkpoints: number[];
  total_length: number;
  start_index: number;
}

export interface ZoneInfo {
  start_idx: number;
  end_idx: number;
  grip_scale: number;
  color: string;
}

export interface StaticPrimitive {
  shape: "terrain" | "flag" | "circle" | "line" | "hill";
  points?: [number, number][];
  x?: number;
  y?: number;
  r?: number;
  color?: string;
}

export type SceneData =
  | { scenario_id: string; kind: "track"; track: TrackGeometry; zones: ZoneInfo[] }
  | { scenario_id: string; kind: "generic"; bounds: [number, number];
      primary_shape: string; statics: StaticPrimitive[] };

export interface CarFrame {
  x: number;
  y: number;
  heading: number;
  speed: number;
  drift: number;
  slip: number;
}

export interface GenericObject {
  shape: "lander" | "rod" | "drone" | "target" | "cartpole" | "mountain-car" | "spacecraft" | "station" | "robotarm" | "ballbeam";
  x: number;
  y: number;
  rot?: number;
  len?: number;
  flame?: number;
  force?: number;
  joint2?: number;
  thrust_x?: number;
  thrust_y?: number;
}

export interface FrameMsg {
  type: "frame";
  scenario_id: string;
  episode: number;
  episode_reward: number;
  terminal?: boolean;
  cause?: string | null;
  terminal_steps?: number | null;
  learning?: { observation: number[]; action: number[]; reward: number } | null;
  car?: CarFrame;
  laps?: number;
  last_lap?: number | null;
  best_lap?: number | null;
  bots?: { x: number; y: number; heading: number }[];
  style?: number;
  overtakes?: number;
  objects?: GenericObject[];
  waypoints?: number;
  balance_time?: number;
  peak_position?: number;
  fuel?: number;
  distance?: number;
  hold?: number;
  speed?: number;
  delta_v?: number;
  tracking_hits?: number;
}

export interface EpisodeRecord {
  episode: number;
  reward: number;
  steps: number;
  cause: string;
  metric: number | null;
  success?: boolean;
  peak_progress?: number;
  progress_fraction?: number;
  failure_progress?: number | null;
  laps?: number;
  best_lap?: number | null;
  rollingMean?: number;
  successMean?: number;
}

export interface PpoUpdateRecord {
  scenario_id: string;
  episode: number;
  total_steps: number;
  update: number;
  sps: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  approx_kl: number;
  clip_frac: number;
  value_scale: number;
  value_clip_frac: number;
  explained_variance: number;
  value_bias: number;
  action_std_mean: number;
  action_std_min: number;
  action_std_max: number;
  action_std_0?: number;
  action_std_1?: number;
}

export type PpoDiagnostics = Omit<PpoUpdateRecord,
  "scenario_id" | "episode" | "total_steps" | "update" | "sps">;

export interface CheckpointMeta {
  episode: number;
  timestamp: string;
  mean_reward: number;
  eval_reward: number;
  eval_metric: number | null;
  eval_reward_std: number;
  eval_metric_std: number | null;
  eval_failure_progress: number | null;
  eval_episodes: number;
  success_rate: number | null;
  success_ci_low: number | null;
  success_ci_high: number | null;
  evaluation_suite: string | null;
  seed: number | null;
  update_count: number;
  total_steps: number | null;
  schema_version: number;
  obs_dim: number | null;
  n_continuous: number | null;
  n_binary: number | null;
  protocol: {
    algorithm: string;
    version: number;
    rollout_steps: number;
    gamma: number;
    gae_lambda: number;
    [key: string]: string | number | boolean;
  } | null;
  training_diagnostics: Partial<PpoDiagnostics> | null;
  metadata_sha256: string | null;
  checkpoint_sha256: string | null;
}

export interface StatusMsg {
  type: "status";
  scenario_id: string;
  scenario_kind: "driving" | "generic";
  metric_label: string;
  metric_mode: "min" | "max";
  training: boolean;
  last_error?: string | null;
  cpu_threads?: number;
  episode: number;
  max_episodes: number;
  run_start_episode: number;
  run_target_episode: number;
  checkpoint_every_n: number;
  total_steps: number;
  update_count: number;
  sps: number;
  seed: number;
  eval_episodes: number;
  evaluation_suite: string;
  evaluation_seed_base: number;
  engine_source_sha256: string;
  best_reward: number | null;
  best_metric: number | null;
  device: string;
  ghost_episode: number | null;
  ppo_diagnostics: Partial<PpoDiagnostics> | null;
}

export interface ScenarioProgress {
  episode: number;
  mean_reward: number;
  best_metric: number | null;
  checkpoints: number;
}

export interface ArchivedRun {
  id: string;
  latest_episode: number;
  checkpoints: number;
  seed: number | null;
  timestamp: string | null;
  schema_version: number;
  compatible: boolean;
}

export interface ScenarioInfo {
  id: string;
  name: string;
  group: string;
  kind: "driving" | "generic";
  description: string;
  metric_label: string;
  metric_mode: "min" | "max";
  objective: string;
  success: string;
  observations: string[];
  observation_dimensions: string[];
  actions: string[];
  reward_terms: string[];
  termination_conditions: string[];
  difficulty: string;
  horizon_steps: number;
  horizon_seconds: number | null;
  progress: ScenarioProgress | null;
  reference_controller?: string | null;
  model_assumptions?: string[];
}

const REQUIRED_PPO_DIAGNOSTICS = [
  "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
  "value_scale", "value_clip_frac", "explained_variance", "value_bias",
  "action_std_mean", "action_std_min", "action_std_max",
] as const satisfies readonly (keyof PpoDiagnostics)[];

/** Rebuild the latest chart point persisted in status/checkpoint metadata. */
export function ppoRecordFromStatus(status: Pick<StatusMsg,
  "scenario_id" | "episode" | "total_steps" | "update_count" | "sps" | "ppo_diagnostics"
>): PpoUpdateRecord | null {
  const diagnostics = status.ppo_diagnostics;
  if (!diagnostics || !REQUIRED_PPO_DIAGNOSTICS.every((key) => (
    typeof diagnostics[key] === "number" && Number.isFinite(diagnostics[key])
  ))) return null;
  return {
    ...diagnostics,
    scenario_id: status.scenario_id,
    episode: status.episode,
    total_steps: status.total_steps,
    update: status.update_count,
    sps: status.sps,
  } as PpoUpdateRecord;
}

// trajectory rows: [x, y, rotation, drift, speed]
export interface GhostLap {
  episode: number;
  dt: number;
  trajectory: [number, number, number, number, number][];
}

export type ServerMessage =
  | FrameMsg
  | StatusMsg
  | ({ type: "episode_end"; scenario_id: string; checkpoint_due: boolean } & EpisodeRecord)
  | ({ type: "ppo_update" } & PpoUpdateRecord)
  | { type: "checkpoint_list"; scenario_id: string; checkpoints: CheckpointMeta[] }
  | { type: "history"; scenario_id: string; history: EpisodeRecord[] }
  | ({ type: "ghost_lap" } & GhostLap)
  | { type: "ghost_clear" }
  | ({ type: "scenario_changed" } & Omit<ScenarioInfo, "progress">)
  | { type: "error"; message: string };

export type ClientMessage =
  | { type: "start_training"; max_episodes: number; checkpoint_every_n: number }
  | { type: "stop_training" }
  | { type: "reset_training"; seed: number }
  | { type: "set_ghost"; episode: number }
  | { type: "clear_ghost" }
  | { type: "load_checkpoint"; episode: number };

export interface ScenarioSearchable {
  id: string;
  name: string;
  group: string;
  description: string;
  difficulty: string;
  objective?: string;
  success?: string;
  observations?: string[];
  observation_dimensions?: string[];
  actions?: string[];
  metric_label?: string;
  reward_terms?: string[];
  termination_conditions?: string[];
}

export function filterScenarios<T extends ScenarioSearchable>(
  scenarios: T[], query: string, group: string,
): T[] {
  const needle = query.trim().toLocaleLowerCase();
  return scenarios.filter((scenario) => {
    if (group !== "All" && scenario.group !== group) return false;
    if (!needle) return true;
    return [
      scenario.name, scenario.description, scenario.group, scenario.difficulty,
      scenario.objective, scenario.success, scenario.metric_label,
      ...(scenario.observations ?? []), ...(scenario.actions ?? []),
      ...(scenario.observation_dimensions ?? []),
      ...(scenario.reward_terms ?? []), ...(scenario.termination_conditions ?? []),
    ].filter((value): value is string => Boolean(value))
      .some((value) => value.toLocaleLowerCase().includes(needle));
  });
}

export function normalizeRunConfig(
  episodes: number, checkpointEvery: number,
): { episodes: number; checkpointEvery: number } {
  const safeEpisodes = Number.isFinite(episodes) && episodes > 0 ? Math.round(episodes) : 100;
  const safeCheckpoint = Number.isFinite(checkpointEvery) && checkpointEvery > 0
    ? Math.round(checkpointEvery) : 10;
  return {
    episodes: Math.min(1_000_000, Math.max(1, safeEpisodes)),
    checkpointEvery: Math.min(100_000, Math.max(1, safeCheckpoint)),
  };
}

export function getRunWindow(status: Pick<StatusMsg,
  "training" | "episode" | "run_start_episode" | "run_target_episode">
): { remaining: number; progress: number; resumable: boolean } {
  const remaining = Math.max(status.run_target_episode - status.episode, 0);
  const budget = Math.max(status.run_target_episode - status.run_start_episode, 1);
  const completed = Math.max(status.episode - status.run_start_episode, 0);
  return {
    remaining,
    progress: Math.min(100, completed / budget * 100),
    resumable: !status.training && remaining > 0,
  };
}

export function rankCheckpoints<T extends {
  success_rate: number | null;
  eval_metric: number | null;
  eval_failure_progress?: number | null;
  eval_reward?: number;
  episode: number;
}>(checkpoints: T[], mode: "min" | "max"): T[] {
  return checkpoints.slice().sort((a, b) => {
    const successDelta = (b.success_rate ?? -1) - (a.success_rate ?? -1);
    if (successDelta !== 0) return successDelta;
    if (a.eval_metric == null && b.eval_metric == null) {
      const progressDelta = (b.eval_failure_progress ?? -Infinity)
        - (a.eval_failure_progress ?? -Infinity);
      if (progressDelta !== 0) return progressDelta;
      const rewardDelta = (b.eval_reward ?? -Infinity) - (a.eval_reward ?? -Infinity);
      return rewardDelta || b.episode - a.episode;
    }
    if (a.eval_metric == null) return 1;
    if (b.eval_metric == null) return -1;
    const metricDelta = mode === "min"
      ? a.eval_metric - b.eval_metric
      : b.eval_metric - a.eval_metric;
    if (metricDelta !== 0) return metricDelta;
    if (a.eval_failure_progress != null || b.eval_failure_progress != null) {
      const progressDelta = (b.eval_failure_progress ?? -Infinity)
        - (a.eval_failure_progress ?? -Infinity);
      if (progressDelta !== 0) return progressDelta;
    }
    const rewardDelta = (b.eval_reward ?? -Infinity) - (a.eval_reward ?? -Infinity);
    return rewardDelta || b.episode - a.episode;
  });
}

export function isComparableCheckpoint(
  checkpoint: {
    evaluation_suite: string | null;
    protocol: { [key: string]: string | number | boolean } | null;
  },
  current: { evaluation_suite: string; engine_source_sha256: string },
): boolean {
  const checkpointSource = checkpoint.protocol?.engine_source_sha256;
  return checkpoint.evaluation_suite === current.evaluation_suite
    && typeof checkpointSource === "string"
    && checkpointSource === current.engine_source_sha256;
}

export function recentCheckpoints<T extends { episode: number }>(checkpoints: T[]): T[] {
  return checkpoints.slice().sort((a, b) => b.episode - a.episode);
}

export function formatMetric(value: number | null | undefined, label: string): string {
  if (value == null) return "Not measured";
  if (label === "best lap" || label === "balance time") return `${value.toFixed(2)} s`;
  if (label === "landing error") return value === 0 ? "Landed" : `${value.toFixed(0)} error`;
  if (label === "control effort") return `${value.toFixed(2)} effort`;
  if (label === "laps on tank") return `${value.toFixed(2)} laps`;
  if (label === "style pts") return `${value.toFixed(1)} pts`;
  if (Number.isInteger(value)) return `${value}`;
  return value.toFixed(2);
}

export interface HeldTerminalFrame {
  frame: FrameMsg;
  receivedAt: number;
}

export const TERMINAL_HOLD_MS = 350;
export const TERMINAL_CAPTURE_COOLDOWN_MS = 1_000;

export function shouldCaptureTerminal(
  previous: HeldTerminalFrame | null,
  now: number,
  cooldownMs = TERMINAL_CAPTURE_COOLDOWN_MS,
): boolean {
  return previous == null || now - previous.receivedAt >= cooldownMs;
}

export function recordTerminalFrame(
  previous: HeldTerminalFrame | null,
  frame: FrameMsg,
  now: number,
): HeldTerminalFrame {
  return {
    frame,
    receivedAt: shouldCaptureTerminal(previous, now)
      ? now
      : previous!.receivedAt,
  };
}

export function selectVisibleFrame(
  live: FrameMsg | null,
  terminal: HeldTerminalFrame | null,
  now: number,
  holdMs = TERMINAL_HOLD_MS,
  terminalPinned = false,
): FrameMsg | null {
  if (terminal && terminalPinned) return terminal.frame;
  if (terminal && now - terminal.receivedAt < holdMs) return terminal.frame;
  return live;
}

export function displayedEpisodeNumber(
  frame: FrameMsg | null,
  completedEpisodes: number,
): number {
  if (!frame) return completedEpisodes;
  // Trainer frames carry the zero-based count before the current episode is
  // committed; both live and terminal views therefore represent episode N+1.
  return frame.episode + 1;
}

export function formatTerminationCause(cause: string): string {
  const labels: Record<string, string> = {
    collision: "Off-track collision",
    contact: "Traffic contact",
    fuel: "Fuel exhausted",
    wrong_way: "Wrong-way travel",
    stall: "No forward progress",
    timeout: "Time limit reached",
    crash: "Crash",
    tipped: "Pole tipped",
    out_of_bounds: "Out of bounds",
    balanced: "Balance target reached",
    summit: "Summit reached",
    landed: "Safe landing",
    docked: "Stable rendezvous",
    escaped: "Escaped approach zone",
    reached: "Target reached and settled",
    tracked: "Moving target tracked",
    fell: "Ball left the beam",
    complete: "Route complete",
  };
  return labels[cause] ?? cause.replaceAll("_", " ");
}

export function formatEpisodeDuration(
  steps: number, horizonSteps: number, horizonSeconds: number | null,
): string {
  if (horizonSeconds == null || horizonSteps <= 0) {
    return `${steps.toLocaleString()} control steps`;
  }
  return `${(steps * horizonSeconds / horizonSteps).toFixed(1)} simulated s`;
}

export function formatSimulationRate(
  stepsPerSecond: number, horizonSteps: number, horizonSeconds: number | null,
): string {
  if (horizonSeconds == null || horizonSteps <= 0) {
    return `${Math.round(stepsPerSecond).toLocaleString()} steps/s`;
  }
  return `${Math.round(stepsPerSecond * horizonSeconds / horizonSteps)}x real time`;
}

export function formatHorizon(steps: number, seconds: number | null): string {
  if (seconds == null) return `${steps.toLocaleString()} control steps`;
  return `${seconds.toLocaleString()} s · ${steps.toLocaleString()} steps`;
}

export interface ViewBounds {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface ScrollPosition {
  left: number;
  top: number;
}

export function getRevealScrollPosition(
  container: ViewBounds,
  target: ViewBounds,
  current: ScrollPosition,
): ScrollPosition | null {
  let left = current.left;
  let top = current.top;

  if (target.left < container.left) left += target.left - container.left;
  else if (target.right > container.right) left += target.right - container.right;
  if (target.top < container.top) top += target.top - container.top;
  else if (target.bottom > container.bottom) top += target.bottom - container.bottom;

  left = Math.max(0, left);
  top = Math.max(0, top);
  return left === current.left && top === current.top ? null : { left, top };
}

export function scrollExperimentStage(
  target: Pick<Element, "scrollIntoView"> | null,
  compactViewport: boolean,
  reducedMotion: boolean,
): boolean {
  if (!target || !compactViewport) return false;
  target.scrollIntoView({
    behavior: reducedMotion ? "auto" : "smooth",
    block: "start",
  });
  return true;
}

export function focusExperimentHeading(
  target: Pick<HTMLElement, "focus"> | null,
  compactViewport: boolean,
): boolean {
  if (!target || !compactViewport) return false;
  target.focus({ preventScroll: true });
  return true;
}

export function replayPosition(
  replay: { lap: GhostLap; startedAt: number }, now: number,
) {
  const { lap, startedAt } = replay;
  if (!lap.trajectory.length || !Number.isFinite(lap.dt) || lap.dt <= 0) return null;
  const duration = lap.trajectory.length * lap.dt;
  const elapsed = Math.min(duration, Math.max(0, (now - startedAt) / 1000));
  return {
    episode: lap.episode,
    index: Math.min(lap.trajectory.length - 1, Math.floor(elapsed / lap.dt)),
    elapsed, duration, finished: elapsed >= duration,
  };
}

export type MobileView = "watch" | "projects" | "train" | "results";

export function mobileViewFromHash(hash: string): MobileView {
  const routes: Record<string, MobileView> = {
    "#watch": "watch", "#experiment-stage": "watch",
    "#projects": "projects", "#experiment-library": "projects",
    "#train": "train", "#run-setup": "train",
    "#results": "results", "#checkpoints-title": "results",
  };
  return routes[hash] ?? "watch";
}
