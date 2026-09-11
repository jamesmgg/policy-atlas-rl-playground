import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import { ppoRecordFromStatus, recordTerminalFrame } from "../api/types";
import type {
  ArchivedRun, CheckpointMeta, ClientMessage, EpisodeRecord, FrameMsg, GhostLap,
  HeldTerminalFrame, PpoUpdateRecord, PublicDemoData, PublicDemoPolicy, ScenarioInfo,
  SceneData, ServerMessage, StatusMsg,
} from "../api/types";

const FLUSH_MS = 400;
const MAX_HISTORY_POINTS = 2400;
const MAX_PPO_POINTS = 600;
const ROLLING_WINDOW = 20;

export interface TrainingSocketValue {
  readOnly: boolean;
  scene: SceneData | null;
  publicQualification: PublicDemoData["qualification"] | null;
  connected: boolean;
  connectionState: "connecting" | "connected" | "reconnecting";
  status: StatusMsg | null;
  scenarios: ScenarioInfo[];
  currentScenario: ScenarioInfo | null;
  scenarioId: string | null;
  scenarioKind: "driving" | "generic";
  metricLabel: string;
  metricMode: "min" | "max";
  history: EpisodeRecord[];
  ppo: PpoUpdateRecord[];
  checkpoints: CheckpointMeta[];
  archivedRuns: ArchivedRun[];
  ghostEpisode: number | null;
  replayRevision: number;
  lastError: string | null;
  /** Latest live frame — read inside rAF loops, never triggers re-renders. */
  frameRef: React.RefObject<FrameMsg | null>;
  terminalFrameRef: React.RefObject<HeldTerminalFrame | null>;
  /** Active ghost lap trajectory, with the time it was activated. */
  ghostRef: React.RefObject<{ lap: GhostLap; startedAt: number } | null>;
  startTraining(maxEpisodes: number, checkpointEveryN: number, pauseOnSuccess?: boolean): boolean;
  stopTraining(): void;
  resetTraining(seed: number): void;
  setGhost(episode: number): void;
  setArchiveGhost(id: string, episode: number): void;
  restartReplay(): void;
  clearGhost(): void;
  loadCheckpoint(episode: number): void;
  restoreArchivedRun(id: string): Promise<void>;
  selectScenario(id: string): Promise<boolean>;
  clearError(): void;
}

const Ctx = createContext<TrainingSocketValue | null>(null);

function decimateHalf(items: EpisodeRecord[]): EpisodeRecord[] {
  // Halve the older half of the array; recent episodes stay at full detail.
  const half = Math.floor(items.length / 2);
  const compressed = items.slice(0, half).filter((_, i) => i % 2 === 0);
  return compressed.concat(items.slice(half));
}

function LiveTrainingSocketProvider({ children }: { children: React.ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<TrainingSocketValue["connectionState"]>("connecting");
  const [status, setStatus] = useState<StatusMsg | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [scenarioKind, setScenarioKind] = useState<"driving" | "generic">("driving");
  const [metricLabel, setMetricLabel] = useState("best lap");
  const [metricMode, setMetricMode] = useState<"min" | "max">("min");
  const [history, setHistory] = useState<EpisodeRecord[]>([]);
  const [ppo, setPpo] = useState<PpoUpdateRecord[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointMeta[]>([]);
  const [archivedRuns, setArchivedRuns] = useState<ArchivedRun[]>([]);
  const [ghostEpisode, setGhostEpisode] = useState<number | null>(null);
  const [replayRevision, setReplayRevision] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);

  const frameRef = useRef<FrameMsg | null>(null);
  const terminalFrameRef = useRef<HeldTerminalFrame | null>(null);
  const ghostRef = useRef<{ lap: GhostLap; startedAt: number } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const scenarioRef = useRef<string | null>(null);
  const pendingEpisodes = useRef<EpisodeRecord[]>([]);
  const pendingPpo = useRef<PpoUpdateRecord[]>([]);
  const persistedPpo = useRef<PpoUpdateRecord | null>(null);
  const rollingWindow = useRef<number[]>([]);
  const connectedOnce = useRef(false);

  const send = useCallback((msg: ClientMessage) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      setLastError(null);
      ws.send(JSON.stringify(msg));
      return true;
    } else {
      setLastError("The trainer is reconnecting. Try the action again in a moment.");
      return false;
    }
  }, []);

  const clearScenarioState = useCallback(() => {
    pendingEpisodes.current = [];
    pendingPpo.current = [];
    persistedPpo.current = null;
    rollingWindow.current = [];
    frameRef.current = null;
    terminalFrameRef.current = null;
    ghostRef.current = null;
    setHistory([]);
    setPpo([]);
    setCheckpoints([]);
    setArchivedRuns([]);
    setGhostEpisode(null);
  }, []);

  useEffect(() => {
    let closed = false;
    let retryDelay = 500;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/training`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (closed || wsRef.current !== ws) return;
        setConnected(true);
        setConnectionState("connected");
        connectedOnce.current = true;
        setLastError(null);
        retryDelay = 500;
      };
      ws.onclose = () => {
        // A closing connection from an earlier effect must not clear its replacement.
        if (closed || wsRef.current !== ws) return;
        setConnected(false);
        setConnectionState(connectedOnce.current ? "reconnecting" : "connecting");
        wsRef.current = null;
        if (!closed) {
          retryTimer = setTimeout(connect, retryDelay);
          retryDelay = Math.min(retryDelay * 2, 5000);
        }
      };
      ws.onmessage = (event) => {
        if (closed || wsRef.current !== ws) return;
        let msg: ServerMessage;
        try {
          msg = JSON.parse(event.data) as ServerMessage;
        } catch {
          setLastError("The trainer sent an unreadable update. Reconnecting may help.");
          return;
        }
        const sid = scenarioRef.current;
        switch (msg.type) {
          case "frame":
            if (sid && msg.scenario_id !== sid) return;
            if (msg.terminal) {
              const receivedAt = performance.now();
              terminalFrameRef.current = recordTerminalFrame(
                terminalFrameRef.current, msg, receivedAt,
              );
            } else {
              frameRef.current = msg;
            }
            break;
          case "episode_end": {
            if (sid && msg.scenario_id !== sid) return;
            const window = rollingWindow.current;
            window.push(msg.reward);
            if (window.length > ROLLING_WINDOW) window.shift();
            const mean = window.reduce((a, b) => a + b, 0) / window.length;
            pendingEpisodes.current.push({ ...msg, rollingMean: Math.round(mean * 10) / 10 });
            break;
          }
          case "ppo_update":
            if (sid && msg.scenario_id !== sid) return;
            persistedPpo.current = msg;
            pendingPpo.current.push(msg);
            break;
          case "status": {
            const savedRecord = ppoRecordFromStatus(msg);
            persistedPpo.current = savedRecord;
            if (savedRecord) {
              setPpo((previous) => previous.some((record) => (
                record.scenario_id === savedRecord.scenario_id
                && record.update === savedRecord.update
              )) ? previous : previous.concat(savedRecord).slice(-MAX_PPO_POINTS));
            }
            setStatus(msg);
            if (msg.last_error) setLastError(`Training stopped: ${msg.last_error}`);
            setGhostEpisode(msg.ghost_episode);
            setScenarioKind(msg.scenario_kind);
            setMetricLabel(msg.metric_label);
            setMetricMode(msg.metric_mode);
            if (scenarioRef.current !== msg.scenario_id) {
              scenarioRef.current = msg.scenario_id;
              setScenarioId(msg.scenario_id);
            }
            break;
          }
          case "scenario_changed":
            scenarioRef.current = msg.id;
            setScenarioId(msg.id);
            setScenarioKind(msg.kind);
            setMetricLabel(msg.metric_label);
            setMetricMode(msg.metric_mode);
            clearScenarioState();
            break;
          case "history": {
            if (sid && msg.scenario_id !== sid) return;
            // History is an authoritative branch sync (connect, reset, load,
            // or restore). Discard locally batched data from the old branch.
            pendingEpisodes.current = [];
            pendingPpo.current = [];
            frameRef.current = null;
            terminalFrameRef.current = null;
            setPpo(persistedPpo.current?.scenario_id === msg.scenario_id
              ? [persistedPpo.current] : []);
            const window: number[] = [];
            const withMeans = msg.history.map((h) => {
              window.push(h.reward);
              if (window.length > ROLLING_WINDOW) window.shift();
              const mean = window.reduce((a, b) => a + b, 0) / window.length;
              return { ...h, rollingMean: Math.round(mean * 10) / 10 };
            });
            rollingWindow.current = window;
            setHistory(withMeans);
            break;
          }
          case "checkpoint_list":
            if (sid && msg.scenario_id !== sid) return;
            setCheckpoints(msg.checkpoints.slice().reverse());
            break;
          case "ghost_lap":
            if (msg.scenario_id && msg.scenario_id !== sid) break;
            ghostRef.current = { lap: msg, startedAt: performance.now() };
            setGhostEpisode(msg.episode);
            setReplayRevision((value) => value + 1);
            break;
          case "ghost_clear":
            if (msg.scenario_id && msg.scenario_id !== sid) break;
            ghostRef.current = null;
            setGhostEpisode(null);
            break;
          case "error":
            setLastError(msg.message);
            break;
        }
      };
    };
    connect();

    const flush = setInterval(() => {
      if (pendingEpisodes.current.length > 0) {
        const batch = pendingEpisodes.current;
        pendingEpisodes.current = [];
        setHistory((prev) => {
          let next = prev.concat(batch);
          while (next.length > MAX_HISTORY_POINTS) next = decimateHalf(next);
          return next;
        });
      }
      if (pendingPpo.current.length > 0) {
        const batch = pendingPpo.current;
        pendingPpo.current = [];
        setPpo((prev) => prev.concat(batch).slice(-MAX_PPO_POINTS));
      }
    }, FLUSH_MS);

    return () => {
      closed = true;
      clearTimeout(retryTimer);
      clearInterval(flush);
      wsRef.current?.close();
    };
  }, [clearScenarioState]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/scenarios", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`catalog request failed (${response.status})`);
        return response.json() as Promise<{ scenarios: ScenarioInfo[] }>;
      })
      .then((data) => setScenarios(data.scenarios))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLastError("The experiment library could not be loaded. Check the backend connection.");
      });
    fetch("/api/runs", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`runs request failed (${response.status})`);
        return response.json() as Promise<{ archives: ArchivedRun[] }>;
      })
      .then((data) => setArchivedRuns(data.archives))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
      });
    return () => controller.abort();
  }, [scenarioId, checkpoints.length]);

  const currentScenario = useMemo(
    () => scenarios.find((scenario) => scenario.id === scenarioId) ?? null,
    [scenarios, scenarioId],
  );

  const value = useMemo<TrainingSocketValue>(() => ({
    readOnly: false, scene: null, publicQualification: null,
    connected, connectionState, status, scenarios, currentScenario,
    scenarioId, scenarioKind, metricLabel, metricMode,
    history, ppo, checkpoints, archivedRuns, ghostEpisode, replayRevision, lastError,
    frameRef, terminalFrameRef, ghostRef,
    startTraining: (maxEpisodes, checkpointEveryN, pauseOnSuccess = true) =>
      send({ type: "start_training", max_episodes: maxEpisodes, checkpoint_every_n: checkpointEveryN, pause_on_success: pauseOnSuccess }),
    stopTraining: () => send({ type: "stop_training" }),
    resetTraining: (seed) => {
      pendingEpisodes.current = [];
      pendingPpo.current = [];
      rollingWindow.current = [];
      frameRef.current = null;
      terminalFrameRef.current = null;
      setHistory([]);
      setPpo([]);
      send({ type: "reset_training", seed });
    },
    setGhost: (episode) => send({ type: "set_ghost", episode }),
    setArchiveGhost: (id, episode) => {
      if (scenarioId) send({ type: "set_archive_ghost", archive_id: id, episode, scenario_id: scenarioId });
    },
    restartReplay: () => {
      const ghost = ghostRef.current;
      if (ghost) ghostRef.current = { ...ghost, startedAt: performance.now() };
    },
    clearGhost: () => send({ type: "clear_ghost" }),
    loadCheckpoint: (episode) => {
      frameRef.current = null;
      terminalFrameRef.current = null;
      send({ type: "load_checkpoint", episode });
    },
    restoreArchivedRun: async (id) => {
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(id)}/restore`, {
          method: "POST",
        });
        if (!response.ok) {
          setLastError(`The archived run could not be restored (${response.status}).`);
          return;
        }
        const runs = await fetch("/api/runs");
        if (runs.ok) {
          const data = await runs.json() as { archives: ArchivedRun[] };
          setArchivedRuns(data.archives);
        }
        setLastError(null);
      } catch {
        setLastError("The archived run could not be restored because the trainer is offline.");
      }
    },
    selectScenario: async (id) => {
      try {
        const res = await fetch("/api/scenario", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
        });
        if (!res.ok) {
          setLastError(`The experiment could not be opened (${res.status}). Try again.`);
          return false;
        } else {
          setLastError(null);
          return true;
        }
      } catch {
        setLastError("The experiment could not be opened because the trainer is offline.");
        return false;
      }
    },
    clearError: () => setLastError(null),
  }), [connected, connectionState, status, scenarios, currentScenario,
       scenarioId, scenarioKind, metricLabel, metricMode,
       history, ppo, checkpoints, archivedRuns, ghostEpisode, replayRevision, lastError, send]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

function publicStatus(scenario: ScenarioInfo, policy: PublicDemoPolicy, engineHash: string): StatusMsg {
  return {
    type: "status",
    scenario_id: scenario.id,
    scenario_kind: scenario.kind,
    metric_label: scenario.metric_label,
    metric_mode: scenario.metric_mode,
    training: false,
    episode: policy.origin_episode,
    max_episodes: policy.origin_episode,
    run_start_episode: policy.origin_episode,
    run_target_episode: policy.origin_episode,
    checkpoint_every_n: 0,
    total_steps: policy.evaluation.total_steps ?? 0,
    update_count: policy.evaluation.update_count ?? 0,
    sps: 0,
    seed: policy.seed,
    eval_episodes: policy.evaluation.eval_episodes,
    evaluation_suite: policy.evaluation.evaluation_suite,
    evaluation_seed_base: 5_200_000,
    engine_source_sha256: engineHash,
    best_reward: policy.canonical_summary.reward,
    best_metric: policy.canonical_summary.metric,
    device: "recorded",
    ghost_episode: policy.origin_episode,
    ppo_diagnostics: null,
  };
}

function PublicDemoProvider({ children }: { children: React.ReactNode }) {
  const [demo, setDemo] = useState<PublicDemoData | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [ghostEpisode, setGhostEpisode] = useState<number | null>(null);
  const [replayRevision, setReplayRevision] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const frameRef = useRef<FrameMsg | null>(null);
  const terminalFrameRef = useRef<HeldTerminalFrame | null>(null);
  const ghostRef = useRef<{ lap: GhostLap; startedAt: number } | null>(null);

  const activatePolicy = useCallback((scenario: string, policy: PublicDemoPolicy) => {
    ghostRef.current = {
      lap: {
        scenario_id: scenario,
        archive_id: policy.id,
        episode: policy.origin_episode,
        dt: policy.dt,
        trajectory: policy.trajectory,
        frames: policy.frames,
      },
      startedAt: performance.now(),
    };
    setGhostEpisode(policy.origin_episode);
    setReplayRevision((revision) => revision + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/demo.json", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`demo request failed (${response.status})`);
        return response.json() as Promise<PublicDemoData>;
      })
      .then((data) => {
        const preferred = data.scenarios.find((scenario) => scenario.id === "lunar-lander")
          ?? data.scenarios[0];
        if (!preferred || data.qualification.policies !== 46) {
          throw new Error("The public replay bundle is incomplete.");
        }
        setDemo(data);
        setScenarioId(preferred.id);
        activatePolicy(preferred.id, data.policies[preferred.id][0]);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLastError("The verified replay library could not be loaded. Refresh to try again.");
      });
    return () => controller.abort();
  }, [activatePolicy]);

  const currentScenario = useMemo(
    () => demo?.scenarios.find((scenario) => scenario.id === scenarioId) ?? null,
    [demo, scenarioId],
  );
  const policies = useMemo(
    () => scenarioId ? demo?.policies[scenarioId] ?? [] : [],
    [demo, scenarioId],
  );
  const status = useMemo(
    () => currentScenario && policies[0] && demo
      ? publicStatus(currentScenario, policies[0], demo.engine_source_sha256) : null,
    [currentScenario, demo, policies],
  );
  const history = useMemo<EpisodeRecord[]>(() => policies.map((policy, index) => ({
    ...policy.canonical_summary,
    episode: index + 1,
    rollingMean: policy.canonical_summary.reward,
  })), [policies]);
  const archivedRuns = useMemo<ArchivedRun[]>(() => policies.map((policy) => ({
    id: policy.id,
    latest_episode: policy.origin_episode,
    checkpoints: 1,
    seed: policy.seed,
    timestamp: policy.timestamp,
    schema_version: policy.evaluation.schema_version,
    compatible: true,
    requalification: {
      holdout_successes: policy.holdout_successes,
      holdout_episodes: policy.holdout_episodes,
      origin: { seed: policy.seed, episode: policy.origin_episode },
    },
  })), [policies]);
  const readOnlyError = useCallback(() => {
    setLastError("This Cloudflare showcase is replay-only. Clone the GitHub repository to train policies locally.");
    return false;
  }, []);

  const value = useMemo<TrainingSocketValue>(() => ({
    readOnly: true,
    scene: scenarioId ? demo?.scenes[scenarioId] ?? null : null,
    publicQualification: demo?.qualification ?? null,
    connected: Boolean(demo),
    connectionState: demo ? "connected" : "connecting",
    status,
    scenarios: demo?.scenarios ?? [],
    currentScenario,
    scenarioId,
    scenarioKind: currentScenario?.kind ?? "generic",
    metricLabel: currentScenario?.metric_label ?? "score",
    metricMode: currentScenario?.metric_mode ?? "max",
    history,
    ppo: [],
    checkpoints: [],
    archivedRuns,
    ghostEpisode,
    replayRevision,
    lastError,
    frameRef,
    terminalFrameRef,
    ghostRef,
    startTraining: readOnlyError,
    stopTraining: () => { readOnlyError(); },
    resetTraining: () => { readOnlyError(); },
    setGhost: () => {
      if (scenarioId && policies[0]) activatePolicy(scenarioId, policies[0]);
    },
    setArchiveGhost: (id) => {
      const policy = policies.find((item) => item.id === id);
      if (scenarioId && policy) activatePolicy(scenarioId, policy);
    },
    restartReplay: () => {
      if (ghostRef.current) ghostRef.current = {
        ...ghostRef.current, startedAt: performance.now(),
      };
      setReplayRevision((revision) => revision + 1);
    },
    clearGhost: () => {
      ghostRef.current = null;
      setGhostEpisode(null);
    },
    loadCheckpoint: () => { readOnlyError(); },
    restoreArchivedRun: async () => { readOnlyError(); },
    selectScenario: async (id) => {
      const scenario = demo?.scenarios.find((item) => item.id === id);
      const policy = demo?.policies[id]?.[0];
      if (!scenario || !policy) return false;
      setScenarioId(id);
      setLastError(null);
      activatePolicy(id, policy);
      return true;
    },
    clearError: () => setLastError(null),
  }), [activatePolicy, archivedRuns, currentScenario, demo, ghostEpisode, history,
    lastError, policies, readOnlyError, replayRevision, scenarioId, status]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function TrainingSocketProvider({ children }: { children: React.ReactNode }) {
  return import.meta.env.VITE_PUBLIC_DEMO === "1"
    ? <PublicDemoProvider>{children}</PublicDemoProvider>
    : <LiveTrainingSocketProvider>{children}</LiveTrainingSocketProvider>;
}

export function useTrainingSocket(): TrainingSocketValue {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTrainingSocket outside provider");
  return ctx;
}
