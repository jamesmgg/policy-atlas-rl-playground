import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import { ppoRecordFromStatus, recordTerminalFrame } from "../api/types";
import type {
  ArchivedRun, CheckpointMeta, ClientMessage, EpisodeRecord, FrameMsg, GhostLap,
  HeldTerminalFrame, PpoUpdateRecord, ScenarioInfo, ServerMessage, StatusMsg,
} from "../api/types";

const FLUSH_MS = 400;
const MAX_HISTORY_POINTS = 2400;
const MAX_PPO_POINTS = 600;
const ROLLING_WINDOW = 20;

export interface TrainingSocketValue {
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
  lastError: string | null;
  /** Latest live frame — read inside rAF loops, never triggers re-renders. */
  frameRef: React.RefObject<FrameMsg | null>;
  terminalFrameRef: React.RefObject<HeldTerminalFrame | null>;
  /** Active ghost lap trajectory, with the time it was activated. */
  ghostRef: React.RefObject<{ lap: GhostLap; startedAt: number } | null>;
  startTraining(maxEpisodes: number, checkpointEveryN: number): boolean;
  stopTraining(): void;
  resetTraining(seed: number): void;
  setGhost(episode: number): void;
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

export function TrainingSocketProvider({ children }: { children: React.ReactNode }) {
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
            ghostRef.current = { lap: msg, startedAt: performance.now() };
            setGhostEpisode(msg.episode);
            break;
          case "ghost_clear":
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
    connected, connectionState, status, scenarios, currentScenario,
    scenarioId, scenarioKind, metricLabel, metricMode,
    history, ppo, checkpoints, archivedRuns, ghostEpisode, lastError,
    frameRef, terminalFrameRef, ghostRef,
    startTraining: (maxEpisodes, checkpointEveryN) =>
      send({ type: "start_training", max_episodes: maxEpisodes, checkpoint_every_n: checkpointEveryN }),
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
       history, ppo, checkpoints, archivedRuns, ghostEpisode, lastError, send]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTrainingSocket(): TrainingSocketValue {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTrainingSocket outside provider");
  return ctx;
}
