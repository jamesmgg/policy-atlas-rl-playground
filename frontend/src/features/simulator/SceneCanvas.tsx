import { useEffect, useRef, useState } from "react";
import {
  displayedEpisodeNumber, formatEpisodeDuration, formatSimulationRate, formatTerminationCause,
  selectVisibleFrame, replayPosition,
} from "../../api/types";
import type { FrameMsg, GenericObject, SceneData, StaticPrimitive, TrackGeometry, ZoneInfo } from "../../api/types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

const W = 1000;
const H = 700;

const COLORS = {
  asphalt: "#182630",
  edge: "rgba(255,255,255,0.85)",
  curbRed: "#d33a2c",
  checkpoint: "rgba(77,166,255,0.25)",
  startLine: "#ffffff",
  car: "#f0a34a",
  carGlow: "rgba(240,163,74,0.55)",
  bot: "#8a90a0",
  ghost: "#62a4ff",
  trail: "240,163,74",
  ghostTrail: "98,164,255",
  skid: "rgba(10,10,12,0.5)",
  terrain: "#3a3f4b",
  flame: "#ffb347",
};

interface TrailPoint { x: number; y: number; drift: number }
interface LapBanner { text: string; pb: boolean; until: number }
interface ReferenceReplay { scenario_id: string; dt: number; frames: FrameMsg[] }

function drawCar(
  ctx: CanvasRenderingContext2D, x: number, y: number, heading: number,
  color: string, alpha: number, drift: number, glowColor: string,
) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);
  ctx.globalAlpha = alpha;
  if (drift > 0.35) {
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 10 + drift * 14;
  }
  ctx.fillStyle = color;
  ctx.fillRect(-5.5, -2.2, 9, 4.4);
  ctx.beginPath();
  ctx.moveTo(3.5, -1.4);
  ctx.lineTo(7.5, 0);
  ctx.lineTo(3.5, 1.4);
  ctx.closePath();
  ctx.fill();
  ctx.fillRect(-7, -3.2, 1.8, 6.4);
  ctx.fillRect(5.6, -2.8, 1.2, 5.6);
  ctx.restore();
}

function drawGenericObject(
  ctx: CanvasRenderingContext2D, obj: GenericObject, alpha = 1,
) {
  ctx.save();
  ctx.globalAlpha = alpha;
  switch (obj.shape) {
    case "station": {
      ctx.translate(obj.x, obj.y);
      ctx.fillStyle = "#547aab";
      ctx.fillRect(-42, -24, 22, 48); ctx.fillRect(20, -24, 22, 48);
      ctx.strokeStyle = "#9cbde3"; ctx.lineWidth = 1;
      for (let y = -16; y <= 16; y += 8) {
        ctx.beginPath(); ctx.moveTo(-42, y); ctx.lineTo(42, y); ctx.stroke();
      }
      ctx.fillStyle = "#ccd8dc"; ctx.fillRect(-12, -18, 24, 36);
      ctx.strokeStyle = "#78b9ad"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(0, 0, 7, 0, Math.PI * 2); ctx.stroke();
      break;
    }
    case "spacecraft": {
      ctx.translate(obj.x, obj.y); ctx.rotate(obj.rot ?? 0);
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.beginPath(); ctx.moveTo(15, 0); ctx.lineTo(-9, -9); ctx.lineTo(-6, 0);
      ctx.lineTo(-9, 9); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#dcebf6"; ctx.fillRect(-5, -19, 5, 10); ctx.fillRect(-5, 9, 5, 10);
      break;
    }
    case "robotarm": {
      const q1 = obj.rot ?? 0, q2 = obj.joint2 ?? 0;
      const ex = 450 + 160 * Math.cos(q1), ey = 390 - 160 * Math.sin(q1);
      const tx = ex + 120 * Math.cos(q1 + q2), ty = ey - 120 * Math.sin(q1 + q2);
      ctx.lineCap = "round"; ctx.lineWidth = 17;
      ctx.strokeStyle = alpha < 1 ? COLORS.ghost : "#839bab";
      ctx.beginPath(); ctx.moveTo(450, 390); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.strokeStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(tx, ty); ctx.stroke();
      for (const [x, y, r] of [[450, 390, 18], [ex, ey, 12], [tx, ty, 8]]) {
        ctx.fillStyle = "#dce9e8"; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#203944"; ctx.beginPath(); ctx.arc(x, y, r / 2, 0, Math.PI * 2); ctx.fill();
      }
      break;
    }
    case "ballbeam": {
      ctx.save(); ctx.translate(500, 420); ctx.rotate(obj.rot ?? 0);
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : "#8ca5b3";
      ctx.fillRect(-330, 0, 660, 8);
      ctx.strokeStyle = "#38505c"; ctx.lineWidth = 1;
      for (let x = -300; x <= 300; x += 30) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 8); ctx.stroke();
      }
      ctx.restore();
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.beginPath(); ctx.arc(obj.x, obj.y, 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#ffe0ad";
      ctx.beginPath(); ctx.arc(obj.x - 3, obj.y - 4, 3, 0, Math.PI * 2); ctx.fill();
      break;
    }
    case "lander": {
      const rot = obj.rot ?? 0;
      // Physics reports the contact point, so keep the lowest sprite outline
      // at that point instead of drawing the landing feet below the surface.
      const c = Math.cos(rot), s = Math.abs(Math.sin(rot));
      const contactDepth = Math.max(12 * c + 11 * s, 6 * c + 8 * s, -8 * c + 5 * s) + 1;
      ctx.translate(obj.x, obj.y - contactDepth);
      ctx.rotate(rot);
      if ((obj.flame ?? 0) > 0.05) {
        ctx.fillStyle = COLORS.flame;
        const f = 8 + (obj.flame ?? 0) * 16;
        ctx.beginPath();
        ctx.moveTo(-4, 9);
        ctx.lineTo(0, 9 + f);
        ctx.lineTo(4, 9);
        ctx.closePath();
        ctx.fill();
      }
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : "#d8dbe2";
      ctx.beginPath();                       // capsule body
      ctx.moveTo(-8, 6);
      ctx.lineTo(-5, -8);
      ctx.lineTo(5, -8);
      ctx.lineTo(8, 6);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = alpha < 1 ? COLORS.ghost : "#9aa0ad";
      ctx.lineWidth = 2;
      ctx.beginPath();                       // legs
      ctx.moveTo(-7, 6); ctx.lineTo(-11, 12);
      ctx.moveTo(7, 6); ctx.lineTo(11, 12);
      ctx.stroke();
      break;
    }
    case "paddle": {
      const w = obj.width ?? 153, h = obj.height ?? 16;
      ctx.fillStyle = "#8fe2cd";
      ctx.beginPath(); ctx.roundRect(obj.x-w/2, obj.y-h/2, w, h, 7); ctx.fill();
      ctx.fillStyle = "#d1fff2"; ctx.fillRect(obj.x-w/2+12, obj.y-h/2+2, w-24, 3);
      break;
    }
    case "ball": {
      ctx.fillStyle = "#ffe397";
      ctx.beginPath(); ctx.arc(obj.x, obj.y, obj.radius ?? 13, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = "#fff5d4";
      ctx.beginPath(); ctx.arc(obj.x-4, obj.y-4, 4, 0, Math.PI*2); ctx.fill();
      break;
    }
    case "bird": {
      // Rotate unit-circle artwork inside the mapped 0.045 m collision radius.
      // Scaling before rotation keeps every tilt inside the same world ellipse.
      ctx.translate(obj.x, obj.y); ctx.scale(0.045 * 350, 0.045 * 285);
      ctx.rotate(obj.rot ?? 0);
      ctx.fillStyle = "#ffd275";
      ctx.beginPath(); ctx.moveTo(-0.96, 0.12); ctx.lineTo(-0.55, -0.26); ctx.lineTo(-0.55, 0.36); ctx.fill();
      ctx.beginPath(); ctx.ellipse(-0.1, 0, 0.68, 0.72, 0, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = "#d79a4d";
      ctx.beginPath(); ctx.ellipse(-0.22, 0.08, 0.35, 0.28, -0.4, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = "#ed9960";
      ctx.beginPath(); ctx.moveTo(0.48, -0.12); ctx.lineTo(0.97, 0.03); ctx.lineTo(0.5, 0.22); ctx.fill();
      ctx.fillStyle = "#122331";
      ctx.beginPath(); ctx.arc(0.32, -0.24, 0.115, 0, Math.PI*2); ctx.fill();
      break;
    }
    case "pipe": {
      const w = obj.width ?? 63, h = obj.height ?? 100;
      ctx.fillStyle = "#246b62"; ctx.fillRect(obj.x-w/2, obj.y-h/2, w, h);
      ctx.fillStyle = "#68b8a1"; ctx.fillRect(obj.x-w/2+5, obj.y-h/2, 7, h);
      ctx.strokeStyle = "#92d6ba"; ctx.lineWidth = 2; ctx.strokeRect(obj.x-w/2, obj.y-h/2, w, h);
      break;
    }
    case "collector": {
      // Include the rounded outline within the 0.045 m × 300 px/m footprint.
      ctx.translate(obj.x, obj.y); ctx.scale(0.045 * 300, 0.045 * 300);
      ctx.rotate(obj.rot ?? 0);
      ctx.fillStyle = "#8fd5f2";
      ctx.beginPath(); ctx.moveTo(0.9, 0); ctx.lineTo(-0.6, -0.65); ctx.lineTo(-0.28, 0); ctx.lineTo(-0.6, 0.65); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "#d5f3ff"; ctx.lineWidth = 0.09; ctx.lineJoin = "round"; ctx.stroke();
      break;
    }
    case "coin": {
      const r = obj.radius ?? 16;
      ctx.fillStyle = "#eab85a"; ctx.beginPath(); ctx.arc(obj.x, obj.y, r, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = "#fff0b4"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(obj.x, obj.y, r-4, 0, Math.PI*2); ctx.stroke();
      ctx.fillStyle = "#865f27"; ctx.fillRect(obj.x-2, obj.y-7, 4, 14);
      break;
    }
    case "rod": {
      const len = obj.len ?? 180;
      const rot = obj.rot ?? 0;
      // theta=0 is upright; y-down world → tip = pivot + len*(sin t, -cos t)
      const tx = obj.x + len * Math.sin(rot);
      const ty = obj.y - len * Math.cos(rot);
      ctx.strokeStyle = alpha < 1 ? COLORS.ghost : "#d8dbe2";
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(obj.x, obj.y);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.beginPath();
      ctx.arc(tx, ty, 11, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "drone": {
      ctx.translate(obj.x, obj.y);
      ctx.rotate(obj.rot ?? 0);
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : "#d8dbe2";
      ctx.fillRect(-14, -2, 28, 4);          // crossbar
      ctx.fillRect(-3, -6, 6, 6);            // hub
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.fillRect(-16, -6, 6, 3);           // rotors
      ctx.fillRect(10, -6, 6, 3);
      break;
    }
    case "cartpole": {
      const len = obj.len ?? 150;
      const rot = obj.rot ?? 0;
      ctx.translate(obj.x, obj.y);
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : "#e6eeeb";
      ctx.fillRect(-30, -18, 60, 18);
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : "#6f817c";
      ctx.beginPath(); ctx.arc(-19, 3, 8, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(19, 3, 8, 0, Math.PI * 2); ctx.fill();
      const pivotY = -18;
      const tx = len * Math.sin(rot);
      const ty = pivotY - len * Math.cos(rot);
      ctx.strokeStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.lineWidth = 7;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(0, pivotY); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.fillStyle = "#f7faf8";
      ctx.beginPath(); ctx.arc(0, pivotY, 7, 0, Math.PI * 2); ctx.fill();
      break;
    }
    case "mountain-car": {
      ctx.translate(obj.x, obj.y - 12);
      ctx.rotate(obj.rot ?? 0);
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : COLORS.car;
      ctx.beginPath();
      ctx.roundRect(-20, -13, 40, 18, 6);
      ctx.fill();
      ctx.fillStyle = alpha < 1 ? COLORS.ghost : "#101b26";
      ctx.beginPath(); ctx.arc(-13, 7, 7, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(13, 7, 7, 0, Math.PI * 2); ctx.fill();
      break;
    }
    case "target": {
      ctx.strokeStyle = COLORS.ghost;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(obj.x, obj.y, 14, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      break;
    }
  }
  ctx.restore();
}

export default function SceneCanvas() {
  const {
    frameRef, terminalFrameRef, ghostRef, ghostEpisode, replayRevision, status, ppo,
    scenarioId, currentScenario, restartReplay, clearGhost,
  } = useTrainingSocket();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const expandButtonRef = useRef<HTMLButtonElement>(null);
  const exitButtonRef = useRef<HTMLButtonElement>(null);
  const fullscreenSessionRef = useRef(false);
  const [scene, setScene] = useState<SceneData | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [sceneAttempt, setSceneAttempt] = useState(0);
  const [telemetry, setTelemetry] = useState<FrameMsg | null>(null);
  const [playback, setPlayback] = useState<ReturnType<typeof replayPosition>>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [fullscreenAvailable, setFullscreenAvailable] = useState(false);
  const [fullscreenError, setFullscreenError] = useState<string | null>(null);
  const referenceRef = useRef<{ replay: ReferenceReplay; startedAt: number } | null>(null);
  const referenceRequest = useRef(0);
  const [referenceState, setReferenceState] = useState<"idle" | "loading" | "playing">("idle");
  const [referenceError, setReferenceError] = useState<string | null>(null);
  const viewingReplay = ghostEpisode != null && referenceState !== "playing";
  const replayOnly = viewingReplay;

  useEffect(() => {
    referenceRequest.current += 1;
    referenceRef.current = null;
    setReferenceState("idle");
    setReferenceError(null);
  }, [scenarioId, status?.training, replayRevision]);

  const referenceFrame = (now: number) => {
    const ref = referenceRef.current;
    if (!ref || ref.replay.scenario_id !== scenarioId) return null;
    const speed = scenarioId === "orbital-docking" ? 8 : 1;
    const index = Math.min(ref.replay.frames.length - 1,
      Math.floor((now - ref.startedAt) / 1000 * speed / ref.replay.dt));
    return ref.replay.frames[index] ?? null;
  };

  const toggleReference = async () => {
    const request = ++referenceRequest.current;
    if (referenceRef.current) {
      referenceRef.current = null;
      setReferenceState("idle");
      return;
    }
    setReferenceState("loading"); setReferenceError(null);
    try {
      const response = await fetch(`/api/reference/${encodeURIComponent(scenarioId ?? "")}`);
      if (!response.ok) throw new Error("Reference demonstration could not load.");
      const replay: ReferenceReplay = await response.json();
      if (request !== referenceRequest.current) return;
      referenceRef.current = { replay, startedAt: performance.now() };
      setReferenceState("playing");
    } catch (error) {
      if (request !== referenceRequest.current) return;
      setReferenceError(error instanceof Error ? error.message : "Reference demonstration failed.");
      setReferenceState("idle");
    }
  };

  useEffect(() => {
    setFullscreenAvailable(
      document.fullscreenEnabled && typeof stageRef.current?.requestFullscreen === "function",
    );
    const update = () => {
      setIsFullscreen(document.fullscreenElement === stageRef.current);
      setFullscreenError(null);
    };
    const reportError = () => setFullscreenError(
      "Fullscreen could not start. Try the browser's own full-screen control.",
    );
    document.addEventListener("fullscreenchange", update);
    document.addEventListener("fullscreenerror", reportError);
    return () => {
      document.removeEventListener("fullscreenchange", update);
      document.removeEventListener("fullscreenerror", reportError);
    };
  }, []);

  useEffect(() => {
    if (isFullscreen) {
      fullscreenSessionRef.current = true;
      requestAnimationFrame(() => exitButtonRef.current?.focus());
    } else if (fullscreenSessionRef.current) {
      fullscreenSessionRef.current = false;
      requestAnimationFrame(() => expandButtonRef.current?.focus());
    }
  }, [isFullscreen]);

  useEffect(() => {
    const update = () => {
      const now = performance.now();
      const ghost = referenceRef.current ? null : ghostRef.current;
      const position = ghost ? replayPosition(ghost, now) : null;
      setPlayback(position);
      // A saved rollout must never inherit a training episode's outcome or HUD.
      setTelemetry(ghost ? ghost.lap.frames?.[position?.index ?? 0] ?? null : referenceFrame(now) ?? selectVisibleFrame(
        frameRef.current, terminalFrameRef.current, now, undefined,
        status?.training === false,
      ));
    };
    update();
    const timer = window.setInterval(update, 250);
    return () => window.clearInterval(timer);
  }, [frameRef, terminalFrameRef, ghostRef, ghostEpisode, scenarioId, status?.training, referenceState]);

  useEffect(() => {
    if (!scenarioId) return;
    let cancelled = false;
    setScene(null);
    setSceneError(null);
    fetch("/api/scene")
      .then((r) => {
        if (!r.ok) throw new Error(`scene request failed (${r.status})`);
        return r.json();
      })
      .then((s: SceneData) => {
        if (!cancelled && s.scenario_id === scenarioId) setScene(s);
      })
      .catch(() => {
        if (!cancelled) setSceneError("The environment view could not be loaded.");
      });
    return () => { cancelled = true; };
  }, [scenarioId, sceneAttempt]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !scene) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d")!;

    const staticLayer = document.createElement("canvas");
    staticLayer.width = W * dpr;
    staticLayer.height = H * dpr;
    const tctx = staticLayer.getContext("2d")!;
    tctx.scale(dpr, dpr);
    if (scene.kind === "track") renderTrack(tctx, scene.track, scene.zones);
    else renderStatics(tctx, scene.statics);

    const skidLayer = document.createElement("canvas");
    skidLayer.width = W * dpr;
    skidLayer.height = H * dpr;
    const sctx = skidLayer.getContext("2d")!;
    sctx.scale(dpr, dpr);

    const liveTrail: TrailPoint[] = [];
    const ghostTrail: TrailPoint[] = [];
    let lastSkidFade = performance.now();
    let banner: LapBanner | null = null;
    let prevLastLap: number | null | undefined;
    let previousEpisode: number | null = null;
    let previousGhostStart: number | null = null;
    let raf = 0;
    let timer = 0;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const draw = () => {
      // Mobile keeps screens mounted to preserve replay state. Avoid drawing
      // a hidden simulator while the user browses projects, settings, or results.
      if (document.hidden || canvas.clientWidth === 0) {
        timer = window.setTimeout(draw, 200);
        return;
      }
      const now = performance.now();

      if (now - lastSkidFade > 2000) {
        lastSkidFade = now;
        sctx.save();
        sctx.globalCompositeOperation = "destination-out";
        sctx.globalAlpha = 0.08;
        sctx.fillRect(0, 0, W, H);
        sctx.restore();
      }

      const ghost = referenceRef.current ? null : ghostRef.current;
      const replayOnly = !!ghost;
      const position = ghost ? replayPosition(ghost, now) : null;
      const recorded = ghost && position ? ghost.lap.frames?.[position.index] : null;
      const frame = replayOnly ? recorded ?? null : referenceFrame(now) ?? selectVisibleFrame(
        frameRef.current, terminalFrameRef.current, now, undefined,
        status?.training === false,
      );
      if (frame && previousEpisode != null && frame.episode !== previousEpisode) {
        liveTrail.length = 0;
        sctx.clearRect(0, 0, W, H);
        lastSkidFade = now;
      }
      if (frame) previousEpisode = frame.episode;
      const primary = frame?.car
        ? { x: frame.car.x, y: frame.car.y, drift: frame.car.drift }
        : frame?.objects?.length
          ? { x: frame.objects[0].x, y: frame.objects[0].y, drift: 0 }
          : null;
      if (primary && !reduceMotion) {
        const last = liveTrail[liveTrail.length - 1];
        if (!last || (last.x - primary.x) ** 2 + (last.y - primary.y) ** 2 > 4) {
          liveTrail.push(primary);
          if (liveTrail.length > 110) liveTrail.shift();
          if (primary.drift > 0.4) {
            sctx.fillStyle = COLORS.skid;
            sctx.beginPath();
            sctx.arc(primary.x, primary.y, 1.6, 0, Math.PI * 2);
            sctx.fill();
          }
        }
      }

      // Lap banner trigger: last_lap changed.
      if (frame?.last_lap != null && frame.last_lap !== prevLastLap) {
        if (prevLastLap !== undefined) {
          banner = {
            text: `LAP ${frame.laps} — ${frame.last_lap.toFixed(2)}s`,
            pb: frame.last_lap === frame.best_lap,
            until: now + 2400,
          };
        }
        prevLastLap = frame.last_lap;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(staticLayer, 0, 0, W, H);
      if (scene.kind === "track") ctx.drawImage(skidLayer, 0, 0, W, H);

      // Saved rollouts play once and hold their terminal position.
      if (ghost && position && !(replayOnly && recorded)) {
        const { lap, startedAt } = ghost;
        if (previousGhostStart !== startedAt) ghostTrail.length = 0;
        previousGhostStart = startedAt;
        const [gx, gy, grot, gd] = lap.trajectory[position.index];
        const gLast = ghostTrail[ghostTrail.length - 1];
        if (!gLast || (gLast.x - gx) ** 2 + (gLast.y - gy) ** 2 > 4) {
          ghostTrail.push({ x: gx, y: gy, drift: gd });
          if (ghostTrail.length > 80) ghostTrail.shift();
        }
        if (scene.kind === "track") {
          drawTrail(ctx, ghostTrail, COLORS.ghostTrail);
          drawCar(ctx, gx, gy, grot, COLORS.ghost, replayOnly ? 1 : 0.55, gd, COLORS.ghost);
        } else {
          drawTrail(ctx, ghostTrail, COLORS.ghostTrail);
          drawGenericObject(ctx, {
            shape: scene.primary_shape as GenericObject["shape"],
            x: gx, y: gy, rot: grot, joint2: gd,
          }, replayOnly ? 1 : 0.5);
        }
      } else {
        ghostTrail.length = 0;
      }

      if (!replayOnly) drawTrail(ctx, liveTrail, COLORS.trail);
      if (frame) {
        if (frame.bots) {
          for (const b of frame.bots) {
            drawCar(ctx, b.x, b.y, b.heading, COLORS.bot, 0.9, 0, COLORS.bot);
          }
        }
        if (frame.car) {
          drawCar(ctx, frame.car.x, frame.car.y, frame.car.heading,
                  COLORS.car, 1, frame.car.drift, COLORS.carGlow);
        }
        for (const obj of frame.objects ?? []) {
          drawGenericObject(ctx, frame.terminal && obj.shape === "lander" ? { ...obj, flame: 0 } : obj);
        }
        if (!replayOnly) drawHud(ctx, frame, null);
        if (banner && now < banner.until) drawBanner(ctx, banner, now);
      }
      if (reduceMotion) timer = window.setTimeout(draw, 200);
      else raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(timer);
    };
  }, [scene, frameRef, terminalFrameRef, ghostRef, status?.training, referenceState]);

  const enterFullscreen = async () => {
    setFullscreenError(null);
    if (!stageRef.current?.requestFullscreen) {
      setFullscreenError("Fullscreen is not available in this browser.");
      return;
    }
    try {
      await stageRef.current.requestFullscreen();
    } catch {
      setFullscreenError("Fullscreen could not start. Try the browser's own full-screen control.");
    }
  };

  const exitFullscreen = async () => {
    try {
      await document.exitFullscreen?.();
    } catch {
      setFullscreenError("Fullscreen could not close. Press Escape to return.");
    }
  };

  const stepsPerSecond = ppo.at(-1)?.sps ?? 0;

  return (
    <section className="simulator-panel" aria-labelledby="simulator-title">
      <header className="simulator-toolbar">
        <div>
          <span className="section-kicker">Live environment</span>
          <h2 id="simulator-title">{referenceState === "playing" ? "Reference demonstration" : replayOnly ? "Saved policy replay" : "Policy rollout"}</h2>
        </div>
        <div className="simulator-legend" aria-label="Simulator legend">
          <span><i className="legend-agent" />{referenceState === "playing" ? "Reference" : "Agent"}</span>
          <span><i className="legend-replay" />Checkpoint replay</span>
        </div>
        {currentScenario?.reference_controller && <button type="button" className="fullscreen-button"
          disabled={status?.training || referenceState === "loading"} onClick={() => void toggleReference()}>
          {referenceState === "loading" ? "Loading reference…" : referenceState === "playing" ? "Close reference" : "Watch reference"}
        </button>}
        <button type="button" className="fullscreen-button" ref={expandButtonRef}
          aria-controls="simulator-well" aria-expanded={isFullscreen}
          disabled={!fullscreenAvailable}
          title={fullscreenAvailable ? "Open the simulation at full-screen size" : "Fullscreen is not available in this browser"}
          onClick={() => void enterFullscreen()}>
          {fullscreenAvailable ? "Expand simulator" : "Fullscreen unavailable"}
        </button>
      </header>
      {referenceState === "playing" && <p className="reference-disclosure" role="status">
        Analytic reference controller · not a learned policy{scenarioId === "orbital-docking" ? " · 8× playback" : ""}
      </p>}
      {referenceError && <p className="reference-disclosure" role="alert">{referenceError}</p>}
      {viewingReplay && <div className="replay-controls">
        <p>Recorded rollout · fixed starting state. Playback stops at its final frame.
          {status?.training ? " Training continues in the background." : " Playback does not train."}</p>
        <button type="button" onClick={restartReplay}>Restart replay</button>
        <button type="button" onClick={clearGhost}>Close replay</button>
      </div>}
      {scenarioId === "lunar-lander" && !viewingReplay && <p className="reference-disclosure">
        Training includes practice close to the pad. A safe landing ends one episode;
        training then starts another. Results measures full descents.
      </p>}
      <div className="simulator-well" id="simulator-well" ref={stageRef}>
        {fullscreenError && (
          <div className="fullscreen-feedback" role="alert">{fullscreenError}</div>
        )}
        {isFullscreen && (
          <button type="button" className="fullscreen-exit" ref={exitButtonRef}
            onClick={() => void exitFullscreen()}>Exit fullscreen</button>
        )}
        <canvas ref={canvasRef} className="track-canvas" role="img"
          aria-label={`${currentScenario?.name ?? "Experiment"} ${referenceState === "playing" ? "analytic reference demonstration" : replayOnly ? "saved policy replay" : "live policy simulation"}`}>
          Live visual simulation for {currentScenario?.name ?? "the active experiment"}.
        </canvas>
        {!scene && !sceneError && <div className="track-loading">Loading environment…</div>}
        {sceneError && (
          <div className="track-loading scene-error" role="alert">
            <strong>{sceneError}</strong>
            <button type="button" onClick={() => setSceneAttempt((value) => value + 1)}>Retry</button>
          </div>
        )}
      {scene && !status?.training && !frameRef.current && !viewingReplay && referenceState !== "playing" && (
          <div className="track-idle"><strong>Environment ready</strong><span>Choose a budget and run the policy.</span></div>
      )}
      {ghostEpisode != null && referenceState !== "playing" && (
          <div className="ghost-chip">{ghostRef.current?.lap.archive_id ? "Verified saved policy" : `${replayOnly ? "Saved policy" : "Comparing"} · episode ${ghostEpisode}`}</div>
      )}
        {replayOnly && playback?.finished && <div className="termination-notice replay-ended" role="status">
          <strong>{telemetry?.terminal && telemetry.cause ? formatTerminationCause(telemetry.cause) : "Replay finished"}</strong>
          <small>Replay finished · final frame held. Restart to watch it again.</small>
        </div>}
        {!replayOnly && telemetry?.terminal && telemetry.cause && (
          <div
            className="termination-notice"
            role="status"
            aria-live="polite"
            aria-atomic="true"
            data-cause={telemetry.cause}
          >
            <span>{referenceState === "playing" ? "Reference demonstration ended" : `Episode ${displayedEpisodeNumber(telemetry, status?.episode ?? 0)} ended`}</span>
            <strong>{formatTerminationCause(telemetry.cause)}</strong>
            <small>
              Ended after {formatEpisodeDuration(
                telemetry.terminal_steps ?? 0,
                currentScenario?.horizon_steps ?? 0,
                currentScenario?.horizon_seconds ?? null,
              )}
              {status?.training && " · Next training attempt starts automatically"}
            </small>
          </div>
        )}
        {replayOnly ? <div className="scene-telemetry" aria-label="Saved replay telemetry">
          <span><small>Saved policy</small><strong>{ghostRef.current?.lap.archive_id ? "Verified replay" : `Episode ${ghostEpisode}`}</strong></span>
          <span><small>Playback</small><strong>{playback?.elapsed.toFixed(1) ?? "0.0"} / {playback?.duration.toFixed(1) ?? "—"} s</strong></span>
          {telemetry?.hits != null && <span><small>Returns</small><strong>{telemetry.hits} / {telemetry.target_hits}</strong></span>}
          {telemetry?.gates != null && <span><small>Gates</small><strong>{telemetry.gates} / {telemetry.target_gates}</strong></span>}
          {telemetry?.coins != null && <span><small>Coins</small><strong>{telemetry.coins} / {telemetry.target_coins}</strong></span>}
        </div> : <div className="scene-telemetry" aria-label="Live simulator telemetry">
          <span><small>Episode</small><strong>{displayedEpisodeNumber(
            telemetry, status?.episode ?? 0,
          )}</strong></span>
          <span><small>Return</small><strong>{telemetry?.episode_reward?.toFixed(1) ?? "—"}</strong></span>
          {telemetry?.car && <span><small>Speed</small><strong>{Math.round(telemetry.car.speed * 3.6)} km/h</strong></span>}
          {telemetry?.laps != null && <span><small>Laps</small><strong>{telemetry.laps}</strong></span>}
          {telemetry?.waypoints != null && <span><small>Waypoints</small><strong>{telemetry.waypoints} / 5</strong></span>}
          {telemetry?.balance_time != null && <span><small>Balanced</small><strong>{telemetry.balance_time.toFixed(2)} s</strong></span>}
          {telemetry?.peak_position != null && <span><small>Peak position</small><strong>{telemetry.peak_position.toFixed(3)}</strong></span>}
          {telemetry?.distance != null && <span><small>Target distance</small><strong>{telemetry.distance.toFixed(3)} m</strong></span>}
          {telemetry?.hold != null && <span><small>Stable hold</small><strong>{telemetry.hold.toFixed(1)} s</strong></span>}
          {telemetry?.hits != null && <span><small>Returns</small><strong>{telemetry.hits} / {telemetry.target_hits}</strong></span>}
          {telemetry?.gates != null && <span><small>Gates</small><strong>{telemetry.gates} / {telemetry.target_gates}</strong></span>}
          {telemetry?.coins != null && <span><small>Coins</small><strong>{telemetry.coins} / {telemetry.target_coins}</strong></span>}
          {status?.training && currentScenario && stepsPerSecond > 0 && (
            <span><small>Simulation</small><strong>{formatSimulationRate(
              stepsPerSecond, currentScenario.horizon_steps, currentScenario.horizon_seconds,
            )}</strong></span>
          )}
        </div>}
      </div>
    </section>
  );
}

function drawTrail(ctx: CanvasRenderingContext2D, trail: TrailPoint[], rgb: string) {
  for (let i = 0; i < trail.length; i++) {
    const p = trail[i];
    const a = (i / trail.length) * 0.4;
    ctx.fillStyle = `rgba(${rgb},${a.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.drift > 0.4 ? 2 : 1.3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderTrack(ctx: CanvasRenderingContext2D, track: TrackGeometry, zones: ZoneInfo[]) {
  const { centerline, normals, half_widths, checkpoints } = track;
  const n = centerline.length;
  const left = (i: number): [number, number] => [
    centerline[i][0] + normals[i][0] * half_widths[i],
    centerline[i][1] + normals[i][1] * half_widths[i],
  ];
  const right = (i: number): [number, number] => [
    centerline[i][0] - normals[i][0] * half_widths[i],
    centerline[i][1] - normals[i][1] * half_widths[i],
  ];

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const [x, y] = left(i);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  for (let i = n - 1; i >= 0; i--) {
    const [x, y] = right(i);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = COLORS.asphalt;
  ctx.fill();

  // Surface zones (rain patches etc.) tinted over the asphalt.
  for (const zone of zones) {
    const idxs: number[] = [];
    if (zone.start_idx <= zone.end_idx) {
      for (let i = zone.start_idx; i <= zone.end_idx; i++) idxs.push(i);
    } else {
      for (let i = zone.start_idx; i < n; i++) idxs.push(i);
      for (let i = 0; i <= zone.end_idx; i++) idxs.push(i);
    }
    ctx.beginPath();
    idxs.forEach((i, k) => {
      const [x, y] = left(i);
      k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    for (let k = idxs.length - 1; k >= 0; k--) {
      const [x, y] = right(idxs[k]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = zone.color;
    ctx.fill();
  }

  for (const edge of [left, right]) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const [x, y] = edge(i % n);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.edge;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.strokeStyle = COLORS.curbRed;
    ctx.lineWidth = 2.4;
    ctx.setLineDash([9, 14]);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  for (const idx of checkpoints) {
    if (idx === track.start_index) continue;
    const [lx, ly] = left(idx);
    const [rx, ry] = right(idx);
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(rx, ry);
    ctx.strokeStyle = COLORS.checkpoint;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  const s = track.start_index;
  const [lx, ly] = left(s);
  const [rx, ry] = right(s);
  const steps = 8;
  for (let k = 0; k < steps; k++) {
    const t0 = k / steps;
    const t1 = (k + 1) / steps;
    ctx.strokeStyle = k % 2 === 0 ? COLORS.startLine : "#555";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(lx + (rx - lx) * t0, ly + (ry - ly) * t0);
    ctx.lineTo(lx + (rx - lx) * t1, ly + (ry - ly) * t1);
    ctx.stroke();
  }
}

function renderStatics(ctx: CanvasRenderingContext2D, statics: StaticPrimitive[]) {
  for (const s of statics) {
    switch (s.shape) {
      case "terrain": {
        const pts = s.points ?? [];
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.lineTo(1000, 700);
        ctx.lineTo(0, 700);
        ctx.closePath();
        ctx.fillStyle = COLORS.terrain;
        ctx.fill();
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.lineWidth = 2;
        ctx.stroke();
        break;
      }
      case "flag": {
        const { x = 0, y = 0 } = s;
        ctx.strokeStyle = "#d8dbe2";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y - 26);
        ctx.stroke();
        ctx.fillStyle = COLORS.car;
        ctx.beginPath();
        ctx.moveTo(x, y - 26);
        ctx.lineTo(x + 14, y - 21);
        ctx.lineTo(x, y - 16);
        ctx.closePath();
        ctx.fill();
        break;
      }
      case "circle": {
        const { x = 0, y = 0, r = 5 } = s;
        if (s.color) {
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 6]);
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        } else {
          ctx.fillStyle = "#d8dbe2";
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
        }
        break;
      }
      case "line": {
        const pts = s.points ?? [];
        if (pts.length < 2) break;
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.strokeStyle = s.color === "danger" ? "#c95663" : s.color ?? "rgba(231,240,237,.7)";
        ctx.lineWidth = s.color === "danger" ? 3 : s.color ? 2 : 5;
        ctx.stroke();
        break;
      }
      case "hill": {
        const pts = s.points ?? [];
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.lineTo(1000, 700);
        ctx.lineTo(0, 700);
        ctx.closePath();
        ctx.fillStyle = "#213b3a";
        ctx.fill();
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.strokeStyle = "#7db0a7";
        ctx.lineWidth = 4;
        ctx.stroke();
        break;
      }
    }
  }
}

function drawHud(ctx: CanvasRenderingContext2D, frame: FrameMsg, ghostEpisode: number | null) {
  ctx.save();
  ctx.font = "600 13px ui-monospace, monospace";
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  let y = 22;
  ctx.fillText(`EP ${displayedEpisodeNumber(frame, 0)}`, 14, y); y += 18;
  ctx.fillText(`R ${frame.episode_reward.toFixed(1)}`, 14, y); y += 18;
  if (frame.laps != null) { ctx.fillText(`LAP ${frame.laps}`, 14, y); y += 18; }
  if (frame.style != null) { ctx.fillText(`STYLE ${frame.style}`, 14, y); y += 18; }
  if (frame.overtakes != null) { ctx.fillText(`PASSES ${frame.overtakes}`, 14, y); y += 18; }
  if (frame.waypoints != null) { ctx.fillText(`WP ${frame.waypoints}/5`, 14, y); y += 18; }
  if (ghostEpisode != null) {
    ctx.fillStyle = COLORS.ghost;
    ctx.fillText(`GHOST EP ${ghostEpisode}`, 14, y);
  }

  // Prominent lap times, top-center.
  if (frame.last_lap != null || frame.best_lap != null) {
    ctx.textAlign = "center";
    ctx.font = "600 12px ui-monospace, monospace";
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText("LAST LAP            BEST LAP", W / 2, 20);
    ctx.font = "700 19px ui-monospace, monospace";
    ctx.fillStyle = "#fff";
    ctx.fillText(frame.last_lap != null ? `${frame.last_lap.toFixed(2)}s` : "—", W / 2 - 64, 42);
    ctx.fillStyle = "#ffd23f";
    ctx.fillText(frame.best_lap != null ? `${frame.best_lap.toFixed(2)}s` : "—", W / 2 + 64, 42);
    ctx.textAlign = "left";
  }

  if (frame.car) {
    const kmh = Math.round(frame.car.speed * 3.6);
    ctx.font = "700 26px ui-monospace, monospace";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "right";
    ctx.fillText(`${kmh}`, W - 48, 34);
    ctx.font = "600 11px ui-monospace, monospace";
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText("km/h", W - 14, 34);

    const bw = 90;
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.15)";
    ctx.fillRect(W - 14 - bw, 46, bw, 6);
    ctx.fillStyle = COLORS.car;
    ctx.fillRect(W - 14 - bw, 46, bw * frame.car.drift, 6);
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText("drift", W - 14 - bw, 64);
  }

  if (frame.fuel != null) {
    const bw = 90;
    const yb = frame.car ? 74 : 46;
    ctx.fillStyle = "rgba(255,255,255,0.15)";
    ctx.fillRect(W - 14 - bw, yb, bw, 6);
    ctx.fillStyle = frame.fuel > 0.25 ? "#5ec46c" : "#d33a2c";
    ctx.fillRect(W - 14 - bw, yb, bw * frame.fuel, 6);
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText("fuel", W - 14 - bw, yb + 18);
  }
  ctx.restore();
}

function drawBanner(ctx: CanvasRenderingContext2D, banner: LapBanner, now: number) {
  const fade = Math.min(1, (banner.until - now) / 500);
  ctx.save();
  ctx.globalAlpha = fade;
  ctx.font = "800 30px ui-monospace, monospace";
  ctx.textAlign = "center";
  const text = banner.pb ? `★ ${banner.text} — PERSONAL BEST` : banner.text;
  ctx.fillStyle = "rgba(10,12,16,0.75)";
  const w = ctx.measureText(text).width + 48;
  ctx.fillRect(W / 2 - w / 2, 72, w, 48);
  ctx.fillStyle = banner.pb ? "#ffd23f" : "#fff";
  ctx.fillText(text, W / 2, 105);
  ctx.restore();
}
