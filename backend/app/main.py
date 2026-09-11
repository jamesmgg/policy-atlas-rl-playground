"""FastAPI app: REST + WebSocket front for the multi-scenario PPO trainer."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .checkpoints import CheckpointRegistry, migrate_flat_layout
from .scenarios import get_spec, list_specs
from .settings import settings
from .trainer import Trainer, decimate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("main")


class ConnectionManager:
    """Fans trainer events (produced on the training thread) out to WS clients."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2000)
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit_threadsafe(self, msg: dict) -> None:
        if self.loop is None:
            return
        try:
            self.loop.call_soon_threadsafe(self._put, msg)
        except RuntimeError:
            pass  # loop shutting down

    def _put(self, msg: dict) -> None:
        try:
            self.queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # drop under backpressure; frames are disposable

    async def broadcaster(self) -> None:
        while True:
            msg = await self.queue.get()
            if not self.clients:
                continue
            data = json.dumps(msg)
            dead = []
            # Membership may change at every await as browsers connect/leave.
            # Concurrent, bounded sends keep one stalled peer from blocking all.
            async def send(ws):
                try:
                    await asyncio.wait_for(ws.send_text(data), timeout=2.0)
                except Exception:
                    dead.append(ws)
            await asyncio.gather(*(send(ws) for ws in tuple(self.clients)))
            self.clients.difference_update(dead)


manager = ConnectionManager()
migrate_flat_layout(settings.checkpoint_dir)
trainer = Trainer(settings)
trainer.emit = manager.emit_threadsafe


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.loop = asyncio.get_running_loop()
    task = asyncio.create_task(manager.broadcaster())
    log.info("rl-simulator backend up on port %d (device=%s, scenario=%s)",
             settings.port, trainer.device, trainer.spec.id)
    yield
    await asyncio.to_thread(trainer.shutdown)
    task.cancel()


app = FastAPI(title="rl-simulator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ REST

class StartRequest(BaseModel):
    max_episodes: int | None = Field(default=None, ge=1, le=1_000_000)
    checkpoint_every_n: int | None = Field(default=None, ge=1, le=100_000)


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, le=2 ** 32 - 1)


class ScenarioRequest(BaseModel):
    id: str


class EvaluationRequest(BaseModel):
    scenario_id: str
    episodes: int = Field(default=10, ge=1, le=50)


@app.get("/api/health")
def health():
    return {"ok": True, "training": trainer.running, "episode": trainer.episode,
            "scenario": trainer.spec.id, "device": str(trainer.device)}


@app.get("/api/scene")
def api_scene():
    return {"scenario_id": trainer.spec.id, **trainer.spec.scene()}


@app.get("/api/scenarios")
def api_scenarios():
    out = []
    for spec in list_specs():
        registry = CheckpointRegistry(
            settings.checkpoint_dir, spec.id, spec.checkpoint_schema)
        metas = registry.list()
        progress = None
        if metas:
            metrics = [m["eval_metric"] for m in metas if m.get("eval_metric") is not None]
            best = None
            if metrics:
                best = min(metrics) if spec.metric_mode == "min" else max(metrics)
            progress = {
                "episode": metas[-1]["episode"],
                "mean_reward": metas[-1]["mean_reward"],
                "best_metric": best,
                "checkpoints": len(metas),
            }
        out.append({**spec.info(), "progress": progress})
    return {"active": trainer.spec.id, "scenarios": out}


@app.post("/api/scenario")
async def api_select_scenario(req: ScenarioRequest):
    try:
        get_spec(req.id)
    except KeyError:
        raise HTTPException(404, f"unknown scenario: {req.id}")
    ok = await asyncio.to_thread(trainer.switch_scenario, req.id)
    if not ok:
        raise HTTPException(409, "training thread did not stop in time; retry")
    return trainer.status()


@app.get("/api/checkpoints")
def api_checkpoints():
    return {"scenario_id": trainer.spec.id, "checkpoints": trainer.registry.list()}


@app.get("/api/runs")
def api_runs():
    return {"scenario_id": trainer.spec.id,
            "archives": trainer.registry.list_archives()}


@app.post("/api/runs/{archive_id}/restore")
async def api_restore_run(archive_id: str):
    if trainer.running:
        raise HTTPException(409, "pause training before restoring an archived run")
    restored = await asyncio.to_thread(trainer.restore_archive, archive_id)
    if not restored:
        raise HTTPException(404, "archived run not found or incompatible")
    return trainer.status()


@app.get("/api/training/status")
def api_status():
    return trainer.status()


@app.get("/api/reference/{scenario_id}")
async def api_reference(scenario_id: str):
    from .evaluation import cached_reference_replay
    try:
        return await asyncio.to_thread(cached_reference_replay, scenario_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/evaluation")
async def api_evaluation(req: EvaluationRequest):
    from .evaluation import compare_controllers

    def evaluate():
        # Deepcopy takes no random draws and freezes the policy. The trainer may
        # resume after this snapshot without changing the evaluation's weights.
        with trainer._lock:
            if trainer.running:
                raise HTTPException(409, "pause training before comparing controllers")
            if trainer.spec.id != req.scenario_id:
                raise HTTPException(409, "the active experiment changed; retry")
            spec, agent, episode = trainer.spec, copy.deepcopy(trainer.agent), trainer.episode
        result = compare_controllers(spec, agent, episodes=req.episodes)
        return {**result, "policy_episode": episode}

    return await asyncio.to_thread(evaluate)


@app.post("/api/training/start")
def api_start(req: StartRequest):
    started = trainer.start(req.max_episodes, req.checkpoint_every_n)
    if not started:
        raise HTTPException(409, "training already running")
    return trainer.status()


@app.post("/api/training/stop")
def api_stop():
    trainer.stop()
    return {"ok": True}


@app.post("/api/training/reset")
def api_reset(req: ResetRequest = ResetRequest()):
    if not trainer.reset_agent(req.seed):
        raise HTTPException(409, "stop training before resetting")
    return trainer.status()


# ------------------------------------------------------------------ WebSocket

@app.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    await ws.accept()
    manager.clients.add(ws)
    try:
        await ws.send_text(json.dumps(trainer.status()))
        await ws.send_text(json.dumps(
            {"type": "checkpoint_list", "scenario_id": trainer.spec.id,
             "checkpoints": trainer.registry.list()}))
        await ws.send_text(json.dumps(
            {"type": "history", "scenario_id": trainer.spec.id,
             "history": decimate(trainer.history)}))
        if trainer.ghost:
            await ws.send_text(json.dumps({"type": "ghost_lap", **trainer.ghost}))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await handle_client_message(ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        manager.clients.discard(ws)


async def handle_client_message(ws: WebSocket, msg: dict) -> None:
    try:
        if not isinstance(msg, dict):
            raise ValueError("command must be a JSON object")
        mtype = msg.get("type")
        if mtype == "start_training":
            request = StartRequest.model_validate(msg)
            if not trainer.start(request.max_episodes, request.checkpoint_every_n):
                raise ValueError("training already running")
        elif mtype == "stop_training":
            trainer.stop()
        elif mtype == "reset_training":
            request = ResetRequest.model_validate(msg)
            if not await asyncio.to_thread(trainer.reset_agent, request.seed):
                raise ValueError("pause training before starting a new seeded run")
        elif mtype == "set_scenario":
            ok = await asyncio.to_thread(trainer.switch_scenario, str(msg["id"]))
            if not ok:
                await ws.send_text(json.dumps(
                    {"type": "error", "message": "could not switch scenario; retry"}))
        elif mtype == "set_ghost":
            ok = await asyncio.to_thread(trainer.set_ghost, int(msg["episode"]))
            if not ok:
                await ws.send_text(json.dumps(
                    {"type": "error", "message": f"no ghost lap for episode {msg['episode']}"}))
        elif mtype == "clear_ghost":
            trainer.clear_ghost()
        elif mtype == "load_checkpoint":
            ok = await asyncio.to_thread(trainer.load_checkpoint, int(msg["episode"]))
            if not ok:
                await ws.send_text(json.dumps(
                    {"type": "error", "message": "stop training before loading a checkpoint"}))
        else:
            raise ValueError(f"unknown command: {mtype}")
    except ValueError as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except KeyError as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except FileNotFoundError:
        await ws.send_text(json.dumps(
            {"type": "error", "message": f"checkpoint {msg.get('episode')} not found"}))
    except Exception as exc:  # surface unexpected errors to the UI
        log.exception("ws message failed: %s", msg)
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
