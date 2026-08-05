"""FastAPI app: REST + WebSocket front for the multi-scenario PPO trainer."""
from __future__ import annotations

import asyncio
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
            for ws in self.clients:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


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
    mtype = msg.get("type")
    try:
        if mtype == "start_training":
            trainer.start(msg.get("max_episodes"), msg.get("checkpoint_every_n"))
        elif mtype == "stop_training":
            trainer.stop()
        elif mtype == "reset_training":
            trainer.reset_agent(msg.get("seed"))
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
    except KeyError as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except FileNotFoundError:
        await ws.send_text(json.dumps(
            {"type": "error", "message": f"checkpoint {msg.get('episode')} not found"}))
    except Exception as exc:  # surface unexpected errors to the UI
        log.exception("ws message failed: %s", msg)
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
