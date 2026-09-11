"""Regressions discovered in the September 2026 engine audit."""
import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from app.ppo.agent import PPOAgent
from app.ppo.buffer import RolloutBuffer
from app.settings import Settings, load_settings
from app.trainer import Trainer


class OptimizerAuditTests(unittest.TestCase):
    def test_kl_guard_rejects_update_before_mutating_parameters(self):
        torch.manual_seed(71)
        agent = PPOAgent(3, 1, 0, torch.device("cpu"))
        buffer = RolloutBuffer(8, 3, 1)
        for index in range(8):
            obs = np.array([index / 8, 0.25, -0.5], np.float32)
            action, log_prob, value = agent.select_action(obs)
            buffer.add(obs, action, log_prob - 2.0, 1.0, True, value)
        buffer.compute_gae(0.0, True)
        before = {k: v.clone() for k, v in agent.network.state_dict().items()}
        stats = agent.update(buffer)
        for key, value in agent.network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(stats["optimizer_steps"], 0)
        self.assertEqual(stats["kl_early_stopped"], 1)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_empty_rollout_is_rejected_explicitly(self):
        agent = PPOAgent(3, 1, 0, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "empty"):
            agent.update(RolloutBuffer(8, 3, 1))

    def test_fast_prediction_matches_policy_and_does_not_draw_randomness(self):
        for binary in (0, 1):
            agent = PPOAgent(3, 2, binary, torch.device("cpu"))
            predict = getattr(agent, "predict", None)
            self.assertTrue(callable(predict), "action-only inference is missing")
            obs = np.array([0.7, -0.2, 0.3], np.float32)
            expected = agent.select_action(obs, deterministic=True)[0]
            state = torch.get_rng_state().clone()
            actual = predict(obs)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(torch.get_rng_state(), state)


class TrainerAuditTests(unittest.TestCase):
    def test_archive_restore_rejects_other_engine_before_moving_live_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(Settings(8901, Path(tmp), 50, 1, False, eval_episodes=1))
            with patch.object(trainer, "_run_eval", return_value={
                "reward": 0.0, "metric": 0.0, "trajectory": [], "episodes": 1,
                "evaluation_suite": "policy-atlas-eval-v1-n1", "seed": 42,
            }):
                trainer._save_checkpoint()
            archived = trainer.registry.archive_current()
            with patch("app.trainer.source_digest", return_value="changed-engine"):
                self.assertFalse(trainer.restore_archive(archived.name))
            self.assertTrue(archived.exists())
            self.assertEqual(trainer.registry.list(), [])

    def test_restoring_a_policy_does_not_retrain_its_demonstration_warm_start(self):
        from app.ppo.demonstrations import BehaviorCloningWarmStart
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "state.json").write_text('{"active_scenario":"drone-hover"}')
            settings = Settings(8901, Path(tmp), 50, 1, False, eval_episodes=1)
            with patch.object(BehaviorCloningWarmStart, "apply", return_value={"samples": 123}) as warm:
                trainer = Trainer(settings)
                trainer.switch_scenario("drone-hover")
                with patch.object(trainer, "_run_eval", return_value={
                    "reward": 0.0, "metric": 0.0, "trajectory": [], "episodes": 1,
                    "evaluation_suite": "policy-atlas-eval-v1-n1", "seed": 42,
                }):
                    trainer._save_checkpoint()
                restored = Trainer(settings)
                self.assertEqual(warm.call_count, 1, "a verified checkpoint already contains the initialized actor")
                self.assertEqual(restored.actor_warm_start_diagnostics, {"samples": 123})

    def test_cpu_thread_budget_has_a_small_explicit_default(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = load_settings()
        self.assertEqual(getattr(settings, "cpu_threads", None), 1)
        with patch.dict("os.environ", {"TORCH_NUM_THREADS": "3"}):
            self.assertEqual(load_settings().cpu_threads, 3)

    def test_failure_reaches_clients_and_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(Settings(8901, Path(tmp), 50, 1, False, eval_episodes=1))
            events = []
            trainer.emit = events.append
            with patch.object(trainer.env, "step", side_effect=RuntimeError("synthetic failure")):
                trainer._run()
            self.assertTrue(any(e["type"] == "error" for e in events))
            self.assertEqual(events[-1]["type"], "status")
            self.assertFalse(events[-1]["training"])
            self.assertIn("synthetic failure", trainer.status()["last_error"])


class SocketAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_during_send_does_not_kill_broadcaster(self):
        from app.main import ConnectionManager
        manager = ConnectionManager()
        received = asyncio.Event()

        class Socket:
            def __init__(self, disconnect=False):
                self.disconnect = disconnect
                self.messages = []

            async def send_text(self, data):
                self.messages.append(json.loads(data))
                if self.disconnect:
                    manager.clients.discard(self)
                elif len(self.messages) == 2:
                    received.set()
                await asyncio.sleep(0)

        stable, departing = Socket(), Socket(True)
        manager.clients.update((stable, departing))
        manager._put({"type": "status", "sequence": 1})
        manager._put({"type": "status", "sequence": 2})
        task = asyncio.create_task(manager.broadcaster())
        try:
            await asyncio.wait_for(received.wait(), 1.0)
            self.assertFalse(task.done())
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_invalid_command_and_busy_start_return_errors(self):
        import app.main as main

        class Socket:
            def __init__(self):
                self.messages = []

            async def send_text(self, data):
                self.messages.append(json.loads(data))

        for payload in ([], None, {"type": "start_training", "max_episodes": -1}):
            socket = Socket()
            await main.handle_client_message(socket, payload)
            self.assertEqual(socket.messages[-1]["type"], "error")
        socket = Socket()
        with patch.object(main.trainer, "start", return_value=False):
            await main.handle_client_message(socket, {"type": "start_training"})
        self.assertEqual(socket.messages[-1]["type"], "error")


if __name__ == "__main__":
    unittest.main()
