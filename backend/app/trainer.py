"""Background training loop with scenario switching.

Runs PPO as fast as the hardware allows in a daemon thread; the UI gets a
throttled live view (~20 frames/s), an event per episode and per PPO update.
Each scenario owns an independent agent, history and checkpoint directory;
switching stops the thread, swaps everything, and restores that scenario's
newest checkpoint. The active scenario persists in state.json on the volume.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import random
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .checkpoints import CheckpointRegistry
from .ppo import agent as ppo_defaults
from .ppo.agent import PPOAgent
from .ppo.buffer import RolloutBuffer
from .ppo.initialization import default_actor_initialization
from .scenarios import get_spec
from .scenarios.registry import DEFAULT_SCENARIO
from .settings import Settings

log = logging.getLogger("trainer")

ROLLOUT_STEPS = 2048
GAMMA = 0.995
GAE_LAMBDA = 0.95
FRAME_INTERVAL = 0.05  # seconds between live frames sent to clients
SWITCH_JOIN_TIMEOUT = 15.0
EVALUATION_SUITE_VERSION = "policy-atlas-eval-v1"
EVALUATION_SEED_BASE = 100_000
TRAINING_REWARD_SCALE = 0.01


def source_digest_for_root(root: Path) -> str:
    """Fingerprint a Python source tree independent of checkout line endings.

    Relative paths and every source byte remain significant. Only CRLF and
    bare CR line endings are canonicalized to LF so the same experiment source
    has one identity on Windows and Linux checkouts.
    """
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def source_digest() -> str:
    """Fingerprint the Python experiment engine stored with each checkpoint."""
    return source_digest_for_root(Path(__file__).resolve().parent)


def seed_everything(seed: int) -> None:
    """Seed every random stream used by the trainer and policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluation_seed(index: int) -> int:
    """A fixed test suite makes independent training seeds comparable."""
    return EVALUATION_SEED_BASE + index


def evaluation_suite_id(episodes: int) -> str:
    """The number of fixed starts is part of the comparison protocol."""
    return f"{EVALUATION_SUITE_VERSION}-n{episodes}"


def aggregate_evaluations(results: list[dict], metric_mode: str) -> dict:
    """Aggregate repeated evaluation episodes without cherry-picking a run."""
    del metric_mode  # direction is display metadata; evaluation reports the mean.
    rewards = np.asarray([r["reward"] for r in results], dtype=np.float64)
    metrics = np.asarray([r["metric"] for r in results if r.get("metric") is not None],
                         dtype=np.float64)
    failure_progress = np.asarray([
        r["failure_progress"] for r in results
        if r.get("failure_progress") is not None
    ], dtype=np.float64)
    successes = [bool(r.get("success", False)) for r in results]
    success_rate = float(np.mean(successes)) if successes else None
    success_ci_low, success_ci_high = wilson_interval(
        sum(successes), len(successes))
    return {
        "episodes": len(results),
        "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_std": float(rewards.std()) if len(rewards) else 0.0,
        "metric": float(metrics.mean()) if len(metrics) else None,
        "metric_std": float(metrics.std()) if len(metrics) else None,
        "failure_progress": (float(failure_progress.mean())
                             if len(failure_progress) else None),
        "success_rate": success_rate,
        "success_ci_low": success_ci_low,
        "success_ci_high": success_ci_high,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96
                    ) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total))
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_time_limit(reward: float, next_value: float, done: bool,
                         info: dict, gamma: float = GAMMA) -> float:
    """Bootstrap only external truncations, never an intrinsic task deadline."""
    if (done and info.get("truncated", False)
            and not info.get("task_deadline", False)):
        return reward + gamma * next_value
    return reward


def training_reward(reward: float, next_value: float, done: bool,
                    info: dict) -> float:
    """Map display rewards into stable critic units before any bootstrap.

    Multiplying every reward by one positive constant preserves the policy
    objective. Raw environment returns remain untouched for the UI, ranking,
    and evaluation reports.
    """
    scaled_reward = reward * TRAINING_REWARD_SCALE
    return bootstrap_time_limit(
        scaled_reward, next_value, done, info, gamma=GAMMA)


def actor_initialization_protocol(spec, env) -> dict:
    """Describe the fresh-policy recipe without widening the agent interface."""
    initialization = getattr(spec, "actor_initialization", None)
    if initialization is None:
        initialization = default_actor_initialization(
            env.n_continuous, env.n_binary)
    return initialization.protocol(env.n_continuous, env.n_binary)


def learning_payload(observation: np.ndarray, action: np.ndarray,
                     reward: float) -> dict:
    """Compact one transition for the explanatory UI without flooding the WS."""
    obs = np.nan_to_num(np.asarray(observation)[:6], nan=0.0,
                        posinf=999.0, neginf=-999.0)
    act = np.nan_to_num(np.asarray(action)[:8], nan=0.0,
                        posinf=1.0, neginf=-1.0)
    return {
        "observation": [round(float(value), 3) for value in obs],
        "action": [round(float(value), 3) for value in act],
        "reward": round(float(reward), 3),
    }


def reset_training_environment(env, episode: int) -> np.ndarray:
    """Reset a training env with its one-based episode when it supports it."""
    episode_reset = getattr(env, "reset_for_training_episode", None)
    if callable(episode_reset):
        return episode_reset(episode)
    return env.reset()


def capture_rng_state(env) -> dict:
    curriculum_state = getattr(env, "training_curriculum_state", None)
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment": env.rng.getstate() if hasattr(env, "rng") else None,
        "training_curriculum": (
            curriculum_state() if callable(curriculum_state) else None),
    }
    return state


def restore_rng_state(state: dict, env) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("environment") is not None and hasattr(env, "rng"):
        env.rng.setstate(state["environment"])
    curriculum_state = state.get("training_curriculum")
    if curriculum_state is not None:
        restore = getattr(env, "restore_training_curriculum_state", None)
        if not callable(restore):
            raise ValueError("checkpoint has curriculum state for an incompatible env")
        restore(curriculum_state)


def evaluate_training_curriculum(curriculum, training_env, agent) -> dict:
    """Run a fixed deterministic suite on only the active training frontier."""
    get_state = getattr(training_env, "training_curriculum_state", None)
    if not callable(get_state):
        raise TypeError("training environment does not expose curriculum state")
    frontier = int(get_state()["frontier"])
    successes = 0
    seeds = []
    for episode_index in range(curriculum.evaluation_episodes):
        seed = curriculum.evaluation_seed(frontier, episode_index)
        seeds.append(seed)
        env = curriculum.make_evaluation_env(frontier)
        if hasattr(env, "rng"):
            env.rng.seed(seed)
        obs = env.reset()
        for _ in range(env.max_steps):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                break
        successes += int(bool(env.episode_summary().get("success", False)))
    return {
        "frontier": frontier,
        "episodes": curriculum.evaluation_episodes,
        "successes": successes,
        "success_rate": successes / curriculum.evaluation_episodes,
        "evaluation_suite": curriculum.evaluation_suite_id(frontier),
        "seeds": seeds,
    }


class Trainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        log.info("training device: %s", self.device)

        self.emit: Callable[[dict], None] = lambda msg: None
        self.max_episodes = settings.max_episodes
        self.checkpoint_every_n = settings.checkpoint_every_n
        self.seed = settings.seed
        self.update_count = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._state_path = settings.checkpoint_dir / "state.json"
        self._load_scenario(self._read_active_scenario())

    # ---------------------------------------------------------- scenario state

    def _read_active_scenario(self) -> str:
        try:
            scenario_id = json.loads(self._state_path.read_text())["active_scenario"]
            get_spec(scenario_id)
            return scenario_id
        except (OSError, json.JSONDecodeError, KeyError):
            return DEFAULT_SCENARIO

    def _write_active_scenario(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps({"active_scenario": self.spec.id}))

    def _load_scenario(self, scenario_id: str) -> None:
        """Build env/agent/registry for a scenario. Caller holds the lock (or init)."""
        self.spec = get_spec(scenario_id)
        seed_everything(self.seed)
        self.env = self.spec.make_training_env()
        if hasattr(self.env, "rng"):
            self.env.rng.seed(self.seed)
            reset_training_environment(self.env, 1)
        self.agent = PPOAgent(self.env.obs_dim, self.env.n_continuous,
                              self.env.n_binary, self.device,
                              actor_initialization=self.spec.actor_initialization)
        self.registry = CheckpointRegistry(
            self.settings.checkpoint_dir, self.spec.id, self.spec.checkpoint_schema)
        self.episode = 0
        self.total_steps = 0
        self.update_count = 0
        self.sps = 0.0
        self.history = []
        self.best_reward: float | None = None
        self.best_metric: float | None = None
        self.ghost: dict | None = None
        self._learning: dict | None = None
        self.latest_update_metrics: dict[str, float] | None = None
        self._restore_latest()
        self.run_start_episode = self.episode
        self.run_target_episode = self.episode
        self._write_active_scenario()

    def switch_scenario(self, scenario_id: str) -> bool:
        get_spec(scenario_id)  # raises KeyError for unknown ids
        with self._lock:
            if scenario_id == self.spec.id:
                return True
            if self.running:
                self._stop.set()
                self._thread.join(timeout=SWITCH_JOIN_TIMEOUT)
                if self._thread.is_alive():
                    log.error("training thread did not stop within %ss",
                              SWITCH_JOIN_TIMEOUT)
                    return False
            self._load_scenario(scenario_id)
            self.emit({"type": "scenario_changed", **self.spec.info()})
            self._emit_status()
            self.emit({"type": "checkpoint_list", "scenario_id": self.spec.id,
                       "checkpoints": self.registry.list()})
            self.emit({"type": "history", "scenario_id": self.spec.id,
                       "history": decimate(self.history)})
            self.emit({"type": "ghost_clear"})
            return True

    # ---------------------------------------------------------------- control

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, max_episodes: int | None = None,
              checkpoint_every_n: int | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            if max_episodes is not None:
                target = max(self.episode + 1, max_episodes)
            else:
                target = max(self.episode + 1, self.max_episodes)
            if not (target == self.run_target_episode
                    and self.episode < self.run_target_episode):
                self.run_start_episode = self.episode
            self.run_target_episode = target
            self.max_episodes = target
            if checkpoint_every_n is not None:
                self.checkpoint_every_n = max(1, checkpoint_every_n)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="ppo-trainer")
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        """Finish the current episode and persist it before process shutdown."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=SWITCH_JOIN_TIMEOUT)
            if thread.is_alive():
                log.error("training thread did not stop before shutdown timeout")

    def reset_agent(self, seed: int | None = None) -> bool:
        """Reinitialize weights and history for the active scenario. Only when stopped."""
        with self._lock:
            if self.running:
                return False
            if seed is not None:
                self.seed = max(0, min(int(seed), 2 ** 32 - 1))
            self.registry.archive_current()
            seed_everything(self.seed)
            env = self.env
            if hasattr(env, "rng"):
                env.rng.seed(self.seed)
            self.agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                                  self.device,
                                  actor_initialization=self.spec.actor_initialization)
            self.episode = 0
            self.total_steps = 0
            self.update_count = 0
            self.run_start_episode = 0
            self.run_target_episode = 0
            self.history = []
            self.best_reward = None
            self.best_metric = None
            self.ghost = None
            self._learning = None
            self.latest_update_metrics = None
            reset_curriculum = getattr(env, "reset_training_curriculum", None)
            if callable(reset_curriculum):
                reset_curriculum()
            reset_training_environment(env, 1)
            self._emit_status()
            self.emit({"type": "history", "scenario_id": self.spec.id, "history": []})
            self.emit({"type": "checkpoint_list", "scenario_id": self.spec.id,
                       "checkpoints": []})
            self.emit({"type": "ghost_clear"})
            return True

    def load_checkpoint(self, episode: int) -> bool:
        with self._lock:
            if self.running:
                return False
            data = self.registry.load_into(episode, self.agent)
            # Loading an older policy creates a new branch. Preserve its newer
            # descendants before their episode-numbered files can be replaced.
            self.registry.archive_after(episode)
            self.history = data.get("history", [])
            self.episode = episode
            self.run_start_episode = episode
            self.run_target_episode = episode
            stored_steps = data.get("total_steps")
            self.total_steps = int(
                stored_steps if stored_steps is not None
                else sum(h.get("steps", 0) for h in self.history))
            self.update_count = int(data.get("update_count", 0))
            if data.get("meta", {}).get("seed") is not None:
                self.seed = int(data["meta"]["seed"])
                seed_everything(self.seed)
            if data.get("rng_state"):
                restore_rng_state(data["rng_state"], self.env)
            self.latest_update_metrics = data.get("meta", {}).get(
                "training_diagnostics")
            self.ghost = None
            self._recompute_bests()
            self._emit_status()
            self.emit({"type": "history", "scenario_id": self.spec.id,
                       "history": decimate(self.history)})
            self.emit({"type": "checkpoint_list", "scenario_id": self.spec.id,
                       "checkpoints": self.registry.list()})
            self.emit({"type": "ghost_clear"})
            return True

    def set_ghost(self, episode: int) -> bool:
        data = self.registry.load(episode)
        trajectory = data.get("trajectory") or []
        if not trajectory:
            return False
        self.ghost = {"episode": episode, "dt": self.env.dt,
                      "trajectory": trajectory}
        self.emit({"type": "ghost_lap", **self.ghost})
        return True

    def clear_ghost(self) -> None:
        self.ghost = None
        self.emit({"type": "ghost_clear"})

    def restore_archive(self, archive_id: str) -> bool:
        """Swap a recoverable run branch into the active workspace."""
        with self._lock:
            if self.running or not self.registry.restore_archive(archive_id):
                return False
            scenario_id = self.spec.id
            self._load_scenario(scenario_id)
            self.emit({"type": "scenario_changed", **self.spec.info()})
            self._emit_status()
            self.emit({"type": "checkpoint_list", "scenario_id": self.spec.id,
                       "checkpoints": self.registry.list()})
            self.emit({"type": "history", "scenario_id": self.spec.id,
                       "history": decimate(self.history)})
            self.emit({"type": "ghost_clear"})
            return True

    def status(self) -> dict:
        return {
            "type": "status",
            "scenario_id": self.spec.id,
            "scenario_kind": self.spec.kind,
            "metric_label": self.spec.metric_label,
            "metric_mode": self.spec.metric_mode,
            "training": self.running,
            "episode": self.episode,
            "max_episodes": self.max_episodes,
            "run_start_episode": getattr(self, "run_start_episode", self.episode),
            "run_target_episode": getattr(self, "run_target_episode", self.episode),
            "checkpoint_every_n": self.checkpoint_every_n,
            "total_steps": self.total_steps,
            "sps": round(self.sps),
            "seed": self.seed,
            "update_count": self.update_count,
            "eval_episodes": self.settings.eval_episodes,
            "evaluation_suite": evaluation_suite_id(self.settings.eval_episodes),
            "evaluation_seed_base": EVALUATION_SEED_BASE,
            "engine_source_sha256": source_digest(),
            "best_reward": self.best_reward,
            "best_metric": self.best_metric,
            "device": str(self.device),
            "ghost_episode": self.ghost["episode"] if self.ghost else None,
            "ppo_diagnostics": getattr(self, "latest_update_metrics", None),
        }

    # ------------------------------------------------------------- train loop

    def _run(self) -> None:
        self._emit_status()
        env = self.env
        # Update after at least ROLLOUT_STEPS, at the next episode boundary.
        # The extra capacity makes checkpoint cadence independent of PPO batch
        # boundaries while ensuring a saved state is exactly resumable.
        buffer = RolloutBuffer(
            ROLLOUT_STEPS + env.max_steps, env.obs_dim, self.agent.act_dim)
        obs = reset_training_environment(env, self.episode + 1)
        done = False
        last_frame = 0.0

        while not self._stop.is_set() and self.episode < self.max_episodes:
            buffer.reset()
            t0 = time.perf_counter()
            checkpoint_due = False
            # A pause/switch request finishes the current fixed-size rollout
            # and then stops at an episode boundary. Pause timing therefore
            # cannot change PPO minibatch/update boundaries.
            while not buffer.full:
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                # Show the state that actually produced this action. Pairing the
                # action with next_obs would make the explanatory policy loop
                # one transition out of phase.
                self._learning = learning_payload(obs, action, reward)
                next_value = (
                    self.agent.get_value(next_obs)
                    if done and info.get("truncated", False) else 0.0
                )
                buffer_reward = training_reward(
                    reward, next_value, done, info)
                buffer.add(obs, action, log_prob, buffer_reward, done, value)
                self.total_steps += 1

                now = time.monotonic()
                if now - last_frame >= FRAME_INTERVAL:
                    last_frame = now
                    self._emit_frame()

                if done:
                    # The regular 20 Hz throttle can otherwise skip the final
                    # state entirely before the environment immediately resets.
                    self._emit_frame(terminal=True)
                    self.episode += 1
                    checkpoint_due = self._on_episode_end() or checkpoint_due
                    if (self.episode >= self.max_episodes
                            or buffer.ptr >= ROLLOUT_STEPS):
                        break
                    obs = reset_training_environment(env, self.episode + 1)
                    done = False
                else:
                    obs = next_obs

            if buffer.ptr > 0:
                last_value = 0.0 if done else self.agent.get_value(obs)
                buffer.compute_gae(last_value, done, gamma=GAMMA,
                                   gae_lambda=GAE_LAMBDA)
                metrics = self.agent.update(buffer)
                self.update_count += 1
                self.sps = buffer.ptr / max(time.perf_counter() - t0, 1e-6)
                self.latest_update_metrics = {
                    key: round(value, 5) for key, value in metrics.items()
                }
                self.emit({"type": "ppo_update", "scenario_id": self.spec.id,
                           "episode": self.episode,
                           "total_steps": self.total_steps,
                           "update": self.update_count,
                           "sps": round(self.sps),
                           **self.latest_update_metrics})
            should_save = done and (
                checkpoint_due or self._stop.is_set()
                or self.episode >= self.max_episodes
            )
            if should_save:
                self._save_checkpoint()
            # Checkpoints capture RNG state before this reset. A restored
            # trainer also begins with exactly one reset, reproducing the same
            # pending initial condition.
            if done and not self._stop.is_set() and self.episode < self.max_episodes:
                obs = reset_training_environment(env, self.episode + 1)
                done = False

        final_status = self.status()
        final_status["training"] = False
        self.emit(final_status)
        log.info("[%s] training stopped at episode %d", self.spec.id, self.episode)

    def _on_episode_end(self) -> bool:
        entry = {"episode": self.episode, **self.env.episode_summary()}
        self.history.append(entry)
        if self.best_reward is None or entry["reward"] > self.best_reward:
            self.best_reward = entry["reward"]
        self._update_best_metric(entry.get("metric"))

        saved = self.episode % self.checkpoint_every_n == 0
        self.emit({"type": "episode_end", "scenario_id": self.spec.id,
                   **entry, "checkpoint_due": saved})
        return saved

    def _update_best_metric(self, metric: float | None) -> None:
        if metric is None:
            return
        if self.best_metric is None:
            self.best_metric = metric
        elif self.spec.metric_mode == "min":
            self.best_metric = min(self.best_metric, metric)
        else:
            self.best_metric = max(self.best_metric, metric)

    def _save_checkpoint(self) -> None:
        rng_state = capture_rng_state(self.env)
        curriculum_result = None
        try:
            eval_result = self._run_eval()
            curriculum_result = self._run_training_curriculum_eval()
        finally:
            restore_rng_state(rng_state, self.env)
        curriculum_diagnostic = None
        if curriculum_result is not None:
            record_result = getattr(
                self.env, "record_training_curriculum_evaluation", None)
            curriculum = getattr(self.spec, "training_curriculum", None)
            if not callable(record_result) or curriculum is None:
                raise TypeError("scenario curriculum contract is incomplete")
            transition = record_result(
                float(curriculum_result["success_rate"]), curriculum,
                evaluation_episode=self.episode)
            curriculum_diagnostic = {
                **curriculum_result,
                **transition,
                "state_after": self.env.training_curriculum_state(),
            }
        # Evaluation is observational with respect to every random stream. The
        # only intended mutation is the serialized curriculum gate transition.
        rng_state = capture_rng_state(self.env)
        eval_result["update_count"] = self.update_count
        eval_result["total_steps"] = self.total_steps
        training_diagnostics = dict(
            getattr(self, "latest_update_metrics", None) or {})
        if curriculum_diagnostic is not None:
            training_diagnostics["training_curriculum"] = curriculum_diagnostic
        eval_result["training_diagnostics"] = training_diagnostics or None
        eval_result["protocol"] = {
            "algorithm": "PPO",
            "version": 10,
            "rollout_steps": ROLLOUT_STEPS,
            "episode_aligned_rollouts": True,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "learning_rate": ppo_defaults.LR,
            "clip_epsilon": ppo_defaults.CLIP_EPS,
            "entropy_coefficient": ppo_defaults.ENT_COEF,
            "value_coefficient": ppo_defaults.VF_COEF,
            "training_reward_scale": TRAINING_REWARD_SCALE,
            "value_loss_scale": "rollout return RMS",
            "training_start_distribution": (
                getattr(
                    self.spec,
                    "training_start_distribution",
                    "scenario default starts",
                )
            ),
            "training_curriculum": (
                self.spec.training_curriculum.protocol()
                if getattr(self.spec, "training_curriculum", None) is not None
                else None
            ),
            "actor_initialization": actor_initialization_protocol(
                self.spec, self.env),
            "update_epochs": ppo_defaults.UPDATE_EPOCHS,
            "minibatch_size": ppo_defaults.MINIBATCH_SIZE,
            "target_kl": ppo_defaults.TARGET_KL,
            "evaluation_episodes": self.settings.eval_episodes,
            "evaluation_suite": evaluation_suite_id(self.settings.eval_episodes),
            "evaluation_seed_base": EVALUATION_SEED_BASE,
            "deterministic_evaluation": True,
            "engine_source_sha256": source_digest(),
            "reproducibility_scope": (
                "RNG-exact continuation on the saved runtime; bitwise identity "
                "is not guaranteed across devices or dependency builds"
            ),
            "device": str(self.device),
            "torch_version": str(torch.__version__),
        }
        eval_result["rng_state"] = rng_state
        meta = self.registry.save(self.episode, self.agent, self.history, eval_result)
        log.info("[%s] checkpoint ep%d: eval_reward=%.1f metric=%s",
                 self.spec.id, meta.episode, meta.eval_reward, meta.eval_metric)
        self.emit({"type": "checkpoint_list", "scenario_id": self.spec.id,
                   "checkpoints": self.registry.list()})

    def _run_training_curriculum_eval(self) -> dict | None:
        curriculum = getattr(self.spec, "training_curriculum", None)
        if curriculum is None:
            return None
        return evaluate_training_curriculum(curriculum, self.env, self.agent)

    def _run_eval(self) -> dict:
        results: list[dict] = []
        for i in range(self.settings.eval_episodes):
            env = self.spec.make_env(True)
            if hasattr(env, "rng"):
                env.rng.seed(evaluation_seed(i))
            obs = env.reset()
            for _ in range(env.max_steps):
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                if done:
                    break
            summary = env.episode_summary()
            results.append({
                "reward": env.episode_reward,
                "metric": summary.get("metric"),
                "failure_progress": summary.get("failure_progress"),
                "success": summary.get("success", False),
            })

        # A fixed canonical start remains the comparable ghost replay while the
        # reported score comes from the fixed, versioned test starts above.
        canonical = self.spec.make_env(False)
        obs = canonical.reset()
        trajectory: list[list[float]] = []
        for _ in range(canonical.max_steps):
            action, _, _ = self.agent.select_action(obs, deterministic=True)
            obs, _, done, _ = canonical.step(action)
            trajectory.append(canonical.ghost_sample())
            if done:
                break

        aggregate = aggregate_evaluations(results, self.spec.metric_mode)
        return {
            "reward": aggregate["reward_mean"],
            "reward_std": aggregate["reward_std"],
            "metric": aggregate["metric"],
            "metric_std": aggregate["metric_std"],
            "failure_progress": aggregate["failure_progress"],
            "episodes": aggregate["episodes"],
            "success_rate": aggregate["success_rate"],
            "success_ci_low": aggregate["success_ci_low"],
            "success_ci_high": aggregate["success_ci_high"],
            "evaluation_suite": evaluation_suite_id(self.settings.eval_episodes),
            "seed": self.seed,
            "trajectory": trajectory,
        }

    # ----------------------------------------------------------------- emits

    def _emit_frame(self, *, terminal: bool = False) -> None:
        terminal_summary = self.env.episode_summary() if terminal else {}
        self.emit({
            "type": "frame",
            "scenario_id": self.spec.id,
            "episode": self.episode,
            "episode_reward": round(self.env.episode_reward, 1),
            "terminal": terminal,
            "cause": terminal_summary.get("cause"),
            "terminal_steps": terminal_summary.get("steps"),
            "learning": self._learning,
            **self.env.frame_payload(),
        })

    def _emit_status(self) -> None:
        self.emit(self.status())

    # ----------------------------------------------------------------- misc

    def _restore_latest(self) -> None:
        for meta in reversed(self.registry.list()):
            latest = meta["episode"]
            try:
                data = self.registry.load_into(latest, self.agent)
                self.history = data.get("history", [])
                self.episode = latest
                stored_steps = data.get("total_steps")
                self.total_steps = int(
                    stored_steps if stored_steps is not None
                    else sum(h.get("steps", 0) for h in self.history))
                self.update_count = int(data.get("update_count", 0))
                if data.get("meta", {}).get("seed") is not None:
                    self.seed = int(data["meta"]["seed"])
                    seed_everything(self.seed)
                if data.get("rng_state"):
                    restore_rng_state(data["rng_state"], self.env)
                self.latest_update_metrics = data.get("meta", {}).get(
                    "training_diagnostics")
                self._recompute_bests()
                log.info("[%s] restored checkpoint ep%d", self.spec.id, latest)
                return
            except Exception:
                log.exception("[%s] quarantining invalid checkpoint ep%d",
                              self.spec.id, latest)
                self.registry.quarantine_episode(latest)

    def _recompute_bests(self) -> None:
        rewards = [h["reward"] for h in self.history]
        self.best_reward = max(rewards) if rewards else None
        self.best_metric = None
        for h in self.history:
            # Legacy histories used "best_lap" before metrics were generalized.
            self._update_best_metric(h.get("metric", h.get("best_lap")))


def decimate(history: list[dict], max_points: int = 2000) -> list[dict]:
    if len(history) <= max_points:
        return history
    stride = len(history) / max_points
    return [history[int(i * stride)] for i in range(max_points)]
