"""Read-only, paired-start diagnostics; these repeatable probes are not holdouts."""
from functools import lru_cache
import numpy as np

from .ppo.agent import deterministic_action
from .scenarios import get_spec
from .trainer import aggregate_evaluations

DIAGNOSTIC_SEED_BASE = 3_000_000


def compare_controllers(spec, agent, *, episodes=10, seed_base=DIAGNOSTIC_SEED_BASE):
    if not 1 <= episodes <= 50:
        raise ValueError("diagnostic evaluation requires 1–50 episodes")
    if not 0 <= seed_base <= 2**32-episodes:
        raise ValueError("invalid diagnostic seed range")
    names = ["policy", "zero", "random"]
    if callable(getattr(spec.make_env(False), "reference_action", None)):
        names.append("reference")
    results = []
    for name in names:
        trials = []
        for index in range(episodes):
            seed = seed_base+index
            env = spec.make_env(True)
            env.rng.seed(seed)
            obs = env.reset()
            action_rng = np.random.default_rng(seed+1_000_000)
            for _ in range(env.max_steps):
                if name == "policy":
                    action = deterministic_action(agent, obs)
                elif name == "reference":
                    action = env.reference_action()
                elif name == "zero":
                    action = np.zeros(env.n_continuous+env.n_binary, np.float32)
                else:
                    action = np.concatenate((action_rng.uniform(-1, 1, env.n_continuous),
                                             action_rng.integers(0, 2, env.n_binary))).astype(np.float32)
                obs, _, done, _ = env.step(action)
                if done:
                    break
            trials.append({"seed": seed, **env.episode_summary()})
        results.append({"controller": name, **aggregate_evaluations(trials, spec.metric_mode),
                        "successes": sum(t["success"] for t in trials),
                        "metric_samples": sum(t.get("metric") is not None for t in trials),
                        "trials": trials})
    return {"scenario_id": spec.id, "seed_base": seed_base, "episodes": episodes,
            "suite": "paired-controller-diagnostic-v1", "is_holdout": False,
            "metric_label": spec.metric_label, "results": results}


def reference_replay(spec):
    env = spec.make_env(False)
    control = getattr(env, "reference_action", None)
    if not callable(control):
        raise ValueError("this experiment does not define a reference controller")
    frames = []
    for _ in range(env.max_steps):
        _, _, done, _ = env.step(control())
        frames.append({"type": "frame", "scenario_id": spec.id, "episode": 0,
                       "episode_reward": env.episode_reward, "terminal": done,
                       "cause": env.episode_summary()["cause"], "terminal_steps": env.steps,
                       **env.frame_payload()})
        if done:
            break
    return {"scenario_id": spec.id, "controller": "reference", "dt": env.dt,
            "frames": frames, "summary": env.episode_summary()}


@lru_cache(maxsize=32)
def cached_reference_replay(scenario_id):
    return reference_replay(get_spec(scenario_id))
