"""Offline arcade validation; runs only against an explicitly supplied empty directory.

Example: python benchmark_arcade.py --output /tmp/arcade-validation
No live API or active checkpoint volume is consulted.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from app.ppo.agent import PPOAgent
from app.scenarios.arcade import ARCADE_SPECS
from app.scenarios.registry import SCENARIOS
from app.settings import Settings
from app.trainer import Trainer, source_digest, aggregate_evaluations


def evaluate(spec, agent=None, episodes=50, seed_base=950_000, controller="policy"):
    trials = []
    for index in range(episodes):
        env = spec.make_env(True)
        env.rng.seed(seed_base+index)
        observation = env.reset()
        rng = np.random.default_rng(seed_base+index)
        for _ in range(env.max_steps):
            if controller == "reference":
                action = env.reference_action()
            elif controller == "zero":
                action = np.zeros(env.n_continuous)
            elif controller == "random":
                action = rng.uniform(-1, 1, env.n_continuous)
            else:
                action = agent.predict(observation)
            observation, _, done, _ = env.step(action)
            if done:
                break
        trials.append({"seed": seed_base+index, **env.episode_summary()})
    return {**aggregate_evaluations(trials, spec.metric_mode),
            "seed_base": seed_base, "successes": sum(t["success"] for t in trials), "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--scenarios", nargs="+", default=[s.id for s in ARCADE_SPECS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty; refusing to modify existing checkpoints")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    SCENARIOS.update({spec.id: spec for spec in ARCADE_SPECS})
    report = {"protocol": "arcade-full-game-ppo-v1", "engine_source_sha256": source_digest(),
              "selection": "Earliest scheduled checkpoint with fixed-suite success >=90% of10; independent50-seed holdout only after selection.",
              "holdout_seed_base": 950_000, "budget": args.episodes,
              "warm_start_disclosure": "Expert BC plus two DAgger rounds, then PPO. Learned actor only at inference; these are not from-scratch PPO runs.",
              "runs": [], "baselines": []}
    def flush():
        (args.output/"report.json").write_text(json.dumps(report, indent=2))
    for spec in ARCADE_SPECS:
        if spec.id not in args.scenarios:
            continue
        baselines = {"scenario_id": spec.id}
        for controller in ("reference", "zero", "random"):
            baselines[controller] = evaluate(spec, controller=controller, seed_base=830_000)
        report["baselines"].append(baselines)
        for seed in args.seeds:
            started = time.monotonic()
            run_dir = args.output/f"{spec.id}-seed{seed}"
            run_dir.mkdir()
            (run_dir/"state.json").write_text(json.dumps({"active_scenario": spec.id}))
            trainer = Trainer(Settings(port=0, checkpoint_dir=run_dir,
                              checkpoint_every_n=50, max_episodes=args.episodes,
                              use_gpu=False, seed=seed, eval_episodes=10, cpu_threads=1))
            warm_diagnostic = evaluate(spec, trainer.agent, seed_base=820_000)
            print(json.dumps({"scenario": spec.id, "seed": seed, "warm_successes": warm_diagnostic["successes"]}), flush=True)
            def emit(event):
                if event.get("type") == "checkpoint_list":
                    rows = event.get("checkpoints", [])
                    print(json.dumps({"scenario": spec.id, "seed": seed,
                                      "checkpoints": [{"episode": x["episode"], "success_rate": x.get("success_rate")} for x in rows]}), flush=True)
            trainer.emit = emit
            trainer._run_loop()
            checkpoints = sorted(trainer.registry.list(), key=lambda row: row["episode"])
            eligible = [row for row in checkpoints if row.get("success_rate", 0) >= .9]
            selected = eligible[0] if eligible else None
            result = {"scenario_id": spec.id, "seed": seed, "warm_diagnostic": warm_diagnostic,
                      "warm_start": trainer.actor_warm_start_diagnostics,
                      "trace": [{key: row.get(key) for key in ("episode", "total_steps", "update_count", "success_rate", "eval_reward")} for row in checkpoints],
                      "selected": selected, "duration_seconds": time.monotonic()-started}
            if selected:
                data = trainer.registry.load(selected["episode"])
                trainer.agent.load_state_dict(data["agent"])
                result["holdout"] = evaluate(spec, trainer.agent)
                canonical = spec.make_env(False)
                obs = canonical.reset()
                for _ in range(canonical.max_steps):
                    obs, _, done, _ = canonical.step(trainer.agent.predict(obs))
                    if done:
                        break
                result["canonical"] = canonical.episode_summary()
            report["runs"].append(result)
            flush()
            print(json.dumps({"scenario": spec.id, "seed": seed, "selected_episode": selected["episode"] if selected else None,
                              "holdout_successes": result.get("holdout", {}).get("successes"), "canonical": result.get("canonical")}), flush=True)
    flush()


if __name__ == "__main__":
    main()
