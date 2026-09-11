"""Compare predefined controllers without starting training or writing checkpoints.

These diagnostics use the same start distribution as learned-policy evaluation.
Repeated inspection makes them unsuitable as untouched holdout evidence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from app.evaluation import compare_controllers
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec
from app.trainer import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="robot-reach,ball-beam,robot-tracking,orbital-docking")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=4_200_000)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for scenario_id in args.scenarios.split(","):
        spec = get_spec(scenario_id.strip())
        if spec.actor_warm_start is not None:
            raise ValueError("this baseline tool does not apply demonstration warm starts")
        env = spec.make_env(False)
        torch.manual_seed(args.policy_seed)
        agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"),
                         actor_initialization=spec.actor_initialization)
        result = compare_controllers(spec, agent, episodes=args.episodes, seed_base=args.seed_base)
        result["policy_role"] = "untrained_actor"
        results.append(result)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "engine_source_sha256": source_digest(), "torch_version": torch.__version__,
              "policy_initialization_seed": args.policy_seed, "training_steps": 0,
              "role": "predefined_controller_diagnostics", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps([{ "scenario": r["scenario_id"], "controllers": [
        {"name": c["controller"], "successes": c["successes"], "episodes": c["episodes"], "metric": c["metric"]}
        for c in r["results"]]} for r in results]))


if __name__ == "__main__":
    main()
