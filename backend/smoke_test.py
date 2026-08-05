"""Backend smoke test across every scenario (no server needed):
env protocol conformance, a random-policy episode, one tiny PPO rollout +
update, checkpoint save/load round-trip, plus targeted physics invariants."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from app import physics  # noqa: E402
from app.checkpoints import CheckpointRegistry, migrate_flat_layout  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.ppo.buffer import RolloutBuffer  # noqa: E402
from app.scenarios import list_specs  # noqa: E402

DEVICE = torch.device("cpu")


def exercise_scenario(spec) -> None:
    env = spec.make_env(True)
    obs = env.reset()
    assert obs.shape == (env.obs_dim,), (spec.id, obs.shape)
    assert env.n_continuous >= 1 and env.n_binary >= 0

    rng = np.random.default_rng(0)
    act_dim = env.n_continuous + env.n_binary
    for _ in range(200):
        obs, reward, done, _ = env.step(rng.uniform(-1, 1, act_dim))
        assert np.all(np.isfinite(obs)), spec.id
        assert np.isfinite(reward), spec.id
        if done:
            summary = env.episode_summary()
            assert {"reward", "steps", "cause", "metric"} <= set(summary), spec.id
            obs = env.reset()
    ghost = env.ghost_sample()
    assert len(ghost) == 5, spec.id
    payload = env.frame_payload()
    assert ("car" in payload) or ("objects" in payload), spec.id

    scene = spec.scene()
    assert scene["kind"] in ("track", "generic"), spec.id

    agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary, DEVICE)
    buffer = RolloutBuffer(256, env.obs_dim, act_dim)
    obs = env.reset()
    done = False
    while not buffer.full:
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, log_prob, reward, done, value)
        obs = env.reset() if done else next_obs
    buffer.compute_gae(0.0 if done else agent.get_value(obs), done)
    metrics = agent.update(buffer)
    assert all(np.isfinite(v) for v in metrics.values()), (spec.id, metrics)

    with tempfile.TemporaryDirectory() as tmp:
        registry = CheckpointRegistry(Path(tmp), spec.id)
        registry.save(10, agent, [{"episode": 10, "reward": 1.0, "steps": 5}],
                      {"reward": 1.0, "metric": 2.0, "trajectory": [ghost]})
        assert registry.list()[0]["eval_metric"] == 2.0
        agent2 = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary, DEVICE)
        registry.load_into(10, agent2)
        p1 = next(iter(agent.network.parameters()))
        p2 = next(iter(agent2.network.parameters()))
        assert torch.equal(p1, p2)
    print(f"  {spec.id}: ok ({env.obs_dim} obs, {act_dim} act, kind={spec.kind})")


def targeted_invariants() -> None:
    from app.envs.drone import DroneEnv
    from app.envs.lander import LanderEnv
    from app.scenarios.registry import SCENARIOS

    # F1 straight-line terminal speed.
    car = physics.CarState(0, 0, 0, 0, 0, 0, 0)
    for _ in range(1000):
        car = physics.step(car, 1.0, 0.0, False)
    assert 70 < car.v_long <= 80, car.v_long

    # Drift slides far more than grip for the same inputs.
    drifty = physics.CarState(0, 0, 0, 40, 0, 0, 0)
    grippy = physics.CarState(0, 0, 0, 40, 0, 0, 0)
    for _ in range(50):
        drifty = physics.step(drifty, 0.3, 1.0, True)
        grippy = physics.step(grippy, 0.3, 1.0, False)
    assert abs(drifty.slip_angle) > abs(grippy.slip_angle) * 2

    # Ice: scaled-down yaw authority means heavy understeer without drift…
    icy = physics.CarState(0, 0, 0, 40, 0, 0, 0)
    for _ in range(50):
        icy = physics.step(icy, 0.3, 1.0, False, grip_scale=0.25)
    assert abs(icy.heading) < abs(grippy.heading) * 0.5, \
        (icy.heading, grippy.heading)
    # …and the drift button is the only way to rotate-and-slide on ice.
    icy_drift = physics.CarState(0, 0, 0, 40, 0, 0, 0)
    for _ in range(50):
        icy_drift = physics.step(icy_drift, 0.3, 1.0, True, grip_scale=0.25)
    assert abs(icy_drift.slip_angle) > abs(icy.slip_angle) * 2
    assert abs(icy_drift.heading) > abs(icy.heading)

    # Kart is genuinely slower than the F1 car.
    kart = physics.CarState(0, 0, 0, 0, 0, 0, 0)
    for _ in range(1000):
        kart = physics.step(kart, 1.0, 0.0, False, physics.KART)
    assert kart.v_long <= physics.KART.max_speed < 40

    # Lander free-falls without thrust; full burn climbs.
    env = LanderEnv(jitter=False)
    y0 = env.y
    for _ in range(25):
        env.step(np.array([-1.0, 0.0]))  # main off
    assert env.y > y0 + 10, "lander should fall under gravity"
    env.reset()
    for _ in range(25):
        env.step(np.array([1.0, 0.0]))   # full main burn
    assert env.vy < 0, "lander should accelerate upward at full burn"

    # Drone climbs at full symmetric thrust.
    d = DroneEnv(jitter=False)
    for _ in range(25):
        d.step(np.array([1.0, 1.0]))
    assert d.vy < 0, "drone should climb at full thrust"

    # Eco fuel drains to zero under sustained full throttle.
    eco = SCENARIOS["eco-gp"].make_env(False)
    for _ in range(eco.max_steps):
        _, _, done, _ = eco.step(np.array([1.0, 0.0, 0.0]))
        if eco.fuel == 0.0:
            break
    assert eco.fuel == 0.0, "eco tank never ran dry"

    # Traffic overtake bonus fires when the car passes a bot. Drive the env's
    # own bookkeeping: bot 0 sits just ahead, bots 1-2 parked far away.
    traffic = SCENARIOS["traffic-rush"].make_env(False)
    far = traffic.s_prev + traffic.track.total_length * 0.5
    traffic._bot_arcs = [traffic.s_prev + 10.0, far, far + 50.0]
    traffic._bot_passed = [False, False, False]
    traffic._bot_prev_gap = [traffic._bot_gap(i) for i in range(3)]
    traffic.s_prev += 25.0  # car arc jumps past bot 0
    gained = traffic._step_bots()
    assert gained > 0, "overtake bonus did not fire"

    # Flat-layout checkpoint migration.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "checkpoint_ep000050.pt").write_bytes(b"x")
        (root / "checkpoint_ep000050.json").write_text("{}")
        migrate_flat_layout(root)
        assert (root / "apex-gp" / "checkpoint_ep000050.pt").exists()
        assert not list(root.glob("checkpoint_ep*"))

    print("  targeted invariants: ok")


def main() -> None:
    specs = list_specs()
    assert len(specs) == 16, len(specs)
    print(f"exercising {len(specs)} scenarios:")
    for spec in specs:
        exercise_scenario(spec)
    targeted_invariants()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
