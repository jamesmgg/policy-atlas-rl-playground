"""Small arcade games with explicit action, success and initialization contracts."""
from ..envs.arcade import (
    PaddleRallyEnv, FlappyFlightEnv, CoinCollectorEnv,
    paddle_scene, flappy_scene, coin_scene,
)
from .spec import ScenarioSpec
from ..ppo.arcade_demonstrations import warm_start
from ..ppo.initialization import ActorInitialization


def initialization(labels):
    return ActorInitialization(scope="arcade_demonstration_assisted_ppo",
                               continuous_action_labels=labels,
                               continuous_action_prior=(0.,)*len(labels),
                               continuous_log_std=(-2.,)*len(labels))


ARCADE_SPECS = [
    ScenarioSpec(
        id="paddle-rally", name="Paddle Rally", group="Arcade", kind="generic",
        description="Chase ricochets and return five balls without a single miss.",
        metric_label="returns", metric_mode="max", make_env=lambda jitter: PaddleRallyEnv(jitter), scene=paddle_scene,
        objective="Move the paddle under the ball's wall-reflected landing point and return five shots.",
        success="Return five consecutive balls before the 18-second deadline; any miss ends the game.",
        observations=("paddle and ball state", "wall-reflected landing sensor", "returns and clock"),
        observation_dimensions=("paddle x", "ball x", "ball height / 1.5", "ball vx / 1.2", "ball vy / 1.8", "predicted landing x", "landing x minus paddle x", "returns / 5", "remaining horizon fraction"),
        actions=("horizontal paddle velocity",),
        reward_terms=("+5 per return", "+30 five returns", "−30 missed ball", "−0.015 time cost", "−0.03 × landing error²", "−15 deadline"),
        termination_conditions=("five returns", "missed ball", "18-second deadline"),
        training_start_distribution="seeded full games with varied paddle and ball starts and independently sampled rebound velocities",
        horizon_steps=450, horizon_seconds=18, difficulty="Beginner",
        actor_initialization=initialization(("paddle velocity",)),
        actor_warm_start=warm_start("paddle-rally", ("paddle velocity",)),
        reference_controller="Proportional paddle control using the observed wall-reflected intercept",
        model_assumptions=("Ideal elastic side/top walls; no spin.", "The exact reflected landing point is an engineered observation, available equally to the learned actor and reference controller.")),
    ScenarioSpec(
        id="flappy-flight", name="Flappy Flight", group="Arcade", kind="generic",
        description="Modulate wing thrust and thread six changing pipe gaps.",
        metric_label="gates", metric_mode="max", make_env=lambda jitter: FlappyFlightEnv(jitter), scene=flappy_scene,
        objective="Fly through all six gates without touching a pipe, ceiling or floor.",
        success="Clear six complete pipe obstacles in one flight before the 15.2-second deadline.",
        observations=("bird height and vertical speed", "two visible pipe gaps and nearest distance", "gates and clock"),
        observation_dimensions=("bird height", "vertical speed / 2", "pipe distance / 1.4", "gap center", "gap minus height", "next gap center", "gates / 6", "remaining horizon fraction"),
        actions=("continuous wing thrust: negative descends, positive climbs",),
        reward_terms=("+5 per gate", "+30 six gates", "−30 collision", "−0.01 time cost", "−0.08 × height error²", "−0.002 × thrust²", "−15 deadline"),
        termination_conditions=("six gates", "pipe collision", "ceiling/floor collision", "15.2-second deadline"),
        training_start_distribution="seeded full flights with random starting height and independently sampled gap centers",
        horizon_steps=380, horizon_seconds=15.2, difficulty="Intermediate",
        actor_initialization=initialization(("wing thrust",)),
        actor_warm_start=warm_start("flappy-flight", ("wing thrust",)),
        reference_controller="PD altitude control toward the observed nearest pipe gap",
        model_assumptions=("Constant horizontal scroll at 0.8 m/s.", "Continuous wing thrust with linear vertical drag; zero command balances gravity.", "Pipe and boundary collisions include the bird radius.")),
    ScenarioSpec(
        id="coin-collector", name="Coin Collector", group="Arcade", kind="generic",
        description="Dash across the arena, brake onto five coins, and avoid the electric boundary.",
        metric_label="coins", metric_mode="max", make_env=lambda jitter: CoinCollectorEnv(jitter), scene=coin_scene,
        objective="Navigate between five separated destinations and brake to collect each coin.",
        success="Collect five coins in 24 seconds; each requires 0.12 seconds within 0.1 m below 0.2 m/s.",
        observations=("collector position and velocity", "current coin and displacement", "capture dwell, coins and clock"),
        observation_dimensions=("collector x", "collector y", "collector vx", "collector vy", "coin x", "coin y", "coin error x", "coin error y", "capture dwell / 3", "coins / 5", "remaining horizon fraction"),
        actions=("horizontal acceleration", "vertical acceleration"),
        reward_terms=("+5 per coin", "+30 five coins", "−30 boundary collision", "+2 × distance reduction", "−0.02 time cost", "−0.03 × distance²", "−0.003 × action norm²", "−15 deadline"),
        termination_conditions=("five coins", "electric boundary collision", "24-second deadline"),
        training_start_distribution="seeded central starts; each successive target lies on a 0.68 m ring at least 0.55 m from the collector",
        horizon_steps=600, horizon_seconds=24, difficulty="Intermediate",
        actor_initialization=initialization(("horizontal acceleration", "vertical acceleration")),
        actor_warm_start=warm_start("coin-collector", ("horizontal acceleration", "vertical acceleration")),
        reference_controller="PD acceleration control with velocity damping toward the current coin",
        model_assumptions=("Two independent acceleration thrusters with linear drag.", "Coin capture requires braking and dwell; simply flying through earns no collection.", "New coin positions are stochastic transitions, conditioned only on the current observed collector position.")),
]
