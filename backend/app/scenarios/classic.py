"""Classic-control scenarios rendered with the generic primitive renderer."""
from __future__ import annotations

from ..envs import cartpole, drone, lander, mountain_car, pendulum
from ..envs import lander_demonstrations
from ..ppo.demonstrations import BehaviorCloningWarmStart
from ..ppo.initialization import ActorInitialization
from ..ppo.classic_warmstarts import MOUNTAIN_CAR_ACTOR_INITIALIZATION, MOUNTAIN_CAR_ACTOR_WARM_START
from .spec import ScenarioSpec


LANDER_ACTOR_INITIALIZATION = ActorInitialization(
    scope="lunar_lander_only",
    continuous_action_labels=(
        "main_engine_throttle",
        "side_thruster_command",
    ),
    continuous_action_prior=(0.0, 0.0),
    continuous_log_std=(-1.2, -1.2),
)

LANDER_ACTOR_WARM_START = BehaviorCloningWarmStart(
    id="lander-physics-demonstrations-v1",
    expert_id="lander-braking-envelope-pd-v1",
    expert_description=(
        "physics feedback tracks a 60-unit/s cruise and 10-unit/s² braking "
        "envelope toward 5-unit/s touchdown, with envelope feed-forward and "
        "lateral/attitude PD control; generates training targets only and "
        "is absent at inference"),
    dataset_seed_base=lander_demonstrations.DATASET_SEED_BASE,
    dataset_episodes=lander_demonstrations.DATASET_EPISODES,
    dataset_start_description=(
        "80 canonical full-height starts and 20 at each of four altitude "
        "frontiers, alternating canonical and curriculum starts; fixed sample "
        "permutation with dataset seed, disjoint from selection and holdout"),
    dataset_builder=lander_demonstrations.behavior_cloning_dataset,
    continuous_action_labels=("main_engine_throttle", "side_thruster_command"),
)

DRONE_ACTOR_INITIALIZATION = ActorInitialization(
    scope="drone_hover_only",
    continuous_action_labels=("left_rotor_thrust", "right_rotor_thrust"),
    continuous_action_prior=(drone.HOVER_ACTION, drone.HOVER_ACTION),
    continuous_log_std=(-2.0, -2.0),
)

DRONE_ACTOR_WARM_START = BehaviorCloningWarmStart(
    id="drone-physics-demonstrations-v1",
    expert_id="drone-physics-pd-v1",
    expert_description=(
        "bounded physics PD controller: 0.5-second desired-velocity response, "
        "50 unit/s² acceleration cap, desired tilt atan2(ax, gravity-ay) "
        "clipped to ±1 radian, and angular gains Kp=25/Kd=7; it generates "
        "training targets only and is absent at inference"),
    dataset_seed_base=600_000,
    dataset_episodes=80,
    dataset_start_description=(
        "ordinary jittered canonical full-course starts, disjoint from fixed "
        "selection and holdout seeds"),
    dataset_builder=drone.behavior_cloning_dataset,
)


CLASSIC_SPECS: list[ScenarioSpec] = [
    ScenarioSpec(
        id="lunar-lander", name="Lunar Lander", group="Classic",
        kind="generic",
        description="Main + side thrusters, limited fuel. Touch down softly on the pad.",
        metric_label="landing error", metric_mode="min",
        make_env=lambda jitter: lander.LanderEnv(jitter=jitter),
        scene=lander.scene,
        training_factory=lambda: lander.LanderEnv(
            jitter=True, approach_curriculum=True),
        training_start_distribution=lander.TRAINING_START_DISTRIBUTION,
        training_curriculum=lander.TRAINING_CURRICULUM,
        objective="Reach the landing pad with low velocity, low tilt, and minimal fuel use.",
        success="Touch the pad below all horizontal, vertical, and tilt limits.",
        observations=("pad-relative position", "linear velocity", "tilt and angular rate", "fuel"),
        observation_dimensions=("horizontal pad offset / 300", "vertical pad offset / 300",
                                "horizontal velocity / 60", "vertical velocity / 60",
                                "sin tilt", "cos tilt", "angular rate / 3", "fuel fraction",
                                "remaining horizon fraction"),
        actions=("main-engine throttle", "signed side-thruster command"),
        reward_terms=(
            "potential shaping toward the pad and a 6–18 unit/s safe descent envelope",
            "terminal potential is zero (policy-invariant at γ = 1)",
            "−0.03 × main-engine throttle",
            "+100 soft landing / −100 crash, out-of-bounds, or timeout",
        ),
        termination_conditions=("terrain contact", "leaving the arena", "24-second horizon"),
        difficulty="Advanced", horizon_steps=lander.LanderEnv.max_steps,
        horizon_seconds=lander.LanderEnv.max_steps * lander.LanderEnv.dt,
        actor_initialization=LANDER_ACTOR_INITIALIZATION,
        actor_warm_start=LANDER_ACTOR_WARM_START,
        training_discount_factor=1.0,
        checkpoint_schema=12),
    ScenarioSpec(
        id="pendulum-swingup", name="Pendulum Swing-Up", group="Classic",
        kind="generic",
        description="One torque motor. Swing the pendulum up and hold it there.",
        metric_label="balance", metric_mode="max",
        make_env=lambda jitter: pendulum.PendulumEnv(jitter=jitter),
        scene=pendulum.scene,
        objective="Minimize angle, angular speed, and control effort around upright.",
        success="Hold near upright through the final quarter of the episode.",
        observations=("cosine and sine of angle", "angular velocity"),
        observation_dimensions=("cos angle", "sin angle", "angular velocity / 8",
                                "remaining horizon fraction"),
        actions=("signed motor torque",),
        reward_terms=("−angle²", "−0.1 × angular-speed²", "−0.001 × torque²"),
        termination_conditions=("fixed 16-second horizon",),
        difficulty="Intermediate", horizon_steps=pendulum.PendulumEnv.max_steps,
        horizon_seconds=pendulum.PendulumEnv.max_steps * pendulum.PendulumEnv.dt,
        checkpoint_schema=3),
    ScenarioSpec(
        id="drone-hover", name="Drone Course", group="Classic",
        kind="generic",
        description="Two rotors, five waypoints. Fly the course without tipping over.",
        metric_label="waypoints", metric_mode="max",
        make_env=lambda jitter: drone.DroneEnv(jitter=jitter),
        scene=drone.scene,
        training_factory=lambda: drone.DroneEnv(
            jitter=True, episode_schedule=True),
        training_start_distribution=drone.TRAINING_START_DISTRIBUTION,
        training_schedule=drone.TRAINING_SCHEDULE,
        objective="Capture all waypoints while controlling attitude and energy use.",
        success="Capture all five waypoints without tipping or leaving the arena.",
        observations=("target-relative position", "linear velocity", "tilt and angular rate", "course progress"),
        observation_dimensions=("target horizontal offset / 300", "target vertical offset / 300",
                                "horizontal velocity / 100", "vertical velocity / 100",
                                "desired horizontal velocity error / 100",
                                "desired vertical velocity error / 100",
                                "desired tilt error / 1.3",
                                "sin tilt", "cos tilt", "angular rate / 4", "waypoint fraction",
                                "remaining horizon fraction"),
        actions=("left-rotor thrust", "right-rotor thrust"),
        reward_terms=(
            "segment-local progress auxiliary: change in −(0.05 distance "
            "+ 0.10 desired-velocity error + 5.0 desired-tilt error)",
            "20–80 unit/s target speed from a 20 unit/s² braking envelope",
            "waypoint captures reset the auxiliary baseline without a "
            "cross-target jump; terminal segment cost is retained on failure",
            "+20 per waypoint and +50 course completion",
            "angular-rate and squared-thrust regularizers charged only on "
            "successful course completion",
            "−50 crash or timeout",
        ),
        termination_conditions=("all waypoints captured", "tip or arena exit", "36-second horizon"),
        difficulty="Advanced", horizon_steps=drone.DroneEnv.max_steps,
        horizon_seconds=drone.DroneEnv.max_steps * drone.DroneEnv.dt,
        training_discount_factor=1.0,
        actor_initialization=DRONE_ACTOR_INITIALIZATION,
        actor_warm_start=DRONE_ACTOR_WARM_START,
        checkpoint_schema=18),
    ScenarioSpec(
        id="cartpole-balance", name="Continuous Cart-Pole", group="Foundations",
        kind="generic",
        description="A disclosed continuous-force, semi-implicit CartPole variant; not Gymnasium CartPole-v1.",
        metric_label="balance time", metric_mode="max",
        make_env=lambda jitter: cartpole.CartPoleEnv(jitter=jitter),
        scene=cartpole.scene,
        objective="Maximize the time the pole remains upright and the cart remains in bounds.",
        success="Balance for the full 10-second horizon.",
        observations=("cart position", "cart velocity", "pole angle", "pole angular velocity"),
        observation_dimensions=("cart position / 2.4", "cart velocity / 3",
                                "pole angle / 12°", "pole angular velocity / 3.5",
                                "remaining horizon fraction"),
        actions=("signed horizontal force",),
        reward_terms=("+1 for every safe control step", "0 on the failure transition"),
        termination_conditions=("cart leaves ±2.4 m", "pole exceeds ±12°", "500-step horizon"),
        difficulty="Introductory", horizon_steps=cartpole.CartPoleEnv.max_steps,
        horizon_seconds=cartpole.CartPoleEnv.max_steps * cartpole.CartPoleEnv.dt,
        checkpoint_schema=2),
    ScenarioSpec(
        id="mountain-car", name="Mountain Car", group="Foundations",
        kind="generic",
        description="Build momentum across a valley to reach a goal the motor cannot climb directly.",
        metric_label="control effort", metric_mode="min",
        make_env=lambda jitter: mountain_car.MountainCarEnv(jitter=jitter),
        scene=mountain_car.scene,
        objective="Reach the summit while minimizing squared control effort.",
        success="Cross the goal flag before the 999-step horizon.",
        observations=("normalized position", "normalized velocity"),
        observation_dimensions=("position mapped to [−1, +1]", "velocity / 0.07",
                                "remaining horizon fraction"),
        actions=("signed motor force",),
        reward_terms=("+100 on reaching the summit", "−0.1 × squared motor command per step"),
        termination_conditions=("position reaches 0.45", "999-control-step horizon"),
        difficulty="Introductory", horizon_steps=mountain_car.MountainCarEnv.max_steps,
        horizon_seconds=None,
        actor_initialization=MOUNTAIN_CAR_ACTOR_INITIALIZATION,
        actor_warm_start=MOUNTAIN_CAR_ACTOR_WARM_START,
        checkpoint_schema=4),
]
