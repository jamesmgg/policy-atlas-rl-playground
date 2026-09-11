"""Explicit contracts for robotics and orbital-control experiments."""
from ..envs import ball_beam, orbital, robot_arm
from ..ppo.control_demonstrations import warm_start
from ..ppo.ballbeam_demonstrations import BALLBEAM_WARM_START
from ..ppo.initialization import ActorInitialization
from .spec import ScenarioSpec


def robot_spec(tracking):
    return ScenarioSpec(
        id="robot-tracking" if tracking else "robot-reach",
        name="Robot Target Tracking" if tracking else "Robot Arm Reach",
        group="Robotics", kind="generic",
        description=("Follow an orbiting target with two damped servo joints." if tracking
                     else "Coordinate two servo joints to reach and settle on a target."),
        metric_label="tracking error" if tracking else "reach error", metric_mode="min",
        make_env=lambda jitter: robot_arm.RobotArmEnv(jitter=jitter, tracking=tracking),
        scene=robot_arm.scene,
        objective="Minimize endpoint error, relative endpoint speed, and actuator effort.",
        success=("Stay within 0.12 m of the moving target for at least 90 of the final 100 steps."
                 if tracking else "Hold within 0.08 m at less than 0.1 m/s for one second."),
        observations=("joint angles and velocities", "target position and velocity", "endpoint error", "success progress and clock"),
        observation_dimensions=("cos shoulder", "sin shoulder", "cos elbow", "sin elbow",
                                "shoulder velocity / 3", "elbow velocity / 3",
                                "target x / 1.75", "target y / 1.75", "target velocity x", "target velocity y",
                                "tip error x / 1.75", "tip error y / 1.75", "success progress", "remaining horizon fraction"),
        actions=("shoulder acceleration command", "elbow acceleration command"),
        reward_terms=("−endpoint distance²", "−0.025 × relative tip speed²", "−0.005 × action norm²", "+30 success"),
        termination_conditions=("tracking horizon" if tracking else "one-second stable reach", "15-second horizon"),
        training_start_distribution=("seeded joint perturbations and reachable target configurations; no demonstrations" if tracking
                                     else "unchanged full-task starts after disclosed reference demonstrations and DAgger initialization"),
        actor_warm_start=None if tracking else warm_start("robot-reach"),
        actor_initialization=None if tracking else ActorInitialization(
            scope="robot-reach-only", continuous_action_labels=("shoulder", "elbow"),
            continuous_action_prior=(0.0, 0.0), continuous_log_std=(-2.5, -2.5)),
        checkpoint_schema=1 if tracking else 3,
        training_learning_rate=None if tracking else 3e-5,
        horizon_steps=300, horizon_seconds=15, difficulty="Advanced" if tracking else "Intermediate",
        training_discount_factor=0.995,
        reference_controller="Inverse kinematics with PD joint control and target-velocity feedforward",
        model_assumptions=("Planar links: 1 m and 0.75 m.", "Ideal independent damped acceleration servos; no gravity, joint coupling, or contacts."))


CONTROL_SPECS = [
    ScenarioSpec(
        id="orbital-docking", name="Orbital Docking", group="Space", kind="generic",
        description="Brake a spacecraft into a stable rendezvous under coupled orbital motion.",
        metric_label="docking error", metric_mode="min",
        make_env=lambda jitter: orbital.OrbitalEnv(jitter=jitter), scene=orbital.scene,
        objective="Reach a circular-orbit reference craft with small position and velocity errors.",
        success="Remain within 2 m and below 0.08 m/s for 10 seconds.",
        observations=("radial and along-track position", "relative velocity", "capture dwell and clock"),
        observation_dimensions=("radial offset / 100", "along-track offset / 100", "radial velocity / 2",
                                "along-track velocity / 2", "capture dwell fraction", "remaining horizon fraction"),
        actions=("radial thruster acceleration", "along-track thruster acceleration"),
        reward_terms=("−0.05 per step", "−(distance / 60)²", "−0.2 × speed²", "−0.005 × action norm²", "+100 docking / −600 escape"),
        termination_conditions=("10-second stable rendezvous", "140 m escape boundary", "300-second horizon"),
        training_start_distribution="radial 30–65 m, along-track ±35 m, each velocity ±0.1 m/s; disclosed reference demonstrations and DAgger initialization",
        actor_warm_start=warm_start("orbital-docking"),
        actor_initialization=ActorInitialization(
            scope="orbital-docking-only", continuous_action_labels=("radial", "along_track"),
            continuous_action_prior=(0.0, 0.0), continuous_log_std=(-2.5, -2.5)),
        checkpoint_schema=2,
        difficulty="Advanced", horizon_steps=600, horizon_seconds=300,
        training_discount_factor=1.0,
        reference_controller="PD rendezvous control with cancellation of the CW orbital terms",
        model_assumptions=("Linear planar Hill/Clohessy-Wiltshire dynamics near a circular orbit.",
                           "Mean motion 0.0011 rad/s; bounded acceleration 0.04 m/s²; RK4 at 0.5 s.",
                           "No attitude, contact, eccentricity, or orbital perturbation model.")),
    robot_spec(False), robot_spec(True),
    ScenarioSpec(
        id="ball-beam", name="Ball & Beam", group="Robotics", kind="generic",
        description="Tilt the beam, brake the rolling ball, and hold it on a target.",
        metric_label="position error", metric_mode="min",
        make_env=lambda jitter: ball_beam.BallBeamEnv(jitter=jitter), scene=ball_beam.scene,
        objective="Settle a rolling solid ball while minimizing tracking error and control effort.",
        success="Hold within 4.5 cm, below 0.06 m/s, and within 0.035 rad tilt for two seconds.",
        observations=("ball position and velocity", "beam angle and angular rate", "target", "settling dwell and clock"),
        observation_dimensions=("ball position", "ball velocity / 2", "beam tilt / 0.3", "beam angular rate / 1.2",
                                "target position", "settling dwell fraction", "remaining horizon fraction"),
        actions=("beam angular acceleration command",),
        reward_terms=("−2 × position error²", "−0.2 × velocity²", "−0.03 × tilt²", "−0.001 × command²", "+30 settled / −200 fell"),
        termination_conditions=("two-second settling dwell", "ball leaves ±1 m beam", "20-second horizon"),
        training_start_distribution="unchanged ball starts within ±0.65 m, target within ±0.3 m, level beam; disclosed reference demonstrations initialize the actor before PPO",
        actor_warm_start=BALLBEAM_WARM_START,
        actor_initialization=ActorInitialization(
            scope="ball-beam-only", continuous_action_labels=("beam_angular_acceleration",),
            continuous_action_prior=(0.0,), continuous_log_std=(-2.5,)),
        checkpoint_schema=2,
        difficulty="Intermediate", horizon_steps=500, horizon_seconds=20,
        reference_controller="Cascaded PD control of ball position and beam tilt",
        model_assumptions=("Solid ball rolling without slip: acceleration = (5/7) g sin(tilt), with linear drag.",
                           "Reduced model omits beam rotational inertial coupling and motor electrical dynamics.")),
]
