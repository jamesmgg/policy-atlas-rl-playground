"""The 11 driving scenarios: tracks x physics x features."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import torch

from .. import physics, track as tracks
from ..envs.base import TrainingControlSpec, TrainingCurriculumSpec
from ..envs.driving import (
    TRAFFIC_BOT3_SPEED_ACTIVE_STAGE_PROBABILITY,
    TRAFFIC_BOT3_SPEED_CONSECUTIVE_CONFIRMATIONS,
    TRAFFIC_BOT3_SPEED_CONTROL_ID,
    TRAFFIC_BOT3_SPEED_STAGE_IDS,
    TRAFFIC_BOT3_SPEED_SUCCESS_RATE_THRESHOLD,
    TRAFFIC_CURRICULUM_ACTIVE_FRONTIER_PROBABILITY,
    TRAFFIC_CURRICULUM_CONSECUTIVE_CONFIRMATIONS,
    TRAFFIC_CURRICULUM_FRONTIER_ORDER,
    TRAFFIC_CURRICULUM_ID,
    TRAFFIC_TRAINING_SCHEDULE,
    Bot,
    DrivingEnv,
    DrivingFeatures,
    FuelConfig,
    RewardConfig,
    TrafficCurriculumEnv,
    Zone,
    traffic_reference_action,
)
from ..ppo.demonstrations import BehaviorCloningWarmStart
from ..ppo.driving_warmstarts import guided_driving_spec
from ..ppo.initialization import ActorInitialization
from ..track import Track, build_track
from .spec import ScenarioSpec


DRIVING_ACTOR_INITIALIZATION = ActorInitialization(
    scope="driving_only",
    continuous_action_labels=("throttle_brake", "steering"),
    continuous_action_prior=(0.25, 0.0),
    binary_action_labels=("drift",),
    binary_probability_prior=(0.05,),
)

THUNDER_ACTOR_INITIALIZATION = ActorInitialization(
    scope="thunder_oval_only",
    continuous_action_labels=("throttle_brake", "steering"),
    continuous_action_prior=(0.0, 0.0),
    binary_action_labels=("drift",),
    binary_probability_prior=(0.05,),
)


@lru_cache(maxsize=None)
def _track(name: str) -> Track:
    return build_track(getattr(tracks, name))


def _scene(track_name: str, features: DrivingFeatures) -> dict:
    track = _track(track_name)
    zones = []
    for z in features.zones:
        zones.append({
            "start_idx": track.index_at_arc(z.start_frac * track.total_length),
            "end_idx": track.index_at_arc(z.end_frac * track.total_length),
            "grip_scale": z.grip_scale,
            "color": z.color,
        })
    return {"kind": "track", "track": track.to_dict(), "zones": zones}


def _driving_spec(id: str, name: str, group: str, description: str,
                  track_name: str,
                  params: physics.PhysicsParams = physics.F1,
                  reward: RewardConfig = RewardConfig(),
                  features: DrivingFeatures = DrivingFeatures(),
                  metric_label: str = "best lap",
                  metric_mode: str = "min",
                  objective: str = "Complete clean laps as quickly as possible.",
                  success: str = "Complete at least one timed lap without leaving the circuit.",
                  difficulty: str = "Intermediate",
                  training_rolling_checkpoints: tuple[int, ...] | None = None,
                  horizon_steps: int = DrivingEnv.max_steps,
                  actor_initialization: ActorInitialization = DRIVING_ACTOR_INITIALIZATION,
                  training_discount_factor: float = 0.995,
                  checkpoint_schema: int = 7) -> ScenarioSpec:
    if features.metric != "overtakes":
        features = replace(features, guidance_observations=True)
    if reward.terminal_zero_course_potential:
        reward_terms = [
            "terminal-zero course potential: live progress/checkpoint/lap "
            "differences are dense, terminal potential is 0, and the episode "
            "sum = -potential(start)",
            f"{reward.time:g} time cost per control step",
            f"{reward.collision:g} off-track, {reward.wrong_way:g} wrong-way, "
            f"and {reward.stall:g} stall penalties",
        ]
    elif reward.retain_terminal_course_potential:
        reward_terms = [
            "retained terminal course potential: live "
            "progress/checkpoint/lap differences are dense, terminal "
            "potential is the physical end state, and the episode sum = "
            "potential(end) - potential(start)",
            f"{reward.time:g} time cost per control step",
            f"{reward.collision:g} off-track, {reward.wrong_way:g} wrong-way, "
            f"and {reward.stall:g} stall penalties",
        ]
    else:
        reward_terms = [
            f"{reward.progress:g} × signed forward arc progress",
            f"{reward.time:g} time cost per control step",
            f"+{reward.checkpoint:g} per checkpoint and +{reward.lap:g} per lap",
            f"{reward.collision:g} off-track, {reward.wrong_way:g} wrong-way, "
            f"and {reward.stall:g} stall penalties",
        ]
    if reward.style_coef:
        reward_terms.append(
            f"up to {reward.style_coef:g} × forward distance × speed × "
            "measured slip × corner intensity")
    if reward.drift_corner:
        reward_terms.append(
            f"up to {reward.drift_corner:g} x positive arc progress for "
            "measured, controlled corner slip")
    if reward.overtake:
        reward_terms.append(f"+{reward.overtake:g} per clean overtake")
    if features.fuel:
        reward_terms.append("quadratic throttle drains the fixed fuel budget")
    if reward.timeout:
        reward_terms.append(
            f"{reward.timeout:g} task-deadline timeout penalty")
    if reward.terminalize_failure_time:
        reward_terms.append(
            f"canonical failures pay the full {reward.time * horizon_steps:g} "
            "horizon time budget; rolling-start failures pay their constant "
            "remaining-suffix budget; successful completion pays elapsed "
            "live-step time only")
    horizon_seconds = horizon_steps * DrivingEnv.dt
    termination = [
        "leaving the circuit", "wrong-way regression",
        "12 simulated seconds without progress",
        f"{horizon_seconds:g}-second horizon",
    ]
    if features.bots:
        termination.append("contact with any traffic car")
    if features.fuel:
        termination.append("fuel exhausted and vehicle stopped")
    observation_dimensions = [
        "longitudinal speed / max", "lateral speed / 25", "yaw rate / 3",
        "drift activation", "lateral offset / track width",
        "sin heading error", "cos heading error", "track half-width / 16",
        "curvature +8 m", "curvature +20 m", "curvature +40 m",
        "curvature +75 m", "curvature +120 m", "current surface grip",
        "surface grip +20 m", "surface grip +75 m", "surface grip +120 m",
        "sin track phase", "cos track phase",
        "forward course progress / lap length",
        "next checkpoint course progress / lap length",
        "wrong-way margin / 25",
        "progress since stall anchor / threshold", "stall counter / limit",
        "objective completion fraction",
    ]
    if features.fuel:
        observation_dimensions.append("fuel fraction")
    for index in range(len(features.bots)):
        observation_dimensions.extend((
            f"traffic {index + 1} signed arc gap / 150",
            f"traffic {index + 1} relative speed / max",
            f"traffic {index + 1} lateral lane fraction",
            f"traffic {index + 1} already passed",
        ))
    if features.metric == "overtakes" or features.guidance_observations:
        guidance_prefix = "Traffic" if features.metric == "overtakes" else "Geometric"
        observation_dimensions.extend((
            f"{guidance_prefix} pursuit target lane fraction",
            f"{guidance_prefix} pursuit heading error / max steering angle",
            f"{guidance_prefix} physics speed target / max speed",
        ))
    observation_dimensions.append("remaining horizon fraction")
    if (training_rolling_checkpoints is not None
            and len(training_rolling_checkpoints) == 1):
        rolling_start_phrase = (
            f"checkpoint {training_rolling_checkpoints[0]} as a rolling state"
        )
    elif training_rolling_checkpoints is None:
        rolling_checkpoint_label = "1..N-1"
        rolling_start_phrase = (
            f"uniform checkpoints {rolling_checkpoint_label} as rolling states"
        )
    elif tuple(training_rolling_checkpoints) == tuple(range(
            training_rolling_checkpoints[0],
            training_rolling_checkpoints[-1] + 1)):
        rolling_checkpoint_label = "..".join((
            str(training_rolling_checkpoints[0]),
            str(training_rolling_checkpoints[-1]),
        ))
        rolling_start_phrase = (
            f"uniform checkpoints {rolling_checkpoint_label} as rolling states"
        )
    else:
        rolling_checkpoint_label = ",".join(
            str(index) for index in training_rolling_checkpoints)
        rolling_start_phrase = (
            f"uniform checkpoints {rolling_checkpoint_label} as rolling states"
        )
    curriculum_state = []
    if features.fuel:
        curriculum_state.append("65% throttle-equivalent fuel")
    if features.bots:
        curriculum_state.append("time-advanced traffic and pass masks")
    if features.metric == "style":
        curriculum_state.append("assumed proportional style prefix")
    curriculum_state.append("no reset reward")
    training_start_distribution = (
        "75% canonical start; 25% "
        f"{rolling_start_phrase} at 70-90% of the "
        "curvature/grip backward-braking envelope; clock integrates an 80% "
        "envelope with a 1-second reserve; "
        + "; ".join(curriculum_state)
    )
    return ScenarioSpec(
        id=id, name=name, group=group, kind="driving", description=description,
        metric_label=metric_label, metric_mode=metric_mode,
        make_env=lambda jitter: DrivingEnv(
            _track(track_name), params=params, reward_cfg=reward,
            features=features, jitter=jitter, max_steps=horizon_steps),
        scene=lambda: _scene(track_name, features),
        training_factory=lambda: DrivingEnv(
            _track(track_name), params=params, reward_cfg=reward,
            features=features, jitter=True, random_start=True,
            start_line_probability=0.75,
            rolling_checkpoint_indices=training_rolling_checkpoints,
            max_steps=horizon_steps),
        training_start_distribution=training_start_distribution,
        objective=objective,
        success=success,
        observations=("speed and lateral slip", "track offset and heading error",
                      "five curvature look-aheads", "current and upcoming grip",
                      "fuel or every traffic car when present",
                      "geometric lane, heading, and braking-envelope speed targets"),
        observation_dimensions=tuple(observation_dimensions),
        actions=("throttle / brake", "steering", "drift toggle"),
        reward_terms=tuple(reward_terms),
        termination_conditions=tuple(termination),
        difficulty=difficulty,
        horizon_steps=horizon_steps,
        horizon_seconds=horizon_seconds,
        actor_initialization=actor_initialization,
        training_discount_factor=training_discount_factor,
        checkpoint_schema=checkpoint_schema,
    )


RAIN = "rgba(90,150,255,0.18)"
ICE_TINT = 0.25
WET_ZONES = (
    Zone(0.15, 0.22, 0.55, RAIN),
    Zone(0.55, 0.63, 0.45, RAIN),
    Zone(0.80, 0.86, 0.55, RAIN),
)

TRAFFIC_REWARD = RewardConfig(
    drift_corner=0.0,
    overtake=8.0,
    contact=-40.0,
    stall=-40.0,
    wrong_way=-40.0,
    timeout=-40.0,
    terminalize_failure_time=True,
    retain_terminal_course_potential=True,
)

TRAFFIC_ACTOR_INITIALIZATION = ActorInitialization(
    scope="traffic_demonstration_assisted_ppo",
    continuous_action_labels=("throttle_brake", "steering"),
    continuous_action_prior=(0.25, 0.0),
    continuous_log_std=(-2.0, -2.0),
    binary_action_labels=("drift",),
    binary_probability_prior=(0.05,),
)
TRAFFIC_FEATURES = DrivingFeatures(
    bots=(Bot(0.25, 18.0, -0.4), Bot(0.50, 24.0, 0.0),
          Bot(0.75, 30.0, 0.4)),
    metric="overtakes",
)
TRAFFIC_TRAINING_START_DISTRIBUTION = (
    "Predeclared Traffic rehearsal schedule: episodes 1-200 sample 75% "
    "physical checkpoint-11 near-pass starts with bot3 at 12 m/s and 25% "
    "nested speed-control starts; episodes 201-400 sample those modes at "
    "25% and 75%; episodes 401 onward use nested speed-control starts only. "
    "The performance-gated reverse curriculum uses audited physical "
    "checkpoints 11, 9, 3, then canonical 0 with 80% active frontier and "
    "20% uniformly sampled mastered stages. Before checkpoint 11 can "
    "unlock, nested bot3 control advances through 18, 24, and canonical "
    "30 m/s after two distinct >=80% fixed 10-seed confirmations per speed, "
    "sampling 80% active speed and 20% uniformly among mastered speeds. "
    "Rolling states use the 70-90% curvature/grip backward-braking envelope, "
    "an 80% reference clock with a 1-second reserve, time-advanced traffic, "
    "reconstructed pass masks, and no reset reward"
)


def _make_traffic_training_env() -> TrafficCurriculumEnv:
    return TrafficCurriculumEnv(
        _track("APEX_GP"),
        params=physics.F1,
        reward_cfg=TRAFFIC_REWARD,
        features=TRAFFIC_FEATURES,
        jitter=True,
        max_steps=2250,
        traffic_stage_curriculum=True,
    )


def _make_traffic_stage_evaluation_env(
        checkpoint: int) -> TrafficCurriculumEnv:
    if checkpoint not in TRAFFIC_CURRICULUM_FRONTIER_ORDER:
        raise ValueError("Traffic checkpoint must be one of 11, 9, 3, or 0")
    return TrafficCurriculumEnv(
        _track("APEX_GP"),
        params=physics.F1,
        reward_cfg=TRAFFIC_REWARD,
        features=TRAFFIC_FEATURES,
        jitter=True,
        max_steps=2250,
        forced_start_checkpoint=checkpoint,
    )


def _make_traffic_bot3_speed_evaluation_env(
        stage: str) -> TrafficCurriculumEnv:
    if stage not in TRAFFIC_BOT3_SPEED_STAGE_IDS:
        raise ValueError("unknown Traffic bot3 speed stage")
    return TrafficCurriculumEnv(
        _track("APEX_GP"),
        params=physics.F1,
        reward_cfg=TRAFFIC_REWARD,
        features=TRAFFIC_FEATURES,
        jitter=True,
        max_steps=2250,
        forced_start_checkpoint=11,
        forced_bot3_speed_stage=stage,
    )


TRAFFIC_DEMONSTRATION_SEED_BASE = 900_000
TRAFFIC_DEMONSTRATION_EPISODES = 30
TRAFFIC_DEMONSTRATION_STATE_STRIDE = 2
TRAFFIC_DAGGER_ROLLOUT_SEED_BASE = 920_000
TRAFFIC_DAGGER_ROUNDS = 2
TRAFFIC_DAGGER_ROUND_SEED_STRIDE = 1_000
TRAFFIC_DAGGER_EPISODES_PER_ROUND = 40
TRAFFIC_DAGGER_STATE_STRIDE = 2


def _make_traffic_demonstration_env() -> DrivingEnv:
    return DrivingEnv(
        _track("APEX_GP"),
        params=physics.F1,
        reward_cfg=TRAFFIC_REWARD,
        features=TRAFFIC_FEATURES,
        jitter=True,
        max_steps=2250,
    )


def traffic_behavior_cloning_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Collect fixed canonical pure-pursuit demonstrations for Traffic."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode_index in range(TRAFFIC_DEMONSTRATION_EPISODES):
        env = _make_traffic_demonstration_env()
        env.rng.seed(TRAFFIC_DEMONSTRATION_SEED_BASE + episode_index)
        observation = env.reset()
        for step in range(env.max_steps):
            action = traffic_reference_action(env)
            if step % TRAFFIC_DEMONSTRATION_STATE_STRIDE == 0:
                observations.append(observation.copy())
                actions.append(action.copy())
            observation, _, done, _ = env.step(action)
            if done:
                break
        if not env.episode_summary()["success"]:
            raise RuntimeError(
                f"Traffic demonstration {episode_index} did not complete")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def _deterministic_model_action(model, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(observation).unsqueeze(0)
        hidden = model.torso(tensor)
        parts = [torch.tanh(model.mu(hidden))]
        if model.drift_logit is not None:
            parts.append((model.drift_logit(hidden) > 0.0).float())
        return torch.cat(parts, dim=-1).squeeze(0).cpu().numpy()


def traffic_dagger_dataset(
        model, round_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Label states visited by the learned actor; the expert never acts."""
    if not 0 <= round_index < TRAFFIC_DAGGER_ROUNDS:
        raise ValueError("Traffic DAgger round is outside the fixed contract")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    seed_base = (
        TRAFFIC_DAGGER_ROLLOUT_SEED_BASE
        + round_index * TRAFFIC_DAGGER_ROUND_SEED_STRIDE
    )
    for episode_index in range(TRAFFIC_DAGGER_EPISODES_PER_ROUND):
        env = _make_traffic_demonstration_env()
        env.rng.seed(seed_base + episode_index)
        observation = env.reset()
        for step in range(env.max_steps):
            if step % TRAFFIC_DAGGER_STATE_STRIDE == 0:
                observations.append(observation.copy())
                actions.append(traffic_reference_action(env))
            learned_action = _deterministic_model_action(model, observation)
            observation, _, done, _ = env.step(learned_action)
            if done:
                break
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


TRAFFIC_ACTOR_WARM_START = BehaviorCloningWarmStart(
    id="traffic-pure-pursuit-dagger-v1",
    expert_id="traffic-physics-pure-pursuit-v1",
    expert_description=(
        "training-only pure pursuit with lookahead 18 m + 0.25 * speed, "
        "95% curvature/grip speed envelope, heading gain 1.6, lateral "
        "damping 0.3, and an opposite +/-0.72 lane target only while the "
        "nearest unpassed bot is between -12 m and +80 m; the expert labels "
        "training states and is absent at inference"
    ),
    dataset_seed_base=TRAFFIC_DEMONSTRATION_SEED_BASE,
    dataset_episodes=TRAFFIC_DEMONSTRATION_EPISODES,
    dataset_start_description=(
        "ordinary jittered canonical full-course starts disjoint from "
        "selection, holdout, outer-gate, and nested-control suites"
    ),
    dataset_builder=traffic_behavior_cloning_dataset,
    learning_rate=1e-3,
    batch_size=2_048,
    epochs=50,
    state_stride=TRAFFIC_DEMONSTRATION_STATE_STRIDE,
    continuous_action_labels=("throttle", "steering"),
    continuous_loss_weights=(1.0, 5.0),
    binary_action_labels=("drift",),
    binary_loss_weight=0.1,
    dagger_rounds=TRAFFIC_DAGGER_ROUNDS,
    dagger_rollout_seed_base=TRAFFIC_DAGGER_ROLLOUT_SEED_BASE,
    dagger_round_seed_stride=TRAFFIC_DAGGER_ROUND_SEED_STRIDE,
    dagger_episodes_per_round=TRAFFIC_DAGGER_EPISODES_PER_ROUND,
    dagger_state_stride=TRAFFIC_DAGGER_STATE_STRIDE,
    dagger_dataset_builder=traffic_dagger_dataset,
    dagger_epochs=50,
)


TRAFFIC_BOT3_SPEED_CONTROL = TrainingControlSpec(
    id=TRAFFIC_BOT3_SPEED_CONTROL_ID,
    scope_description=(
        "only bot3 speed at outer frontier checkpoint 11 before it may unlock"
    ),
    stage_ids=TRAFFIC_BOT3_SPEED_STAGE_IDS,
    stage_start_descriptions=(
        "checkpoint 11 with bot3 at 18 m/s, reconstructed gap 319.42 m, "
        "48.64 s remaining, required catch-up average 24.57 m/s, pass masks "
        "[true,true,false], time-advanced bot arc, and no reset reward",
        "checkpoint 11 with bot3 at 24 m/s, reconstructed gap 567.58 m, "
        "48.64 s remaining, required catch-up average 35.67 m/s, pass masks "
        "[true,true,false], time-advanced bot arc, and no reset reward",
        "checkpoint 11 with canonical bot3 at 30 m/s, reconstructed gap "
        "815.74 m, 48.64 s remaining, required catch-up average 46.77 m/s, "
        "pass masks [true,true,false], time-advanced bot arc, and no reset "
        "reward",
    ),
    active_stage_probability=TRAFFIC_BOT3_SPEED_ACTIVE_STAGE_PROBABILITY,
    success_rate_threshold=TRAFFIC_BOT3_SPEED_SUCCESS_RATE_THRESHOLD,
    consecutive_confirmations=(
        TRAFFIC_BOT3_SPEED_CONSECUTIVE_CONFIRMATIONS),
    evaluation_suite_version="traffic-cp11-bot3-speed-control-v1",
    evaluation_episodes=10,
    evaluation_seed_base=800_000,
    stage_seed_stride=1_000,
    evaluation_start_state_description=(
        "the active speed stage's disclosed physical checkpoint-11 state"
    ),
    outer_gate_dependency_description=(
        "the checkpoint-11 outer gate remains locked until canonical 30 m/s "
        "bot3 speed proficiency is complete"
    ),
    checkpoint_selection_description=(
        "bot3 speed control is training-only; fixed canonical full-course "
        "evaluation remains the checkpoint-selection signal"
    ),
    make_evaluation_env=_make_traffic_bot3_speed_evaluation_env,
)


TRAFFIC_TRAINING_CURRICULUM = TrainingCurriculumSpec(
    id=TRAFFIC_CURRICULUM_ID,
    frontier_order=TRAFFIC_CURRICULUM_FRONTIER_ORDER,
    active_frontier_probability=(
        TRAFFIC_CURRICULUM_ACTIVE_FRONTIER_PROBABILITY),
    success_rate_threshold=0.8,
    consecutive_confirmations=TRAFFIC_CURRICULUM_CONSECUTIVE_CONFIRMATIONS,
    evaluation_suite_version="traffic-stage-eval-v1",
    evaluation_episodes=10,
    evaluation_seed_base=700_000,
    segment_seed_stride=1_000,
    start_state_description=(
        "audited APEX_GP checkpoint with curvature/grip speed envelope, "
        "time-advanced bots, reconstructed pass masks, and no reset reward"
    ),
    make_evaluation_env=_make_traffic_stage_evaluation_env,
    frontier_success_rate_thresholds=(
        (11, 0.8), (9, 0.8), (3, 0.8), (0, 0.9),
    ),
    training_control=TRAFFIC_BOT3_SPEED_CONTROL,
)

DRIVING_SPECS: list[ScenarioSpec] = [
    _driving_spec(
        "apex-gp", "Apex GP", "Circuits",
        "The flagship circuit: straights, a sweeper, a chicane and two hairpins.",
        "APEX_GP"),
    _driving_spec(
        "velocita", "Velocità", "Circuits",
        "Speed temple — two huge straights and sweeping ends. Top speed is king.",
        "VELOCITA"),
    _driving_spec(
        "grandville", "Grandville Streets", "Circuits",
        "Narrow street circuit with relentless 90° kinks. Precision over power.",
        "GRANDVILLE"),
    _driving_spec(
        "thunder-oval", "Thunder Oval", "Circuits",
        "Pure oval speedway. Find the line, keep your foot in.",
        "THUNDER_OVAL",
        actor_initialization=THUNDER_ACTOR_INITIALIZATION),
    _driving_spec(
        "apex-gp-wet", "Apex GP — Wet", "Weather",
        "Rain at Apex GP: reduced grip everywhere, standing water in three zones.",
        "APEX_GP",
        reward=RewardConfig(stall=-40.0, wrong_way=-40.0),
        features=DrivingFeatures(global_grip=0.75, zones=WET_ZONES),
        training_rolling_checkpoints=(11,),
        horizon_steps=2250, checkpoint_schema=9),
    _driving_spec(
        "glacier", "Glacier Lake", "Weather",
        "A circuit on ice. Gentle inputs preserve momentum; controlled slides can help rotation.",
        "THUNDER_OVAL",
        features=DrivingFeatures(global_grip=ICE_TINT), difficulty="Advanced"),
    _driving_spec(
        "rally-ridge", "Rally Ridge", "Vehicles",
        "Rally car on a flowing dirt course. Loose grip, lives sideways.",
        "RALLY_RIDGE",
        params=physics.RALLY,
        reward=RewardConfig(drift_corner=0.008), difficulty="Advanced"),
    _driving_spec(
        "kart-sprint", "Kart Sprint", "Vehicles",
        "A go-kart on a tight mini circuit. Slow, nimble, unforgiving.",
        "KART_SPRINT",
        params=physics.KART),
    _driving_spec(
        "drift-trial", "Drift Trial", "Objectives",
        "Style over speed: score points by drifting fast through corners.",
        "APEX_GP",
        reward=RewardConfig(progress=0.05, lap=5.0, drift_corner=0.0,
                            style_coef=0.12),
        features=DrivingFeatures(metric="style"),
        metric_label="style pts", metric_mode="max",
        objective="Accumulate controlled high-speed slip through corners.",
        success="Score at least 10 style points before the horizon.",
        difficulty="Advanced", checkpoint_schema=8),
    _driving_spec(
        "eco-gp", "Eco GP", "Objectives",
        "One tank of fuel — throttle² drains it. Go far, not just fast.",
        "APEX_GP",
        reward=RewardConfig(fuel_empty=-5.0),
        features=DrivingFeatures(fuel=FuelConfig(rate=0.018), metric="tank"),
        metric_label="laps on tank", metric_mode="max",
        objective="Maximize distance while minimizing quadratic throttle use.",
        success="Complete one lap on the fixed fuel budget."),
    replace(
        _driving_spec(
            "traffic-rush", "Traffic Rush", "Objectives",
            "Three slower cars share the track. Overtake cleanly — contact ends it.",
            "APEX_GP",
            reward=TRAFFIC_REWARD,
            features=TRAFFIC_FEATURES,
            metric_label="overtakes", metric_mode="max",
            objective=(
                "Pass traffic without contact while maintaining forward "
                "progress."),
            success="Overtake all three traffic cars in one episode.",
            difficulty="Advanced",
            horizon_steps=2250, training_discount_factor=1.0,
            actor_initialization=TRAFFIC_ACTOR_INITIALIZATION,
            checkpoint_schema=18),
        training_factory=_make_traffic_training_env,
        training_start_distribution=TRAFFIC_TRAINING_START_DISTRIBUTION,
        training_curriculum=TRAFFIC_TRAINING_CURRICULUM,
        training_schedule=TRAFFIC_TRAINING_SCHEDULE,
        actor_warm_start=TRAFFIC_ACTOR_WARM_START,
    ),
]

DRIVING_SPECS = [
    guided_driving_spec(spec) if spec.id != "traffic-rush" else spec
    for spec in DRIVING_SPECS
]
