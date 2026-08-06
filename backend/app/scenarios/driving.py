"""The 11 driving scenarios: tracks x physics x features."""
from __future__ import annotations

from functools import lru_cache

from .. import physics, track as tracks
from ..envs.driving import (Bot, DrivingEnv, DrivingFeatures, FuelConfig,
                            RewardConfig, Zone)
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
                  checkpoint_schema: int = 7) -> ScenarioSpec:
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
                      "fuel or every traffic car when present"),
        observation_dimensions=tuple(observation_dimensions),
        actions=("throttle / brake", "steering", "drift toggle"),
        reward_terms=tuple(reward_terms),
        termination_conditions=tuple(termination),
        difficulty=difficulty,
        horizon_steps=horizon_steps,
        horizon_seconds=horizon_seconds,
        actor_initialization=actor_initialization,
        checkpoint_schema=checkpoint_schema,
    )


RAIN = "rgba(90,150,255,0.18)"
ICE_TINT = 0.25
WET_ZONES = (
    Zone(0.15, 0.22, 0.55, RAIN),
    Zone(0.55, 0.63, 0.45, RAIN),
    Zone(0.80, 0.86, 0.55, RAIN),
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
    _driving_spec(
        "traffic-rush", "Traffic Rush", "Objectives",
        "Three slower cars share the track. Overtake cleanly — contact ends it.",
        "APEX_GP",
        reward=RewardConfig(overtake=8.0, contact=-40.0),
        features=DrivingFeatures(
            bots=(Bot(0.25, 18.0, -0.4), Bot(0.50, 24.0, 0.0),
                  Bot(0.75, 30.0, 0.4)),
            metric="overtakes"),
        metric_label="overtakes", metric_mode="max",
        objective="Pass traffic without contact while maintaining forward progress.",
        success="Overtake all three traffic cars in one episode.",
        difficulty="Advanced", training_rolling_checkpoints=(1, 2, 3),
        horizon_steps=2250, checkpoint_schema=9),
]
