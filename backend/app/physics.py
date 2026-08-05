"""Arcade-grade 2D car model with a real drift mechanic.

Velocity lives in the body frame (v_long forward, v_lat to the car's right),
so the travel direction can diverge from the heading. Normal grip bleeds
lateral velocity quickly (the car goes where it points); engaging drift cuts
that lateral grip while granting extra yaw authority — the car rotates into
the corner while momentum carries it sideways.

All vehicle characteristics live in PhysicsParams so scenarios can swap cars
(F1 / rally / kart). `grip_scale` scales grip and both force caps per step,
which is how surfaces work: ice both slides AND refuses to rotate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DT = 0.02  # physics timestep, 50 Hz


@dataclass(frozen=True)
class PhysicsParams:
    max_speed: float = 80.0          # m/s (~288 km/h)
    max_reverse: float = 8.0
    max_accel: float = 22.0          # m/s^2
    max_brake: float = 38.0
    drag: float = 0.0034             # quadratic; terminal ~78 m/s
    roll: float = 0.02               # linear rolling resistance
    wheelbase: float = 3.2
    max_steer: float = 0.45          # rad
    grip_normal: float = 9.0         # 1/s lateral velocity decay rate
    grip_drift: float = 1.6
    a_lat_grip_normal: float = 38.0  # m/s^2 cap on lateral scrub (~3.9g)
    a_lat_grip_drift: float = 8.0
    a_yaw_cap_normal: float = 38.0   # m/s^2-equivalent cap on commanded yaw
    a_yaw_cap_drift: float = 52.0    # drifting trades grip for rotation
    drift_on_tau: float = 0.10       # s, drift engagement lag
    drift_off_tau: float = 0.40
    drift_scrub: float = 0.35        # fwd speed bled per unit lateral speed


F1 = PhysicsParams()

RALLY = PhysicsParams(
    max_speed=52.0, max_accel=16.0, max_brake=26.0, drag=0.005,
    wheelbase=2.6, max_steer=0.55,
    grip_normal=4.5, grip_drift=1.2,
    a_lat_grip_normal=22.0, a_lat_grip_drift=7.0,
    a_yaw_cap_normal=26.0, a_yaw_cap_drift=44.0,
    drift_off_tau=0.6, drift_scrub=0.2,
)

KART = PhysicsParams(
    max_speed=34.0, max_reverse=5.0, max_accel=18.0, max_brake=30.0,
    drag=0.014, wheelbase=1.1, max_steer=0.5,
    grip_normal=12.0, grip_drift=2.5,
    a_lat_grip_normal=45.0, a_lat_grip_drift=12.0,
    a_yaw_cap_normal=45.0, a_yaw_cap_drift=55.0,
    drift_scrub=0.5,
)


@dataclass
class CarState:
    x: float
    y: float
    heading: float          # rad, y-down world
    v_long: float           # m/s along heading
    v_lat: float            # m/s toward car's right
    omega: float            # rad/s, last applied yaw rate
    drift: float            # drift intensity 0..1

    @property
    def speed(self) -> float:
        return math.hypot(self.v_long, self.v_lat)

    @property
    def slip_angle(self) -> float:
        return math.atan2(self.v_lat, max(abs(self.v_long), 1e-3))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def step(state: CarState, throttle: float, steer: float, drift_on: bool,
         p: PhysicsParams = F1, grip_scale: float = 1.0,
         dt: float = DT) -> CarState:
    # Drift intensity follows the input with asymmetric lag.
    target = 1.0 if drift_on else 0.0
    tau = p.drift_on_tau if target > state.drift else p.drift_off_tau
    drift = state.drift + (target - state.drift) * dt / tau
    drift = min(max(drift, 0.0), 1.0)

    # Longitudinal: throttle forward, brake/reverse backward, drag always.
    if throttle >= 0.0:
        a_long = throttle * p.max_accel
    elif state.v_long > 0.5:
        a_long = throttle * p.max_brake
    else:
        a_long = throttle * p.max_accel * 0.5  # reverse, weak
    a_long -= p.drag * state.v_long * abs(state.v_long) + p.roll * state.v_long
    v_long = min(max(state.v_long + a_long * dt, -p.max_reverse), p.max_speed)

    # Commanded yaw from kinematic bicycle, capped by available lateral grip.
    steer_angle = steer * p.max_steer
    omega_cmd = v_long * math.tan(steer_angle) / p.wheelbase
    a_yaw_cap = _lerp(p.a_yaw_cap_normal, p.a_yaw_cap_drift, drift) * grip_scale
    omega_max = a_yaw_cap / max(abs(v_long), 2.0)
    omega = min(max(omega_cmd, -omega_max), omega_max)

    # Rotating the body frame converts forward momentum into lateral velocity.
    dtheta = omega * dt
    cos_d, sin_d = math.cos(dtheta), math.sin(dtheta)
    v_long_r = v_long * cos_d + state.v_lat * sin_d
    v_lat_r = -v_long * sin_d + state.v_lat * cos_d
    heading = state.heading + dtheta

    # Lateral grip scrubs sideways velocity, force-limited; drift slashes it.
    grip = _lerp(p.grip_normal, p.grip_drift, drift) * grip_scale
    a_lat_cap = _lerp(p.a_lat_grip_normal, p.a_lat_grip_drift, drift) * grip_scale
    scrub = min(max(grip * v_lat_r, -a_lat_cap), a_lat_cap) * dt
    if abs(scrub) > abs(v_lat_r):
        scrub = v_lat_r
    v_lat = v_lat_r - scrub

    # Sliding bleeds forward speed.
    v_long_f = v_long_r - p.drift_scrub * drift * abs(v_lat) * dt
    if v_long_r > 0.0:
        v_long_f = max(v_long_f, 0.0)

    fx, fy = math.cos(heading), math.sin(heading)
    rx, ry = -fy, fx  # right-hand vector in y-down coords
    x = state.x + (fx * v_long_f + rx * v_lat) * dt
    y = state.y + (fy * v_long_f + ry * v_lat) * dt

    return CarState(x=x, y=y, heading=heading, v_long=v_long_f, v_lat=v_lat,
                    omega=omega, drift=drift)
