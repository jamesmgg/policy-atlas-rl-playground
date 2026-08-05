"""Closed circuits built from Catmull-Rom splines.

World coordinates are canvas-style (y grows downward), 1000x700 units,
1 unit ~= 1 meter. `build_track(waypoints)` is parametric so each scenario
supplies its own layout; all layouts share the same world so the canvas
never rescales.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# (x, y, half_width) control points. Closed loops, traversed in index order.

# Flagship GP circuit: straight, sweeper, chicane, two hairpins.
APEX_GP = [
    (150, 550, 16), (330, 560, 16), (520, 555, 16), (700, 540, 16),
    (850, 480, 14), (900, 340, 14), (850, 215, 14),
    (700, 160, 13), (550, 180, 11),
    (460, 110, 9), (340, 95, 9),
    (200, 130, 12), (120, 250, 12),
    (170, 350, 13), (300, 330, 13),
    (390, 410, 12), (330, 470, 9), (220, 460, 9),
]

# Speed temple: two long straights, sweeping ends, one tight final corner.
VELOCITA = [
    (120, 600, 16), (450, 615, 16), (780, 600, 16),
    (900, 520, 12),
    (880, 380, 14), (900, 240, 14),
    (820, 120, 12),
    (550, 90, 16), (280, 95, 16),
    (140, 140, 11),
    (90, 280, 13), (95, 430, 13),
    (100, 520, 12),
]

# Street circuit: narrow, many 90-degree kinks between "city blocks".
GRANDVILLE = [
    (150, 600, 9), (400, 610, 9),
    (520, 560, 8),
    (530, 440, 8),
    (660, 430, 8),
    (680, 300, 8),
    (560, 250, 8),
    (430, 280, 8),
    (330, 200, 8),
    (180, 170, 8),
    (120, 300, 9),
    (140, 450, 9),
    (110, 550, 9),
]

# Oval speedway.
THUNDER_OVAL = [
    (500, 620, 15), (760, 560, 15), (880, 350, 15), (760, 140, 15),
    (500, 90, 15), (240, 140, 15), (120, 350, 15), (240, 560, 15),
]

# Compact kart loop.
KART_SPRINT = [
    (380, 480, 8), (550, 500, 8), (660, 440, 8),
    (680, 330, 8), (600, 260, 8), (640, 180, 7),
    (540, 130, 7), (420, 160, 7), (350, 240, 8),
    (420, 310, 8), (340, 380, 8), (300, 450, 8),
]

# Flowing dirt circuit with esses for the rally car.
RALLY_RIDGE = [
    (150, 560, 12), (380, 590, 12), (600, 560, 11),
    (750, 480, 10), (820, 360, 10), (760, 240, 10),
    (620, 190, 10), (480, 230, 10), (360, 170, 10),
    (220, 130, 10), (110, 220, 11), (130, 350, 11),
    (220, 420, 10), (160, 480, 11),
]

SAMPLES_PER_SEGMENT = 60
N_CHECKPOINTS = 12


def _catmull_rom(p0, p1, p2, p3, t):
    """Uniform Catmull-Rom basis evaluated at t in [0,1). Vectorized over t."""
    t = t[:, None]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _smooth_closed(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average on a closed (wrapping) 1-D signal."""
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:-window]


@dataclass
class Track:
    centerline: np.ndarray   # (N, 2)
    tangents: np.ndarray     # (N, 2) unit vectors
    normals: np.ndarray      # (N, 2) unit vectors (left of travel direction)
    half_widths: np.ndarray  # (N,)
    curvature: np.ndarray    # (N,) signed, 1/units
    arc: np.ndarray          # (N,) cumulative arc length, arc[0] == 0
    total_length: float
    checkpoints: list[int] = field(default_factory=list)  # sample indices
    checkpoint_arcs: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.centerline)

    def localize(self, x: float, y: float, hint: int) -> tuple[int, float]:
        """Nearest centerline index near `hint` and signed lateral offset.

        Windowed search (biased forward) keeps localization stable where
        distinct track sections pass near each other.
        """
        n = self.n
        idx = (np.arange(hint - 50, hint + 90)) % n
        pts = self.centerline[idx]
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        best = int(idx[int(np.argmin(d2))])
        dx = x - self.centerline[best, 0]
        dy = y - self.centerline[best, 1]
        lat = dx * self.normals[best, 0] + dy * self.normals[best, 1]
        return best, float(lat)

    def index_ahead(self, idx: int, distance: float) -> int:
        """Nearest sample measured `distance` arc-units ahead of idx."""
        return self.index_at_arc(float(self.arc[idx]) + distance)

    def index_at_arc(self, s: float) -> int:
        """Sample index nearest to arc position s (wrapped)."""
        target = s % self.total_length
        right = int(np.searchsorted(self.arc, target, side="left")) % self.n
        left = (right - 1) % self.n

        def circular_error(i: int) -> float:
            delta = abs(float(self.arc[i]) - target)
            return min(delta, self.total_length - delta)

        return left if circular_error(left) <= circular_error(right) else right

    def to_dict(self) -> dict:
        return {
            "centerline": np.round(self.centerline, 2).tolist(),
            "normals": np.round(self.normals, 4).tolist(),
            "half_widths": np.round(self.half_widths, 2).tolist(),
            "checkpoints": self.checkpoints,
            "total_length": round(self.total_length, 2),
            "start_index": 0,
        }


def build_track(waypoints: list[tuple[float, float, float]],
                samples_per_segment: int = SAMPLES_PER_SEGMENT,
                n_checkpoints: int = N_CHECKPOINTS) -> Track:
    pts = np.array([(w[0], w[1]) for w in waypoints], dtype=np.float64)
    widths = np.array([w[2] for w in waypoints], dtype=np.float64)
    n_ctrl = len(pts)

    t = np.arange(samples_per_segment) / samples_per_segment
    samples = []
    sample_widths = []
    for i in range(n_ctrl):
        p0 = pts[(i - 1) % n_ctrl]
        p1 = pts[i]
        p2 = pts[(i + 1) % n_ctrl]
        p3 = pts[(i + 2) % n_ctrl]
        samples.append(_catmull_rom(p0, p1, p2, p3, t))
        sample_widths.append(widths[i] + (widths[(i + 1) % n_ctrl] - widths[i]) * t)
    centerline = np.concatenate(samples)
    half_widths = _smooth_closed(np.concatenate(sample_widths), 31)

    diffs = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    tangents = diffs / lengths[:, None]
    # Left-hand normal in y-down coordinates.
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)

    seg = np.linalg.norm(centerline - np.roll(centerline, 1, axis=0), axis=1)
    arc = np.cumsum(seg) - seg[0]
    total_length = float(arc[-1] + seg[0])

    # Signed curvature from tangent rotation per arc length.
    t_next = np.roll(tangents, -1, axis=0)
    cross = tangents[:, 0] * t_next[:, 1] - tangents[:, 1] * t_next[:, 0]
    ds = np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1)
    curvature = _smooth_closed(np.arcsin(np.clip(cross, -1, 1)) / np.maximum(ds, 1e-6), 21)

    checkpoint_arcs = [k * total_length / n_checkpoints for k in range(n_checkpoints)]
    checkpoints = [int(np.argmin(np.abs(arc - s))) for s in checkpoint_arcs]

    return Track(
        centerline=centerline,
        tangents=tangents,
        normals=normals,
        half_widths=half_widths,
        curvature=curvature,
        arc=arc,
        total_length=total_length,
        checkpoints=checkpoints,
        checkpoint_arcs=checkpoint_arcs,
    )


def heading_at(track: Track, idx: int) -> float:
    return math.atan2(track.tangents[idx, 1], track.tangents[idx, 0])
