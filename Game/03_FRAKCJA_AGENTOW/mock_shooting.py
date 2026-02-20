"""
Simple shooting mock: closest hit in a +/- cone within range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Hit:
    hit_type: str  # "tank" or "obstacle"
    hit_id: str
    distance: float
    hit_position: Tuple[float, float]


def _normalize_angle(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def _angle_to(src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def fire_projectile_mock(
    shooter_pos: Tuple[float, float],
    heading: float,
    barrel_angle: float,
    targets: Sequence[Tuple[str, Tuple[float, float]]],
    obstacles: Sequence[Tuple[str, Tuple[float, float]]],
    max_range: float = math.inf,
    half_angle: float = 5.0,
) -> Optional[Hit]:
    """
    Returns closest hit in a cone (+/- half_angle degrees) within max_range.

    targets: list of (id, (x, y))
    obstacles: list of (id, (x, y))
    """
    shoot_dir = _normalize_angle(heading + barrel_angle)
    closest_dist = max_range
    best: Optional[Hit] = None

    for tid, tpos in targets:
        dist = _distance(shooter_pos, tpos)
        if dist >= closest_dist:
            continue
        ang = _angle_to(shooter_pos, tpos)
        if abs(_normalize_angle(ang - shoot_dir)) <= half_angle:
            closest_dist = dist
            best = Hit("tank", tid, dist, tpos)

    for oid, opos in obstacles:
        dist = _distance(shooter_pos, opos)
        if dist >= closest_dist:
            continue
        ang = _angle_to(shooter_pos, opos)
        if abs(_normalize_angle(ang - shoot_dir)) <= half_angle:
            closest_dist = dist
            best = Hit("obstacle", oid, dist, opos)

    return best


__all__ = ["Hit", "fire_projectile_mock"]
