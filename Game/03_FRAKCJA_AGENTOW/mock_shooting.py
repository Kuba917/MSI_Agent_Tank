"""
Simple shooting mock: closest hit in a +/- cone within range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Hit:
    hit_type: str  # "enemy", "ally", or "obstacle"
    hit_id: str
    distance: float
    hit_position: Tuple[float, float]


@dataclass(frozen=True)
class ShotLabel:
    label: float  # 1.0 = enemy hit, 0.0 = miss/blocked, -1.0 = ally in line
    reason: str
    hit: Optional[Hit]


def _normalize_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


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


def rotate_barrel_mock(barrel_angle: float, delta: float, max_spin: float) -> float:
    """Clamp delta by max_spin and normalize angle to [-180, 180]."""
    clamped = max(-max_spin, min(delta, max_spin))
    return _normalize_angle(barrel_angle + clamped)


def label_shot(
    shooter_pos: Tuple[float, float],
    heading: float,
    barrel_angle: float,
    enemies: Sequence[Tuple[str, Tuple[float, float]]],
    allies: Sequence[Tuple[str, Tuple[float, float]]],
    obstacles: Sequence[Tuple[str, Tuple[float, float]]],
    max_range: float = math.inf,
    half_angle: float = 5.0,
) -> ShotLabel:
    """
    Label a shot:
        - 1.0 if an enemy is the closest hit in the cone
        - -1.0 if an ally is the closest hit in the cone
        - 0.0 if blocked by obstacle or no valid hit
    """
    shoot_dir = _normalize_angle(heading + barrel_angle)
    closest_dist = max_range
    best: Optional[Hit] = None

    for tid, tpos in enemies:
        dist = _distance(shooter_pos, tpos)
        if dist >= closest_dist:
            continue
        ang = _angle_to(shooter_pos, tpos)
        if abs(_normalize_angle(ang - shoot_dir)) <= half_angle:
            closest_dist = dist
            best = Hit("enemy", tid, dist, tpos)

    for aid, apos in allies:
        dist = _distance(shooter_pos, apos)
        if dist >= closest_dist:
            continue
        ang = _angle_to(shooter_pos, apos)
        if abs(_normalize_angle(ang - shoot_dir)) <= half_angle:
            closest_dist = dist
            best = Hit("ally", aid, dist, apos)

    for oid, opos in obstacles:
        dist = _distance(shooter_pos, opos)
        if dist >= closest_dist:
            continue
        ang = _angle_to(shooter_pos, opos)
        if abs(_normalize_angle(ang - shoot_dir)) <= half_angle:
            closest_dist = dist
            best = Hit("obstacle", oid, dist, opos)

    if best is None:
        return ShotLabel(0.0, "no_target", None)
    if best.hit_type == "enemy":
        return ShotLabel(1.0, "enemy_hit", best)
    if best.hit_type == "ally":
        return ShotLabel(-1.0, "ally_in_line", best)
    return ShotLabel(0.0, "blocked", best)


__all__ = ["Hit", "ShotLabel", "fire_projectile_mock", "rotate_barrel_mock", "label_shot"]
