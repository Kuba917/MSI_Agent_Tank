"""
Simple shooting mock: closest hit in a +/- cone within range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Optional, Sequence, Tuple, List, Any, Callable


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


DEFAULT_MAX_RANGE = 100.0  # Align with engine LONG_DISTANCE ammo range.
DEFAULT_HALF_ANGLE = 5.0   # Engine hit cone half-angle (degrees).
DEFAULT_BARREL_SPIN = 90.0 # Typical engine barrel_spin_rate (deg/tick).


def fire_projectile_mock(
    shooter_pos: Tuple[float, float],
    heading: float,
    barrel_angle: float,
    targets: Sequence[Tuple[str, Tuple[float, float]]],
    obstacles: Sequence[Tuple[str, Tuple[float, float]]],
    max_range: float = DEFAULT_MAX_RANGE,
    half_angle: float = DEFAULT_HALF_ANGLE,
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


def rotate_barrel_mock(barrel_angle: float, delta: float, max_spin: float = DEFAULT_BARREL_SPIN) -> float:
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
    max_range: float = DEFAULT_MAX_RANGE,
    half_angle: float = DEFAULT_HALF_ANGLE,
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


# =============================
# ANFIS training helpers
# =============================

@dataclass
class Scene:
    shooter_pos: Tuple[float, float]
    heading: float
    barrel_angle: float
    enemies: List[Tuple[str, Tuple[float, float]]]
    allies: List[Tuple[str, Tuple[float, float]]]
    obstacles: List[Tuple[str, Tuple[float, float]]]


def _rand_pos(map_width: float, map_height: float) -> Tuple[float, float]:
    return (random.uniform(0.0, map_width), random.uniform(0.0, map_height))


def _rand_scene(
    map_width: float,
    map_height: float,
    max_enemies: int = 3,
    max_allies: int = 2,
    max_obstacles: int = 3,
) -> Scene:
    shooter_pos = _rand_pos(map_width, map_height)
    heading = random.uniform(-180.0, 180.0)
    barrel_angle = random.uniform(-180.0, 180.0)
    enemies = [(f"e{i}", _rand_pos(map_width, map_height)) for i in range(random.randint(0, max_enemies))]
    allies = [(f"a{i}", _rand_pos(map_width, map_height)) for i in range(random.randint(0, max_allies))]
    obstacles = [(f"o{i}", _rand_pos(map_width, map_height)) for i in range(random.randint(0, max_obstacles))]
    return Scene(shooter_pos, heading, barrel_angle, enemies, allies, obstacles)


def _features_for_scene(
    scene: Scene,
    max_range: float,
    half_angle: float,
) -> Tuple["np.ndarray", Optional[Tuple[str, Tuple[float, float]]]]:
    """Return feature vector and nearest enemy (if any)."""
    import numpy as np

    enemy_visible = 1.0 if scene.enemies else 0.0
    nearest_enemy = None
    enemy_dist = 1.0
    angle_error = 0.0
    if scene.enemies:
        nearest_enemy = min(scene.enemies, key=lambda t: _distance(scene.shooter_pos, t[1]))
        enemy_dist = min(_distance(scene.shooter_pos, nearest_enemy[1]) / max(max_range, 1.0), 1.0)
        enemy_angle = _angle_to(scene.shooter_pos, nearest_enemy[1])
        shoot_dir = _normalize_angle(scene.heading + scene.barrel_angle)
        angle_error = _normalize_angle(enemy_angle - shoot_dir) / 180.0
    # Obstacle/ally checks based on current barrel direction
    shot = label_shot(
        scene.shooter_pos,
        scene.heading,
        scene.barrel_angle,
        scene.enemies,
        scene.allies,
        scene.obstacles,
        max_range=max_range,
        half_angle=half_angle,
    )
    obstacle_blocked = 1.0 if shot.reason == "blocked" else 0.0
    ally_in_line = 1.0 if shot.reason == "ally_in_line" else 0.0
    feats = np.array(
        [enemy_visible, enemy_dist, angle_error, obstacle_blocked, ally_in_line],
        dtype=np.float32,
    )
    return feats, nearest_enemy


def _barrel_target(
    scene: Scene,
    max_range: float,
    half_angle: float,
    max_spin: float,
) -> float:
    """Return desired barrel delta normalized to [-1, 1] (per-step)."""
    if not scene.enemies:
        return 0.0

    candidates = sorted(scene.enemies, key=lambda t: _distance(scene.shooter_pos, t[1]))
    for _, pos in candidates:
        target_angle = _angle_to(scene.shooter_pos, pos)
        target_barrel = _normalize_angle(target_angle - scene.heading)
        shot = label_shot(
            scene.shooter_pos,
            scene.heading,
            target_barrel,
            scene.enemies,
            scene.allies,
            scene.obstacles,
            max_range=max_range,
            half_angle=half_angle,
        )
        if shot.reason == "enemy_hit":
            delta = _normalize_angle(target_barrel - scene.barrel_angle)
            delta = max(-max_spin, min(delta, max_spin))
            return float(delta / max_spin)

    return 0.0


def _shoot_label(scene: Scene, max_range: float, half_angle: float) -> float:
    shot = label_shot(
        scene.shooter_pos,
        scene.heading,
        scene.barrel_angle,
        scene.enemies,
        scene.allies,
        scene.obstacles,
        max_range=max_range,
        half_angle=half_angle,
    )
    return float(shot.label)


def train_anfis_shooting_models(
    n_samples: int = 1000,
    batch_size: int = 128,
    epochs: int = 10,
    seed: int = 1,
    map_width: float = 200.0,
    map_height: float = 200.0,
    max_range: float = DEFAULT_MAX_RANGE,
    half_angle: float = DEFAULT_HALF_ANGLE,
    max_spin: float = DEFAULT_BARREL_SPIN,
    n_rules: int = 16,
    lr: float = 1e-3,
) -> Tuple[Any, Any]:
    """
    Train two ANFIS models:
    1) Barrel regressor: predicts normalized barrel delta in [-1, 1].
    2) Shoot regressor: predicts label in {-1, 0, 1}.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ANFISDQN import ANFISDQN

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_list: List["np.ndarray"] = []
    y_barrel: List[float] = []
    y_shoot: List[float] = []
    for _ in range(n_samples):
        scene = _rand_scene(map_width, map_height)
        feats, _ = _features_for_scene(scene, max_range, half_angle)
        x_list.append(feats)
        y_barrel.append(_barrel_target(scene, max_range, half_angle, max_spin))
        y_shoot.append(_shoot_label(scene, max_range, half_angle))

    x = np.stack(x_list)
    yb = np.array(y_barrel, dtype=np.float32)
    ys = np.array(y_shoot, dtype=np.float32)

    x_t = torch.from_numpy(x)
    yb_t = torch.from_numpy(yb).unsqueeze(1)
    ys_t = torch.from_numpy(ys).unsqueeze(1)

    barrel_model = ANFISDQN(n_inputs=x.shape[1], n_rules=n_rules, n_actions=1).train()
    shoot_model = ANFISDQN(n_inputs=x.shape[1], n_rules=n_rules, n_actions=1).train()

    opt_b = torch.optim.Adam(barrel_model.parameters(), lr=lr)
    opt_s = torch.optim.Adam(shoot_model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(x_t.size(0))
        for start in range(0, x_t.size(0), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = x_t[batch_idx]
            yb_b = yb_t[batch_idx]
            ys_b = ys_t[batch_idx]

            pred_b = barrel_model(xb)
            loss_b = F.mse_loss(pred_b, yb_b)
            opt_b.zero_grad(set_to_none=True)
            loss_b.backward()
            opt_b.step()

            pred_s = shoot_model(xb)
            loss_s = F.mse_loss(pred_s, ys_b)
            opt_s.zero_grad(set_to_none=True)
            loss_s.backward()
            opt_s.step()

        print(f"epoch {epoch}: barrel_loss={loss_b.item():.4f} shoot_loss={loss_s.item():.4f}")

    return barrel_model, shoot_model


def predict_anfis_action(
    barrel_model: Any,
    shoot_model: Any,
    scene: Scene,
    max_range: float = DEFAULT_MAX_RANGE,
    half_angle: float = DEFAULT_HALF_ANGLE,
    max_spin: float = DEFAULT_BARREL_SPIN,
) -> Tuple[float, float]:
    """
    Returns (barrel_delta_degrees, shoot_score).
    shoot_score is in approx [-1, 1] (trained label).
    """
    import numpy as np
    import torch

    feats, _ = _features_for_scene(scene, max_range, half_angle)
    x = torch.from_numpy(np.array([feats], dtype=np.float32))
    with torch.no_grad():
        barrel_norm = float(barrel_model(x).squeeze(0).item())
        shoot_score = float(shoot_model(x).squeeze(0).item())
    barrel_norm = max(-1.0, min(1.0, barrel_norm))
    barrel_delta = barrel_norm * max_spin
    return barrel_delta, shoot_score


__all__ += [
    "Scene",
    "train_anfis_shooting_models",
    "predict_anfis_action",
]
