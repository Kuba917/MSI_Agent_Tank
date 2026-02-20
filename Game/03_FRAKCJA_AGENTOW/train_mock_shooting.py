"""
Train two ANFIS models on a simple shooting mock:
1) Barrel regressor: predict barrel delta to aim at a shootable enemy.
2) Shoot classifier (regression): predict label in {-1, 0, 1}.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ANFISDQN import ANFISDQN
from mock_shooting import label_shot, rotate_barrel_mock, _normalize_angle


MAP_WIDTH = 200.0
MAP_HEIGHT = 200.0
MAX_RANGE = 100.0
HALF_ANGLE = 5.0
MAX_SPIN = 90.0  # degrees per step for mock; align with engine barrel_spin_rate


@dataclass
class Scene:
    shooter_pos: Tuple[float, float]
    heading: float
    barrel_angle: float
    enemies: List[Tuple[str, Tuple[float, float]]]
    allies: List[Tuple[str, Tuple[float, float]]]
    obstacles: List[Tuple[str, Tuple[float, float]]]


def _rand_pos() -> Tuple[float, float]:
    return (random.uniform(0.0, MAP_WIDTH), random.uniform(0.0, MAP_HEIGHT))


def _rand_scene(
    max_enemies: int = 3,
    max_allies: int = 2,
    max_obstacles: int = 3,
) -> Scene:
    shooter_pos = _rand_pos()
    heading = random.uniform(-180.0, 180.0)
    barrel_angle = random.uniform(-180.0, 180.0)
    enemies = [(f"e{i}", _rand_pos()) for i in range(random.randint(0, max_enemies))]
    allies = [(f"a{i}", _rand_pos()) for i in range(random.randint(0, max_allies))]
    obstacles = [(f"o{i}", _rand_pos()) for i in range(random.randint(0, max_obstacles))]
    return Scene(shooter_pos, heading, barrel_angle, enemies, allies, obstacles)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _angle_to(src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))


def _features_for_scene(scene: Scene) -> Tuple[np.ndarray, Optional[Tuple[str, Tuple[float, float]]]]:
    """Return feature vector and nearest enemy (if any)."""
    enemy_visible = 1.0 if scene.enemies else 0.0
    nearest_enemy = None
    enemy_dist = 1.0
    angle_error = 0.0
    if scene.enemies:
        nearest_enemy = min(scene.enemies, key=lambda t: _dist(scene.shooter_pos, t[1]))
        enemy_dist = min(_dist(scene.shooter_pos, nearest_enemy[1]) / MAX_RANGE, 1.0)
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
        max_range=MAX_RANGE,
        half_angle=HALF_ANGLE,
    )
    obstacle_blocked = 1.0 if shot.reason == "blocked" else 0.0
    ally_in_line = 1.0 if shot.reason == "ally_in_line" else 0.0
    feats = np.array(
        [enemy_visible, enemy_dist, angle_error, obstacle_blocked, ally_in_line],
        dtype=np.float32,
    )
    return feats, nearest_enemy


def _barrel_target(scene: Scene) -> float:
    """Return desired barrel delta normalized to [-1, 1] (per-step)."""
    if not scene.enemies:
        return 0.0

    # Find nearest enemy that is shootable (no ally/obstacle blocking when aimed directly at it).
    candidates = sorted(scene.enemies, key=lambda t: _dist(scene.shooter_pos, t[1]))
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
            max_range=MAX_RANGE,
            half_angle=HALF_ANGLE,
        )
        if shot.reason == "enemy_hit":
            delta = _normalize_angle(target_barrel - scene.barrel_angle)
            delta = max(-MAX_SPIN, min(delta, MAX_SPIN))
            return float(delta / MAX_SPIN)

    return 0.0


def _shoot_label(scene: Scene) -> float:
    shot = label_shot(
        scene.shooter_pos,
        scene.heading,
        scene.barrel_angle,
        scene.enemies,
        scene.allies,
        scene.obstacles,
        max_range=MAX_RANGE,
        half_angle=HALF_ANGLE,
    )
    return float(shot.label)


def build_dataset(n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_barrel: List[float] = []
    y_shoot: List[float] = []
    for _ in range(n_samples):
        scene = _rand_scene()
        feats, _ = _features_for_scene(scene)
        x_list.append(feats)
        y_barrel.append(_barrel_target(scene))
        y_shoot.append(_shoot_label(scene))
    return np.stack(x_list), np.array(y_barrel, dtype=np.float32), np.array(y_shoot, dtype=np.float32)


def train_models(
    n_samples: int = 1000,
    batch_size: int = 128,
    epochs: int = 10,
    seed: int = 1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x, y_barrel, y_shoot = build_dataset(n_samples)
    x_t = torch.from_numpy(x)
    yb_t = torch.from_numpy(y_barrel).unsqueeze(1)
    ys_t = torch.from_numpy(y_shoot).unsqueeze(1)

    barrel_model = ANFISDQN(n_inputs=x.shape[1], n_rules=16, n_actions=1).train()
    shoot_model = ANFISDQN(n_inputs=x.shape[1], n_rules=16, n_actions=1).train()

    opt_b = torch.optim.Adam(barrel_model.parameters(), lr=1e-3)
    opt_s = torch.optim.Adam(shoot_model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(x_t.size(0))
        for start in range(0, x_t.size(0), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = x_t[batch_idx]
            yb = yb_t[batch_idx]
            ys = ys_t[batch_idx]

            pred_b = barrel_model(xb)
            loss_b = F.mse_loss(pred_b, yb)
            opt_b.zero_grad(set_to_none=True)
            loss_b.backward()
            opt_b.step()

            pred_s = shoot_model(xb)
            loss_s = F.mse_loss(pred_s, ys)
            opt_s.zero_grad(set_to_none=True)
            loss_s.backward()
            opt_s.step()

        print(f"epoch {epoch}: barrel_loss={loss_b.item():.4f} shoot_loss={loss_s.item():.4f}")

    return barrel_model, shoot_model


if __name__ == "__main__":
    train_models()
