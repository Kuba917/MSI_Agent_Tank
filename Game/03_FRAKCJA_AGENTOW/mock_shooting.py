"""
Minimal radial shooting mock used for ANFIS training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Optional, Tuple
import torch
import numpy as np


# TODO: replace this single default with ammo-dependent range from engine payload.
# Current default mirrors HEAVY ammo range.
DEFAULT_MAX_RANGE = 25.0
DEFAULT_HALF_ANGLE = 5.0
DEFAULT_BARREL_STEP = 5.0

# Hard-coded per-tank vision properties (engine-aligned).
TANK_PROFILES = {
    "LIGHT": {"vision_range": 70.0, "vision_half_angle": 20.0, "barrel_spin": 90.0},
    "HEAVY": {"vision_range": 40.0, "vision_half_angle": 30.0, "barrel_spin": 70.0},
    "SNIPER": {"vision_range": 120.0, "vision_half_angle": 10.0, "barrel_spin": 100.0},
}
DEFAULT_VISION_RANGE = 40.0

CASE_ENEMY_VISIBLE = "enemy_visible"
CASE_ENEMY_OUT_OF_VIEW = "enemy_out_of_view"
CASE_NO_TANKS = "no_tanks"  # legacy alias; treated as enemy_out_of_view

DEFAULT_SCENE_CASE_WEIGHTS = {
    CASE_ENEMY_VISIBLE: 0.90,
    CASE_ENEMY_OUT_OF_VIEW: 0.10,
}


@dataclass(frozen=True)
class Scene:
    tank_type: str
    enemy_visible: bool
    enemy_dist: float
    enemy_error_deg: float
    obstacle_blocks_shot: bool
    friendly_blocks_shot: bool
    vision_range: float


def _normalize_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")


def _validate_scene(scene: Scene) -> None:
    if scene.tank_type not in TANK_PROFILES:
        raise ValueError(f"Unsupported scene.tank_type: {scene.tank_type}")
    _require_finite("scene.enemy_dist", scene.enemy_dist)
    _require_finite("scene.enemy_error_deg", scene.enemy_error_deg)
    _require_finite("scene.vision_range", scene.vision_range)
    if scene.enemy_dist < 0.0:
        raise ValueError(f"scene.enemy_dist must be >= 0, got {scene.enemy_dist}")
    if scene.vision_range <= 0.0:
        raise ValueError(f"scene.vision_range must be > 0, got {scene.vision_range}")
    if scene.obstacle_blocks_shot and not scene.enemy_visible:
        raise ValueError("scene.obstacle_blocks_shot cannot be True when enemy_visible is False")
    if scene.friendly_blocks_shot and not scene.enemy_visible:
        raise ValueError("scene.friendly_blocks_shot cannot be True when enemy_visible is False")


def _rand_scene(
    case_weights: Optional[Mapping[str, float]] = None,
    tank_type: Optional[str] = None,
    obstacle_prob: float = 0.35,
    obstacle_in_cone_prob: float = 0.7,
    friendly_prob: float = 0.2,
    friendly_in_cone_prob: float = 0.7,
    hit_focused_error_prob: float = 0.6,
    in_range_focus_prob: float = 0.6,
    out_of_range_visible_prob: float = 0.5,
) -> Scene:
    _require_finite("obstacle_prob", obstacle_prob)
    _require_finite("obstacle_in_cone_prob", obstacle_in_cone_prob)
    _require_finite("friendly_prob", friendly_prob)
    _require_finite("friendly_in_cone_prob", friendly_in_cone_prob)
    _require_finite("hit_focused_error_prob", hit_focused_error_prob)
    _require_finite("in_range_focus_prob", in_range_focus_prob)
    _require_finite("out_of_range_visible_prob", out_of_range_visible_prob)
    if not (0.0 <= obstacle_prob <= 1.0):
        raise ValueError(f"obstacle_prob must be in [0, 1], got {obstacle_prob}")
    if not (0.0 <= obstacle_in_cone_prob <= 1.0):
        raise ValueError(f"obstacle_in_cone_prob must be in [0, 1], got {obstacle_in_cone_prob}")
    if not (0.0 <= friendly_prob <= 1.0):
        raise ValueError(f"friendly_prob must be in [0, 1], got {friendly_prob}")
    if not (0.0 <= friendly_in_cone_prob <= 1.0):
        raise ValueError(f"friendly_in_cone_prob must be in [0, 1], got {friendly_in_cone_prob}")
    if not (0.0 <= hit_focused_error_prob <= 1.0):
        raise ValueError(f"hit_focused_error_prob must be in [0, 1], got {hit_focused_error_prob}")
    if not (0.0 <= in_range_focus_prob <= 1.0):
        raise ValueError(f"in_range_focus_prob must be in [0, 1], got {in_range_focus_prob}")
    if not (0.0 <= out_of_range_visible_prob <= 1.0):
        raise ValueError(
            f"out_of_range_visible_prob must be in [0, 1], got {out_of_range_visible_prob}"
        )

    weights = case_weights or DEFAULT_SCENE_CASE_WEIGHTS
    if not weights:
        raise ValueError("case_weights cannot be empty")
    invalid_keys = [
        k for k in weights if k not in {CASE_ENEMY_VISIBLE, CASE_ENEMY_OUT_OF_VIEW, CASE_NO_TANKS}
    ]
    if invalid_keys:
        raise ValueError(f"Unsupported scene case keys: {invalid_keys}")
    if any(w < 0.0 for w in weights.values()):
        raise ValueError(f"Negative case weight in {weights}")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError(f"case_weights must sum to > 0, got {weights}")

    # Backward compatibility: old "no_tanks" key now maps to "enemy_out_of_view"
    merged_weights = {
        CASE_ENEMY_VISIBLE: float(weights.get(CASE_ENEMY_VISIBLE, 0.0)),
        CASE_ENEMY_OUT_OF_VIEW: float(weights.get(CASE_ENEMY_OUT_OF_VIEW, 0.0))
        + float(weights.get(CASE_NO_TANKS, 0.0)),
    }
    keys = list(merged_weights.keys())
    probs = np.array([float(merged_weights[k]) for k in keys], dtype=np.float64)
    probs /= probs.sum()
    case = str(np.random.choice(keys, p=probs))
    if tank_type is None:
        sampled_tank_type = random.choice(list(TANK_PROFILES.keys()))
    else:
        sampled_tank_type = str(tank_type).upper()
        if sampled_tank_type not in TANK_PROFILES:
            raise ValueError(f"Unsupported tank_type: {tank_type}")
    sampled_vision_range = float(TANK_PROFILES[sampled_tank_type]["vision_range"])
    sampled_vision_half_angle = float(TANK_PROFILES[sampled_tank_type]["vision_half_angle"])

    # TODO: replace fixed DEFAULT_MAX_RANGE with selected ammo range per sample.
    shoot_range_cap = min(sampled_vision_range, DEFAULT_MAX_RANGE)
    if case == CASE_ENEMY_VISIBLE:
        if shoot_range_cap < sampled_vision_range and random.random() < out_of_range_visible_prob:
            # Visible but outside shoot range -> explicit "do not fire" labels.
            dist = random.uniform(shoot_range_cap, sampled_vision_range)
        elif random.random() < in_range_focus_prob:
            dist = random.uniform(0.0, shoot_range_cap)
        else:
            dist = random.uniform(0.0, sampled_vision_range)
        # Visible means enemy is inside current vision cone.
        if random.random() < hit_focused_error_prob:
            # Bias toward shootable states while preserving the same global range constants.
            error_deg = random.uniform(-DEFAULT_HALF_ANGLE, DEFAULT_HALF_ANGLE)
        else:
            error_deg = random.uniform(-sampled_vision_half_angle, sampled_vision_half_angle)
        enemy_visible = True
    else:
        dist = random.uniform(0.0, sampled_vision_range)
        # Out-of-view means enemy exists but is currently outside the vision cone.
        # Match DQN fallback semantics where hidden target angle can be anywhere.
        out_of_view_margin = 1e-3
        if random.random() < 0.5:
            error_deg = random.uniform(
                sampled_vision_half_angle + out_of_view_margin,
                180.0,
            )
        else:
            error_deg = random.uniform(
                -180.0,
                -sampled_vision_half_angle - out_of_view_margin,
            )
        enemy_visible = False

    obstacle_blocks_shot = False
    friendly_blocks_shot = False
    if enemy_visible and random.random() < obstacle_prob:
        obstacle_dist = random.uniform(0.0, sampled_vision_range)
        if random.random() < obstacle_in_cone_prob:
            # Cone-aware sampling so blocked shots are represented in data.
            obstacle_error = random.uniform(-DEFAULT_HALF_ANGLE, DEFAULT_HALF_ANGLE)
        else:
            obstacle_error = random.uniform(-180.0, 180.0)
        obstacle_blocks_shot = (
            abs(_normalize_angle(obstacle_error)) <= DEFAULT_HALF_ANGLE
            and obstacle_dist <= min(dist, DEFAULT_MAX_RANGE)
        )
    if enemy_visible and random.random() < friendly_prob:
        max_block_dist = min(dist, DEFAULT_MAX_RANGE)
        ally_dist = random.uniform(0.0, max_block_dist) if max_block_dist > 0.0 else 0.0
        if random.random() < friendly_in_cone_prob:
            ally_error = random.uniform(-DEFAULT_HALF_ANGLE, DEFAULT_HALF_ANGLE)
        else:
            ally_error = random.uniform(-180.0, 180.0)
        # Friendly blocks the shot if closer than enemy and in firing cone/range.
        friendly_blocks_shot = (
            abs(_normalize_angle(ally_error)) <= DEFAULT_HALF_ANGLE
            and ally_dist < dist
            and ally_dist <= max_block_dist
        )

    scene = Scene(
        tank_type=sampled_tank_type,
        enemy_visible=enemy_visible,
        enemy_dist=dist,
        enemy_error_deg=error_deg,
        obstacle_blocks_shot=obstacle_blocks_shot,
        friendly_blocks_shot=friendly_blocks_shot,
        vision_range=sampled_vision_range,
    )
    _validate_scene(scene)
    return scene


def _features_for_scene(scene: Scene) -> Tuple[np.ndarray, None]:
    """Returns [enemy_visible, enemy_dist_norm, enemy_barrel_error_norm, shot_blocked]."""
    _validate_scene(scene)

    enemy_visible = 1.0 if scene.enemy_visible else 0.0
    # DQN alignment: with target fallback, dist/error stay informative even if enemy isn't visible.
    enemy_dist = min(max(scene.enemy_dist / scene.vision_range, 0.0), 1.0)
    angle_error = (_normalize_angle(scene.enemy_error_deg) / 180.0 + 1.0) * 0.5
    shot_blocked = scene.obstacle_blocks_shot or scene.friendly_blocks_shot

    feats = np.array(
        [enemy_visible, enemy_dist, angle_error, 1.0 if shot_blocked else 0.0],
        dtype=np.float32,
    )
    return feats, None


def _barrel_target(scene: Scene, max_spin: float) -> float:
    """Desired barrel delta normalized to [-1, 1] for one step."""
    _validate_scene(scene)
    _require_finite("max_spin", max_spin)
    if max_spin <= 0.0:
        raise ValueError(f"max_spin must be > 0, got {max_spin}")

    delta = _normalize_angle(scene.enemy_error_deg)
    delta = max(-max_spin, min(delta, max_spin))
    return float(delta / max_spin)


def _shoot_label(scene: Scene, max_range: float, half_angle: float) -> float:
    """Binary shoot label: 1.0 fire, 0.0 hold."""
    _validate_scene(scene)
    _require_finite("max_range", max_range)
    _require_finite("half_angle", half_angle)
    if max_range <= 0.0:
        raise ValueError(f"max_range must be > 0, got {max_range}")
    if not (0.0 <= half_angle <= 180.0):
        raise ValueError(f"half_angle must be in [0, 180], got {half_angle}")

    if not scene.enemy_visible:
        return 0.0
    if scene.obstacle_blocks_shot or scene.friendly_blocks_shot:
        return 0.0
    # TODO: max_range should come from currently loaded ammo (HEAVY/LIGHT/LONG_DISTANCE).
    if scene.enemy_dist > max_range:
        return 0.0
    if abs(_normalize_angle(scene.enemy_error_deg)) > half_angle:
        return 0.0
    return 1.0


def train_anfis_shooting_models(
    n_samples: int = 20000,
    batch_size: int = 128,
    epochs: int = 10,
    seed: int = 1,
    max_range: float = DEFAULT_MAX_RANGE,
    half_angle: float = DEFAULT_HALF_ANGLE,
    max_spin: float = DEFAULT_BARREL_STEP,
    n_rules: int = 16,
    lr: float = 1e-3,
) -> Tuple[Any, Any]:
    """Compatibility wrapper. Uses the single trainer in train_mock_shooting.py."""
    if max_range != DEFAULT_MAX_RANGE:
        raise ValueError(f"Unsupported max_range override: {max_range}")
    if half_angle != DEFAULT_HALF_ANGLE:
        raise ValueError(f"Unsupported half_angle override: {half_angle}")
    if max_spin != DEFAULT_BARREL_STEP:
        raise ValueError(f"Unsupported max_spin override: {max_spin}")
    if n_rules != 16:
        raise ValueError(f"Unsupported n_rules override: {n_rules}")
    if lr != 1e-3:
        raise ValueError(f"Unsupported lr override: {lr}")

    from train_mock_shooting import train_models

    return train_models(
        n_samples=n_samples,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )


def predict_anfis_action(
    barrel_model: Any,
    shoot_model: Any,
    scene: Scene,
    max_range: float = DEFAULT_MAX_RANGE,
    half_angle: float = DEFAULT_HALF_ANGLE,
    max_spin: float = DEFAULT_BARREL_STEP,
) -> Tuple[float, float]:
    """
    Returns (barrel_delta_degrees, shoot_score).
    """
    _require_finite("max_range", max_range)
    _require_finite("half_angle", half_angle)
    _require_finite("max_spin", max_spin)
    if max_spin <= 0.0:
        raise ValueError(f"max_spin must be > 0, got {max_spin}")
    if max_range <= 0.0:
        raise ValueError(f"max_range must be > 0, got {max_range}")
    if not (0.0 <= half_angle <= 180.0):
        raise ValueError(f"half_angle must be in [0, 180], got {half_angle}")

    feats, _ = _features_for_scene(scene)
    x = torch.from_numpy(np.array([feats], dtype=np.float32))
    with torch.no_grad():
        barrel_norm = float(barrel_model(x).squeeze(0).item())
        shoot_logit = float(shoot_model(x).squeeze(0).item())
        shoot_score = float(1.0 / (1.0 + math.exp(-shoot_logit)))
    barrel_norm = max(-1.0, min(1.0, barrel_norm))
    barrel_delta = barrel_norm * max_spin
    return barrel_delta, shoot_score


__all__ = [
    "Scene",
    "TANK_PROFILES",
    "DEFAULT_MAX_RANGE",
    "DEFAULT_HALF_ANGLE",
    "DEFAULT_BARREL_STEP",
    "DEFAULT_VISION_RANGE",
    "_rand_scene",
    "_features_for_scene",
    "_barrel_target",
    "_shoot_label",
    "train_anfis_shooting_models",
    "predict_anfis_action",
]