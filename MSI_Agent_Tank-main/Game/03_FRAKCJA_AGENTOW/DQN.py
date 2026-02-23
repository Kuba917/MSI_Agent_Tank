"""
Fuzzy DQN agent for 5v5 tank battles.

Key features:
- ANFIS-style Q-network (fuzzy rules + Sugeno consequents),
- replay buffer + target network,
- reward shaping focused on useful combat behavior,
- optional online training mode,
- FastAPI endpoints compatible with the game engine.
"""

from __future__ import annotations

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

import matplotlib.pyplot as plt
import argparse
import copy
from collections import deque
import math
import os
import random
import sys
import threading
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn
import matplotlib
matplotlib.use("Agg")       # zeby uniknac bledow

from ANFISDQN import ANFISDQN
from mock_shooting import (
    AMMO_RANGES,
    DEFAULT_HALF_ANGLE as MOCK_DEFAULT_HALF_ANGLE,
    DEFAULT_MAX_RANGE as MOCK_DEFAULT_MAX_RANGE,
    MOCK_FEATURE_DIM as DEFAULT_MOCK_FEATURE_DIM,
    build_mock_features,
)


# Add engine paths for local runs.
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA", "controller")
engine_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA")
sys.path.insert(0, controller_dir)
sys.path.insert(0, engine_dir)


STATE_DIM = 23
DEFAULT_MODEL_PATH = os.path.join(current_dir, "fuzzy_dqn_model_agent1_final.pt")
MAP_WIDTH = 200.0
MAP_HEIGHT = 200.0
COMET_LOG_EVERY = 20
ACTION_DIM = 2
# TODO: Scale actions using per-tank limits from my_tank_status (_top_speed, _heading_spin_rate, _barrel_spin_rate)
MAX_MOVE_SPEED = 5.0
MAX_HEADING_DELTA = 5.0
MAX_BARREL_DELTA = 5.0
AMMO_RANGE_BY_NAME = {str(name).upper(): float(rng) for name, rng in AMMO_RANGES.items()}


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: Optional[str] = None
    should_fire: bool = False


@dataclass(frozen=True)
class ActionSpec:
    name: str
    move_speed: float
    heading_rotation_angle: float
    barrel_rotation_angle: float
    should_fire: bool


@dataclass
class ShotTrace:
    x: float
    y: float
    angle_deg: float
    step_idx: int
    outcome: str = "pending"


class ReplayBuffer:
    """Simple ring buffer for off-policy learning."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.index = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = float(reward)
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[idx]).to(device)
        actions = torch.from_numpy(self.actions[idx]).to(device)
        rewards = torch.from_numpy(self.rewards[idx]).to(device)
        next_states = torch.from_numpy(self.next_states[idx]).to(device)
        dones = torch.from_numpy(self.dones[idx]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


@dataclass
class Observation:
    vector: np.ndarray
    enemy_visible: bool
    enemy_dist: float
    enemy_barrel_error: float
    shot_blocked: bool
    enemy_hull_error: float
    ally_fire_risk: bool
    obstacle_ahead: bool
    danger_ahead: bool
    powerup_visible: bool
    powerup_dist: float
    hp_ratio: float
    shield_ratio: float
    can_fire: bool
    reload_norm: float


class StateEncoder:
    """Converts engine payload into a normalized feature vector and helper signals."""

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(value, hi))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.hypot(b["x"] - a["x"], b["y"] - a["y"])

    @staticmethod
    def _angle_to(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))

    @staticmethod
    def _ammo_range_for_name(ammo_name: Optional[str]) -> float:
        if not ammo_name:
            return float(MOCK_DEFAULT_MAX_RANGE)
        return float(AMMO_RANGE_BY_NAME.get(str(ammo_name).upper(), MOCK_DEFAULT_MAX_RANGE))

    def _max_available_ammo_range(self, my_status: Dict[str, Any]) -> float:
        ammo_data = my_status.get("ammo", {}) or {}
        available_ranges: List[float] = []
        for ammo_name, slot in ammo_data.items():
            count = int((slot or {}).get("count", 0) or 0)
            if count > 0:
                available_ranges.append(self._ammo_range_for_name(str(ammo_name)))
        if available_ranges:
            return float(max(available_ranges))
        loaded = str(my_status.get("ammo_loaded") or "").upper()
        return self._ammo_range_for_name(loaded if loaded else None)

    def _can_fire(self, my_status: Dict[str, Any], reload_timer: float) -> bool:
        if reload_timer > 0.0:
            return False

        ammo = my_status.get("ammo", {})
        for slot in ammo.values():
            count = int((slot or {}).get("count", 0) or 0)
            if count > 0:
                return True
        return False

    @staticmethod
    def _sensor_enemy_id(tank: Dict[str, Any]) -> Optional[str]:
        raw = tank.get("id")
        if raw is None:
            return None
        sid = str(raw).strip()
        return sid if sid else None

    def _nearest_enemy(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        seen_tanks: List[Dict[str, Any]],
        preferred_enemy_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        enemies = [tank for tank in seen_tanks if tank.get("team") != my_team]
        if not enemies:
            return None

        if preferred_enemy_id:
            for tank in enemies:
                if self._sensor_enemy_id(tank) == preferred_enemy_id:
                    return tank

        return min(
            enemies,
            key=lambda tank: float(
                tank.get("distance")
                if tank.get("distance") is not None
                else self._distance(my_pos, tank.get("position", {"x": 0.0, "y": 0.0}))
            ),
        )

    def _ally_in_fire_line(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        barrel_abs: float,
        seen_tanks: List[Dict[str, Any]],
    ) -> bool:
        for tank in seen_tanks:
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            angle = self._angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 4.0:
                return True
        return False

    def _friendly_blocks_shot(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        barrel_abs: float,
        seen_tanks: List[Dict[str, Any]],
        target_distance: float,
        max_range: float = 25.0,
        half_angle: float = 5.0,
    ) -> bool:
        max_dist = min(max_range, max(target_distance, 0.0))
        for tank in seen_tanks:
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position")
            if not ally_pos:
                raise ValueError(f"Missing ally position in sensor data: {tank}")
            ally_dist = self._distance(my_pos, ally_pos)
            if ally_dist >= target_distance:
                continue
            if ally_dist > max_dist:
                continue
            angle = self._angle_to(my_pos, ally_pos)
            rel = abs(self.normalize_angle(angle - barrel_abs))
            if rel <= half_angle:
                return True
        return False

    def _has_object_ahead(
        self,
        my_pos: Dict[str, float],
        my_heading: float,
        objects: List[Dict[str, Any]],
        max_dist: float,
        half_angle: float,
    ) -> bool:
        for item in objects:
            pos = item.get("position")
            if not pos:
                raise ValueError(f"Missing position for object in sensor data: {item}")

            distance = self._distance(my_pos, pos)
            if distance > max_dist:
                continue

            angle = self._angle_to(my_pos, pos)
            rel = abs(self.normalize_angle(angle - my_heading))
            if rel < half_angle:
                return True

        return False

    def _shot_blocked_by_obstacle(
        self,
        my_pos: Dict[str, float],
        barrel_abs: float,
        seen_obstacles: List[Dict[str, Any]],
        target_distance: float,
        # TODO: use loaded-ammo range from my_status instead of one shared default.
        max_range: float = 25.0,
        half_angle: float = 5.0,
    ) -> bool:
        max_dist = min(max_range, max(target_distance, 0.0))
        for item in seen_obstacles:
            pos = item.get("position")
            if not pos:
                raise ValueError(f"Missing position for obstacle in sensor data: {item}")
            distance = self._distance(my_pos, pos)
            if distance > max_dist:
                continue
            angle = self._angle_to(my_pos, pos)
            rel = abs(self.normalize_angle(angle - barrel_abs))
            if rel <= half_angle:
                return True
        return False

    def encode(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
        enemy_target_pos: Optional[Tuple[float, float]] = None,
        preferred_enemy_id: Optional[str] = None,
    ) -> Observation:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")

        my_heading = float(my_status.get("heading", 0.0) or 0.0)
        my_barrel = float(my_status.get("barrel_angle", 0.0) or 0.0)
        barrel_abs = my_heading + my_barrel

        max_hp = float(my_status.get("_max_hp", 100.0) or 100.0)
        max_shield = float(my_status.get("_max_shield", 100.0) or 100.0)

        hp_ratio = self._clamp(float(my_status.get("hp", 0.0) or 0.0) / max_hp, 0.0, 1.0)
        shield_ratio = self._clamp(float(my_status.get("shield", 0.0) or 0.0) / max_shield, 0.0, 1.0)

        reload_timer = float(my_status.get("_reload_timer", 0.0) or 0.0)
        reload_norm = self._clamp(reload_timer / 10.0, 0.0, 1.0)

        ammo = my_status.get("ammo", {})
        heavy_ratio = self._clamp(float(ammo.get("HEAVY", {}).get("count", 0) or 0) / 5.0, 0.0, 1.0)
        light_ratio = self._clamp(float(ammo.get("LIGHT", {}).get("count", 0) or 0) / 15.0, 0.0, 1.0)
        long_ratio = self._clamp(
            float(ammo.get("LONG_DISTANCE", {}).get("count", 0) or 0) / 10.0,
            0.0,
            1.0,
        )

        seen_tanks = sensor_data.get("seen_tanks", [])
        nearest_enemy = self._nearest_enemy(
            my_pos,
            my_team,
            seen_tanks,
            preferred_enemy_id=preferred_enemy_id,
        )

        enemy_visible = nearest_enemy is not None
        enemy_dist = 1.0
        enemy_hull_error = 0.5
        enemy_barrel_error = 0.5
        enemy_distance_raw: Optional[float] = None

        enemy_pos = None
        if nearest_enemy is not None:
            enemy_pos = nearest_enemy.get("position", {"x": 0.0, "y": 0.0})
            distance_raw = nearest_enemy.get("distance")
            if distance_raw is None:
                raise ValueError(f"Missing enemy distance in sensor data: {nearest_enemy}")
            enemy_distance_raw = float(distance_raw)

            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

        else:
            enemy_pos = {"x": float(enemy_target_pos[0]), "y": float(enemy_target_pos[1])}
            distance_raw = self._distance(my_pos, enemy_pos)
            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

        target_angle = self._angle_to(my_pos, enemy_pos)
        enemy_hull_error = (self.normalize_angle(target_angle - my_heading) / 180.0 + 1.0) * 0.5
        enemy_barrel_error = (self.normalize_angle(target_angle - barrel_abs) / 180.0 + 1.0) * 0.5

        ally_fire_risk = self._ally_in_fire_line(my_pos, my_team, barrel_abs, seen_tanks)

        seen_obstacles = sensor_data.get("seen_obstacles", [])
        shot_blocked = False
        shot_range = self._max_available_ammo_range(my_status)
        if enemy_visible and enemy_distance_raw is not None:
            obstacle_blocks = self._shot_blocked_by_obstacle(
                my_pos,
                barrel_abs,
                seen_obstacles,
                target_distance=enemy_distance_raw,
                max_range=shot_range,
                half_angle=5.0,
            )
            ally_blocks = self._friendly_blocks_shot(
                my_pos,
                my_team,
                barrel_abs,
                seen_tanks,
                target_distance=enemy_distance_raw,
                max_range=shot_range,
                half_angle=5.0,
            )
            shot_blocked = obstacle_blocks or ally_blocks
        obstacle_ahead = self._has_object_ahead(
            my_pos,
            my_heading,
            seen_obstacles,
            max_dist=12.0,
            half_angle=28.0,
        )

        seen_terrains = sensor_data.get("seen_terrains", [])
        dangerous_terrains = [
            terrain
            for terrain in seen_terrains
            if terrain.get("type") in {"Water", "PotholeRoad"}
        ]
        danger_ahead = self._has_object_ahead(
            my_pos,
            my_heading,
            dangerous_terrains,
            max_dist=10.0,
            half_angle=35.0,
        )

        seen_powerups = sensor_data.get("seen_powerups", [])
        powerup_visible = len(seen_powerups) > 0
        powerup_dist = 1.0
        if powerup_visible:
            nearest_powerup = min(
                seen_powerups,
                key=lambda p: self._distance(my_pos, p.get("position", {"x": 0.0, "y": 0.0})),
            )
            dist_raw = self._distance(my_pos, nearest_powerup.get("position", {"x": 0.0, "y": 0.0}))
            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            powerup_dist = self._clamp(dist_raw / max(vision_range, 1.0), 0.0, 1.0)

        top_speed = float(my_status.get("_top_speed", 5.0) or 5.0)
        speed = float(my_status.get("move_speed", 0.0) or 0.0)
        speed_ratio = self._clamp(speed / max(top_speed, 1.0), -1.0, 1.0)
        speed_ratio = (speed_ratio + 1.0) * 0.5

        enemies_remaining_norm = self._clamp(float(enemies_remaining) / 5.0, 0.0, 1.0)
        can_fire = self._can_fire(my_status, reload_timer)

        vector = np.array(
            [
                hp_ratio,
                shield_ratio,
                reload_norm,
                heavy_ratio,
                light_ratio,
                long_ratio,
                1.0 if enemy_visible else 0.0,
                enemy_dist,
                enemy_barrel_error,
                enemy_hull_error,
                1.0 if ally_fire_risk else 0.0,
                1.0 if obstacle_ahead else 0.0,
                1.0 if danger_ahead else 0.0,
                1.0 if powerup_visible else 0.0,
                powerup_dist,
                speed_ratio,
                enemies_remaining_norm,
            ],
            dtype=np.float32,
        )

        return Observation(
            vector=vector,
            enemy_visible=enemy_visible,
            enemy_dist=enemy_dist,
            enemy_barrel_error=enemy_barrel_error,
            shot_blocked=shot_blocked,
            enemy_hull_error=enemy_hull_error,
            ally_fire_risk=ally_fire_risk,
            obstacle_ahead=obstacle_ahead,
            danger_ahead=danger_ahead,
            powerup_visible=powerup_visible,
            powerup_dist=powerup_dist,
            hp_ratio=hp_ratio,
            shield_ratio=shield_ratio,
            can_fire=can_fire,
            reload_norm=reload_norm,
        )


@dataclass
class AgentConfig:
    state_dim: int = STATE_DIM
    n_rules: int = 2 * STATE_DIM
    mf_type: str = "triangular"

    gamma: float = 0.97
    actor_lr: float = 4e-5
    critic_lr: float = 0.004
    tau: float = 0.04
    action_noise_start: float = 0.0003
    action_noise_end: float = 0.05
    action_noise_decay_steps: int = 16_177
    batch_size: int = 512
    replay_capacity: int = 50_000
    warmup_steps: int = 512
    train_every: int = 2
    target_sync_every: int = 1

    frame_skip: int = 1
    save_every_games: int = 1

    model_path: str = DEFAULT_MODEL_PATH
    best_model_path: Optional[str] = None
    mock_barrel_model_path: Optional[str] = None
    mock_shoot_model_path: Optional[str] = None
    mock_shoot_threshold: float = 0.55
    mock_half_angle_deg: float = float(MOCK_DEFAULT_HALF_ANGLE)
    movement_bootstrap_episodes: int = 30
    movement_bootstrap_alpha_start: float = 0.8
    movement_bootstrap_alpha_end: float = 0.0
    movement_bootstrap_visible_scale: float = 0.65
    movement_bootstrap_hidden_scale: float = 1.0
    movement_waypoint_min_distance: float = 20.0
    movement_waypoint_max_distance: float = 65.0
    movement_waypoint_lateral_max: float = 32.0
    movement_waypoint_replan_ticks: int = 120
    movement_waypoint_reach_radius: float = 9.0
    movement_waypoint_stuck_window: int = 70
    movement_waypoint_stuck_distance: float = 4.0
    movement_enemy_zone_patrol_activate_distance: float = 30.0
    movement_enemy_zone_patrol_exit_distance: float = 48.0
    movement_enemy_zone_patrol_radius_min: float = 10.0
    movement_enemy_zone_patrol_radius_max: float = 22.0
    movement_enemy_zone_patrol_replan_ticks: int = 28
    movement_enemy_zone_patrol_step_deg: float = 60.0
    target_lock_lost_patience_ticks: int = 90
    target_lock_relaxed_shoot_margin: float = 0.15
    progress_reward_scale: float = 0.9
    exploration_reward_scale: float = 0.25
    retreat_hp_threshold: float = 0.35
    seed: int = 1
    map_name: str = ""


class FuzzyDQNAgent:
    def __init__(self, name: str, config: AgentConfig, training_enabled: bool, load_checkpoint: bool = True):
        self.name = name
        self.config = config
        self.training_enabled = training_enabled
        self.map_name = str(config.map_name or "")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StateEncoder()
        self.config.mock_shoot_threshold = float(
            max(0.0, min(1.0, float(self.config.mock_shoot_threshold)))
        )
        self.config.mock_half_angle_deg = float(
            max(0.0, min(180.0, float(self.config.mock_half_angle_deg)))
        )
        self.config.movement_bootstrap_episodes = max(
            0, int(self.config.movement_bootstrap_episodes)
        )
        self.config.movement_bootstrap_alpha_start = float(
            max(0.0, min(1.0, float(self.config.movement_bootstrap_alpha_start)))
        )
        self.config.movement_bootstrap_alpha_end = float(
            max(0.0, min(1.0, float(self.config.movement_bootstrap_alpha_end)))
        )
        self.config.movement_bootstrap_visible_scale = float(
            max(0.0, min(1.0, float(self.config.movement_bootstrap_visible_scale)))
        )
        self.config.movement_bootstrap_hidden_scale = float(
            max(0.0, min(1.0, float(self.config.movement_bootstrap_hidden_scale)))
        )
        self.config.movement_waypoint_min_distance = float(
            max(1.0, float(self.config.movement_waypoint_min_distance))
        )
        self.config.movement_waypoint_max_distance = float(
            max(
                self.config.movement_waypoint_min_distance,
                float(self.config.movement_waypoint_max_distance),
            )
        )
        self.config.movement_waypoint_lateral_max = float(
            max(0.0, float(self.config.movement_waypoint_lateral_max))
        )
        self.config.movement_waypoint_replan_ticks = max(
            1, int(self.config.movement_waypoint_replan_ticks)
        )
        self.config.movement_waypoint_reach_radius = float(
            max(1.0, float(self.config.movement_waypoint_reach_radius))
        )
        self.config.movement_waypoint_stuck_window = max(
            5, int(self.config.movement_waypoint_stuck_window)
        )
        self.config.movement_waypoint_stuck_distance = float(
            max(0.0, float(self.config.movement_waypoint_stuck_distance))
        )
        self.config.movement_enemy_zone_patrol_activate_distance = float(
            max(4.0, float(self.config.movement_enemy_zone_patrol_activate_distance))
        )
        self.config.movement_enemy_zone_patrol_exit_distance = float(
            max(
                self.config.movement_enemy_zone_patrol_activate_distance + 1.0,
                float(self.config.movement_enemy_zone_patrol_exit_distance),
            )
        )
        self.config.movement_enemy_zone_patrol_radius_min = float(
            max(2.0, float(self.config.movement_enemy_zone_patrol_radius_min))
        )
        self.config.movement_enemy_zone_patrol_radius_max = float(
            max(
                self.config.movement_enemy_zone_patrol_radius_min + 1.0,
                float(self.config.movement_enemy_zone_patrol_radius_max),
            )
        )
        self.config.movement_enemy_zone_patrol_replan_ticks = max(
            1, int(self.config.movement_enemy_zone_patrol_replan_ticks)
        )
        self.config.movement_enemy_zone_patrol_step_deg = float(
            max(5.0, min(180.0, float(self.config.movement_enemy_zone_patrol_step_deg)))
        )
        self.config.target_lock_lost_patience_ticks = max(
            0, int(self.config.target_lock_lost_patience_ticks)
        )
        self.config.target_lock_relaxed_shoot_margin = float(
            max(0.0, min(0.49, float(self.config.target_lock_relaxed_shoot_margin)))
        )
        self.config.progress_reward_scale = float(
            max(0.0, float(self.config.progress_reward_scale))
        )
        self.config.exploration_reward_scale = float(
            max(0.0, float(self.config.exploration_reward_scale))
        )
        self.config.retreat_hp_threshold = float(
            max(0.0, min(1.0, float(self.config.retreat_hp_threshold)))
        )

        self.actor = ANFISDQN(
            n_inputs=config.state_dim,
            n_rules=config.n_rules,
            n_actions=ACTION_DIM,
            mf_type=config.mf_type,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.eval()

        self.critic = ANFISDQN(
            n_inputs=config.state_dim + ACTION_DIM,
            n_rules=config.n_rules,
            n_actions=1,
            mf_type=config.mf_type,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()

        # Optional mock shooting models.
        self.mock_barrel_model: Optional[ANFISDQN] = None
        self.mock_shoot_model: Optional[nn.Module] = None
        self.mock_feature_dim: int = int(DEFAULT_MOCK_FEATURE_DIM)
        if (config.mock_barrel_model_path is None) != (config.mock_shoot_model_path is None):
            raise ValueError("Both mock model paths must be provided (barrel and shoot).")
        if config.mock_barrel_model_path is not None:
            barrel_payload = torch.load(config.mock_barrel_model_path, map_location=self.device)
            shoot_payload = torch.load(config.mock_shoot_model_path, map_location=self.device)
            barrel_state = self._extract_mock_state_dict(barrel_payload)
            shoot_state = self._extract_mock_state_dict(shoot_payload)
            barrel_inputs, barrel_rules = self._infer_mock_model_shape(barrel_state)
            shoot_inputs, shoot_rules = self._infer_mock_model_shape(shoot_state)
            if barrel_inputs != shoot_inputs:
                raise ValueError(
                    "Mock model input mismatch: "
                    f"barrel_inputs={barrel_inputs}, shoot_inputs={shoot_inputs}"
                )
            if barrel_rules != shoot_rules:
                raise ValueError(
                    "Mock model rule mismatch: "
                    f"barrel_rules={barrel_rules}, shoot_rules={shoot_rules}"
                )
            self.mock_feature_dim = int(barrel_inputs)
            self.mock_barrel_model = ANFISDQN(
                n_inputs=barrel_inputs,
                n_rules=barrel_rules,
                n_actions=1,
            ).to(self.device)
            self.mock_barrel_model.load_state_dict(barrel_state, strict=True)
            self.mock_barrel_model.eval()
            self.mock_shoot_model = ANFISDQN(
                n_inputs=shoot_inputs,
                n_rules=shoot_rules,
                n_actions=1,
            ).to(self.device)
            self.mock_shoot_model.load_state_dict(shoot_state, strict=True)
            self.mock_shoot_model.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity, config.state_dim, ACTION_DIM)

        self.total_steps = 0
        self.train_steps = 0
        self.games_played = 0
        self.last_loss: Optional[float] = None
        self.current_episode_score = 0.0
        self.last_episode_score = 0.0

        self.experiment = None
        if self.training_enabled:
            if Experiment:
                threading.Thread(target=self._init_comet, daemon=True).start()
            else:
                print(f"[{self.name}] WARNING: comet_ml module not found. Experiment logging disabled.")

        self.last_observation: Optional[Observation] = None
        self.last_action_vector: Optional[np.ndarray] = None
        self.last_command = ActionCommand()
        self.last_fire_tick = -10_000
        self.prev_enemies_remaining: Optional[int] = None
        self.was_destroyed = False
        self.best_score = float("-inf")
        self.last_status: Optional[Dict[str, Any]] = None

        self.lock = threading.Lock()

        self.best_model_path = self._resolve_best_model_path()
        if load_checkpoint:
            self._load_checkpoint_if_available()
        print(
            f"[{self.name}] ready | training={self.training_enabled} "
            f"device={self.device} rules={self.config.n_rules} mf={self.config.mf_type} "
            f"save_path={self.config.model_path}"
        )

        self.trace_positions: List[Tuple[float, float]] = []
        self.trace_hp: List[float] = []
        self.trace_shots: List[ShotTrace] = []
        self.pending_shot_indices: Deque[int] = deque()
        self.trace_allies: List[Tuple[float, float]] = []
        self.trace_enemies: List[Tuple[float, float]] = []
        self.trace_labels: List[str] = []
        self.pos_history: List[Tuple[float, float]] = []
        self.trace_actor_raw: List[np.ndarray] = []
        self.last_actor_raw: Optional[np.ndarray] = None
        self.trace_mock_actions: List[np.ndarray] = []
        self.last_mock_action: Optional[np.ndarray] = None
        self.trace_damage_taken: List[float] = []
        self.trace_hit_target: List[float] = []
        self.trace_friendly_hit: List[float] = []
        self.trace_should_fire: List[float] = []
        self.episode_reward_total = 0.0
        self.episode_reward_parts: Dict[str, float] = {}
        self.episode_reward_parts_steps = 0
        self.frontier_min_x: Optional[float] = None
        self.frontier_max_x: Optional[float] = None
        self.frontier_min_y: Optional[float] = None
        self.frontier_max_y: Optional[float] = None
        self.reward_parts_history: Dict[str, List[float]] = {}
        self.enemy_target_pos: Optional[Tuple[float, float]] = None
        self.enemy_base_pos: Optional[Tuple[float, float]] = None
        self.nav_waypoint: Optional[Tuple[float, float]] = None
        self.nav_waypoint_age = 0
        self.in_enemy_zone_patrol = False
        self.patrol_orbit_angle_deg: Optional[float] = None
        self.patrol_orbit_direction = 1.0
        self.locked_enemy_id: Optional[str] = None
        self.locked_enemy_last_seen_tick = -10_000
        self.locked_enemy_last_pos: Optional[Tuple[float, float]] = None
        self.locked_enemy_visible = False

    def _init_comet(self) -> None:
        try:
            self.experiment = Experiment(
                api_key="L2PzW7c3YM3WqM5hNfCsloeLZ",
                project_name="msi-projekt",
                workspace="kluski777",
                auto_output_logging="simple"
            )
            self.experiment.set_name(self.name)
            self.experiment.log_parameters(vars(self.config))
        except Exception as e:
            print(f"[{self.name}] Failed to initialize Comet ML: {e}")

    def _resolve_best_model_path(self) -> Optional[str]:
        if self.config.best_model_path:
            return self.config.best_model_path
        if not self.config.model_path:
            return None
        root, ext = os.path.splitext(self.config.model_path)
        ext = ext or ".pt"
        return f"{root}_best{ext}"

    def _scale_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        move = action[:, 0] * MAX_MOVE_SPEED
        heading = action[:, 1] * MAX_HEADING_DELTA
        return torch.stack([move, heading], dim=1)

    def _current_action_noise_std(self) -> float:
        decay_steps = max(1, int(self.config.action_noise_decay_steps))
        t = min(max(self.total_steps, 0), decay_steps)
        frac = 1.0 - (t / float(decay_steps))
        return float(self.config.action_noise_end + (self.config.action_noise_start - self.config.action_noise_end) * frac)

    def _movement_bootstrap_alpha(self) -> float:
        if not self.training_enabled:
            return 0.0
        episodes = max(0, int(self.config.movement_bootstrap_episodes))
        if episodes <= 0:
            return 0.0
        if self.games_played >= episodes:
            return 0.0
        start = float(self.config.movement_bootstrap_alpha_start)
        end = float(self.config.movement_bootstrap_alpha_end)
        if episodes == 1:
            return float(max(0.0, min(1.0, start)))
        frac = float(self.games_played) / float(max(1, episodes - 1))
        alpha = start + (end - start) * frac
        return float(max(0.0, min(1.0, alpha)))

    @staticmethod
    def _clamp_to_map_point(point: Tuple[float, float], margin: float = 6.0) -> Tuple[float, float]:
        x = float(max(margin, min(MAP_WIDTH - margin, float(point[0]))))
        y = float(max(margin, min(MAP_HEIGHT - margin, float(point[1]))))
        return (x, y)

    def _enemy_base_anchor(self, my_team: Optional[int]) -> Tuple[float, float]:
        if my_team == 1:
            return (MAP_WIDTH * 0.82, MAP_HEIGHT * 0.20)
        if my_team == 2:
            return (MAP_WIDTH * 0.18, MAP_HEIGHT * 0.20)
        return (MAP_WIDTH * 0.5, MAP_HEIGHT * 0.5)

    def _is_navigation_stuck(self, current_pos: Tuple[float, float]) -> bool:
        window = max(5, int(self.config.movement_waypoint_stuck_window))
        if len(self.pos_history) < window:
            return False
        old_pos = self.pos_history[-window]
        moved = math.hypot(current_pos[0] - old_pos[0], current_pos[1] - old_pos[1])
        return moved < float(self.config.movement_waypoint_stuck_distance)

    def _sample_navigation_waypoint(
        self,
        current_pos: Tuple[float, float],
        final_target: Tuple[float, float],
        force_wide: bool,
    ) -> Tuple[float, float]:
        dx = float(final_target[0] - current_pos[0])
        dy = float(final_target[1] - current_pos[1])
        dist = math.hypot(dx, dy)
        reach = float(self.config.movement_waypoint_reach_radius)
        if dist <= max(1.0, reach * 1.5):
            return self._clamp_to_map_point(final_target)

        ux, uy = dx / max(1e-6, dist), dy / max(1e-6, dist)
        px, py = -uy, ux

        min_forward = float(self.config.movement_waypoint_min_distance)
        max_forward = float(self.config.movement_waypoint_max_distance)
        forward = random.uniform(min_forward, max_forward)
        forward = min(forward, dist * 0.85)
        forward = max(min_forward * 0.5, forward)

        lateral_max = float(self.config.movement_waypoint_lateral_max)
        if force_wide:
            lateral_max *= 1.35
        lateral_limit = min(lateral_max, max(6.0, dist * 0.6))
        lateral = random.uniform(-lateral_limit, lateral_limit)
        if abs(lateral) < 2.0 and lateral_limit > 2.0:
            side = -1.0 if ((self.games_played + (self.total_steps // 120)) % 2 == 0) else 1.0
            lateral = side * random.uniform(2.0, lateral_limit)

        wx = current_pos[0] + ux * forward + px * lateral
        wy = current_pos[1] + uy * forward + py * lateral
        return self._clamp_to_map_point((wx, wy))

    def _sample_enemy_zone_patrol_waypoint(
        self,
        anchor: Tuple[float, float],
        current_pos: Tuple[float, float],
        force_wide: bool,
    ) -> Tuple[float, float]:
        ax, ay = float(anchor[0]), float(anchor[1])
        dx = float(current_pos[0]) - ax
        dy = float(current_pos[1]) - ay
        if self.patrol_orbit_angle_deg is None:
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                self.patrol_orbit_angle_deg = math.degrees(math.atan2(dy, dx))
            else:
                self.patrol_orbit_angle_deg = random.uniform(-180.0, 180.0)
            self.patrol_orbit_direction = -1.0 if random.random() < 0.5 else 1.0
        else:
            if random.random() < 0.12:
                self.patrol_orbit_direction *= -1.0
            if force_wide and random.random() < 0.35:
                self.patrol_orbit_direction *= -1.0

        step = float(self.config.movement_enemy_zone_patrol_step_deg)
        step *= random.uniform(0.75, 1.25)
        self.patrol_orbit_angle_deg = self.encoder.normalize_angle(
            float(self.patrol_orbit_angle_deg) + self.patrol_orbit_direction * step
        )

        r_min = float(self.config.movement_enemy_zone_patrol_radius_min)
        r_max = float(self.config.movement_enemy_zone_patrol_radius_max)
        radius = random.uniform(r_min, r_max)
        if force_wide:
            radius = min(r_max, radius * 1.15)

        ang = math.radians(float(self.patrol_orbit_angle_deg))
        wx = ax + radius * math.cos(ang)
        wy = ay + radius * math.sin(ang)
        return self._clamp_to_map_point((wx, wy))

    def _resolve_bootstrap_target(
        self,
        my_status: Dict[str, Any],
        obs: Observation,
    ) -> Optional[Tuple[float, float]]:
        pos = my_status.get("position", {}) or {}
        current_pos = (
            float(pos.get("x", 0.0) or 0.0),
            float(pos.get("y", 0.0) or 0.0),
        )
        my_team = my_status.get("_team")
        if self.enemy_base_pos is None:
            self.enemy_base_pos = self._enemy_base_anchor(my_team if isinstance(my_team, int) else None)

        # If enemy is visible, prioritize combat target and clear route waypoint.
        if obs.enemy_visible and self.enemy_target_pos is not None:
            self.in_enemy_zone_patrol = False
            self.patrol_orbit_angle_deg = None
            self.nav_waypoint = None
            self.nav_waypoint_age = 0
            return self.enemy_target_pos

        final_target = self.enemy_base_pos or self.enemy_target_pos
        if final_target is None:
            return None
        final_target = self._clamp_to_map_point(final_target)
        self.enemy_target_pos = final_target

        reach = float(self.config.movement_waypoint_reach_radius)
        dist_to_final = math.hypot(
            current_pos[0] - final_target[0],
            current_pos[1] - final_target[1],
        )
        patrol_activate_dist = float(self.config.movement_enemy_zone_patrol_activate_distance)
        patrol_exit_dist = float(self.config.movement_enemy_zone_patrol_exit_distance)
        if not self.in_enemy_zone_patrol and dist_to_final <= patrol_activate_dist:
            self.in_enemy_zone_patrol = True
            self.nav_waypoint = None
            self.nav_waypoint_age = 0
        elif self.in_enemy_zone_patrol and dist_to_final > patrol_exit_dist:
            self.in_enemy_zone_patrol = False
            self.patrol_orbit_angle_deg = None

        if self.in_enemy_zone_patrol:
            is_stuck = self._is_navigation_stuck(current_pos)
            need_replan_patrol = False
            if self.nav_waypoint is None:
                need_replan_patrol = True
            else:
                dist_to_wp = math.hypot(
                    current_pos[0] - self.nav_waypoint[0],
                    current_pos[1] - self.nav_waypoint[1],
                )
                if dist_to_wp <= reach:
                    need_replan_patrol = True
                elif self.nav_waypoint_age >= int(self.config.movement_enemy_zone_patrol_replan_ticks):
                    need_replan_patrol = True
                elif is_stuck:
                    need_replan_patrol = True
            if need_replan_patrol:
                self.nav_waypoint = self._sample_enemy_zone_patrol_waypoint(
                    anchor=final_target,
                    current_pos=current_pos,
                    force_wide=is_stuck,
                )
                self.nav_waypoint_age = 0
            else:
                self.nav_waypoint_age += 1
            active_target = self.nav_waypoint or final_target
            self.enemy_target_pos = active_target
            return active_target

        if dist_to_final <= max(1.0, 1.5 * reach):
            self.nav_waypoint = final_target
            self.nav_waypoint_age = 0
            self.enemy_target_pos = final_target
            return final_target

        is_stuck = self._is_navigation_stuck(current_pos)
        need_replan = False
        if self.nav_waypoint is None:
            need_replan = True
        else:
            dist_to_wp = math.hypot(
                current_pos[0] - self.nav_waypoint[0],
                current_pos[1] - self.nav_waypoint[1],
            )
            if dist_to_wp <= reach:
                need_replan = True
            elif self.nav_waypoint_age >= int(self.config.movement_waypoint_replan_ticks):
                need_replan = True
            elif is_stuck:
                need_replan = True

        if need_replan:
            self.nav_waypoint = self._sample_navigation_waypoint(
                current_pos=current_pos,
                final_target=final_target,
                force_wide=is_stuck,
            )
            self.nav_waypoint_age = 0
        else:
            self.nav_waypoint_age += 1

        active_target = self.nav_waypoint or final_target
        self.enemy_target_pos = active_target
        return active_target

    def _teacher_move_heading(
        self,
        my_status: Dict[str, Any],
        target_pos: Tuple[float, float],
        obs: Observation,
    ) -> Tuple[float, float]:
        pos = my_status.get("position", {})
        x = float(pos.get("x", 0.0) or 0.0)
        y = float(pos.get("y", 0.0) or 0.0)
        heading = float(my_status.get("heading", 0.0) or 0.0)
        dx = float(target_pos[0]) - x
        dy = float(target_pos[1]) - y
        target_angle = math.degrees(math.atan2(dy, dx))
        heading_error = self.encoder.normalize_angle(target_angle - heading)
        heading_cmd = max(-MAX_HEADING_DELTA, min(MAX_HEADING_DELTA, heading_error))

        top_speed = float(my_status.get("_top_speed", MAX_MOVE_SPEED) or MAX_MOVE_SPEED)
        max_forward = max(0.0, min(MAX_MOVE_SPEED, top_speed))
        target_distance = math.hypot(dx, dy)
        abs_err = abs(heading_error)
        if target_distance < 8.0:
            move_cmd = 0.0
        elif obs.obstacle_ahead:
            move_cmd = max_forward * (0.2 if abs_err < 45.0 else 0.0)
        elif abs_err > 110.0:
            move_cmd = 0.0
        elif abs_err > 60.0:
            move_cmd = max_forward * 0.35
        else:
            move_cmd = max_forward
        return float(move_cmd), float(heading_cmd)

    def _apply_movement_bootstrap(
        self,
        action_vec: np.ndarray,
        my_status: Dict[str, Any],
        obs: Observation,
    ) -> np.ndarray:
        if action_vec.shape[0] < 2:
            return action_vec
        alpha = self._movement_bootstrap_alpha()
        if obs.enemy_visible:
            alpha *= float(self.config.movement_bootstrap_visible_scale)
        else:
            alpha *= float(self.config.movement_bootstrap_hidden_scale)
        if alpha <= 1e-6:
            return action_vec
        teacher_target = self._resolve_bootstrap_target(my_status=my_status, obs=obs)
        if teacher_target is None:
            return action_vec
        teacher_move, teacher_heading = self._teacher_move_heading(
            my_status=my_status,
            target_pos=teacher_target,
            obs=obs,
        )
        mixed = np.array(action_vec, dtype=np.float32, copy=True)
        mixed[0] = float((1.0 - alpha) * mixed[0] + alpha * teacher_move)
        mixed[1] = float((1.0 - alpha) * mixed[1] + alpha * teacher_heading)
        mixed[0] = float(max(-MAX_MOVE_SPEED, min(MAX_MOVE_SPEED, mixed[0])))
        mixed[1] = float(max(-MAX_HEADING_DELTA, min(MAX_HEADING_DELTA, mixed[1])))
        return mixed

    def _select_action(self, state_vector: np.ndarray, training: bool) -> np.ndarray:
        state = torch.from_numpy(state_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.actor(state)
            self.last_actor_raw = raw.squeeze(0).detach().cpu().numpy()
            action = torch.tanh(raw)
            if training:
                noise_std = self._current_action_noise_std()
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=action.shape,
                    device=action.device,
                )
                action = torch.clamp(action + noise, -1.0, 1.0)
            scaled = self._scale_action_tensor(action)
        return scaled.squeeze(0).cpu().numpy()

    @staticmethod
    def _ammo_counts(my_status: Dict[str, Any]) -> Dict[str, int]:
        ammo_data = my_status.get("ammo", {}) or {}
        counts: Dict[str, int] = {}
        for ammo_name, slot in ammo_data.items():
            key = str(ammo_name).upper()
            count = int((slot or {}).get("count", 0) or 0)
            counts[key] = max(0, count)
        return counts

    @staticmethod
    def _sensor_enemy_id(tank: Dict[str, Any]) -> Optional[str]:
        raw = tank.get("id")
        if raw is None:
            return None
        sid = str(raw).strip()
        return sid if sid else None

    def _distance_to_sensor_enemy(
        self,
        my_pos: Tuple[float, float],
        tank: Dict[str, Any],
    ) -> float:
        distance = tank.get("distance")
        if distance is not None:
            try:
                return float(distance)
            except (TypeError, ValueError):
                pass
        pos = tank.get("position", {}) or {}
        ex = float(pos.get("x", 0.0) or 0.0)
        ey = float(pos.get("y", 0.0) or 0.0)
        return float(math.hypot(ex - my_pos[0], ey - my_pos[1]))

    def _select_focus_enemy(
        self,
        my_pos: Tuple[float, float],
        my_team: Optional[int],
        seen_tanks: List[Dict[str, Any]],
        current_tick: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Tuple[float, float]], bool]:
        enemies = [tank for tank in seen_tanks if tank.get("team") != my_team]
        enemies = [tank for tank in enemies if isinstance(tank.get("position"), dict)]

        lock_patience = max(0, int(self.config.target_lock_lost_patience_ticks))
        self.locked_enemy_visible = False

        if self.locked_enemy_id:
            for enemy in enemies:
                if self._sensor_enemy_id(enemy) == self.locked_enemy_id:
                    epos = enemy.get("position", {}) or {}
                    locked_pos = (
                        float(epos.get("x", 0.0) or 0.0),
                        float(epos.get("y", 0.0) or 0.0),
                    )
                    self.locked_enemy_last_seen_tick = int(current_tick)
                    self.locked_enemy_last_pos = locked_pos
                    self.locked_enemy_visible = True
                    return enemy, self.locked_enemy_id, locked_pos, True

            if (
                self.locked_enemy_last_pos is not None
                and (int(current_tick) - int(self.locked_enemy_last_seen_tick)) <= lock_patience
            ):
                return None, self.locked_enemy_id, self.locked_enemy_last_pos, False

            self.locked_enemy_id = None
            self.locked_enemy_last_pos = None
            self.locked_enemy_last_seen_tick = -10_000

        if enemies:
            chosen = min(enemies, key=lambda t: self._distance_to_sensor_enemy(my_pos, t))
            epos = chosen.get("position", {}) or {}
            chosen_pos = (
                float(epos.get("x", 0.0) or 0.0),
                float(epos.get("y", 0.0) or 0.0),
            )
            chosen_id = self._sensor_enemy_id(chosen)
            self.locked_enemy_id = chosen_id
            self.locked_enemy_last_seen_tick = int(current_tick)
            self.locked_enemy_last_pos = chosen_pos
            self.locked_enemy_visible = True
            return chosen, chosen_id, chosen_pos, True

        if (
            self.locked_enemy_last_pos is not None
            and self.locked_enemy_id is not None
            and (int(current_tick) - int(self.locked_enemy_last_seen_tick)) <= lock_patience
        ):
            return None, self.locked_enemy_id, self.locked_enemy_last_pos, False

        self.locked_enemy_id = None
        self.locked_enemy_last_pos = None
        self.locked_enemy_last_seen_tick = -10_000
        return None, None, None, False

    @staticmethod
    def _extract_mock_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported mock checkpoint payload type: {type(payload)}")
        if "value_weights" in payload:
            return payload
        for key in ("state_dict", "model_state_dict", "actor_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and "value_weights" in candidate:
                return candidate
        keys = list(payload.keys())[:10]
        raise ValueError(
            "Unsupported mock checkpoint format: missing value_weights. "
            f"Top-level keys={keys}"
        )

    @staticmethod
    def _infer_mock_model_shape(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        vw = state_dict.get("value_weights")
        if vw is None:
            raise ValueError("Mock state dict does not contain value_weights")
        if getattr(vw, "ndim", None) != 3:
            raise ValueError(f"Unexpected value_weights shape: {getattr(vw, 'shape', None)}")
        n_rules = int(vw.shape[0])
        n_inputs = int(vw.shape[2])
        if n_rules <= 0 or n_inputs <= 0:
            raise ValueError(
                f"Invalid mock model shape from value_weights: rules={n_rules}, inputs={n_inputs}"
            )
        return n_inputs, n_rules

    @staticmethod
    def _ammo_range_for_name(ammo_name: Optional[str]) -> float:
        if not ammo_name:
            return float(MOCK_DEFAULT_MAX_RANGE)
        return float(AMMO_RANGE_BY_NAME.get(str(ammo_name).upper(), MOCK_DEFAULT_MAX_RANGE))

    def _reload_norm_from_status(self, my_status: Dict[str, Any]) -> float:
        reload_timer = self._safe_nonnegative_float(my_status.get("_reload_timer", 0.0), default=0.0)
        return float(max(0.0, min(1.0, reload_timer / 10.0)))

    def _can_fire_with_ammo(self, my_status: Dict[str, Any], ammo_name: Optional[str]) -> bool:
        reload_timer = self._safe_nonnegative_float(my_status.get("_reload_timer", 0.0), default=0.0)
        if reload_timer > 0.0:
            return False
        counts = self._ammo_counts(my_status)
        if ammo_name:
            return counts.get(str(ammo_name).upper(), 0) > 0
        loaded = str(my_status.get("ammo_loaded") or "").upper()
        if loaded and counts.get(loaded, 0) > 0:
            return True
        return any(c > 0 for c in counts.values())

    def _build_mock_feature_vector(
        self,
        obs: Observation,
        can_fire: bool,
        reload_norm: float,
        ammo_range: float,
    ) -> np.ndarray:
        full = build_mock_features(
            enemy_visible=obs.enemy_visible,
            enemy_dist_norm=float(obs.enemy_dist),
            enemy_barrel_error_norm=float(obs.enemy_barrel_error),
            shot_blocked=bool(obs.shot_blocked),
            can_fire=bool(can_fire),
            reload_norm=float(reload_norm),
            ammo_range=float(ammo_range),
        )
        if self.mock_feature_dim == full.shape[0]:
            return full
        if self.mock_feature_dim <= 0:
            raise ValueError(f"Invalid mock_feature_dim={self.mock_feature_dim}")
        if self.mock_feature_dim < full.shape[0]:
            return full[: self.mock_feature_dim]
        padded = np.zeros((self.mock_feature_dim,), dtype=np.float32)
        padded[: full.shape[0]] = full
        return padded

    def _passes_fire_safety_gates(
        self,
        obs: Observation,
        my_status: Dict[str, Any],
        ammo_name: Optional[str],
    ) -> bool:
        if not obs.enemy_visible:
            return False
        if obs.shot_blocked:
            return False
        if not self._can_fire_with_ammo(my_status, ammo_name):
            return False
        vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
        enemy_distance = float(obs.enemy_dist) * max(vision_range, 1.0)
        ammo_range = self._ammo_range_for_name(ammo_name)
        if enemy_distance > ammo_range:
            return False
        barrel_error_deg = abs((float(obs.enemy_barrel_error) - 0.5) * 360.0)
        if barrel_error_deg > float(self.config.mock_half_angle_deg):
            return False
        return True

    def _passes_fire_gates(
        self,
        shoot_val: float,
        obs: Observation,
        my_status: Dict[str, Any],
        ammo_name: Optional[str],
    ) -> bool:
        if not math.isfinite(shoot_val):
            return False
        if shoot_val <= float(self.config.mock_shoot_threshold):
            return False
        return self._passes_fire_safety_gates(
            obs=obs,
            my_status=my_status,
            ammo_name=ammo_name,
        )

    @staticmethod
    def _safe_nonnegative_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return float(max(0.0, parsed))

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "t", "yes", "y"}:
                return True
            if v in {"0", "false", "f", "no", "n"}:
                return False
            return default
        return default

    def _select_ammo_for_action(
        self,
        my_status: Dict[str, Any],
        obs: Observation,
        action: ActionSpec,
    ) -> Optional[str]:
        counts = self._ammo_counts(my_status)
        if not counts:
            return None

        current = str(my_status.get("ammo_loaded") or "").upper()

        if obs.enemy_visible:
            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            enemy_distance = obs.enemy_dist * max(vision_range, 1.0)

            if enemy_distance > 50.0:
                preferred = ["LONG_DISTANCE", "LIGHT", "HEAVY"]
            elif enemy_distance > 25.0:
                preferred = ["LIGHT", "LONG_DISTANCE", "HEAVY"]
            else:
                preferred = ["HEAVY", "LIGHT", "LONG_DISTANCE"]
        else:
            preferred = ["LIGHT", "HEAVY", "LONG_DISTANCE"]

        if current and counts.get(current, 0) > 0:
            if current in preferred:
                return current
            if not action.should_fire and not obs.enemy_visible:
                return current

        for ammo_name in preferred:
            if counts.get(ammo_name, 0) > 0:
                return ammo_name

        if current and counts.get(current, 0) > 0:
            return current

        return max(counts.items(), key=lambda item: item[1])[0]

    def _to_command(
        self,
        action_vec: np.ndarray,
        my_status: Dict[str, Any],
        obs: Observation,
    ) -> ActionCommand:
        move_speed = float(action_vec[0])
        heading_rotation = float(action_vec[1])
        # Actor no longer controls barrel rotation or firing directly.
        if self.mock_barrel_model is None or self.mock_shoot_model is None:
            raise ValueError("Mock barrel/shoot models are required to control barrel and fire.")
        barrel_rotation = 0.0
        should_fire = False
        action_stub = ActionSpec(
            name="ddpg",
            move_speed=move_speed,
            heading_rotation_angle=heading_rotation,
            barrel_rotation_angle=0.0,
            should_fire=False,
        )
        ammo_to_load = self._select_ammo_for_action(my_status, obs, action_stub)
        selected_ammo = ammo_to_load or str(my_status.get("ammo_loaded") or "").upper() or None
        ammo_range = self._ammo_range_for_name(selected_ammo)
        reload_norm = self._reload_norm_from_status(my_status)
        can_fire_now = self._can_fire_with_ammo(my_status, selected_ammo)
        mock_feats = self._build_mock_feature_vector(
            obs=obs,
            can_fire=can_fire_now,
            reload_norm=reload_norm,
            ammo_range=ammo_range,
        )
        mock_t = torch.from_numpy(mock_feats).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.mock_barrel_model is None:
                raise ValueError("No mock barrel model")
            barrel_norm = float(self.mock_barrel_model(mock_t).squeeze(0).squeeze(0).cpu().item())
            barrel_rotation = barrel_norm * MAX_BARREL_DELTA
            if self.mock_shoot_model is None:
                raise ValueError("No shooting model")
            shoot_logit = float(self.mock_shoot_model(mock_t).squeeze(0).squeeze(0).cpu().item())
            shoot_val = 1.0 / (1.0 + math.exp(-shoot_logit))
            should_fire = self._passes_fire_gates(
                shoot_val=shoot_val,
                obs=obs,
                my_status=my_status,
                ammo_name=selected_ammo,
            )
            if (
                (not should_fire)
                and self.locked_enemy_visible
                and self._passes_fire_safety_gates(
                    obs=obs,
                    my_status=my_status,
                    ammo_name=selected_ammo,
                )
            ):
                relaxed_thr = max(
                    0.35,
                    float(self.config.mock_shoot_threshold)
                    - float(self.config.target_lock_relaxed_shoot_margin),
                )
                if shoot_val >= relaxed_thr:
                    should_fire = True
            self.last_mock_action = np.array([barrel_norm, shoot_val], dtype=np.float32)

        action_stub = ActionSpec(
            name="ddpg",
            move_speed=move_speed,
            heading_rotation_angle=heading_rotation,
            barrel_rotation_angle=barrel_rotation,
            should_fire=should_fire,
        )
        ammo_to_load = self._select_ammo_for_action(my_status, obs, action_stub)
        fire_ammo = ammo_to_load or selected_ammo
        if should_fire and not self._passes_fire_gates(
            shoot_val=float(self.last_mock_action[1]),
            obs=obs,
            my_status=my_status,
            ammo_name=fire_ammo,
        ):
            should_fire = False
        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=move_speed,
            should_fire=should_fire,
            ammo_to_load=ammo_to_load,
        )

    def _compute_step_reward(
        self,
        prev_obs: Observation,
        current_obs: Observation,
        action: ActionCommand,
        enemies_remaining: int,
        current_tick: int,
        current_pos: Tuple[float, float],
        vision_range: float = 40.0,
        damage_taken: float = 0.0,
        hit_target: bool = False,
        friendly_hit: bool = False,
    ) -> float:
        from collections import defaultdict
        parts: Dict[str, float] = defaultdict(float)
        # Observation fields:
            # - vector
                # - hp_ratio
                # - shield_ratio
                # - reload_norm
                # - heavy_ratio
                # - light_ratio
                # - long_ratio
                # - enemy_visible (1.0/0.0)
                # - enemy_dist
                # - enemy_barrel_error
                # - enemy_hull_error
                # - ally_fire_risk (1.0/0.0)
                # - obstacle_ahead (1.0/0.0)
                # - danger_ahead (1.0/0.0)
                # - powerup_visible (1.0/0.0)
                # - powerup_dist
                # - speed_ratio
                # - enemies_remaining_norm
                # - x_norm
                # - y_norm
                # - dx_recent
                # - dy_recent
                # - dx_prev
                # - dy_prev
            # - enemy_visible: nearest enemy is visible
            # - enemy_dist: normalized distance to nearest enemy (0..1)
            # - enemy_barrel_error: normalized barrel angle error to enemy (0..1, 0.5=center)
            # - enemy_hull_error: normalized hull angle error to enemy (0..1, 0.5=center)
            # - ally_fire_risk: ally within firing line
            # - obstacle_ahead: obstacle within forward cone
            # - danger_ahead: dangerous terrain ahead
            # - powerup_visible: any powerup visible
            # - powerup_dist: normalized distance to nearest powerup (0..1)
            # - hp_ratio: hp / max_hp (0..1)
            # - shield_ratio: shield / max_shield (0..1)
            # - can_fire: reloaded + ammo available
            # - reload_norm: normalized reload timer (0..1)
        # Action fields:
            # barrel_rotation_angle: float = 0.0
            # heading_rotation_angle: float = 0.0
            # move_speed: float = 0.0
            # should_fire: bool = False

        hp_delta = current_obs.hp_ratio - prev_obs.hp_ratio

        # Keep target alignment rewards moderate so combat feedback can dominate.
        parts["approaching_enemy"] = -0.25 * abs(current_obs.enemy_hull_error - 0.5)
        if current_obs.enemy_visible:
            parts["barrel_aim"] = -0.25 * abs(current_obs.enemy_barrel_error - 0.5)

        progress_scale = float(self.config.progress_reward_scale)
        enemy_dist_delta_norm = float(prev_obs.enemy_dist - current_obs.enemy_dist)
        if math.isfinite(enemy_dist_delta_norm):
            enemy_dist_delta_world = enemy_dist_delta_norm * max(1.0, float(vision_range))
            enemy_dist_delta_world = max(-0.6, min(0.6, enemy_dist_delta_world))
            parts["enemy_progress"] = 0.35 * progress_scale * enemy_dist_delta_world
        if self.enemy_target_pos is not None and len(self.pos_history) >= 2:
            prev_pos = self.pos_history[-2]
            tx, ty = self.enemy_target_pos
            prev_target_dist = math.hypot(float(prev_pos[0]) - float(tx), float(prev_pos[1]) - float(ty))
            curr_target_dist = math.hypot(float(current_pos[0]) - float(tx), float(current_pos[1]) - float(ty))
            target_delta_world = prev_target_dist - curr_target_dist
            target_delta_world = max(-0.6, min(0.6, target_delta_world))
            parts["target_progress"] = 0.8 * progress_scale * target_delta_world
            if target_delta_world < 0.0:
                parts["wrong_direction"] = 0.4 * progress_scale * target_delta_world
        if (
            current_obs.hp_ratio <= float(self.config.retreat_hp_threshold)
            and current_obs.enemy_visible
            and current_obs.enemy_dist < 0.55
            and action.move_speed < 0.0
        ):
            retreat_strength = min(1.0, abs(float(action.move_speed)) / max(1.0, float(MAX_MOVE_SPEED)))
            parts["tactical_retreat"] = 0.12 * retreat_strength
        if not np.isclose(hp_delta, 0.0):
            parts["hp_loss"] = hp_delta * 5
        if damage_taken > 0.0:
            parts["damage_taken"] = -0.02 * float(damage_taken)
        if not prev_obs.danger_ahead and current_obs.danger_ahead:
            parts["danger_ahead"] = -0.3

        if hit_target:
            parts["enemy_hit"] = 2.5
        if friendly_hit:
            parts["friendly_fire"] = -3.0
        if action.should_fire:
            if not prev_obs.can_fire:
                parts["invalid_fire"] = -0.8
            elif prev_obs.shot_blocked:
                parts["blocked_fire"] = -0.4
            elif (not hit_target) and (not friendly_hit):
                parts["missed_shot"] = -0.15

        delta = action.heading_rotation_angle / MAX_HEADING_DELTA
        parts["rotation"] = -0.5 * (delta) ** 2

        recent = self.pos_history[-200:] or [current_pos]
        prev = self.pos_history[-400:-200] or recent
        spawn = self.pos_history[0] if self.pos_history else current_pos
        
        centroid = lambda pts: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        
        if current_tick < 200:
            rc = pc = spawn
        else:
            rc = centroid(recent)
            pc = spawn if current_tick < 400 else centroid(prev)

        var_r = sum((p[0] - rc[0]) ** 2 + (p[1] - rc[1]) ** 2 for p in recent) / len(recent)
        var_p = sum((p[0] - pc[0]) ** 2 + (p[1] - pc[1]) ** 2 for p in prev) / len(prev)

        exploration_scale = float(self.config.exploration_reward_scale)
        parts["variance_recent"] = exploration_scale * (var_r / 200.0)
        parts["variance_prev"] = exploration_scale * (var_p / 200.0)
        parts["centroid_bonus"] = exploration_scale * 0.2 * ((var_r + var_p) / 200.0)

        frontier_bonus = 0.0
        if self.frontier_min_x is None:
            self.frontier_min_x = current_pos[0]
            self.frontier_max_x = current_pos[0]
            self.frontier_min_y = current_pos[1]
            self.frontier_max_y = current_pos[1]
        else:                                       # wyjezdza w nieznane
            if current_pos[0] < self.frontier_min_x:
                frontier_bonus += (self.frontier_min_x - current_pos[0]) / MAP_WIDTH
                self.frontier_min_x = current_pos[0]
            if current_pos[0] > self.frontier_max_x:
                frontier_bonus += (current_pos[0] - self.frontier_max_x) / MAP_WIDTH
                self.frontier_max_x = current_pos[0]
            if current_pos[1] < self.frontier_min_y:
                frontier_bonus += (self.frontier_min_y - current_pos[1]) / MAP_HEIGHT
                self.frontier_min_y = current_pos[1]
            if current_pos[1] > self.frontier_max_y:
                frontier_bonus += (current_pos[1] - self.frontier_max_y) / MAP_HEIGHT
                self.frontier_max_y = current_pos[1]
        parts["frontier_bonus"] = exploration_scale * frontier_bonus

        reward = sum(parts.values())
        self.episode_reward_total += reward
        for key, value in parts.items():
            self.episode_reward_parts[key] = self.episode_reward_parts.get(key, 0.0) + value
        self.episode_reward_parts_steps += 1

        return reward


    def _maybe_train(self) -> None:
        # min_required = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size,
            self.device,
        )

        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            next_action = torch.tanh(next_raw)
            next_action = self._scale_action_tensor(next_action)
            next_action = torch.sigmoid(next_action)
            next_q = self.critic_target(torch.cat([next_states, next_action], dim=1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        actions = torch.sigmoid(actions)
        current_q = self.critic(torch.cat([states, actions], dim=1)).squeeze(1)
        critic_loss = F.mse_loss(current_q, target_q)
        self.last_loss = float(critic_loss.item())

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_raw = self.actor(states)
        actor_action = torch.tanh(actor_raw)
        actor_action = self._scale_action_tensor(actor_action)
        actor_action = torch.sigmoid(actor_action)
        raw_penalty = 5e-2 * actor_raw.pow(2).mean()
        actor_loss = -self.critic(torch.cat([states, actor_action], dim=1)).mean() + raw_penalty

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
        self.actor_optimizer.step()

        tau = float(self.config.tau)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        if self.experiment and (self.train_steps % COMET_LOG_EVERY == 0):
            self.experiment.log_metric("critic_loss", self.last_loss, step=self.train_steps)
            self.experiment.log_metric("actor_loss", float(actor_loss.item()), step=self.train_steps)
            self.experiment.log_metric("q_mean", current_q.mean().item(), step=self.train_steps)
            self.experiment.log_metric("action_noise_std", self._current_action_noise_std(), step=self.train_steps)

        self.train_steps += 1

    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "best_score": self.best_score,
            "config": {
                "state_dim": self.config.state_dim,
                "n_rules": self.config.n_rules,
                "mf_type": self.config.mf_type,
            },
        }

    def save_checkpoint(self, path: Optional[str] = None, label: str = "checkpoint") -> None:
        save_path = path or self.config.model_path
        if not save_path:
            return

        model_dir = os.path.dirname(save_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        torch.save(self._checkpoint_payload(), save_path)
        print(f"[{self.name}] {label} saved: {save_path}")

    def _load_checkpoint_if_available(self) -> None:
        path = self.config.model_path
        if not path or not os.path.exists(path):
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
                saved_conf = checkpoint.get("config", {})
                if saved_conf:
                    if saved_conf.get("n_rules") != self.config.n_rules:
                        print(f"[{self.name}] WARNING: Rules count mismatch! Saved: {saved_conf.get('n_rules')}, Current: {self.config.n_rules}")
                    if saved_conf.get("mf_type") != self.config.mf_type:
                        print(f"[{self.name}] WARNING: MF type mismatch! Saved: {saved_conf.get('mf_type')}, Current: {self.config.mf_type}")

                self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                self.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)

                self.actor_target.load_state_dict(checkpoint.get("actor_target_state_dict", checkpoint["actor_state_dict"]), strict=True)
                self.critic_target.load_state_dict(checkpoint.get("critic_target_state_dict", checkpoint["critic_state_dict"]), strict=True)

                actor_opt = checkpoint.get("actor_optimizer_state_dict")
                critic_opt = checkpoint.get("critic_optimizer_state_dict")
                if actor_opt:
                    self.actor_optimizer.load_state_dict(actor_opt)
                if critic_opt:
                    self.critic_optimizer.load_state_dict(critic_opt)

                self.total_steps = int(checkpoint.get("total_steps", 0) or 0)
                self.train_steps = int(checkpoint.get("train_steps", 0) or 0)
                self.games_played = int(checkpoint.get("games_played", 0) or 0)
                self.best_score = float(checkpoint.get("best_score", float("-inf")))
            else:
                print(f"[{self.name}] checkpoint format not recognized; starting fresh.")

            print(f"[{self.name}] checkpoint loaded: {path}")
        except Exception as exc:
            print(f"[{self.name}] failed to load checkpoint {path}: {exc}")

    def _reset_episode_state(self) -> None:
        self.last_observation = None
        self.last_action_vector = None
        self.last_command = ActionCommand()
        self.prev_enemies_remaining = None
        self.current_episode_score = 0.0
        self.last_status = None
        self.trace_positions.clear()
        self.trace_hp.clear()
        self.trace_shots.clear()
        self.pending_shot_indices.clear()
        self.trace_allies.clear()
        self.trace_enemies.clear()
        self.trace_labels.clear()
        self.pos_history.clear()
        self.trace_actor_raw.clear()
        self.last_actor_raw = None
        self.trace_mock_actions.clear()
        self.last_mock_action = None
        self.trace_damage_taken.clear()
        self.trace_hit_target.clear()
        self.trace_friendly_hit.clear()
        self.trace_should_fire.clear()
        self.episode_reward_total = 0.0
        self.episode_reward_parts = {}
        self.episode_reward_parts_steps = 0
        self.frontier_min_x = None
        self.frontier_max_x = None
        self.frontier_min_y = None
        self.frontier_max_y = None
        self.enemy_target_pos = None
        self.enemy_base_pos = None
        self.nav_waypoint = None
        self.nav_waypoint_age = 0
        self.in_enemy_zone_patrol = False
        self.patrol_orbit_angle_deg = None
        self.patrol_orbit_direction = 1.0
        self.locked_enemy_id = None
        self.locked_enemy_last_seen_tick = -10_000
        self.locked_enemy_last_pos = None
        self.locked_enemy_visible = False

    def _resolve_pending_shot_feedback(self, hit_target: bool, friendly_hit: bool) -> None:
        if not self.pending_shot_indices:
            return
        shot_idx = self.pending_shot_indices.popleft()
        if shot_idx < 0 or shot_idx >= len(self.trace_shots):
            return
        if friendly_hit:
            outcome = "friendly_hit"
        elif hit_target:
            outcome = "enemy_hit"
        else:
            outcome = "miss"
        self.trace_shots[shot_idx].outcome = outcome

    def _record_trace(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        current_obs: Observation,
        action: ActionSpec,
        damage_taken: float,
        hit_target: bool,
        friendly_hit: bool,
    ) -> None:
        pos = my_status.get("position")
        if not pos or "x" not in pos or "y" not in pos:
            raise ValueError(f"Plotting {pos=}")
        x = float(pos.get("x") or 0.0)
        y = float(pos.get("y") or 0.0)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Plotting non-finite position x={x} y={y}")
        hp = float(current_obs.hp_ratio)
        if not math.isfinite(hp):
            raise ValueError(f"Plotting non-finite hp={hp}")
        # Engine feedback is for previous physics tick, so resolve previous shot now.
        self._resolve_pending_shot_feedback(hit_target=hit_target, friendly_hit=friendly_hit)
        self.trace_positions.append((x, y))
        self.trace_hp.append(hp)
        self.trace_labels.append(
            f"actor: v={action.move_speed:.2f} h={action.heading_rotation_angle:.2f} "
            f"mock: b={action.barrel_rotation_angle:.2f} f={int(action.should_fire)}"
        )
        self.trace_damage_taken.append(float(max(0.0, damage_taken)))
        self.trace_hit_target.append(1.0 if hit_target else 0.0)
        self.trace_friendly_hit.append(1.0 if friendly_hit else 0.0)
        self.trace_should_fire.append(1.0 if action.should_fire else 0.0)

        seen_tanks = sensor_data.get("seen_tanks", [])
        my_team = my_status.get("_team")
        for tank in seen_tanks:
            tpos = tank.get("position")
            if not tpos or "x" not in tpos or "y" not in tpos:
                raise ValueError(f'Plotting {tpos=}')
            tx = float(tpos.get("x") or 0.0)
            ty = float(tpos.get("y") or 0.0)
            if not (math.isfinite(tx) and math.isfinite(ty)):
                raise ValueError(f"Plotting non-finite tank position x={tx} y={ty} tpos={tpos}")
            if tank.get("team") == my_team:
                self.trace_allies.append((tx, ty))
            else:
                self.trace_enemies.append((tx, ty))

        if action.should_fire:
            heading = float(my_status.get("heading", 0.0) or 0.0)
            barrel = float(my_status.get("barrel_angle", 0.0) or 0.0)
            shot_idx = len(self.trace_shots)
            self.trace_shots.append(
                ShotTrace(
                    x=x,
                    y=y,
                    angle_deg=heading + barrel,
                    step_idx=len(self.trace_positions) - 1,
                )
            )
            self.pending_shot_indices.append(shot_idx)

    def _save_episode_plot(self) -> None:
        if len(self.trace_positions) == 0:
            raise ValueError("Plotting: no trace positions")

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        hps = self.trace_hp
        fig, ax = plt.subplots(figsize=(8, 8))
        segments = []
        colors = []
        for i in range(len(self.trace_positions) - 1):
            segments.append([self.trace_positions[i], self.trace_positions[i + 1]])
            colors.append((hps[i] + hps[i + 1]) * 0.5)
        lc = LineCollection(segments, cmap="RdYlGn", linewidths=2.0)
        lc.set_array(np.array(colors, dtype=np.float32))
        ax.add_collection(lc)

        xs = [p[0] for p in self.trace_positions]
        ys = [p[1] for p in self.trace_positions]
        sc_self = ax.scatter(xs, ys, c=hps, cmap="RdYlGn", s=10, alpha=0.7, label="self")

        if self.trace_allies:
            ax.scatter(
                [p[0] for p in self.trace_allies],
                [p[1] for p in self.trace_allies],
                c="blue",
                s=12,
                alpha=0.6,
                label="allies",
            )
            ax_x, ax_y = self.trace_allies[0]
            ax.text(ax_x, ax_y, "A", color="blue", fontsize=7, ha="left", va="bottom")
        if self.trace_enemies:
            ax.scatter(
                [p[0] for p in self.trace_enemies],
                [p[1] for p in self.trace_enemies],
                c="black",
                s=12,
                alpha=0.6,
                label="enemies",
            )
            ex, ey = self.trace_enemies[0]
            ax.text(ex, ey, "E", color="black", fontsize=7, ha="left", va="bottom")
        if self.trace_shots:
            shot_colors = {
                "enemy_hit": "#2ca02c",
                "friendly_hit": "#d62728",
                "miss": "#b58900",
                "pending": "#7f7f7f",
            }
            shot_labels = {
                "enemy_hit": "shot: enemy hit",
                "friendly_hit": "shot: friendly hit",
                "miss": "shot: miss/no-hit",
                "pending": "shot: pending",
            }
            shot_counts = {k: 0 for k in shot_colors.keys()}
            added_labels: set[str] = set()
            for shot in self.trace_shots:
                outcome = shot.outcome if shot.outcome in shot_colors else "pending"
                shot_counts[outcome] += 1
                dx = math.cos(math.radians(shot.angle_deg)) * 30.0
                dy = math.sin(math.radians(shot.angle_deg)) * 30.0
                label = None
                if outcome not in added_labels:
                    label = shot_labels[outcome]
                    added_labels.add(outcome)
                ax.arrow(
                    shot.x,
                    shot.y,
                    dx,
                    dy,
                    color=shot_colors[outcome],
                    width=0.12,
                    head_width=0.6,
                    head_length=1.0,
                    length_includes_head=True,
                    alpha=0.9,
                    label=label,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, MAP_WIDTH)
        ax.set_ylim(0.0, MAP_HEIGHT)
        ax.autoscale(False)
        if self.trace_shots:
            ax.set_title(
                f"{self.name} trajectory (game {self.games_played}) | "
                f"shots: enemy_hit={shot_counts['enemy_hit']}, "
                f"friendly_hit={shot_counts['friendly_hit']}, "
                f"miss={shot_counts['miss']}, pending={shot_counts['pending']}"
            )
        else:
            ax.set_title(f"{self.name} trajectory (game {self.games_played}) | mock barrel+shoot")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        cbar = fig.colorbar(sc_self, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("hp (normalized)")
        ax.grid(True, alpha=0.3)

        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"game_{self.games_played}_agent_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_action(
        self,
        current_tick: int,
        my_tank_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
        damage_taken: float = 0.0,
        hit_target: bool = False,
        friendly_hit: bool = False,
    ) -> ActionCommand:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent get_action call detected")
        try:
            pos = my_tank_status.get("position")
            if not pos or "x" not in pos or "y" not in pos:
                raise ValueError(f"Missing position in my_tank_status: {pos}")
            current_pos = (float(pos["x"]), float(pos["y"]))
            my_team = my_tank_status.get("_team")
            if self.enemy_base_pos is None and my_team in (1, 2):
                self.enemy_base_pos = self._enemy_base_anchor(my_team)
            if self.enemy_target_pos is None:
                self.enemy_target_pos = self.enemy_base_pos
            if self.enemy_target_pos is None:
                raise ValueError("enemy_target_pos is None; spawn point not initialized")
            seen_tanks = sensor_data.get("seen_tanks", [])
            focus_enemy, focus_enemy_id, focus_pos, focus_visible = self._select_focus_enemy(
                my_pos=current_pos,
                my_team=my_team,
                seen_tanks=seen_tanks,
                current_tick=current_tick,
            )
            if focus_pos is not None:
                self.enemy_target_pos = focus_pos
            elif self.enemy_base_pos is not None:
                self.enemy_target_pos = self.enemy_base_pos
            current_obs = self.encoder.encode(
                my_tank_status,
                sensor_data,
                enemies_remaining,
                enemy_target_pos=self.enemy_target_pos,
                preferred_enemy_id=(focus_enemy_id if focus_visible else None),
            )
            damage_taken_value = self._safe_nonnegative_float(damage_taken, default=0.0)
            hit_target_value = self._safe_bool(hit_target, default=False)
            friendly_hit_value = self._safe_bool(friendly_hit, default=False)
            x_norm = current_pos[0] / MAP_WIDTH
            y_norm = current_pos[1] / MAP_HEIGHT

            recent = self.pos_history[-200:] or [current_pos]
            prev = self.pos_history[-400:-200] or recent
            rcx = sum(p[0] for p in recent) / float(len(recent))
            rcy = sum(p[1] for p in recent) / float(len(recent))
            pcx = sum(p[0] for p in prev) / float(len(prev))
            pcy = sum(p[1] for p in prev) / float(len(prev))

            dx_recent = (current_pos[0] - rcx) / MAP_WIDTH
            dy_recent = (current_pos[1] - rcy) / MAP_HEIGHT
            dx_prev = (current_pos[0] - pcx) / MAP_WIDTH
            dy_prev = (current_pos[1] - pcy) / MAP_HEIGHT
            dx_recent = (dx_recent + 1.0) * 0.5
            dy_recent = (dy_recent + 1.0) * 0.5
            dx_prev = (dx_prev + 1.0) * 0.5
            dy_prev = (dy_prev + 1.0) * 0.5
            current_obs.vector = np.concatenate(
                [current_obs.vector, np.array([x_norm, y_norm, dx_recent, dy_recent, dx_prev, dy_prev], dtype=np.float32)]
            )
            self.pos_history.append(current_pos)

            if (
                self.config.frame_skip > 1
                and self.last_action_vector is not None
                and current_tick % self.config.frame_skip != 0
            ):
                self._record_trace(
                    my_tank_status,
                    sensor_data,
                    current_obs,
                    ActionSpec("ddpg", self.last_command.move_speed, self.last_command.heading_rotation_angle, self.last_command.barrel_rotation_angle, self.last_command.should_fire),
                    damage_taken=damage_taken_value,
                    hit_target=hit_target_value,
                    friendly_hit=friendly_hit_value,
                )
                return self.last_command

            if current_tick > 0:
                if self.last_observation is None or self.last_action_vector is None:
                    if current_tick <= 1:
                        pass
                    elif current_tick > 10:
                        print(
                            f"[{self.name}] WARNING: missing previous state/action at tick={current_tick}"
                        )
                    else:
                        raise ValueError("Missing previous state/action at tick > 0")
                else:
                    reward = self._compute_step_reward(
                        prev_obs=self.last_observation,
                        current_obs=current_obs,
                        action=self.last_command,
                        enemies_remaining=enemies_remaining,
                        current_tick=current_tick,
                        current_pos=current_pos,
                        vision_range=float(my_tank_status.get("_vision_range", 40.0) or 40.0),
                        damage_taken=damage_taken_value,
                        hit_target=hit_target_value,
                        friendly_hit=friendly_hit_value,
                    )
                    self.current_episode_score += reward
                    self.replay.add(
                        state=self.last_observation.vector,
                        action=self.last_action_vector,
                        reward=reward,
                        next_state=current_obs.vector,
                        done=0.0,
                    )
                    self._maybe_train()

            action_vec = self._select_action(current_obs.vector, training=self.training_enabled)
            action_vec = self._apply_movement_bootstrap(
                action_vec=action_vec,
                my_status=my_tank_status,
                obs=current_obs,
            )
            command = self._to_command(action_vec, my_tank_status, current_obs)

            if command.should_fire:
                self.last_fire_tick = current_tick

            self.last_observation = current_obs
            self.last_action_vector = action_vec
            self.last_command = command
            self.prev_enemies_remaining = enemies_remaining
            self.total_steps += 1
            self.last_status = my_tank_status
            self._record_trace(
                my_tank_status,
                sensor_data,
                current_obs,
                ActionSpec("ddpg", action_vec[0], action_vec[1], command.barrel_rotation_angle, command.should_fire),
                damage_taken=damage_taken_value,
                hit_target=hit_target_value,
                friendly_hit=friendly_hit_value,
            )
            if self.last_actor_raw is not None:
                self.trace_actor_raw.append(self.last_actor_raw.copy())
            if self.last_mock_action is not None:
                self.trace_mock_actions.append(self.last_mock_action.copy())

            return command
        finally:
            self.lock.release()

    def destroy(self, payload: Optional[Dict[str, Any]] = None) -> None:
        with self.lock:
            print(f"[{self.name}] destroyed")
            if self.last_observation is not None:
                tags = []
                if self.last_observation.danger_ahead:
                    tags.append("rough_terrain")
                if self.last_observation.enemy_visible:
                    tags.append("shot_likely")
                if self.last_observation.ally_fire_risk:
                    tags.append("ally_fire_risk")
                if self.last_observation.obstacle_ahead:
                    tags.append("obstacle_ahead")
                if not tags:
                    tags.append("unknown")
                print(f"[{self.name}] destroy_context: {','.join(tags)}, {self.last_observation}\n\n")
            status = self.last_status or {}
            print(f"[{self.name}] destroy_state: hp={status.get('hp')} shield={status.get('shield')}")
            if payload:
                print(f"[{self.name}] destroy_reason: {payload.get('cause')} damage_events={payload.get('damage_events')}")
            if self.last_observation is not None and self.last_action_vector is not None:
                self.current_episode_score -= 8.0
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_vector,
                    reward=-8.0,
                    next_state=self.last_observation.vector,
                    done=1.0,
                )
                for _ in range(3):
                    self._maybe_train()

            self.was_destroyed = True
            # Do not reset trace here; /agent/end is called after destroy
            # and should finalize the episode plot.

    def end(self, damage_dealt: float, tanks_killed: int) -> None:
        with self.lock:
            final_reward = tanks_killed * 4.0 + (damage_dealt / 75.0)
            if not self.was_destroyed:
                final_reward += 1.5

            self.current_episode_score += final_reward
            self.episode_reward_total += final_reward
            if self.last_observation is not None and self.last_action_vector is not None:
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_vector,
                    reward=final_reward,
                    next_state=self.last_observation.vector,
                    done=1.0,
                )
                for _ in range(5):
                    self._maybe_train()

            self.last_episode_score = self.current_episode_score
            self.games_played += 1

            print(
                f"[{self.name}] end | games={self.games_played} "
                f"damage={damage_dealt:.1f} kills={tanks_killed} "
                f"replay={len(self.replay)} train_steps={self.train_steps}"
            )

            if self.experiment:
                step = self.train_steps
                self.experiment.log_metric("total_episode_reward", self.current_episode_score, step=step)
                self.experiment.log_metric("damage_dealt", damage_dealt, step=step)
                self.experiment.log_metric("tanks_killed", tanks_killed, step=step)
                self.experiment.log_metric("was_destroyed", int(self.was_destroyed), step=step)
                self.experiment.log_metric("replay_size", len(self.replay), step=step)

            if self.training_enabled and self.games_played % max(1, self.config.save_every_games) == 0:
                episode_score = final_reward + (4.0 * tanks_killed) + (damage_dealt / 40.0)
                if episode_score > self.best_score:
                    self.best_score = episode_score
                    self.save_checkpoint(self.best_model_path, label="best")

                # Save latest after potential best-score update, so
                # best_score persists across process restarts.
                self.save_checkpoint(self.config.model_path, label="latest")

            self._save_episode_plot()
            self._save_actor_raw_plot()
            self._save_combat_feedback_plot()
            steps = max(1, self.episode_reward_parts_steps)
            episode_parts = {k: v / steps for k, v in self.episode_reward_parts.items()}
            episode_parts["total_reward_avg"] = self.episode_reward_total / steps
            all_keys = set(self.reward_parts_history) | set(episode_parts)
            history_len = max((len(v) for v in self.reward_parts_history.values()), default=0)
            for key in all_keys:
                self.reward_parts_history.setdefault(key, [])
                # When a new reward component appears mid-training, backfill
                # previous episodes with zeros so all series keep equal length.
                if len(self.reward_parts_history[key]) < history_len:
                    self.reward_parts_history[key].extend(
                        [0.0] * (history_len - len(self.reward_parts_history[key]))
                    )
                self.reward_parts_history[key].append(episode_parts.get(key, 0.0))
            self._save_reward_plot()
            
            self.was_destroyed = False
            self._reset_episode_state()

    def _save_reward_plot(self) -> None:
        if not self.reward_parts_history:
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        first_key = next(iter(self.reward_parts_history))
        xs = list(range(1, len(self.reward_parts_history[first_key]) + 1))
        colors = list(plt.cm.tab20.colors)
        for idx, (label, ys) in enumerate(self.reward_parts_history.items()):
            local_xs = list(range(1, len(ys) + 1))
            ax.plot(
                local_xs,
                ys,
                label=label,
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=colors[idx % len(colors)],
            )
        ax.set_xlabel("Game")
        ax.set_ylabel("Reward")
        ax.set_title(f"{self.name} reward history")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            fontsize=7,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.8,
            borderaxespad=0.2,
        )
        fig.subplots_adjust(bottom=0.25)
        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"rewards_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        last_total = None
        if "total_reward_avg" in self.reward_parts_history and self.reward_parts_history["total_reward_avg"]:
            last_total = self.reward_parts_history["total_reward_avg"][-1]
        suffix = f" last_total={last_total:.3f}" if last_total is not None else ""
        print(f"[{self.name}] reward plot saved: {out_path}{suffix}")

    def _save_actor_raw_plot(self) -> None:
        if not self.trace_actor_raw and not self.trace_mock_actions:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        window = 5
        if self.trace_actor_raw:
            data = np.stack(self.trace_actor_raw, axis=0)
            labels = [
                "actor_move_raw",
                "actor_heading_raw",
            ]
            for idx in range(data.shape[1]):
                label = labels[idx] if idx < len(labels) else f"raw_{idx}"
                series = data[:, idx]
                if len(series) >= window:
                    kernel = np.ones(window, dtype=np.float32) / float(window)
                    series = np.convolve(series, kernel, mode="same")
                    label = f"{label}"
                ax.plot(series, label=label)
        if self.trace_mock_actions:
            mock_data = np.stack(self.trace_mock_actions, axis=0)
            mock_labels = [
                "mock barrel norm",
                "mock shoot val",
            ]
            for idx in range(mock_data.shape[1]):
                label = mock_labels[idx] if idx < len(mock_labels) else f"mock_{idx}"
                series = mock_data[:, idx]
                if len(series) >= window:
                    kernel = np.ones(window, dtype=np.float32) / float(window)
                    series = np.convolve(series, kernel, mode="same")
                ax.plot(series, label=f'{label}')
        ax.set_title(f"{self.name} actor raw outputs (game {self.games_played})")
        ax.set_xlabel("step")
        ax.set_ylabel("raw value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"actor_raw_{self.games_played}_agent_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{self.name}] actor raw plot saved: {out_path}")

    def _save_combat_feedback_plot(self) -> None:
        if (
            not self.trace_damage_taken
            and not self.trace_hit_target
            and not self.trace_friendly_hit
            and not self.trace_should_fire
        ):
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        if self.trace_damage_taken:
            damage = np.array(self.trace_damage_taken, dtype=np.float32)
            ax1.plot(damage, label="damage_taken_step", linewidth=1.2)
            ax1.plot(np.cumsum(damage), label="damage_taken_cum", linewidth=1.0)
        ax1.set_ylabel("Damage")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        should_fire = np.array(self.trace_should_fire, dtype=np.int32) if self.trace_should_fire else np.zeros((0,), dtype=np.int32)
        hit = np.array(self.trace_hit_target, dtype=np.int32) if self.trace_hit_target else np.zeros((0,), dtype=np.int32)
        friendly = np.array(self.trace_friendly_hit, dtype=np.int32) if self.trace_friendly_hit else np.zeros((0,), dtype=np.int32)
        # Feedback arrives one tick later (tick t carries outcomes from tick t-1).
        n = min(
            max(0, len(should_fire) - 1),
            max(0, len(hit) - 1),
            max(0, len(friendly) - 1),
        )
        pending_last_fire = int(len(should_fire) > 0 and should_fire[-1] == 1)
        if n == 0:
            cm = np.zeros((2, 3), dtype=np.int32)
        else:
            sf = should_fire[:n]
            ht = hit[1:1 + n]
            fh = friendly[1:1 + n]
            # Rows: shot command (no/yes), Cols: outcome (miss/no-hit, enemy hit, friendly hit).
            cm = np.array(
                [
                    [
                        int(np.sum((sf == 0) & (ht == 0) & (fh == 0))),
                        int(np.sum((sf == 0) & (ht == 1) & (fh == 0))),
                        int(np.sum((sf == 0) & (fh == 1))),
                    ],
                    [
                        int(np.sum((sf == 1) & (ht == 0) & (fh == 0))),
                        int(np.sum((sf == 1) & (ht == 1) & (fh == 0))),
                        int(np.sum((sf == 1) & (fh == 1))),
                    ],
                ],
                dtype=np.int32,
            )

        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(["Miss/NoHit", "Enemy Hit", "Friendly Hit"])
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["No Fire", "Fire"])
        ax2.set_xlabel("Actual Outcome")
        ax2.set_ylabel("Shot Command")
        for i in range(2):
            for j in range(3):
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fired_enemy_hits = int(cm[1, 1])
        fired_friendly_hits = int(cm[1, 2])
        fired_misses = int(cm[1, 0])
        ax2.set_title(
            "Shooting Confusion Matrix | "
            f"enemy_hit={fired_enemy_hits}, friendly_hit={fired_friendly_hits}, "
            f"miss={fired_misses}, pending={pending_last_fire}"
        )
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        fig.suptitle(f"{self.name} combat feedback (game {self.games_played})")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"combat_feedback_{self.games_played}_agent_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{self.name}] combat feedback plot saved: {out_path}")

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "training_enabled": self.training_enabled,
            "map_name": self.map_name,
            "steps": self.total_steps,
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "replay_size": len(self.replay),
            "last_loss": self.last_loss,
            "model_path": self.config.model_path,
            "best_model_path": self.best_model_path,
            "best_score": self.best_score,
            "last_episode_score": self.last_episode_score,
        }


app = FastAPI(
    title="Fuzzy DQN Agent",
    description="ANFIS-based DQN agent for tank battles",
    version="2.0.0",
)

agent: Optional[FuzzyDQNAgent] = None


def _get_agent() -> FuzzyDQNAgent:
    global agent
    if agent is None:
        agent = FuzzyDQNAgent(
            name="FuzzyDQN",
            config=AgentConfig(model_path=""),
            training_enabled=False,
            load_checkpoint=False,
        )
    return agent


@app.get("/")
async def root() -> Dict[str, Any]:
    return _get_agent().status()


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)) -> ActionCommand:
    damage_taken = FuzzyDQNAgent._safe_nonnegative_float(payload.get("damage_taken", 0.0), default=0.0)
    hit_target = FuzzyDQNAgent._safe_bool(payload.get("hit_target", False), default=False)
    friendly_hit = FuzzyDQNAgent._safe_bool(payload.get("friendly_hit", False), default=False)
    return _get_agent().get_action(
        current_tick=int(payload.get("current_tick", 0) or 0),
        my_tank_status=payload.get("my_tank_status", {}),
        sensor_data=payload.get("sensor_data", {}),
        enemies_remaining=int(payload.get("enemies_remaining", 0) or 0),
        damage_taken=damage_taken,
        hit_target=hit_target,
        friendly_hit=friendly_hit,
    )


@app.post("/agent/context")
async def update_context(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    map_name = str(payload.get("map_name", "") or "")
    agent_obj = _get_agent()
    with agent_obj.lock:
        agent_obj.map_name = map_name
    return {"ok": True, "map_name": map_name}


@app.post("/agent/destroy", status_code=204, response_model=None)
async def destroy(payload: Dict[str, Any] = Body(None)) -> None:
    _get_agent().destroy(payload)


@app.post("/agent/end", status_code=204, response_model=None)
async def end(payload: Dict[str, Any] = Body(...)) -> None:
    _get_agent().end(
        damage_dealt=float(payload.get("damage_dealt", 0.0) or 0.0),
        tanks_killed=int(payload.get("tanks_killed", 0) or 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fuzzy DQN tank agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--train", action="store_true", help="Enable online learning")

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--best-model-path", type=str, default=None)
    parser.add_argument("--rules", type=int, default=32)
    parser.add_argument("--mf-type", choices=["gaussian", "bell", "triangular"], default="triangular")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every-games", type=int, default=1)
    parser.add_argument("--mock-barrel-model-path", type=str, default="./anfis_barrel_model.pt")
    parser.add_argument("--mock-shoot-model-path", type=str, default="./anfis_shoot_model.pt")
    parser.add_argument("--mock-shoot-threshold", type=float, default=0.55)
    parser.add_argument("--mock-half-angle-deg", type=float, default=float(MOCK_DEFAULT_HALF_ANGLE))
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--target-sync-every", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--action-noise-start", type=float, default=0.3)
    parser.add_argument("--action-noise-end", type=float, default=0.05)
    parser.add_argument("--action-noise-decay-steps", type=int, default=50_000)
    parser.add_argument("--movement-bootstrap-episodes", type=int, default=30)
    parser.add_argument("--movement-bootstrap-alpha-start", type=float, default=0.8)
    parser.add_argument("--movement-bootstrap-alpha-end", type=float, default=0.0)
    parser.add_argument("--movement-bootstrap-visible-scale", type=float, default=0.65)
    parser.add_argument("--movement-bootstrap-hidden-scale", type=float, default=1.0)
    parser.add_argument("--movement-waypoint-min-distance", type=float, default=20.0)
    parser.add_argument("--movement-waypoint-max-distance", type=float, default=65.0)
    parser.add_argument("--movement-waypoint-lateral-max", type=float, default=32.0)
    parser.add_argument("--movement-waypoint-replan-ticks", type=int, default=120)
    parser.add_argument("--movement-waypoint-reach-radius", type=float, default=9.0)
    parser.add_argument("--movement-waypoint-stuck-window", type=int, default=70)
    parser.add_argument("--movement-waypoint-stuck-distance", type=float, default=4.0)
    parser.add_argument("--movement-enemy-zone-patrol-activate-distance", type=float, default=30.0)
    parser.add_argument("--movement-enemy-zone-patrol-exit-distance", type=float, default=48.0)
    parser.add_argument("--movement-enemy-zone-patrol-radius-min", type=float, default=10.0)
    parser.add_argument("--movement-enemy-zone-patrol-radius-max", type=float, default=22.0)
    parser.add_argument("--movement-enemy-zone-patrol-replan-ticks", type=int, default=28)
    parser.add_argument("--movement-enemy-zone-patrol-step-deg", type=float, default=60.0)
    parser.add_argument("--target-lock-lost-patience-ticks", type=int, default=90)
    parser.add_argument("--target-lock-relaxed-shoot-margin", type=float, default=0.15)
    parser.add_argument("--progress-reward-scale", type=float, default=0.9)
    parser.add_argument("--exploration-reward-scale", type=float, default=0.25)
    parser.add_argument("--retreat-hp-threshold", type=float, default=0.35)
    parser.add_argument("--no-load", action="store_true", help="Do not load checkpoints on startup.")
    parser.add_argument("--map-name", type=str, default="", help="Current map identifier (for context/debugging).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = AgentConfig(
        n_rules=max(4, int(args.rules)),
        mf_type=args.mf_type,
        frame_skip=max(1, int(args.frame_skip)),
        model_path=args.model_path,
        best_model_path=args.best_model_path,
        mock_barrel_model_path=args.mock_barrel_model_path,
        mock_shoot_model_path=args.mock_shoot_model_path,
        mock_shoot_threshold=float(args.mock_shoot_threshold),
        mock_half_angle_deg=float(args.mock_half_angle_deg),
        seed=int(args.seed),
        save_every_games=max(1, int(args.save_every_games)),
        warmup_steps=max(0, int(args.warmup_steps)),
        batch_size=max(16, int(args.batch_size)),
        train_every=max(1, int(args.train_every)),
        target_sync_every=max(1, int(args.target_sync_every)),
        gamma=float(args.gamma),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        tau=float(args.tau),
        action_noise_start=float(args.action_noise_start),
        action_noise_end=float(args.action_noise_end),
        action_noise_decay_steps=max(1, int(args.action_noise_decay_steps)),
        movement_bootstrap_episodes=max(0, int(args.movement_bootstrap_episodes)),
        movement_bootstrap_alpha_start=float(args.movement_bootstrap_alpha_start),
        movement_bootstrap_alpha_end=float(args.movement_bootstrap_alpha_end),
        movement_bootstrap_visible_scale=float(args.movement_bootstrap_visible_scale),
        movement_bootstrap_hidden_scale=float(args.movement_bootstrap_hidden_scale),
        movement_waypoint_min_distance=float(args.movement_waypoint_min_distance),
        movement_waypoint_max_distance=float(args.movement_waypoint_max_distance),
        movement_waypoint_lateral_max=float(args.movement_waypoint_lateral_max),
        movement_waypoint_replan_ticks=max(1, int(args.movement_waypoint_replan_ticks)),
        movement_waypoint_reach_radius=float(args.movement_waypoint_reach_radius),
        movement_waypoint_stuck_window=max(5, int(args.movement_waypoint_stuck_window)),
        movement_waypoint_stuck_distance=float(args.movement_waypoint_stuck_distance),
        movement_enemy_zone_patrol_activate_distance=float(args.movement_enemy_zone_patrol_activate_distance),
        movement_enemy_zone_patrol_exit_distance=float(args.movement_enemy_zone_patrol_exit_distance),
        movement_enemy_zone_patrol_radius_min=float(args.movement_enemy_zone_patrol_radius_min),
        movement_enemy_zone_patrol_radius_max=float(args.movement_enemy_zone_patrol_radius_max),
        movement_enemy_zone_patrol_replan_ticks=max(1, int(args.movement_enemy_zone_patrol_replan_ticks)),
        movement_enemy_zone_patrol_step_deg=float(args.movement_enemy_zone_patrol_step_deg),
        target_lock_lost_patience_ticks=max(0, int(args.target_lock_lost_patience_ticks)),
        target_lock_relaxed_shoot_margin=float(args.target_lock_relaxed_shoot_margin),
        progress_reward_scale=float(args.progress_reward_scale),
        exploration_reward_scale=float(args.exploration_reward_scale),
        retreat_hp_threshold=float(args.retreat_hp_threshold),
        map_name=str(args.map_name or ""),
    )

    agent_name = args.name or f"FuzzyDQN_{args.port}"

    # Replace default global agent with runtime configuration.
    agent = FuzzyDQNAgent(
        name=agent_name,
        config=config,
        training_enabled=bool(args.train),
        load_checkpoint=not bool(args.no_load),
    )

    print(
        f"Starting {agent_name} on {args.host}:{args.port} "
        f"| train={args.train} | model={config.model_path}"
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
