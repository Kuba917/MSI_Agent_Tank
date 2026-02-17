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

import argparse
import math
import os
import random
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn

from ANFISDQN import ANFISDQN


# Add engine paths for local runs.
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA", "controller")
engine_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA")
sys.path.insert(0, controller_dir)
sys.path.insert(0, engine_dir)


STATE_DIM = 17
DEFAULT_MODEL_PATH = os.path.join(current_dir, "fuzzy_dqn_model.pt")


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


ACTION_SPACE: List[ActionSpec] = [
    ActionSpec("scan_left", 0.0, 0.0, -2.5, False),
    ActionSpec("scan_right", 0.0, 0.0, 2.5, False),
    ActionSpec("forward", 100.0, 0.0, 0.0, False),
    ActionSpec("forward_left", 100.0, -1.2, 0.0, False),
    ActionSpec("forward_right", 100.0, 1.2, 0.0, False),
    ActionSpec("retreat", -80.0, 0.0, 0.0, False),
    ActionSpec("retreat_left", -80.0, -1.2, 0.0, False),
    ActionSpec("retreat_right", -80.0, 1.2, 0.0, False),
    ActionSpec("turn_left", 0.0, -2.0, 0.0, False),
    ActionSpec("turn_right", 0.0, 2.0, 0.0, False),
    ActionSpec("aim_left_small", 0.0, 0.0, -2.0, False),
    ActionSpec("aim_right_small", 0.0, 0.0, 2.0, False),
    ActionSpec("aim_left_large", 0.0, 0.0, -4.0, False),
    ActionSpec("aim_right_large", 0.0, 0.0, 4.0, False),
    ActionSpec("fire_still", 0.0, 0.0, 0.0, True),
    ActionSpec("fire_push", 60.0, 0.0, 0.0, True),
    ActionSpec("fire_turn_left", 0.0, -1.0, 0.0, True),
    ActionSpec("fire_turn_right", 0.0, 1.0, 0.0, True),
]


class ReplayBuffer:
    """Simple ring buffer for off-policy learning."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.index = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = int(action)
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
    enemy_hull_error: float
    ally_fire_risk: bool
    obstacle_ahead: bool
    danger_ahead: bool
    powerup_visible: bool
    powerup_dist: float
    hp_ratio: float
    shield_ratio: float
    speed_ratio: float
    can_fire: bool
    reload_norm: float
    enemies_remaining: int


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

    def _can_fire(self, my_status: Dict[str, Any], reload_timer: float) -> bool:
        if reload_timer > 0.0:
            return False

        ammo = my_status.get("ammo", {})
        for slot in ammo.values():
            count = int((slot or {}).get("count", 0) or 0)
            if count > 0:
                return True
        return False

    def _nearest_enemy(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        seen_tanks: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        enemies = [tank for tank in seen_tanks if tank.get("team") != my_team]
        if not enemies:
            return None

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
        max_distance: Optional[float] = None,
    ) -> bool:
        for tank in seen_tanks:
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            ally_dist = self._distance(my_pos, ally_pos)
            if max_distance is not None and ally_dist > max_distance:
                continue
            angle = self._angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 6.0:
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
                continue

            distance = self._distance(my_pos, pos)
            if distance > max_dist:
                continue

            angle = self._angle_to(my_pos, pos)
            rel = abs(self.normalize_angle(angle - my_heading))
            if rel < half_angle:
                return True

        return False

    def encode(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
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
        nearest_enemy = self._nearest_enemy(my_pos, my_team, seen_tanks)

        enemy_visible = nearest_enemy is not None
        enemy_dist = 1.0
        enemy_hull_error = 0.0
        enemy_barrel_error = 0.0

        if nearest_enemy is not None:
            enemy_pos = nearest_enemy.get("position", {"x": 0.0, "y": 0.0})
            distance_raw = nearest_enemy.get("distance")
            if distance_raw is None:
                distance_raw = self._distance(my_pos, enemy_pos)

            vision_range = float(my_status.get("_vision_range", 25.0) or 25.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

            target_angle = self._angle_to(my_pos, enemy_pos)
            enemy_hull_error = self.normalize_angle(target_angle - my_heading) / 180.0
            enemy_barrel_error = self.normalize_angle(target_angle - barrel_abs) / 180.0

        ally_risk_max_distance: Optional[float] = None
        if nearest_enemy is not None:
            enemy_pos = nearest_enemy.get("position", {"x": 0.0, "y": 0.0})
            enemy_distance_raw = nearest_enemy.get("distance")
            if enemy_distance_raw is None:
                enemy_distance_raw = self._distance(my_pos, enemy_pos)
            ally_risk_max_distance = max(2.0, float(enemy_distance_raw) + 1.0)

        ally_fire_risk = self._ally_in_fire_line(
            my_pos,
            my_team,
            barrel_abs,
            seen_tanks,
            max_distance=ally_risk_max_distance,
        )

        seen_obstacles = sensor_data.get("seen_obstacles", [])
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
            vision_range = float(my_status.get("_vision_range", 25.0) or 25.0)
            powerup_dist = self._clamp(dist_raw / max(vision_range, 1.0), 0.0, 1.0)

        top_speed = float(my_status.get("_top_speed", 5.0) or 5.0)
        speed = float(my_status.get("move_speed", 0.0) or 0.0)
        speed_ratio = self._clamp(speed / max(top_speed, 1.0), -1.0, 1.0)

        enemies_remaining_safe = max(0, int(enemies_remaining))
        enemies_remaining_norm = self._clamp(float(enemies_remaining_safe) / 5.0, 0.0, 1.0)
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
            enemy_hull_error=enemy_hull_error,
            ally_fire_risk=ally_fire_risk,
            obstacle_ahead=obstacle_ahead,
            danger_ahead=danger_ahead,
            powerup_visible=powerup_visible,
            powerup_dist=powerup_dist,
            hp_ratio=hp_ratio,
            shield_ratio=shield_ratio,
            speed_ratio=speed_ratio,
            can_fire=can_fire,
            reload_norm=reload_norm,
            enemies_remaining=enemies_remaining_safe,
        )


@dataclass
class AgentConfig:
    state_dim: int = STATE_DIM
    n_rules: int = 32
    mf_type: str = "gaussian"

    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 2_000
    train_every: int = 2
    target_sync_every: int = 500

    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 120_000

    frame_skip: int = 1
    save_every_games: int = 1

    model_path: str = DEFAULT_MODEL_PATH
    best_model_path: Optional[str] = None
    final_model_path: Optional[str] = None
    seed: int = 1


class FuzzyDQNAgent:
    def __init__(self, name: str, config: AgentConfig, training_enabled: bool):
        self.name = name
        self.config = config
        self.training_enabled = training_enabled

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StateEncoder()

        self.policy_net = ANFISDQN(
            n_inputs=config.state_dim,
            n_rules=config.n_rules,
            n_actions=len(ACTION_SPACE),
            mf_type=config.mf_type,
        ).to(self.device)
        self.target_net = ANFISDQN(
            n_inputs=config.state_dim,
            n_rules=config.n_rules,
            n_actions=len(ACTION_SPACE),
            mf_type=config.mf_type,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, config.state_dim)

        self.total_steps = 0
        self.train_steps = 0
        self.games_played = 0
        self.current_epsilon = config.epsilon_start
        self.last_loss: Optional[float] = None

        self.last_observation: Optional[Observation] = None
        self.last_action_index: Optional[int] = None
        self.last_command = ActionCommand()
        self.last_known_enemies_remaining: Optional[int] = None
        self.was_destroyed = False
        self.best_score = float("-inf")
        self.episode_reward_sum = 0.0

        self.lock = threading.Lock()

        self.best_model_path = self._resolve_best_model_path()
        self.final_model_path = self._resolve_final_model_path()

        self._load_checkpoint_if_available()
        print(
            f"[{self.name}] ready | training={self.training_enabled} "
            f"device={self.device} rules={self.config.n_rules} mf={self.config.mf_type}"
        )

    def _resolve_best_model_path(self) -> Optional[str]:
        if self.config.best_model_path:
            return self.config.best_model_path
        if not self.config.model_path:
            return None
        root, ext = os.path.splitext(self.config.model_path)
        ext = ext or ".pt"
        return f"{root}_best{ext}"

    def _resolve_final_model_path(self) -> Optional[str]:
        if self.config.final_model_path:
            return self.config.final_model_path
        if not self.config.model_path:
            return None
        root, ext = os.path.splitext(self.config.model_path)
        ext = ext or ".pt"
        return f"{root}_final{ext}"

    def _epsilon(self) -> float:
        ratio = min(1.0, self.total_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + ratio * (self.config.epsilon_final - self.config.epsilon_start)

    def _select_action(self, state_vector: np.ndarray) -> int:
        state = torch.from_numpy(state_vector).unsqueeze(0).to(self.device)

        if not self.training_enabled:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return int(q_values.argmax(dim=1).item())

        self.current_epsilon = self._epsilon()
        if random.random() < self.current_epsilon:
            return random.randrange(len(ACTION_SPACE))

        with torch.no_grad():
            q_values = self.policy_net(state)
            return int(q_values.argmax(dim=1).item())

    @staticmethod
    def _ammo_counts(my_status: Dict[str, Any]) -> Dict[str, int]:
        ammo_data = my_status.get("ammo", {}) or {}
        counts: Dict[str, int] = {}
        for ammo_name, slot in ammo_data.items():
            key = str(ammo_name).upper()
            count = int((slot or {}).get("count", 0) or 0)
            counts[key] = max(0, count)
        return counts

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
            vision_range = float(my_status.get("_vision_range", 25.0) or 25.0)
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
        action: ActionSpec,
        my_status: Dict[str, Any],
        obs: Observation,
    ) -> ActionCommand:
        ammo_to_load = self._select_ammo_for_action(my_status, obs, action)
        should_fire = bool(action.should_fire)
        aligned_for_fire = (
            obs.enemy_visible
            and abs(obs.enemy_barrel_error) < 0.02
            and abs(obs.enemy_hull_error) < 0.08
            and obs.enemy_dist <= 0.95
        )

        # Friendly-fire guard: never fire into an ally line and avoid blind shots.
        if should_fire:
            if (
                obs.ally_fire_risk
                or (not obs.enemy_visible)
                or (not obs.can_fire)
                or obs.reload_norm > 0.02
                or (not aligned_for_fire)
            ):
                should_fire = False

        # If a clean shot is already lined up, force immediate fire.
        clean_shot = (
            obs.enemy_visible
            and (not obs.ally_fire_risk)
            and obs.can_fire
            and obs.reload_norm <= 0.02
            and aligned_for_fire
        )
        if clean_shot:
            should_fire = True

        move_speed = action.move_speed
        heading_rotation = action.heading_rotation_angle
        barrel_rotation = action.barrel_rotation_angle
        if should_fire:
            # Keep the platform stable at the firing tick to reduce accidental hits.
            move_speed = 0.0
            heading_rotation = 0.0
            barrel_rotation = 0.0

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
        prev_command: ActionCommand,
    ) -> float:
        move_speed = float(prev_command.move_speed)
        heading_rotation = float(prev_command.heading_rotation_angle)
        barrel_rotation = float(prev_command.barrel_rotation_angle)
        fired = bool(prev_command.should_fire)
        reward = -0.02

        hp_delta = current_obs.hp_ratio - prev_obs.hp_ratio
        if hp_delta < 0.0:
            reward += hp_delta * 10.0

        shield_delta = current_obs.shield_ratio - prev_obs.shield_ratio
        if shield_delta < 0.0:
            reward += shield_delta * 4.0

        if current_obs.enemy_visible:
            if current_obs.enemy_dist < prev_obs.enemy_dist:
                reward += 0.12
            if abs(current_obs.enemy_barrel_error) < abs(prev_obs.enemy_barrel_error):
                reward += 0.16
            if abs(current_obs.enemy_hull_error) < abs(prev_obs.enemy_hull_error):
                reward += 0.06
            if (
                move_speed != 0.0
                and abs(current_obs.speed_ratio) < 0.02
                and current_obs.obstacle_ahead
            ):
                reward -= 0.12
        else:
            # Reward exploration and discourage stationary spinning without a target.
            if move_speed != 0.0 and not current_obs.obstacle_ahead and not current_obs.danger_ahead:
                reward += 0.04
            if move_speed == 0.0 and abs(heading_rotation) > 0.0:
                reward -= 0.06
            if move_speed == 0.0 and abs(barrel_rotation) > 0.0:
                reward -= 0.04

        # Push policy toward objective completion (eliminate enemy team).
        enemy_delta = int(prev_obs.enemies_remaining) - int(current_obs.enemies_remaining)
        if enemy_delta > 0:
            reward += 3.0 * float(enemy_delta)
        elif enemy_delta < 0:
            reward -= 0.6 * float(-enemy_delta)

        if current_obs.obstacle_ahead and move_speed > 0.0:
            reward -= 0.25

        if current_obs.danger_ahead and move_speed > 0.0:
            reward -= 0.22

        if current_obs.ally_fire_risk:
            # Encourage moving barrel away from teammates even before pressing fire.
            reward -= 0.18

        if prev_obs.powerup_visible and prev_obs.hp_ratio < 0.7:
            if current_obs.powerup_dist < prev_obs.powerup_dist:
                reward += 0.06

        if fired:
            if prev_obs.ally_fire_risk:
                reward -= 5.0
            elif not prev_obs.enemy_visible:
                reward -= 1.20
            elif not prev_obs.can_fire or prev_obs.reload_norm > 0.02:
                reward -= 1.00
            elif (
                abs(prev_obs.enemy_barrel_error) < 0.02
                and abs(prev_obs.enemy_hull_error) < 0.08
                and prev_obs.enemy_dist <= 0.95
            ):
                reward += 1.20
            else:
                reward -= 0.55
        else:
            if (
                prev_obs.enemy_visible
                and prev_obs.can_fire
                and (not prev_obs.ally_fire_risk)
                and abs(prev_obs.enemy_barrel_error) < 0.02
                and abs(prev_obs.enemy_hull_error) < 0.08
                and prev_obs.enemy_dist <= 0.95
            ):
                reward -= 0.35

        return reward

    def _maybe_train(self) -> None:
        if not self.training_enabled:
            return

        min_required = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < min_required:
            return

        if self.total_steps % self.config.train_every != 0:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size,
            self.device,
        )

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q)
        self.last_loss = float(loss.item())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["model_state_dict"], strict=False)

                target_state = checkpoint.get("target_state_dict")
                if target_state:
                    self.target_net.load_state_dict(target_state, strict=False)
                else:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                optimizer_state = checkpoint.get("optimizer_state_dict")
                if optimizer_state:
                    self.optimizer.load_state_dict(optimizer_state)

                self.total_steps = int(checkpoint.get("total_steps", 0) or 0)
                self.train_steps = int(checkpoint.get("train_steps", 0) or 0)
                self.games_played = int(checkpoint.get("games_played", 0) or 0)
                self.best_score = float(checkpoint.get("best_score", float("-inf")))
            else:
                # Compatibility with plain state_dict files.
                self.policy_net.load_state_dict(checkpoint, strict=False)
                self.target_net.load_state_dict(self.policy_net.state_dict())

            print(f"[{self.name}] checkpoint loaded: {path}")
        except Exception as exc:
            print(f"[{self.name}] failed to load checkpoint {path}: {exc}")

    def _reset_episode_state(self) -> None:
        self.last_observation = None
        self.last_action_index = None
        self.last_command = ActionCommand()

    def get_action(
        self,
        current_tick: int,
        my_tank_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
    ) -> ActionCommand:
        with self.lock:
            if (
                self.config.frame_skip > 1
                and self.last_action_index is not None
                and current_tick % self.config.frame_skip != 0
            ):
                return self.last_command

            current_obs = self.encoder.encode(my_tank_status, sensor_data, enemies_remaining)
            self.last_known_enemies_remaining = int(current_obs.enemies_remaining)

            if self.last_observation is not None and self.last_action_index is not None:
                reward = self._compute_step_reward(
                    prev_obs=self.last_observation,
                    current_obs=current_obs,
                    prev_command=self.last_command,
                )
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_index,
                    reward=reward,
                    next_state=current_obs.vector,
                    done=0.0,
                )
                self.episode_reward_sum += reward
                self._maybe_train()

            action_index = self._select_action(current_obs.vector)
            action = ACTION_SPACE[action_index]
            command = self._to_command(action, my_tank_status, current_obs)

            self.last_observation = current_obs
            self.last_action_index = action_index
            self.last_command = command
            self.total_steps += 1

            return command

    def destroy(self) -> None:
        with self.lock:
            print(f"[{self.name}] destroyed")
            if self.last_observation is not None and self.last_action_index is not None:
                destroy_penalty = -8.0
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_index,
                    reward=destroy_penalty,
                    next_state=self.last_observation.vector,
                    done=1.0,
                )
                self.episode_reward_sum += destroy_penalty
                for _ in range(3):
                    self._maybe_train()

            self.was_destroyed = True
            self._reset_episode_state()

    def end(self, damage_dealt: float, tanks_killed: int) -> None:
        with self.lock:
            # Keep episode-end reward independent from engine kill/damage counters,
            # which can include friendly-fire side effects in this project setup.
            final_reward = -0.5
            if not self.was_destroyed:
                final_reward = 1.5
            if self.last_known_enemies_remaining is not None and self.last_known_enemies_remaining <= 0:
                final_reward += 6.0

            if self.last_observation is not None and self.last_action_index is not None:
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_index,
                    reward=final_reward,
                    next_state=self.last_observation.vector,
                    done=1.0,
                )
                for _ in range(5):
                    self._maybe_train()

            self.episode_reward_sum += final_reward
            self.games_played += 1

            print(
                f"[{self.name}] end | games={self.games_played} "
                f"damage={damage_dealt:.1f} kills={tanks_killed} "
                f"ep_reward={self.episode_reward_sum:.3f} "
                f"epsilon={self.current_epsilon:.3f} replay={len(self.replay)} "
                f"train_steps={self.train_steps}"
            )

            if self.training_enabled and self.games_played % max(1, self.config.save_every_games) == 0:
                episode_score = float(self.episode_reward_sum)
                if episode_score > self.best_score:
                    self.best_score = episode_score
                    self.save_checkpoint(self.best_model_path, label="best")

                # Save latest and final after potential best-score update, so
                # best_score persists across process restarts.
                self.save_checkpoint(self.config.model_path, label="latest")
                self.save_checkpoint(self.final_model_path, label="final")

            self.was_destroyed = False
            self.last_known_enemies_remaining = None
            self.episode_reward_sum = 0.0
            self._reset_episode_state()

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "training_enabled": self.training_enabled,
            "steps": self.total_steps,
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "epsilon": round(self.current_epsilon, 5),
            "replay_size": len(self.replay),
            "last_loss": self.last_loss,
            "model_path": self.config.model_path,
            "best_model_path": self.best_model_path,
            "final_model_path": self.final_model_path,
            "best_score": self.best_score,
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
        # Keep import-time initialization light; runtime __main__ overrides it.
        agent = FuzzyDQNAgent(
            name="FuzzyDQN",
            config=AgentConfig(model_path=""),
            training_enabled=False,
        )
    return agent


@app.get("/")
async def root() -> Dict[str, Any]:
    return _get_agent().status()


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)) -> ActionCommand:
    return _get_agent().get_action(
        current_tick=int(payload.get("current_tick", 0) or 0),
        my_tank_status=payload.get("my_tank_status", {}),
        sensor_data=payload.get("sensor_data", {}),
        enemies_remaining=int(payload.get("enemies_remaining", 0) or 0),
    )


@app.post("/agent/destroy", status_code=204)
async def destroy() -> None:
    _get_agent().destroy()


@app.post("/agent/end", status_code=204)
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
    parser.add_argument("--final-model-path", type=str, default=None)
    parser.add_argument("--rules", type=int, default=32)
    parser.add_argument("--mf-type", choices=["gaussian", "bell", "triangular"], default="gaussian")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every-games", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--target-sync-every", type=int, default=500)
    parser.add_argument("--epsilon-decay-steps", type=int, default=120000)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = AgentConfig(
        n_rules=max(4, int(args.rules)),
        mf_type=args.mf_type,
        frame_skip=max(1, int(args.frame_skip)),
        model_path=args.model_path,
        best_model_path=args.best_model_path,
        final_model_path=args.final_model_path,
        seed=int(args.seed),
        save_every_games=max(1, int(args.save_every_games)),
        warmup_steps=max(0, int(args.warmup_steps)),
        batch_size=max(16, int(args.batch_size)),
        train_every=max(1, int(args.train_every)),
        target_sync_every=max(1, int(args.target_sync_every)),
        epsilon_decay_steps=max(1000, int(args.epsilon_decay_steps)),
    )

    agent_name = args.name or f"FuzzyDQN_{args.port}"

    # Replace default global agent with runtime configuration.
    agent = FuzzyDQNAgent(
        name=agent_name,
        config=config,
        training_enabled=bool(args.train),
    )

    print(
        f"Starting {agent_name} on {args.host}:{args.port} "
        f"| train={args.train} | model={config.model_path}"
    )
    uvicorn.run(app, host=args.host, port=args.port)

