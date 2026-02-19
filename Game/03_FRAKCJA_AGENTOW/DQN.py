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

import argparse
import copy
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
import matplotlib
matplotlib.use("Agg")       # zeby uniknac bledow

from ANFISDQN import ANFISDQN


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
ACTION_DIM = 4
# TODO: Scale actions using per-tank limits from my_tank_status (_top_speed, _heading_spin_rate, _barrel_spin_rate)
MAX_MOVE_SPEED = 5.0
MAX_HEADING_DELTA = 5.0
MAX_BARREL_DELTA = 5.0


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
    ActionSpec("idle", 0.0, 0.0, 0.0, False),

    # Movement.
    ActionSpec("forward", 1.0, 0.0, 0.0, False),
    ActionSpec("forward_left", 0.75, -1.5, 0.0, False),
    ActionSpec("forward_right", 0.75, 1.5, 0.0, False),
    ActionSpec("retreat", -0.5, 0.0, 0.0, False),

    # Rotation
    ActionSpec("turn_left", 0.0, -2.5, 0.0, False),
    ActionSpec("turn_right", 0.0, 2.5, 0.0, False),

    # Aiming (ONLY ONE SCALE) - serio boje sie ze nie beda mogly dobrze do siebie przymierzyc.
    ActionSpec("aim_left", 0.0, 0.0, -1.0, False),
    ActionSpec("aim_right", 0.0, 0.0, 1.0, False),
    ActionSpec("aim_left_fast", 0.0, 0.0, -60.0, False),
    ActionSpec("aim_right", 0.0, 0.0, 60.0, False),

    # Fire
    ActionSpec("fire", 0.0, 0.0, 0.0, True),
    ActionSpec("fire_forward", 1.0, 0.0, 0.0, True),
]


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
        enemy_hull_error = 0.5
        enemy_barrel_error = 0.5

        if nearest_enemy is not None:
            enemy_pos = nearest_enemy.get("position", {"x": 0.0, "y": 0.0})
            distance_raw = nearest_enemy.get("distance")
            if distance_raw is None:
                raise ValueError(f"Missing enemy distance in sensor data: {nearest_enemy}")

            vision_range = float(my_status.get("_vision_range", 25.0) or 25.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

            target_angle = self._angle_to(my_pos, enemy_pos)
            enemy_hull_error = (self.normalize_angle(target_angle - my_heading) / 180.0 + 1.0) * 0.5
            enemy_barrel_error = (self.normalize_angle(target_angle - barrel_abs) / 180.0 + 1.0) * 0.5

        ally_fire_risk = self._ally_in_fire_line(my_pos, my_team, barrel_abs, seen_tanks)

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
    n_rules: int = 32
    mf_type: str = "gaussian"

    gamma: float = 0.95
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    tau: float = 0.01
    action_noise_start: float = 0.3
    action_noise_end: float = 0.05
    action_noise_decay_steps: int = 50_000
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 2_000
    train_every: int = 2
    target_sync_every: int = 1


    frame_skip: int = 1
    save_every_games: int = 1

    model_path: str = DEFAULT_MODEL_PATH
    best_model_path: Optional[str] = None
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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity, config.state_dim, ACTION_DIM)

        self.total_steps = 0
        self.train_steps = 0
        self.games_played = 0
        self.last_loss: Optional[float] = None
        self.current_episode_score = 0.0

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
        self._load_checkpoint_if_available()
        print(
            f"[{self.name}] ready | training={self.training_enabled} "
            f"device={self.device} rules={self.config.n_rules} mf={self.config.mf_type} "
            f"save_path={self.config.model_path}"
        )

        self.trace_positions: List[Tuple[float, float]] = []
        self.trace_hp: List[float] = []
        self.trace_shots: List[Tuple[float, float, float]] = []
        self.trace_allies: List[Tuple[float, float]] = []
        self.trace_enemies: List[Tuple[float, float]] = []
        self.pos_history: List[Tuple[float, float]] = []
        self.episode_reward_total = 0.0
        self.episode_reward_parts: Dict[str, float] = {}
        self.episode_reward_parts_steps = 0
        self.frontier_min_x: Optional[float] = None
        self.frontier_max_x: Optional[float] = None
        self.frontier_min_y: Optional[float] = None
        self.frontier_max_y: Optional[float] = None
        self.reward_parts_history: Dict[str, List[float]] = {}

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
        barrel = action[:, 2] * MAX_BARREL_DELTA
        fire = (action[:, 3] + 1.0) * 0.5
        return torch.stack([move, heading, barrel, fire], dim=1)

    def _current_action_noise_std(self) -> float:
        decay_steps = max(1, int(self.config.action_noise_decay_steps))
        t = min(max(self.total_steps, 0), decay_steps)
        frac = 1.0 - (t / float(decay_steps))
        return float(self.config.action_noise_end + (self.config.action_noise_start - self.config.action_noise_end) * frac)

    def _select_action(self, state_vector: np.ndarray, training: bool) -> np.ndarray:
        state = torch.from_numpy(state_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.actor(state)
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
        action_vec: np.ndarray,
        my_status: Dict[str, Any],
        obs: Observation,
    ) -> ActionCommand:
        move_speed = float(action_vec[0])
        heading_rotation = float(action_vec[1])
        barrel_rotation = float(action_vec[2])
        fire_prob = float(action_vec[3])
        should_fire = fire_prob > 0.5

        action_stub = ActionSpec(
            name="ddpg",
            move_speed=move_speed,
            heading_rotation_angle=heading_rotation,
            barrel_rotation_angle=barrel_rotation,
            should_fire=should_fire,
        )
        ammo_to_load = self._select_ammo_for_action(my_status, obs, action_stub)
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
    ) -> float:         # Exploration is all you need
        parts: Dict[str, float] = {}
        # Damage avoidance (biggest issue: entering danger zone).
        # parts["hp_delta"] = (current_obs.hp_ratio - prev_obs.hp_ratio) * 10.0
        # parts["shield_delta"] = (current_obs.shield_ratio - prev_obs.shield_ratio) * 2.0

        # Danger/obstacle penalties (angles already normalized).
        # parts["danger_ahead"] = -1.0 if current_obs.danger_ahead else 0.0
        # parts["danger_move"] = -7.5 if current_obs.danger_ahead and action.move_speed > 1e-2 else 0.0
        # parts["danger_enter"] = -1.0 if (not prev_obs.danger_ahead and current_obs.danger_ahead) else 0.0
        # parts["danger_exit"] = 0.5 if (prev_obs.danger_ahead and not current_obs.danger_ahead) else 0.0
        # parts["obstacle_move"] = -0.2 if current_obs.obstacle_ahead and action.move_speed > 0.0 else 0.0

        # Enemy tracking/aiming.
        # if current_obs.enemy_visible:
        #     parts["enemy_visible"] = 0.1
        #     barrel_improve = abs(prev_obs.enemy_barrel_error - 0.5) - abs(current_obs.enemy_barrel_error - 0.5)
        #     hull_improve = abs(prev_obs.enemy_hull_error - 0.5) - abs(current_obs.enemy_hull_error - 0.5)
        #     parts["barrel_improve"] = barrel_improve * 3.0
        #     parts["hull_improve"] = hull_improve * 0.2
        # else:
        #     parts["enemy_visible"] = -0.05
        #     parts["barrel_improve"] = 0.0
        #     parts["hull_improve"] = 0.0

        # Ally safety.
        # parts["ally_fire_risk"] = -0.2 if current_obs.ally_fire_risk else 0.0

        # Firing quality based on previous state (action chosen there).
        # parts["fire_no_ammo"] = 0.0
        # parts["fire_no_enemy"] = 0.0
        # parts["fire_ally_risk"] = 0.0
        # parts["fire_aim_error"] = 0.0
        # if action.should_fire:
        #     if not prev_obs.can_fire:
        #         parts["fire_no_ammo"] = -0.4
        #     elif not prev_obs.enemy_visible:
        #         parts["fire_no_enemy"] = -2.0
        #     elif prev_obs.ally_fire_risk:
        #         parts["fire_ally_risk"] = -3.5
        #     else:
        #         aim_error = abs(prev_obs.enemy_barrel_error - 0.5)
        #         parts["fire_aim_error"] = -(aim_error + 0.25)

        # Powerup pursuit.
        # parts["powerup_pursuit"] = (
        #     (prev_obs.powerup_dist - current_obs.powerup_dist) * 0.1
        #     if current_obs.powerup_visible
        #     else 0.0
        # )

        # Small penalty for idling while enemy is visible.
        # parts["idle_visible"] = -0.05 if abs(action.move_speed) < 1e-3 and current_obs.enemy_visible else 0.0

        recent = self.pos_history[-200:] or [current_pos]
        prev = self.pos_history[-400:-200] or recent
        rcx = sum(p[0] for p in recent) / float(len(recent))
        rcy = sum(p[1] for p in recent) / float(len(recent))
        pcx = sum(p[0] for p in prev) / float(len(prev))
        pcy = sum(p[1] for p in prev) / float(len(prev))
        var_r = sum((p[0] - rcx) ** 2 + (p[1] - rcy) ** 2 for p in recent) / float(len(recent))
        var_p = sum((p[0] - pcx) ** 2 + (p[1] - pcy) ** 2 for p in prev) / float(len(prev))
        parts["variance_recent"] = var_r / 10
        parts["variance_prev"] = var_p / 10
        parts["centroid_bonus"] = parts["variance_recent"] + parts["variance_prev"]

        frontier_bonus = 0.0
        if self.frontier_min_x is None:
            self.frontier_min_x = current_pos[0]
            self.frontier_max_x = current_pos[0]
            self.frontier_min_y = current_pos[1]
            self.frontier_max_y = current_pos[1]
        else:                                       # wyjezdza w nieznane
            if current_pos[0] < self.frontier_min_x:
                frontier_bonus += (self.frontier_min_x - current_pos[0]) / 50.0
                self.frontier_min_x = current_pos[0]
            if current_pos[0] > self.frontier_max_x:
                frontier_bonus += (current_pos[0] - self.frontier_max_x) / 50.0
                self.frontier_max_x = current_pos[0]
            if current_pos[1] < self.frontier_min_y:
                frontier_bonus += (self.frontier_min_y - current_pos[1]) / 50.0
                self.frontier_min_y = current_pos[1]
            if current_pos[1] > self.frontier_max_y:
                frontier_bonus += (current_pos[1] - self.frontier_max_y) / 50.0
                self.frontier_max_y = current_pos[1]
        parts["frontier_bonus"] = frontier_bonus

        reward = sum(parts.values())
        self.episode_reward_total += reward
        for key, value in parts.items():
            self.episode_reward_parts[key] = self.episode_reward_parts.get(key, 0.0) + value
        self.episode_reward_parts_steps += 1

        return reward


    def _maybe_train(self) -> None:
        # if not self.training_enabled:
        #     return

        # min_required = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < self.config.batch_size:
            return

        # if self.total_steps % self.config.train_every != 0:
        #     return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size,
            self.device,
        )

        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            next_action = torch.tanh(next_raw)
            next_action = self._scale_action_tensor(next_action)
            next_action = F.hardsigmoid(next_action)
            next_q = self.critic_target(torch.cat([next_states, next_action], dim=1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        actions = F.hardsigmoid(actions)
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
        actor_action = F.hardsigmoid(actor_action)
        actor_loss = -self.critic(torch.cat([states, actor_action], dim=1)).mean()

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
        self.trace_allies.clear()
        self.trace_enemies.clear()
        self.pos_history.clear()
        self.episode_reward_total = 0.0
        self.episode_reward_parts = {}
        self.episode_reward_parts_steps = 0
        self.frontier_min_x = None
        self.frontier_max_x = None
        self.frontier_min_y = None
        self.frontier_max_y = None

    def _record_trace(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        current_obs: Observation,
        action: ActionSpec,
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
        self.trace_positions.append((x, y))
        self.trace_hp.append(hp)

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
            self.trace_shots.append((x, y, heading + barrel))

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
        ax.scatter(xs, ys, c=hps, cmap="RdYlGn", s=10, alpha=0.7)

        if self.trace_allies:
            ax.scatter(
                [p[0] for p in self.trace_allies],
                [p[1] for p in self.trace_allies],
                c="blue",
                s=12,
                alpha=0.6,
                label="allies",
            )
        if self.trace_enemies:
            ax.scatter(
                [p[0] for p in self.trace_enemies],
                [p[1] for p in self.trace_enemies],
                c="black",
                s=12,
                alpha=0.6,
                label="enemies",
            )
        if self.trace_shots:
            for sx, sy, ang in self.trace_shots:
                dx = math.cos(math.radians(ang)) * 10.0
                dy = math.sin(math.radians(ang)) * 10.0
                ax.arrow(
                    sx,
                    sy,
                    dx,
                    dy,
                    color="#8a7500",
                    width=0.12,
                    head_width=0.6,
                    head_length=1.0,
                    length_includes_head=True,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, MAP_WIDTH)
        ax.set_ylim(0.0, MAP_HEIGHT)
        ax.autoscale(False)
        ax.set_title(f"{self.name} trajectory (game {self.games_played})")
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
    ) -> ActionCommand:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent get_action call detected")
        try:
            pos = my_tank_status.get("position")
            if not pos or "x" not in pos or "y" not in pos:
                raise ValueError(f"Missing position in my_tank_status: {pos}")
            current_pos = (float(pos["x"]), float(pos["y"]))
            current_obs = self.encoder.encode(my_tank_status, sensor_data, enemies_remaining)
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
            # 19 - 22 
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
            command = self._to_command(action_vec, my_tank_status, current_obs)

            if command.should_fire:
                self.last_fire_tick = current_tick

            self.last_observation = current_obs
            self.last_action_vector = action_vec
            self.last_command = command
            self.prev_enemies_remaining = enemies_remaining
            self.total_steps += 1
            self.last_status = my_tank_status
            self._record_trace(my_tank_status, sensor_data, current_obs, ActionSpec("ddpg", action_vec[0], action_vec[1], action_vec[2], command.should_fire))

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
            steps = max(1, self.episode_reward_parts_steps)
            episode_parts = {k: v / steps for k, v in self.episode_reward_parts.items()}
            episode_parts["total_reward_avg"] = self.episode_reward_total / steps
            all_keys = set(self.reward_parts_history) | set(episode_parts)
            for key in all_keys:
                self.reward_parts_history.setdefault(key, [])
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
            ax.plot(
                xs,
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

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "training_enabled": self.training_enabled,
            "steps": self.total_steps,
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "replay_size": len(self.replay),
            "last_loss": self.last_loss,
            "model_path": self.config.model_path,
            "best_model_path": self.best_model_path,
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
    parser.add_argument("--mf-type", choices=["gaussian", "bell", "triangular"], default="gaussian")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every-games", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--target-sync-every", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = AgentConfig(
        n_rules=max(4, int(args.rules)),
        mf_type=args.mf_type,
        frame_skip=max(1, int(args.frame_skip)),
        model_path=args.model_path,
        best_model_path=args.best_model_path,
        seed=int(args.seed),
        save_every_games=max(1, int(args.save_every_games)),
        warmup_steps=max(0, int(args.warmup_steps)),
        batch_size=max(16, int(args.batch_size)),
        train_every=max(1, int(args.train_every)),
        target_sync_every=max(1, int(args.target_sync_every)),
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
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)