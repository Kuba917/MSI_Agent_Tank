"""
DQN Agent for Tank Battle
Agent uczący się metodą Deep Q-Network (Reinforcement Learning).

Wymagania:
    pip install torch numpy

Uruchomienie:
    python dqn_agent.py --port 8001 --name DQNAgent_1
"""

import argparse
import sys
import os
import random
import math
import numpy as np
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

try:
    from comet_ml import Experiment, ExistingExperiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

# --- PyTorch Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("Brak biblioteki PyTorch. Zainstaluj: pip install torch")
    sys.exit(1)

# --- Ścieżki importów (zgodne z random_agent.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)
parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

# ============================================================================
# KONFIGURACJA DQN
# ============================================================================

BATCH_SIZE = 128    # Zwiększono dla stabilniejszego uczenia
GAMMA = 0.95        # Discount factor
EPS_START = 1.0     # Początkowa losowość
EPS_END = 0.05      # Końcowa losowość
EPS_DECAY = 15000   # Znacznie wolniejszy zanik losowości (więcej eksploracji)
TARGET_UPDATE = 10  # Co ile epizodów aktualizować sieć docelową
LR = 0.001          # Learning Rate
MEMORY_SIZE = 10000
MODEL_PATH = os.path.join(current_dir, "dqn_model.pth")
COMET_KEY_PATH = os.path.join(current_dir, "comet_key.txt")
FRAME_SKIP = 4      # Podejmuj decyzję co 4 klatki (ok. 15 razy na sekundę przy 60 FPS)

# Konfiguracja Comet ML
API_KEY = "RoqFxUQ2dJHm8RjW1YatD0VQw"
PROJECT_NAME = "MSI_Tank_DQN"
WORKSPACE = "jbuka"

# Definicja akcji niskopoziomowych (Kombinatoryka)
# Move (3) * Hull (3) * Barrel (3) * Fire (2) = 54 akcje
MOVES = [0.0, 100.0, -100.0]  # Stop, Przód, Tył
TURNS = [0.0, -15.0, 15.0]    # Brak, Lewo, Prawo
FIRES = [False, True]         # Nie strzelaj, Strzelaj

NUM_ACTIONS = 3 * 3 * 3 * 2  # 54

# ============================================================================
# MODEL SIECI NEURONOWEJ
# ============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ============================================================================
# PAMIĘĆ DOŚWIADCZEŃ (REPLAY BUFFER)
# ============================================================================

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ============================================================================
# MODEL KOMENDY (API)
# ============================================================================

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

# ============================================================================
# LOGIKA AGENTA DQN
# ============================================================================

class DQNAgent:
    def __init__(self, name: str = "DQNAgent"):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stan wejściowy:
        # [MyHP, MyAmmo, EnemyDist, EnemyAngle, EnemyHP, PowerupDist, PowerupAngle, AimedAtFriend, ObsDist, ObsAngle, DangDist, DangAngle]
        self.input_dim = 12
        
        self.policy_net = DQN(self.input_dim, NUM_ACTIONS).to(self.device)
        self.target_net = DQN(self.input_dim, NUM_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.steps_done = 0
        self.last_state = None
        self.last_action = None
        
        # Zmienne do obliczania nagród
        self.prev_enemies_count = 0
        self.prev_my_hp = 100
        self.prev_enemy_hp_sum = 0
        self.last_fire_tick = -100
        self.current_tick = 0
        self.last_action_cmd = None # Cache ostatniej komendy dla Frame Skipping
        
        # Inicjalizacja Comet ML (opóźniona)
        self.experiment = None

        # Ładowanie modelu jeśli istnieje
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=self.device)
                # Obsługa nowego formatu zapisu (słownik ze stanem)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.steps_done = checkpoint.get('steps_done', 0)
                else:
                    self.policy_net.load_state_dict(checkpoint)
                
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"[{self.name}] Załadowano model z {MODEL_PATH} (Kroki: {self.steps_done})")
            except Exception as e:
                print(f"[{self.name}] Błąd ładowania modelu: {e}")

    def init_comet(self):
        """Bezpieczna inicjalizacja Comet ML."""
        if COMET_AVAILABLE:
            try:
                # Sprawdź, czy istnieje klucz poprzedniego eksperymentu
                if os.path.exists(COMET_KEY_PATH):
                    with open(COMET_KEY_PATH, 'r') as f:
                        previous_key = f.read().strip()
                    
                    self.experiment = ExistingExperiment(
                        api_key=API_KEY,
                        previous_experiment=previous_key,
                        auto_output_logging="simple"
                    )
                    print(f"[{self.name}] Comet ML resumed (Key: {previous_key}).")
                else:
                    # Tworzymy nowy eksperyment i zapisujemy jego klucz
                    self.experiment = Experiment(
                        api_key=API_KEY,
                        project_name=PROJECT_NAME,
                        workspace=WORKSPACE,
                        auto_output_logging="simple"
                    )
                    self.experiment.set_name(self.name)
                    with open(COMET_KEY_PATH, 'w') as f:
                        f.write(self.experiment.get_key())
                    print(f"[{self.name}] Comet ML initialized (New Experiment).")
            except Exception as e:
                print(f"[{self.name}] Comet ML init failed (logging disabled): {e}")

<<<<<<< Updated upstream
    def normalize_angle(self, angle):
        """Sprowadza kąt do zakresu [-180, 180]."""
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
=======
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
    destructible_ahead: bool
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
>>>>>>> Stashed changes
        return angle

    def get_state_vector(self, my_status, sensor_data, enemies_remaining):
        """Przetwarza surowe dane JSON na wektor wejściowy sieci."""
        my_pos = my_status.get('position', {'x': 0, 'y': 0})
        my_heading = my_status.get('heading', 0)
        my_barrel = my_status.get('barrel_angle', 0)
        my_team = my_status.get('_team')
        
        # 1. Najbliższy wróg
        seen_tanks = sensor_data.get('seen_tanks', [])
        enemies = [t for t in seen_tanks if t.get('team') != my_team]
        
        enemy_dist = 1.0 # Znormalizowane (1.0 = max range lub brak)
        enemy_angle = 0.0
        enemy_hp = 0.0
        
        nearest_enemy = None
        if enemies:
            nearest_enemy = min(enemies, key=lambda t: math.hypot(t['position']['x'] - my_pos['x'], t['position']['y'] - my_pos['y']))
            dx = nearest_enemy['position']['x'] - my_pos['x']
            dy = nearest_enemy['position']['y'] - my_pos['y']
            dist = math.hypot(dx, dy)
            abs_angle = math.degrees(math.atan2(dy, dx))
            
            # Kąt relatywny do lufy
            current_barrel_abs = my_heading + my_barrel
            rel_angle = self.normalize_angle(abs_angle - current_barrel_abs)
            
            enemy_dist = min(dist / 500.0, 1.0)
            enemy_angle = rel_angle / 180.0
            enemy_hp = nearest_enemy.get('hp', 0) / 100.0 # Przybliżenie max HP

        # 2. Najbliższy PowerUp
        powerups = sensor_data.get('seen_powerups', [])
        pup_dist = 1.0
        pup_angle = 0.0
        
        if powerups:
            nearest_pup = min(powerups, key=lambda p: math.hypot(p.get('position', {}).get('x', 0) - my_pos['x'], p.get('position', {}).get('y', 0) - my_pos['y']))
            dx = nearest_pup.get('position', {}).get('x', 0) - my_pos['x']
            dy = nearest_pup.get('position', {}).get('y', 0) - my_pos['y']
            dist = math.hypot(dx, dy)
            abs_angle = math.degrees(math.atan2(dy, dx))
            
            current_heading = my_heading # Powerupy zbieramy kadłubem
            rel_angle = self.normalize_angle(abs_angle - current_heading)
            
            pup_dist = min(dist / 500.0, 1.0)
            pup_angle = rel_angle / 180.0

        # 3. Friendly Fire Check
        aimed_at_friend = 0.0
        friends = [t for t in seen_tanks if t.get('team') == my_team]
        for f in friends:
            dx = f['position']['x'] - my_pos['x']
            dy = f['position']['y'] - my_pos['y']
            abs_angle = math.degrees(math.atan2(dy, dx))
            current_barrel_abs = my_heading + my_barrel
            diff = abs(self.normalize_angle(abs_angle - current_barrel_abs))
            if diff < 5.0: # Celujemy w sojusznika
                aimed_at_friend = 1.0
                break

        # 4. Najbliższa przeszkoda (Obstacle)
        obstacles = sensor_data.get('seen_obstacles', [])
        obs_dist = 1.0
        obs_angle = 0.0
        
        if obstacles:
            nearest_obs = min(obstacles, key=lambda o: math.hypot(o.get('position', {}).get('x', 0) - my_pos['x'], o.get('position', {}).get('y', 0) - my_pos['y']))
            o_pos = nearest_obs.get('position', {'x': 0, 'y': 0})
            dx = o_pos['x'] - my_pos['x']
            dy = o_pos['y'] - my_pos['y']
            dist = math.hypot(dx, dy)
            abs_angle = math.degrees(math.atan2(dy, dx))
            
            # Przeszkody omijamy kadłubem
            rel_angle = self.normalize_angle(abs_angle - my_heading)
            
            obs_dist = min(dist / 500.0, 1.0)
            obs_angle = rel_angle / 180.0

        # 5. Najbliższy niebezpieczny teren (Water, PotholeRoad)
        terrains = sensor_data.get('seen_terrains', [])
        danger_terrains = [t for t in terrains if t.get('terrain_type') in ['Water', 'PotholeRoad']]
        dang_dist = 1.0
        dang_angle = 0.0
        
        if danger_terrains:
            nearest_dang = min(danger_terrains, key=lambda t: math.hypot(t.get('position', {}).get('x', 0) - my_pos['x'], t.get('position', {}).get('y', 0) - my_pos['y']))
            t_pos = nearest_dang.get('position', {'x': 0, 'y': 0})
            dx = t_pos['x'] - my_pos['x']
            dy = t_pos['y'] - my_pos['y']
            dist = math.hypot(dx, dy)
            abs_angle = math.degrees(math.atan2(dy, dx))
            
            # Teren omijamy kadłubem
            rel_angle = self.normalize_angle(abs_angle - my_heading)
            
            dang_dist = min(dist / 500.0, 1.0)
            dang_angle = rel_angle / 180.0

<<<<<<< Updated upstream
        # Budowa wektora
        state = [
            my_status.get('hp', 0) / 100.0,
            1.0 if my_status.get('ammo', {}).get('HEAVY', {}).get('count', 0) > 0 else 0.0,
            enemy_dist,
            enemy_angle,
            enemy_hp,
            pup_dist,
            pup_angle,
            aimed_at_friend,
            obs_dist,
            obs_angle,
            dang_dist,
            dang_angle
        ]
        return torch.tensor([state], device=self.device, dtype=torch.float32), nearest_enemy

    def select_action(self, state):
        """Epsilon-Greedy Policy."""
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if self.experiment:
            self.experiment.log_metric("epsilon", eps_threshold, step=self.steps_done)
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
=======
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
        max_dist: float = 999.0,
    ) -> bool:
        for tank in seen_tanks:
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            if self._distance(my_pos, ally_pos) > max_dist:
                continue
            angle = self._angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 4.0:
                return True
        return False

    def _scan_obstacles(
        self,
        my_pos: Dict[str, float],
        my_heading: float,
        objects: List[Dict[str, Any]],
        max_dist: float,
        half_angle: float,
    ) -> Tuple[bool, bool]:
        """Returns (found_any_obstacle, found_destructible_obstacle)"""
        found_any = False
        found_destructible = False

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
                found_any = True
                if item.get("type") == "Tree":
                    found_destructible = True
        
        return found_any, found_destructible

    def encode(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
    ) -> Observation:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("team")
        if my_team is None:
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
        enemy_dist_raw = 999.0

        if nearest_enemy is not None:
            enemy_pos = nearest_enemy.get("position", {"x": 0.0, "y": 0.0})
            distance_raw = nearest_enemy.get("distance")
            if distance_raw is None:
                raise ValueError(f"Missing enemy distance in sensor data: {nearest_enemy}")

            enemy_dist_raw = float(distance_raw)
            vision_range = float(my_status.get("_vision_range", 25.0) or 25.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

            target_angle = self._angle_to(my_pos, enemy_pos)
            enemy_hull_error = (self.normalize_angle(target_angle - my_heading) / 180.0 + 1.0) * 0.5
            enemy_barrel_error = (self.normalize_angle(target_angle - barrel_abs) / 180.0 + 1.0) * 0.5

        ally_fire_risk = self._ally_in_fire_line(my_pos, my_team, barrel_abs, seen_tanks, max_dist=enemy_dist_raw)

        seen_obstacles = sensor_data.get("seen_obstacles", [])
        obstacle_ahead, destructible_ahead = self._scan_obstacles(
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
        # Reuse scan logic for terrain, ignoring destructible flag
        danger_ahead, _ = self._scan_obstacles(
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
            destructible_ahead=destructible_ahead,
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
    seed: int = 1


class FuzzyDQNAgent:
    def __init__(self, name: str, config: AgentConfig, training_enabled: bool, load_checkpoint: bool = True):
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
        self.trace_shots: List[Tuple[float, float, float]] = []
        self.trace_allies: List[Tuple[float, float]] = []
        self.trace_enemies: List[Tuple[float, float]] = []
        self.pos_history: List[Tuple[float, float]] = []
        self.trace_actor_raw: List[np.ndarray] = []
        self.last_actor_raw: Optional[np.ndarray] = None
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

        # Priority: if firing, keep current ammo to avoid reload interruption
        if action.should_fire and current and counts.get(current, 0) > 0:
            return current

        if obs.enemy_visible:
            vision_range = float(my_status.get("_vision_range", 25.0) or 25.0)
            enemy_distance = obs.enemy_dist * max(vision_range, 1.0)

            if enemy_distance > 50.0:
                preferred = ["LONG_DISTANCE", "LIGHT", "HEAVY"]
            elif enemy_distance > 25.0:
                preferred = ["LIGHT", "LONG_DISTANCE", "HEAVY"]
            else:
                preferred = ["HEAVY", "LIGHT", "LONG_DISTANCE"]
>>>>>>> Stashed changes
        else:
            return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=self.device, dtype=torch.long)

    def decode_action(self, action_idx: int) -> Tuple[float, float, float, bool]:
        """Zamienia indeks akcji (0-53) na konkretne wartości sterowania."""
        # Dekodowanie od końca (jak system liczbowy)
        fire_idx = action_idx % 2
        barrel_idx = (action_idx // 2) % 3
        hull_idx = (action_idx // 6) % 3
        move_idx = (action_idx // 18) % 3

        return MOVES[move_idx], TURNS[hull_idx], TURNS[barrel_idx], FIRES[fire_idx]

    def calculate_reward(self, current_state_vec, enemies_remaining, my_hp, enemy_hp_sum, action_taken):
        """Oblicza nagrodę na podstawie zmiany stanu."""
        reward = 0.0
        
        # --- Reward Shaping (Zagęszczanie nagród) ---
        # 0. Kara za upływ czasu (wymusza aktywność i kończenie gry)
        reward -= 0.01
        
        # Dane z aktualnego stanu
        curr_enemy_dist = current_state_vec[0][2].item()
        curr_enemy_angle = current_state_vec[0][3].item()
        curr_pup_dist = current_state_vec[0][5].item()
        
        # Dane z poprzedniego stanu (jeśli istnieje)
        prev_enemy_dist = 1.0
        prev_pup_dist = 1.0
        if self.last_state is not None:
            prev_enemy_dist = self.last_state[0][2].item()
            prev_pup_dist = self.last_state[0][5].item()

        # Dekodujemy akcję, aby wiedzieć co agent próbował zrobić
        move_val, hull_val, barrel_val, fire_val = self.decode_action(action_taken)

<<<<<<< Updated upstream
        # 1. Nagroda za celowanie (jeśli wróg widoczny)
        if curr_enemy_dist < 1.0:
            # Kąt jest znormalizowany (-1 do 1). 0.1 to ok. 18 stopni.
            if abs(curr_enemy_angle) < 0.1: 
                reward += 0.1
            
            # Nagroda za zbliżanie się przy ataku
            if curr_enemy_dist < prev_enemy_dist:
                reward += 0.1
=======
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
                # - destructible_ahead (1.0/0.0) <-- NEW
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
            # ammo_to_load: Optional[str] = None
            # should_fire: bool = False

        delta = action.heading_rotation_angle / MAX_HEADING_DELTA
        if abs(delta) > 1:
            raise ValueError("action.heading_rotation_angle not normalized in _compute_step_reward")
        parts["rotation"] = -(delta) ** 2 / 10

        recent = self.pos_history[-200:] or [current_pos]
        prev = self.pos_history[-400:-200] or recent
        rcx = sum(p[0] for p in recent) / float(len(recent))
        rcy = sum(p[1] for p in recent) / float(len(recent))
        pcx = sum(p[0] for p in prev) / float(len(prev))
        pcy = sum(p[1] for p in prev) / float(len(prev))
        var_r = sum((p[0] - rcx) ** 2 + (p[1] - rcy) ** 2 for p in recent) / float(len(recent))
        var_p = sum((p[0] - pcx) ** 2 + (p[1] - pcy) ** 2 for p in prev) / float(len(prev))
        parts["variance_recent"] = var_r / 50
        parts["variance_prev"] = var_p / 50
        parts["centroid_bonus"] = parts["variance_recent"] + parts["variance_prev"]

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
        parts["frontier_bonus"] = frontier_bonus

        reward = sum(parts.values())
        self.episode_reward_total += reward
        for key, value in parts.items():
            self.episode_reward_parts[key] = self.episode_reward_parts.get(key, 0.0) + value
        self.episode_reward_parts_steps += 1
>>>>>>> Stashed changes

        # 2. Nagroda za PowerUpy
        if curr_pup_dist < 1.0:
            # Duża nagroda za zebranie (bardzo blisko)
            if curr_pup_dist < 0.05:
                reward += 1.0
            # Mała nagroda za zbliżanie się
            elif curr_pup_dist < prev_pup_dist:
                reward += 0.1
        
        # Sprawdź czy strzelaliśmy niedawno (np. w ciągu ostatnich 25 ticków)
        ticks_since_fire = self.current_tick - self.last_fire_tick
        recently_fired = ticks_since_fire < 25
        
        # 1. Nagroda za zabicie wroga (Globalna zmiana licznika)
        if enemies_remaining < self.prev_enemies_count:
            if recently_fired:
                reward += 100.0
                print(f"[{self.name}] +100 Reward: Enemy Killed!")
            else:
                print(f"[{self.name}] Enemy died (passive). No reward.")
            
        # 2. Nagroda za zadanie obrażeń (Lokalna suma HP widocznych wrogów)
        # Jeśli suma HP wrogów spadła, a liczba wrogów jest ta sama (lub mniejsza), zadaliśmy dmg
        if enemy_hp_sum < self.prev_enemy_hp_sum:
            diff = self.prev_enemy_hp_sum - enemy_hp_sum
            # Ignorujemy ogromne skoki (np. wróg zniknął z pola widzenia)
            if diff < 50 and recently_fired: 
                reward += diff * 0.5
        
        # 3. Kara za otrzymanie obrażeń
        if my_hp < self.prev_my_hp:
            reward -= (self.prev_my_hp - my_hp) * 0.5
            
        # 4. Kara za Friendly Fire (z wektora stanu)
        # state[0][7] to aimed_at_friend. Jeśli akcja to ATTACK (0) i celujemy w przyjaciela -> kara
        aimed_at_friend = current_state_vec[0][7].item()
        if fire_val and aimed_at_friend > 0.5:
            reward -= 5.0 # Mniejsza kara, bo agent uczy się dopiero kontroli
            
        if self.experiment:
            self.experiment.log_metric("reward", reward, step=self.steps_done)
        return reward

<<<<<<< Updated upstream
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
=======

    def _maybe_train(self) -> None:
        if not self.training_enabled:
>>>>>>> Stashed changes
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

<<<<<<< Updated upstream
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])
=======
        min_required = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < min_required:
            return

        if self.total_steps % self.config.train_every != 0:
            return
>>>>>>> Stashed changes

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        # Dla stanów, które nie są końcowe (done=0)
        non_final_mask = (done_batch == 0)
        if non_final_mask.sum() > 0:
            next_state_values[non_final_mask] = self.target_net(next_state_batch[non_final_mask]).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        if self.experiment:
            self.experiment.log_metric("loss", loss.item(), step=self.steps_done)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_action(self, current_tick, my_tank_status, sensor_data, enemies_remaining):
        try:
            # --- FRAME SKIPPING ---
            # Jeśli to nie jest klatka decyzyjna i mamy poprzednią akcję, powtarzamy ją.
            if current_tick % FRAME_SKIP != 0 and self.last_action_cmd is not None:
                return self.last_action_cmd

            self.current_tick = current_tick
            # 1. Przygotowanie stanu
            state_vec, nearest_enemy = self.get_state_vector(my_tank_status, sensor_data, enemies_remaining)
            
            # 2. Obliczenie nagrody za POPRZEDNIĄ akcję i zapis do pamięci
            current_enemy_hp_sum = sum(t.get('hp', 0) for t in sensor_data.get('seen_tanks', []) if t.get('team') != my_tank_status.get('_team'))
            
            if self.last_state is not None:
                reward = self.calculate_reward(
                    state_vec, 
                    enemies_remaining, 
                    my_tank_status.get('hp', 100), 
                    current_enemy_hp_sum,
                    self.last_action.item()
                )
                
                # Zapisz tranzycję: (s, a, r, s', done=False)
                self.memory.push(
                    self.last_state, 
                    self.last_action, 
                    torch.tensor([reward], device=self.device), 
                    state_vec,
                    torch.tensor([0], device=self.device) # Done=0 bo gra trwa
                )
                
                # Trenuj sieć
                self.optimize_model()

            # 3. Wybór nowej akcji
            action_tensor = self.select_action(state_vec)
            action_idx = action_tensor.item()
            
            # 4. Aktualizacja zmiennych pomocniczych
            self.last_state = state_vec
            self.last_action = action_tensor
            self.prev_enemies_count = enemies_remaining
            self.prev_my_hp = my_tank_status.get('hp', 100)
            self.prev_enemy_hp_sum = current_enemy_hp_sum

<<<<<<< Updated upstream
            # 5. Tłumaczenie akcji wysokopoziomowej na ActionCommand
            cmd = ActionCommand()
            my_pos = my_tank_status.get('position')
            my_heading = my_tank_status.get('heading', 0)
            my_barrel = my_tank_status.get('barrel_angle', 0)
=======
            action_vec = self._select_action(current_obs.vector, training=self.training_enabled)

            # Deterministic Aiming & Firing
            if current_obs.enemy_visible:
                # Align hull toward target to keep it in vision
                hull_error_deg = (current_obs.enemy_hull_error - 0.5) * 360.0
                action_vec[1] = max(-MAX_HEADING_DELTA, min(MAX_HEADING_DELTA, hull_error_deg))

                # Aiming to the enemy based on barrel error; the model's barrel output can be used for fine-tuning, but here we rely on direct error for more consistent aiming.
                error_deg = (current_obs.enemy_barrel_error - 0.5) * 360.0
                # Use full speed for deterministic aiming (engine will clamp to tank limits)
                # FIX: Clamp to network limits to avoid breaking ReplayBuffer/Training
                action_vec[2] = max(-MAX_BARREL_DELTA, min(MAX_BARREL_DELTA, error_deg))

                # Fire if aimed within 15 degrees
                if abs(error_deg) < 15.0 and not current_obs.ally_fire_risk:
                    action_vec[3] = 1.0
                else:
                    action_vec[3] = 0.0
            elif current_obs.destructible_ahead:
                # Stop hull rotation to stabilize aiming
                action_vec[1] = 0.0

                # Jeśli brak wroga, ale jest ZNISZCZALNA przeszkoda na drodze -> strzelaj w nią
                my_pos = my_tank_status.get("position", {"x": 0.0, "y": 0.0})
                my_heading = float(my_tank_status.get("heading", 0.0) or 0.0)
                my_barrel = float(my_tank_status.get("barrel_angle", 0.0) or 0.0)
                barrel_abs = my_heading + my_barrel

                seen_obstacles = sensor_data.get("seen_obstacles", [])
                target_obs = None
                min_dist = float('inf')

                # Znajdź przeszkodę, która uruchomiła flagę obstacle_ahead (parametry zgodne ze StateEncoder)
                for obj in seen_obstacles:
                    # Celujemy tylko w drzewa
                    if obj.get("type") != "Tree":
                        continue

                    pos = obj.get("position")
                    if not pos: continue
                    dist = self.encoder._distance(my_pos, pos)
                    if dist > 12.0: continue
                    
                    # Find nearest tree in range regardless of angle
                    if dist < min_dist:
                        min_dist = dist
                        target_obs = obj
                
                if target_obs:
                    t_pos = target_obs.get("position")
                    target_angle = self.encoder._angle_to(my_pos, t_pos)
                    barrel_error = self.encoder.normalize_angle(target_angle - barrel_abs)
                    
                    # FIX: Clamp to network limits
                    action_vec[2] = max(-MAX_BARREL_DELTA, min(MAX_BARREL_DELTA, barrel_error))
                    if abs(barrel_error) < 15.0:
                        action_vec[3] = 1.0
                    else:
                        action_vec[3] = 0.0
                else:
                    # Fallback: skanowanie, jeśli nie udało się namierzyć konkretnej przeszkody
                    # FIX: Clamp to network limits (15.0 > 5.0 was invalid for training)
                    action_vec[2] = MAX_BARREL_DELTA
                    action_vec[3] = 0.0
            else:
                # Steady searching pattern when no enemy is visible
                # FIX: Clamp to network limits
                action_vec[2] = MAX_BARREL_DELTA
                action_vec[3] = 0.0

            command = self._to_command(action_vec, my_tank_status, current_obs)
>>>>>>> Stashed changes

            # Dekodowanie akcji z sieci neuronowej na sterowanie
            move_val, hull_val, barrel_val, fire_val = self.decode_action(action_idx)
            
            cmd.move_speed = move_val
            cmd.heading_rotation_angle = hull_val
            cmd.barrel_rotation_angle = barrel_val
            cmd.should_fire = fire_val
            
            if fire_val:
                self.last_fire_tick = current_tick

            self.last_action_cmd = cmd
            return cmd
        except Exception as e:
            print(f"[{self.name}] ERROR in get_action: {e}")
            import traceback
            traceback.print_exc()
            return ActionCommand()

    def destroy(self):
        """Kara za śmierć."""
        print(f"[{self.name}] DESTROYED! Penalty applied.")
        if self.last_state is not None:
            reward = -100.0
            # Kara -100, stan końcowy (done=1)
            self.memory.push(
                self.last_state,
                self.last_action,
                torch.tensor([reward], device=self.device),
                self.last_state, # Stan nie ma znaczenia bo done=1
                torch.tensor([1], device=self.device)
            )
            self.optimize_model()
            if self.experiment:
                self.experiment.log_metric("reward", reward, step=self.steps_done)
        
        # Reset stanu epizodu
        self.last_state = None
        self.last_action = None

    def end(self, damage_dealt, tanks_killed):
        """Zapis modelu na koniec gry."""
        print(f"[{self.name}] Game Over. Saving model...")
        
        # Logowanie wyników końcowych do Comet ML
        if self.experiment:
            self.experiment.log_metrics({
                "game_damage_dealt": damage_dealt,
                "game_tanks_killed": tanks_killed
            }, step=self.steps_done)

        # Zapis pełnego stanu (model + optymalizator + kroki)
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, MODEL_PATH)

# ============================================================================
# SERWER FASTAPI
# ============================================================================

app = FastAPI()
agent = DQNAgent()

@app.get("/")
async def root():
    return {"message": f"Agent {agent.name} is running"}

@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    return agent.get_action(
        payload.get('current_tick', 0),
        payload.get('my_tank_status', {}),
        payload.get('sensor_data', {}),
        payload.get('enemies_remaining', 0)
    )

@app.post("/agent/destroy", status_code=204)
async def destroy():
    agent.destroy()

@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    agent.end(
        payload.get('damage_dealt', 0.0),
        payload.get('tanks_killed', 0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default="DQNAgent")
    args = parser.parse_args()
    
    agent.name = args.name
    agent.init_comet() # Inicjalizacja logowania dopiero po ustawieniu nazwy
    print(f"Starting DQN Agent {agent.name} on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
