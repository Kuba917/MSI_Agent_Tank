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
from ANFISDQN import ANFISDQN

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
GAMMA = 0.99        # Discount factor
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
API_KEY = "L2PzW7c3YM3WqM5hNfCsloeLZ"
PROJECT_NAME = "msi-projekt"
WORKSPACE = "kluski777"

# Definicja akcji niskopoziomowych (Kombinatoryka) - poki co niech zostana hardkodowane, ale czy my wiemy czy to jest duzo czy malo chyba nie.
# Move (3) * Hull (3) * Barrel (3) * Fire (2) = 54 akcje
MOVES = [0.0, 100.0, -100.0]  # Stop, Przód, Tył
TURNS = [0.0, -15.0, 15.0]    # Brak, Lewo, Prawo
FIRES = [False, True]         # Nie strzelaj, Strzelaj

NUM_ACTIONS = 3 * 3 * 3 * 2  # 54

# ============================================================================
# MODEL SIECI NEURONOWEJ
# ============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): # najgorsze jest to ze to nie jest ANFIS
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
    barrel_rotation_angle: float = 0.0      # -> Wartosc ciagla
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
        
        # generalnie bym rozdzielil akcje na te ktore sa od siebie niezalezne
        self.policy_net = ANFISDQN(self.input_dim, NUM_ACTIONS).to(self.device)
        self.target_net = ANFISDQN(self.input_dim, NUM_ACTIONS).to(self.device)
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

    def normalize_angle(self, angle):
        """Sprowadza kąt do zakresu [-180, 180]."""
        return ((angle + 180) % 360) - 180
    
    def get_state_vector(self, my_status, sensor_data, enemies_remaining): # chyba wszystko jest normalizowane
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
        
        # imo w powerupach logika jest bledna, jak nie istieje jeszcze zaden powerup (mozna by zalozyc ze jednak istnieje)
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
        else:
            return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=self.device, dtype=torch.long)

    def decode_action(self, action_idx: int) -> Tuple[float, float, float, bool]: #! IMO oddzielnie to powinno isc
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

        # 1. Nagroda za celowanie (jeśli wróg widoczny)
        if curr_enemy_dist < 1.0:
            # Kąt jest znormalizowany (-1 do 1). 0.1 to ok. 18 stopni.
            if abs(curr_enemy_angle) < 0.1: 
                reward += 0.1
            
            # Nagroda za zbliżanie się przy ataku
            if curr_enemy_dist < prev_enemy_dist:
                reward += 0.1

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

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])

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

            # 5. Tłumaczenie akcji wysokopoziomowej na ActionCommand
            cmd = ActionCommand()
            my_pos = my_tank_status.get('position')
            my_heading = my_tank_status.get('heading', 0)
            my_barrel = my_tank_status.get('barrel_angle', 0)

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
