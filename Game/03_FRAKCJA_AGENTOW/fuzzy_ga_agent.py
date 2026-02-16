"""
Parametric Agent for Fuzzy/GA Optimization.
Agent przyjmujący parametry zachowania z pliku JSON (dla algorytmu genetycznego).
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, Optional

# --- Konfiguracja ścieżek importu (podobnie jak w random_agent.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dodanie ścieżki do kontrolera (dla definicji API jeśli potrzebne)
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

# Dodanie ścieżki do głównego katalogu silnika
parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

# ============================================================================
# MODEL DANYCH
# ============================================================================

class ActionCommand(BaseModel):
    """Komenda wysyłana do silnika."""
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: Optional[str] = None
    should_fire: bool = False

# Domyślne parametry (używane, jeśli nie podano pliku lub brakuje klucza)
DEFAULT_PARAMS = {
    "near_distance": 22.0,
    "mid_distance": 55.0,
    "far_distance": 110.0,
    "preferred_distance": 42.0,
    "low_hp_ratio": 0.35,
    "aggression": 0.75,
    "retreat_bias": 0.55,
    "heading_gain": 1.25,
    "barrel_gain": 1.7,
    "distance_hold_gain": 0.85,
    "advance_speed": 0.85,
    "retreat_speed": -0.75,
    "explore_speed": 0.30,
    "explore_turn": 14.0,
    "fire_alignment_deg": 8.0,
    "fire_threshold": 0.46,
    "scan_speed": 20.0,
    "scan_arc": 55.0,
}

# ============================================================================
# LOGIKA AGENTA
# ============================================================================

class FuzzyGenAgent:
    def __init__(self, name: str, params: Dict[str, float]):
        self.name = name
        self.params = params
        self.is_destroyed = False
        
        # Stan wewnętrzny dla eksploracji
        self.scan_dir = 1.0
        self.scan_angle_current = 0.0
        self.explore_timer = 0
        self.explore_heading_target = 0.0

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        p = self.params
        
        # Pobranie stanu własnego
        my_pos = my_tank_status.get('position', {'x': 0.0, 'y': 0.0})
        my_hp = my_tank_status.get('hp', 100)
        my_max_hp = my_tank_status.get('_max_hp', 100)
        my_heading = my_tank_status.get('heading', 0.0)
        my_barrel = my_tank_status.get('barrel_angle', 0.0)
        my_team = my_tank_status.get('_team')

        # Wyjścia
        barrel_rot = 0.0
        heading_rot = 0.0
        speed = 0.0
        fire = False

        # 1. Wybór celu
        seen_tanks = sensor_data.get('seen_tanks', [])
        # Filtrujemy wrogów (inny team)
        enemies = [t for t in seen_tanks if t.get('team') != my_team]
        
        target = None
        if enemies:
            # Wybierz najbliższego wroga
            target = min(enemies, key=lambda t: math.hypot(
                t['position']['x'] - my_pos['x'], 
                t['position']['y'] - my_pos['y']
            ))
        
        if target:
            # --- TRYB WALKI ---
            t_pos = target['position']
            dx = t_pos['x'] - my_pos['x']
            dy = t_pos['y'] - my_pos['y']
            dist = math.hypot(dx, dy)
            target_angle_abs = math.degrees(math.atan2(dy, dx))
            
            # Obliczanie różnicy kątów (znormalizowane do -180..180)
            current_barrel_abs = (my_heading + my_barrel) % 360
            angle_diff = (target_angle_abs - current_barrel_abs + 180) % 360 - 180
            
            # Obrót wieży (Proportional Controller)
            barrel_rot = angle_diff * p['barrel_gain']
            
            # Obrót kadłuba (chcemy być przodem do wroga)
            heading_diff = (target_angle_abs - my_heading + 180) % 360 - 180
            heading_rot = heading_diff * p['heading_gain']
            
            # Logika ruchu (Fuzzy-like)
            hp_ratio = my_hp / max(1, my_max_hp)
            is_low_hp = hp_ratio < p['low_hp_ratio']
            
            # Ustalanie preferowanego dystansu
            desired_dist = p['preferred_distance']
            if is_low_hp:
                # Jeśli mało HP, zwiększ dystans (ucieczka/kampienie)
                desired_dist *= (1.0 + p['retreat_bias'])
            
            dist_error = dist - desired_dist
            
            # Kontrola prędkości w zależności od błędu dystansu
            if dist_error > 0: 
                # Za daleko -> Jedź do przodu
                # distance_hold_gain reguluje jak agresywnie dążymy do dystansu
                speed = p['advance_speed'] * min(1.0, dist_error * 0.1 * p['distance_hold_gain']) * 100.0
            else: 
                # Za blisko -> Cofaj
                speed = p['retreat_speed'] * min(1.0, abs(dist_error) * 0.1 * p['distance_hold_gain']) * 100.0
                
            # Modyfikator agresji (jeśli mamy dużo HP, atakuj mocniej)
            if not is_low_hp:
                speed *= p['aggression']
            
            # Ograniczenie prędkości do zakresu zdefiniowanego w genach
            # (parametry speed są znormalizowane 0-1 lub -1-0, mnożymy przez 100 dla silnika)
            min_s = p['retreat_speed'] * 100.0
            max_s = p['advance_speed'] * 100.0
            speed = max(min_s, min(max_s, speed))
            
            # Decyzja o strzale
            # Strzelaj tylko jeśli lufa jest w miarę wycelowana
            if abs(angle_diff) < p['fire_alignment_deg']:
                 # Prosty próg losowości (symulacja czasu reakcji/błędu)
                 if random.random() > (1.0 - p['fire_threshold']):
                     fire = True

        else:
            # --- TRYB EKSPLORACJI ---
            # Losowe błądzenie (Random Walk)
            self.explore_timer -= 1
            if self.explore_timer <= 0:
                self.explore_timer = random.randint(20, 60)
                # Losowy zwrot w zakresie +/- explore_turn
                self.explore_heading_target = random.uniform(-p['explore_turn'], p['explore_turn'])
            
            heading_rot = self.explore_heading_target
            speed = p['explore_speed'] * 100.0
            
            # Skanowanie wieżą (radar)
            self.scan_angle_current += p['scan_speed'] * self.scan_dir
            # Jeśli przekroczyliśmy kąt skanowania, zmień kierunek
            if abs(self.scan_angle_current) > p['scan_arc']:
                self.scan_dir *= -1
                self.scan_angle_current = p['scan_arc'] * self.scan_dir
            
            barrel_rot = p['scan_speed'] * self.scan_dir

        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=heading_rot,
            move_speed=speed,
            should_fire=fire
        )

    def destroy(self):
        self.is_destroyed = True

    def end(self, damage, kills):
        pass

# ============================================================================
# SERWER FASTAPI
# ============================================================================

app = FastAPI()
agent_instance: Optional[FuzzyGenAgent] = None

@app.post("/agent/action", response_model=ActionCommand)
async def action(payload: Dict[str, Any] = Body(...)):
    return agent_instance.get_action(
        payload.get('current_tick', 0),
        payload.get('my_tank_status', {}),
        payload.get('sensor_data', {}),
        payload.get('enemies_remaining', 0)
    )

@app.post("/agent/destroy")
async def destroy():
    if agent_instance: agent_instance.destroy()

@app.post("/agent/end")
async def end(payload: Dict[str, Any] = Body(...)):
    if agent_instance: agent_instance.end(payload.get('damage_dealt', 0), payload.get('tanks_killed', 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default="FuzzyBot")
    parser.add_argument("--params-file", type=str, default=None, help="Path to JSON with agent parameters")
    args = parser.parse_args()
    
    # Wczytywanie parametrów
    params = DEFAULT_PARAMS.copy()
    if args.params_file and os.path.exists(args.params_file):
        try:
            with open(args.params_file, 'r') as f:
                loaded_params = json.load(f)
                # Aktualizuj tylko znane klucze
                for k, v in loaded_params.items():
                    if k in params:
                        params[k] = v
            print(f"[{args.name}] Loaded params from {args.params_file}")
        except Exception as e:
            print(f"[{args.name}] Failed to load params: {e}")
    else:
        print(f"[{args.name}] Using default params")
            
    agent_instance = FuzzyGenAgent(args.name, params)
    print(f"Starting {args.name} on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)