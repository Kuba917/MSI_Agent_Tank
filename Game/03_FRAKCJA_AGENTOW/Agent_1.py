"""
Random Walking and Shooting Agent for Testing
Agent losowo chodzący i strzelający do testów
Rule-based tank agent used as a strong and readable baseline.

This agent implements IAgentController and performs random actions:
- Random barrel and heading rotation
- Random movement speed
- Random shooting
The agent has two modes:
1. Combat mode when at least one enemy is visible.
2. Exploration mode when no enemy is visible.

Usage:
    python random_agent.py --port 8001
    
To run multiple agents:
    python random_agent.py --port 8001  # Tank 1
    python random_agent.py --port 8002  # Tank 2
    ...
It avoids friendly fire, keeps moving to discover enemies, and applies
simple aiming logic before shooting.
"""

<<<<<<< Updated upstream
=======
from __future__ import annotations

import argparse
import math
import os
import heapq
>>>>>>> Stashed changes
import random
import argparse
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body
from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn
import math


# ============================================================================
# ACTION COMMAND MODEL
# ============================================================================
# Add engine paths for local runs.
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA", "controller")
engine_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA")
sys.path.insert(0, controller_dir)
sys.path.insert(0, engine_dir)


class ActionCommand(BaseModel):
    """Output action from agent to engine."""
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    ammo_to_load: Optional[str] = None
    should_fire: bool = False


# ============================================================================
# RANDOM AGENT LOGIC
# ============================================================================
@dataclass
class TargetInfo:
    distance: float
    hull_error: float
    barrel_error: float
    position: Dict[str, float]

class RandomAgent:
    """
    Agent with more structured, stateful behavior for testing purposes.
    Drives in one direction for a while, then changes.
    Scans with its turret.
    """
    
    def __init__(self, name: str = "TestBot"):

class RuleBasedCombatAgent:
    """Simple but effective baseline for 5v5 training opponents."""

    def __init__(self, name: str = "RuleBasedBot"):
        self.name = name
        self.is_destroyed = False
        print(f"[{self.name}] Agent initialized")

<<<<<<< Updated upstream
        # State for movement
        self.move_timer = 15
        self.current_move_speed = 30.0

        # State for hull rotation
        self.heading_timer = 0
        self.current_heading_rotation = 0.0
=======
        # State machine: "SCAN" (initial 360) -> "MOVE" (to center)
        self.mode = "SCAN"
        self.total_scan_angle = 0.0
        
        # A* Pathfinding state
        self.grid_size = 10.0  # Size of each tile in the navigation grid
        self.blocked_cells = set() # Set of (x, y) grid coordinates that are blocked
        self.current_path = []     # List of (x, y) world coordinates to follow
        self.last_path_calc_tick = -100
>>>>>>> Stashed changes

        # State for barrel scanning
        self.barrel_scan_direction = 1.0  # 1.0 for right, -1.0 for left
        self.barrel_rotation_speed = 15.0
        print(f"[{self.name}] initialized")

<<<<<<< Updated upstream
        # State for aiming before shooting
        self.aim_timer = 0  # Ticks to wait before firing
    
    def get_action(
        self, 
        current_tick: int, 
        my_tank_status: Dict[str, Any], 
        sensor_data: Dict[str, Any], 
        enemies_remaining: int
=======
    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def angle_to(source: Dict[str, float], target: Dict[str, float]) -> float:
        dx = target["x"] - source["x"]
        dy = target["y"] - source["y"]
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _ammo_counts(my_status: Dict[str, Any]) -> Dict[str, int]:
        ammo_data = my_status.get("ammo", {}) or {}
        counts: Dict[str, int] = {}
        for ammo_name, slot in ammo_data.items():
            key = str(ammo_name).upper()
            count = int((slot or {}).get("count", 0) or 0)
            counts[key] = max(0, count)
        return counts

    def _select_ammo(
        self,
        my_status: Dict[str, Any],
        target: Optional[TargetInfo],
    ) -> Optional[str]:
        counts = self._ammo_counts(my_status)
        if not counts:
            return None

        if target is None:
            current = str(my_status.get("ammo_loaded") or "").upper()
            if current and counts.get(current, 0) > 0:
                return current
            return max(counts.items(), key=lambda item: item[1])[0] if counts else None

        if target.distance > 50.0:
            preferred = ["LONG_DISTANCE", "LIGHT", "HEAVY"]
        elif target.distance > 25.0:
            preferred = ["LIGHT", "LONG_DISTANCE", "HEAVY"]
        else:
            preferred = ["HEAVY", "LIGHT", "LONG_DISTANCE"]

        for ammo_name in preferred:
            if counts.get(ammo_name, 0) > 0:
                return ammo_name

        current = str(my_status.get("ammo_loaded") or "").upper()
        if current and counts.get(current, 0) > 0:
            return current

        return max(counts.items(), key=lambda item: item[1])[0] if counts else None

    def _find_nearest_enemy(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> Optional[TargetInfo]:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)

        enemies = [
            tank
            for tank in sensor_data.get("seen_tanks", [])
            if tank.get("team") != my_team
        ]
        if not enemies:
            return None

        nearest = min(enemies, key=lambda tank: float(tank.get("distance", 1e9)))
        enemy_pos = nearest.get("position", {"x": 0.0, "y": 0.0})

        target_angle = self.angle_to(my_pos, enemy_pos)
        hull_error = self.normalize_angle(target_angle - my_heading)
        barrel_abs = my_heading + my_barrel
        barrel_error = self.normalize_angle(target_angle - barrel_abs)

        return TargetInfo(
            distance=float(nearest.get("distance", 999.0)),
            hull_error=hull_error,
            barrel_error=barrel_error,
            position=enemy_pos,
        )

    def _ally_in_fire_line(self, my_status: Dict[str, Any], sensor_data: Dict[str, Any]) -> bool:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)
        barrel_abs = my_heading + my_barrel

        for tank in sensor_data.get("seen_tanks", []):
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            angle = self.angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 4.0:
                return True
        return False

    def _danger_ahead(self, my_status: Dict[str, Any], sensor_data: Dict[str, Any]) -> bool:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_heading = my_status.get("heading", 0.0)
        my_team = my_status.get("_team")

        for obstacle in sensor_data.get("seen_obstacles", []):
            pos = obstacle.get("position", {"x": 0.0, "y": 0.0})
            dx = pos["x"] - my_pos["x"]
            dy = pos["y"] - my_pos["y"]
            distance = math.hypot(dx, dy)
            if distance > 12.0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            rel = abs(self.normalize_angle(angle - my_heading))
            if rel < 30.0:
                return True

        for terrain in sensor_data.get("seen_terrains", []):
            terrain_type = terrain.get("type")
            if terrain_type not in {"Water", "PotholeRoad"}:
                continue
            pos = terrain.get("position", {"x": 0.0, "y": 0.0})
            dx = pos["x"] - my_pos["x"]
            dy = pos["y"] - my_pos["y"]
            distance = math.hypot(dx, dy)
            if distance > 10.0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            rel = abs(self.normalize_angle(angle - my_heading))
            if rel < 35.0:
                return True

        # Avoid allies to prevent overlapping
        for tank in sensor_data.get("seen_tanks", []):
            if tank.get("team") != my_team:
                continue
            pos = tank.get("position", {"x": 0.0, "y": 0.0})
            dx = pos["x"] - my_pos["x"]
            dy = pos["y"] - my_pos["y"]
            distance = math.hypot(dx, dy)
            if distance > 15.0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            rel = abs(self.normalize_angle(angle - my_heading))
            if rel < 35.0:
                return True

        return False

    def _combat_action(
        self,
        my_status: Dict[str, Any],
        target: TargetInfo,
        sensor_data: Dict[str, Any],
>>>>>>> Stashed changes
    ) -> ActionCommand:
        """Generate a stateful, predictable action for testing."""

        my_pos = my_tank_status.get('position', {'x': 0.0, 'y': 0.0})
        my_heading = my_tank_status.get('heading', 0.0)
        my_barrel_rel = my_tank_status.get('barrel_angle', 0.0)

        should_fire = False
        heading_rotation = 0.0
        barrel_rotation = 0.0
        move_speed = 0.0
        should_fire = False

        seen_tanks = sensor_data.get('seen_tanks', [])
        my_team = my_tank_status.get('_team')
        
        # Filtrowanie: wybieramy tylko czołgi, które mają inny numer drużyny niż my
        enemies = [t for t in seen_tanks if t.get('team') != my_team]
        
        if enemies:
            # ==========================================
            # FAZA WALKI: WIDZIMY WROGA
            # ==========================================
            
            # Pobieramy pierwszego WROGA z przefiltrowanej listy
            target = enemies[0] 
            target_pos = target.get('position', {'x': 0.0, 'y': 0.0})
        # Keep rotations small: engine applies rotation per tick.
        max_hull_step = 2.0
        max_barrel_step = 3.0

            # Obliczanie wektora kierunku do wroga
            dx = target_pos['x'] - my_pos['x']
            dy = target_pos['y'] - my_pos['y']
        # Align hull toward target if the error is large.
        if abs(target.hull_error) > 4.0:
            heading_rotation = max(-max_hull_step, min(max_hull_step, target.hull_error))

            # Obliczanie bezwzględnego kąta do wroga (w stopniach)
            target_angle_abs = math.degrees(math.atan2(dy, dx))
        # Always align barrel.
        barrel_rotation = max(-max_barrel_step, min(max_barrel_step, target.barrel_error))

<<<<<<< Updated upstream
            # Nasz obecny, bezwzględny kąt lufy w świecie gry
            current_barrel_abs = my_heading + my_barrel_rel

            # Obliczamy różnicę kątów i normalizujemy ją do przedziału [-180, 180]
            angle_diff = (target_angle_abs - current_barrel_abs) % 360
            if angle_diff > 180:
                angle_diff -= 360

            # --- OBRACANIE LUFY W STRONĘ CELU ---
            max_rotation = self.barrel_rotation_speed # np. 15.0
            
            # Jeśli jesteśmy blisko celu, obracamy się tylko o tyle, ile brakuje
            if abs(angle_diff) <= max_rotation:
                barrel_rotation = angle_diff 
            else:
                # W przeciwnym razie kręcimy się z maksymalną prędkością w odpowiednią stronę
                barrel_rotation = max_rotation if angle_diff > 0 else -max_rotation

            # --- DECYZJA O STRZALE ---
            # Jeśli lufa celuje prawie idealnie we wroga (margines błędu 3 stopnie)
            if abs(angle_diff) < 3.0:
                should_fire = True

            # Czołg zatrzymuje kadłub na czas celowania (zwiększa precyzję)
            move_speed = 0.0
            heading_rotation = 0.0
=======
        # Check if ally is blocking the shot
        ally_blocking = self._ally_in_fire_line(my_status, sensor_data)

        if ally_blocking:
            # Flanking maneuver: turn 45 degrees relative to target and move to find a new angle
            # We use a fixed offset to try to go around
            heading_rotation = 4.0  # Sharp turn
            move_speed = 100.0      # Full speed to reposition
            should_fire = False
        else:
            # Standard combat movement
            # Keep distance: close in when far, retreat when too close.
            if target.distance > 7.0:
                move_speed = 80.0
            elif target.distance < 3.5:
                move_speed = -70.0
            else:
                move_speed = 0.0
            
            reload_timer = float(my_status.get("_reload_timer", 0.0) or 0.0)
            if abs(target.barrel_error) < 2.8 and reload_timer <= 0.0:
                should_fire = True
>>>>>>> Stashed changes

        else:
            # ==========================================
            # FAZA EKSPLORACJI: SZUKAMY WROGA
            # ==========================================
            
            # --- Ruch kadłuba (jazda i skręcanie) ---
            self.heading_timer -= 1
            if self.heading_timer <= 0:
                # Losujemy, czy skręcać, czy jechać prosto
                self.current_heading_rotation = random.choice([-15.0, 0.0, 15.0])
                self.heading_timer = random.randint(20, 60)
            heading_rotation = self.current_heading_rotation
        if self._danger_ahead(my_status, sensor_data) and move_speed > 0:
            # Avoid deadlocks near obstacles/water: move slowly and turn away.
            move_speed = 40.0
            heading_rotation = 2.5 if random.random() > 0.5 else -2.5

<<<<<<< Updated upstream
            self.move_timer -= 1
            if self.move_timer <= 0:
                # Większa szansa na jazdę do przodu, żeby sprawniej odkrywać mapę
                self.current_move_speed = random.choice([30.0])
                self.move_timer = random.randint(10, 40)
            move_speed = self.current_move_speed
=======
        ammo_to_load = self._select_ammo(my_status, target)
>>>>>>> Stashed changes

            # --- Skanowanie radarem (lufą) w trakcie jazdy ---
            if my_barrel_rel > 45.0:
                self.barrel_scan_direction = -1.0
            elif my_barrel_rel < -45.0:
                self.barrel_scan_direction = 1.0
            barrel_rotation = self.barrel_rotation_speed * self.barrel_scan_direction
        
        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=move_speed,
            should_fire=should_fire
            ammo_to_load=ammo_to_load,
            should_fire=should_fire,
        )
<<<<<<< Updated upstream
    
    def destroy(self):
        """Called when tank is destroyed."""
=======

    def _grid_coords(self, x: float, y: float) -> tuple[int, int]:
        """Converts world coordinates to grid coordinates."""
        return int(x // self.grid_size), int(y // self.grid_size)

    def _world_coords(self, gx: int, gy: int) -> tuple[float, float]:
        """Converts grid coordinates to world coordinates (center of tile)."""
        return gx * self.grid_size + self.grid_size / 2, gy * self.grid_size + self.grid_size / 2

    def _update_map(self, sensor_data: Dict[str, Any]):
        """Updates the internal blocked cells map based on sensor data."""
        # Mark static obstacles
        for obj in sensor_data.get("seen_obstacles", []):
            pos = obj.get("position")
            if pos:
                gx, gy = self._grid_coords(pos["x"], pos["y"])
                self.blocked_cells.add((gx, gy))
                
        # Mark dangerous terrain (Water)
        for terrain in sensor_data.get("seen_terrains", []):
            if terrain.get("type") == "Water":
                pos = terrain.get("position")
                if pos:
                    gx, gy = self._grid_coords(pos["x"], pos["y"])
                    self.blocked_cells.add((gx, gy))

    def _astar(self, start_pos: Dict[str, float], end_pos: tuple[float, float]) -> List[tuple[float, float]]:
        """Calculates A* path from start to end."""
        start_node = self._grid_coords(start_pos["x"], start_pos["y"])
        end_node = self._grid_coords(end_pos[0], end_pos[1])

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: math.hypot(start_node[0]-end_node[0], start_node[1]-end_node[1])}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end_node:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self._world_coords(current[0], current[1]))
                    current = came_from[current]
                path.reverse()
                return path

            # 8-way movement
            neighbors = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds (assuming 200x200 map approx)
                if not (0 <= neighbor[0] <= 20 and 0 <= neighbor[1] <= 20):
                    continue
                
                if neighbor in self.blocked_cells:
                    continue

                # Cost: 1 for straight, 1.41 for diagonal
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = math.hypot(neighbor[0]-end_node[0], neighbor[1]-end_node[1])
                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [] # No path found

    def _move_to_center_action(self, my_status: Dict[str, Any], sensor_data: Dict[str, Any]) -> ActionCommand:
        self._update_map(sensor_data)
        
        # Map center approximation (assuming 200x200 map based on DQN config, or generic center)
        center_x, center_y = 100.0, 100.0
        
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_heading = float(my_status.get("heading", 0.0) or 0.0)

        # Recalculate path every 20 ticks or if empty
        current_tick = int(time.time() * 10) # Mock tick using time if not passed, but we rely on state
        # Note: In real usage we should pass tick to this function, but for now we just check if path is empty
        # Calculate vector to center
        dx = center_x - my_pos["x"]
        dy = center_y - my_pos["y"]
        dist_to_center = math.hypot(dx, dy)
        
        if not self.current_path:
             self.current_path = self._astar(my_pos, (center_x, center_y))

        target_x, target_y = center_x, center_y
        
        # Follow the path
        if self.current_path:
            # Get next waypoint
            wp_x, wp_y = self.current_path[0]
            dist_to_wp = math.hypot(wp_x - my_pos["x"], wp_y - my_pos["y"])
            
            if dist_to_wp < 5.0:
                self.current_path.pop(0) # Reached waypoint
            else:
                target_x, target_y = wp_x, wp_y

        # Navigate to current target (waypoint or center)
        dx = target_x - my_pos["x"]
        dy = target_y - my_pos["y"]
        
        desired_angle = math.degrees(math.atan2(dy, dx))
        heading_error = self.normalize_angle(desired_angle - my_heading)
        heading_rotation = max(-3.0, min(3.0, heading_error))
        move_speed = 80.0

        dist_to_center = math.hypot(center_x - my_pos["x"], center_y - my_pos["y"])
        if dist_to_center < 10.0:
             move_speed = 0.0
             heading_rotation = 2.0
            # Arrived at center area, just patrol slowly
            move_speed = 20.0
            heading_rotation = 2.0
        else:
            # Navigate to center
            desired_angle = math.degrees(math.atan2(dy, dx))
            heading_error = self.normalize_angle(desired_angle - my_heading)
            heading_rotation = max(-3.0, min(3.0, heading_error))
            move_speed = 80.0

        # Reactive avoidance for dynamic objects (allies) not in grid
        # Obstacle avoidance overrides path to center
        if self._danger_ahead(my_status, sensor_data):
            move_speed = 30.0
            # Turn away sharply
            heading_rotation = 4.0
            # Force path recalculation next time
            self.current_path = []

        # Continuous 360 barrel rotation while moving
        barrel_rotation = 5.0

        ammo_to_load = self._select_ammo(my_status, target=None)

        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=move_speed,
            ammo_to_load=ammo_to_load,
            should_fire=False,
        )

    def get_action(
        self,
        current_tick: int,
        my_tank_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
    ) -> ActionCommand:
        if self.is_destroyed:
            return ActionCommand()

        # 1. Combat Priority: If enemy seen, stop and shoot
        target = self._find_nearest_enemy(my_tank_status, sensor_data)
        if target is not None:
            return self._combat_action(my_tank_status, target, sensor_data)

        # 2. Initial Scan Phase
        if self.mode == "SCAN":
            scan_step = 5.0
            self.total_scan_angle += scan_step
            if self.total_scan_angle >= 360.0:
                self.mode = "MOVE"
            
            return ActionCommand(
                barrel_rotation_angle=scan_step,
                heading_rotation_angle=0.0,
                move_speed=0.0
            )

        # 3. Move to Center Phase (A*)
        return self._move_to_center_action(my_tank_status, sensor_data)

    def destroy(self) -> None:
>>>>>>> Stashed changes
        self.is_destroyed = True
        print(f"[{self.name}] Tank destroyed!")
    
    def end(self, damage_dealt: float, tanks_killed: int):
        """Called when game ends."""
        print(f"[{self.name}] Game ended!")
        print(f"[{self.name}] Damage dealt: {damage_dealt}")
        print(f"[{self.name}] Tanks killed: {tanks_killed}")
        print(f"[{self.name}] destroyed")

<<<<<<< Updated upstream
=======
    def end(self, damage_dealt: float, tanks_killed: int) -> None:
        print(
            f"[{self.name}] end | damage={damage_dealt:.1f} kills={tanks_killed}"
        )
        # Allow persistent trainer mode: revive local episode state for next match.
        self.is_destroyed = False
        self.mode = "SCAN"
        self.total_scan_angle = 0.0
        self.blocked_cells.clear()
        self.current_path = []
>>>>>>> Stashed changes

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(
    title="Random Test Agent",
    description="Random walking and shooting agent for testing",
    version="1.0.0"
    title="Rule-Based Agent",
    description="Readable baseline agent for tank battles",
    version="2.0.0",
)
agent = RuleBasedCombatAgent()

# Global agent instance
agent = RandomAgent()


@app.get("/")
async def root():
async def root() -> Dict[str, Any]:
    return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    """Main endpoint called each tick by the engine."""
    print(payload)
    action = agent.get_action(
        current_tick=payload.get('current_tick', 0),
        my_tank_status=payload.get('my_tank_status', {}),
        sensor_data=payload.get('sensor_data', {}),
        enemies_remaining=payload.get('enemies_remaining', 0)
async def get_action(payload: Dict[str, Any] = Body(...)) -> ActionCommand:
    return agent.get_action(
        current_tick=payload.get("current_tick", 0),
        my_tank_status=payload.get("my_tank_status", {}),
        sensor_data=payload.get("sensor_data", {}),
        enemies_remaining=payload.get("enemies_remaining", 0),
    )
    return action


@app.post("/agent/destroy", status_code=204)
async def destroy():
    """Called when the tank is destroyed."""
@app.post("/agent/destroy", status_code=204, response_model=None)
async def destroy(payload: Dict[str, Any] = Body(None)) -> None:
    agent.destroy()


@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    """Called when the game ends."""
@app.post("/agent/end", status_code=204, response_model=None)
async def end(payload: Dict[str, Any] = Body(...)) -> None:
    agent.end(
        damage_dealt=payload.get('damage_dealt', 0.0),
        tanks_killed=payload.get('tanks_killed', 0)
        damage_dealt=float(payload.get("damage_dealt", 0.0) or 0.0),
        tanks_killed=int(payload.get("tanks_killed", 0) or 0),
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random test agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default=None, help="Agent name")
    parser = argparse.ArgumentParser(description="Run rule-based tank agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    

    if args.name:
        agent.name = args.name
    else:
        agent.name = f"RandomBot_{args.port}"
    
        agent.name = f"RuleBot_{args.port}"

    print(f"Starting {agent.name} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
