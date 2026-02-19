"""
Rule-based tank agent used as a strong and readable baseline.

The agent has two modes:
1. Combat mode when at least one enemy is visible.
2. Exploration mode when no enemy is visible.

It avoids friendly fire, keeps moving to discover enemies, and applies
simple aiming logic before shooting.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn


# Add engine paths for local runs.
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA", "controller")
engine_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA")
sys.path.insert(0, controller_dir)
sys.path.insert(0, engine_dir)


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: Optional[str] = None
    should_fire: bool = False


@dataclass
class TargetInfo:
    enemy_id: str
    distance: float
    hull_error: float
    barrel_error: float
    position: Dict[str, float]


class RuleBasedCombatAgent:
    """Simple but effective baseline for 5v5 training opponents."""

    def __init__(self, name: str = "RuleBasedBot"):
        self.name = name
        self.is_destroyed = False

        self.scan_direction = 1.0
        self.scan_speed = 3.0
        self.scan_burst_ticks_left = 0
        self.scan_pause_ticks_left = 0

        self.turn_direction = 1.0
        self.turn_ticks_left = 0
        self.idle_ticks = 0
        self.locked_enemy_id: Optional[str] = None
        self.last_enemy_position: Optional[Dict[str, float]] = None
        self.last_enemy_seen_tick = -10_000

        print(f"[{self.name}] initialized")

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
    def _distance(source: Dict[str, float], target: Dict[str, float]) -> float:
        dx = target["x"] - source["x"]
        dy = target["y"] - source["y"]
        return math.hypot(dx, dy)

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

        def _enemy_key(tank: Dict[str, Any]) -> Tuple[int, float, str]:
            enemy_id = str(tank.get("id", ""))
            enemy_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            fallback_distance = self._distance(my_pos, enemy_pos)
            try:
                dist = float(tank.get("distance", fallback_distance) or fallback_distance)
            except (TypeError, ValueError):
                dist = fallback_distance
            lock_priority = 0 if enemy_id and enemy_id == self.locked_enemy_id else 1
            return (lock_priority, dist, enemy_id)

        nearest = min(enemies, key=_enemy_key)
        enemy_id = str(nearest.get("id", ""))
        if enemy_id:
            self.locked_enemy_id = enemy_id
        enemy_pos = nearest.get("position", {"x": 0.0, "y": 0.0})
        try:
            enemy_distance = float(nearest.get("distance", self._distance(my_pos, enemy_pos)) or 999.0)
        except (TypeError, ValueError):
            enemy_distance = self._distance(my_pos, enemy_pos)

        target_angle = self.angle_to(my_pos, enemy_pos)
        hull_error = self.normalize_angle(target_angle - my_heading)
        barrel_abs = my_heading + my_barrel
        barrel_error = self.normalize_angle(target_angle - barrel_abs)

        return TargetInfo(
            enemy_id=enemy_id,
            distance=enemy_distance,
            hull_error=hull_error,
            barrel_error=barrel_error,
            position=enemy_pos,
        )

    def _ally_in_fire_line(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        target_distance: Optional[float] = None,
    ) -> bool:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)
        barrel_abs = my_heading + my_barrel

        for tank in sensor_data.get("seen_tanks", []):
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            ally_distance = self._distance(my_pos, ally_pos)
            if target_distance is not None and ally_distance >= max(0.0, target_distance - 0.5):
                continue
            angle = self.angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 3.0:
                return True
        return False

    def _compute_search_barrel_rotation(self, my_status: Dict[str, Any]) -> float:
        vision_angle = float(my_status.get("_vision_angle", 40.0) or 40.0)
        sweep_limit = max(10.0, min(48.0, vision_angle * 0.95))
        barrel_angle = float(my_status.get("barrel_angle", 0.0) or 0.0)

        if barrel_angle > sweep_limit:
            self.scan_direction = -1.0
            self.scan_burst_ticks_left = 0
        elif barrel_angle < -sweep_limit:
            self.scan_direction = 1.0
            self.scan_burst_ticks_left = 0

        if self.scan_pause_ticks_left > 0:
            self.scan_pause_ticks_left -= 1
            return 0.0

        if self.scan_burst_ticks_left <= 0:
            self.scan_burst_ticks_left = max(4, int(vision_angle / 6.0) + 2)
            self.scan_pause_ticks_left = max(6, int(22.0 - min(vision_angle, 18.0)))
            if abs(barrel_angle) > sweep_limit * 0.75:
                self.scan_direction *= -1.0

        self.scan_burst_ticks_left -= 1
        return self.scan_speed * self.scan_direction

    def _nearest_destructible_obstacle(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)
        barrel_abs = my_heading + my_barrel

        best_distance = float("inf")
        best_error = 0.0

        for obstacle in sensor_data.get("seen_obstacles", []):
            if not bool(obstacle.get("is_destructible", False)):
                continue
            obstacle_pos = obstacle.get("position", {"x": 0.0, "y": 0.0})
            distance = self._distance(my_pos, obstacle_pos)
            if distance >= best_distance:
                continue
            angle = self.angle_to(my_pos, obstacle_pos)
            error = self.normalize_angle(angle - barrel_abs)
            best_distance = distance
            best_error = error

        if best_distance == float("inf"):
            return None

        return best_distance, best_error

    def _barrel_error_to_position(self, my_status: Dict[str, Any], target_pos: Dict[str, float]) -> float:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)
        target_angle = self.angle_to(my_pos, target_pos)
        barrel_abs = my_heading + my_barrel
        return self.normalize_angle(target_angle - barrel_abs)

    def _danger_ahead(self, my_status: Dict[str, Any], sensor_data: Dict[str, Any]) -> bool:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_heading = my_status.get("heading", 0.0)

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

        return False

    def _combat_action(
        self,
        my_status: Dict[str, Any],
        target: TargetInfo,
        sensor_data: Dict[str, Any],
    ) -> ActionCommand:
        heading_rotation = 0.0
        barrel_rotation = 0.0
        move_speed = 0.0
        should_fire = False

        # Keep rotations small: engine applies rotation per tick.
        max_hull_step = 2.0
        max_barrel_step = 3.5

        # Align hull toward target if the error is large.
        if abs(target.hull_error) > 4.0:
            heading_rotation = max(-max_hull_step, min(max_hull_step, target.hull_error))

        # Always align barrel.
        barrel_rotation = max(-max_barrel_step, min(max_barrel_step, target.barrel_error))

        # Keep distance: close in when far, retreat when too close.
        if target.distance > 7.0:
            move_speed = 80.0
        elif target.distance < 3.5:
            move_speed = -70.0
        else:
            move_speed = 0.0

        if self._danger_ahead(my_status, sensor_data) and move_speed > 0:
            # Avoid deadlocks near obstacles/water: move slowly and turn away.
            move_speed = 40.0
            heading_rotation = 2.5 if random.random() > 0.5 else -2.5

        ammo_to_load = self._select_ammo(my_status, target)
        ammo_counts = self._ammo_counts(my_status)
        reload_timer = float(my_status.get("_reload_timer", 0.0) or 0.0)
        tank_type = str(my_status.get("_tank_type", "")).upper()
        aim_tolerance = 3.6
        if "HEAVY" in tank_type:
            aim_tolerance = 4.2
        elif "SNIPER" in tank_type:
            aim_tolerance = 2.4

        if (
            any(count > 0 for count in ammo_counts.values())
            and abs(target.barrel_error) <= aim_tolerance
            and not self._ally_in_fire_line(my_status, sensor_data, target_distance=target.distance)
            and reload_timer <= 0.0
        ):
            should_fire = True

        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=move_speed,
            ammo_to_load=ammo_to_load,
            should_fire=should_fire,
        )

    def _exploration_action(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        current_tick: int,
    ) -> ActionCommand:
        self.idle_ticks += 1

        if self.turn_ticks_left <= 0:
            self.turn_ticks_left = random.randint(20, 55)
            self.turn_direction = random.choice([-1.0, 1.0])

        self.turn_ticks_left -= 1

        my_heading = float(my_status.get("heading", 0.0) or 0.0)
        my_team = my_status.get("_team")

        # Push both teams toward each other during exploration.
        # Team 2 initially faces east too, so moving backward drives it west.
        desired_heading = 0.0
        heading_error = self.normalize_angle(desired_heading - my_heading)
        heading_rotation = max(-0.25, min(0.25, heading_error))
        move_speed = 100.0 if my_team == 1 else -100.0

        # Short, periodic turret pulses for enemy searching.
        barrel_rotation = self._compute_search_barrel_rotation(my_status)

        # After contact loss, briefly bias the turret toward last known enemy position.
        if (
            self.last_enemy_position is not None
            and (current_tick - self.last_enemy_seen_tick) <= 140
        ):
            last_error = self._barrel_error_to_position(my_status, self.last_enemy_position)
            if abs(last_error) <= 20.0:
                barrel_rotation = max(-5.0, min(5.0, last_error))

        if self._danger_ahead(my_status, sensor_data):
            move_speed = 35.0
            heading_rotation = 0.4 if self.turn_direction > 0 else -0.4

        ammo_to_load = self._select_ammo(my_status, target=None)
        should_fire = False

        # Clear destructible obstacles that block scanning/engagement.
        obstacle_target = self._nearest_destructible_obstacle(my_status, sensor_data)
        reload_timer = float(my_status.get("_reload_timer", 0.0) or 0.0)
        if obstacle_target is not None:
            obstacle_distance, obstacle_barrel_error = obstacle_target
            if abs(obstacle_barrel_error) <= 14.0:
                barrel_rotation = max(-5.0, min(5.0, obstacle_barrel_error))
            if (
                reload_timer <= 0.0
                and abs(obstacle_barrel_error) <= 2.8
                and not self._ally_in_fire_line(my_status, sensor_data, target_distance=obstacle_distance)
            ):
                should_fire = True

        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=move_speed,
            ammo_to_load=ammo_to_load,
            should_fire=should_fire,
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

        target = self._find_nearest_enemy(my_tank_status, sensor_data)
        if target is not None:
            self.locked_enemy_id = target.enemy_id or self.locked_enemy_id
            self.last_enemy_position = {
                "x": float(target.position.get("x", 0.0)),
                "y": float(target.position.get("y", 0.0)),
            }
            self.last_enemy_seen_tick = int(current_tick)
            return self._combat_action(my_tank_status, target, sensor_data)

        if self.locked_enemy_id and (current_tick - self.last_enemy_seen_tick) > 220:
            self.locked_enemy_id = None
            self.last_enemy_position = None

        return self._exploration_action(my_tank_status, sensor_data, current_tick)

    def destroy(self) -> None:
        self.is_destroyed = True
        print(f"[{self.name}] destroyed")

    def end(self, damage_dealt: float, tanks_killed: int) -> None:
        print(
            f"[{self.name}] end | damage={damage_dealt:.1f} kills={tanks_killed}"
        )
        # Allow persistent trainer mode: revive local episode state for next match.
        self.is_destroyed = False
        self.locked_enemy_id = None
        self.last_enemy_position = None
        self.last_enemy_seen_tick = -10_000
        self.scan_burst_ticks_left = 0
        self.scan_pause_ticks_left = 0


app = FastAPI(
    title="Rule-Based Agent",
    description="Readable baseline agent for tank battles",
    version="2.0.0",
)
agent = RuleBasedCombatAgent()


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)) -> ActionCommand:
    return agent.get_action(
        current_tick=payload.get("current_tick", 0),
        my_tank_status=payload.get("my_tank_status", {}),
        sensor_data=payload.get("sensor_data", {}),
        enemies_remaining=payload.get("enemies_remaining", 0),
    )


@app.post("/agent/destroy", status_code=204, response_model=None)
async def destroy(payload: Dict[str, Any] = Body(None)) -> None:
    agent.destroy()


@app.post("/agent/end", status_code=204, response_model=None)
async def end(payload: Dict[str, Any] = Body(...)) -> None:
    agent.end(
        damage_dealt=float(payload.get("damage_dealt", 0.0) or 0.0),
        tanks_killed=int(payload.get("tanks_killed", 0) or 0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rule-based tank agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if args.name:
        agent.name = args.name
    else:
        agent.name = f"RuleBot_{args.port}"

    print(f"Starting {agent.name} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
