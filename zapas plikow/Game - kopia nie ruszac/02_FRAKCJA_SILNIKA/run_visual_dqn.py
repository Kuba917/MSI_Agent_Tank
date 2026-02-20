"""
Skrypt do wizualnej oceny wyuczonego agenta DQN.
Uruchamia grę w trybie graficznym:
- TEAM 1 (Niebiescy): Twój agent DQN (ładuje wagi z pliku .pt)
- TEAM 2 (Czerwoni): Agent regułowy (Baseline)

Użycie:
  py -3 Game/02_FRAKCJA_SILNIKA/run_visual_dqn.py
"""
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

import subprocess
import sys
import os
import time
import random
import math
from typing import Dict, Any, List
from pygame.math import Vector2
import pygame

# --- Konfiguracja Ścieżek ---
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(current_file_dir)
    agents_dir = os.path.join(main_dir, "03_FRAKCJA_AGENTOW")

    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    from backend.engine.game_loop import GameLoop, TEAM_A_NBR, TEAM_B_NBR, AGENT_BASE_PORT
    from backend.utils.logger import set_log_level
    from backend.tank.base_tank import Tank
    from controller.api import AmmoType

except ImportError as e:
    print(f"Błąd importu: {e}")
    sys.exit(1)

# --- Konfiguracja ---
LOG_LEVEL = "INFO"
MAP_SEED = "advanced_road_trees.csv" # Mapa do testów
TARGET_FPS = 60
SCALE = 5
TILE_SIZE = 10

# Nazwy plików w folderze 03_FRAKCJA_AGENTOW
DQN_SCRIPT = "DQN.py"
BASELINE_SCRIPT = "Agent_1.py"
MODEL_FILE = "fuzzy_dqn_model.pt" # Lub "fuzzy_dqn_model_best.pt"

# --- Zasoby Graficzne (Skopiowane z engine_v1_beta.py) ---
ASSETS_BASE_PATH = os.path.join(current_file_dir, 'frontend', 'assets')
TILE_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'tiles')
POWERUP_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'power-ups')
TANK_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'tanks')
ICONS_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'icons')

BACKGROUND_COLOR = (20, 20, 30)
TEAM_COLORS = {1: (50, 150, 255), 2: (255, 50, 50)}
TANK_ASSET_MAP = {"LIGHT": "light_tank", "HEAVY": "heavy_tank", "Sniper": "sniper_tank"}
POWERUP_ASSET_MAP = {
    "MEDKIT": "Medkit", "SHIELD": "Shield", "OVERCHARGE": "Overcharge",
    "AMMO_HEAVY": "AmmoBox_Heavy", "AMMO_LIGHT": "AmmoBox_Light", "AMMO_LONG_DISTANCE": "AmmoBox_Sniper"
}

class ExplosionParticle:
    def __init__(self, pos, velocity, start_size, lifetime):
        self.pos = list(pos)
        self.velocity = list(velocity)
        self.size = start_size
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = random.choice([(255, 0, 0), (255, 100, 0), (255, 215, 0), (139, 0, 0)])

    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.lifetime -= 1
        self.velocity[0] += random.uniform(-0.05, 0.05)
        self.velocity[1] += random.uniform(-0.05, 0.05)

    def draw(self, surface):
        if self.lifetime > 0:
            lerp = self.lifetime / self.max_lifetime
            sz = int(self.size * lerp)
            if sz > 0:
                col = (int(self.color[0]*lerp), int(self.color[1]*lerp), int(self.color[2]*lerp))
                pygame.draw.circle(surface, col, self.pos, sz)

def load_assets():
    assets = {'tiles': {}, 'powerups': {}, 'tanks': {}, 'icons': {}}
    # Kafelki
    for name in ['Wall', 'Tree', 'AntiTankSpike', 'Grass', 'Road', 'Swamp', 'PotholeRoad', 'Water']:
        try:
            img = pygame.image.load(os.path.join(TILE_ASSETS_PATH, f"{name}.png")).convert_alpha()
            assets['tiles'][name] = pygame.transform.scale(img, (TILE_SIZE * SCALE, TILE_SIZE * SCALE))
        except: pass
    # Powerupy
    for name in ['Medkit', 'Shield', 'Overcharge', 'AmmoBox_Heavy', 'AmmoBox_Light', 'AmmoBox_Sniper']:
        try:
            img = pygame.image.load(os.path.join(POWERUP_ASSETS_PATH, f"{name}.png")).convert_alpha()
            assets['powerups'][name] = pygame.transform.scale(img, (int(TILE_SIZE*SCALE*0.8), int(TILE_SIZE*SCALE*0.8)))
        except: pass
    # Czołgi
    ts = (TILE_SIZE * SCALE, TILE_SIZE * SCALE)
    for ttype, folder in TANK_ASSET_MAP.items():
        try:
            bp = os.path.join(TANK_ASSETS_PATH, folder)
            assets['tanks'][ttype] = {
                'body': pygame.transform.scale(pygame.image.load(os.path.join(bp, 'tnk1.png')).convert_alpha(), ts),
                'mask_body': pygame.transform.scale(pygame.image.load(os.path.join(bp, 'msk1.png')).convert_alpha(), ts),
                'turret': pygame.transform.scale(pygame.image.load(os.path.join(bp, 'tnk2.png')).convert_alpha(), ts),
                'mask_turret': pygame.transform.scale(pygame.image.load(os.path.join(bp, 'msk2.png')).convert_alpha(), ts),
            }
        except: pass
    # Ikony
    for ttype, folder in TANK_ASSET_MAP.items():
        try:
            img = pygame.image.load(os.path.join(ICONS_ASSETS_PATH, f"{folder}.png")).convert_alpha()
            assets['icons'][ttype] = pygame.transform.scale(img, (128, 64))
        except:
            assets['icons'][ttype] = pygame.Surface((128, 64), pygame.SRCALPHA)
    return assets

def create_background_surface(map_info, assets, scale, w, h):
    bg = pygame.Surface((w, h))
    bg.fill(BACKGROUND_COLOR)
    for obj in map_info.terrain_list + map_info.obstacle_list:
        asset = assets['tiles'].get(obj.__class__.__name__)
        if asset:
            px = obj._position.x * scale
            py = h - (obj._position.y * scale)
            bg.blit(asset, (px - asset.get_width()/2, py - asset.get_height()/2))
    return bg

def draw_tank(surface, tank, assets, scale, map_h):
    t_assets = assets['tanks'].get(tank._tank_type)
    if not t_assets: return
    alive = tank.is_alive()
    color = TEAM_COLORS.get(tank.team, (255, 255, 255))
    center = (tank.position.x * scale, map_h - (tank.position.y * scale))

    # Body
    body = t_assets['body'].copy()
    if not alive: body.set_alpha(100)
    rot_body = pygame.transform.rotate(body, -tank.heading - 180)
    surface.blit(rot_body, rot_body.get_rect(center=center).topleft)

    # Mask Body
    mask = t_assets['mask_body'].copy()
    if not alive: mask.set_alpha(100)
    col_layer = pygame.Surface(mask.get_size())
    col_layer.fill(color)
    col_layer.blit(mask, (0,0), special_flags=pygame.BLEND_RGB_MULT)
    col_layer.set_colorkey((0,0,0))
    surface.blit(pygame.transform.rotate(col_layer, -tank.heading - 180), rot_body.get_rect(center=center).topleft)

    if alive:
        # Turret
        turret = t_assets['turret']
        angle = tank.heading - tank.barrel_angle
        rot_tur = pygame.transform.rotate(turret, -angle - 180)
        surface.blit(rot_tur, rot_tur.get_rect(center=center).topleft)
        
        # Mask Turret
        mask_t = t_assets['mask_turret']
        col_t = pygame.Surface(mask_t.get_size())
        col_t.fill(color)
        col_t.blit(mask_t, (0,0), special_flags=pygame.BLEND_RGB_MULT)
        col_t.set_colorkey((0,0,0))
        surface.blit(pygame.transform.rotate(col_t, -angle - 180), rot_tur.get_rect(center=center).topleft)

        # HP Bar
        hp_w, hp_h = 40, 5
        hp_r = max(0, tank.hp / tank._max_hp)
        bx, by = center[0] - hp_w/2, center[1] - body.get_height()/2 - 15
        pygame.draw.rect(surface, (50,50,50), (bx, by, hp_w, hp_h))
        pygame.draw.rect(surface, (0,255,0), (bx, by, hp_w * hp_r, hp_h))

def draw_ui(screen, font, game_loop, win_w, map_rect, assets):
    detail_font = pygame.font.Font(None, 22)
    
    # Team 1 (Left)
    p1_x = map_rect.left / 2
    cy = 50
    t1_alive = sum(1 for t in game_loop.tanks.values() if t.team == 1 and t.is_alive())
    screen.blit(font.render("TEAM 1 (DQN)", True, TEAM_COLORS[1]), (p1_x - 60, cy))
    cy += 40
    screen.blit(font.render(f"Alive: {t1_alive}", True, (200,200,200)), (p1_x - 40, cy))
    cy += 40
    
    for t in sorted([t for t in game_loop.tanks.values() if t.team == 1], key=lambda x: x._id):
        if not t.is_alive(): continue
        icon = assets['icons'].get(t._tank_type)
        if icon: screen.blit(icon, (p1_x - 64, cy))
        cy += 50
        hp_txt = f"{int(t.hp)}/{int(t._max_hp)}"
        screen.blit(detail_font.render(hp_txt, True, (255,255,255)), (p1_x - 20, cy))
        cy += 20
        # Ammo
        loaded = t.ammo_loaded.name if t.ammo_loaded else "NONE"
        screen.blit(detail_font.render(f"L: {loaded}", True, (200,200,0)), (p1_x - 40, cy))
        cy += 30

    # Team 2 (Right)
    p2_x = map_rect.right + (win_w - map_rect.right) / 2
    cy = 50
    t2_alive = sum(1 for t in game_loop.tanks.values() if t.team == 2 and t.is_alive())
    screen.blit(font.render("TEAM 2 (Base)", True, TEAM_COLORS[2]), (p2_x - 60, cy))
    cy += 40
    screen.blit(font.render(f"Alive: {t2_alive}", True, (200,200,200)), (p2_x - 40, cy))
    cy += 40

    for t in sorted([t for t in game_loop.tanks.values() if t.team == 2], key=lambda x: x._id):
        if not t.is_alive(): continue
        icon = assets['icons'].get(t._tank_type)
        if icon: screen.blit(icon, (p2_x - 64, cy))
        cy += 50
        hp_txt = f"{int(t.hp)}/{int(t._max_hp)}"
        screen.blit(detail_font.render(hp_txt, True, (255,255,255)), (p2_x - 20, cy))
        cy += 20
        loaded = t.ammo_loaded.name if t.ammo_loaded else "NONE"
        screen.blit(detail_font.render(f"L: {loaded}", True, (200,200,0)), (p2_x - 40, cy))
        cy += 30

def main():
    set_log_level(LOG_LEVEL)
    
    # Sprawdzenie ścieżek
    dqn_path = os.path.join(agents_dir, DQN_SCRIPT)
    baseline_path = os.path.join(agents_dir, BASELINE_SCRIPT)
    model_path = os.path.join(agents_dir, MODEL_FILE)

    if not os.path.exists(dqn_path):
        print(f"BŁĄD: Nie znaleziono skryptu DQN: {dqn_path}")
        return
    if not os.path.exists(model_path):
        print(f"BŁĄD: Nie znaleziono modelu: {model_path}")
        print("Upewnij się, że wytrenowałeś model i plik .pt istnieje.")
        return

    agent_processes = []
    total_tanks = TEAM_A_NBR + TEAM_B_NBR

    print(f"--- Uruchamianie Wizualizacji DQN vs Baseline ---")
    print(f"Model: {MODEL_FILE}")
    print(f"Mapa: {MAP_SEED}")

    try:
        # 1. Uruchamianie Agentów
        for i in range(total_tanks):
            port = AGENT_BASE_PORT + i
            
            if i < TEAM_A_NBR:
                # TEAM 1: DQN Agent
                name = f"DQN_Agent_{i+1}"
                cmd = [
                    sys.executable, dqn_path,
                    "--port", str(port),
                    "--name", name,
                    "--model-path", model_path,
                    # Ważne: brak flagi --train oznacza tryb ewaluacji (inference)
                ]
                print(f"Startuje {name} (DQN) na porcie {port}")
            else:
                # TEAM 2: Baseline Agent
                name = f"Baseline_{i+1}"
                cmd = [
                    sys.executable, baseline_path,
                    "--port", str(port),
                    "--name", name
                ]
                print(f"Startuje {name} (Baseline) na porcie {port}")

            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=agents_dir)
            agent_processes.append(proc)

        print("Oczekiwanie na start serwerów...")
        time.sleep(4)

        # 2. Inicjalizacja Gry
        game_loop = GameLoop(headless=False)
        if not game_loop.initialize_game(map_seed=MAP_SEED):
            raise RuntimeError("Błąd inicjalizacji gry")

        # 3. Pygame Setup
        pygame.init()
        map_w, map_h = game_loop.map_info._size
        render_w, render_h = map_w * SCALE, map_h * SCALE
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        win_w, win_h = screen.get_size()
        pygame.display.set_caption("DQN Visual Evaluation")
        
        assets = load_assets()
        font = pygame.font.Font(None, 36)
        clock = pygame.time.Clock()
        
        map_surf = pygame.Surface((render_w, render_h))
        map_rect = map_surf.get_rect(center=(win_w/2, win_h/2))
        bg_surf = create_background_surface(game_loop.map_info, assets, SCALE, render_w, render_h)

        if not game_loop.game_core.start_game_loop():
            raise RuntimeError("Błąd startu pętli core")

        shot_effects = []
        particles = []

        running = True
        print("--- Start Symulacji ---")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            if not game_loop.game_core.can_continue_game():
                running = False
                continue

            # Logic Tick
            tick_res = game_loop._process_game_tick()
            phys_res = game_loop.last_physics_results

            # Visual Effects
            for hit in phys_res.get("projectile_hits", []):
                if hit.hit_position:
                    spos = Vector2(hit.hit_position.x * SCALE, render_h - (hit.hit_position.y * SCALE))
                    for _ in range(20):
                        angle = random.uniform(0, 360)
                        vel = Vector2(1, 0).rotate(angle) * random.uniform(0.5, 2.0)
                        particles.append(ExplosionParticle(spos, vel, random.randint(3,5), 30))
                    
                    shooter = game_loop.tanks.get(hit.shooter_id)
                    if shooter:
                        shot_effects.append({
                            's': shooter.position, 
                            'e': hit.hit_position, 
                            'l': 10
                        })

            # Rendering
            screen.fill(BACKGROUND_COLOR)
            map_surf.blit(bg_surf, (0,0))

            # Powerups
            for p in game_loop.map_info.powerup_list:
                asset = assets['powerups'].get(POWERUP_ASSET_MAP.get(p._powerup_type.name))
                if asset:
                    px = p.position.x * SCALE
                    py = render_h - (p.position.y * SCALE)
                    map_surf.blit(asset, (px - asset.get_width()/2, py - asset.get_height()/2))

            # Tanks
            for t in game_loop.tanks.values():
                draw_tank(map_surf, t, assets, SCALE, render_h)

            # Shots
            rem_shots = []
            for s in shot_effects:
                if s['l'] > 0:
                    start = (s['s'].x * SCALE, render_h - (s['s'].y * SCALE))
                    end = (s['e'].x * SCALE, render_h - (s['e'].y * SCALE))
                    col = (255, 255, 0, int(255 * (s['l']/10)))
                    ls = pygame.Surface(map_surf.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(ls, col, start, end, 2)
                    map_surf.blit(ls, (0,0))
                    s['l'] -= 1
                    rem_shots.append(s)
            shot_effects = rem_shots

            # Particles
            rem_parts = []
            for p in particles:
                p.update()
                if p.lifetime > 0:
                    p.draw(map_surf)
                    rem_parts.append(p)
            particles = rem_parts

            screen.blit(map_surf, map_rect)
            draw_ui(screen, font, game_loop, win_w, map_rect, assets)
            
            pygame.display.flip()
            clock.tick(TARGET_FPS)

        # End Screen
        res = game_loop.game_core.end_game("normal")
        winner = res.get("winner_team")
        print(f"Koniec gry. Wygrał Team: {winner}")
        time.sleep(2)

    except Exception as e:
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()
    finally:
        game_loop.cleanup_game()
        for p in agent_processes: p.terminate()
        pygame.quit()

if __name__ == "__main__":
    main()
