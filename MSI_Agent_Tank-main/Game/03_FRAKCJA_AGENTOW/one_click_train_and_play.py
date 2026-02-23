"""
One-click launcher for fuzzy DQN training in this project.

Use case:
- edit CONFIG values below,
- run this file from IDE,
- script prepares shooting mock models (if needed) and starts training.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


THIS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = THIS_DIR.parent / "02_FRAKCJA_SILNIKA"
TRAIN_SCRIPT = THIS_DIR / "train_fuzzy_dqn.py"
MOCK_TRAIN_SCRIPT = THIS_DIR / "train_mock_shooting.py"
MOCK_BARREL_PATH = THIS_DIR / "anfis_barrel_model.pt"
MOCK_SHOOT_PATH = THIS_DIR / "anfis_shoot_model.pt"


@dataclass
class LauncherConfig:
    # Core run
    episodes: int = 20
    team_size: int = 5
    learning_agents: int = 1
    model_path: Path = THIS_DIR / "fuzzy_dqn_model.pt"

    # Map setup: use either single map_seed or map_curriculum
    map_seed: str = "road_trees.csv"
    map_curriculum: str = ""

    # Engine/training timing
    max_ticks: int = 3500
    base_port: int = 8001
    log_level: str = "INFO"
    restart_agents_every: int = 0
    ready_timeout: float = 90.0
    start_delay: float = 4.0
    episode_delay: float = 0.5

    # Learner hyperparameters
    warmup_steps: int = 1024
    batch_size: int = 128
    train_every: int = 2
    target_sync_every: int = 500
    save_every_games: int = 1
    seed: int = 1

    gamma: float = 0.95
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    tau: float = 0.01
    action_noise_start: float = 0.12
    action_noise_end: float = 0.06
    action_noise_decay_steps: int = 250000
    movement_bootstrap_episodes: int = 25
    movement_bootstrap_alpha_start: float = 0.95
    movement_bootstrap_alpha_end: float = 0.15
    movement_bootstrap_visible_scale: float = 0.55
    movement_bootstrap_hidden_scale: float = 1.0
    movement_waypoint_min_distance: float = 18.0
    movement_waypoint_max_distance: float = 55.0
    movement_waypoint_lateral_max: float = 55.0
    movement_waypoint_replan_ticks: int = 45
    movement_waypoint_reach_radius: float = 8.0
    movement_waypoint_stuck_window: int = 40
    movement_waypoint_stuck_distance: float = 5.0
    movement_enemy_zone_patrol_activate_distance: float = 34.0
    movement_enemy_zone_patrol_exit_distance: float = 56.0
    movement_enemy_zone_patrol_radius_min: float = 10.0
    movement_enemy_zone_patrol_radius_max: float = 24.0
    movement_enemy_zone_patrol_replan_ticks: int = 20
    movement_enemy_zone_patrol_step_deg: float = 72.0
    target_lock_lost_patience_ticks: int = 120
    target_lock_relaxed_shoot_margin: float = 0.26
    progress_reward_scale: float = 1.2
    exploration_reward_scale: float = 0.09
    retreat_hp_threshold: float = 0.35
    mock_shoot_threshold: float = 0.47
    mock_half_angle_deg: float = 7.0

    # Self-play (optional)
    selfplay_start_episode: int = 0
    selfplay_opponents: int = 0
    selfplay_model_path: str = ""

    # Runtime flags
    verbose: bool = True
    continue_on_error: bool = False
    render_training: bool = False
    no_load: bool = False

    # Mock shooting models (required for barrel+fire control)
    ensure_mock_models: bool = True
    force_retrain_mock_models: bool = True #jednorazowo True pozniej False zeby nie trenowac mocka za kazdym razem
    mock_samples: int = 20000
    mock_batch_size: int = 128
    mock_epochs: int = 200


CONFIG = LauncherConfig()


def _print_command(prefix: str, command: Sequence[str]) -> None:
    print(f"{prefix}: {subprocess.list2cmdline(list(command))}")


def _run(command: Sequence[str], cwd: Path) -> int:
    _print_command("RUN", command)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def validate_config(cfg: LauncherConfig) -> None:
    if cfg.team_size != 5:
        _warn("Engine is currently used as 5v5; team_size != 5 may cause roster/port mismatch.")

    if int(cfg.learning_agents) <= 0:
        raise ValueError("learning_agents must be > 0")

    if int(cfg.learning_agents) > 1:
        _warn(
            "learning_agents > 1 starts independent learners (separate checkpoints), "
            "which is heavier and often less stable than one learner in this codebase."
        )

    if cfg.map_curriculum.strip() and cfg.map_seed.strip():
        _warn("map_curriculum is set, so map_seed will be ignored by train_fuzzy_dqn.py")

    if not cfg.map_curriculum.strip() and not cfg.map_seed.strip():
        raise ValueError("Set map_seed or map_curriculum")

    if cfg.selfplay_opponents > 0 and cfg.selfplay_start_episode <= 0:
        _warn("selfplay_opponents > 0 but selfplay_start_episode <= 0, so self-play is disabled")



def _mock_models_exist() -> bool:
    return MOCK_BARREL_PATH.exists() and MOCK_SHOOT_PATH.exists()


def build_mock_training_command(cfg: LauncherConfig) -> List[str]:
    inline = (
        "from train_mock_shooting import train_models; "
        f"train_models(n_samples={int(cfg.mock_samples)}, "
        f"batch_size={int(cfg.mock_batch_size)}, "
        f"epochs={int(cfg.mock_epochs)}, "
        f"seed={int(cfg.seed)})"
    )
    return [sys.executable, "-c", inline]


def maybe_prepare_mock_models(cfg: LauncherConfig) -> int:
    if not cfg.ensure_mock_models:
        if not _mock_models_exist():
            _warn(
                "Mock shooting models are missing. DQN agent will fail to start without "
                f"{MOCK_BARREL_PATH.name} and {MOCK_SHOOT_PATH.name}."
            )
        return 0

    need_train = bool(cfg.force_retrain_mock_models or not _mock_models_exist())
    if not need_train:
        print(
            "Mock shooting models found: "
            f"{MOCK_BARREL_PATH.name}, {MOCK_SHOOT_PATH.name}"
        )
        return 0

    print("=== PREPARE SHOOTING MODELS START ===")
    rc = _run(build_mock_training_command(cfg), THIS_DIR)
    print(f"=== PREPARE SHOOTING MODELS END (rc={rc}) ===")
    if rc != 0:
        return rc

    if not _mock_models_exist():
        print(
            "Mock model training finished but expected files are missing: "
            f"{MOCK_BARREL_PATH.name}, {MOCK_SHOOT_PATH.name}"
        )
        return 1
    return 0


def build_training_command(cfg: LauncherConfig) -> List[str]:
    cmd = [
        sys.executable,
        TRAIN_SCRIPT.name,
        "--episodes",
        str(int(cfg.episodes)),
        "--team-size",
        str(int(cfg.team_size)),
        "--learning-agents",
        str(int(cfg.learning_agents)),
        "--base-port",
        str(int(cfg.base_port)),
        "--model-path",
        str(Path(cfg.model_path)),
        "--max-ticks",
        str(int(cfg.max_ticks)),
        "--selfplay-start-episode",
        str(int(cfg.selfplay_start_episode)),
        "--selfplay-opponents",
        str(int(cfg.selfplay_opponents)),
        "--warmup-steps",
        str(int(cfg.warmup_steps)),
        "--batch-size",
        str(int(cfg.batch_size)),
        "--train-every",
        str(int(cfg.train_every)),
        "--target-sync-every",
        str(int(cfg.target_sync_every)),
        "--restart-agents-every",
        str(int(cfg.restart_agents_every)),
        "--ready-timeout",
        str(float(cfg.ready_timeout)),
        "--start-delay",
        str(float(cfg.start_delay)),
        "--episode-delay",
        str(float(cfg.episode_delay)),
        "--log-level",
        str(cfg.log_level),
        "--save-every-games",
        str(int(cfg.save_every_games)),
        "--seed",
        str(int(cfg.seed)),
        "--gamma",
        str(float(cfg.gamma)),
        "--actor-lr",
        str(float(cfg.actor_lr)),
        "--critic-lr",
        str(float(cfg.critic_lr)),
        "--tau",
        str(float(cfg.tau)),
        "--action-noise-start",
        str(float(cfg.action_noise_start)),
        "--action-noise-end",
        str(float(cfg.action_noise_end)),
        "--action-noise-decay-steps",
        str(int(cfg.action_noise_decay_steps)),
        "--movement-bootstrap-episodes",
        str(int(cfg.movement_bootstrap_episodes)),
        "--movement-bootstrap-alpha-start",
        str(float(cfg.movement_bootstrap_alpha_start)),
        "--movement-bootstrap-alpha-end",
        str(float(cfg.movement_bootstrap_alpha_end)),
        "--movement-bootstrap-visible-scale",
        str(float(cfg.movement_bootstrap_visible_scale)),
        "--movement-bootstrap-hidden-scale",
        str(float(cfg.movement_bootstrap_hidden_scale)),
        "--movement-waypoint-min-distance",
        str(float(cfg.movement_waypoint_min_distance)),
        "--movement-waypoint-max-distance",
        str(float(cfg.movement_waypoint_max_distance)),
        "--movement-waypoint-lateral-max",
        str(float(cfg.movement_waypoint_lateral_max)),
        "--movement-waypoint-replan-ticks",
        str(int(cfg.movement_waypoint_replan_ticks)),
        "--movement-waypoint-reach-radius",
        str(float(cfg.movement_waypoint_reach_radius)),
        "--movement-waypoint-stuck-window",
        str(int(cfg.movement_waypoint_stuck_window)),
        "--movement-waypoint-stuck-distance",
        str(float(cfg.movement_waypoint_stuck_distance)),
        "--movement-enemy-zone-patrol-activate-distance",
        str(float(cfg.movement_enemy_zone_patrol_activate_distance)),
        "--movement-enemy-zone-patrol-exit-distance",
        str(float(cfg.movement_enemy_zone_patrol_exit_distance)),
        "--movement-enemy-zone-patrol-radius-min",
        str(float(cfg.movement_enemy_zone_patrol_radius_min)),
        "--movement-enemy-zone-patrol-radius-max",
        str(float(cfg.movement_enemy_zone_patrol_radius_max)),
        "--movement-enemy-zone-patrol-replan-ticks",
        str(int(cfg.movement_enemy_zone_patrol_replan_ticks)),
        "--movement-enemy-zone-patrol-step-deg",
        str(float(cfg.movement_enemy_zone_patrol_step_deg)),
        "--target-lock-lost-patience-ticks",
        str(int(cfg.target_lock_lost_patience_ticks)),
        "--target-lock-relaxed-shoot-margin",
        str(float(cfg.target_lock_relaxed_shoot_margin)),
        "--progress-reward-scale",
        str(float(cfg.progress_reward_scale)),
        "--exploration-reward-scale",
        str(float(cfg.exploration_reward_scale)),
        "--retreat-hp-threshold",
        str(float(cfg.retreat_hp_threshold)),
        "--mock-shoot-threshold",
        str(float(cfg.mock_shoot_threshold)),
        "--mock-half-angle-deg",
        str(float(cfg.mock_half_angle_deg)),
    ]

    if cfg.map_curriculum.strip():
        cmd.extend(["--map-curriculum", cfg.map_curriculum.strip()])
    else:
        cmd.extend(["--map-seed", cfg.map_seed.strip()])

    if cfg.selfplay_model_path.strip():
        cmd.extend(["--selfplay-model-path", cfg.selfplay_model_path.strip()])

    if cfg.verbose:
        cmd.append("--verbose")
    if cfg.continue_on_error:
        cmd.append("--continue-on-error")
    if cfg.render_training:
        cmd.append("--render")
    if cfg.no_load:
        cmd.append("--no-load")

    return cmd


def run_training(cfg: LauncherConfig) -> int:
    print("=== TRAINING START ===")
    train_cmd = build_training_command(cfg)
    rc = _run(train_cmd, THIS_DIR)
    print(f"=== TRAINING END (rc={rc}) ===")
    return rc


def main() -> int:
    cfg = CONFIG
    validate_config(cfg)

    if not ENGINE_DIR.exists():
        print(f"Engine directory not found: {ENGINE_DIR}")
        return 1
    if not TRAIN_SCRIPT.exists():
        print(f"Training script not found: {TRAIN_SCRIPT}")
        return 1
    if not MOCK_TRAIN_SCRIPT.exists():
        print(f"Mock training script not found: {MOCK_TRAIN_SCRIPT}")
        return 1

    prep_rc = maybe_prepare_mock_models(cfg)
    if prep_rc != 0:
        return prep_rc

    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
