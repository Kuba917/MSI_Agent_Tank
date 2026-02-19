"""
One-click launcher for training.

Use case:
- edit CONFIG values below in this file,
- press Play/Run in your IDE,
- script runs training only.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


THIS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = THIS_DIR.parent / "02_FRAKCJA_SILNIKA"


@dataclass
class LauncherConfig:
    # -------------------------
    # Training config
    # -------------------------
    episodes: int = 80
    team_size: int = 5
    learning_agents: int = 1
    shared_team1_policy: bool = True
    # Keep one learner slot by default so reward history is continuous.
    rotate_learner_slot: bool = False
    learner_slot: int = 0
    model_path: Path = THIS_DIR / "fuzzy_dqn_model.pt"
    map_curriculum: str = "open.csv*20,semi-open.csv*20,road_trees.csv*40"
    max_ticks: int = 2600
    log_level: str = "INFO"
    base_port: int = 8001

    warmup_steps: int = 2000
    batch_size: int = 128
    train_every: int = 1
    target_sync_every: int = 50
    restart_agents_every: int = 5
    ready_timeout: float = 120.0
    save_every_games: int = 1
    seed: int = 1

    selfplay_start_episode: int = 0
    selfplay_opponents: int = 0
    selfplay_model_path: str = ""

    verbose: bool = True
    continue_on_error: bool = False
    render_training: bool = False


CONFIG = LauncherConfig()


def _print_command(prefix: str, command: Sequence[str]) -> None:
    print(f"{prefix}: {' '.join(command)}")


def _run(command: Sequence[str], cwd: Path) -> int:
    _print_command("RUN", command)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def build_training_command(cfg: LauncherConfig) -> List[str]:
    effective_learning_agents = 1 if cfg.shared_team1_policy else int(cfg.learning_agents)
    cmd = [
        sys.executable,
        "train_fuzzy_dqn.py",
        "--episodes",
        str(int(cfg.episodes)),
        "--team-size",
        str(int(cfg.team_size)),
        "--learning-agents",
        str(int(effective_learning_agents)),
        "--base-port",
        str(int(cfg.base_port)),
        "--model-path",
        str(Path(cfg.model_path)),
        "--map-curriculum",
        str(cfg.map_curriculum),
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
        "--log-level",
        str(cfg.log_level),
        "--save-every-games",
        str(int(cfg.save_every_games)),
        "--seed",
        str(int(cfg.seed)),
    ]

    if cfg.selfplay_model_path.strip():
        cmd.extend(["--selfplay-model-path", cfg.selfplay_model_path.strip()])
    if cfg.shared_team1_policy:
        cmd.append("--shared-team1-policy")
        if cfg.rotate_learner_slot:
            cmd.append("--rotate-learner-slot")
        else:
            cmd.extend(["--learner-slot", str(max(0, int(cfg.learner_slot)))])
    if cfg.verbose:
        cmd.append("--verbose")
    if cfg.continue_on_error:
        cmd.append("--continue-on-error")
    if cfg.render_training:
        cmd.append("--render")

    return cmd


def run_training(cfg: LauncherConfig) -> int:
    print("=== TRAINING START ===")
    train_cmd = build_training_command(cfg)
    rc = _run(train_cmd, THIS_DIR)
    print(f"=== TRAINING END (rc={rc}) ===")
    return rc


def validate_config(cfg: LauncherConfig) -> None:
    if cfg.team_size != 5:
        print(
            "[WARN] Engine currently spawns 5v5 in game_loop.py. "
            "Using team_size != 5 may cause port/roster mismatch."
        )
    if cfg.shared_team1_policy and cfg.learning_agents != 1:
        print(
            "[WARN] shared_team1_policy=True forces one learner slot; "
            "learning_agents value will be ignored."
        )
    if cfg.shared_team1_policy and cfg.rotate_learner_slot:
        print(
            "[WARN] rotate_learner_slot=True restarts Team-1 agents each episode; "
            "prefer False for more stable reward history."
        )
    if (not cfg.shared_team1_policy) and cfg.learning_agents != 1:
        print(
            "[WARN] learning_agents != 1 means separate learners/models, "
            "not one shared policy."
        )


def main() -> int:
    cfg = CONFIG
    validate_config(cfg)

    if not ENGINE_DIR.exists():
        print(f"Engine directory not found: {ENGINE_DIR}")
        return 1

    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
