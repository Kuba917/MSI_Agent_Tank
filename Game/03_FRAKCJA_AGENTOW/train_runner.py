"""
Batch training/evaluation runner for agent development.

Runs many headless games, starts agent servers for both teams,
computes fitness, and writes per-match CSV + summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


THIS_FILE = Path(__file__).resolve()
AGENTS_DIR = THIS_FILE.parent
ROOT_DIR = AGENTS_DIR.parent
ENGINE_DIR = ROOT_DIR / "02_FRAKCJA_SILNIKA"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from backend.engine import game_loop as game_loop_module
from backend.engine.game_loop import run_game
from backend.utils.logger import set_log_level
from backend.utils.config import game_config


@dataclass
class FitnessWeights:
    win: float = 200.0
    loss: float = -200.0
    draw: float = 0.0
    kill: float = 30.0
    damage: float = 0.25
    alive: float = 10.0
    enemy_alive: float = -10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch headless games and compute training fitness."
    )

    default_agent = str(AGENTS_DIR / "random_agent.py")

    parser.add_argument("--episodes", type=int, default=20, help="Number of matches.")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["advanced_road_trees.csv"],
        help="Map files from 02_FRAKCJA_SILNIKA/backend/maps.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base RNG seed. Match seed is base_seed + episode_idx.",
    )

    parser.add_argument(
        "--agent-team1",
        type=str,
        default=default_agent,
        help="Path to Team 1 agent server script.",
    )
    parser.add_argument(
        "--agent-team2",
        type=str,
        default=default_agent,
        help="Path to Team 2 agent server script.",
    )

    parser.add_argument(
        "--agent1-arg",
        action="append",
        default=[],
        help="Extra arg for Team 1 agent (repeatable).",
    )
    parser.add_argument(
        "--agent2-arg",
        action="append",
        default=[],
        help="Extra arg for Team 2 agent (repeatable).",
    )

    parser.add_argument("--team-a", type=int, default=5, help="Team 1 tank count.")
    parser.add_argument("--team-b", type=int, default=5, help="Team 2 tank count.")
    parser.add_argument("--base-port", type=int, default=8001, help="Base agent port.")
    parser.add_argument(
        "--agent-host", type=str, default="127.0.0.1", help="Agent host used by engine."
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=1.0,
        help="HTTP timeout for engine->agent requests.",
    )

    parser.add_argument(
        "--eval-team",
        type=int,
        choices=[1, 2],
        default=1,
        help="Team for which fitness is computed.",
    )

    parser.add_argument(
        "--restart-agents-per-game",
        action="store_true",
        default=True,
        help="Restart agent processes before each match (recommended).",
    )
    parser.add_argument(
        "--keep-agents",
        dest="restart_agents_per_game",
        action="store_false",
        help="Keep agent processes for all episodes.",
    )

    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for agent HTTP servers to become ready.",
    )
    parser.add_argument(
        "--startup-poll-interval",
        type=float,
        default=0.2,
        help="Health-check polling interval (seconds).",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=30.0,
        help="Print a heartbeat every N seconds while a match is running.",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Engine log level during training.",
    )
    parser.add_argument(
        "--sudden-death-tick",
        type=int,
        default=2500,
        help="Tick at which sudden death starts (smaller = shorter matches).",
    )
    parser.add_argument(
        "--sudden-death-damage",
        type=int,
        default=-2,
        help="HP damage per tick during sudden death (negative value in engine config).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "logs"),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="training_run",
        help="Filename prefix for outputs.",
    )

    parser.add_argument(
        "--verbose-agents",
        action="store_true",
        help="Show agent stdout/stderr in console.",
    )
    parser.add_argument(
        "--set-agent-names",
        action="store_true",
        help="Pass --name TEAM_X_N to agent scripts when spawning processes.",
    )

    parser.add_argument("--w-win", type=float, default=200.0)
    parser.add_argument("--w-loss", type=float, default=-200.0)
    parser.add_argument("--w-draw", type=float, default=0.0)
    parser.add_argument("--w-kill", type=float, default=30.0)
    parser.add_argument("--w-damage", type=float, default=0.25)
    parser.add_argument("--w-alive", type=float, default=10.0)
    parser.add_argument("--w-enemy-alive", type=float, default=-10.0)

    return parser.parse_args()


def ensure_maps_exist(map_names: List[str]) -> None:
    maps_dir = ENGINE_DIR / "backend" / "maps"
    available = {p.name for p in maps_dir.glob("*.csv")}
    missing = [m for m in map_names if m not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing map(s): {missing}. Available maps: {sorted(available)}"
        )


def configure_engine(args: argparse.Namespace) -> None:
    game_loop_module.TEAM_A_NBR = args.team_a
    game_loop_module.TEAM_B_NBR = args.team_b
    game_loop_module.AGENT_BASE_PORT = args.base_port
    game_loop_module.AGENT_HOST = args.agent_host
    game_loop_module.AGENT_TIMEOUT = args.agent_timeout
    game_config.game_rules.sudden_death_tick = int(args.sudden_death_tick)
    game_config.game_rules.sudden_death_damage_per_tick = int(args.sudden_death_damage)
    set_log_level(args.log_level)


def _http_ping(url: str, timeout: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def wait_for_agents(
    host: str, ports: List[int], timeout_s: float, poll_interval_s: float
) -> bool:
    deadline = time.time() + timeout_s
    urls = [f"http://{host}:{port}/" for port in ports]
    while time.time() < deadline:
        if all(_http_ping(url) for url in urls):
            return True
        time.sleep(poll_interval_s)
    return False


def _proc_stdio(verbose: bool):
    if verbose:
        return None, None
    return subprocess.DEVNULL, subprocess.DEVNULL


def start_agent_group(
    script_path: Path,
    count: int,
    start_port: int,
    name_prefix: str,
    extra_args: List[str],
    verbose: bool,
    set_agent_names: bool,
) -> List[subprocess.Popen]:
    if not script_path.exists():
        raise FileNotFoundError(f"Agent script does not exist: {script_path}")

    stdout_target, stderr_target = _proc_stdio(verbose)
    processes: List[subprocess.Popen] = []

    for i in range(count):
        port = start_port + i
        cmd = [sys.executable, str(script_path), "--port", str(port), *extra_args]

        # Optional: some agent scripts expose --name, others don't.
        if set_agent_names:
            cmd += ["--name", f"{name_prefix}_{i + 1}"]

        proc = subprocess.Popen(cmd, stdout=stdout_target, stderr=stderr_target)
        processes.append(proc)

    return processes


def stop_processes(processes: List[subprocess.Popen]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=2.0)


def team_stats(scoreboards: List[Dict[str, Any]], team: int) -> Dict[str, float]:
    rows = [r for r in scoreboards if int(r.get("team", -1)) == team]
    return {
        "kills": float(sum(r.get("tanks_killed", 0) for r in rows)),
        "damage": float(sum(r.get("damage_dealt", 0.0) for r in rows)),
    }


def normalize_team_counts(raw_counts: Any) -> Dict[int, int]:
    if not isinstance(raw_counts, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in raw_counts.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def is_game_success(results: Dict[str, Any]) -> bool:
    """
    Normalize success status across slightly inconsistent engine return schemas.
    """
    if "success" in results:
        return bool(results["success"])
    if results.get("error"):
        return False
    if "total_ticks" in results:
        return True
    return False


def compute_fitness(
    results: Dict[str, Any], eval_team: int, w: FitnessWeights
) -> Dict[str, float]:
    if not is_game_success(results):
        return {
            "fitness": -1_000_000.0,
            "kills": 0.0,
            "damage": 0.0,
            "alive": 0.0,
            "enemy_alive": 0.0,
            "win_component": w.loss,
        }

    scoreboards = results.get("scoreboards", [])
    winner = results.get("winner_team")
    final_counts = normalize_team_counts(results.get("final_team_counts", {}))
    enemy_team = 2 if eval_team == 1 else 1

    own = team_stats(scoreboards, eval_team)
    alive = float(final_counts.get(eval_team, 0))
    enemy_alive = float(final_counts.get(enemy_team, 0))

    if winner == eval_team:
        win_component = w.win
    elif winner is None:
        win_component = w.draw
    else:
        win_component = w.loss

    fitness = (
        win_component
        + w.kill * own["kills"]
        + w.damage * own["damage"]
        + w.alive * alive
        + w.enemy_alive * enemy_alive
    )

    return {
        "fitness": float(fitness),
        "kills": own["kills"],
        "damage": own["damage"],
        "alive": alive,
        "enemy_alive": enemy_alive,
        "win_component": float(win_component),
    }


def run_single_match(
    episode_idx: int,
    map_name: str,
    seed: int,
    args: argparse.Namespace,
    weights: FitnessWeights,
) -> Dict[str, Any]:
    random.seed(seed)
    started_processes: List[subprocess.Popen] = []

    if args.restart_agents_per_game:
        team1_script = Path(args.agent_team1).resolve()
        team2_script = Path(args.agent_team2).resolve()

        team1 = start_agent_group(
            script_path=team1_script,
            count=args.team_a,
            start_port=args.base_port,
            name_prefix="T1",
            extra_args=args.agent1_arg,
            verbose=args.verbose_agents,
            set_agent_names=args.set_agent_names,
        )
        team2 = start_agent_group(
            script_path=team2_script,
            count=args.team_b,
            start_port=args.base_port + args.team_a,
            name_prefix="T2",
            extra_args=args.agent2_arg,
            verbose=args.verbose_agents,
            set_agent_names=args.set_agent_names,
        )
        started_processes = team1 + team2

        ports = list(range(args.base_port, args.base_port + args.team_a + args.team_b))
        if not wait_for_agents(
            host=args.agent_host,
            ports=ports,
            timeout_s=args.startup_timeout,
            poll_interval_s=args.startup_poll_interval,
        ):
            # Diagnostyka: Sprawdź, czy któryś z procesów agentów zakończył się przedwcześnie
            dead_procs = [p for p in started_processes if p.poll() is not None]
            if dead_procs:
                print(f"\n[ERROR] {len(dead_procs)} agentów zakończyło działanie (crash) podczas startu!")
                print("Użyj flagi --verbose-agents aby zobaczyć błędy (np. brakujące biblioteki).")
            stop_processes(started_processes)
            return {
                "episode": episode_idx,
                "map": map_name,
                "seed": seed,
                "success": False,
                "error": "Agents failed to start in time",
                "fitness": -1_000_000.0,
                "winner_team": None,
                "total_ticks": 0,
                "kills": 0.0,
                "damage": 0.0,
                "alive": 0.0,
                "enemy_alive": 0.0,
            }

    started_at = time.time()
    game_result_holder: Dict[str, Any] = {}
    game_exc_holder: Dict[str, Exception] = {}

    def _run_game():
        try:
            game_result_holder["result"] = run_game(map_seed=map_name, headless=True)
        except Exception as e:
            game_exc_holder["exc"] = e

    game_thread = threading.Thread(target=_run_game, daemon=True)
    game_thread.start()

    heartbeat_sec = max(1.0, float(getattr(args, "heartbeat_sec", 30.0)))
    while game_thread.is_alive():
        game_thread.join(timeout=heartbeat_sec)
        if game_thread.is_alive():
            elapsed_now = time.time() - started_at
            print(
                f"[match {episode_idx + 1}] still running... "
                f"map={map_name} elapsed={int(elapsed_now)}s",
                flush=True,
            )

    elapsed = time.time() - started_at
    if "exc" in game_exc_holder:
        raise game_exc_holder["exc"]
    game_result = game_result_holder.get(
        "result",
        {"success": False, "error": "run_game returned no result"},
    )

    if started_processes:
        stop_processes(started_processes)

    parts = compute_fitness(
        results=game_result,
        eval_team=args.eval_team,
        w=weights,
    )

    out = {
        "episode": episode_idx,
        "map": map_name,
        "seed": seed,
        "success": is_game_success(game_result),
        "error": game_result.get("error", ""),
        "winner_team": game_result.get("winner_team"),
        "total_ticks": int(game_result.get("total_ticks", 0)),
        "elapsed_sec": round(elapsed, 4),
        **parts,
    }
    return out


def start_persistent_agents(args: argparse.Namespace) -> List[subprocess.Popen]:
    team1_script = Path(args.agent_team1).resolve()
    team2_script = Path(args.agent_team2).resolve()

    team1 = start_agent_group(
        script_path=team1_script,
        count=args.team_a,
        start_port=args.base_port,
        name_prefix="T1",
        extra_args=args.agent1_arg,
        verbose=args.verbose_agents,
        set_agent_names=args.set_agent_names,
    )
    team2 = start_agent_group(
        script_path=team2_script,
        count=args.team_b,
        start_port=args.base_port + args.team_a,
        name_prefix="T2",
        extra_args=args.agent2_arg,
        verbose=args.verbose_agents,
        set_agent_names=args.set_agent_names,
    )
    processes = team1 + team2

    ports = list(range(args.base_port, args.base_port + args.team_a + args.team_b))
    ok = wait_for_agents(
        host=args.agent_host,
        ports=ports,
        timeout_s=args.startup_timeout,
        poll_interval_s=args.startup_poll_interval,
    )
    if not ok:
        stop_processes(processes)
        raise RuntimeError("Agents failed to start in time")

    return processes


def write_outputs(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    weights: FitnessWeights,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.output_prefix}_{ts}.csv"
    json_path = output_dir / f"{args.output_prefix}_{ts}.json"

    fieldnames = [
        "episode",
        "map",
        "seed",
        "success",
        "error",
        "winner_team",
        "total_ticks",
        "elapsed_sec",
        "fitness",
        "kills",
        "damage",
        "alive",
        "enemy_alive",
        "win_component",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ok_rows = [r for r in rows if r.get("success")]
    avg_fitness = (
        sum(float(r.get("fitness", 0.0)) for r in ok_rows) / len(ok_rows)
        if ok_rows
        else float("-inf")
    )

    summary = {
        "generated_at": ts,
        "config": {
            "episodes": args.episodes,
            "maps": args.maps,
            "base_seed": args.base_seed,
            "agent_team1": str(Path(args.agent_team1).resolve()),
            "agent_team2": str(Path(args.agent_team2).resolve()),
            "team_a": args.team_a,
            "team_b": args.team_b,
            "base_port": args.base_port,
            "agent_host": args.agent_host,
            "agent_timeout": args.agent_timeout,
            "eval_team": args.eval_team,
            "restart_agents_per_game": args.restart_agents_per_game,
            "log_level": args.log_level,
            "weights": weights.__dict__,
        },
        "results": {
            "matches_total": len(rows),
            "matches_successful": len(ok_rows),
            "avg_fitness_successful": avg_fitness,
        },
        "files": {"csv": str(csv_path), "json": str(json_path)},
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(
        f"Matches: {summary['results']['matches_successful']}/{summary['results']['matches_total']} | "
        f"Avg fitness: {summary['results']['avg_fitness_successful']:.3f}"
    )


def main() -> int:
    args = parse_args()
    ensure_maps_exist(args.maps)
    configure_engine(args)

    weights = FitnessWeights(
        win=args.w_win,
        loss=args.w_loss,
        draw=args.w_draw,
        kill=args.w_kill,
        damage=args.w_damage,
        alive=args.w_alive,
        enemy_alive=args.w_enemy_alive,
    )

    output_dir = Path(args.output_dir).resolve()

    persistent_processes: List[subprocess.Popen] = []
    try:
        if not args.restart_agents_per_game:
            persistent_processes = start_persistent_agents(args)

        rows: List[Dict[str, Any]] = []
        total_match_time = 0.0
        for episode_idx in range(args.episodes):
            map_name = args.maps[episode_idx % len(args.maps)]
            seed = args.base_seed + episode_idx

            row = run_single_match(
                episode_idx=episode_idx,
                map_name=map_name,
                seed=seed,
                args=args,
                weights=weights,
            )
            rows.append(row)
            total_match_time += float(row.get("elapsed_sec", 0.0))

            status = "OK" if row["success"] else "FAIL"
            done = episode_idx + 1
            avg_match_time = total_match_time / max(1, done)
            remaining_matches = args.episodes - done
            eta_sec = remaining_matches * avg_match_time
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            eta_s = int(eta_sec % 60)
            print(
                f"[{episode_idx + 1:03d}/{args.episodes:03d}] "
                f"{status} map={map_name} seed={seed} "
                f"winner={row['winner_team']} ticks={row['total_ticks']} "
                f"fitness={row['fitness']:.3f} "
                f"eta={eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
            )

        write_outputs(rows=rows, args=args, weights=weights, output_dir=output_dir)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"\nFatal error: {e}")
        return 1
    finally:
        if persistent_processes:
            stop_processes(persistent_processes)


if __name__ == "__main__":
    raise SystemExit(main())
