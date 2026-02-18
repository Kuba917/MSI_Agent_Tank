"""
Training runner for the fuzzy DQN tank agent.

Default setup:
- 10 total agents (5v5),
- first N agents are learning agents (DQN.py --train),
- remaining agents are rule-based baselines (Agent_1.py),
- game engine runs in headless mode for each episode.

This runner also supports:
- map curriculum (small map first, then harder maps),
- persistent agent processes across episodes (so replay buffer is not reset),
- optional self-play stage (some opponents switched from baseline to DQN).
"""

from __future__ import annotations

import ast
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request


THIS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = THIS_DIR.parent / "02_FRAKCJA_SILNIKA"
AGENT_DIR = THIS_DIR


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fuzzy DQN agent in 5v5 matches")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--learning-agents", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=8001)

    parser.add_argument("--model-path", type=str, default=str(AGENT_DIR / "fuzzy_dqn_model.pt"))
    parser.add_argument("--rules", type=int, default=32)
    parser.add_argument("--mf-type", choices=["gaussian", "bell", "triangular"], default="gaussian")
    parser.add_argument("--map-seed", type=str, default="road_trees.csv")
    parser.add_argument(
        "--map-curriculum",
        type=str,
        default="",
        help=(
            "Comma-separated map schedule, e.g. "
            "'road_trees.csv:300,semi-open.csv:150,advanced_road_trees.csv:150'. "
            "If omitted, --map-seed is used for all episodes."
        ),
    )

    parser.add_argument("--max-ticks", type=int, default=5000)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING")

    parser.add_argument("--restart-agents-every", type=int, default=0, help="Restart agent processes every N episodes (0 = keep alive).")
    parser.add_argument("--start-delay", type=float, default=4.0)
    parser.add_argument("--ready-timeout", type=float, default=120.0, help="Max seconds to wait for all agent HTTP servers to become ready.")
    parser.add_argument("--episode-delay", type=float, default=0.5)

    parser.add_argument("--selfplay-start-episode", type=int, default=0, help="Enable self-play opponents from this episode (1-based, 0 = disabled).")
    parser.add_argument("--selfplay-opponents", type=int, default=0, help="How many non-learning agents to run as DQN inference in self-play stage.")
    parser.add_argument("--selfplay-model-path", type=str, default="", help="Checkpoint path for self-play opponents. Default: first learner checkpoint.")

    parser.add_argument("--warmup-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--target-sync-every", type=int, default=500)
    parser.add_argument("--save-every-games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--render", action="store_true", help="Show game window (disable headless mode)")

    return parser.parse_args()


def terminate_processes(processes: List[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + 3.0
    for proc in processes:
        if proc.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def wait_for_agents_ready(
    base_port: int,
    total_agents: int,
    timeout_seconds: float,
) -> List[int]:
    pending = {base_port + idx for idx in range(total_agents)}
    deadline = time.time() + max(0.0, float(timeout_seconds))

    while pending and time.time() < deadline:
        for port in list(pending):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.7) as response:
                    if int(getattr(response, "status", 200)) < 500:
                        pending.remove(port)
            except urllib.error.URLError:
                continue
            except Exception:
                continue
        if pending:
            time.sleep(0.2)

    return sorted(pending)


def learner_model_path(base_path: Path, learner_idx: int, total_learners: int) -> Path:
    if total_learners == 1:
        return base_path

    suffix = base_path.suffix or ".pt"
    return base_path.with_name(f"{base_path.stem}_agent{learner_idx + 1}{suffix}")


def model_path_with_tag(path: Path, tag: str) -> Path:
    suffix = path.suffix or ".pt"
    return path.with_name(f"{path.stem}_{tag}{suffix}")


def latest_summary_file() -> Optional[Path]:
    logs_dir = ENGINE_DIR / "logs"
    if not logs_dir.exists():
        return None

    summaries = sorted(logs_dir.glob("summary_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not summaries:
        return None
    return summaries[0]


def parse_summary_metrics(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics: Dict[str, Any] = {
        "shots_fired": None,
        "hits_landed": None,
        "tanks_killed": None,
        "projectile_deaths": None,
        "non_projectile_deaths": None,
        "reason": None,
        "winner_team": None,
        "sudden_death_reached": None,
        "move_command_ratio": None,
        "heading_turn_ratio": None,
        "barrel_turn_ratio": None,
        "idle_ratio": None,
        "movement_success_ratio": None,
        "mobility_label": None,
    }

    patterns = {
        "shots_fired": r"Shots Fired:\s*([0-9]+)",
        "hits_landed": r"Hits Landed:\s*([0-9]+)",
        "tanks_killed": r"Tanks Killed:\s*([0-9]+)",
        "projectile_deaths": r"Projectile Deaths:\s*([0-9]+)",
        "non_projectile_deaths": r"Non-Projectile Deaths:\s*([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = int(match.group(1))

    result_line = None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            result_line = stripped
            break

    if result_line:
        try:
            result = ast.literal_eval(result_line)
            if isinstance(result, dict):
                metrics["reason"] = result.get("reason")
                metrics["winner_team"] = result.get("winner_team")
                metrics["sudden_death_reached"] = result.get("sudden_death_reached")
                behavior = result.get("behavior", {})
                if isinstance(behavior, dict):
                    metrics["move_command_ratio"] = behavior.get("move_command_ratio")
                    metrics["heading_turn_ratio"] = behavior.get("heading_turn_ratio")
                    metrics["barrel_turn_ratio"] = behavior.get("barrel_turn_ratio")
                    metrics["idle_ratio"] = behavior.get("idle_ratio")
                    metrics["movement_success_ratio"] = behavior.get("movement_success_ratio")
                    metrics["mobility_label"] = behavior.get("mobility_label")
        except Exception:
            pass

    return metrics


def classify_elimination_source(metrics: Dict[str, Any]) -> str:
    tanks_killed = int(metrics.get("tanks_killed") or 0)
    hits_landed = int(metrics.get("hits_landed") or 0)
    projectile_deaths = metrics.get("projectile_deaths")
    non_projectile_deaths = metrics.get("non_projectile_deaths")
    reason = str(metrics.get("reason") or "").lower()
    sudden_death = bool(metrics.get("sudden_death_reached"))

    if tanks_killed <= 0:
        return "none"
    if projectile_deaths is not None and non_projectile_deaths is not None:
        projectile = int(projectile_deaths)
        non_projectile = int(non_projectile_deaths)
        if projectile > 0 and non_projectile > 0:
            return "mixed_combat_and_time"
        if projectile > 0:
            return "combat_projectile"
        if non_projectile > 0 and (sudden_death or "sudden_death" in reason):
            return "time_sudden_death"
        if non_projectile > 0:
            return "environment_or_time"
    if hits_landed > 0:
        return "combat_shots"
    if sudden_death or "sudden_death" in reason:
        return "time_sudden_death"
    return "environment_or_time"


def classify_mobility(metrics: Dict[str, Any]) -> str:
    label = metrics.get("mobility_label")
    if isinstance(label, str) and label:
        return label

    move_ratio = float(metrics.get("move_command_ratio") or 0.0)
    heading_turn_ratio = float(metrics.get("heading_turn_ratio") or 0.0)
    barrel_turn_ratio = float(metrics.get("barrel_turn_ratio") or 0.0)
    idle_ratio = float(metrics.get("idle_ratio") or 0.0)
    movement_success_ratio = float(metrics.get("movement_success_ratio") or 0.0)

    turning_ratio = heading_turn_ratio + barrel_turn_ratio
    if idle_ratio >= 0.55:
        return "mostly_idle"
    if move_ratio >= 0.45 and movement_success_ratio >= 0.6:
        return "mostly_moving"
    if move_ratio <= 0.25 and turning_ratio >= 0.55:
        return "mostly_spinning"
    return "mixed_motion"


def build_map_schedule(args: argparse.Namespace) -> List[str]:
    if not args.map_curriculum.strip():
        return [args.map_seed for _ in range(args.episodes)]

    schedule: List[str] = []
    chunks = [chunk.strip() for chunk in args.map_curriculum.split(",") if chunk.strip()]

    for chunk in chunks:
        match = re.fullmatch(r"(.+?)\s*[:*]\s*([0-9]+)", chunk)
        if match:
            map_name = match.group(1).strip()
            count = int(match.group(2))
        else:
            map_name = chunk
            count = 1

        if not map_name:
            continue
        if count <= 0:
            continue

        schedule.extend([map_name] * count)

    if not schedule:
        return [args.map_seed for _ in range(args.episodes)]

    if len(schedule) < args.episodes:
        schedule.extend([schedule[-1]] * (args.episodes - len(schedule)))
    return schedule[: args.episodes]


def planned_selfplay_opponents(args: argparse.Namespace, episode: int, non_learner_count: int) -> int:
    if args.selfplay_start_episode <= 0:
        return 0
    if episode < args.selfplay_start_episode:
        return 0
    return max(0, min(int(args.selfplay_opponents), non_learner_count))


def resolve_selfplay_model_path(args: argparse.Namespace, model_path: Path, learner_count: int) -> Path:
    if args.selfplay_model_path.strip():
        return Path(args.selfplay_model_path)
    return learner_model_path(model_path, 0, max(1, learner_count))


def launch_agents(args: argparse.Namespace, episode: int) -> Tuple[List[subprocess.Popen], Dict[str, Any]]:
    total_agents = args.team_size * 2
    learner_count = max(0, min(args.learning_agents, total_agents))
    non_learner_count = total_agents - learner_count
    selfplay_requested = planned_selfplay_opponents(args, episode, non_learner_count)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    selfplay_model_path = resolve_selfplay_model_path(args, model_path, learner_count)
    selfplay_enabled = selfplay_requested > 0 and selfplay_model_path.exists()

    processes: List[subprocess.Popen] = []
    launch_info: Dict[str, Any] = {
        "learners": learner_count,
        "selfplay_requested": selfplay_requested,
        "selfplay": 0,
        "baselines": 0,
        "selfplay_enabled": selfplay_enabled,
        "selfplay_model_path": str(selfplay_model_path),
    }

    for idx in range(total_agents):
        port = args.base_port + idx
        seed_for_agent = int(args.seed) + idx

        if idx < learner_count:
            model_for_agent = learner_model_path(model_path, idx, learner_count)
            best_for_agent = model_path_with_tag(model_for_agent, "best")
            final_for_agent = model_path_with_tag(model_for_agent, "final")
            cmd = [
                sys.executable,
                "DQN.py",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--name",
                f"Learner_{idx + 1}",
                "--train",
                "--model-path",
                str(model_for_agent),
                "--best-model-path",
                str(best_for_agent),
                "--final-model-path",
                str(final_for_agent),
                "--rules",
                str(args.rules),
                "--mf-type",
                args.mf_type,
                "--warmup-steps",
                str(max(0, int(args.warmup_steps))),
                "--batch-size",
                str(max(16, int(args.batch_size))),
                "--train-every",
                str(max(1, int(args.train_every))),
                "--target-sync-every",
                str(max(1, int(args.target_sync_every))),
                "--save-every-games",
                str(max(1, int(args.save_every_games))),
                "--seed",
                str(seed_for_agent),
            ]
        else:
            non_learner_idx = idx - learner_count
            use_selfplay = selfplay_enabled and non_learner_idx < selfplay_requested
            if use_selfplay:
                launch_info["selfplay"] += 1
                cmd = [
                    sys.executable,
                    "DQN.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--name",
                    f"SelfPlay_{idx + 1}",
                    "--model-path",
                    str(selfplay_model_path),
                    "--rules",
                    str(args.rules),
                    "--mf-type",
                    args.mf_type,
                    "--seed",
                    str(seed_for_agent),
                ]
            else:
                launch_info["baselines"] += 1
                cmd = [
                    sys.executable,
                    "Agent_1.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--name",
                    f"Baseline_{idx + 1}",
                ]

        stdout = None if args.verbose else subprocess.DEVNULL
        stderr = None if args.verbose else subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            cwd=str(AGENT_DIR),
            stdout=stdout,
            stderr=stderr,
        )
        processes.append(proc)

    return processes, launch_info


def parse_winner(output: str) -> Optional[str]:
    match = re.search(r"winner_team[\"']?\s*[:=]\s*([0-9]+)", output)
    if match:
        return match.group(1)

    match = re.search(r"Winner:\s*Team\s*([0-9]+)", output)
    if match:
        return match.group(1)

    return None


def run_engine_episode(args: argparse.Namespace, map_seed: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "run_game.py",
        "--log-level",
        args.log_level,
        "--map-seed",
        map_seed,
    ]

    if not args.render:
        cmd.append("--headless")

    if args.max_ticks > 0:
        cmd.extend(["--max-ticks", str(args.max_ticks)])

    if args.verbose:
        return subprocess.run(cmd, cwd=str(ENGINE_DIR), check=False)

    return subprocess.run(
        cmd,
        cwd=str(ENGINE_DIR),
        check=False,
        text=True,
        capture_output=True,
    )


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_episode_reports(episode_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_episodes = len(episode_reports)
    episodes_ok = sum(1 for row in episode_reports if bool(row.get("episode_ok")))
    with_summary = sum(1 for row in episode_reports if bool(row.get("has_new_summary")))

    winner_counts: Dict[str, int] = {}
    elimination_counts: Dict[str, int] = {}
    mobility_counts: Dict[str, int] = {}
    sudden_death_true = 0

    shots_total = 0
    hits_total = 0
    tanks_killed_total = 0
    projectile_deaths_total = 0
    non_projectile_deaths_total = 0

    turn_ratios: List[float] = []
    move_ratios: List[float] = []
    idle_ratios: List[float] = []
    movement_success_ratios: List[float] = []

    for row in episode_reports:
        winner_key = str(row.get("winner_team")) if row.get("winner_team") is not None else "unknown"
        winner_counts[winner_key] = winner_counts.get(winner_key, 0) + 1

        elim = row.get("elimination_source")
        if elim:
            elimination_counts[str(elim)] = elimination_counts.get(str(elim), 0) + 1

        mobility = row.get("mobility")
        if mobility:
            mobility_counts[str(mobility)] = mobility_counts.get(str(mobility), 0) + 1

        if bool(row.get("sudden_death_reached")):
            sudden_death_true += 1

        shots_total += int(row.get("shots_fired") or 0)
        hits_total += int(row.get("hits_landed") or 0)
        tanks_killed_total += int(row.get("tanks_killed") or 0)
        projectile_deaths_total += int(row.get("projectile_deaths") or 0)
        non_projectile_deaths_total += int(row.get("non_projectile_deaths") or 0)

        tr = _to_float(row.get("turn_ratio"))
        mr = _to_float(row.get("move_command_ratio"))
        ir = _to_float(row.get("idle_ratio"))
        msr = _to_float(row.get("movement_success_ratio"))
        if tr is not None:
            turn_ratios.append(tr)
        if mr is not None:
            move_ratios.append(mr)
        if ir is not None:
            idle_ratios.append(ir)
        if msr is not None:
            movement_success_ratios.append(msr)

    total_deaths = projectile_deaths_total + non_projectile_deaths_total
    projectile_share = (projectile_deaths_total / total_deaths) if total_deaths > 0 else 0.0
    hit_accuracy = (hits_total / shots_total) if shots_total > 0 else 0.0

    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "episodes_total": total_episodes,
        "episodes_ok": episodes_ok,
        "episodes_with_summary": with_summary,
        "winner_counts": winner_counts,
        "elimination_counts": elimination_counts,
        "mobility_counts": mobility_counts,
        "sudden_death_episodes": sudden_death_true,
        "shots_fired_total": shots_total,
        "hits_landed_total": hits_total,
        "tanks_killed_total": tanks_killed_total,
        "projectile_deaths_total": projectile_deaths_total,
        "non_projectile_deaths_total": non_projectile_deaths_total,
        "hit_accuracy": round(hit_accuracy, 4),
        "projectile_death_share": round(projectile_share, 4),
        "avg_turn_ratio": _mean(turn_ratios),
        "avg_move_command_ratio": _mean(move_ratios),
        "avg_idle_ratio": _mean(idle_ratios),
        "avg_movement_success_ratio": _mean(movement_success_ratios),
    }


def save_training_report(
    args: argparse.Namespace,
    episode_reports: List[Dict[str, Any]],
    map_schedule: List[str],
    exit_code: int,
    start_unix: float,
) -> Tuple[Path, Path]:
    reports_dir = AGENT_DIR / "training_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"training_report_{timestamp}.json"
    csv_path = reports_dir / f"training_report_{timestamp}.csv"

    aggregate = summarize_episode_reports(episode_reports)
    payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(max(0.0, time.time() - start_unix), 2),
        "exit_code": int(exit_code),
        "config": vars(args),
        "map_schedule": map_schedule,
        "aggregate": aggregate,
        "episodes": episode_reports,
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    csv_fields = [
        "episode",
        "map_seed",
        "return_code",
        "episode_ok",
        "has_new_summary",
        "summary_file",
        "winner_team",
        "learners",
        "selfplay",
        "baselines",
        "shots_fired",
        "hits_landed",
        "tanks_killed",
        "projectile_deaths",
        "non_projectile_deaths",
        "elimination_source",
        "mobility",
        "move_command_ratio",
        "idle_ratio",
        "turn_ratio",
        "movement_success_ratio",
        "reason",
        "sudden_death_reached",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in episode_reports:
            writer.writerow({field: row.get(field) for field in csv_fields})

    return json_path, csv_path


def main() -> int:
    args = build_args()

    if not ENGINE_DIR.exists():
        print(f"Engine directory not found: {ENGINE_DIR}")
        return 1

    total_agents = args.team_size * 2
    learner_count = max(0, min(args.learning_agents, total_agents))
    map_schedule = build_map_schedule(args)

    print("=== Fuzzy DQN Training Runner ===")
    print(f"Episodes: {args.episodes}")
    print(f"Team size: {args.team_size} (total agents: {total_agents})")
    print(f"Learning agents: {learner_count}")
    print(f"Model path: {args.model_path}")
    print(f"MF type: {args.mf_type}, rules: {args.rules}")
    print(
        "Learner training params: "
        f"warmup={max(0, int(args.warmup_steps))} "
        f"batch={max(16, int(args.batch_size))} "
        f"train_every={max(1, int(args.train_every))} "
        f"target_sync={max(1, int(args.target_sync_every))}"
    )
    print(f"Agent restart policy: every {args.restart_agents_every} episodes (0 = persistent)")
    print(
        "Self-play policy: "
        f"start_episode={args.selfplay_start_episode}, "
        f"opponents={args.selfplay_opponents}, "
        f"model={args.selfplay_model_path or '(first learner checkpoint)'}"
    )
    if args.map_curriculum.strip():
        print(f"Map curriculum: {args.map_curriculum}")
    else:
        print(f"Map seed: {args.map_seed}")
    print()

    processes: List[subprocess.Popen] = []
    launch_info: Optional[Dict[str, Any]] = None
    last_launch_episode = 0
    episode_reports: List[Dict[str, Any]] = []
    run_start_unix = time.time()
    exit_code = 0

    try:
        for episode in range(1, args.episodes + 1):
            map_seed = map_schedule[episode - 1]
            non_learner_count = total_agents - learner_count
            desired_selfplay = planned_selfplay_opponents(args, episode, non_learner_count)

            processes_dead = any(proc.poll() is not None for proc in processes)
            role_change = bool(
                launch_info
                and desired_selfplay != int(launch_info.get("selfplay_requested", 0))
            )
            periodic_restart = bool(
                processes
                and args.restart_agents_every > 0
                and (episode - last_launch_episode) >= args.restart_agents_every
            )
            retry_selfplay_enable = bool(
                launch_info
                and desired_selfplay > 0
                and not bool(launch_info.get("selfplay_enabled", False))
            )

            need_restart = (not processes) or processes_dead or role_change or periodic_restart or retry_selfplay_enable
            if need_restart:
                if processes:
                    terminate_processes(processes)
                    processes = []
                    launch_info = None

                print(f"[Episode {episode}/{args.episodes}] starting agents...")
                processes, launch_info = launch_agents(args, episode)
                last_launch_episode = episode

                print(
                    f"[Episode {episode}/{args.episodes}] roster: "
                    f"learners={launch_info['learners']} "
                    f"selfplay={launch_info['selfplay']} "
                    f"baselines={launch_info['baselines']}"
                )
                if int(launch_info.get("selfplay_requested", 0)) > 0 and not bool(launch_info.get("selfplay_enabled", False)):
                    print(
                        f"[Episode {episode}/{args.episodes}] self-play requested but disabled "
                        f"(model not found: {launch_info.get('selfplay_model_path')})."
                    )

                time.sleep(max(0.0, args.start_delay))
                unready_ports = wait_for_agents_ready(
                    base_port=int(args.base_port),
                    total_agents=total_agents,
                    timeout_seconds=float(args.ready_timeout),
                )
                if unready_ports:
                    print(
                        f"[Episode {episode}/{args.episodes}] warning: agents not ready on ports: "
                        f"{', '.join(str(p) for p in unready_ports)}"
                    )
                    if not args.continue_on_error:
                        print("Stopping training because some agent servers did not become ready.")
                        exit_code = 1
                        break
            else:
                print(
                    f"[Episode {episode}/{args.episodes}] reusing agents "
                    f"(learners={launch_info['learners']} selfplay={launch_info['selfplay']} baselines={launch_info['baselines']})"
                )

            print(f"[Episode {episode}/{args.episodes}] running engine on map={map_seed}...")
            summary_before = latest_summary_file()
            summary_before_mtime = summary_before.stat().st_mtime if summary_before else -1.0
            result = run_engine_episode(args, map_seed=map_seed)

            summary_after = latest_summary_file()
            has_new_summary = bool(
                summary_after and summary_after.stat().st_mtime > summary_before_mtime
            )
            episode_metrics = (
                parse_summary_metrics(summary_after)
                if summary_after and has_new_summary
                else None
            )
            rc = result.returncode
            winner = parse_winner((result.stdout or "") if hasattr(result, "stdout") else "")
            if winner is None and summary_after and has_new_summary:
                winner = parse_winner(summary_after.read_text(encoding="utf-8", errors="ignore"))
            if winner is None and episode_metrics and episode_metrics.get("winner_team") is not None:
                winner = str(episode_metrics["winner_team"])

            if args.verbose:
                print(
                    f"[Episode {episode}/{args.episodes}] engine return code: {rc}, "
                    f"new_summary={has_new_summary}"
                )
                if episode_metrics:
                    turn_ratio = (
                        float(episode_metrics.get("heading_turn_ratio") or 0.0)
                        + float(episode_metrics.get("barrel_turn_ratio") or 0.0)
                    )
                    print(
                        f"[Episode {episode}/{args.episodes}] combat_stats: "
                        f"shots={episode_metrics.get('shots_fired')} "
                        f"hits={episode_metrics.get('hits_landed')} "
                        f"tanks_killed={episode_metrics.get('tanks_killed')} "
                        f"projectile_deaths={episode_metrics.get('projectile_deaths')} "
                        f"non_projectile_deaths={episode_metrics.get('non_projectile_deaths')} "
                        f"reason={episode_metrics.get('reason')} "
                        f"eliminations={classify_elimination_source(episode_metrics)} "
                        f"move_ratio={episode_metrics.get('move_command_ratio')} "
                        f"idle_ratio={episode_metrics.get('idle_ratio')} "
                        f"turn_ratio={round(turn_ratio, 4)} "
                        f"movement_success_ratio={episode_metrics.get('movement_success_ratio')} "
                        f"mobility={classify_mobility(episode_metrics)}"
                    )
            else:
                winner_text = winner if winner is not None else "unknown"
                print(
                    f"[Episode {episode}/{args.episodes}] return code={rc}, "
                    f"winner_team={winner_text}, new_summary={has_new_summary}"
                )
                if episode_metrics:
                    turn_ratio = (
                        float(episode_metrics.get("heading_turn_ratio") or 0.0)
                        + float(episode_metrics.get("barrel_turn_ratio") or 0.0)
                    )
                    print(
                        f"[Episode {episode}/{args.episodes}] combat_stats: "
                        f"shots={episode_metrics.get('shots_fired')} "
                        f"hits={episode_metrics.get('hits_landed')} "
                        f"tanks_killed={episode_metrics.get('tanks_killed')} "
                        f"projectile_deaths={episode_metrics.get('projectile_deaths')} "
                        f"non_projectile_deaths={episode_metrics.get('non_projectile_deaths')} "
                        f"reason={episode_metrics.get('reason')} "
                        f"eliminations={classify_elimination_source(episode_metrics)} "
                        f"move_ratio={episode_metrics.get('move_command_ratio')} "
                        f"idle_ratio={episode_metrics.get('idle_ratio')} "
                        f"turn_ratio={round(turn_ratio, 4)} "
                        f"movement_success_ratio={episode_metrics.get('movement_success_ratio')} "
                        f"mobility={classify_mobility(episode_metrics)}"
                    )

                if rc != 0 and not has_new_summary:
                    print("Engine stderr:")
                    print((result.stderr or "").strip()[:2000])

            elimination_source = None
            mobility = None
            turn_ratio_value = None
            if episode_metrics:
                elimination_source = classify_elimination_source(episode_metrics)
                mobility = classify_mobility(episode_metrics)
                turn_ratio_value = (
                    float(episode_metrics.get("heading_turn_ratio") or 0.0)
                    + float(episode_metrics.get("barrel_turn_ratio") or 0.0)
                )

            episode_report: Dict[str, Any] = {
                "episode": episode,
                "map_seed": map_seed,
                "return_code": rc,
                "episode_ok": False,  # filled below after episode_ok evaluation
                "has_new_summary": has_new_summary,
                "summary_file": str(summary_after) if (summary_after and has_new_summary) else None,
                "winner_team": _to_int(winner),
                "learners": int(launch_info.get("learners", 0)) if launch_info else 0,
                "selfplay": int(launch_info.get("selfplay", 0)) if launch_info else 0,
                "baselines": int(launch_info.get("baselines", 0)) if launch_info else 0,
                "shots_fired": _to_int((episode_metrics or {}).get("shots_fired")),
                "hits_landed": _to_int((episode_metrics or {}).get("hits_landed")),
                "tanks_killed": _to_int((episode_metrics or {}).get("tanks_killed")),
                "projectile_deaths": _to_int((episode_metrics or {}).get("projectile_deaths")),
                "non_projectile_deaths": _to_int((episode_metrics or {}).get("non_projectile_deaths")),
                "elimination_source": elimination_source,
                "mobility": mobility,
                "move_command_ratio": _to_float((episode_metrics or {}).get("move_command_ratio")),
                "idle_ratio": _to_float((episode_metrics or {}).get("idle_ratio")),
                "turn_ratio": turn_ratio_value,
                "movement_success_ratio": _to_float((episode_metrics or {}).get("movement_success_ratio")),
                "reason": (episode_metrics or {}).get("reason"),
                "sudden_death_reached": (episode_metrics or {}).get("sudden_death_reached"),
            }

            # In this project run_game.py may return code 1 even after a completed match
            # (known 'success' key mismatch in wrapper). A fresh summary file confirms
            # the episode finished, so treat that case as successful.
            episode_ok = (rc == 0) or has_new_summary
            episode_report["episode_ok"] = bool(episode_ok)
            episode_reports.append(episode_report)

            if not episode_ok and not args.continue_on_error:
                print("Stopping training due to engine error.")
                exit_code = int(rc) if int(rc) != 0 else 1
                break

            if episode < args.episodes:
                time.sleep(max(0.0, args.episode_delay))

    finally:
        if processes:
            terminate_processes(processes)
        try:
            report_json, report_csv = save_training_report(
                args=args,
                episode_reports=episode_reports,
                map_schedule=map_schedule[: max(0, len(episode_reports))],
                exit_code=exit_code,
                start_unix=run_start_unix,
            )
            print(f"Training report JSON: {report_json}")
            print(f"Training report CSV: {report_csv}")
        except Exception as exc:
            print(f"Failed to save training report: {exc}")

    if exit_code == 0:
        print("Training run finished.")
    else:
        print(f"Training run stopped with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
