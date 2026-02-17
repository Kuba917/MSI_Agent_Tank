"""
Analyze friendly-fire kill ratio from engine logs.

Reads recent sessions from summary_*.txt files and matches them with
game_*.log to count:
- enemy kills (team A killing team B),
- friendly kills (same-team projectile kills).
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS_DIR = THIS_DIR.parent / "02_FRAKCJA_SILNIKA" / "logs"

KILL_RE = re.compile(
    r"Tank\s+tank_(?P<victim_team>[12])_(?P<victim_idx>\d+)\s+"
    r"killed by\s+tank_(?P<killer_team>[12])_(?P<killer_idx>\d+)\s+\(projectile\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Friendly-fire session analyzer")
    parser.add_argument("--logs-dir", type=str, default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--last", type=int, default=20, help="How many recent sessions to include.")
    parser.add_argument(
        "--only-with-kills",
        action="store_true",
        help="Show only sessions where kill lines were found in game log.",
    )
    return parser.parse_args()


def parse_summary_metrics(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    default_run_id = path.stem.replace("summary_", "")
    metrics: Dict[str, Any] = {
        "run_id": default_run_id,
        "engine_session_id": None,
        "winner_team": None,
        "shots_fired": 0,
        "hits_landed": 0,
        "tanks_killed": 0,
        "projectile_deaths": 0,
        "non_projectile_deaths": 0,
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

    session_match = re.search(r"Session ID:\s*([A-Za-z0-9_:-]+)", text)
    if session_match:
        metrics["run_id"] = session_match.group(1).strip()

    result_line: Optional[str] = None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            result_line = stripped
            break

    if result_line:
        try:
            result = ast.literal_eval(result_line)
            if isinstance(result, dict):
                metrics["engine_session_id"] = result.get("session_id")
                winner = result.get("winner_team")
                metrics["winner_team"] = int(winner) if winner is not None else None
        except Exception:
            pass

    return metrics


def count_kills_from_game_log(path: Path) -> Tuple[int, int, int]:
    """
    Returns:
    - friendly_kills
    - enemy_kills
    - total_kill_lines
    """
    if not path.exists():
        return 0, 0, 0

    friendly = 0
    enemy = 0
    total = 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in KILL_RE.finditer(text):
        total += 1
        victim_team = int(match.group("victim_team"))
        killer_team = int(match.group("killer_team"))
        if victim_team == killer_team:
            friendly += 1
        else:
            enemy += 1
    return friendly, enemy, total


def format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(100.0 * numerator / denominator):5.1f}%"


def session_row(session: Dict[str, Any]) -> str:
    winner = session.get("winner_team")
    winner_txt = "-" if winner is None else str(winner)
    total_counted = int(session["friendly_kills"]) + int(session["enemy_kills"])
    return (
        f"{session['session_id']:<16}  "
        f"{winner_txt:>6}  "
        f"{session['shots_fired']:>5}  "
        f"{session['hits_landed']:>4}  "
        f"{session['projectile_deaths']:>8}  "
        f"{session['friendly_kills']:>8}  "
        f"{session['enemy_kills']:>5}  "
        f"{format_pct(int(session['friendly_kills']), total_counted):>7}  "
        f"{session['game_log_name']}"
    )


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir).expanduser().resolve()
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return 1

    summaries = sorted(
        logs_dir.glob("summary_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        print(f"No summary_*.txt files found in: {logs_dir}")
        return 1

    selected = summaries[: max(1, int(args.last))]
    rows: List[Dict[str, Any]] = []

    for summary_path in selected:
        metrics = parse_summary_metrics(summary_path)
        run_id = str(metrics["run_id"])
        game_log = logs_dir / f"game_{run_id}.log"
        friendly, enemy, kill_lines = count_kills_from_game_log(game_log)

        row = {
            "session_id": run_id,
            "winner_team": metrics["winner_team"],
            "engine_session_id": metrics["engine_session_id"],
            "shots_fired": int(metrics["shots_fired"]),
            "hits_landed": int(metrics["hits_landed"]),
            "projectile_deaths": int(metrics["projectile_deaths"]),
            "friendly_kills": int(friendly),
            "enemy_kills": int(enemy),
            "kill_lines": int(kill_lines),
            "game_log_name": game_log.name if game_log.exists() else "(missing)",
        }
        rows.append(row)

    if args.only_with_kills:
        rows = [row for row in rows if int(row["kill_lines"]) > 0]
        if not rows:
            print("No sessions with kill lines found. Run training with --log-level INFO.")
            return 0

    print(
        "session_id         winner  shots  hits  proj_kill  friendly  enemy  ff_ratio  game_log"
    )
    print(
        "----------------  ------  -----  ----  ---------  --------  -----  -------  ------------------------"
    )
    for row in rows:
        print(session_row(row))

    total_friendly = sum(int(row["friendly_kills"]) for row in rows)
    total_enemy = sum(int(row["enemy_kills"]) for row in rows)
    total_counted = total_friendly + total_enemy
    total_kill_lines = sum(int(row["kill_lines"]) for row in rows)
    sessions_with_kills = sum(1 for row in rows if int(row["kill_lines"]) > 0)

    print()
    print(
        f"TOTAL: sessions={len(rows)} sessions_with_kills={sessions_with_kills} "
        f"kill_lines={total_kill_lines} friendly={total_friendly} enemy={total_enemy} "
        f"ff_ratio={format_pct(total_friendly, total_counted)}"
    )
    if total_kill_lines == 0:
        print("Hint: friendly/enemy breakdown needs INFO logs (kill lines in game_*.log).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
