"""
Optimize fuzzy-agent parameters with GA or PSO using real headless games.

Outputs:
- generation-level metrics CSV
- per-generation candidate scores CSV
- best_params.json
- final_evaluation.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import train_runner as tr


# Parameter bounds for fuzzy_ga_agent.py.
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "near_distance": (8.0, 45.0),
    "mid_distance": (25.0, 90.0),
    "far_distance": (70.0, 160.0),
    "preferred_distance": (10.0, 90.0),
    "low_hp_ratio": (0.10, 0.80),
    "aggression": (0.00, 1.20),
    "retreat_bias": (0.00, 1.20),
    "heading_gain": (0.20, 3.00),
    "barrel_gain": (0.20, 3.50),
    "distance_hold_gain": (0.00, 2.50),
    "advance_speed": (0.05, 1.00),
    "retreat_speed": (-1.00, -0.05),
    "explore_speed": (0.00, 1.00),
    "explore_turn": (0.0, 45.0),
    "fire_alignment_deg": (2.0, 25.0),
    "fire_threshold": (0.10, 0.95),
    "scan_speed": (0.0, 70.0),
    "scan_arc": (10.0, 120.0),
}

DEFAULT_PARAMS: Dict[str, float] = {
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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_params(params: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key, (low, high) in PARAM_BOUNDS.items():
        out[key] = clamp(float(params[key]), low, high)
    return out


def random_params(rng: random.Random) -> Dict[str, float]:
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in PARAM_BOUNDS.items()}


def load_params_file(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    merged = {**DEFAULT_PARAMS, **raw}
    for key in PARAM_BOUNDS:
        if key not in merged:
            raise ValueError(f"Missing parameter: {key}")
    return clamp_params(merged)


def save_params_file(path: Path, params: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


@dataclass
class IndividualScore:
    params: Dict[str, float]
    score: float
    success_rate: float
    avg_ticks: float


class Evaluator:
    def __init__(self, args: argparse.Namespace, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.tmp_dir = run_dir / "tmp_params"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.base_eval_args = argparse.Namespace(
            team_a=args.team_a,
            team_b=args.team_b,
            base_port=args.base_port,
            agent_host=args.agent_host,
            agent_timeout=args.agent_timeout,
            sudden_death_tick=args.sudden_death_tick,
            sudden_death_damage=args.sudden_death_damage,
            log_level=args.log_level,
            restart_agents_per_game=True,
            agent_team1=str(Path(args.agent_script).resolve()),
            agent_team2=str(Path(args.opponent_script).resolve()),
            agent1_arg=[],
            agent2_arg=list(args.opponent_arg),
            eval_team=1,
            verbose_agents=args.verbose_agents,
            startup_timeout=args.startup_timeout,
            startup_poll_interval=args.startup_poll_interval,
            heartbeat_sec=args.heartbeat_sec,
            set_agent_names=args.set_agent_names,
        )

        tr.configure_engine(self.base_eval_args)
        self.weights = tr.FitnessWeights(
            win=args.w_win,
            loss=args.w_loss,
            draw=args.w_draw,
            kill=args.w_kill,
            damage=args.w_damage,
            alive=args.w_alive,
            enemy_alive=args.w_enemy_alive,
        )

    def evaluate(
        self,
        params: Dict[str, float],
        eval_seed_offset: int,
        eval_episodes: int,
        progress_label: str = "",
    ) -> IndividualScore:
        params = clamp_params(params)
        params_file = self.tmp_dir / f"params_{eval_seed_offset}_{random.randint(0, 10**9)}.json"
        save_params_file(params_file, params)

        args = copy.deepcopy(self.base_eval_args)
        args.agent1_arg = ["--params-file", str(params_file)]

        rows: List[Dict[str, Any]] = []
        eval_start = time.time()
        if progress_label:
            print(f"[{progress_label}] start eval ({eval_episodes} matches)", flush=True)
        for ep in range(eval_episodes):
            ep_start = time.time()
            map_name = self.args.maps[ep % len(self.args.maps)]
            seed = self.args.base_seed + eval_seed_offset + ep
            row = tr.run_single_match(
                episode_idx=ep,
                map_name=map_name,
                seed=seed,
                args=args,
                weights=self.weights,
            )
            rows.append(row)

            ep_elapsed = time.time() - ep_start
            eval_elapsed = time.time() - eval_start
            ep_done = ep + 1
            avg_ep = eval_elapsed / max(1, ep_done)
            eta_eval = (eval_episodes - ep_done) * avg_ep
            if progress_label:
                print(
                    f"[{progress_label}] match {ep_done}/{eval_episodes} "
                    f"done in {format_duration(ep_elapsed)} "
                    f"score={float(row.get('fitness', 0.0)):.3f} "
                    f"eta_eval={format_duration(eta_eval)}",
                    flush=True,
                )

        # Cleanup temporary params file after processes exit.
        try:
            params_file.unlink(missing_ok=True)
        except Exception:
            pass

        score = mean(float(r.get("fitness", -1_000_000.0)) for r in rows)
        success_rate = mean(1.0 if r.get("success") else 0.0 for r in rows)
        avg_ticks = mean(float(r.get("total_ticks", 0)) for r in rows)

        return IndividualScore(
            params=params,
            score=float(score),
            success_rate=float(success_rate),
            avg_ticks=float(avg_ticks),
        )


def tournament_select(
    scored: List[IndividualScore], k: int, rng: random.Random
) -> IndividualScore:
    sample = rng.sample(scored, min(k, len(scored)))
    return max(sample, key=lambda s: s.score)


def crossover(
    p1: Dict[str, float],
    p2: Dict[str, float],
    rng: random.Random,
    blend: bool,
) -> Dict[str, float]:
    child: Dict[str, float] = {}
    for key in PARAM_BOUNDS:
        if blend:
            alpha = rng.random()
            child[key] = alpha * p1[key] + (1.0 - alpha) * p2[key]
        else:
            child[key] = p1[key] if rng.random() < 0.5 else p2[key]
    return clamp_params(child)


def mutate(
    params: Dict[str, float],
    rng: random.Random,
    mutation_rate: float,
    mutation_sigma: float,
) -> Dict[str, float]:
    out = dict(params)
    for key, (low, high) in PARAM_BOUNDS.items():
        if rng.random() < mutation_rate:
            span = high - low
            out[key] = out[key] + rng.gauss(0.0, mutation_sigma * span)
    return clamp_params(out)


def write_generation_files(
    run_dir: Path,
    generation: int,
    scored: List[IndividualScore],
) -> None:
    path = run_dir / f"generation_{generation:03d}_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "score", "success_rate", "avg_ticks", *PARAM_BOUNDS.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        ranked = sorted(scored, key=lambda s: s.score, reverse=True)
        for rank, ind in enumerate(ranked, start=1):
            row = {
                "rank": rank,
                "score": ind.score,
                "success_rate": ind.success_rate,
                "avg_ticks": ind.avg_ticks,
            }
            row.update(ind.params)
            writer.writerow(row)


def run_ga(
    evaluator: Evaluator,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    population_size = args.population
    elite_count = min(args.elite_count, population_size)
    if elite_count < 1:
        elite_count = 1

    population: List[Dict[str, float]] = []
    if args.init_params:
        population.append(load_params_file(Path(args.init_params).resolve()))
    else:
        population.append(dict(DEFAULT_PARAMS))

    while len(population) < population_size:
        population.append(random_params(rng))

    history: List[Dict[str, Any]] = []
    best_global = IndividualScore(params=dict(DEFAULT_PARAMS), score=-1e18, success_rate=0.0, avg_ticks=0.0)
    total_evals = args.generations * population_size
    completed_evals = 0
    total_eval_time = 0.0

    for gen in range(args.generations):
        scored: List[IndividualScore] = []
        seed_offset = gen * 10000
        gen_start = time.time()
        for idx, individual in enumerate(population):
            eval_start = time.time()
            ind_score = evaluator.evaluate(
                params=individual,
                eval_seed_offset=seed_offset + idx * 100,
                eval_episodes=args.eval_episodes,
                progress_label=f"GA gen {gen + 1}/{args.generations} ind {idx + 1}/{population_size}",
            )
            eval_time = time.time() - eval_start
            scored.append(ind_score)
            completed_evals += 1
            total_eval_time += eval_time

            avg_eval_time = total_eval_time / max(1, completed_evals)
            remaining_total = (total_evals - completed_evals) * avg_eval_time

            gen_completed = idx + 1
            gen_elapsed = time.time() - gen_start
            gen_avg_eval = gen_elapsed / max(1, gen_completed)
            gen_remaining = (population_size - gen_completed) * gen_avg_eval

            print(
                f"[GA gen {gen + 1}/{args.generations}] "
                f"ind {idx + 1}/{population_size} "
                f"score={ind_score.score:.3f} success={ind_score.success_rate:.2f} "
                f"eval={eval_time:.1f}s eta_gen={format_duration(gen_remaining)} "
                f"eta_total={format_duration(remaining_total)}"
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        write_generation_files(evaluator.run_dir, gen, scored)

        best_gen = scored[0]
        if best_gen.score > best_global.score:
            best_global = best_gen
            save_params_file(evaluator.run_dir / "best_params.json", best_global.params)

        mean_score = mean(s.score for s in scored)
        history.append(
            {
                "generation": gen,
                "best_score": best_gen.score,
                "mean_score": mean_score,
                "worst_score": scored[-1].score,
                "best_success_rate": best_gen.success_rate,
                "best_avg_ticks": best_gen.avg_ticks,
            }
        )
        print(
            f"[GA gen {gen + 1}] best={best_gen.score:.3f} "
            f"mean={mean_score:.3f} global_best={best_global.score:.3f}"
        )
        print(
            f"[GA gen {gen + 1}] elapsed={format_duration(time.time() - gen_start)} "
            f"progress={completed_evals}/{total_evals}"
        )

        elites = [copy.deepcopy(s.params) for s in scored[:elite_count]]
        next_pop: List[Dict[str, float]] = elites[:]

        while len(next_pop) < population_size:
            p1 = tournament_select(scored, args.tournament_k, rng).params
            p2 = tournament_select(scored, args.tournament_k, rng).params
            do_blend = rng.random() < args.crossover_rate
            child = crossover(p1, p2, rng, blend=do_blend)
            child = mutate(
                child,
                rng=rng,
                mutation_rate=args.mutation_rate,
                mutation_sigma=args.mutation_sigma,
            )
            next_pop.append(child)

        population = next_pop

    return best_global.params, history


def run_pso(
    evaluator: Evaluator,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    n = args.population
    keys = list(PARAM_BOUNDS.keys())

    particles: List[Dict[str, float]] = []
    if args.init_params:
        particles.append(load_params_file(Path(args.init_params).resolve()))
    else:
        particles.append(dict(DEFAULT_PARAMS))
    while len(particles) < n:
        particles.append(random_params(rng))

    velocities: List[Dict[str, float]] = []
    vmax: Dict[str, float] = {
        k: (PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0]) * args.pso_vmax_ratio for k in keys
    }
    for _ in range(n):
        velocities.append({k: rng.uniform(-vmax[k], vmax[k]) for k in keys})

    pbest = [dict(p) for p in particles]
    pbest_score = [-1e18 for _ in range(n)]

    gbest = dict(particles[0])
    gbest_score = -1e18

    history: List[Dict[str, Any]] = []
    total_evals = args.generations * n
    completed_evals = 0
    total_eval_time = 0.0

    for gen in range(args.generations):
        scored: List[IndividualScore] = []
        seed_offset = gen * 10000
        gen_start = time.time()
        for i in range(n):
            eval_start = time.time()
            ind_score = evaluator.evaluate(
                params=particles[i],
                eval_seed_offset=seed_offset + i * 100,
                eval_episodes=args.eval_episodes,
                progress_label=f"PSO gen {gen + 1}/{args.generations} particle {i + 1}/{n}",
            )
            eval_time = time.time() - eval_start
            scored.append(ind_score)
            completed_evals += 1
            total_eval_time += eval_time

            if ind_score.score > pbest_score[i]:
                pbest_score[i] = ind_score.score
                pbest[i] = dict(ind_score.params)
            if ind_score.score > gbest_score:
                gbest_score = ind_score.score
                gbest = dict(ind_score.params)
                save_params_file(evaluator.run_dir / "best_params.json", gbest)

            avg_eval_time = total_eval_time / max(1, completed_evals)
            remaining_total = (total_evals - completed_evals) * avg_eval_time

            gen_completed = i + 1
            gen_elapsed = time.time() - gen_start
            gen_avg_eval = gen_elapsed / max(1, gen_completed)
            gen_remaining = (n - gen_completed) * gen_avg_eval

            print(
                f"[PSO gen {gen + 1}/{args.generations}] "
                f"particle {i + 1}/{n} score={ind_score.score:.3f} "
                f"eval={eval_time:.1f}s eta_gen={format_duration(gen_remaining)} "
                f"eta_total={format_duration(remaining_total)}"
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        write_generation_files(evaluator.run_dir, gen, scored)

        best_gen = scored[0]
        mean_score = mean(s.score for s in scored)
        history.append(
            {
                "generation": gen,
                "best_score": best_gen.score,
                "mean_score": mean_score,
                "worst_score": scored[-1].score,
                "best_success_rate": best_gen.success_rate,
                "best_avg_ticks": best_gen.avg_ticks,
            }
        )
        print(
            f"[PSO gen {gen + 1}] best={best_gen.score:.3f} "
            f"mean={mean_score:.3f} global_best={gbest_score:.3f}"
        )
        print(
            f"[PSO gen {gen + 1}] elapsed={format_duration(time.time() - gen_start)} "
            f"progress={completed_evals}/{total_evals}"
        )

        # Position update.
        for i in range(n):
            for k in keys:
                r1 = rng.random()
                r2 = rng.random()
                v = (
                    args.pso_w * velocities[i][k]
                    + args.pso_c1 * r1 * (pbest[i][k] - particles[i][k])
                    + args.pso_c2 * r2 * (gbest[k] - particles[i][k])
                )
                v = clamp(v, -vmax[k], vmax[k])
                velocities[i][k] = v
                particles[i][k] = particles[i][k] + v
            particles[i] = clamp_params(particles[i])

    return gbest, history


def write_history_csv(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    path = run_dir / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        if not history:
            return
        fieldnames = list(history[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def final_evaluation(
    evaluator: Evaluator,
    best_params: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    score = evaluator.evaluate(
        params=best_params,
        eval_seed_offset=999_000_000,
        eval_episodes=args.final_episodes,
    )
    return {
        "final_avg_score": score.score,
        "final_success_rate": score.success_rate,
        "final_avg_ticks": score.avg_ticks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fuzzy agent with GA or PSO.")

    default_agent = str((Path(__file__).parent / "fuzzy_ga_agent.py").resolve())
    default_opponent = str((Path(__file__).parent / "Agent_1.py").resolve())
    default_output = str((Path(__file__).resolve().parent.parent / "logs").resolve())

    parser.add_argument("--method", choices=["ga", "pso"], default="ga")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--final-episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1234)

    parser.add_argument(
        "--maps",
        nargs="+",
        default=["advanced_road_trees.csv", "semi-open.csv", "open.csv"],
    )

    parser.add_argument("--agent-script", type=str, default=default_agent)
    parser.add_argument("--opponent-script", type=str, default=default_opponent)
    parser.add_argument("--opponent-arg", action="append", default=[])
    parser.add_argument("--init-params", type=str, default=None)

    parser.add_argument("--team-a", type=int, default=2)
    parser.add_argument("--team-b", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=8001)
    parser.add_argument("--agent-host", type=str, default="127.0.0.1")
    parser.add_argument("--agent-timeout", type=float, default=1.0)
    parser.add_argument("--sudden-death-tick", type=int, default=2500)
    parser.add_argument("--sudden-death-damage", type=int, default=-2)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    parser.add_argument("--startup-poll-interval", type=float, default=0.2)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="ERROR")
    parser.add_argument("--verbose-agents", action="store_true")
    parser.add_argument("--set-agent-names", action="store_true")

    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--run-name", type=str, default="fuzzy_opt")

    # Fitness weights passed to train_runner fitness.
    parser.add_argument("--w-win", type=float, default=200.0)
    parser.add_argument("--w-loss", type=float, default=-200.0)
    parser.add_argument("--w-draw", type=float, default=0.0)
    parser.add_argument("--w-kill", type=float, default=30.0)
    parser.add_argument("--w-damage", type=float, default=0.25)
    parser.add_argument("--w-alive", type=float, default=10.0)
    parser.add_argument("--w-enemy-alive", type=float, default=-10.0)

    # GA-specific.
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--mutation-sigma", type=float, default=0.08)

    # PSO-specific.
    parser.add_argument("--pso-w", type=float, default=0.72)
    parser.add_argument("--pso-c1", type=float, default=1.5)
    parser.add_argument("--pso-c2", type=float, default=1.5)
    parser.add_argument("--pso-vmax-ratio", type=float, default=0.18)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.base_seed)

    output_root = Path(args.output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{args.run_name}_{args.method}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    evaluator = Evaluator(args=args, run_dir=run_dir)

    if args.method == "ga":
        best_params, history = run_ga(evaluator=evaluator, args=args, rng=rng)
    else:
        best_params, history = run_pso(evaluator=evaluator, args=args, rng=rng)

    write_history_csv(run_dir, history)
    save_params_file(run_dir / "best_params.json", best_params)

    final_stats = final_evaluation(evaluator=evaluator, best_params=best_params, args=args)
    with (run_dir / "final_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    print("\nOptimization finished.")
    print(f"Run dir: {run_dir}")
    print(f"Best params: {run_dir / 'best_params.json'}")
    print(f"Final eval: {final_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
