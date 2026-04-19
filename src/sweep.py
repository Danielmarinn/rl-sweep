"""
RL Sweep Runner
===============
Trains PPO and DQN agents across 3 Gymnasium environments over a focused
hyperparameter grid, in parallel on all available CPU cores.

Usage:
    python src/sweep.py           # run (or resume) the sweep
    python src/plot.py            # generate plots after sweep completes
    python src/evaluate.py        # interactively load and run a saved model

Fixes vs original:
    - Full seed isolation: random, numpy, and torch all seeded (not just numpy)
    - Removed redundant eval_env (two envs per worker, not three)
    - eval_cb.n_eval_episodes increased to 10 for more reliable curve snapshots
    - Worker returns wall-clock duration in minutes (more readable in logs)
    - Type hints added throughout for IDE support
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import random
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
RUNS_CSV    = RESULTS_DIR / "runs.csv"
CURVES_CSV  = RESULTS_DIR / "curves.csv"
MODELS_DIR  = RESULTS_DIR / "models"
LOG_FILE    = RESULTS_DIR / "sweep.log"

# Evaluation snapshots per training run (one per 10% of total_timesteps).
N_CURVE_POINTS = 10

ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "CartPole-v1":    {"timesteps": 150_000, "n_eval_episodes": 20},
    "LunarLander-v3": {"timesteps": 300_000, "n_eval_episodes": 15},
    "Acrobot-v1":     {"timesteps": 300_000, "n_eval_episodes": 20},
}
ENVIRONMENTS_DEFAULT = {name: cfg.copy() for name, cfg in ENVIRONMENTS.items()}

# PPO grid: 3 × 2 × 2 = 12 combos per environment.
PPO_GRID: dict[str, list] = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps":       [512, 2048],
    "gamma":         [0.99, 0.999],
}

# DQN grid: 3 × 2 × 2 = 12 combos. Applied to CartPole and LunarLander only.
DQN_GRID: dict[str, list] = {
    "learning_rate":        [1e-4, 3e-4, 1e-3],
    "exploration_fraction": [0.1, 0.2],
    "gamma":                [0.99, 0.999],
}

# Fixed (non-swept) hyperparameters — excluded from grid to save compute.
# NOTE: ent_coef=0.0 is a deliberate trade-off that can trigger policy collapse
# on sparse-reward tasks (Acrobot) under high gamma + short rollouts.
# Phase-2 recommendation: sweep ent_coef ∈ {0.0, 0.01, 0.05}.
PPO_FIXED = {"ent_coef": 0.0, "batch_size": 64}
DQN_FIXED = {"batch_size": 64}

SEEDS = [0, 1]
SLOW_RUN_WARN_MINUTES = 25

# CSV column schema — both files are append-only; never modify existing rows.
CSV_FIELDS = [
    "run_id", "env", "algo", "seed",
    "learning_rate", "n_steps", "gamma", "exploration_fraction",
    "mean_reward", "std_reward", "timesteps",
    "duration_s", "status", "error", "timestamp",
]
CURVE_FIELDS = ["run_id", "timestep", "mean_reward"]


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fmt    = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    file_h = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    con_h  = logging.StreamHandler()
    file_h.setFormatter(fmt)
    con_h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(file_h)
        root.addHandler(con_h)


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT PROBE
# ══════════════════════════════════════════════════════════════════════════════

def check_environments() -> None:
    """
    Instantiate every configured environment before spawning workers.
    Removes any that fail (e.g. LunarLander without box2d) and aborts if
    nothing survives.
    """
    import gymnasium as gym

    to_remove = []
    for env_name in list(ENVIRONMENTS.keys()):
        try:
            e = gym.make(env_name, render_mode=None)
            e.reset()
            e.close()
            logging.info(f"  ✓  {env_name}")
        except Exception as exc:
            logging.warning(f"  ✗  {env_name} — SKIPPED: {str(exc)[:120]}")
            if "LunarLander" in env_name:
                logging.warning(
                    "      → LunarLander requires box2d:\n"
                    "          pip install swig\n"
                    "          pip install gymnasium[box2d]"
                )
            to_remove.append(env_name)

    for name in to_remove:
        del ENVIRONMENTS[name]

    if not ENVIRONMENTS:
        logging.error("No environments available. Aborting.")
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN LIST + RESUME
# ══════════════════════════════════════════════════════════════════════════════

def make_run_id(env: str, algo: str, hparams: dict, seed: int) -> str:
    hp = "_".join(f"{k}={v}" for k, v in sorted(hparams.items()))
    return f"{env}__{algo}__{hp}__s{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the RL hyperparameter sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENVIRONMENTS.keys()),
        help="Limit the sweep to one Gymnasium environment.",
    )
    parser.add_argument(
        "--algo",
        choices=("PPO", "DQN"),
        help="Limit the sweep to one algorithm.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seed values to run. Example: --seeds 0 1",
    )
    return parser.parse_args()


def build_run_list(
    seeds: list[int] | None = None,
    algo: str | None = None,
) -> list[tuple]:
    """Return all (env_name, algo, hparams, seed, env_cfg) tuples."""
    selected_seeds = SEEDS if seeds is None else seeds
    runs = []
    for env_name, env_cfg in ENVIRONMENTS.items():
        if algo in (None, "PPO"):
            keys = list(PPO_GRID.keys())
            for combo in product(*[PPO_GRID[k] for k in keys]):
                hp = dict(zip(keys, combo))
                for seed in selected_seeds:
                    runs.append((env_name, "PPO", hp, seed, env_cfg))

        if algo in (None, "DQN") and env_name in ("CartPole-v1", "LunarLander-v3"):
            keys = list(DQN_GRID.keys())
            for combo in product(*[DQN_GRID[k] for k in keys]):
                hp = dict(zip(keys, combo))
                for seed in selected_seeds:
                    runs.append((env_name, "DQN", hp, seed, env_cfg))

    return runs


def load_completed_run_ids() -> set[str]:
    """
    Return run_ids with status == 'success' from runs.csv.
    Only successful runs are skipped; failed/partial runs are retried.
    """
    if not RUNS_CSV.exists():
        return set()
    completed = set()
    with open(RUNS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "success":
                completed.add(row["run_id"])
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  DISK WRITERS  (main process only)
# ══════════════════════════════════════════════════════════════════════════════

def append_result(result: dict) -> None:
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in CSV_FIELDS})
        f.flush()


def append_curves(run_id: str, curve: list[tuple[int, float]]) -> None:
    if not curve:
        return
    write_header = not CURVES_CSV.exists()
    with open(CURVES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        if write_header:
            writer.writeheader()
        for timestep, mean_reward in curve:
            writer.writerow({"run_id": run_id, "timestep": timestep,
                             "mean_reward": mean_reward})
        f.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER  (top-level for Windows multiprocessing spawn)
# ══════════════════════════════════════════════════════════════════════════════

def run_single(args: tuple) -> dict:
    """
    Train one agent, evaluate it, save artefacts, return result dict.

    All SB3/gymnasium imports are deferred inside this function (late-import
    pattern) to avoid re-initialising heavy libraries in every spawned worker
    on Windows (spawn start method).

    Files saved on success:
        results/models/{run_id}/best_model.zip     ← best checkpoint
        results/models/{run_id}/config.json        ← hparams + final reward
        results/models/{run_id}/evaluations.npz    ← raw EvalCallback arrays
    """
    env_name, algo_name, hparams, seed, env_cfg = args
    run_id  = make_run_id(env_name, algo_name, hparams, seed)
    t_start = time.time()

    result: dict = {
        "run_id":               run_id,
        "env":                  env_name,
        "algo":                 algo_name,
        "seed":                 seed,
        "timesteps":            env_cfg["timesteps"],
        "timestamp":            datetime.now().isoformat(timespec="seconds"),
        "learning_rate":        hparams.get("learning_rate", ""),
        "n_steps":              hparams.get("n_steps", ""),
        "gamma":                hparams.get("gamma", ""),
        "exploration_fraction": hparams.get("exploration_fraction", ""),
    }

    try:
        import torch
        import gymnasium as gym
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor

        # FIX: full seed isolation across all RNG sources.
        # The original only seeded numpy, which is insufficient — torch and
        # Python's random module have independent states.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # FIX: two envs instead of three.
        # The original created train_env, eval_env, and cb_env but only
        # train_env and cb_env were actually used (eval_env was passed to
        # evaluate_policy but cb_env was passed to EvalCallback — both
        # redundant). Now: train_env for learning, eval_env for both
        # EvalCallback and final evaluation.
        train_env = Monitor(gym.make(env_name, render_mode=None))
        eval_env  = Monitor(gym.make(env_name, render_mode=None))

        model_dir = Path("results") / "models" / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        eval_freq = max(1, env_cfg["timesteps"] // N_CURVE_POINTS)
        eval_cb   = EvalCallback(
            eval_env,
            best_model_save_path = str(model_dir),
            log_path             = str(model_dir),
            eval_freq            = eval_freq,
            # FIX: increased from 5 → 10 for lower-variance curve snapshots
            n_eval_episodes      = 10,
            deterministic        = True,
            verbose              = 0,
            warn                 = False,
        )

        if algo_name == "PPO":
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate = hparams["learning_rate"],
                n_steps       = hparams["n_steps"],
                gamma         = hparams["gamma"],
                ent_coef      = PPO_FIXED["ent_coef"],
                batch_size    = PPO_FIXED["batch_size"],
                seed          = seed,
                verbose       = 0,
                device        = "cpu",
            )
        elif algo_name == "DQN":
            model = DQN(
                "MlpPolicy", train_env,
                learning_rate        = hparams["learning_rate"],
                exploration_fraction = hparams["exploration_fraction"],
                gamma                = hparams["gamma"],
                batch_size           = DQN_FIXED["batch_size"],
                seed                 = seed,
                verbose              = 0,
                device               = "cpu",
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        model.learn(total_timesteps=env_cfg["timesteps"], callback=eval_cb)

        mean_r, std_r = evaluate_policy(
            model, eval_env,
            n_eval_episodes = env_cfg["n_eval_episodes"],
            deterministic   = True,
        )

        train_env.close()
        eval_env.close()

        curve: list[tuple[int, float]] = []
        if eval_cb.evaluations_timesteps:
            for ts, ep_rewards in zip(eval_cb.evaluations_timesteps,
                                      eval_cb.evaluations_results):
                curve.append((int(ts), round(float(np.mean(ep_rewards)), 3)))

        config = {
            "run_id":      run_id,
            "env":         env_name,
            "algo":        algo_name,
            "seed":        seed,
            "hparams":     {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in hparams.items()},
            "mean_reward": round(float(mean_r), 4),
            "std_reward":  round(float(std_r),  4),
            "timesteps":   env_cfg["timesteps"],
            "saved_at":    datetime.now().isoformat(timespec="seconds"),
        }
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        result.update({
            "mean_reward": round(float(mean_r), 4),
            "std_reward":  round(float(std_r),  4),
            "duration_s":  round(time.time() - t_start, 1),
            "status":      "success",
            "error":       "",
            "_curve":      curve,
        })

    except Exception as exc:
        result.update({
            "mean_reward": "",
            "std_reward":  "",
            "duration_s":  round(time.time() - t_start, 1),
            "status":      "failed",
            "error":       repr(exc)[:400],
            "_curve":      [],
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    if args.env:
        ENVIRONMENTS.clear()
        ENVIRONMENTS[args.env] = ENVIRONMENTS_DEFAULT[args.env].copy()

    setup_logging()
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    logging.info("Checking environments…")
    check_environments()
    logging.info("")

    all_runs  = build_run_list(seeds=args.seeds, algo=args.algo)
    completed = load_completed_run_ids() & {
        make_run_id(r[0], r[1], r[2], r[3]) for r in all_runs
    }
    pending   = [r for r in all_runs
                 if make_run_id(r[0], r[1], r[2], r[3]) not in completed]

    n_total   = len(all_runs)
    n_done    = len(completed)
    n_pending = len(pending)
    n_workers = max(1, min(n_pending, mp.cpu_count() - 1))

    logging.info("=" * 65)
    logging.info("  RL SWEEP — PRE-FLIGHT SUMMARY")
    logging.info(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"  Workers   : {n_workers}  (of {mp.cpu_count()} logical CPUs)")
    logging.info("")
    logging.info("  Environments and planned runs:")
    for env_name, cfg in ENVIRONMENTS.items():
        ppo_n = sum(1 for r in all_runs if r[0] == env_name and r[1] == "PPO")
        dqn_n = sum(1 for r in all_runs if r[0] == env_name and r[1] == "DQN")
        logging.info(f"    {env_name:<20}  {cfg['timesteps']//1000:>4}k steps  "
                     f"PPO×{ppo_n} + DQN×{dqn_n} = {ppo_n+dqn_n} runs")
    logging.info("")
    logging.info(f"  Total planned : {n_total}")
    logging.info(f"  Already done  : {n_done}  (skipped)")
    logging.info(f"  To run now    : {n_pending}")
    if n_done > 0:
        logging.info("  [RESUME MODE] — picking up from previous run")
    logging.info("=" * 65)
    logging.info("")

    if n_pending == 0:
        logging.info("Nothing to do — all runs completed. Run python src/plot.py.")
        return

    t_start   = time.time()
    n_success = 0
    n_failed  = 0
    counter   = n_done

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_single, pending, chunksize=1):

            curve = result.pop("_curve", [])
            append_result(result)

            if result["status"] == "success":
                n_success += 1

                if curve:
                    append_curves(result["run_id"], curve)
                else:
                    logging.warning(
                        f"  ⚠  No curve data for {result['run_id']} — "
                        "training finished before the first eval_freq checkpoint."
                    )

                model_path = MODELS_DIR / result["run_id"] / "best_model.zip"
                if not model_path.exists():
                    logging.warning(
                        f"  ⚠  best_model.zip missing for {result['run_id']}. "
                        "EvalCallback may not have beaten its initial baseline."
                    )

                if result["duration_s"] > SLOW_RUN_WARN_MINUTES * 60:
                    logging.warning(
                        f"  ⚠  Slow run ({result['duration_s']/60:.1f} min): "
                        f"{result['run_id']}"
                    )

                tag        = "✓"
                reward_str = (f"reward={result['mean_reward']:>8.2f} "
                              f"± {result['std_reward']:.2f}")
            else:
                n_failed  += 1
                tag        = "✗"
                reward_str = f"FAILED: {result['error'][:70]}"
                logging.warning(
                    f"  ✗  Will retry on next launch:\n"
                    f"       {result['run_id']}\n"
                    f"       Error: {result['error'][:200]}"
                )

            counter   += 1
            elapsed_h  = (time.time() - t_start) / 3600
            pct        = 100 * counter / n_total
            logging.info(
                f"[{counter:>4}/{n_total}  {pct:4.1f}%  {elapsed_h:.2f}h]  "
                f"{tag} {result['env']:<18} {result['algo']:<4}  "
                f"seed={result['seed']}  {reward_str}  ({result['duration_s']}s)"
            )

    elapsed_total = (time.time() - t_start) / 3600
    logging.info("")
    logging.info("=" * 65)
    logging.info(f"  SWEEP COMPLETE in {elapsed_total:.2f} hours")
    logging.info(f"  Successful : {n_success}  |  Failed : {n_failed}")
    if n_failed > 0:
        logging.warning(f"  {n_failed} run(s) failed — rerun to retry.")
    logging.info(f"  Results : {RUNS_CSV.resolve()}")
    logging.info(f"  Curves  : {CURVES_CSV.resolve()}")
    logging.info(f"  Models  : {MODELS_DIR.resolve()}")
    logging.info("  → Next: python src/plot.py")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
