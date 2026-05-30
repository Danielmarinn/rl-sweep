"""
RL Sweep Runner
===============
Trains PPO and DQN agents across Gymnasium environments over a factorial
hyperparameter grid, in parallel on all available CPU cores.  Results are
appended to results/runs.csv and results/curves.csv after every worker
returns, enabling crash-safe resume.

Usage
-----
    rl-sweep run                   # run (or resume) the full sweep
    rl-sweep run --env CartPole-v1 # limit to one environment
    rl-sweep run --algo PPO        # limit to one algorithm
    python -m rlsweep.sweep        # equivalent direct invocation
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
from typing import Any

import numpy as np

from rlsweep import config

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════


def setup_logging() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    file_h = logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")
    con_h = logging.StreamHandler()
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


def check_environments(environments: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Instantiate every environment in the dict before spawning workers.
    Returns a filtered dict containing only the survivors.
    Aborts if no environment is available.
    """
    import gymnasium as gym

    survivors: dict[str, dict[str, Any]] = {}
    for env_name, env_cfg in environments.items():
        try:
            e = gym.make(env_name, render_mode=None)
            e.reset()
            e.close()
            logging.info(f"  ok  {env_name}")
            survivors[env_name] = env_cfg
        except Exception as exc:
            logging.warning(f"  !!  {env_name} — SKIPPED: {str(exc)[:120]}")
            if "LunarLander" in env_name:
                logging.warning(
                    "      -> LunarLander requires box2d:\n"
                    "          pip install swig\n"
                    "          pip install gymnasium[box2d]"
                )

    if not survivors:
        logging.error("No environments available. Aborting.")
        raise SystemExit(1)

    return survivors


# ══════════════════════════════════════════════════════════════════════════════
#  RUN LIST + RESUME
# ══════════════════════════════════════════════════════════════════════════════


def make_run_id(env: str, algo: str, hparams: dict, seed: int) -> str:
    hp = "_".join(f"{k}={v}" for k, v in sorted(hparams.items()))
    return f"{env}__{algo}__{hp}__s{seed}"


def build_run_list(
    environments: dict[str, dict[str, Any]],
    seeds: list[int] | None = None,
    algo: str | None = None,
) -> list[tuple]:
    """Return all (env_name, algo, hparams, seed, env_cfg) tuples."""
    selected_seeds = config.SEEDS if seeds is None else seeds
    runs = []
    for env_name, env_cfg in environments.items():
        if algo in (None, "PPO"):
            keys = list(config.PPO_GRID.keys())
            for combo in product(*[config.PPO_GRID[k] for k in keys]):
                hp = dict(zip(keys, combo, strict=True))
                for seed in selected_seeds:
                    runs.append((env_name, "PPO", hp, seed, env_cfg))

        if algo in (None, "DQN") and env_name in ("CartPole-v1", "LunarLander-v3"):
            keys = list(config.DQN_GRID.keys())
            for combo in product(*[config.DQN_GRID[k] for k in keys]):
                hp = dict(zip(keys, combo, strict=True))
                for seed in selected_seeds:
                    runs.append((env_name, "DQN", hp, seed, env_cfg))

    return runs


def load_completed_run_ids() -> set[str]:
    """
    Return run_ids with status == 'success' from runs.csv.
    Only successful runs are skipped; failed or partial runs are retried.
    """
    if not config.RUNS_CSV.exists():
        return set()
    completed = set()
    with open(config.RUNS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "success":
                completed.add(row["run_id"])
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  DISK WRITERS  (main process only)
# ══════════════════════════════════════════════════════════════════════════════


def append_result(result: dict) -> None:
    write_header = not config.RUNS_CSV.exists()
    with open(config.RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in config.CSV_FIELDS})
        f.flush()


def append_curves(run_id: str, curve: list[tuple[int, float]]) -> None:
    if not curve:
        return
    write_header = not config.CURVES_CSV.exists()
    with open(config.CURVES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.CURVE_FIELDS)
        if write_header:
            writer.writeheader()
        for timestep, mean_reward in curve:
            writer.writerow({"run_id": run_id, "timestep": timestep, "mean_reward": mean_reward})
        f.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER  (top-level for multiprocessing spawn)
# ══════════════════════════════════════════════════════════════════════════════


def run_single(args: tuple) -> dict:
    """
    Train one agent, evaluate it, save artefacts, return result dict.

    Heavy SB3/gym imports are deferred inside this function so they are not
    re-initialised in every spawned worker on Windows (spawn start method).

    Files saved on success:
        results/models/{run_id}/best_model.zip     <- best checkpoint
        results/models/{run_id}/config.json        <- hparams + final reward
        results/models/{run_id}/evaluations.npz    <- raw EvalCallback arrays
    """
    env_name, algo_name, hparams, seed, env_cfg = args
    run_id = make_run_id(env_name, algo_name, hparams, seed)
    t_start = time.time()

    result: dict = {
        "run_id": run_id,
        "env": env_name,
        "algo": algo_name,
        "seed": seed,
        "timesteps": env_cfg["timesteps"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "learning_rate": hparams.get("learning_rate", ""),
        "n_steps": hparams.get("n_steps", ""),
        "gamma": hparams.get("gamma", ""),
        "exploration_fraction": hparams.get("exploration_fraction", ""),
    }

    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor

        # Seed every RNG source for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_env = Monitor(gym.make(env_name, render_mode=None))
        eval_env = Monitor(gym.make(env_name, render_mode=None))

        model_dir = config.MODELS_DIR / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        eval_freq = max(1, env_cfg["timesteps"] // config.N_CURVE_POINTS)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(model_dir),
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            verbose=0,
            warn=False,
        )

        if algo_name == "PPO":
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=hparams["learning_rate"],
                n_steps=hparams["n_steps"],
                gamma=hparams["gamma"],
                ent_coef=config.PPO_FIXED["ent_coef"],
                batch_size=config.PPO_FIXED["batch_size"],
                seed=seed,
                verbose=0,
                device="cpu",
            )
        elif algo_name == "DQN":
            model = DQN(
                "MlpPolicy",
                train_env,
                learning_rate=hparams["learning_rate"],
                exploration_fraction=hparams["exploration_fraction"],
                gamma=hparams["gamma"],
                batch_size=config.DQN_FIXED["batch_size"],
                seed=seed,
                verbose=0,
                device="cpu",
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        model.learn(total_timesteps=env_cfg["timesteps"], callback=eval_cb)

        mean_r, std_r = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=env_cfg["n_eval_episodes"],
            deterministic=True,
        )

        train_env.close()
        eval_env.close()

        curve: list[tuple[int, float]] = []
        if eval_cb.evaluations_timesteps:
            for ts, ep_rewards in zip(
                eval_cb.evaluations_timesteps, eval_cb.evaluations_results, strict=True
            ):
                curve.append((int(ts), round(float(np.mean(ep_rewards)), 3)))

        run_config = {
            "run_id": run_id,
            "env": env_name,
            "algo": algo_name,
            "seed": seed,
            "hparams": {
                k: float(v) if isinstance(v, (int, float)) else v for k, v in hparams.items()
            },
            "mean_reward": round(float(mean_r), 4),
            "std_reward": round(float(std_r), 4),
            "timesteps": env_cfg["timesteps"],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        result.update(
            {
                "mean_reward": round(float(mean_r), 4),
                "std_reward": round(float(std_r), 4),
                "duration_s": round(time.time() - t_start, 1),
                "status": "success",
                "error": "",
                "_curve": curve,
            }
        )

    except Exception as exc:
        result.update(
            {
                "mean_reward": "",
                "std_reward": "",
                "duration_s": round(time.time() - t_start, 1),
                "status": "failed",
                "error": repr(exc)[:400],
                "_curve": [],
            }
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the RL hyperparameter sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=sorted(config.ENVIRONMENTS.keys()),
        help="Limit the sweep to one Gymnasium environment.",
    )
    parser.add_argument(
        "--algo",
        choices=("PPO", "DQN"),
        help="Limit to one algorithm.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=config.SEEDS,
        help="Seed values to run. Example: --seeds 0 1",
    )
    args = parser.parse_args(argv)

    # Build a local environments dict; do not mutate config globals.
    if args.env:
        environments = {args.env: config.ENVIRONMENTS[args.env]}
    else:
        environments = dict(config.ENVIRONMENTS)

    setup_logging()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)

    logging.info("Checking environments...")
    environments = check_environments(environments)
    logging.info("")

    all_runs = build_run_list(environments, seeds=args.seeds, algo=args.algo)
    completed = load_completed_run_ids() & {make_run_id(r[0], r[1], r[2], r[3]) for r in all_runs}
    pending = [r for r in all_runs if make_run_id(r[0], r[1], r[2], r[3]) not in completed]

    n_total = len(all_runs)
    n_done = len(completed)
    n_pending = len(pending)
    n_workers = max(1, min(n_pending, mp.cpu_count() - 1))

    logging.info("=" * 65)
    logging.info("  RL SWEEP — PRE-FLIGHT SUMMARY")
    logging.info(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"  Workers   : {n_workers}  (of {mp.cpu_count()} logical CPUs)")
    logging.info("")
    logging.info("  Environments and planned runs:")
    for env_name, cfg in environments.items():
        ppo_n = sum(1 for r in all_runs if r[0] == env_name and r[1] == "PPO")
        dqn_n = sum(1 for r in all_runs if r[0] == env_name and r[1] == "DQN")
        logging.info(
            f"    {env_name:<20}  {cfg['timesteps'] // 1000:>4}k steps  "
            f"PPO x{ppo_n} + DQN x{dqn_n} = {ppo_n + dqn_n} runs"
        )
    logging.info("")
    logging.info(f"  Total planned : {n_total}")
    logging.info(f"  Already done  : {n_done}  (skipped)")
    logging.info(f"  To run now    : {n_pending}")
    if n_done > 0:
        logging.info("  [RESUME MODE] — picking up from previous run")
    logging.info("=" * 65)
    logging.info("")

    if n_pending == 0:
        logging.info("Nothing to do — all runs completed. Run: rl-sweep plot")
        return

    t_start = time.time()
    n_success = 0
    n_failed = 0
    counter = n_done

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
                        f"  No curve data for {result['run_id']} — "
                        "training finished before the first eval_freq checkpoint."
                    )

                model_path = config.MODELS_DIR / result["run_id"] / "best_model.zip"
                if not model_path.exists():
                    logging.warning(
                        f"  best_model.zip missing for {result['run_id']}. "
                        "EvalCallback may not have beaten its initial baseline."
                    )

                if result["duration_s"] > config.SLOW_RUN_WARN_MINUTES * 60:
                    logging.warning(
                        f"  Slow run ({result['duration_s'] / 60:.1f} min): {result['run_id']}"
                    )

                tag = "ok"
                reward_str = f"reward={result['mean_reward']:>8.2f} +/- {result['std_reward']:.2f}"
            else:
                n_failed += 1
                tag = "!!"
                reward_str = f"FAILED: {result['error'][:70]}"
                logging.warning(
                    f"  !!  Will retry on next launch:\n"
                    f"       {result['run_id']}\n"
                    f"       Error: {result['error'][:200]}"
                )

            counter += 1
            elapsed_h = (time.time() - t_start) / 3600
            pct = 100 * counter / n_total
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
    logging.info(f"  Results : {config.RUNS_CSV.resolve()}")
    logging.info(f"  Curves  : {config.CURVES_CSV.resolve()}")
    logging.info(f"  Models  : {config.MODELS_DIR.resolve()}")
    logging.info("  -> Next: rl-sweep plot")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
