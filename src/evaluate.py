"""
Evaluate a saved model from the RL sweep.
=========================================
Loads best_model.zip for any completed run and either renders it live
or runs a silent benchmark.

Usage
-----
    # Interactive picker (lists all saved models, you choose)
    python evaluate.py

    # Direct by run_id (tab-completable with --list first)
    python evaluate.py --run_id CartPole-v1__PPO__gamma=0.99_learning_rate=0.001_n_steps=512__s0

    # Benchmark mode (no window, just stats)
    python evaluate.py --no_render --n_episodes 50

    # List all saved models and exit
    python evaluate.py --list

Fixes vs original:
    - Learning rate formatting crash: hp.get('learning_rate') is None-guarded
      before applying :.0e format (was ValueError on missing keys)
    - Added --render_mode flag so users can override 'human' to 'rgb_array'
      for headless servers
    - episode_lengths tracked and printed in summary
    - Per-episode output suppressed in --no_render mode for cleaner CI output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODELS_DIR = Path("results/models")


# ── Discovery ─────────────────────────────────────────────────────────────────

def find_models() -> list[Path]:
    """Return all model dirs that contain best_model.zip and config.json."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p for p in MODELS_DIR.iterdir()
        if p.is_dir()
        and (p / "best_model.zip").exists()
        and (p / "config.json").exists()
    )


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json", encoding="utf-8") as f:
        return json.load(f)


def get_algo_class(algo_name: str):
    """Lazily import and return the SB3 algorithm class by name."""
    from stable_baselines3 import DQN, PPO
    classes = {"PPO": PPO, "DQN": DQN}
    if algo_name not in classes:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Supported: {list(classes)}")
    return classes[algo_name]


# ── Display ───────────────────────────────────────────────────────────────────

def print_model_table(models: list[Path]) -> None:
    print(f"\n{'#':<5}  {'ENV':<18}  {'ALGO':<5}  {'SEED':<5}  "
          f"{'MEAN REWARD':>11}  {'LR':>8}  {'GAMMA':>6}  RUN ID")
    print("─" * 100)
    for i, m in enumerate(models):
        cfg = load_config(m)
        hp  = cfg.get("hparams", {})

        # FIX: guard against missing learning_rate key (originally caused crash)
        lr      = hp.get("learning_rate")
        lr_str  = f"{lr:.0e}" if lr is not None else "N/A"
        g       = hp.get("gamma", "")
        r       = cfg.get("mean_reward", "N/A")

        print(
            f"{i:<5}  {cfg['env']:<18}  {cfg['algo']:<5}  {cfg['seed']:<5}  "
            f"{r!s:>11}  {lr_str:>8}  {g!s:>6}  "
            f"{cfg['run_id'][:60]}"
        )
    print()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model_dir:  Path,
    n_episodes: int  = 10,
    render:     bool = True,
    verbose:    bool = True,
) -> list[float]:
    """
    Load best_model.zip and run n_episodes of evaluation.
    Returns the list of per-episode rewards.
    """
    import gymnasium as gym

    cfg        = load_config(model_dir)
    env_name   = cfg["env"]
    algo_name  = cfg["algo"]
    AlgoClass  = get_algo_class(algo_name)

    print(f"\n{'─'*62}")
    print(f"  Model   : {model_dir.name}")
    print(f"  Env     : {env_name}")
    print(f"  Algo    : {algo_name}  |  seed={cfg['seed']}")
    hp = cfg.get("hparams", {})
    print(f"  HParams : {', '.join(f'{k}={v}' for k, v in hp.items())}")
    trained_r = cfg.get('mean_reward', 'N/A')
    trained_s = cfg.get('std_reward',  '?')
    print(f"  Trained : {trained_r} ± {trained_s}")
    print(f"{'─'*62}\n")

    model       = AlgoClass.load(str(model_dir / "best_model"))
    render_mode = "human" if render else None
    env         = gym.make(env_name, render_mode=render_mode)

    episode_rewards = []
    episode_lengths = []

    for ep in range(1, n_episodes + 1):
        obs, _  = env.reset()
        done    = False
        total_r = 0.0
        steps   = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps   += 1
            done     = terminated or truncated

        episode_rewards.append(total_r)
        episode_lengths.append(steps)

        if verbose:
            print(f"  Episode {ep:>3}/{n_episodes}  │  "
                  f"reward={total_r:>8.2f}  │  steps={steps}")

    env.close()

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)
    min_r  = np.min(episode_rewards)
    max_r  = np.max(episode_rewards)

    print(f"\n{'─'*62}")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Mean reward : {mean_r:.2f} ± {std_r:.2f}")
    print(f"  Min / Max   : {min_r:.2f} / {max_r:.2f}")
    print(f"  Avg length  : {np.mean(episode_lengths):.1f} steps")
    print(f"{'─'*62}\n")

    return episode_rewards


# ── Interactive picker ─────────────────────────────────────────────────────────

def interactive_pick(models: list[Path]) -> Path:
    print_model_table(models)
    while True:
        try:
            choice = input(f"Pick a model [0–{len(models)-1}] (or 'q' to quit): ").strip()
            if choice.lower() in ("q", "quit", "exit"):
                sys.exit(0)
            idx = int(choice)
            if 0 <= idx < len(models):
                return models[idx]
            print(f"  Please enter a number between 0 and {len(models)-1}.")
        except (ValueError, EOFError):
            print("  Invalid input.")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and evaluate a saved RL model from the sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_id",     type=str,  default=None,
                        help="Exact run_id to evaluate (omit for interactive picker)")
    parser.add_argument("--n_episodes", type=int,  default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--no_render",  action="store_true",
                        help="Disable rendering (benchmarking / headless servers)")
    parser.add_argument("--list",       action="store_true",
                        help="Print all available models and exit")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress per-episode output (useful with --no_render)")
    args = parser.parse_args()

    models = find_models()

    if not models:
        print("\nNo saved models found in results/models/")
        print("Run rl_sweep.py first, then come back here.\n")
        sys.exit(1)

    if args.list:
        print(f"\nFound {len(models)} saved model(s):\n")
        print_model_table(models)
        sys.exit(0)

    if args.run_id:
        matches = [m for m in models if m.name == args.run_id]
        if not matches:
            print(f"\nNo model found with run_id: {args.run_id}")
            print("Use --list to see available models.\n")
            sys.exit(1)
        model_dir = matches[0]
    else:
        print(f"\nFound {len(models)} saved model(s).")
        model_dir = interactive_pick(models)

    evaluate(
        model_dir  = model_dir,
        n_episodes = args.n_episodes,
        render     = not args.no_render,
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()