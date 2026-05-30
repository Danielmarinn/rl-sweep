"""
Shared configuration constants for the RL sweep.

All path and hyperparameter definitions live here so every module reads
from a single source of truth.
"""

from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
RUNS_CSV = RESULTS_DIR / "runs.csv"
CURVES_CSV = RESULTS_DIR / "curves.csv"
MODELS_DIR = RESULTS_DIR / "models"
LOG_FILE = RESULTS_DIR / "sweep.log"
PLOTS_DIR = RESULTS_DIR / "plots"

# Evaluation snapshots per training run (one per 10% of total_timesteps).
N_CURVE_POINTS = 10

ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "CartPole-v1": {"timesteps": 150_000, "n_eval_episodes": 20},
    "LunarLander-v3": {"timesteps": 300_000, "n_eval_episodes": 15},
    "Acrobot-v1": {"timesteps": 300_000, "n_eval_episodes": 20},
}

# PPO grid: 3 × 2 × 2 = 12 combos per environment.
PPO_GRID: dict[str, list] = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps": [512, 2048],
    "gamma": [0.99, 0.999],
}

# DQN grid: 3 × 2 × 2 = 12 combos. Applied to CartPole and LunarLander only.
DQN_GRID: dict[str, list] = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "exploration_fraction": [0.1, 0.2],
    "gamma": [0.99, 0.999],
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
    "run_id",
    "env",
    "algo",
    "seed",
    "learning_rate",
    "n_steps",
    "gamma",
    "exploration_fraction",
    "mean_reward",
    "std_reward",
    "timesteps",
    "duration_s",
    "status",
    "error",
    "timestamp",
]
CURVE_FIELDS = ["run_id", "timestep", "mean_reward"]
