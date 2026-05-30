"""
Fast unit tests for sweep logic — no SB3/torch/gym required.
Only numpy and stdlib are imported transitively through rlsweep.sweep.
"""

from __future__ import annotations

from rlsweep import config
from rlsweep.sweep import build_run_list, make_run_id

# ── make_run_id ────────────────────────────────────────────────────────────────


def test_make_run_id_deterministic() -> None:
    """Same inputs always produce the same run_id."""
    hp = {"gamma": 0.99, "learning_rate": 1e-4, "n_steps": 512}
    assert make_run_id("CartPole-v1", "PPO", hp, 0) == make_run_id("CartPole-v1", "PPO", hp, 0)


def test_make_run_id_sorts_hparams() -> None:
    """Insertion order of hparams dict must not affect the id."""
    hp1 = {"gamma": 0.99, "learning_rate": 1e-4, "n_steps": 512}
    hp2 = {"n_steps": 512, "learning_rate": 1e-4, "gamma": 0.99}
    assert make_run_id("CartPole-v1", "PPO", hp1, 0) == make_run_id("CartPole-v1", "PPO", hp2, 0)


def test_make_run_id_differs_by_seed() -> None:
    hp = {"gamma": 0.99, "learning_rate": 1e-4, "n_steps": 512}
    assert make_run_id("CartPole-v1", "PPO", hp, 0) != make_run_id("CartPole-v1", "PPO", hp, 1)


# ── build_run_list ─────────────────────────────────────────────────────────────


def test_build_run_list_total_count() -> None:
    """Full sweep must produce exactly 120 runs."""
    runs = build_run_list(dict(config.ENVIRONMENTS))
    assert len(runs) == 120


def test_build_run_list_ppo_count() -> None:
    """PPO runs: 3 envs × 12 configs × 2 seeds = 72."""
    runs = build_run_list(dict(config.ENVIRONMENTS))
    ppo_runs = [r for r in runs if r[1] == "PPO"]
    assert len(ppo_runs) == 72


def test_build_run_list_dqn_count() -> None:
    """DQN runs: 2 envs × 12 configs × 2 seeds = 48."""
    runs = build_run_list(dict(config.ENVIRONMENTS))
    dqn_runs = [r for r in runs if r[1] == "DQN"]
    assert len(dqn_runs) == 48


def test_build_run_list_dqn_envs() -> None:
    """DQN must only appear for CartPole-v1 and LunarLander-v3."""
    runs = build_run_list(dict(config.ENVIRONMENTS))
    dqn_envs = {r[0] for r in runs if r[1] == "DQN"}
    assert dqn_envs == {"CartPole-v1", "LunarLander-v3"}


def test_build_run_list_algo_filter() -> None:
    """Passing algo='PPO' must exclude all DQN runs."""
    runs = build_run_list(dict(config.ENVIRONMENTS), algo="PPO")
    assert all(r[1] == "PPO" for r in runs)
    assert len(runs) == 72


def test_build_run_list_single_env() -> None:
    """Single-env dict with both algos: 12*2 PPO + 12*2 DQN = 48 (CartPole)."""
    envs = {"CartPole-v1": config.ENVIRONMENTS["CartPole-v1"]}
    runs = build_run_list(envs)
    assert len(runs) == 48
    assert all(r[0] == "CartPole-v1" for r in runs)


def test_build_run_list_single_env_ppo_only() -> None:
    """Acrobot has no DQN: 12 configs * 2 seeds = 24 runs."""
    envs = {"Acrobot-v1": config.ENVIRONMENTS["Acrobot-v1"]}
    runs = build_run_list(envs)
    assert len(runs) == 24
    assert all(r[1] == "PPO" for r in runs)
