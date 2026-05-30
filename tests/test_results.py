"""
Pytest suite: validate headline metrics against results/runs.csv.

These tests are deliberately lightweight — stdlib + pandas only, no heavy
imports — so they run instantly even without GPU or RL libraries installed.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "results" / "runs.csv"

EXPECTED = {
    ("CartPole-v1", "PPO"): {"count": 24, "best": 500.0, "mean": 499.7},
    ("CartPole-v1", "DQN"): {"count": 24, "best": 238.6, "mean": 146.4},
    ("LunarLander-v3", "PPO"): {"count": 24, "best": 254.3, "mean": 55.3},
    ("LunarLander-v3", "DQN"): {"count": 24, "best": 109.4, "mean": -0.4},
    ("Acrobot-v1", "PPO"): {"count": 24, "best": -69.3, "mean": -97.0, "worst": -500.0},
}

ABS_TOL = 0.2


def _load_rows() -> list[dict]:
    if not RUNS_CSV.exists():
        pytest.skip(f"results file not found: {RUNS_CSV}")
    with RUNS_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _grouped_rewards(rows: list[dict]) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["algo"])].append(float(row["mean_reward"]))
    return grouped


# ── Top-level invariants ───────────────────────────────────────────────────────


def test_total_row_count() -> None:
    rows = _load_rows()
    assert len(rows) == 120, f"Expected 120 rows, got {len(rows)}"


def test_all_rows_success() -> None:
    rows = _load_rows()
    statuses = Counter(row["status"] for row in rows)
    assert statuses == {"success": 120}, f"Expected all success, got {dict(statuses)}"


# ── Per-(env, algo) parametrised checks ───────────────────────────────────────


@pytest.mark.parametrize("key,exp", list(EXPECTED.items()))
def test_per_group_count(key: tuple[str, str], exp: dict) -> None:
    rows = _load_rows()
    grouped = _grouped_rewards(rows)
    values = grouped[key]
    assert len(values) == exp["count"], f"{key}: expected {exp['count']} rows, got {len(values)}"


@pytest.mark.parametrize("key,exp", list(EXPECTED.items()))
def test_per_group_best(key: tuple[str, str], exp: dict) -> None:
    rows = _load_rows()
    grouped = _grouped_rewards(rows)
    best = max(grouped[key])
    assert math.isclose(best, exp["best"], rel_tol=0.0, abs_tol=ABS_TOL), (
        f"{key} best: expected ~{exp['best']}, got {best}"
    )


@pytest.mark.parametrize("key,exp", list(EXPECTED.items()))
def test_per_group_mean(key: tuple[str, str], exp: dict) -> None:
    rows = _load_rows()
    grouped = _grouped_rewards(rows)
    values = grouped[key]
    mean = sum(values) / len(values)
    assert math.isclose(mean, exp["mean"], rel_tol=0.0, abs_tol=ABS_TOL), (
        f"{key} mean: expected ~{exp['mean']}, got {mean}"
    )


@pytest.mark.parametrize(
    "key,exp",
    [(k, v) for k, v in EXPECTED.items() if "worst" in v],
)
def test_per_group_worst(key: tuple[str, str], exp: dict) -> None:
    rows = _load_rows()
    grouped = _grouped_rewards(rows)
    worst = min(grouped[key])
    assert math.isclose(worst, exp["worst"], rel_tol=0.0, abs_tol=ABS_TOL), (
        f"{key} worst: expected ~{exp['worst']}, got {worst}"
    )
