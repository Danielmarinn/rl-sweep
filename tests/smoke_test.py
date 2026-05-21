#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "results" / "runs.csv"


EXPECTED = {
    ("CartPole-v1", "PPO"): {"count": 24, "best": 500.0, "mean": 499.7},
    ("CartPole-v1", "DQN"): {"count": 24, "best": 238.6, "mean": 146.4},
    ("LunarLander-v3", "PPO"): {"count": 24, "best": 254.3, "mean": 55.3},
    ("LunarLander-v3", "DQN"): {"count": 24, "best": 109.4, "mean": -0.4},
    ("Acrobot-v1", "PPO"): {"count": 24, "best": -69.3, "mean": -97.0, "worst": -500.0},
}


def assert_close(name: str, actual: float, expected: float, abs_tol: float = 0.2) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=abs_tol):
        raise AssertionError(f"{name}: expected about {expected}, got {actual}")


def main() -> None:
    if not RUNS_CSV.exists():
        raise AssertionError(f"Missing results file: {RUNS_CSV}")

    with RUNS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 120:
        raise AssertionError(f"Expected 120 rows, got {len(rows)}")

    statuses = Counter(row["status"] for row in rows)
    if statuses != {"success": 120}:
        raise AssertionError(f"Expected 120 successful runs, got {dict(statuses)}")

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["algo"])].append(float(row["mean_reward"]))

    for key, expected in EXPECTED.items():
        values = grouped[key]
        if len(values) != expected["count"]:
            raise AssertionError(f"{key}: expected {expected['count']} rows, got {len(values)}")
        assert_close(f"{key} best", max(values), expected["best"])
        assert_close(f"{key} mean", sum(values) / len(values), expected["mean"])
        if "worst" in expected:
            assert_close(f"{key} worst", min(values), expected["worst"])

    print("Smoke test passed: 120 successful runs and headline metrics verified.")


if __name__ == "__main__":
    main()
