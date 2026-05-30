.PHONY: install test lint format sweep plots clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff format .

sweep:
	rl-sweep run

plots:
	rl-sweep plot

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
