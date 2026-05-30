"""
rl-sweep — unified CLI entry point.

Dispatches to the three sub-commands: run, plot, evaluate.
"""

import argparse

from rlsweep import evaluate, plot, sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rl-sweep",
        description="PPO vs DQN hyperparameter sweep toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run or resume the sweep.")
    sub.add_parser("plot", help="Generate plots from results CSVs.")
    sub.add_parser("evaluate", help="Evaluate a saved model.")
    args, rest = parser.parse_known_args()
    {"run": sweep.main, "plot": plot.main, "evaluate": evaluate.main}[args.command](rest)


if __name__ == "__main__":
    main()
