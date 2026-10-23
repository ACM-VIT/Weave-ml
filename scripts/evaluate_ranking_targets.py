"""Evaluate an owner-reviewed offline export: python -m scripts.evaluate_ranking_targets FILE."""
import argparse
import json
from pathlib import Path

from evaluation.ranking_targets import evaluate_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate_targets(json.loads(args.dataset.read_text()))
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
        parser.error(f"invalid target dataset ({type(exc).__name__}); see offline contract")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
