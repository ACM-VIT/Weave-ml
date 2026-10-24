"""Evaluate an owner-reviewed offline export: python -m scripts.evaluate_ranking_targets FILE."""
import argparse
import json
from pathlib import Path

from evaluation.ranking_targets import evaluate_targets


def unique_object(pairs):
    """Reject repeated JSON keys at every depth before claims can be overwritten."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate_targets(json.loads(args.dataset.read_text(), object_pairs_hook=unique_object))
    except (ValueError, KeyError, TypeError, OverflowError, OSError, RecursionError) as exc:
        parser.error(f"invalid target dataset ({type(exc).__name__}); see offline contract")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
