"""Measure exported encoder artifacts using only a newly owned pinned Qdrant.

The input reports/registry/fixture are read-only. No live URL, service credential,
encoder download, model promotion or existing collection is accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import signal
import subprocess
from pathlib import Path
import sys
import zipfile

from evaluation.qdrant_ann import load_artifact, measure_artifact, publish_new_json

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "tests/fixtures/embedding_models.json")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/readme_retrieval.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt("ANN measurement terminated")

    try:
        signal.signal(signal.SIGTERM, terminate)
        if not 1 <= len(args.reports) <= 3 or not 1 <= args.k <= 100 or not 1 <= args.repeats <= 5:
            raise ValueError("ANN model/query budget out of bounds")
        if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("output must be a new file in an existing directory")
        artifacts = [load_artifact(path, args.registry, args.fixture) for path in args.reports]
        if len({artifact.configuration.version for artifact in artifacts}) != len(artifacts):
            raise ValueError("comparison models must have distinct embedding identities")
        if len({artifact.repository_ids for artifact in artifacts}) != 1:
            raise ValueError("comparison corpus identities differ")
        protected = {args.registry.resolve(), args.fixture.resolve(), *(p.resolve() for p in args.reports),
                     *(Path(artifact.artifact_path) for artifact in artifacts)}
        if args.output.resolve() in protected:
            raise ValueError("output cannot replace input evidence")
        reports = [measure_artifact(artifact, k=args.k, repeats=args.repeats) for artifact in artifacts]
        publish_new_json(args.output, {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "execution": "serial freshly owned pinned Qdrant servers; original encoded artifacts",
            "models": reports, "promotion_decision": "none; no serving state changed"})
        return 0
    except KeyboardInterrupt:
        print("ANN measurement interrupted; owned resource cleanup attempted", file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, TimeoutError,
            subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        # No response body, vectors or environment credentials are printed.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
