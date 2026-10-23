"""Evaluate frozen recommendation evidence without contacting serving services.

This scores supplied stage outputs; it does not run or qualify the live ranker.
Exit 0 means requested offline checks passed, 1 means gates failed, 2 means invalid
input. Qualification records still require externally audited provenance and an
authorized promotion decision; this command never changes model configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from evaluation.benchmark_gates import Baseline, Policy, assess, qualification_manifest
from evaluation.recommendation_benchmark import Bundle, Run, evaluate

MAX_INPUT_BYTES = 64 * 1024 * 1024


def read_json(path: Path, *, max_input_bytes: int | None = None):
    if max_input_bytes is None:
        max_input_bytes = MAX_INPUT_BYTES
    # Read at most the budget, including when a file grows after stat().
    with path.open('rb') as source:
        raw = source.read(max_input_bytes + 1)
    if len(raw) > max_input_bytes:
        raise ValueError(f'{path.name}: input exceeds byte budget')
    def reject_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError('duplicate JSON object key')
            output[key] = value
        return output
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)


def atomic_json(path: Path, payload: dict):
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: Path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'run', 'baseline', 'policy', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true', help='Check evidence for a later authorized promotion review')
    parser.add_argument('--model-artifact', type=Path)
    parser.add_argument('--scaler-artifact', type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = [args.bundle, args.run, args.baseline, args.policy, args.model_artifact, args.scaler_artifact]
        if any(path is not None and path.resolve() == args.output.resolve() for path in inputs):
            raise ValueError('output must not overwrite an input artifact')
        bundle = Bundle.model_validate(read_json(args.bundle))
        raw_run = read_json(args.run)
        local_stage_replay = False
        if isinstance(raw_run, dict) and raw_run.get('format') == 'frozen-model-stages-v1':
            from evaluation.stage_runner import StageRunExport
            from evaluation.recommendation_benchmark import digest
            exported = StageRunExport.model_validate(raw_run)
            if exported.bundle_sha256 != digest(bundle):
                raise ValueError('stage export belongs to another frozen bundle')
            run = exported.run
            local_stage_replay = True
        else:
            run = Run.model_validate(raw_run)
        baseline = Baseline.model_validate(read_json(args.baseline))
        policy = Policy.model_validate(read_json(args.policy))
        report = evaluate(bundle, run, k=policy.k)
        assessment = assess(report, baseline, policy, qualify=args.qualify)
        if args.qualify:
            if local_stage_replay:
                assessment['failures'].append('local exact-search stage replay is diagnostic, not production qualification evidence')
            if args.model_artifact is None or args.scaler_artifact is None:
                assessment['failures'].append('qualification requires actual model and scaler artifacts')
            else:
                if file_sha256(args.model_artifact) != run.model_sha256:
                    assessment['failures'].append('actual model artifact digest mismatch')
                if file_sha256(args.scaler_artifact) != run.scaler_sha256:
                    assessment['failures'].append('actual scaler artifact digest mismatch')
            assessment['passed'] = not assessment['failures']
        payload = {'report': report, 'assessment': assessment, 'qualification_manifest': None}
        if args.qualify and assessment['passed']:
            payload['qualification_manifest'] = qualification_manifest(report, baseline, policy)
        atomic_json(args.output, payload)
        print(json.dumps({'passed': assessment['passed'], 'failures': assessment['failures'],
                          'report_sha256': assessment['report_sha256']}, sort_keys=True))
        return 0 if assessment['passed'] else 1
    except (ValueError, OSError, RecursionError) as exc:
        # Pydantic errors can contain private input values. Export only a type.
        print(json.dumps({'passed': False, 'error': type(exc).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
