from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluate_recommendations import main, read_json

FIXTURE = Path(__file__).parent / 'fixtures' / 'recommendation_benchmark'


def arguments(output):
    return [arg for name in ('bundle', 'run', 'baseline', 'policy') for arg in (f'--{name}', str(FIXTURE / f'{name}.json'))] + ['--output', str(output)]


def test_real_cli_process_passes_diagnostic_and_rejects_synthetic_qualification(tmp_path):
    output = tmp_path / 'report.json'
    command = [sys.executable, '-m', 'scripts.evaluate_recommendations', *arguments(output)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert payload['assessment']['passed']
    assert payload['qualification_manifest'] is None
    first = output.read_bytes()
    assert subprocess.run(command, capture_output=True, timeout=10).returncode == 0
    assert first == output.read_bytes()
    assert subprocess.run(command + ['--qualify'], capture_output=True, timeout=10).returncode == 1
    payload = json.loads(output.read_text())
    assert not payload['assessment']['passed']
    assert payload['qualification_manifest'] is None
    assert any('synthetic' in value for value in payload['assessment']['failures'])


def test_cli_exits_one_for_actual_output_regression(tmp_path):
    changed = json.loads((FIXTURE / 'run.json').read_text())
    changed['outputs'][0]['stages']['assembly'][0]['repo_id'] = changed['outputs'][0]['stages']['assembly'][2]['repo_id']
    run = tmp_path / 'run.json'
    run.write_text(json.dumps(changed))
    args = arguments(tmp_path / 'report.json')
    args[args.index('--run') + 1] = str(run)
    assert main(args) == 1
    assert not json.loads((tmp_path / 'report.json').read_text())['assessment']['passed']


def test_invalid_or_oversized_input_does_not_create_a_report(tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    bad = tmp_path / 'bad.json'
    bad.write_text('{"secret": "do not echo"}')
    args = arguments(output)
    args[args.index('--bundle') + 1] = str(bad)
    assert main(args) == 2
    assert not output.exists()
    monkeypatch.setattr('scripts.evaluate_recommendations.MAX_INPUT_BYTES', 2)
    with pytest.raises(ValueError, match='budget'): read_json(bad)


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_ambiguous_or_nonfinite_json_fails(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises(ValueError): read_json(path)


def test_output_cannot_overwrite_input_including_symlink(tmp_path):
    output = tmp_path / 'output.json'
    output.symlink_to(FIXTURE.resolve() / 'bundle.json')
    assert main(arguments(output)) == 2
    assert output.is_symlink()


@pytest.mark.parametrize('corrupt', [None, 'model', 'scaler'])
def test_test_only_qualification_binds_actual_artifact_files(tmp_path, corrupt):
    """Fake declarations/files test hashing; no real model is qualified."""
    import hashlib
    from tests.test_benchmark_gates import qualification_fixture
    from tests.test_recommendation_benchmark import diagnostic
    from evaluation.recommendation_benchmark import digest
    from evaluation.benchmark_gates import Baseline

    _, policy, _ = qualification_fixture()
    bundle, run = diagnostic()
    bundle['provenance'] = dict(kind='human_judgment', reference='test-only invented declaration')
    run['timing_kind'] = 'measured'
    artifacts = {}
    for name in ('model', 'scaler'):
        artifacts[name] = tmp_path / (name + '.bin')
        artifacts[name].write_bytes(('invented test-only ' + name).encode())
        run[name + '_sha256'] = hashlib.sha256(artifacts[name].read_bytes()).hexdigest()
    from evaluation.recommendation_benchmark import Bundle, Run, evaluate
    result = evaluate(Bundle.model_validate(bundle), Run.model_validate(run), k=policy.k)
    baseline = Baseline.model_validate(dict(schema_version=1, status='promoted', reference='test-only',
        policy_sha256=digest(policy), report_sha256=digest(result), report=result))
    values = dict(bundle=bundle, run=run, baseline=baseline.model_dump(), policy=policy.model_dump())
    args = []
    for name, value in values.items():
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(value))
        args += ['--' + name, str(path)]
    if corrupt:
        artifacts[corrupt].write_bytes(b'tampered artifact')
    output = tmp_path / 'qualified.json'
    args += ['--qualify', '--output', str(output), '--model-artifact', str(artifacts['model']),
             '--scaler-artifact', str(artifacts['scaler'])]
    assert main(args) == (1 if corrupt else 0)
    payload = json.loads(output.read_text())
    if corrupt:
        assert payload['qualification_manifest'] is None
        assert payload['assessment']['failures'] == [f'actual {corrupt} artifact digest mismatch']
    else:
        manifest = payload['qualification_manifest']
        assert manifest['model_artifact']['model_sha256'] == run['model_sha256']
        assert manifest['model_artifact']['scaler_sha256'] == run['scaler_sha256']
        assert manifest['report_sha256'] == digest(payload['report'])
