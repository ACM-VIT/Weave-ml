"""Offline comparison failure/provenance tests with no model or network."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import compare_embedding_models as comparison


def arguments(tmp_path):
    fixture = tmp_path / 'fixture.json'
    fixture.write_bytes(comparison.FIXTURE.read_bytes())
    return ['--fixture', str(fixture), '--output', str(tmp_path / 'comparison.json'),
            '--models', 'minilm-384', 'qwen3-384']


def fake_worker(command, **kwargs):
    output = Path(command[command.index('--output') + 1])
    fixture = Path(command[command.index('--fixture') + 1])
    registry = Path(command[command.index('--registry') + 1])
    vector = output.with_suffix('.vectors.npz')
    vector.write_bytes(fixture.read_bytes())
    payload = {'registry_sha256': hashlib.sha256(registry.read_bytes()).hexdigest(),
        'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'cost': {'shadow_artifact': vector.name}}
    output.write_text(json.dumps(payload))
    return SimpleNamespace(returncode=0, stderr='')


def test_failed_rerun_keeps_previous_aggregate_and_referenced_vectors(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(comparison.subprocess, 'run', fake_worker)
    assert comparison.main(args) == 0
    aggregate = tmp_path / 'comparison.json'
    saved = aggregate.read_bytes()
    vector_paths = [tmp_path / r['cost']['shadow_artifact'] for r in json.loads(saved)['models']]
    vectors = [p.read_bytes() for p in vector_paths]
    fixture = tmp_path / 'fixture.json'
    changed = json.loads(fixture.read_text())
    changed['provenance'] += ' changed test-only input'
    fixture.write_text(json.dumps(changed))
    calls = []
    def fail_second(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            return SimpleNamespace(returncode=1, stderr='intentional isolated worker failure')
        return fake_worker(command, **kwargs)
    monkeypatch.setattr(comparison.subprocess, 'run', fail_second)
    assert comparison.main(args) == 2
    assert aggregate.read_bytes() == saved
    assert [p.read_bytes() for p in vector_paths] == vectors


@pytest.mark.parametrize('field', ['registry_sha256', 'fixture_sha256'])
def test_parent_rejects_worker_input_hash_drift_before_publishing(tmp_path, monkeypatch, field):
    args = arguments(tmp_path)
    aggregate = tmp_path / 'comparison.json'
    aggregate.write_bytes(b'previous valid evidence')
    def changed_worker(command, **kwargs):
        result = fake_worker(command, **kwargs)
        output = Path(command[command.index('--output') + 1])
        report = json.loads(output.read_text())
        report[field] = 'f' * 64
        output.write_text(json.dumps(report))
        return result
    monkeypatch.setattr(comparison.subprocess, 'run', changed_worker)
    assert comparison.main(args) == 2
    assert aggregate.read_bytes() == b'previous valid evidence'


@pytest.mark.parametrize('missing', ['fixture_version', 'full_name'])
def test_missing_required_fixture_fields_fail_before_any_worker(tmp_path, monkeypatch, missing):
    args = arguments(tmp_path)
    fixture = tmp_path / 'fixture.json'
    value = json.loads(fixture.read_text())
    del (value if missing == 'fixture_version' else value['repositories'][0])[missing]
    fixture.write_text(json.dumps(value))
    monkeypatch.setattr(comparison.subprocess, 'run', lambda *a, **k: pytest.fail('Model worker launched'))
    assert comparison.main(args) == 2
    assert not (tmp_path / 'comparison.json').exists()


def test_offline_cli_import_ignores_serving_collection_configuration():
    env = dict(os.environ, QDRANT_COLLECTION_NAME='unrelated-local-offline-collection')
    result = subprocess.run([sys.executable, '-m', 'scripts.compare_embedding_models', '--help'],
        env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--registry' in result.stdout


def test_import_help_and_memory_unknown_without_unix_resource():
    # A fresh interpreter prevents an already-imported resource module masking
    # the native-Windows failure during test collection and CLI startup.
    code = """
import builtins
real_import = builtins.__import__
def without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError("No module named 'resource'")
    return real_import(name, *args, **kwargs)
builtins.__import__ = without_resource
from scripts import compare_embedding_models as comparison
assert comparison._peak_process_rss_bytes() is None
comparison.main(['--help'])
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--registry' in result.stdout


@pytest.mark.parametrize('platform_name, expected', [('linux', 128 * 1024), ('darwin', 128)])
def test_peak_rss_retains_actual_platform_units(monkeypatch, platform_name, expected):
    fake = SimpleNamespace(RUSAGE_SELF=0,
                           getrusage=lambda who: SimpleNamespace(ru_maxrss=128))
    monkeypatch.setitem(sys.modules, 'resource', fake)
    monkeypatch.setattr(comparison.sys, 'platform', platform_name)
    assert comparison._peak_process_rss_bytes() == expected
