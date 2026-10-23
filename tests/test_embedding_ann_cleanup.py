"""Safe owned-resource cleanup diagnostics on all CLI failure paths."""
import pytest
from evaluation import qdrant_ann as ann
from scripts import measure_embedding_ann as cli
from tests.test_embedding_artifacts import evidence


def args_for(evidence, output):
    return ['--reports', str(evidence[0]), '--registry', str(evidence[1]),
            '--fixture', str(evidence[2]), '--output', str(output)]


@pytest.mark.parametrize('phase', ['body', 'entry'])
@pytest.mark.parametrize('failure,expected', [(ValueError, 2), (KeyboardInterrupt, 130)])
def test_cli_cleanup_warning_survives_error_and_interruption(evidence, tmp_path, monkeypatch, capsys, phase, failure, expected):
    owned = ann.OwnedQdrant()
    owned.root = tmp_path / 'weave-embedding-ann-safe123'
    def fail_cleanup(self):
        raise RuntimeError('SECRET-CLEANUP-CREDENTIAL')
    monkeypatch.setattr(ann.OwnedQdrant, 'cleanup', fail_cleanup)
    if phase == 'entry':
        monkeypatch.setattr(owned, '_command', lambda *a, **kw: (_ for _ in ()).throw(failure('original')))
        def measure(*a, **kw):
            owned.__enter__()
    else:
        monkeypatch.setattr(ann.OwnedQdrant, '__enter__', lambda self: self)
        def measure(*a, **kw):
            with owned:
                raise failure('original')
    monkeypatch.setattr(cli, 'measure_artifact', measure)
    output = tmp_path / 'result.json'
    assert cli.main(args_for(evidence, output)) == expected
    error = capsys.readouterr().err
    assert 'WARNING: owned ANN cleanup failed' in error
    assert 'container=' + owned.name in error
    assert 'storage=' + owned.root.name in error
    assert 'SECRET' not in error and str(tmp_path) not in error and 'Traceback' not in error
    assert not output.exists()


def test_cleanup_only_failure_is_safe_and_prevents_publication(evidence, tmp_path, monkeypatch, capsys):
    owned = ann.OwnedQdrant()
    monkeypatch.setattr(ann.OwnedQdrant, '__enter__', lambda self: self)
    monkeypatch.setattr(owned, 'cleanup', lambda: (_ for _ in ()).throw(RuntimeError('SECRET-CLEANUP-CREDENTIAL')))
    def measure(*a, **kw):
        with owned:
            return {'complete': True}
    monkeypatch.setattr(cli, 'measure_artifact', measure)
    output = tmp_path / 'result.json'
    assert cli.main(args_for(evidence, output)) == 2
    error = capsys.readouterr().err
    assert 'WARNING:' in error and 'container=' + owned.name in error
    assert 'SECRET' not in error and 'Traceback' not in error and not output.exists()
