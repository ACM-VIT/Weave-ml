"""Genuine Git-object replay provenance and tower-hit storage fencing."""
import json
import subprocess
from uuid import uuid4

import pytest

from scripts.benchmark_tower_history import git, replay, snapshots


def require_local_only_git():
    # Probe before fixture mutation. Replay intentionally fails closed if this
    # no-network option is absent, including on distribution Git 2.39 builds.
    try:
        probe = subprocess.run(['git', '--no-lazy-fetch', '--version'],
                               capture_output=True, timeout=15, check=False)
    except FileNotFoundError:
        pytest.skip('Git is required for the local-history fixture')
    if probe.returncode:
        pytest.skip('Git with --no-lazy-fetch support is required for history replay')


@pytest.fixture
def history(tmp_path):
    require_local_only_git()
    # Synthetic bare fixture only, never a checkout/copy of either project.
    repo = tmp_path / 'objects.git'
    subprocess.run(['git', 'init', '--bare', str(repo)], check=True, capture_output=True)
    parent = None
    commits = []
    for index, readme in enumerate([None, b'# Useful\nInstall with pip.\n', b'# Useful\nInstall with pip.\n', b'# Updated\nUse the new command.\n']):
        entries = []
        for path, text in [('README.md', readme), ('code.txt', str(index).encode())]:
            if text is None:
                continue
            blob = git(repo, 'hash-object', '-w', '--stdin', input=text).decode().strip()
            entries.append(f'100644 blob {blob}\t{path}\n')
        tree = git(repo, 'mktree', input=''.join(entries).encode()).decode().strip()
        args = ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit-tree', tree]
        if parent:
            args += ['-p', parent]
        parent = git(repo, *args, input=f'fixture {index}\n'.encode()).decode().strip()
        commits.append(parent)
    git(repo, 'update-ref', 'refs/heads/fixture', parent)
    return repo, commits


def test_history_reports_exact_immutable_provenance_and_missing_inputs(history):
    repo, commits = history
    frozen, rows, skipped = snapshots(repo, ref='fixture', limit=4)
    assert frozen == commits[-1]
    assert [row['commit'] for row in rows] == commits[1:]
    assert skipped == [{'commit': commits[0], 'reason': 'README missing'}]
    assert rows[0]['blob'] == rows[1]['blob'] != rows[2]['blob']
    report = replay(rows)
    assert report['readme_tower_reuses'] == 1
    assert report['readme_tower_reuse_ratio'] == pytest.approx(1/3)
    assert report['cold_vs_reused_max_absolute_error'] == 0
    assert all('text' not in row for row in report['records'])
    assert '# Useful' not in json.dumps(report)
    assert all(row['commit_time'] for row in report['records'])


def test_readme_change_cohort_does_not_count_code_only_commits_as_updates(history):
    repo, commits = history
    _, rows, _ = snapshots(repo, ref='fixture', limit=4, changes_only=True)
    assert [row['commit'] for row in rows] == [commits[1], commits[3]]
    assert replay(rows)['readme_tower_reuses'] == 0


@pytest.mark.parametrize('path', ['../README.md', '/README.md', 'package.json'])
def test_sampling_path_is_bounded_to_relative_readmes(history, path):
    with pytest.raises(ValueError, match='README'):
        snapshots(history[0], ref='fixture', path=path)


def test_small_cache_preserves_numerical_reference_when_eviction_prevents_reuse(history):
    _, rows, _ = snapshots(history[0], ref='fixture')
    report = replay([rows[0], rows[-1], rows[0]], max_entries=1)
    assert report['readme_tower_reuses'] == 0
    assert report['cache']['entries'] == 1
    assert report['cold_vs_reused_max_absolute_error'] == 0


def test_all_tower_hits_preserve_current_content_cas_and_independent_summary():
    from tests.test_embedding_validation import _cached_tower_pipeline, _repository, REPO_ID
    from tests.test_v2_qdrant_fencing import _repository_store, _repository_point
    pipeline, encoder, cache = _cached_tower_pipeline()
    results = []
    jobs = [str(uuid4()) for _ in range(3)]
    for version, job in enumerate(jobs, 1):
        result = pipeline.embed_repository(_repository(content_version=version, content_hash=f'hash-{version}'))
        result.payload['content_job_id'] = job
        results.append(result)
    assert len(encoder.calls) == 1
    assert cache.snapshot()['hits'] == 6
    assert pipeline.card_summarizer.calls == 3
    assert results[0].final_embedding == results[1].final_embedding == results[2].final_embedding
    assert [result.payload['content_version'] for result in results] == [1, 2, 3]
    assert len({id(result.card_summary) for result in results}) == 3
    assert all(result.payload['card_summary_source'] == 'description_fallback' for result in results)
    client, store = _repository_store()
    try:
        store.upsert([results[0]])
        stale_snapshot = _repository_point(client, store, REPO_ID)
        store.compare_and_set_content(results[2], expected_point=stale_snapshot)
        observed = store.compare_and_set_content(results[1], expected_point=stale_snapshot)
        assert observed.payload['content_version'] == 3
        assert observed.payload['content_job_id'] == jobs[2]
        assert observed.payload['content_hash'] == 'hash-3'
        assert observed.payload['serving_eligibility_version'] == 'repository-vector-v2'
    finally:
        client.close()


@pytest.mark.parametrize('content,reason', [(b'x'*1_000_001, 'README exceeds bounded input'),
                                            (b'\xff\xfe', 'README is not UTF-8')], ids=['oversized', 'non-utf8'])
def test_oversized_or_non_utf8_git_objects_are_reported_without_replay(history, content, reason):
    repo, commits = history
    blob = git(repo, 'hash-object', '-w', '--stdin', input=content).decode().strip()
    tree = git(repo, 'mktree', input=f'100644 blob {blob}\tREADME.md\n'.encode()).decode().strip()
    commit = git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit-tree', tree, '-p', commits[-1], input=b'bounded input fixture\n').decode().strip()
    frozen, rows, skipped = snapshots(repo, ref=commit, limit=1)
    assert frozen == commit and rows == []
    assert skipped == [{'commit': commit, 'reason': reason}]
    assert replay(rows)['readme_tower_reuse_ratio'] is None



@pytest.mark.parametrize('mode', ['120000', '040000'])
@pytest.mark.parametrize('changes_only', [False, True])
def test_nonregular_readme_entries_are_explicitly_skipped(history, mode, changes_only):
    repo, commits = history
    if mode == '120000':
        object_id = git(repo, 'hash-object', '-w', '--stdin', input=b'docs/README.md').decode().strip()
        kind = 'blob'
    else:
        object_id = git(repo, 'mktree', input=b'').decode().strip()
        kind = 'tree'
    tree = git(repo, 'mktree', input=f'{mode} {kind} {object_id}\tREADME.md\n'.encode()).decode().strip()
    commit = git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit-tree', tree, '-p', commits[-1], input=b'nonregular fixture\n').decode().strip()
    frozen, rows, skipped = snapshots(repo, ref=commit, limit=1, changes_only=changes_only)
    assert frozen == commit and rows == []
    assert skipped == [{'commit': commit, 'reason': 'README is not a regular file'}]
    assert replay(rows)['readme_tower_reuse_ratio'] is None


def test_missing_promisor_readme_never_invokes_git_transport(tmp_path, monkeypatch):
    import os
    require_local_only_git()
    if os.name != "posix":
        pytest.skip("This promisor fixture uses a POSIX transport helper")
    repo = tmp_path / 'partial'
    repo.mkdir()
    def raw_git(*args, data=None):
        return subprocess.check_output(['git', *args], cwd=repo, input=data, text=True).strip()
    raw_git('init', '--quiet')
    blob = raw_git('hash-object', '-w', '--stdin', data='README content')
    raw_git('config', 'remote.origin.url', 'proof::missing')
    raw_git('config', 'remote.origin.promisor', 'true')
    raw_git('config', 'extensions.partialClone', 'origin')
    (repo / '.git' / 'objects' / blob[:2] / blob[2:]).unlink()
    helper = tmp_path / 'git-remote-proof'
    marker = tmp_path / 'transport-invoked'
    helper.write_text('#!/bin/sh\nprintf attempted > "$PROOF_MARKER"\nexit 1\n')
    helper.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PROOF_MARKER', str(marker))
    with pytest.raises(RuntimeError, match='lazy fetching disabled'):
        git(repo, 'show', blob)
    assert not marker.exists()
    # The same actual missing object invokes transport without the guard;
    # absence above cannot be explained by an inert promisor fixture.
    env = dict(os.environ)
    env.pop('GIT_NO_LAZY_FETCH', None)
    subprocess.run(['git', 'show', blob], cwd=repo, env=env,
                   capture_output=True, text=True, timeout=15, check=False)
    assert marker.read_text() == 'attempted'


@pytest.mark.parametrize('changes_only', [False, True])
@pytest.mark.parametrize('path', ['./README.md', '././README.md'])
def test_relative_dot_readme_paths_preserve_actual_object_snapshots(history, changes_only, path):
    repo, _ = history
    canonical = snapshots(repo, ref='fixture', path='README.md', changes_only=changes_only)
    actual = snapshots(repo, ref='fixture', path=path, changes_only=changes_only)
    assert actual == canonical
    assert actual[1]
    assert all(row['path'] == 'README.md' for row in actual[1])


def test_cli_reports_canonical_readme_path(history):
    import sys
    result = subprocess.run([sys.executable, '-m', 'scripts.benchmark_tower_history',
                             '--repo', str(history[0]), '--ref', 'fixture',
                             '--path', './README.md'], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['path'] == 'README.md'
    assert report['snapshots'] > 0
    assert all(row['path'] == 'README.md' for row in report['records'])


def test_unsupported_git_skips_before_any_fixture_setup(tmp_path, monkeypatch):
    calls = []
    def unsupported(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 129, b'', b'unknown option: --no-lazy-fetch')
    monkeypatch.setattr(subprocess, 'run', unsupported)
    with pytest.raises(pytest.skip.Exception, match='no-lazy-fetch'):
        history.__wrapped__(tmp_path)
    assert not list(tmp_path.iterdir())
    assert calls == [['git', '--no-lazy-fetch', '--version']]


def test_replay_on_unsupported_git_still_fails_closed(tmp_path, monkeypatch):
    def unsupported(args, **kwargs):
        raise subprocess.CalledProcessError(129, args, stderr=b'unknown option: --no-lazy-fetch')
    monkeypatch.setattr(subprocess, 'run', unsupported)
    with pytest.raises(RuntimeError, match='lazy fetching disabled'):
        git(tmp_path, 'show', 'HEAD:README.md')
