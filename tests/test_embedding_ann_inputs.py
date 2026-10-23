"""Regression checks for bounded reads and malformed encoder archives."""
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest

from scripts import measure_embedding_ann as cli
from tests.test_embedding_artifacts import evidence, mutate_report


def args_for(evidence, output):
    return ['--reports', str(evidence[0]), '--registry', str(evidence[1]),
            '--fixture', str(evidence[2]), '--output', str(output)]


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO unavailable')
@pytest.mark.parametrize('target', ['report', 'registry', 'fixture', 'archive'])
def test_fifo_inputs_fail_promptly_in_bounded_child(evidence, tmp_path, target):
    paths = {'report': evidence[0], 'registry': evidence[1], 'fixture': evidence[2],
             'archive': evidence[0].parent / 'vectors.npz'}
    path = paths[target]
    path.unlink()
    os.mkfifo(path)
    output = tmp_path / 'result.json'
    result = subprocess.run([sys.executable, '-c',
        'from scripts.measure_embedding_ann import main; raise SystemExit(main())',
        *args_for(evidence, output)], capture_output=True, text=True, timeout=5,
        cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 2
    assert 'bounded regular file' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not output.exists()


def replace_archive(evidence, mode, version=2, length=10):
    path = evidence[0].parent / 'vectors.npz'
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if mode == 'npy':
        members['documents.npy'] = b'\x93NUMPY' + bytes([version, 0]) + bytes(length - 8)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    raw = bytearray(output.getvalue())
    if mode == 'deflate':
        # Invalid BTYPE=3 in the first member, while ZIP directory stays valid.
        offset = 30 + struct.unpack_from('<H', raw, 26)[0] + struct.unpack_from('<H', raw, 28)[0]
        raw[offset] = 7
    path.write_bytes(raw)
    mutate_report(evidence, lambda report: report['cost'].__setitem__('shadow_artifact_bytes', len(raw)))


@pytest.mark.parametrize('version,length', [(2, 10), (2, 11), (3, 10), (3, 11)])
def test_truncated_npy_version_lengths_fail_cli_without_traceback(evidence, tmp_path, monkeypatch, capsys, version, length):
    replace_archive(evidence, 'npy', version, length)
    monkeypatch.setattr(cli, 'measure_artifact', lambda *a, **kw: pytest.fail('must not start Docker'))
    output = tmp_path / 'result.json'
    assert cli.main(args_for(evidence, output)) == 2
    error = capsys.readouterr().err
    assert 'ValueError:' in error and 'Traceback' not in error
    assert not output.exists()


def test_invalid_deflate_fails_cli_without_traceback(evidence, tmp_path, monkeypatch, capsys):
    replace_archive(evidence, 'deflate')
    monkeypatch.setattr(cli, 'measure_artifact', lambda *a, **kw: pytest.fail('must not start Docker'))
    output = tmp_path / 'result.json'
    assert cli.main(args_for(evidence, output)) == 2
    error = capsys.readouterr().err
    assert 'ValueError: invalid compressed NPZ member' in error
    assert 'Traceback' not in error and not output.exists()
