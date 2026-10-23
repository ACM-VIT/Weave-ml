"""Offline registry and cross-model safety checks; no model loads in unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.embedding_registry import EmbeddingConfig, Registry, VectorBatch, compatible_scores, encode_role


@pytest.fixture
def config():
    return EmbeddingConfig.model_validate(dict(name='test-model', model_id='test/model', revision='a'*40,
        native_dimension=3, output_dimension=2, max_tokens=128, pooling='last_token',
        normalize=True, instruction_aware=True, query_prompt='Instruct: test\nQuery:', document_prompt=''))


def test_embedding_identity_binds_dimensions_prompts_revision_and_token_recipe(config):
    version = config.version
    assert version == config.version
    for change in [{'output_dimension': 3}, {'query_prompt': 'different'}, {'revision': 'b'*40}, {'max_tokens': 256}]:
        other = EmbeddingConfig.model_validate({**config.model_dump(), **change})
        assert other.version != version
    with pytest.raises(ValueError): EmbeddingConfig.model_validate({**config.model_dump(), 'output_dimension': 4})
    with pytest.raises(ValueError): EmbeddingConfig.model_validate({**config.model_dump(), 'revision': 'main'})
    with pytest.raises(ValueError): EmbeddingConfig.model_validate({**config.model_dump(), 'normalize': False})
    with pytest.raises(ValueError): EmbeddingConfig.model_validate({**config.model_dump(), 'instruction_aware': False})


def test_role_prompts_truncation_and_normalization_are_explicit(config):
    class Model:
        def __init__(self): self.calls = []
        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            # The adapter validates output instead of trusting normalize=True.
            return np.asarray([[.6, .8] for _ in texts])
    model = Model()
    query = encode_role(model, ['interest'], config, role='query', batch_size=2)
    documents = encode_role(model, ['repository'], config, role='document', batch_size=2)
    assert model.calls[0][1]['prompt'] == config.query_prompt
    assert model.calls[1][1]['prompt'] == ''
    assert all(call[1]['truncate_dim'] == 2 and call[1]['normalize_embeddings'] for call in model.calls)
    assert np.allclose(compatible_scores(query, documents), [[1.]])


def test_equal_dimensions_do_not_make_different_models_or_prompts_compatible(config):
    first = VectorBatch.create(np.asarray([[1., 0.]]), config, role='query')
    changed = EmbeddingConfig.model_validate({**config.model_dump(), 'model_id': 'another/model'})
    other = VectorBatch.create(np.asarray([[1., 0.]]), changed, role='document')
    with pytest.raises(ValueError, match='identity'): compatible_scores(first, other)
    same_role = VectorBatch.create(np.asarray([[1., 0.]]), config, role='query')
    with pytest.raises(ValueError, match='roles'): compatible_scores(first, same_role)


@pytest.mark.parametrize('values', [[[0., 0.]], [[1., float('nan')]], [[1., 1.]], [[1., 0., 0.]]])
def test_invalid_model_output_fails_before_cosine(config, values):
    with pytest.raises(ValueError): VectorBatch.create(np.asarray(values), config, role='query')


def test_registry_switching_uses_only_pinned_config():
    path = Path(__file__).parent / 'fixtures' / 'embedding_models.json'
    registry = Registry.model_validate(json.loads(path.read_text()))
    assert registry.get('minilm-384').output_dimension == 384
    assert registry.get('qwen3-384').output_dimension == 384
    assert registry.get('qwen3-1024').output_dimension == 1024
    assert registry.get('qwen3-384').version != registry.get('qwen3-1024').version
    with pytest.raises(ValueError, match='unknown'): registry.get('missing')
    with pytest.raises(ValueError, match='unique'):
        Registry.model_validate({'schema_version': 1, 'models': [registry.models[0].model_dump()] * 2})


def test_invalid_fixture_cannot_start_model_work():
    import copy
    from scripts.compare_embedding_models import validate_fixture
    fixture = json.loads((Path(__file__).parent / 'fixtures/readme_retrieval.json').read_text())
    validate_fixture(fixture)
    for mutation in ['duplicate', 'unknown_relevant', 'bad_text', 'blank_query', 'no_provenance']:
        changed = copy.deepcopy(fixture)
        if mutation == 'duplicate': changed['repositories'].append(changed['repositories'][0])
        if mutation == 'unknown_relevant': changed['queries'][0]['relevant_repo_ids'] = ['not-in-corpus']
        if mutation == 'bad_text': changed['repositories'][0]['readme'] = 1
        if mutation == 'blank_query': changed['queries'][0]['query'] = ' '
        if mutation == 'no_provenance': changed['provenance'] = ''
        with pytest.raises(ValueError): validate_fixture(changed)


def test_runner_rejects_invalid_budget_and_missing_model_before_loading(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from scripts import compare_embedding_models as runner
    worker = MagicMock(side_effect=AssertionError('must not load any model'))
    monkeypatch.setattr(runner, 'run_worker', worker)
    output = tmp_path / 'report.json'
    assert runner.main(['--worker', '--output', str(output), '--threads', '0']) == 2
    assert runner.main(['--worker', '--output', str(output), '--models', 'missing']) == 2
    assert not output.exists()
    worker.assert_not_called()


def test_worker_failure_preserves_previous_aggregate_report(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import compare_embedding_models as runner
    output = tmp_path / 'report.json'
    output.write_text('previous complete evidence')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr='test failure'))
    assert runner.main(['--output', str(output), '--models', 'minilm-384']) == 2
    assert output.read_text() == 'previous complete evidence'
