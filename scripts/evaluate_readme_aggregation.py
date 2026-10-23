"""Compare frozen README representations without Qdrant or provider calls.

python3 -m scripts.evaluate_readme_aggregation --output /tmp/readme-evaluation.json

The model must already be cached at the pinned revision. The fixture is a small
synthetic diagnostic, not evidence to qualify the production heavy ranker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.retrieval_metrics import retrieval_metrics

from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL, TOKEN_SECTION_EMBEDDING_VERSION
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.repository_embedding import RepositoryEmbeddingConfig
from summarization.pipeline import description_fallback_artifact


FIXTURE = Path(__file__).resolve().parents[1] / 'tests/fixtures/readme_retrieval.json'


class _SummaryFallback:
    def summarize(self, source, repo, **kwargs):
        return description_fallback_artifact(repo)


class _MeasuredModel:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_seq_length = model.max_seq_length
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.extend(texts)
        return self.model.encode(texts, **kwargs)


def evaluate(model, fixture_path=FIXTURE):
    fixture_bytes = Path(fixture_path).read_bytes()
    fixture = json.loads(fixture_bytes)
    repositories, queries = fixture['repositories'], fixture['queries']
    query_vectors = np.asarray(model.encode([query['query'] for query in queries], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
    report = {'fixture_version': fixture['fixture_version'], 'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest(), 'provenance': fixture['provenance'], 'repository_count': len(repositories), 'query_count': len(queries), 'model': REPOSITORY_EMBEDDING_MODEL, 'model_revision': EMBEDDING_MODEL_REVISION, 'max_seq_length': model.max_seq_length, 'strategies': {}}
    repo_ids = [repo['repo_id'] for repo in repositories]
    for name, version in [('character_mean_baseline', 'repo-embedding-v2'), ('token_sections', TOKEN_SECTION_EMBEDDING_VERSION)]:
        measured = _MeasuredModel(model)
        pipeline = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(
            version=version, readme_chunk_chars=2500, readme_chunk_overlap_chars=250,
        ), embedder=SentenceTransformerEmbedder(model=measured), card_summarizer=_SummaryFallback())
        started = perf_counter()
        results = pipeline.embed_batch(repositories)
        elapsed = perf_counter() - started
        counts = [len(model.tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']) for text in measured.inputs]
        readme_inputs = []
        offset = 0
        for result in results:
            readme_inputs.extend(measured.inputs[offset:offset + result.readme_chunks])
            offset += result.readme_chunks + 2
        readme_counts = [len(model.tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']) for text in readme_inputs]
        final_scores = query_vectors @ np.asarray([result.final_embedding for result in results]).T
        readme_scores = query_vectors @ np.asarray([result.readme_embedding for result in results]).T
        report['strategies'][name] = {
            'embedding_version': version,
            'readme_only': retrieval_metrics(readme_scores, queries, repo_ids),
            'final_tower': retrieval_metrics(final_scores, queries, repo_ids),
            'cost': {'wall_seconds': elapsed, 'encoder_input_count': len(counts), 'input_tokens_before_truncation': sum(counts), 'encoded_tokens_after_truncation': sum(min(count, model.max_seq_length) for count in counts), 'readme_chunk_count': len(readme_counts), 'readme_chunks_over_limit': sum(count > model.max_seq_length for count in readme_counts), 'max_readme_chunk_tokens': max(readme_counts, default=0)},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save the complete JSON comparison')
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(1)
    model = SentenceTransformer(REPOSITORY_EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION, local_files_only=True, device='cpu')
    report = evaluate(model)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
