"""Pure retrieval measurements, independent of serving configuration."""
import math

import numpy as np


def retrieval_metrics(scores, queries, repo_ids, *, k=3):
    reciprocal_ranks, recall, ndcg, reports = [], [], [], []
    for row, query in zip(scores, queries):
        # Stable fixture order resolves exact ties reproducibly.
        ranking = sorted(range(len(repo_ids)), key=lambda index: (-float(row[index]), index))
        relevant = set(query['relevant_repo_ids'])
        ranked_ids = [repo_ids[index] for index in ranking]
        first = next((index + 1 for index, repo_id in enumerate(ranked_ids) if repo_id in relevant), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
        recall.append(len(set(ranked_ids[:k]) & relevant) / len(relevant))
        dcg = sum(1 / math.log2(index + 2) for index, repo_id in enumerate(ranked_ids[:k]) if repo_id in relevant)
        ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
        ndcg.append(dcg / ideal)
        reports.append({'query': query['query'], 'first_relevant_rank': first, 'top_repo_ids': ranked_ids[:k]})
    return {'mrr': float(np.mean(reciprocal_ranks)), f'recall_at_{k}': float(np.mean(recall)), f'ndcg_at_{k}': float(np.mean(ndcg)), 'queries': reports}
