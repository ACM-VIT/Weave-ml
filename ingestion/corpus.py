import math
import threading
from collections import Counter
from numbers import Real

import numpy as np

class CorpusStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = []
        self.timeline = []

    def add_node(self, node: dict, novelty_score: float | None):
        with self.lock:
            self.history.append(node)
            self.timeline.append({
                "growth_index": len(self.history),
                "repo_id": node["repo_id"],
                "novelty_index_point": novelty_score
            })

    def get_history(self) -> list[dict]:
        with self.lock:
            return list(self.history)
            
    def get_timeline(self) -> list[dict]:
        with self.lock:
            return list(self.timeline)

    def size(self) -> int:
        with self.lock:
            return len(self.history)


def dynamic_cluster_discovery(corpus: list[dict]) -> dict:
    """
    Executes real-time mathematical centroid tracking across incoming vector configurations.
    Measures dimensional saturation density distributions and topological boundary distances.
    """
    vectors = []
    for node in corpus:
        vector = node.get("vector")
        if not isinstance(vector, (list, tuple, np.ndarray)):
            continue
        try:
            if len(vector) == 0:
                continue
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in vector):
                continue
        except TypeError:
            continue
        try:
            vector_array = np.asarray(vector, dtype=float)
        except (OverflowError, TypeError, ValueError):
            continue
        if vector_array.ndim != 1 or vector_array.size == 0 or not np.isfinite(vector_array).all():
            continue
        norm = float(np.linalg.norm(vector_array))
        if not math.isfinite(norm) or norm == 0:
            continue
        vectors.append((node, vector_array / norm))

    if len(vectors) < 4:
        return {
            "current_cluster": "Unavailable",
            "nearest_cluster": "Unavailable",
            "distance": None,
            "measurement_status": "unmeasured",
            "reason": "Fewer than four valid non-zero repository vectors are available.",
        }

    dimensions = {vector.size for _, vector in vectors}
    if len(dimensions) != 1:
        return {
            "current_cluster": "Unavailable",
            "nearest_cluster": "Unavailable",
            "distance": None,
            "measurement_status": "unmeasured",
            "reason": "Repository vectors use incompatible dimensions.",
        }

    categories = [node["category"] for node, _ in vectors]
    distribution = Counter(categories)

    if len(distribution) < 2:
        return {
            "current_cluster": distribution.most_common(1)[0][0],
            "nearest_cluster": "Unavailable",
            "distance": None,
            "measurement_status": "unmeasured",
            "reason": "At least two observed categories are required for a centroid comparison.",
        }

    top_cluster = distribution.most_common(1)[0][0]
    alt_cluster = distribution.most_common(2)[1][0]

    # Calculate dimensional centroid arrays
    vectors_top = [vector for node, vector in vectors if node["category"] == top_cluster]
    vectors_alt = [vector for node, vector in vectors if node["category"] == alt_cluster] if len(distribution) > 1 else vectors_top

    centroid_top = np.mean(vectors_top, axis=0)
    centroid_alt = np.mean(vectors_alt, axis=0)
    
    norm_t = np.linalg.norm(centroid_top)
    norm_a = np.linalg.norm(centroid_alt)
    
    if not norm_t or not norm_a:
        return {
            "current_cluster": top_cluster,
            "nearest_cluster": alt_cluster,
            "distance": None,
            "measurement_status": "unmeasured",
            "reason": "A category centroid has no usable direction.",
        }
    cosine_distance = 1.0 - float(np.dot(centroid_top, centroid_alt) / (norm_t * norm_a))
    return {
        "current_cluster": top_cluster,
        "nearest_cluster": alt_cluster,
        "distance": round(float(np.clip(cosine_distance, 0.0, 2.0)), 4),
        "measurement_status": "measured",
    }
