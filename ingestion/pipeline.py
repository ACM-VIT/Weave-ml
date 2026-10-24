import uuid
from typing import Optional

from .features import extract_tags, score_documentation, activity_score, trend_velocity, build_structured_summary, score_code_health
from .classification import classify_category
from .corpus import CorpusStore, dynamic_cluster_discovery
from .result import IngestionResult, NoveltyMatrix
from config import GATE_APPROVAL_THRESHOLD, MIN_STARS_PREFILTER, MIN_README_PREFILTER
from contracts.admission import observe_repository_admission

def _rejected_prefilter_result(repo_id: str, reason: str, repo: dict) -> IngestionResult:
    mock_novelty = NoveltyMatrix(
        final=0.0,
        semantic_penalty=0.0,
        category_penalty=0.0,
        tech_stack_penalty=0.0,
        activity_penalty=0.0,
        top_k=[],
        anomaly_tag="NONE",
        explanation="Failed pre-filter gate."
    )
    return IngestionResult(
        repo_id=repo_id, decision="REJECTED", rejection_reason=reason,
        doc_quality=0.0, code_health=0.0, activity_score=0.0, trend_velocity=0.0,
        novelty=mock_novelty, tags=[], category="General / Other",
        structured_summary="(Failed pre-filter gate)", doc_found=[], doc_missing=[],
        embedding=[0.0] * 384, topological_metrics={},
        quadrant="💤 Dormant Ecosystem Nodes",
        admission_observation=observe_repository_admission(repo),
    )

def ingest_repository(
    repo:           dict,
    corpus_store:   'CorpusStore',
    qdrant_url:     Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    auto_index:     bool          = True,
    compute_topology: bool        = False,
) -> IngestionResult:
    """Run the compatibility scorer with its existing legacy gates.

    APPROVED is a legacy local decision, not canonical quality approval.
    The shared admission proposal is observation-only pending owner approval.
    """
    repo_id = repo.get("id", f"anonymous/unidentified-node-{uuid.uuid4().hex[:6]}")
    
    # 1. Fast-Fail Pre-Filters
    star_count = repo.get("star_count", 0)
    readme_len = repo.get("readme_length", 0)
    
    if star_count < MIN_STARS_PREFILTER:
        reason = f"Failed pre-filter: star count ({star_count}) is below minimum threshold of {MIN_STARS_PREFILTER}."
        return _rejected_prefilter_result(repo_id, reason, repo)
        
    if readme_len < MIN_README_PREFILTER:
        reason = f"Failed pre-filter: README length ({readme_len} chars) is below minimum threshold of {MIN_README_PREFILTER} chars."
        return _rejected_prefilter_result(repo_id, reason, repo)
        
    # 2. Main Feature Scoring
    extracted_tags = extract_tags(repo_id, repo.get("extracted_paragraphs", []))
    documentation_metrics = score_documentation(repo)
    code_health = score_code_health(repo)
    ecosystem_activity = activity_score(repo)
    flux_trends = trend_velocity(repo)
    taxonomy_category = classify_category(repo, extracted_tags)
    
    structured_abstract = build_structured_summary(repo, extracted_tags, taxonomy_category)
    
    # Bypassed embeddings, but compute topology if requested
    spatial_topography_data = {}
    if compute_topology:
        spatial_topography_data = dynamic_cluster_discovery(corpus_store.get_history())

    
    # Gate Decision based on 50/50 blend of Doc Quality and Code Health
    blended_score = 0.50 * documentation_metrics.score + 0.50 * code_health
    
    if blended_score >= GATE_APPROVAL_THRESHOLD:
        decision_status = "APPROVED"
        rejection_reason = ""
        # Map to Approved Quadrants based on traction/popularity
        if flux_trends >= 0.40:
            matrix_quadrant = "🔥 Viral Rockets"
        else:
            matrix_quadrant = "💎 Hidden Gems"
            
        if auto_index:
            corpus_store.add_node({
                "repo_id":  repo_id,
                "category": taxonomy_category,
                "activity": ecosystem_activity,
                "tags":     extracted_tags,
                "doc_quality": documentation_metrics.score,
                "code_health": code_health,
                "trend": flux_trends,
                "vector": [0.0] * 384
            }, 1.0)
    else:
        decision_status = "REJECTED"
        rejection_reason = (
            f"Rejected: Blended score {blended_score:.4f} is below threshold {GATE_APPROVAL_THRESHOLD:.4f} "
            f"(Doc Quality: {documentation_metrics.score:.4f}, Code Health: {code_health:.4f})."
        )
        # Map to Rejected Quadrants based on traction/popularity
        if flux_trends >= 0.40:
            matrix_quadrant = "⚠️ Copycats / Clones"
        else:
            matrix_quadrant = "💤 Dormant Ecosystem Nodes"
            
    mock_novelty = NoveltyMatrix(
        final=1.0,
        semantic_penalty=0.0,
        category_penalty=0.0,
        tech_stack_penalty=0.0,
        activity_penalty=0.0,
        top_k=[],
        anomaly_tag="NONE",
        explanation="Novelty checks bypassed. Ingestion decided purely by Doc Quality & Code Health."
    )

    return IngestionResult(
        repo_id=repo_id, decision=decision_status, rejection_reason=rejection_reason,
        doc_quality=documentation_metrics.score, code_health=code_health,
        activity_score=ecosystem_activity, trend_velocity=flux_trends,
        novelty=mock_novelty, tags=extracted_tags, category=taxonomy_category,
        structured_summary=structured_abstract, doc_found=documentation_metrics.found, doc_missing=documentation_metrics.missing,
        embedding=[0.0] * 384, topological_metrics=spatial_topography_data,
        quadrant=matrix_quadrant,
        admission_observation=observe_repository_admission(repo),
    )



def ingest_batch(repos: list[dict], corpus_store: 'CorpusStore' = None, verbose: bool = True) -> list[IngestionResult]:
    """Processes sequential stream vectors into the target processing system maps."""
    if corpus_store is None:
        corpus_store = CorpusStore()
    execution_results = []
    for iteration, repository_map in enumerate(repos):
        result_node = ingest_repository(repository_map, corpus_store=corpus_store)
        if verbose:
            print(result_node)
        execution_results.append(result_node)
    return execution_results

def print_batch_summary(results: list[IngestionResult], corpus_timeline: list[dict] = None) -> None:
    """Generates an extensive, elite terminal dashboard charting structural ecosystem growth profiles."""
    approved_nodes = [node for node in results if node.decision == "APPROVED"]
    rejected_nodes = [node for node in results if node.decision == "REJECTED"]
    novelty_floats = [node.novelty.final for node in results]
    
    width = 76
    double_boundary = "═" * width
    thin_boundary = "─" * width
    
    from collections import Counter
    taxonomy_distribution = Counter(node.category for node in approved_nodes)
    quadrant_distribution = Counter(node.quadrant for node in results)
    
    print(thin_boundary)
    print("  Ecosystem Retrieval 2x2 Matrix Quadrant Breakdown:")
    quadrants_to_check = ["🔥 Viral Rockets", "💎 Hidden Gems", "⚠️ Copycats / Clones", "💤 Dormant Ecosystem Nodes"]
    for quad in quadrants_to_check:
        frequency = quadrant_distribution.get(quad, 0)
        bar_graph_visualization = "▓" * min(frequency, 30)
        print(f"    {quad:<30} [{frequency:>2}]  {bar_graph_visualization}")
    
    highest_novelty_rankings = sorted(approved_nodes, key=lambda x: x.novelty.final, reverse=True)[:5]
    convergent_density_rankings = sorted(approved_nodes, key=lambda x: x.novelty.final)[:5]
    flagged_anomalies = [node for node in results if node.novelty.anomaly_tag != "NONE"]

    print("\\n" + double_boundary)
    print(f"               OSIRIS RESEARCH ENGINE STREAM ANALYTICS REPORT")
    print(double_boundary)
    print(f"  Processed Node Evaluation Streams : {len(results)}")
    print(f"  Approved Active Ecosystem Signatures: {len(approved_nodes)}")
    print(f"  Rejected Conflict Vector Drops     : {len(rejected_nodes)}")
    print(f"  Total Flagged System Anomalies    : {len(flagged_anomalies)}")
    
    if novelty_floats:
        print(thin_boundary)
        print(f"  Corpus Ecosystem Novelty Mean Value : {sum(novelty_floats)/len(novelty_floats):.4f}")
        print(f"  Absolute Delta Floor Min Recorded   : {min(novelty_floats):.4f}")
        print(f"  Absolute Delta Peak Max Recorded   : {max(novelty_floats):.4f}")
    print(thin_boundary)
    
    print(f"  Active Taxonomy Spatial Distributions (Registered Approved Nodes):")
    for category_name, frequency in taxonomy_distribution.most_common():
        bar_graph_visualization = "█" * min(frequency, 30)
        print(f"    {category_name:<30} [{frequency:>2}]  {bar_graph_visualization}")
    print(thin_boundary)
    
    print(f"  Top 5 High-Novelty Ecosystem Disruptors:")
    for rank, node in enumerate(highest_novelty_rankings, 1):
        print(f"    {rank}. {node.repo_id:<44} Novelty Vector Index: {node.novelty.final:.4f} [{node.category}]")
    print(thin_boundary)
    
    print(f"  Top 5 Base-Convergent Alternates (Approved Near Minimum Floor Boundaries):")
    for rank, node in enumerate(convergent_density_rankings, 1):
        nearest_match_signature = node.novelty.top_k[0]["repo_id"].split("/")[-1] if node.novelty.top_k else "None (Corpus Origin Seed)"
        print(f"    {rank}. {node.repo_id:<44} Novelty Vector Index: {node.novelty.final:.4f} (Match: {nearest_match_signature})")
        
    if corpus_timeline:
        print(thin_boundary)
        print(f"  Ecosystem Corpus Growth & Novelty Vector Decay Timeline Analysis:")
        sampling_step_interval = max(1, len(corpus_timeline) // 5)
        for position in range(0, len(corpus_timeline), sampling_step_interval):
            timeline_node = corpus_timeline[position]
            print(f"    [Corpus Size Node: {timeline_node['growth_index']:>2}]  Source Module: {timeline_node['repo_id']:<32} Registered Novelty Tracking: {timeline_node['novelty_index_point']:.4f}")
            
    if rejected_nodes:
        print(thin_boundary)
        print(f"  Rejected System Drop Registers ({len(rejected_nodes)} Conflicting Nodes):")
        for node in rejected_nodes:
            print(f"    [!] Drop Track -> {node.repo_id:<42} Matrix Evaluation Value: {node.novelty.final:.4f}")
            
    print(double_boundary + "\\n")
