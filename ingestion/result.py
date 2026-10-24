from dataclasses import dataclass, field
from typing import Literal, Optional
from config import NOVELTY_THRESHOLD

@dataclass
class NoveltyMatrix:
    measurement_status: Literal["measured", "unmeasured"]
    final: Optional[float]
    semantic_penalty: Optional[float]
    category_penalty: Optional[float]
    tech_stack_penalty: Optional[float]
    activity_penalty: Optional[float]
    top_k: list[dict]
    anomaly_tag: str
    explanation: str

@dataclass
class IngestionResult:
    repo_id: str
    decision: str 
    rejection_reason: str
    doc_quality: float
    code_health: float
    activity_score: float
    trend_velocity: float
    novelty: NoveltyMatrix
    tags: list[str]
    category: str
    structured_summary: str
    doc_found: list[str]
    doc_missing: list[str]
    embedding: list[float]
    topological_metrics: dict
    quadrant: str
    # Legacy decision is path-local; this observation never claims approval.
    admission_observation: dict = field(default_factory=dict)

    def __str__(self):
        width = 72
        thin_divider = "─" * width
        thick_divider = "═" * width
        status_banner = "✅ REGISTRATION APPROVED" if self.decision == "APPROVED" else "❌ ECOSYSTEM DROP REJECTED"
        
        anomaly_warning = ""
        if self.novelty.measurement_status == "measured" and self.novelty.anomaly_tag != "NONE":
            anomaly_warning = f"\n  ⚠️  SECURITY/INTEGRITY ANOMALY DETECTED: [{self.novelty.anomaly_tag}]"

        novelty_value = (
            f"{self.novelty.final:.4f}"
            if self.novelty.measurement_status == "measured" and self.novelty.final is not None
            else "UNMEASURED"
        )

        output_blocks = [
            thick_divider,
            f"  {status_banner}   ::   {self.repo_id}",
            f"  Assigned Taxonomy Core Module Node : {self.category}{anomaly_warning}",
            f"  Ecosystem Matrix Placement         : {self.quadrant}",
            thin_divider,
            f"  Documentation Quality : {self.doc_quality:.4f}  (Found Components: {', '.join(self.doc_found) or 'None'})",
            f"  Code Health Score     : {self.code_health:.4f}",
            f"  Activity Density Index: {self.activity_score:.4f}",
            f"  Trend Flux Velocity   : {self.trend_velocity:.4f}",
            thin_divider,
            f"  Standardized Structural Presentation Abstract:",
            "\n".join(f"    {line}" for line in self.structured_summary.split("\n")),
            thin_divider,
            f"  Corpus Novelty Measurement: {novelty_value}",
            thin_divider,
            self.novelty.explanation
        ]
        
        if self.topological_metrics and self.topological_metrics.get("measurement_status") == "measured":
            output_blocks += [
                thin_divider,
                f"  Corpus Cluster Topology Structural Space Vector:",
                f"    Core Target Centroid Node : {self.topological_metrics.get('current_cluster')}",
                f"    Adjacent Cluster Center   : {self.topological_metrics.get('nearest_cluster')}",
                f"    Spatial Centroid Distance : {self.topological_metrics.get('distance'):.4f}"
            ]
            
        if self.rejection_reason:
            output_blocks += ["", f"  Rejection Gate Analysis Logic: {self.rejection_reason}"]
        output_blocks.append(thick_divider)
        return "\n".join(output_blocks)
