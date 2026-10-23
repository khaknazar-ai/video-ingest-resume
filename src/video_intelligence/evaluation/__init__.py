"""Explicit evaluation contracts for retrieval and other pipeline stages."""

from video_intelligence.evaluation.dataset_guard import (
    evaluate_real_detection,
    evaluate_real_retrieval,
)
from video_intelligence.evaluation.detection_comparison import (
    ComparisonStatus,
    DetectionComparisonHarness,
    DetectionComparisonReport,
    DetectionRunReport,
)
from video_intelligence.evaluation.detection_metrics import (
    DetectionAnnotation,
    DetectionMetrics,
    GroundTruthBox,
    evaluate_detection,
)
from video_intelligence.evaluation.retrieval_metrics import (
    RetrievalAnnotation,
    RetrievalMetrics,
    evaluate_retrieval,
)

__all__ = [
    "ComparisonStatus",
    "DetectionAnnotation",
    "DetectionComparisonHarness",
    "DetectionComparisonReport",
    "DetectionMetrics",
    "DetectionRunReport",
    "GroundTruthBox",
    "RetrievalAnnotation",
    "RetrievalMetrics",
    "evaluate_detection",
    "evaluate_real_detection",
    "evaluate_real_retrieval",
    "evaluate_retrieval",
]
