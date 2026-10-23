"""CPU-first detection contracts and coordinate-safe postprocessing."""

from video_intelligence.detection.base import (
    DetectionBatch,
    DetectionInput,
    DetectionObservation,
    UnavailableDetectorAdapter,
    postprocess_detections,
)
from video_intelligence.detection.closed_set import (
    ClosedSetCandidate,
    UnavailableClosedSetAdapter,
)
from video_intelligence.detection.grounding import (
    GroundingBatch,
    GroundingBudget,
    GroundingCandidate,
    GroundingComparisonHarness,
    GroundingComparisonReport,
    GroundingDetection,
    GroundingInput,
    GroundingRequest,
    GroundingRunReport,
    GroundingStatus,
    UnavailableGroundingAdapter,
)
from video_intelligence.detection.postprocess import (
    BoundingBox,
    BoxFormat,
    CoordinateSpace,
    EncodedDetection,
    RawDetection,
    decode_box,
    decode_model_detections,
)
from video_intelligence.detection.prompts import PhraseSpec, PromptPlan, PromptRegistry

__all__ = [
    "BoundingBox",
    "BoxFormat",
    "CoordinateSpace",
    "ClosedSetCandidate",
    "DetectionBatch",
    "DetectionInput",
    "DetectionObservation",
    "EncodedDetection",
    "GroundingBatch",
    "GroundingBudget",
    "GroundingCandidate",
    "GroundingComparisonHarness",
    "GroundingComparisonReport",
    "GroundingDetection",
    "GroundingInput",
    "GroundingRequest",
    "GroundingRunReport",
    "GroundingStatus",
    "PhraseSpec",
    "PromptPlan",
    "PromptRegistry",
    "RawDetection",
    "UnavailableDetectorAdapter",
    "UnavailableClosedSetAdapter",
    "UnavailableGroundingAdapter",
    "decode_box",
    "decode_model_detections",
    "postprocess_detections",
]
