"""Data provenance and synthetic fixture utilities."""

from video_intelligence.data.dataset import (
    REAL_QUALITY_SPLITS,
    DatasetKind,
    DatasetManifest,
    DatasetManifestError,
    DatasetPurpose,
    SyntheticDataUsageError,
    load_dataset_manifest,
    require_real_quality,
    require_synthetic_contract,
    write_dataset_manifest,
)
from video_intelligence.data.intake import (
    REAL_ANNOTATION_SCHEMA,
    REAL_DETECTION_ANNOTATION_SCHEMA,
    RealDatasetIntakeError,
    build_real_quality_manifest,
    load_verified_real_annotations,
    validate_real_quality_split_policy,
)

__all__ = [
    "DatasetKind",
    "DatasetManifest",
    "DatasetManifestError",
    "DatasetPurpose",
    "REAL_QUALITY_SPLITS",
    "REAL_ANNOTATION_SCHEMA",
    "REAL_DETECTION_ANNOTATION_SCHEMA",
    "RealDatasetIntakeError",
    "SyntheticDataUsageError",
    "load_dataset_manifest",
    "require_real_quality",
    "require_synthetic_contract",
    "write_dataset_manifest",
    "build_real_quality_manifest",
    "load_verified_real_annotations",
    "validate_real_quality_split_policy",
]
