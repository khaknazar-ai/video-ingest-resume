"""Versioned model-facing views; full evidence/provenance stays in the host request."""

from video_intelligence.reasoning.semantic import SemanticRequest

FULL_VIEW = "full_request.v1"
EVIDENCE_VIEW = "evidence_only.v1"


def validate_input_view(value):
    if value not in (FULL_VIEW, EVIDENCE_VIEW):
        raise ValueError("unsupported semantic input view")
    return value


def model_input_view(request: SemanticRequest, input_view=FULL_VIEW):
    """Remove detector assertions through an allowlist, without removing candidates.

    This is an experimental input policy, not a qualified semantic correction.
    Query entities remain the claim to assess. Scores, decoded labels and arbitrary
    upstream provenance are retained in the full hash-bound host request only.
    """
    validate_input_view(input_view)
    body = request.to_dict()
    if input_view == FULL_VIEW:
        return body
    view = {key: body[key] for key in (
        "query", "frames", "units", "coverage_complete", "gaps", "omitted_evidence_ids",
        "source_sha256", "snapshot_id",
    )}
    view.update(schema_version="video_intelligence.semantic_evidence_view.v1",
                input_view=EVIDENCE_VIEW, source_request_sha256=request.sha256)
    view["candidates"] = [{key: item[key] for key in (
        "evidence_id", "frame_ref", "bbox_xyxy", "role",
    )} for item in body["candidates"]]
    return view
