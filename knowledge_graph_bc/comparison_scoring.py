from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ComparisonScoringWeights:
    issue_overlap: float = 5.0
    shared_keyword: float = 2.0
    shared_entity: float = 3.0
    semantic_similarity: float = 4.0
    analysis_success: float = 1.0
    view_score: float = 1.0
    published_at: float = 0.3


@dataclass(frozen=True)
class ComparisonScoringFeatures:
    issue_overlap_count: int = 0
    shared_keyword_count: int = 0
    shared_entity_count: int = 0
    semantic_similarity: float = 0.0
    analysis_success: bool = False
    view_score: float = 0.0
    published_at: bool = False


DEFAULT_COMPARISON_SCORING_WEIGHTS = ComparisonScoringWeights()


def clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def features_from_mapping(raw: dict[str, Any] | None) -> ComparisonScoringFeatures:
    data = raw or {}
    return ComparisonScoringFeatures(
        issue_overlap_count=max(0, int(data.get("issue_overlap_count") or 0)),
        shared_keyword_count=max(0, int(data.get("shared_keyword_count") or 0)),
        shared_entity_count=max(0, int(data.get("shared_entity_count") or 0)),
        semantic_similarity=clamp_unit(data.get("semantic_similarity")),
        analysis_success=bool(data.get("analysis_success")),
        view_score=clamp_unit(data.get("view_score")),
        published_at=bool(data.get("published_at")),
    )


def weights_from_mapping(raw: dict[str, Any] | None) -> ComparisonScoringWeights:
    data = raw or {}
    baseline = asdict(DEFAULT_COMPARISON_SCORING_WEIGHTS)
    for key in baseline:
        if key not in data:
            continue
        try:
            baseline[key] = max(0.0, float(data[key]))
        except (TypeError, ValueError):
            continue
    return ComparisonScoringWeights(**baseline)


def with_weight_multiplier(
    weights: ComparisonScoringWeights,
    field_name: str,
    multiplier: float,
) -> ComparisonScoringWeights:
    if not hasattr(weights, field_name):
        raise ValueError(f"unknown comparison weight: {field_name}")
    return replace(weights, **{field_name: max(0.0, getattr(weights, field_name) * multiplier)})


def score_comparison_features(
    features: ComparisonScoringFeatures,
    weights: ComparisonScoringWeights = DEFAULT_COMPARISON_SCORING_WEIGHTS,
) -> float:
    return (
        features.issue_overlap_count * weights.issue_overlap
        + features.shared_keyword_count * weights.shared_keyword
        + features.shared_entity_count * weights.shared_entity
        + clamp_unit(features.semantic_similarity) * weights.semantic_similarity
        + (weights.analysis_success if features.analysis_success else 0.0)
        + clamp_unit(features.view_score) * weights.view_score
        + (weights.published_at if features.published_at else 0.0)
    )


def score_breakdown(
    features: ComparisonScoringFeatures,
    weights: ComparisonScoringWeights = DEFAULT_COMPARISON_SCORING_WEIGHTS,
) -> dict[str, float]:
    return {
        "issue_overlap": features.issue_overlap_count * weights.issue_overlap,
        "shared_keyword": features.shared_keyword_count * weights.shared_keyword,
        "shared_entity": features.shared_entity_count * weights.shared_entity,
        "semantic_similarity": clamp_unit(features.semantic_similarity) * weights.semantic_similarity,
        "analysis_success": weights.analysis_success if features.analysis_success else 0.0,
        "view_score": clamp_unit(features.view_score) * weights.view_score,
        "published_at": weights.published_at if features.published_at else 0.0,
    }
