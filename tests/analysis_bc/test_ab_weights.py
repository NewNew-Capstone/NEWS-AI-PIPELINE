"""
A/B 가중치 variant 검증 테스트.
ML/Qdrant/API 의존성 없이 극단 fixture로 동작.
"""
from __future__ import annotations

import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.scorer import BiasScorer, ScorerWeights

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------
_N = 5
_CONFIDENCE = 0.95
_GAP_A = 0.8   # 극단 opinion 기사 (사설) — 제목-본문 갭 높음
_GAP_B = 0.1   # 극단 fact 기사 (보도자료) — 갭 낮음

WEIGHT_VARIANTS = [
    ScorerWeights(),
    ScorerWeights(w_opinion=0.4, w_emotion=0.6),
    ScorerWeights(
        w_opinion=0.7,
        w_emotion=0.3,
        position_weight_front=1.0,
        position_weight_mid=1.0,
        position_weight_back=1.0,
    ),
]
WEIGHT_IDS = ["default", "emotion_heavy", "position_flat"]


# ------------------------------------------------------------------
# fixture helpers
# ------------------------------------------------------------------

def _opinion_classified(n: int = _N) -> list[ClassifiedSentenceDto]:
    return [
        ClassifiedSentenceDto(
            content_sentence_id=i,
            sentence_text=f"의견문장{i}",
            sentence_order=i,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=_CONFIDENCE,
        )
        for i in range(1, n + 1)
    ]


def _emotion_spans(n: int = _N) -> list[SpanLabelDto]:
    return [
        SpanLabelDto(
            content_sentence_id=i,
            start_offset=0,
            end_offset=5,
            label_type=SentenceLabelType.EMOTIONALLY_LOADED,
            score=0.95,
        )
        for i in range(1, n + 1)
    ]


def _fact_classified(n: int = _N) -> list[ClassifiedSentenceDto]:
    return [
        ClassifiedSentenceDto(
            content_sentence_id=i,
            sentence_text=f"사실문장{i}",
            sentence_order=i,
            start_time_ms=None,
            end_time_ms=None,
            label="fact_like",
            confidence=_CONFIDENCE,
        )
        for i in range(1, n + 1)
    ]


# ------------------------------------------------------------------
# 점수 범위 검증
# ------------------------------------------------------------------

@pytest.mark.parametrize("weights", WEIGHT_VARIANTS, ids=WEIGHT_IDS)
class TestScoreRange:
    """모든 점수는 0.0~1.0 범위 안에 있어야 한다."""

    def test_opinion_fixture_scores_in_range(self, weights: ScorerWeights) -> None:
        result = BiasScorer(weights).calculate(
            classified=_opinion_classified(),
            span_labels=_emotion_spans(),
            headline_body_gap=_GAP_A,
        )
        assert 0.0 <= result["overall_bias_score"] <= 1.0
        assert 0.0 <= result["opinion_score"] <= 1.0
        assert 0.0 <= result["emotion_score"] <= 1.0
        assert 0.0 <= result["subjectivity_score"] <= 100.0

    def test_fact_fixture_scores_in_range(self, weights: ScorerWeights) -> None:
        result = BiasScorer(weights).calculate(
            classified=_fact_classified(),
            span_labels=[],
            headline_body_gap=_GAP_B,
        )
        assert 0.0 <= result["overall_bias_score"] <= 1.0
        assert 0.0 <= result["opinion_score"] <= 1.0
        assert 0.0 <= result["emotion_score"] <= 1.0
        assert 0.0 <= result["subjectivity_score"] <= 100.0


# ------------------------------------------------------------------
# 극단 케이스 방향성 검증
# ------------------------------------------------------------------

@pytest.mark.parametrize("weights", WEIGHT_VARIANTS, ids=WEIGHT_IDS)
def test_opinion_always_higher_than_fact(weights: ScorerWeights) -> None:
    """A타입(극단 opinion)은 B타입(극단 fact)보다 overall_bias_score가 항상 높아야 한다."""
    opinion_result = BiasScorer(weights).calculate(
        classified=_opinion_classified(),
        span_labels=_emotion_spans(),
        headline_body_gap=_GAP_A,
    )
    fact_result = BiasScorer(weights).calculate(
        classified=_fact_classified(),
        span_labels=[],
        headline_body_gap=_GAP_B,
    )
    assert opinion_result["overall_bias_score"] > fact_result["overall_bias_score"], (
        f"[{weights}] opinion={opinion_result['overall_bias_score']} "
        f"should > fact={fact_result['overall_bias_score']}"
    )


# ------------------------------------------------------------------
# 기본값 호환성
# ------------------------------------------------------------------

def test_default_weights_backward_compatible() -> None:
    """ScorerWeights() 기본값이 기존 하드코딩 상수와 동일해야 한다."""
    w = ScorerWeights()
    assert w.w_opinion == pytest.approx(0.7)
    assert w.w_emotion == pytest.approx(0.3)
    assert w.gap_weight_slope == pytest.approx(0.3)
    assert w.position_weight_front == pytest.approx(1.3)
    assert w.position_weight_mid == pytest.approx(1.0)
    assert w.position_weight_back == pytest.approx(0.8)
    assert w.position_front_threshold == pytest.approx(0.33)
    assert w.position_mid_threshold == pytest.approx(0.66)
    assert w.gap_evidence_high_threshold == pytest.approx(0.7)
    assert w.gap_evidence_mid_threshold == pytest.approx(0.4)


# ------------------------------------------------------------------
# 커스텀 가중치 반영 확인
# ------------------------------------------------------------------

def test_custom_weights_applied() -> None:
    """emotion_heavy variant는 default와 다른 overall_bias_score를 반환해야 한다.

    감정 스팬을 절반(2개)만 사용해 emotion_score=0.4로 설정한다.
    전체 스팬 사용 시 두 variant 모두 상한(1.0)에 걸려 동일해지는 것을 방지.
    """
    classified = _opinion_classified()
    partial_spans = _emotion_spans(n=2)  # emotion_score = 2/5 = 0.4

    default_result = BiasScorer(ScorerWeights()).calculate(
        classified, partial_spans, _GAP_A
    )
    heavy_result = BiasScorer(ScorerWeights(w_opinion=0.4, w_emotion=0.6)).calculate(
        classified, partial_spans, _GAP_A
    )
    assert default_result["overall_bias_score"] != heavy_result["overall_bias_score"]


# ------------------------------------------------------------------
# worst-case 안정성
# ------------------------------------------------------------------

@pytest.mark.parametrize("weights", WEIGHT_VARIANTS, ids=WEIGHT_IDS)
def test_weight_sum_stability(weights: ScorerWeights) -> None:
    """gap=1.0 worst case에서도 overall_bias_score <= 1.0 을 유지해야 한다."""
    result = BiasScorer(weights).calculate(
        classified=_opinion_classified(),
        span_labels=_emotion_spans(),
        headline_body_gap=1.0,
    )
    assert result["overall_bias_score"] <= 1.0


# ------------------------------------------------------------------
# 잘못된 가중치 검증
# ------------------------------------------------------------------

def test_invalid_weights_raise() -> None:
    """w_opinion + w_emotion != 1.0 이면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError, match="must equal 1.0"):
        ScorerWeights(w_opinion=0.6, w_emotion=0.6)
