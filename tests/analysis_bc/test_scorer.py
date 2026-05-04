from __future__ import annotations

import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.scorer import BiasScorer


# ------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------

def _make_classified(
    sentence_order: int,
    label: str = "opinion_like",
    confidence: float = 0.9,
) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=sentence_order,
        sentence_text=f"문장{sentence_order}",
        sentence_order=sentence_order,
        start_time_ms=None,
        end_time_ms=None,
        label=label,
        confidence=confidence,
    )


def _make_span(
    label_type: SentenceLabelType,
    content_sentence_id: int = 1,
    score: float = 0.9,
) -> SpanLabelDto:
    return SpanLabelDto(
        content_sentence_id=content_sentence_id,
        start_offset=0,
        end_offset=5,
        label_type=label_type,
        score=score,
    )


# ------------------------------------------------------------------
# tests
# ------------------------------------------------------------------

class TestBiasScorer:
    def setup_method(self) -> None:
        self.scorer = BiasScorer()

    def test_empty_input(self) -> None:
        result = self.scorer.calculate(classified=[], span_labels=[], headline_body_gap=0.0)
        assert result["opinion_score"] == 0.0
        assert result["emotion_score"] == 0.0
        assert result["fact_ratio"] == 0.0
        assert result["overall_bias_score"] == 0.0
        assert result["score_evidence"] == ""

    def test_all_fact(self) -> None:
        classified = [_make_classified(i, label="fact_like", confidence=0.99) for i in range(1, 6)]
        result = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.0)
        assert result["opinion_score"] == 0.0
        assert result["fact_ratio"] == 1.0
        assert result["overall_bias_score"] == pytest.approx(
            0.4 * 0.0 + 0.3 * 0.0 + 0.3 * (1 - 1.0), abs=1e-4
        )

    def test_all_opinion_positive_score(self) -> None:
        classified = [_make_classified(i, label="opinion_like", confidence=1.0) for i in range(1, 6)]
        result = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.0)
        assert result["opinion_score"] > 0.0
        assert result["overall_bias_score"] > 0.0

    def test_low_fact_ratio_evidence(self) -> None:
        # opinion 문장만 있으면 fact_ratio=0.0 → "사실 기반 문장 비율이 낮습니다." 포함
        classified = [
            _make_classified(i, label="opinion_like", confidence=0.9)
            for i in range(1, 6)
        ]
        result = self.scorer.calculate(
            classified=classified, span_labels=[], headline_body_gap=0.0
        )
        assert "사실 기반" in result["score_evidence"]

    def test_opinion_ratio_increases_score(self) -> None:
        classified_all_opinion = [
            _make_classified(i, label="opinion_like", confidence=0.9)
            for i in range(1, 6)
        ]
        classified_half_opinion = (
            [_make_classified(i, label="opinion_like", confidence=0.9) for i in range(1, 4)]
            + [_make_classified(i, label="fact_like", confidence=0.9) for i in range(4, 6)]
        )
        score_all = self.scorer.calculate(
            classified=classified_all_opinion, span_labels=[], headline_body_gap=0.0
        )["overall_bias_score"]
        score_half = self.scorer.calculate(
            classified=classified_half_opinion, span_labels=[], headline_body_gap=0.0
        )["overall_bias_score"]
        assert score_all > score_half

    def test_bias_type_scores_keys(self) -> None:
        classified = [_make_classified(1)]
        result = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.0)
        assert set(result["bias_type_scores"].keys()) == {"OPINION", "EMOTIONAL", "FACT"}

    def test_score_evidence_emotional(self) -> None:
        classified = [_make_classified(1)]
        span_labels = [_make_span(SentenceLabelType.EMOTIONALLY_LOADED)]
        result = self.scorer.calculate(classified=classified, span_labels=span_labels, headline_body_gap=0.0)
        assert "감정적 표현" in result["score_evidence"]
        assert result["emotion_score"] > 0.0

    def test_overall_bias_score_max_1(self) -> None:
        classified = [
            _make_classified(i, label="opinion_like", confidence=1.0)
            for i in range(1, 11)
        ]
        result = self.scorer.calculate(
            classified=classified, span_labels=[], headline_body_gap=1.0
        )
        assert result["overall_bias_score"] <= 1.0

    def test_overall_bias_score_weighted_no_spans(self) -> None:
        # span 없음 → emotion=0, fact_ratio=0
        # overall = 0.4 * opinion + 0.3 * 0 + 0.3 * (1 - 0)
        classified = [_make_classified(i) for i in range(1, 6)]
        result = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.5)
        expected = pytest.approx(
            0.4 * result["opinion_score"]
            + 0.3 * result["emotion_score"]
            + 0.3 * (1 - result["fact_ratio"]),
            abs=1e-4,
        )
        assert result["overall_bias_score"] == expected

    def test_overall_includes_emotion(self) -> None:
        # emotion span 있을 때 overall이 opinion-only보다 높아야 함
        classified = [_make_classified(i) for i in range(1, 6)]
        result_no_span = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.0)
        span_labels = [_make_span(SentenceLabelType.EMOTIONALLY_LOADED, i) for i in range(1, 6)]
        result_with_span = self.scorer.calculate(classified=classified, span_labels=span_labels, headline_body_gap=0.0)
        assert result_with_span["overall_bias_score"] > result_no_span["overall_bias_score"]

    def test_emotion_score_uses_span_confidence(self) -> None:
        classified = [_make_classified(1), _make_classified(2)]
        low = self.scorer.calculate(
            classified=classified,
            span_labels=[_make_span(SentenceLabelType.EMOTIONALLY_LOADED, 1, score=0.2)],
            headline_body_gap=0.0,
        )
        high = self.scorer.calculate(
            classified=classified,
            span_labels=[_make_span(SentenceLabelType.EMOTIONALLY_LOADED, 1, score=0.9)],
            headline_body_gap=0.0,
        )
        assert high["emotion_score"] > low["emotion_score"]

    def test_emotion_score_applies_position_weight(self) -> None:
        classified = [_make_classified(i) for i in range(1, 7)]
        front = self.scorer.calculate(
            classified=classified,
            span_labels=[_make_span(SentenceLabelType.EMOTIONALLY_LOADED, 1, score=0.8)],
            headline_body_gap=0.0,
        )
        back = self.scorer.calculate(
            classified=classified,
            span_labels=[_make_span(SentenceLabelType.EMOTIONALLY_LOADED, 6, score=0.8)],
            headline_body_gap=0.0,
        )
        assert front["emotion_score"] > back["emotion_score"]

    def test_emotion_sentence_intensity_is_capped(self) -> None:
        classified = [_make_classified(1), _make_classified(2)]
        duplicate_spans = [
            _make_span(SentenceLabelType.EMOTIONALLY_LOADED, 1, score=0.9),
            _make_span(SentenceLabelType.EMOTIONALLY_LOADED, 1, score=0.9),
        ]
        result = self.scorer.calculate(
            classified=classified,
            span_labels=duplicate_spans,
            headline_body_gap=0.0,
        )
        assert result["emotion_score"] == pytest.approx(0.65)

    def test_gap_evidence_high(self) -> None:
        classified = [_make_classified(1)]
        result = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.8)
        assert "차이가 큽니다" in result["score_evidence"]

    def test_gap_evidence_medium(self) -> None:
        classified = [_make_classified(1)]
        result = self.scorer.calculate(classified=classified, span_labels=[], headline_body_gap=0.5)
        assert "다소 차이" in result["score_evidence"]
