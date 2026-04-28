from __future__ import annotations

import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import EvidenceType, SentenceLabelType
from analysis_bc.evidence_extractor import EvidenceExtractor
from analysis_bc.schemas import SentenceInputDto, SpanLabelDto


def _input_sentence(cid: int, text: str) -> SentenceInputDto:
    return SentenceInputDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=cid,
    )


def _classified(cid: int, text: str, label: str = "opinion_like", conf: float = 0.85) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=cid,
        start_time_ms=None,
        end_time_ms=None,
        label=label,
        confidence=conf,
    )


def _span(
    cid: int,
    label: SentenceLabelType,
    score: float = 0.8,
    matched_word: str | None = None,
    start: int = 0,
    end: int = 3,
) -> SpanLabelDto:
    return SpanLabelDto(
        content_sentence_id=cid,
        start_offset=start,
        end_offset=end,
        label_type=label,
        score=score,
        matched_word=matched_word,
    )


@pytest.fixture
def extractor() -> EvidenceExtractor:
    return EvidenceExtractor()


class TestEvidenceExtractorEmpty:
    def test_all_empty(self, extractor):
        result = extractor.extract(sentences=[], classified=[], span_labels=[])
        assert result == []

    def test_no_span_labels(self, extractor):
        sentences = [_input_sentence(1, "정부가 정책을 발표했다")]
        classified = [_classified(1, "정부가 정책을 발표했다", label="fact_like")]
        result = extractor.extract(sentences=sentences, classified=classified, span_labels=[])
        assert result == []


class TestEvidenceExtractorMapping:
    def test_emotionally_loaded_maps_to_emotion(self, extractor):
        sentences = [_input_sentence(1, "최악의 선택이다")]
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.9, matched_word="최악")
        result = extractor.extract(sentences=sentences, classified=[], span_labels=[span])
        assert len(result) == 1
        assert result[0].evidence_type == EvidenceType.EMOTION
        assert result[0].title == "감정적 표현"
        assert result[0].confidence_score == 0.9

    def test_opinion_like_classified_adds_evidence(self, extractor):
        sentences = [_input_sentence(1, "이건 잘못된 정책이다")]
        classified = [_classified(1, "이건 잘못된 정책이다", label="opinion_like", conf=0.88)]
        result = extractor.extract(sentences=sentences, classified=classified, span_labels=[])
        opinion_evs = [e for e in result if e.evidence_type == EvidenceType.OPINION]
        assert len(opinion_evs) == 1
        assert opinion_evs[0].title == "주관적 의견"
        assert opinion_evs[0].confidence_score == 0.88

    def test_fact_like_classified_excluded(self, extractor):
        sentences = [_input_sentence(1, "GDP가 3% 성장했다")]
        classified = [_classified(1, "GDP가 3% 성장했다", label="fact_like")]
        result = extractor.extract(sentences=sentences, classified=classified, span_labels=[])
        assert result == []


class TestEvidenceExtractorSourceText:
    def test_source_text_uses_matched_word(self, extractor):
        sentences = [_input_sentence(1, "분노한 시민들이 모였다")]
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, matched_word="분노", start=0, end=2)
        result = extractor.extract(sentences=sentences, classified=[], span_labels=[span])
        assert result[0].source_text == "분노"

    def test_source_text_fallback_to_slice(self, extractor):
        sentences = [_input_sentence(1, "관계자에 따르면")]
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, matched_word=None, start=0, end=3)
        result = extractor.extract(sentences=sentences, classified=[], span_labels=[span])
        assert result[0].source_text == "관계자"

    def test_description_is_full_sentence(self, extractor):
        text = "최악의 결정이라고 할 수 있다"
        sentences = [_input_sentence(1, text)]
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, matched_word="최악")
        result = extractor.extract(sentences=sentences, classified=[], span_labels=[span])
        assert result[0].description == text
