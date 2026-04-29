from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import BiasKeywordType, SentenceLabelType
from analysis_bc.keyword_extractor import KeywordExtractor
from analysis_bc.schemas import BiasAnalysisKeywordDto, SentenceInputDto, SpanLabelDto


def _sentence(cid: int, text: str, label: str = "opinion_like", conf: float = 0.9) -> ClassifiedSentenceDto:
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
    matched_word: str | None = "분노",
) -> SpanLabelDto:
    return SpanLabelDto(
        content_sentence_id=cid,
        start_offset=0,
        end_offset=2,
        label_type=label,
        score=score,
        matched_word=matched_word,
    )


@pytest.fixture
def extractor():
    with patch("analysis_bc.keyword_extractor.Kiwi") as MockKiwi:
        mock_kiwi = MagicMock()
        MockKiwi.return_value = mock_kiwi
        yield KeywordExtractor(), mock_kiwi


class TestKeywordExtractorEmpty:
    def test_empty_all_inputs(self, extractor):
        kw_extractor, _ = extractor
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=[])
        assert result == []

    def test_no_emotion_span(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        classified = [_sentence(1, "정부가 발표했다", label="fact_like", conf=0.85)]
        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])
        assert all(k.keyword_type != BiasKeywordType.EMOTION for k in result)


class TestKeywordExtractorEmotion:
    def test_emotion_uses_matched_word(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.9, matched_word="분노")
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=[span])
        emotion_kws = [k for k in result if k.keyword_type == BiasKeywordType.EMOTION]
        assert len(emotion_kws) == 1
        assert emotion_kws[0].keyword_text == "분노"
        assert emotion_kws[0].score == 0.9

    def test_emotion_span_without_matched_word_skipped(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, matched_word=None)
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=[span])
        assert result == []


class TestKeywordExtractorFilter:
    def test_min_length_filter(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, matched_word="화")
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=[span])
        assert result == []

    def test_dedup_keeps_higher_score(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.7, matched_word="분노"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=0.95, matched_word="분노"),
        ]
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)
        emotion_kws = [k for k in result if k.keyword_text == "분노"]
        assert len(emotion_kws) == 1
        assert emotion_kws[0].score == 0.95

    def test_top5_limit(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        words = ["단어가", "단어나", "단어다", "단어라", "단어마", "단어바"]
        spans = [
            _span(i, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5 + i * 0.05, matched_word=w)
            for i, w in enumerate(words)
        ]
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)
        emotion_kws = [k for k in result if k.keyword_type == BiasKeywordType.EMOTION]
        assert len(emotion_kws) <= 5


class TestKeywordExtractorMorpheme:
    def _make_token(self, form: str, tag: str) -> MagicMock:
        t = MagicMock()
        t.form = form
        t.tag = tag
        return t

    def test_topic_uses_fact_sentences(self, extractor):
        kw_extractor, mock_kiwi = extractor
        token = self._make_token("부동산", "NNG")
        mock_kiwi.tokenize.return_value = [token]
        classified = [_sentence(1, "부동산 가격이 올랐다", label="fact_like", conf=0.8)]
        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])
        topic_kws = [k for k in result if k.keyword_type == BiasKeywordType.TOPIC]
        assert any(k.keyword_text == "부동산" for k in topic_kws)

    def test_frame_uses_opinion_sentences(self, extractor):
        kw_extractor, mock_kiwi = extractor
        token = self._make_token("붕괴하다", "VV")
        mock_kiwi.tokenize.return_value = [token]
        classified = [_sentence(1, "정책이 붕괴하다", label="opinion_like", conf=0.75)]
        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])
        frame_kws = [k for k in result if k.keyword_type == BiasKeywordType.FRAME]
        assert any(k.keyword_text == "붕괴하다" for k in frame_kws)
