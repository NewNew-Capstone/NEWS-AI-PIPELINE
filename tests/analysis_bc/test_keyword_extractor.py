from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import BiasKeywordType, SentenceLabelType
from analysis_bc.keyword_extractor import KeywordExtractor
from analysis_bc.schemas import SpanLabelDto


def _token(form: str, tag: str, start: int = 0, length: int | None = None) -> MagicMock:
    token = MagicMock()
    token.form = form
    token.tag = tag
    token.start = start
    token.len = len(form) if length is None else length
    return token


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
        classified = [_sentence(1, "정부가 정책을 발표했다", label="fact_like", conf=0.85)]
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
        assert emotion_kws[0].score == 1.0

    def test_emotion_predicate_matched_words_are_grouped_by_base_form(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("재밌", "VA", 0, 2)],
            [_token("싸우", "VV", 0, 2)],
            [_token("죄송하", "VA", 0, 3)],
        ]
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5, matched_word="재밌어요"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5, matched_word="싸우고"),
            _span(3, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word="죄송합니다"),
        ]

        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)

        scores = {k.keyword_text: k.score for k in result if k.keyword_type == BiasKeywordType.EMOTION}
        assert scores == {
            "재밌다": 0.5,
            "싸우다": 0.5,
        }
        assert "죄송하다" not in scores

    def test_emotion_bare_predicate_stem_keyword_uses_base_form(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = [_token("재밌", "VA", 0, 2)]
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word="재밌")

        result = kw_extractor.extract(sentences=[], classified=[], span_labels=[span])

        emotion_kws = [k for k in result if k.keyword_type == BiasKeywordType.EMOTION]
        assert len(emotion_kws) == 1
        assert emotion_kws[0].keyword_text == "재밌다"

    def test_emotion_xr_hada_keyword_uses_base_form(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = [
            _token("끔찍", "XR", 0, 2),
            _token("하", "XSA", 2, 1),
        ]
        span = _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word="끔찍하다")

        result = kw_extractor.extract(sentences=[], classified=[], span_labels=[span])

        emotion_kws = [k for k in result if k.keyword_type == BiasKeywordType.EMOTION]
        assert len(emotion_kws) == 1
        assert emotion_kws[0].keyword_text == "끔찍하다"

    def test_emotion_stopword_after_normalization_is_skipped(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("궁금", "XR", 0, 2), _token("하", "XSA", 2, 1)],
            [_token("분노", "NNG", 0, 2)],
        ]
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word="궁금하실"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word="분노"),
        ]

        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)

        scores = {k.keyword_text: k.score for k in result if k.keyword_type == BiasKeywordType.EMOTION}
        assert scores == {"분노": 1.0}

    def test_emotion_nouns_keep_existing_keyword_text(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("불안", "NNG", 0, 2)],
            [_token("분노", "NNG", 0, 2)],
        ]
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5, matched_word="불안"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5, matched_word="분노"),
        ]

        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)

        scores = {k.keyword_text: k.score for k in result if k.keyword_type == BiasKeywordType.EMOTION}
        assert scores == {"불안": 0.5, "분노": 0.5}

    def test_extract_emotion_filter_candidates_uses_all_normalized_spans(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("아름답", "VA", 0, 3)],
            [_token("분노", "NNG", 0, 2)],
            [_token("분노", "NNG", 0, 2)],
        ]
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.9, matched_word="아름답다"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=0.8, matched_word="분노"),
            _span(3, SentenceLabelType.EMOTIONALLY_LOADED, score=0.7, matched_word="분노"),
        ]

        result = kw_extractor.extract_emotion_filter_candidates(spans)

        assert result == ["아름답다", "분노"]

    def test_emotion_score_is_relative_importance(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5, matched_word="분노"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=0.5, matched_word="분노"),
            _span(3, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word="불안"),
        ]
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)
        scores = {k.keyword_text: k.score for k in result if k.keyword_type == BiasKeywordType.EMOTION}
        assert scores == {"분노": 0.5, "불안": 0.5}

    def test_emotion_filter_preserves_nouns_when_predicates_dominate(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("재밌", "VA", 0, 2)],
            [_token("싸우", "VV", 0, 2)],
            [_token("놀라", "VV", 0, 2)],
            [_token("반갑", "VA", 0, 2)],
            [_token("기쁘", "VA", 0, 2)],
            [_token("즐겁", "VA", 0, 2)],
            [_token("분노", "NNG", 0, 2)],
        ]
        spans = [
            _span(index, SentenceLabelType.EMOTIONALLY_LOADED, score=1.0, matched_word=word)
            for index, word in enumerate(
                ["재밌어요", "싸우고", "놀라다", "반갑다", "기쁘다", "즐겁다", "분노"],
                start=1,
            )
        ]

        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)

        emotion_kws = [k.keyword_text for k in result if k.keyword_type == BiasKeywordType.EMOTION]
        assert len(emotion_kws) == 5
        assert "분노" in emotion_kws

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

    def test_stopword_filter(self, extractor):
        kw_extractor, mock_kiwi = extractor
        news = MagicMock()
        news.form = "뉴스"
        news.tag = "NNG"
        taiwan = MagicMock()
        taiwan.form = "대만"
        taiwan.tag = "NNP"
        related = MagicMock()
        related.form = "관련"
        related.tag = "NNG"
        mock_kiwi.tokenize.return_value = [news, taiwan, related]
        classified = [_sentence(1, "뉴스 대만 관련", label="fact_like", conf=0.8)]

        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])

        topic_kws = [k.keyword_text for k in result if k.keyword_type == BiasKeywordType.TOPIC]
        assert topic_kws == ["대만"]

    def test_dedup_sums_duplicate_weights(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = []
        spans = [
            _span(1, SentenceLabelType.EMOTIONALLY_LOADED, score=0.7, matched_word="분노"),
            _span(2, SentenceLabelType.EMOTIONALLY_LOADED, score=0.95, matched_word="분노"),
        ]
        result = kw_extractor.extract(sentences=[], classified=[], span_labels=spans)
        emotion_kws = [k for k in result if k.keyword_text == "분노"]
        assert len(emotion_kws) == 1
        assert emotion_kws[0].score == 1.0

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
        return _token(form, tag)

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
        token = self._make_token("비판하다", "VV")
        mock_kiwi.tokenize.return_value = [token]
        classified = [_sentence(1, "정책을 비판하다", label="opinion_like", conf=0.75)]
        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])
        frame_kws = [k for k in result if k.keyword_type == BiasKeywordType.FRAME]
        assert any(k.keyword_text == "비판하다" for k in frame_kws)

    def test_frame_normalizes_verbs_and_preserves_nouns(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [
                self._make_token("강조하", "VV"),
                self._make_token("밝히", "VV"),
                self._make_token("주장하", "VV"),
                self._make_token("내세우", "VV"),
                self._make_token("꺼내", "VV"),
                self._make_token("정책", "NNG"),
            ],
            [
                self._make_token("강조하", "VV"),
                self._make_token("밝히", "VV"),
                self._make_token("주장하", "VV"),
                self._make_token("내세우", "VV"),
                self._make_token("꺼내", "VV"),
            ],
        ]
        classified = [
            _sentence(1, "정책을 강조하고 밝혔다", label="opinion_like", conf=0.9),
            _sentence(2, "다시 강조하고 밝혔다", label="opinion_like", conf=0.9),
        ]

        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])

        frame_kws = [k.keyword_text for k in result if k.keyword_type == BiasKeywordType.FRAME]
        assert "정책" in frame_kws
        assert "강조하다" in frame_kws
        assert "강조하" not in frame_kws

    def test_morpheme_score_is_relative_importance(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [self._make_token("정책", "NNG"), self._make_token("실패", "NNG")],
            [self._make_token("정책", "NNG")],
        ]
        classified = [
            _sentence(1, "정책 실패", label="opinion_like", conf=0.5),
            _sentence(2, "정책", label="opinion_like", conf=0.5),
        ]

        result = kw_extractor.extract(sentences=[], classified=classified, span_labels=[])

        scores = {k.keyword_text: k.score for k in result if k.keyword_type == BiasKeywordType.FRAME}
        assert scores == {"정책": 0.6667, "실패": 0.3333}


class TestKeywordExtractorFocus:
    def test_focus_keywords_use_repeated_terms_from_all_sentences(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("정책", "NNG", 0, 2), _token("실패", "NNG", 3, 2)],
            [_token("정책", "NNG", 0, 2), _token("논란", "NNG", 3, 2)],
            [_token("정책", "NNG", 0, 2), _token("발표", "NNG", 3, 2)],
        ]
        classified = [
            _sentence(1, "정책 실패입니다", label="opinion_like"),
            _sentence(2, "정책 논란입니다", label="opinion_like"),
            _sentence(3, "정책 발표입니다", label="fact_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        assert len(result) == 1
        assert result[0].keyword_text == "정책"
        assert result[0].occurrence_count == 3
        assert result[0].sentence_count == 3
        assert result[0].score == 1.0

    def test_focus_keywords_include_fact_only_repetition(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("정책", "NNG", 0, 2), _token("발표", "NNG", 3, 2)],
            [_token("정책", "NNG", 0, 2), _token("설명", "NNG", 3, 2)],
        ]
        classified = [
            _sentence(1, "정책 발표입니다", label="fact_like"),
            _sentence(2, "정책 설명입니다", label="fact_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        assert len(result) == 1
        assert result[0].keyword_text == "정책"
        assert result[0].occurrence_count == 2
        assert result[0].sentence_count == 2

    def test_focus_keywords_require_repeated_sentences(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.return_value = [
            _token("정책", "NNG", 0, 2),
            _token("정책", "NNG", 3, 2),
        ]
        classified = [_sentence(1, "정책 정책", label="opinion_like")]

        result = kw_extractor.extract_focus_keywords(classified)

        assert result == []

    def test_focus_keywords_filter_stopwords_and_short_tokens(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("오늘", "NNG", 0, 2), _token("나", "NNG", 3, 1)],
            [_token("오늘", "NNG", 0, 2), _token("나", "NNG", 3, 1)],
        ]
        classified = [
            _sentence(1, "오늘 나", label="opinion_like"),
            _sentence(2, "오늘 나", label="opinion_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        assert result == []

    def test_focus_keywords_group_verbs_by_base_form(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token("싸우", "VV", 0, 2)],
            [_token("싸우", "VV", 0, 2)],
            [_token("싸우", "VV", 0, 2)],
        ]
        classified = [
            _sentence(1, "싸우고 있다", label="opinion_like"),
            _sentence(2, "싸운다", label="opinion_like"),
            _sentence(3, "싸웠다", label="fact_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        assert len(result) == 1
        assert result[0].keyword_text == "싸우다"
        assert result[0].occurrence_count == 3
        assert result[0].sentence_count == 3

    def test_focus_keywords_exclude_adjectives_roots_and_adverbs(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [
                _token("똑같", "VA", 0, 2),
                _token("궁금", "XR", 4, 2),
                _token("매우", "MAG", 8, 2),
            ],
            [
                _token("똑같", "VA", 0, 2),
                _token("궁금", "XR", 4, 2),
                _token("매우", "MAG", 8, 2),
            ],
        ]
        classified = [
            _sentence(1, "똑같다 궁금하다 매우", label="opinion_like"),
            _sentence(2, "똑같다 궁금하다 매우", label="fact_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        assert result == []

    def test_focus_keywords_limit_and_sort_top5(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [_token(word, "NNG") for word in ["가가", "나나", "다다", "라라", "마마", "바바"]],
            [_token(word, "NNG") for word in ["가가", "나나", "다다", "라라", "마마", "바바"]],
            [_token(word, "NNG") for word in ["가가", "나나"]],
        ]
        classified = [
            _sentence(1, "첫 의견", label="opinion_like"),
            _sentence(2, "둘 의견", label="opinion_like"),
            _sentence(3, "셋 의견", label="opinion_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        assert [k.keyword_text for k in result] == ["가가", "나나", "다다", "라라", "마마"]
        assert len(result) == 5

    def test_focus_keywords_preserve_nouns_when_verbs_dominate(self, extractor):
        kw_extractor, mock_kiwi = extractor
        mock_kiwi.tokenize.side_effect = [
            [
                _token("강조하", "VV"),
                _token("밝히", "VV"),
                _token("주장하", "VV"),
                _token("내세우", "VV"),
                _token("꺼내", "VV"),
                _token("정책", "NNG"),
            ],
            [
                _token("강조하", "VV"),
                _token("밝히", "VV"),
                _token("주장하", "VV"),
                _token("내세우", "VV"),
                _token("꺼내", "VV"),
                _token("정책", "NNG"),
            ],
        ]
        classified = [
            _sentence(1, "정책을 강조하고 밝혔다", label="opinion_like"),
            _sentence(2, "정책을 다시 강조하고 밝혔다", label="fact_like"),
        ]

        result = kw_extractor.extract_focus_keywords(classified)

        focus_kws = [k.keyword_text for k in result]
        assert "정책" in focus_kws
        assert "강조하다" in focus_kws
        assert "강조하" not in focus_kws
