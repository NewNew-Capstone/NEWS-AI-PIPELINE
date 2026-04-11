"""SpanTagger 유닛 테스트.

실제 Kiwi / SentenceTransformer / FastText / Qdrant 연결 없이
mock으로 전체 흐름을 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.tagger.span_tagger import SpanTagger


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_sentence(cid: int, text: str, order: int = 0) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=order,
        start_time_ms=None,
        end_time_ms=None,
        label="opinion_like",
        confidence=0.9,
    )


def _make_token(form: str, tag: str, start: int, length: int) -> MagicMock:
    token = MagicMock()
    token.form = form
    token.tag = tag
    token.start = start
    token.len = length
    return token


def _make_qdrant_hit(score: float, payload: dict) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = payload
    return hit


def _make_batch_result(hits: list) -> MagicMock:
    result = MagicMock()
    result.points = hits
    return result


@pytest.fixture
def mock_tagger() -> SpanTagger:
    """실제 모델/Qdrant 연결 없이 SpanTagger 인스턴스 반환."""
    with (
        patch("analysis_bc.tagger.span_tagger.Kiwi") as mock_kiwi_cls,
        patch("analysis_bc.tagger.span_tagger.SentenceTransformer") as mock_sbert_cls,
        patch("analysis_bc.tagger.span_tagger.fasttext") as mock_ft_module,
        patch("analysis_bc.tagger.span_tagger.QdrantClient") as mock_qdrant_cls,
    ):
        mock_kiwi_cls.return_value = MagicMock()
        mock_sbert_cls.return_value = MagicMock()

        mock_ft = MagicMock()
        mock_ft.get_word_vector.return_value = np.ones(300)
        mock_ft_module.load_model.return_value = mock_ft

        mock_qdrant_cls.return_value = MagicMock()

        tagger = SpanTagger()
        # Qdrant 헬스체크 통과 상태로 설정
        tagger._qdrant_healthy = True
        tagger._health_checked_at = float("inf")

    return tagger


# ── 빈 입력 / Qdrant 비정상 ───────────────────────────────────────────────────

def test_tag_empty_sentences(mock_tagger: SpanTagger) -> None:
    """빈 입력 → 빈 리스트 반환."""
    assert mock_tagger.tag([]) == []


def test_tag_returns_empty_when_qdrant_unhealthy(mock_tagger: SpanTagger) -> None:
    """Qdrant 비정상 → 빈 리스트 반환."""
    mock_tagger._qdrant_healthy = False
    mock_tagger._health_checked_at = 0.0
    mock_tagger.qdrant.get_collections.side_effect = Exception("connection refused")

    result = mock_tagger.tag([_make_sentence(1, "분노가 폭발했다")])

    assert result == []


# ── emotion 태깅 ──────────────────────────────────────────────────────────────

def test_search_emotion_returns_span(mock_tagger: SpanTagger) -> None:
    """감정 단어 토큰 → EMOTIONALLY_LOADED span 반환, matched_word는 표면형."""
    sentence = _make_sentence(1, "분노가 폭발했다")

    # Kiwi: "분노" 토큰 (NNG, start=0, len=2) → 표면형 "분노"
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("분노", "NNG", 0, 2),
    ]

    # FastText + Qdrant 매칭
    hit = _make_qdrant_hit(0.95, {"word": "분노", "word_root": "분노", "polarity": "-2"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    span = result[0]
    assert span.content_sentence_id == 1
    assert span.label_type == SentenceLabelType.EMOTIONALLY_LOADED
    assert span.start_offset == 0
    assert span.end_offset == 2
    assert span.score == pytest.approx(0.95)
    assert span.matched_word == "분노"  # 표면형 사용


def test_search_emotion_no_match_returns_empty(mock_tagger: SpanTagger) -> None:
    """유사도 미달 → 감정 태깅 없음."""
    sentence = _make_sentence(2, "국회는 오늘 예산안을 처리했다")

    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("국회", "NNG", 0, 2),
        _make_token("오늘", "MAG", 3, 2),
    ]
    mock_tagger.qdrant.query_batch_points.return_value = [
        _make_batch_result([]),
        _make_batch_result([]),
    ]

    result = mock_tagger.tag([sentence])

    assert result == []


def test_search_emotion_filters_short_tokens(mock_tagger: SpanTagger) -> None:
    """len < 2 토큰은 필터링되어 쿼리하지 않는다."""
    sentence = _make_sentence(3, "이 법안")

    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("이", "NNG", 0, 1),   # len=1 → 필터링
        _make_token("법안", "NNG", 2, 2),
    ]
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([])]

    mock_tagger.tag([sentence])

    # query_batch_points는 "법안" 1개만 쿼리 (keyword 인수로 전달됨)
    calls = mock_tagger.qdrant.query_batch_points.call_args_list
    emotion_calls = [c for c in calls if "emotion" in str(c.kwargs.get("collection_name", ""))]
    assert len(emotion_calls[0].kwargs["requests"]) == 1


def test_search_emotion_xr_tag_included(mock_tagger: SpanTagger) -> None:
    """XR(어근) 태그 토큰도 처리된다 (심각하다 → 심각(XR))."""
    sentence = _make_sentence(4, "매우 심각한 상황이다")

    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("심각", "XR", 3, 2),  # 표면형 "심각"
    ]
    hit = _make_qdrant_hit(1.0, {"word": "심각한", "word_root": "심각", "polarity": "-2"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "심각"
    assert result[0].label_type == SentenceLabelType.EMOTIONALLY_LOADED


# ── anonymous 태깅 ────────────────────────────────────────────────────────────

def test_search_anonymous_returns_span(mock_tagger: SpanTagger) -> None:
    """익명 출처 패턴 매칭 → ANONYMOUS_SOURCE span 반환."""
    sentence = _make_sentence(5, "관계자에 따르면 사태가 심각하다")

    tokens = [
        _make_token("관계자", "NNG", 0, 3),
        _make_token("에", "JKB", 3, 1),
        _make_token("따르", "VV", 5, 2),
        _make_token("면", "EC", 7, 1),
    ]
    mock_tagger.kiwi.tokenize.return_value = tokens
    mock_tagger.st_model.encode.return_value = np.ones((6, 768))  # 슬라이딩 윈도우 후보 수

    hit = _make_qdrant_hit(1.0, {"phrase": "관계자에 따르면", "label_type": "ANONYMOUS_SOURCE"})
    # anonymous 쿼리 결과 (emotion은 빈 결과)
    mock_tagger.qdrant.query_batch_points.side_effect = [
        [_make_batch_result([hit])] + [_make_batch_result([])] * 5,  # anonymous
        [],  # emotion (valid_tokens 없음)
    ]

    result = mock_tagger.tag([sentence])

    anon_spans = [s for s in result if s.label_type == SentenceLabelType.ANONYMOUS_SOURCE]
    assert len(anon_spans) == 1
    assert anon_spans[0].matched_word == "관계자에 따르면"
    assert anon_spans[0].score == pytest.approx(1.0)


def test_search_anonymous_no_match(mock_tagger: SpanTagger) -> None:
    """패턴 없는 문장 → ANONYMOUS_SOURCE 없음."""
    sentence = _make_sentence(6, "정부는 오늘 예산안을 발표했다")

    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("정부", "NNG", 0, 2),
        _make_token("오늘", "MAG", 3, 2),
    ]
    mock_tagger.st_model.encode.return_value = np.ones((1, 768))
    mock_tagger.qdrant.query_batch_points.return_value = [
        _make_batch_result([]),
        _make_batch_result([]),
    ]

    result = mock_tagger.tag([sentence])

    assert all(s.label_type != SentenceLabelType.ANONYMOUS_SOURCE for s in result)


# ── _merge ────────────────────────────────────────────────────────────────────

def test_merge_no_overlap(mock_tagger: SpanTagger) -> None:
    """겹치지 않는 anon + emotion → 둘 다 유지."""
    anon = SpanLabelDto(
        content_sentence_id=1, start_offset=0, end_offset=7,
        label_type=SentenceLabelType.ANONYMOUS_SOURCE, score=1.0, matched_word="관계자에 따르면",
    )
    emotion = SpanLabelDto(
        content_sentence_id=1, start_offset=10, end_offset=12,
        label_type=SentenceLabelType.EMOTIONALLY_LOADED, score=0.95, matched_word="분노",
    )

    result = mock_tagger._merge(anon, [emotion])

    assert len(result) == 2
    assert result[0].start_offset == 0
    assert result[1].start_offset == 10


def test_merge_overlap_anon_wins(mock_tagger: SpanTagger) -> None:
    """겹칠 때 anon score 더 높으면 anon 유지."""
    anon = SpanLabelDto(
        content_sentence_id=1, start_offset=0, end_offset=7,
        label_type=SentenceLabelType.ANONYMOUS_SOURCE, score=1.0, matched_word="관계자에 따르면",
    )
    emotion = SpanLabelDto(
        content_sentence_id=1, start_offset=3, end_offset=6,
        label_type=SentenceLabelType.EMOTIONALLY_LOADED, score=0.85, matched_word="따르",
    )

    result = mock_tagger._merge(anon, [emotion])

    assert len(result) == 1
    assert result[0].label_type == SentenceLabelType.ANONYMOUS_SOURCE


def test_merge_overlap_emotion_wins(mock_tagger: SpanTagger) -> None:
    """겹칠 때 emotion score 더 높으면 emotion 유지."""
    anon = SpanLabelDto(
        content_sentence_id=1, start_offset=0, end_offset=7,
        label_type=SentenceLabelType.ANONYMOUS_SOURCE, score=0.80, matched_word="관계자에 따르면",
    )
    emotion = SpanLabelDto(
        content_sentence_id=1, start_offset=3, end_offset=6,
        label_type=SentenceLabelType.EMOTIONALLY_LOADED, score=0.95, matched_word="분노",
    )

    result = mock_tagger._merge(anon, [emotion])

    assert len(result) == 1
    assert result[0].label_type == SentenceLabelType.EMOTIONALLY_LOADED


def test_merge_anon_none_returns_emotion_spans(mock_tagger: SpanTagger) -> None:
    """anon_span이 None이면 emotion_spans 그대로 반환."""
    emotion = SpanLabelDto(
        content_sentence_id=1, start_offset=0, end_offset=2,
        label_type=SentenceLabelType.EMOTIONALLY_LOADED, score=0.95, matched_word="분노",
    )

    result = mock_tagger._merge(None, [emotion])

    assert result == [emotion]
