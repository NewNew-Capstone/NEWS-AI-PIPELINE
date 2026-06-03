"""SpanTagger 유닛 테스트.

실제 Kiwi / FastText / Qdrant 연결 없이 mock으로 전체 흐름을 검증한다.
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
    hit.payload = dict(payload)
    if "clean_seed" not in hit.payload and "embed_text" not in hit.payload:
        clean_seed = hit.payload.get("word_root") or hit.payload.get("word")
        if clean_seed:
            hit.payload["clean_seed"] = clean_seed
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
        patch("analysis_bc.tagger.span_tagger.fasttext") as mock_ft_module,
        patch("analysis_bc.tagger.span_tagger.QdrantClient") as mock_qdrant_cls,
    ):
        mock_kiwi_cls.return_value = MagicMock()

        mock_ft = MagicMock()
        mock_ft.get_word_vector.return_value = np.ones(300)
        mock_ft_module.load_model.return_value = mock_ft

        mock_qdrant_cls.return_value = MagicMock()

        tagger = SpanTagger()
        # Qdrant 헬스체크 통과 상태로 설정
        tagger._qdrant_healthy = True
        tagger._health_checked_at = float("inf")
        tagger._gate_load_attempted = True

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


def test_search_emotion_clamps_score_above_one(mock_tagger: SpanTagger) -> None:
    sentence = _make_sentence(1, "분노가 커졌다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("분노", "NNG", 0, 2),
    ]
    hit = _make_qdrant_hit(1.0000002, {"word": "분노"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].score == 1.0


def test_search_emotion_ignores_duplicate_clean_seed_for_margin(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(1, "우려됩니다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("우려", "NNG", 0, 2),
    ]
    hits = [
        _make_qdrant_hit(1.0, {"clean_seed": "우려", "polarity": "-2"}),
        _make_qdrant_hit(1.0, {"clean_seed": "우려", "polarity": "-1"}),
        _make_qdrant_hit(0.2, {"clean_seed": "걱정", "polarity": "-1"}),
    ]
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result(hits)]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "우려"


def test_search_emotion_uses_surface_not_morph_form_for_matched_word(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(1, "깨진다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("깨지", "VV", 0, 2),
    ]
    hit = _make_qdrant_hit(
        1.0,
        {"word": "깨지다", "word_root": "깨지다", "clean_seed": "깨지다", "polarity": "-1"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "깨진다"
    assert result[0].end_offset == 3
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("깨지다")


@pytest.mark.parametrize(
    ("text", "form", "tag", "start", "length", "expected_word", "expected_end"),
    [
        ("힘들다", "힘들", "VA", 0, 2, "힘들다", 3),
        ("힘들어지고 있다", "힘들", "VA", 0, 2, "힘들어지고", 5),
        ("힘든 상황이다", "힘들", "VA", 0, 2, "힘든", 2),
    ],
)
def test_search_emotion_predicate_queries_base_form_and_expands_span(
    mock_tagger: SpanTagger,
    text: str,
    form: str,
    tag: str,
    start: int,
    length: int,
    expected_word: str,
    expected_end: int,
) -> None:
    sentence = _make_sentence(1, text)
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token(form, tag, start, length),
    ]
    hit = _make_qdrant_hit(1.0, {"word": "힘들다", "word_root": "힘들다", "polarity": "-2"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].start_offset == start
    assert result[0].end_offset == expected_end
    assert result[0].matched_word == expected_word
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("힘들다")


@pytest.mark.parametrize(
    ("text", "form", "tag", "start", "length", "expected_query", "expected_word", "expected_end"),
    [
        ("재밌어요", "재밌", "VA", 0, 2, "재밌다", "재밌어요", 4),
        ("재밌는 영상이다", "재밌", "VA", 0, 2, "재밌다", "재밌는", 3),
        ("싸우고 있다", "싸우", "VV", 0, 2, "싸우다", "싸우고", 3),
        ("싸운다", "싸우", "VV", 0, 2, "싸우다", "싸운다", 3),
    ],
)
def test_search_emotion_common_predicate_examples_expand_span(
    mock_tagger: SpanTagger,
    text: str,
    form: str,
    tag: str,
    start: int,
    length: int,
    expected_query: str,
    expected_word: str,
    expected_end: int,
) -> None:
    sentence = _make_sentence(1, text)
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token(form, tag, start, length),
    ]
    hit = _make_qdrant_hit(1.0, {"word": expected_query, "word_root": expected_query, "polarity": "-1"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].start_offset == start
    assert result[0].end_offset == expected_end
    assert result[0].matched_word == expected_word
    mock_tagger.ft_model.get_word_vector.assert_called_once_with(expected_query)


def test_search_emotion_noun_plus_xsa_keeps_existing_noun_span(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(1, "불안하다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("불안", "NNG", 0, 2),
    ]
    hit = _make_qdrant_hit(1.0, {"word": "불안", "word_root": "불안", "polarity": "-2"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "불안"
    assert result[0].end_offset == 2
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("불안")


def test_search_emotion_noun_keeps_existing_span_before_predicate(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(1, "분노가 크다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("분노", "NNG", 0, 2),
    ]
    hit = _make_qdrant_hit(1.0, {"word": "분노", "word_root": "분노", "polarity": "-2"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "분노"
    assert result[0].end_offset == 2
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("분노")


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


def test_search_emotion_blocks_known_general_false_positive_words(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(
        2,
        (
            "지우다 흐르다 궁금하실 정한 똑같다 그렇다 어렵다 죄송합니다 "
            "부르다 통하다 미치다 위대하다 내세우다 꺼내다 물어보다"
        ),
    )
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("지우", "VV", 0, 2),
        _make_token("흐르", "VV", 4, 2),
        _make_token("궁금", "XR", 8, 2),
        _make_token("하", "XSA", 10, 1),
        _make_token("정한", "NNG", 13, 2),
        _make_token("똑같", "VA", 16, 2),
        _make_token("그렇", "VA", 20, 2),
        _make_token("어렵", "VA", 24, 2),
        _make_token("죄송하", "VA", 28, 3),
        _make_token("부르", "VV", 34, 2),
        _make_token("통하", "VV", 38, 2),
        _make_token("미치", "VV", 42, 2),
        _make_token("위대", "XR", 46, 2),
        _make_token("하", "XSA", 48, 1),
        _make_token("내세우", "VV", 51, 3),
        _make_token("꺼내", "VV", 56, 2),
        _make_token("물어보", "VV", 60, 3),
    ]

    result = mock_tagger.tag([sentence])

    assert result == []
    mock_tagger.ft_model.get_word_vector.assert_not_called()
    mock_tagger.qdrant.query_batch_points.assert_not_called()


def test_search_emotion_blocks_broad_knu_general_nouns_but_keeps_real_emotions(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "친구 이야기 추진 재밌다 놀라다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("친구", "NNG", 0, 2),
        _make_token("이야기", "NNG", 3, 3),
        _make_token("추진", "NNG", 7, 2),
        _make_token("재밌", "VA", 10, 2),
        _make_token("놀라", "VV", 14, 2),
    ]
    hits = [
        _make_batch_result([
            _make_qdrant_hit(
                1.0,
                {"clean_seed": "재밌다", "polarity": "1", "category": "EVALUATIVE_WORD"},
            )
        ]),
        _make_batch_result([
            _make_qdrant_hit(
                1.0,
                {"clean_seed": "놀라다", "polarity": "1", "category": "EVALUATIVE_WORD"},
            )
        ]),
    ]
    mock_tagger.qdrant.query_batch_points.return_value = hits

    result = mock_tagger.tag([sentence])

    assert [span.matched_word for span in result] == ["재밌다", "놀라다"]
    calls = [call.args[0] for call in mock_tagger.ft_model.get_word_vector.call_args_list]
    assert calls == ["재밌다", "놀라다"]


def test_search_emotion_requires_qdrant_clean_seed_match(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "싸우고 있다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("싸우", "VV", 0, 2),
    ]
    hit = _make_qdrant_hit(
        1.0,
        {"word": "화나다", "word_root": "화나다", "clean_seed": "화나다", "polarity": "-2"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence], debug=True)

    assert result == []
    assert mock_tagger.last_debug_trace is not None
    token_traces = mock_tagger.last_debug_trace.sentences[0].tokens
    assert token_traces[0].reject_reason == "seed_mismatch"


def test_search_emotion_matches_broad_seed_canonical_seed(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "화난다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("화나", "VV", 0, 2),
    ]
    hit = _make_qdrant_hit(
        0.95,
        {
            "clean_seed": "화가 치밀어 오르다",
            "canonical_seed": "화나다",
            "word_root": "화가 치밀어 오르다",
            "polarity": "-2",
            "category": "NEGATIVE_STATE",
        },
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence], debug=True)

    assert len(result) == 1
    assert result[0].matched_word == "화난다"
    assert mock_tagger.last_debug_trace is not None
    assert mock_tagger.last_debug_trace.sentences[0].tokens[0].accepted is True


def test_search_emotion_allows_high_confidence_direct_semantic_fallback_for_adjectives(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "서글픈 마음이다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("서글프", "VA", 0, 3),
    ]
    hit = _make_qdrant_hit(
        0.96,
        {"clean_seed": "슬픔", "polarity": "-2", "category": "DIRECT_EMOTION"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "서글픈"
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("서글프다")


def test_search_emotion_allows_direct_semantic_fallback_for_emotion_like_nouns(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "공포감이 커졌다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("공포감", "NNG", 0, 3),
    ]
    hit = _make_qdrant_hit(
        0.96,
        {"clean_seed": "공포", "polarity": "-2", "category": "DIRECT_EMOTION"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "공포감"


def test_search_emotion_rejects_semantic_fallback_for_generic_nouns(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "정책을 발표했다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("정책", "NNG", 0, 2),
    ]
    hit = _make_qdrant_hit(
        0.99,
        {"clean_seed": "불안", "polarity": "-2", "category": "DIRECT_EMOTION"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence], debug=True)

    assert result == []
    assert mock_tagger.last_debug_trace is not None
    token_traces = mock_tagger.last_debug_trace.sentences[0].tokens
    assert token_traces[0].reject_reason == "seed_mismatch"


def test_search_emotion_rejects_low_score_semantic_fallback(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(2, "서글픈 마음이다")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("서글프", "VA", 0, 3),
    ]
    hit = _make_qdrant_hit(
        0.90,
        {"clean_seed": "슬픔", "polarity": "-2", "category": "DIRECT_EMOTION"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence], debug=True)

    assert result == []
    assert mock_tagger.last_debug_trace is not None
    token_traces = mock_tagger.last_debug_trace.sentences[0].tokens
    assert token_traces[0].reject_reason == "seed_mismatch"


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


def test_search_emotion_blocks_stopwords_before_qdrant(mock_tagger: SpanTagger) -> None:
    sentence = _make_sentence(4, "중국 과연 부리 1부리")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("중국", "NNP", 0, 2),
        _make_token("과연", "MAG", 3, 2),
        _make_token("부리", "NNG", 6, 2),
        _make_token("1부리", "NNG", 9, 3),
    ]

    result = mock_tagger.tag([sentence])

    assert result == []
    mock_tagger.ft_model.get_word_vector.assert_not_called()
    mock_tagger.qdrant.query_batch_points.assert_not_called()


def test_search_emotion_blocks_proper_nouns_before_qdrant(mock_tagger: SpanTagger) -> None:
    sentence = _make_sentence(5, "트럼프 바이든")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("트럼프", "NNP", 0, 3),
        _make_token("바이든", "NNP", 4, 3),
    ]

    result = mock_tagger.tag([sentence], debug=True)

    assert result == []
    mock_tagger.ft_model.get_word_vector.assert_not_called()
    mock_tagger.qdrant.query_batch_points.assert_not_called()
    assert mock_tagger.last_debug_trace is not None
    token_traces = mock_tagger.last_debug_trace.sentences[0].tokens
    assert [t.reject_reason for t in token_traces] == ["proper_noun", "proper_noun"]


def test_search_emotion_excludes_stopword_and_keeps_emotion_token(mock_tagger: SpanTagger) -> None:
    sentence = _make_sentence(5, "중국 분노")
    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("중국", "NNP", 0, 2),
        _make_token("분노", "NNG", 3, 2),
    ]
    hit = _make_qdrant_hit(0.95, {"word": "분노", "word_root": "분노", "polarity": "-2"})
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "분노"
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("분노")
    calls = mock_tagger.qdrant.query_batch_points.call_args_list
    assert len(calls) == 1
    assert len(calls[0].kwargs["requests"]) == 1


def test_search_emotion_xr_tag_included(mock_tagger: SpanTagger) -> None:
    """VA 감정 서술어는 기본형으로 쿼리하고 표면형 span을 유지한다."""
    sentence = _make_sentence(4, "매우 심각한 상황이다")

    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("심각하", "VA", 3, 3),
    ]
    hit = _make_qdrant_hit(
        1.0,
        {"word": "심각하다", "word_root": "심각하다", "clean_seed": "심각하다", "polarity": "-2"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "심각한"
    assert result[0].label_type == SentenceLabelType.EMOTIONALLY_LOADED
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("심각하다")


def test_search_emotion_xr_plus_xsa_keeps_existing_root_span(
    mock_tagger: SpanTagger,
) -> None:
    sentence = _make_sentence(4, "끔찍하다")

    mock_tagger.kiwi.tokenize.return_value = [
        _make_token("끔찍", "XR", 0, 2),
        _make_token("하", "XSA", 2, 1),
    ]
    hit = _make_qdrant_hit(
        1.0,
        {"word": "끔찍하다", "word_root": "끔찍하다", "clean_seed": "끔찍하다", "polarity": "-2"},
    )
    mock_tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].matched_word == "끔찍하다"
    assert result[0].end_offset == 4
    mock_tagger.ft_model.get_word_vector.assert_called_once_with("끔찍하다")


def test_gate_failed_sentence_is_skipped(mock_tagger: SpanTagger) -> None:
    sentence = _make_sentence(1, "분노가 폭발했다")
    mock_tagger._gate_sentences = MagicMock(return_value={
        1: MagicMock(gate_passed=False, gate_score=0.1, top_labels=[], skip_reason="low_gate_score")
    })

    result = mock_tagger.tag([sentence], debug=True)

    assert result == []
    assert mock_tagger.qdrant.query_batch_points.call_count == 0
    assert mock_tagger.last_debug_trace is not None
    assert mock_tagger.last_debug_trace.sentences[0].skip_reason == "low_gate_score"
