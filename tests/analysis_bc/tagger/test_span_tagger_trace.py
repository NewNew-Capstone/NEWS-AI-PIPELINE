from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.tagger.span_tagger import SpanTagger


def _make_sentence(cid: int, text: str) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=0,
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


def _make_qdrant_hit(score: float) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {"word": "x"}
    return hit


def _make_batch_result(hits: list[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.points = hits
    return result


def _build_tagger() -> SpanTagger:
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
        tagger._qdrant_healthy = True
        tagger._health_checked_at = float("inf")
    return tagger


def test_trace_collects_reasons_and_span() -> None:
    tagger = _build_tagger()
    sentence = _make_sentence(1, "토큰 테스트 문장")

    tagger.kiwi.tokenize.return_value = [
        _make_token("조사", "JKS", 0, 2),   # invalid_pos
        _make_token("가", "NNG", 3, 1),     # too_short
        _make_token("무덤", "NNG", 5, 2),   # no_hit
        _make_token("약함", "NNG", 8, 2),   # below_threshold
        _make_token("애매", "NNG", 11, 2),  # low_margin
        _make_token("분노", "NNG", 14, 2),  # accepted
    ]
    tagger.qdrant.query_batch_points.return_value = [
        _make_batch_result([]),
        _make_batch_result([_make_qdrant_hit(0.40)]),
        _make_batch_result([_make_qdrant_hit(0.95), _make_qdrant_hit(0.93)]),
        _make_batch_result([_make_qdrant_hit(0.95), _make_qdrant_hit(0.70)]),
    ]

    spans = tagger.tag([sentence], debug=True)
    trace = tagger.last_debug_trace

    assert len(spans) == 1
    assert trace is not None
    assert trace.qdrant_healthy is True
    assert len(trace.sentences) == 1
    token_traces = trace.sentences[0].tokens
    assert len(token_traces) == 6

    assert token_traces[0].reject_reason == "invalid_pos"
    assert token_traces[1].reject_reason == "too_short"
    assert token_traces[2].reject_reason == "no_hit"
    assert token_traces[3].reject_reason == "below_threshold"
    assert token_traces[4].reject_reason == "low_margin"
    assert token_traces[5].accepted is True
    assert token_traces[5].created_span is not None


def test_debug_false_keeps_return_contract() -> None:
    tagger = _build_tagger()
    sentence = _make_sentence(1, "분노가 폭발했다")
    tagger.kiwi.tokenize.return_value = [_make_token("분노", "NNG", 0, 2)]
    tagger.qdrant.query_batch_points.return_value = [
        _make_batch_result([_make_qdrant_hit(0.95)])
    ]

    result = tagger.tag([sentence], debug=False)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].matched_word == "분노"
