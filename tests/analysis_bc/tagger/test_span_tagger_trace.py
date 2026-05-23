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


def _make_qdrant_hit(score: float, clean_seed: str = "x") -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {"clean_seed": clean_seed, "word": clean_seed}
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
        tagger._gate_load_attempted = True
    return tagger


def test_trace_collects_reasons_and_span() -> None:
    tagger = _build_tagger()
    sentence = _make_sentence(1, "조사가 중국 무덤 약함 모호 분노")

    tagger.kiwi.tokenize.return_value = [
        _make_token("조사", "JKS", 0, 2),   # invalid_pos
        _make_token("가", "NNG", 2, 1),     # too_short
        _make_token("중국", "NNP", 4, 2),   # proper_noun
        _make_token("무덤", "NNG", 7, 2),   # no_hit
        _make_token("약함", "NNG", 10, 2),  # below_threshold
        _make_token("모호", "NNG", 13, 2),  # low_margin
        _make_token("분노", "NNG", 16, 2),  # accepted
    ]
    tagger.qdrant.query_batch_points.return_value = [
        _make_batch_result([]),
        _make_batch_result([_make_qdrant_hit(0.40)]),
        _make_batch_result([_make_qdrant_hit(0.95, "모호"), _make_qdrant_hit(0.93, "애매")]),
        _make_batch_result([_make_qdrant_hit(0.95, "분노"), _make_qdrant_hit(0.70, "화")]),
    ]

    spans = tagger.tag([sentence], debug=True)
    trace = tagger.last_debug_trace

    assert len(spans) == 1
    assert trace is not None
    assert trace.qdrant_healthy is True
    assert len(trace.sentences) == 1
    token_traces = trace.sentences[0].tokens
    assert len(token_traces) == 7

    assert token_traces[0].reject_reason == "invalid_pos"
    assert token_traces[1].reject_reason == "too_short"
    assert token_traces[2].reject_reason == "proper_noun"
    assert token_traces[3].reject_reason == "no_hit"
    assert token_traces[4].reject_reason == "below_threshold"
    assert token_traces[5].reject_reason == "low_margin"
    assert token_traces[6].accepted is True
    assert token_traces[6].created_span is not None


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


def test_trace_records_polarity_mismatch() -> None:
    tagger = _build_tagger()
    sentence = _make_sentence(1, "감동적이다")

    tagger.kiwi.tokenize.return_value = [_make_token("감동", "NNG", 0, 2)]
    hit = _make_qdrant_hit(0.97)
    hit.payload = {"word": "감동", "polarity": "-1"}
    tagger.qdrant.query_batch_points.return_value = [_make_batch_result([hit])]
    tagger._gate_sentences = MagicMock(return_value={
        1: MagicMock(gate_passed=True, gate_score=0.9, top_labels=["환영/호의"], skip_reason=None)
    })

    spans = tagger.tag([sentence], debug=True)

    assert spans == []
    trace = tagger.last_debug_trace
    assert trace is not None
    assert trace.sentences[0].tokens[0].reject_reason == "polarity_mismatch"
