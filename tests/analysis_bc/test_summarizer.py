from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.summarizer import BiasSummarizer

_MOCK_JSON = {
    "summary_text": "요약",
    "perspective_summary": "관점",
    "evidence_summary": "근거",
    "tone_label": "비판적",
}
_MOCK_TEXT = json.dumps(_MOCK_JSON, ensure_ascii=False)


def _make_mock_response(response_text: str) -> MagicMock:
    mock_block = MagicMock(spec=TextBlock)
    mock_block.text = response_text
    mock_msg = MagicMock()
    mock_msg.content = [mock_block]
    return mock_msg


def _make_mock_summarizer(response_text: str = _MOCK_TEXT) -> BiasSummarizer:
    mock_msg = _make_mock_response(response_text)
    with patch("analysis_bc.summarizer.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_msg
        summarizer = BiasSummarizer()
    summarizer.client.messages.create.return_value = mock_msg  # type: ignore[attr-defined]
    return summarizer


def _classified(order: int, label: str = "fact_like") -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=order,
        sentence_text=f"문장{order}",
        sentence_order=order,
        start_time_ms=None,
        end_time_ms=None,
        label=label,
        confidence=0.9,
    )


def _span(label_type: SentenceLabelType) -> SpanLabelDto:
    return SpanLabelDto(
        content_sentence_id=1,
        start_offset=0,
        end_offset=5,
        label_type=label_type,
        score=0.9,
    )


# ------------------------------------------------------------------
# tests
# ------------------------------------------------------------------

class TestBiasSummarizer:
    def test_summarize_returns_all_keys(self) -> None:
        summarizer = _make_mock_summarizer()
        result = summarizer.summarize(
            fact_sentences=[_classified(1)],
            opinion_sentences=[_classified(2, label="opinion_like")],
            span_labels=[],
            title="테스트 제목",
            language="ko",
        )
        assert set(result.keys()) >= {"summary_text", "perspective_summary", "evidence_summary", "tone_label"}

    def test_summarize_parses_json(self) -> None:
        summarizer = _make_mock_summarizer()
        result = summarizer.summarize(
            fact_sentences=[_classified(1)],
            opinion_sentences=[],
            span_labels=[],
            title="제목",
            language="ko",
        )
        assert result["summary_text"] == "요약"
        assert result["tone_label"] == "비판적"

    def test_summarize_json_with_surrounding_text(self) -> None:
        wrapped = "다음은 결과입니다.\n" + _MOCK_TEXT + "\n이상입니다."
        summarizer = _make_mock_summarizer(response_text=wrapped)
        result = summarizer.summarize(
            fact_sentences=[],
            opinion_sentences=[],
            span_labels=[],
            title="제목",
            language="ko",
        )
        assert result["tone_label"] == "비판적"

    def test_summarize_api_failure_returns_empty(self) -> None:
        with patch("analysis_bc.summarizer.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = RuntimeError("API 오류")
            summarizer = BiasSummarizer()
        result = summarizer.summarize(
            fact_sentences=[],
            opinion_sentences=[],
            span_labels=[],
            title="제목",
            language="ko",
        )
        assert result == {
            "summary_text": "",
            "perspective_summary": "",
            "evidence_summary": "",
            "tone_label": "",
        }

    def test_summarize_empty_sentences(self) -> None:
        summarizer = _make_mock_summarizer()
        result = summarizer.summarize(
            fact_sentences=[],
            opinion_sentences=[],
            span_labels=[],
            title="제목",
            language="ko",
        )
        assert "summary_text" in result

    def test_summarize_opinion_truncated_to_10(self) -> None:
        opinions = [_classified(i, label="opinion_like") for i in range(1, 16)]
        mock_msg = _make_mock_response(_MOCK_TEXT)
        with patch("analysis_bc.summarizer.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            summarizer = BiasSummarizer()
            summarizer.summarize(
                fact_sentences=[],
                opinion_sentences=opinions,
                span_labels=[],
                title="제목",
                language="ko",
            )
            call_args = MockClient.return_value.messages.create.call_args
        prompt: str = call_args.kwargs["messages"][0]["content"]
        # 문장11~15는 포함되지 않아야 함
        assert "문장11" not in prompt
        assert "문장10" in prompt
