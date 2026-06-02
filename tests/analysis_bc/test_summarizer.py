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
_MOCK_SCORE_REASON_JSON = {
    "score_reason_summary": "이 영상은 보도자의 해석이나 주장이 일부 포함되어 주관성 점수가 보통 수준입니다."
}
_MOCK_SCORE_REASON_TEXT = json.dumps(_MOCK_SCORE_REASON_JSON, ensure_ascii=False)


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
            "summary_text": "[LLM 안 탐] 영상 요약 생성에 실패했습니다.",
            "perspective_summary": "",
            "evidence_summary": "",
            "tone_label": "",
        }

    def test_summarize_empty_summary_marks_non_llm_fallback(self) -> None:
        summarizer = _make_mock_summarizer(response_text=json.dumps({"summary_text": ""}, ensure_ascii=False))

        result = summarizer.summarize(
            fact_sentences=[_classified(1)],
            opinion_sentences=[],
            span_labels=[],
            title="제목",
            language="ko",
        )

        assert result["summary_text"].startswith("[LLM 안 탐]")

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

    def test_summarize_score_reason_parses_json(self) -> None:
        summarizer = _make_mock_summarizer(response_text=_MOCK_SCORE_REASON_TEXT)
        result = summarizer.summarize_score_reason(
            overall_bias_score=0.4362,
            opinion_score=0.5,
            emotion_score=0.12,
            fact_ratio=0.33,
            headline_body_gap_score=0.2,
            score_evidence="전체 문장 중 50%가 주관적 문장입니다.",
            opinion_sentences=[_classified(1, label="opinion_like")],
            span_labels=[],
            language="ko",
        )

        assert result == _MOCK_SCORE_REASON_JSON["score_reason_summary"]

    def test_summarize_score_reason_api_failure_returns_fallback(self) -> None:
        with patch("analysis_bc.summarizer.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = RuntimeError("API 오류")
            summarizer = BiasSummarizer()

        result = summarizer.summarize_score_reason(
            overall_bias_score=0.4362,
            opinion_score=0.5,
            emotion_score=0.12,
            fact_ratio=0.33,
            headline_body_gap_score=0.2,
            score_evidence="전체 문장 중 50%가 주관적 문장입니다.",
            opinion_sentences=[],
            span_labels=[],
            language="ko",
        )

        assert "주관성 점수" in result
        assert "보도자의 해석이나 주장" in result
        assert "overall_bias_score" not in result
        assert "제목-본문 괴리" not in result

    def test_summarize_score_reason_prompt_uses_user_friendly_terms(self) -> None:
        mock_msg = _make_mock_response(_MOCK_SCORE_REASON_TEXT)
        with patch("analysis_bc.summarizer.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            summarizer = BiasSummarizer()
            summarizer.summarize_score_reason(
                overall_bias_score=0.4362,
                opinion_score=0.5,
                emotion_score=0.12,
                fact_ratio=0.33,
                headline_body_gap_score=0.2,
                score_evidence="전체 문장 중 50%가 주관적 문장입니다.",
                opinion_sentences=[_classified(1, label="opinion_like")],
                span_labels=[_span(SentenceLabelType.EMOTIONALLY_LOADED)],
                language="ko",
            )
            call_args = MockClient.return_value.messages.create.call_args

        prompt: str = call_args.kwargs["messages"][0]["content"]
        assert "주관성 점수: 44점 / 100점" in prompt
        assert "의견과 정보가 섞여 있는 중간 구간" in prompt
        assert "보도자의 해석이나 주장이 들어간 문장 비율: 50%" in prompt
        assert "사실을 전달하는 문장 비율: 33%" in prompt
        assert "41~60점: 의견과 정보가 섞여 있음" in prompt
        assert '41~60점 구간은 "낮은 편"이라고 표현하지 말고' in prompt
        assert "영상의 보도 관점 또는 표현 방식" in prompt
        assert "문장 전체를 숫자 설명으로만 채우지 않기" in prompt
        assert "수치만 반복하는 문장" in prompt
        assert "overall_bias_score = 0.4 * opinion_score" not in prompt
        assert "전체 편향 점수" in prompt
        assert "제목-본문 괴리 점수는 언급하지 않기" in prompt

    def test_score_reason_fallback_uses_score_band_boundaries(self) -> None:
        summarizer = _make_mock_summarizer()

        result = summarizer.build_score_reason_fallback(
            overall_bias_score=0.41,
            opinion_score=0.5,
            emotion_score=0.19,
            fact_ratio=0.5,
        )

        assert "중간 수준" in result
        assert "낮은 편" not in result
