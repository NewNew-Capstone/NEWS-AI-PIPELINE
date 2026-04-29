from __future__ import annotations

from unittest.mock import MagicMock, patch

import analysis_bc.router as router_mod
from analysis_bc.schemas import AnalyzeRawTextRequestDto, AnalyzeRequestDto, SentenceInputDto


def _make_analyze_request() -> AnalyzeRequestDto:
    return AnalyzeRequestDto(
        target_id=1,
        title="제목",
        language="ko",
        target_type=None,
        transcript_id=10,
        country="KR",
        sentences=[
            SentenceInputDto(
                content_sentence_id=1,
                sentence_text="첫 문장입니다.",
                sentence_order=1,
            )
        ],
    )


def _make_raw_request() -> AnalyzeRawTextRequestDto:
    return AnalyzeRawTextRequestDto(
        target_id=1,
        title="제목",
        language="ko",
        raw_text="첫 문장입니다. 둘째 문장입니다.",
        target_type=None,
        transcript_id=10,
        country="KR",
    )


def test_analyze_continues_when_log_insert_fails() -> None:
    request = _make_analyze_request()

    mock_service = MagicMock()
    mock_service.analyze.return_value = "OK"

    with patch("analysis_bc.router.get_analysis_service", return_value=mock_service):
        result = router_mod.analyze(request)

    assert result == "OK"
    assert mock_service.analyze.call_count == 1


def test_analyze_raw_continues_when_log_insert_fails() -> None:
    request = _make_raw_request()

    fake_result = MagicMock()
    fake_result.model_dump.return_value = {
        "target_id": 1,
        "transcript_id": 10,
        "overall_bias_score": 0.1,
        "opinion_score": 0.1,
        "emotion_score": 0.1,
        "anonymous_source_score": 0.1,
        "summary_text": "요약",
        "perspective_summary": "관점",
        "evidence_summary": "근거",
        "tone_label": "중립",
        "keywords": [],
        "sentence_labels": [],
        "evidences": [],
    }

    mock_service = MagicMock()
    mock_service.analyze.return_value = fake_result

    with patch("analysis_bc.router.get_analysis_service", return_value=mock_service):
        result = router_mod.analyze_raw(request)

    assert result.target_id == 1
    assert mock_service.analyze.call_count == 1
