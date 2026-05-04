import json
from unittest.mock import MagicMock

from analysis_bc.request_log_repository import AnalysisRequestLogRepository


def test_build_payload_sentence_array_and_count() -> None:
    repo = AnalysisRequestLogRepository(engine=None, use_default_engine=False)
    payload = repo.build_payload(
        source_endpoint="/analyze",
        target_id=1,
        transcript_id=2,
        language="ko",
        target_type="YOUTUBE_VIDEO",
        country="KR",
        sentences=["  첫 문장  ", "", "둘째 문장"],
    )

    assert payload["source_endpoint"] == "/analyze"
    assert payload["target_id"] == 1
    assert payload["sentence_count"] == 2
    assert payload["sentences_json"] == "[\"첫 문장\", \"둘째 문장\"]"
    assert payload["created_at"] is not None


def test_insert_returns_false_when_engine_missing() -> None:
    repo = AnalysisRequestLogRepository(
        engine=None,
        local_log_path="",
        use_default_engine=False,
        local_daily_split=False,
    )
    ok = repo.insert_request_log(
        source_endpoint="/analyze",
        target_id=1,
        transcript_id=None,
        language="ko",
        target_type=None,
        country=None,
        sentences=["문장"],
    )
    assert ok is False
    assert repo.last_sink == "disabled"


def test_insert_writes_local_jsonl_when_path_set(tmp_path) -> None:
    log_path = tmp_path / "analysis_requests_local.jsonl"
    repo = AnalysisRequestLogRepository(
        engine=None,
        local_log_path=str(log_path),
        use_default_engine=False,
        local_daily_split=False,
    )
    ok = repo.insert_request_log(
        source_endpoint="/analyze/raw",
        target_id=11,
        transcript_id=22,
        language="ko",
        target_type="YOUTUBE_VIDEO",
        country="KR",
        sentences=[" 첫 문장 ", "둘째 문장"],
    )

    assert ok is True
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["source_endpoint"] == "/analyze/raw"
    assert row["sentence_count"] == 2
    assert row["sentences"] == [{"sentence_text": "첫 문장"}, {"sentence_text": "둘째 문장"}]
    assert repo.last_sink == "local"


def test_insert_falls_back_to_local_when_db_insert_fails(tmp_path) -> None:
    log_path = tmp_path / "analysis_requests_local.jsonl"
    mock_engine = MagicMock()
    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.side_effect = RuntimeError("db down")
    mock_engine.begin.return_value = mock_begin_ctx

    repo = AnalysisRequestLogRepository(
        engine=mock_engine,
        local_log_path=str(log_path),
        use_default_engine=False,
        local_daily_split=False,
    )
    ok = repo.insert_request_log(
        source_endpoint="/analyze",
        target_id=1,
        transcript_id=2,
        language="ko",
        target_type=None,
        country=None,
        sentences=["문장 하나"],
    )

    assert ok is True
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["source_endpoint"] == "/analyze"
    assert row["sentences"] == [{"sentence_text": "문장 하나"}]
    assert repo.last_sink == "local_fallback"
