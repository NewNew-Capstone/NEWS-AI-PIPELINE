from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_emotion_eval_from_logs import _extract_rows_from_record, _parse_dt


def test_parse_dt_iso_z() -> None:
    dt = _parse_dt("2026-04-25T12:34:56Z")
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2026
    assert dt.month == 4
    assert dt.day == 25


def test_extract_rows_from_sentences() -> None:
    record = {
        "created_at": "2026-04-25T01:00:00Z",
        "sentences": [
            {"sentence_text": "첫 번째 문장"},
            {"sentence_text": "두 번째 문장"},
        ],
    }
    rows = _extract_rows_from_record(record)
    assert [r.sentence_text for r in rows] == ["첫 번째 문장", "두 번째 문장"]
    assert all(r.source == "sentences" for r in rows)
    assert all(isinstance(r.created_at, datetime) for r in rows)


def test_extract_rows_from_raw_text() -> None:
    record = {
        "raw_text": "첫 문장입니다. 둘째 문장입니다.",
        "language": "ko",
    }
    rows = _extract_rows_from_record(record)
    assert len(rows) >= 1
    assert all(r.source == "raw_text" for r in rows)
