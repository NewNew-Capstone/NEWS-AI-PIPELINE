from __future__ import annotations

import csv
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_demo_curation import EXPECTED_ISSUE_ID, EXPECTED_ISSUE_NAME, validate


HEADER = [
    "issue_id",
    "issue_name",
    "country_code",
    "language",
    "youtube_video_id",
    "title",
    "channel_name",
    "channel_id",
    "published_at",
    "curation_reason",
    "stance_hint",
    "frame_hint",
    "backup_rank",
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _row(country: str, index: int, channel: int | None = None) -> dict[str, str]:
    language = {"KR": "ko", "US": "en", "CN": "zh"}[country]
    channel_index = channel if channel is not None else index
    return {
        "issue_id": EXPECTED_ISSUE_ID,
        "issue_name": EXPECTED_ISSUE_NAME,
        "country_code": country,
        "language": language,
        "youtube_video_id": f"{country.lower()}-{index:02d}",
        "title": f"{country} Trump Taiwan video {index}",
        "channel_name": f"{country} channel {channel_index}",
        "channel_id": f"{country.lower()}-channel-{channel_index}",
        "published_at": "2026-05-20T00:00:00",
        "curation_reason": "Trump and Taiwan are directly connected in the issue framing.",
        "stance_hint": "",
        "frame_hint": "",
        "backup_rank": "",
    }


def test_validate_passes_balanced_demo_dataset(tmp_path: Path) -> None:
    rows = []
    for country in ("KR", "US", "CN"):
        rows.extend(_row(country, index) for index in range(1, 11))
    path = tmp_path / "curation.csv"
    _write_csv(path, rows)

    result = validate(path)

    assert result.ok is True
    assert result.country_counts == {"KR": 10, "US": 10, "CN": 10}


def test_validate_rejects_channel_dominance(tmp_path: Path) -> None:
    rows = []
    for country in ("KR", "US", "CN"):
        rows.extend(_row(country, index, channel=1 if index <= 5 else index) for index in range(1, 11))
    path = tmp_path / "curation.csv"
    _write_csv(path, rows)

    result = validate(path)

    assert result.ok is False
    assert any("top channel share" in error for error in result.errors)


def test_validate_allow_draft_downgrades_count_errors(tmp_path: Path) -> None:
    rows = [_row("KR", 1), _row("US", 1), _row("CN", 1)]
    path = tmp_path / "curation.csv"
    _write_csv(path, rows)

    result = validate(path, allow_draft=True)

    assert result.ok is True
    assert any("needs at least" in warning for warning in result.warnings)
