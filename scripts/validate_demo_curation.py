"""Validate the manual Trump-Taiwan demo curation CSV.

The validator intentionally fails before the demo dataset reaches the minimum
shape needed by feature 2: KR/US/CN, 10-15 videos per country, unique videos,
language/country consistency, and no single-channel dominance.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_ISSUE_ID = "trump_taiwan_2026_demo"
EXPECTED_ISSUE_NAME = "트럼프-대만/미중 갈등"
COUNTRY_LANGUAGE = {"KR": "ko", "US": "en", "CN": "zh"}
REQUIRED_COLUMNS = {
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
}
OPTIONAL_COLUMNS = {"stance_hint", "frame_hint", "backup_rank"}
MIN_PER_COUNTRY = 10
MAX_PER_COUNTRY = 15
MAX_CHANNEL_SHARE = 0.40


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    row_count: int
    errors: list[str]
    warnings: list[str]
    country_counts: dict[str, int]
    channel_top1_share: dict[str, float]
    backup_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Trump-Taiwan demo curation CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Curation CSV path")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow fewer than 10 videos per country; still reports errors as warnings",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        return rows, list(reader.fieldnames)


def _valid_datetime(value: str) -> bool:
    if not value:
        return False
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
        return True
    except ValueError:
        return False


def validate(path: Path, *, allow_draft: bool = False) -> ValidationResult:
    rows, columns = _read_rows(path)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
    unexpected_columns = sorted(set(columns) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
    if unexpected_columns:
        warnings.append(f"unexpected columns ignored by tooling: {unexpected_columns}")

    country_counts: Counter[str] = Counter()
    by_country_channel: dict[str, Counter[str]] = defaultdict(Counter)
    seen_video_ids: set[str] = set()
    backup_count = 0

    for index, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if not row.get(column):
                errors.append(f"row {index}: missing {column}")

        issue_id = row.get("issue_id", "")
        if issue_id and issue_id != EXPECTED_ISSUE_ID:
            errors.append(f"row {index}: issue_id must be {EXPECTED_ISSUE_ID!r}")

        issue_name = row.get("issue_name", "")
        if issue_name and issue_name != EXPECTED_ISSUE_NAME:
            warnings.append(f"row {index}: issue_name differs from fixed demo name")

        country = row.get("country_code", "").upper()
        language = row.get("language", "").lower()
        if country not in COUNTRY_LANGUAGE:
            errors.append(f"row {index}: country_code must be one of {sorted(COUNTRY_LANGUAGE)}")
        elif language != COUNTRY_LANGUAGE[country]:
            errors.append(f"row {index}: language for {country} must be {COUNTRY_LANGUAGE[country]!r}")

        youtube_video_id = row.get("youtube_video_id", "")
        if youtube_video_id:
            if youtube_video_id in seen_video_ids:
                errors.append(f"row {index}: duplicate youtube_video_id {youtube_video_id!r}")
            seen_video_ids.add(youtube_video_id)

        if row.get("published_at") and not _valid_datetime(row["published_at"]):
            errors.append(f"row {index}: published_at must be ISO-8601")

        country_counts[country] += 1
        if row.get("channel_id"):
            by_country_channel[country][row["channel_id"]] += 1
        if row.get("backup_rank"):
            backup_count += 1

    for country in COUNTRY_LANGUAGE:
        count = country_counts.get(country, 0)
        if count < MIN_PER_COUNTRY:
            message = f"{country}: needs at least {MIN_PER_COUNTRY} videos, found {count}"
            if allow_draft:
                warnings.append(message)
            else:
                errors.append(message)
        if count > MAX_PER_COUNTRY:
            errors.append(f"{country}: max {MAX_PER_COUNTRY} videos allowed, found {count}")

    top1_share: dict[str, float] = {}
    for country in COUNTRY_LANGUAGE:
        total = country_counts.get(country, 0)
        top1 = by_country_channel[country].most_common(1)[0][1] if by_country_channel[country] else 0
        share = 0.0 if total == 0 else top1 / total
        top1_share[country] = round(share, 4)
        if total >= MIN_PER_COUNTRY and share > MAX_CHANNEL_SHARE:
            errors.append(f"{country}: top channel share {share:.2%} exceeds {MAX_CHANNEL_SHARE:.0%}")

    ok = not errors
    return ValidationResult(
        ok=ok,
        row_count=len(rows),
        errors=errors,
        warnings=warnings,
        country_counts=dict(country_counts),
        channel_top1_share=top1_share,
        backup_count=backup_count,
    )


def _write_json(path: Path, result: ValidationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ok": result.ok,
        "row_count": result.row_count,
        "errors": result.errors,
        "warnings": result.warnings,
        "country_counts": result.country_counts,
        "channel_top1_share": result.channel_top1_share,
        "backup_count": result.backup_count,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, result: ValidationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trump-Taiwan Demo Curation Validation",
        "",
        f"- Status: {'PASS' if result.ok else 'FAIL'}",
        f"- Rows: {result.row_count}",
        f"- Backup rows: {result.backup_count}",
        "",
        "## Country Counts",
        "",
        "| Country | Count | Top Channel Share |",
        "| --- | ---: | ---: |",
    ]
    for country in COUNTRY_LANGUAGE:
        lines.append(
            f"| {country} | {result.country_counts.get(country, 0)} | "
            f"{result.channel_top1_share.get(country, 0.0):.2%} |"
        )
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {error}" for error in result.errors] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    result = validate(args.csv, allow_draft=args.allow_draft)
    if args.output_json:
        _write_json(args.output_json, result)
    if args.output_md:
        _write_md(args.output_md, result)

    print("PASS" if result.ok else "FAIL")
    print(f"rows={result.row_count} country_counts={result.country_counts}")
    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"- {error}")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
