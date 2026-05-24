"""Apply the manual Trump-Taiwan demo curation CSV through Spring APIs.

This script does not create the curated decisions itself. It consumes the CSV
after validation, ensures each YouTube video exists in Spring, creates one
CURATION_MANUAL set, adds the items, optionally triggers analysis, and
optionally locks the set to sync Neo4j relationships.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from validate_demo_curation import EXPECTED_ISSUE_NAME, validate


@dataclass(frozen=True)
class CurationRow:
    country_code: str
    youtube_video_id: str
    published_at: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Trump-Taiwan demo curation set")
    parser.add_argument("--csv", type=Path, required=True, help="Validated curation CSV path")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8080", help="Spring backend URL")
    parser.add_argument("--auth-token", default=None, help="Optional bearer token")
    parser.add_argument("--period-start", default=None, help="Override period start date YYYY-MM-DD")
    parser.add_argument("--period-end", default=None, help="Override period end date YYYY-MM-DD")
    parser.add_argument("--trigger-analysis", action="store_true", help="Call /api/v1/analysis/analyze/{youtubeVideoId}")
    parser.add_argument("--lock", action="store_true", help="Lock curation set after adding all items")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow applying a dataset that does not yet meet 10-per-country validation",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[CurationRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [
            CurationRow(
                country_code=(row.get("country_code") or "").strip().upper(),
                youtube_video_id=(row.get("youtube_video_id") or "").strip(),
                published_at=(row.get("published_at") or "").strip(),
            )
            for row in reader
            if (row.get("youtube_video_id") or "").strip()
        ]


def _headers(auth_token: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def _request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    auth_token: str | None = None,
) -> Any:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(auth_token), method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc
    payload = json.loads(raw) if raw else None
    if isinstance(payload, dict) and "body" in payload:
        return payload["body"]
    return payload


def _date_part(value: str) -> str | None:
    if not value:
        return None
    return value[:10]


def _default_period(rows: list[CurationRow]) -> tuple[str, str]:
    dates = sorted(d for d in (_date_part(row.published_at) for row in rows) if d)
    today = date.today().isoformat()
    if not dates:
        return today, today
    return dates[0], dates[-1]


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def main() -> int:
    args = _parse_args()
    validation = validate(args.csv, allow_draft=args.allow_draft)
    if not validation.ok:
        print("curation CSV failed validation; use --allow-draft only for local dry-run datasets")
        for error in validation.errors:
            print(f"- {error}")
        return 1

    rows = _load_rows(args.csv)
    period_start, period_end = _default_period(rows)
    period_start = args.period_start or period_start
    period_end = args.period_end or period_end
    backend = args.backend_url.rstrip("/")

    created = _request(
        "POST",
        f"{backend}/api/v1/issues/curation/sets",
        body={
            "searchKeyword": EXPECTED_ISSUE_NAME,
            "periodStartDate": period_start,
            "periodEndDate": period_end,
        },
        auth_token=args.auth_token,
    )
    issue_cluster_id = created["issueClusterId"]
    print(f"created curation set issueClusterId={issue_cluster_id}")

    for row in rows:
        video_id = _quote(row.youtube_video_id)
        _request("GET", f"{backend}/api/v1/youtube/{video_id}", auth_token=args.auth_token)
        target = _request("GET", f"{backend}/api/v1/comparison/videos/{video_id}/target", auth_token=args.auth_token)
        target_id = target["targetId"]
        _request(
            "POST",
            f"{backend}/api/v1/issues/curation/sets/{issue_cluster_id}/items",
            body={"videoId": target_id, "countryCode": row.country_code},
            auth_token=args.auth_token,
        )
        print(f"added {row.country_code} {row.youtube_video_id} targetId={target_id}")
        if args.trigger_analysis:
            _request("POST", f"{backend}/api/v1/analysis/analyze/{video_id}", auth_token=args.auth_token)
            print(f"analysis triggered {row.youtube_video_id}")

    if args.lock:
        status = _request(
            "POST",
            f"{backend}/api/v1/issues/curation/sets/{issue_cluster_id}/lock",
            auth_token=args.auth_token,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"not locked; lock later with POST /api/v1/issues/curation/sets/{issue_cluster_id}/lock")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
