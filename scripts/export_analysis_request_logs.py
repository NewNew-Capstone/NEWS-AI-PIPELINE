"""analysis_request_log 테이블에서 최근 로그를 JSONL로 내보낸다.

사용 예시:
    .venv/bin/python scripts/export_analysis_request_logs.py \
      --output logs/analyze_requests.jsonl \
      --days 7 \
      --limit 5000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sqlalchemy import create_engine, text


def main() -> None:
    parser = argparse.ArgumentParser(description="운영 분석 요청 로그 export")
    parser.add_argument("--output", type=Path, required=True, help="출력 JSONL 경로")
    parser.add_argument("--days", type=int, default=7, help="최근 N일")
    parser.add_argument("--limit", type=int, default=5000, help="최대 row 수")
    parser.add_argument("--db-url", type=str, default="", help="미지정 시 ANALYSIS_DB_URL 사용")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("ANALYSIS_DB_URL", "")
    if not db_url:
        raise RuntimeError("DB URL이 없습니다. --db-url 또는 ANALYSIS_DB_URL을 설정하세요.")

    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    stmt = text(
        """
        SELECT
            created_at,
            source_endpoint,
            target_id,
            transcript_id,
            language,
            target_type,
            country,
            sentence_count,
            sentences_json
        FROM analysis_request_log
        WHERE created_at >= NOW() - make_interval(days => :days)
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with engine.connect() as conn, args.output.open("w", encoding="utf-8") as f:
        rows = conn.execute(stmt, {"days": args.days, "limit": args.limit})
        for row in rows:
            sentences_json = row.sentences_json or []
            if isinstance(sentences_json, str):
                try:
                    sentences_json = json.loads(sentences_json)
                except json.JSONDecodeError:
                    sentences_json = []
            sentences = []
            for s in sentences_json:
                if isinstance(s, str) and s.strip():
                    sentences.append({"sentence_text": s})

            payload = {
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "source_endpoint": row.source_endpoint,
                "target_id": row.target_id,
                "transcript_id": row.transcript_id,
                "language": row.language,
                "target_type": row.target_type,
                "country": row.country,
                "sentence_count": row.sentence_count,
                "sentences": sentences,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"export 완료: {written} rows -> {args.output}")


if __name__ == "__main__":
    main()
