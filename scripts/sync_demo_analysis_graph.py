"""Sync analyzed demo videos from Postgres into Neo4j HAS_ANALYSIS edges."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync demo AnalysisResult nodes to Neo4j")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-name", default="newsdb")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD"))
    return parser.parse_args()


def _load_video_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [
            row["youtube_video_id"].strip()
            for row in csv.DictReader(f)
            if row.get("youtube_video_id") and row["youtube_video_id"].strip()
        ]


def _quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _load_analysis_rows(args: argparse.Namespace, video_ids: list[str]) -> list[dict[str, Any]]:
    quoted = ",".join(_quote_sql(video_id) for video_id in video_ids)
    sql = f"""
select v.youtube_video_id,
       r.bias_analysis_result_id,
       coalesce(r.overall_bias_score, 0),
       coalesce(r.opinion_score, 0)
from youtube_video v
join bias_analysis_result r
  on r.target_id = v.id
 and r.target_type = 'YOUTUBE_VIDEO'
where v.youtube_video_id in ({quoted})
order by v.youtube_video_id;
"""
    env = os.environ.copy()
    if args.db_password:
        env["PGPASSWORD"] = args.db_password
    raw = subprocess.check_output(
        [
            "psql",
            "-h",
            args.db_host,
            "-U",
            args.db_user,
            "-d",
            args.db_name,
            "-At",
            "-F",
            "|",
            "-c",
            sql,
        ],
        env=env,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        video_id, result_id, overall, opinion = line.split("|")
        rows.append(
            {
                "video_id": video_id,
                "analysis_result_id": int(result_id),
                "overall_bias_score": float(overall),
                "opinion_score": float(opinion),
            }
        )
    return rows


def _sync_neo4j(rows: list[dict[str, Any]]) -> int:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    query = """
UNWIND $rows AS row
MATCH (v:Video {video_id: row.video_id})
MERGE (a:AnalysisResult {analysis_result_id: row.analysis_result_id})
SET a.status = 'SUCCESS',
    a.overall_bias_score = row.overall_bias_score,
    a.opinion_score = row.opinion_score
MERGE (v)-[:HAS_ANALYSIS]->(a)
RETURN count(*) AS synced
"""
    try:
        with driver.session() as session:
            result = session.run(query, rows=rows).single()
            return int(result["synced"] if result else 0)
    finally:
        driver.close()


def main() -> int:
    args = _parse_args()
    video_ids = _load_video_ids(args.csv)
    rows = _load_analysis_rows(args, video_ids)
    missing = sorted(set(video_ids) - {row["video_id"] for row in rows})
    synced = _sync_neo4j(rows)
    print(f"db_analysis_rows={len(rows)}")
    print(f"neo4j_has_analysis_synced={synced}")
    if missing:
        print("missing_db_analysis=" + ",".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
