"""Verify demo curation readiness from Spring comparison endpoints."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Trump-Taiwan demo curation APIs")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--issue-cluster-id", type=int, required=True)
    parser.add_argument("--source-video-id", required=True, help="YouTube video id used for graph verification")
    parser.add_argument("--keyword", default="트럼프 대만")
    return parser.parse_args()


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("body", payload)


def _country_sections_ok(body: dict[str, Any]) -> bool:
    sections = body.get("sections") or []
    countries = {section.get("country_code") or section.get("countryCode") for section in sections}
    return {"KR", "US", "CN"} <= countries


def main() -> int:
    args = _parse_args()
    backend = args.backend_url.rstrip("/")
    keyword = urllib.parse.quote(args.keyword)
    video_id = urllib.parse.quote(args.source_video_id, safe="")

    report = _get_json(f"{backend}/api/v1/issues/comparison/report?issueClusterId={args.issue_cluster_id}")
    home = _get_json(f"{backend}/api/v1/comparison/home")
    search = _get_json(f"{backend}/api/v1/comparison/search?keyword={keyword}")
    graph = _get_json(f"{backend}/api/v1/comparison/videos/{video_id}/graph")

    checks = {
        "report_ready": bool(report.get("ready")),
        "report_country_count_min_10": all((c.get("videoCount") or 0) >= 10 for c in report.get("countries", [])),
        "home_has_three_sections": _country_sections_ok(home),
        "search_has_three_sections": _country_sections_ok(search),
        "graph_has_edges": bool(graph.get("edges")),
        "graph_has_related_nodes": any(node.get("node_type") == "related" for node in graph.get("nodes", [])),
    }
    print(json.dumps({"checks": checks, "report": report}, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
