from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from typing import Any


DEFAULT_TITLE = "[지식뉴스] 시진핑의 대만 야욕..오히려 벼랑 끝 트럼프 살렸다?"


def _request_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw}
        return exc.code, parsed


def _edge_missing_fields(edge: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in ("weight", "keywords", "reasons", "similarity_score"):
        if field not in edge:
            missing.append(field)
    if "score_breakdown" in edge and not isinstance(edge["score_breakdown"], dict):
        missing.append("score_breakdown_type")
    if "keywords" in edge and not edge["keywords"]:
        missing.append("keywords_empty")
    if "reasons" in edge and not edge["reasons"]:
        missing.append("reasons_empty")
    return missing


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "keyword": args.keyword,
        "max_per_country": args.max_per_country,
        "selected_video": {
            "video_id": args.video_id,
            "title": args.title,
            "description": args.description,
            "country_code": args.country_code,
            "language": args.language,
            "channel_name": args.channel_name,
        },
    }


def verify(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    failures: list[str] = []

    health_status, health_body = _request_json(f"{base_url}/kg/health")
    if health_status != 200:
        failures.append(f"health_not_ok:{health_status}:{health_body}")

    compare_status, compare_body = _request_json(
        f"{base_url}/kg/realtime-compare/clicked-video",
        method="POST",
        payload=_build_payload(args),
    )
    if compare_status != 200:
        failures.append(f"compare_not_ok:{compare_status}:{compare_body}")
        print("FAIL feature2 demo")
        for failure in failures:
            print(f"- {failure}")
        return 1

    graph = compare_body.get("current_graph") or {}
    source = graph.get("source_video") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    if source.get("country_code") != args.country_code:
        failures.append(f"source_country_expected_{args.country_code}_got_{source.get('country_code')}")

    related_nodes = [node for node in nodes if node.get("node_type") == "related"]
    related_counts = Counter(node.get("country_code") for node in related_nodes)
    if related_counts.get(args.country_code, 0) > 0:
        failures.append(f"source_country_leaked_into_related:{args.country_code}")

    for country in args.expected_countries:
        count = related_counts.get(country, 0)
        if count < args.min_per_country:
            failures.append(f"related_shortage:{country}:{count}/{args.min_per_country}")

    if len(edges) < args.min_edges:
        failures.append(f"edge_shortage:{len(edges)}/{args.min_edges}")

    edge_field_failures: list[str] = []
    for index, edge in enumerate(edges):
        missing = _edge_missing_fields(edge)
        if missing:
            edge_field_failures.append(f"edge[{index}]={','.join(missing)}")
    if edge_field_failures:
        failures.append("edge_field_missing:" + ";".join(edge_field_failures))

    status = "PASS" if not failures else "FAIL"
    print(f"{status} feature2 demo")
    print(f"source={source.get('video_id')} country={source.get('country_code')}")
    print(f"related_counts={dict(sorted((key, value) for key, value in related_counts.items() if key))}")
    print(f"edges={len(edges)}")
    if failures:
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify feature2 demo graph response.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video-id", default="q0Jo5F8pHbs")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--description", default="")
    parser.add_argument("--country-code", default="KR")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--channel-name", default="교양이를 부탁해")
    parser.add_argument("--keyword", default="트럼프 대만")
    parser.add_argument("--max-per-country", type=int, default=3)
    parser.add_argument("--expected-countries", nargs="+", default=["US", "CN"])
    parser.add_argument("--min-per-country", type=int, default=3)
    parser.add_argument("--min-edges", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(verify(parse_args()))
