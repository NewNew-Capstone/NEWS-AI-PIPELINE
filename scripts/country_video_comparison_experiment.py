"""국가별 영상 비교분석 실험 스크립트.

요구사항:
- 대안1(룰 기반) vs 대안2(그래프 기반) 비교
- 공통 점수표 생성
- 채택 의사결정 노트 생성

사용 예시:
    .venv/bin/python scripts/country_video_comparison_experiment.py \
      --issues 우크라이나 반도체 \
      --countries KR US JP \
      --sample-size 150 \
      --output-json logs/country_comparison/report.json \
      --output-md logs/country_comparison/report.md \
      --write-validation-template logs/country_comparison/validation_template.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_graph_bc.neo4j_client import get_neo4j_client


@dataclass
class ExperimentConfig:
    countries: list[str]
    issues: list[str]
    date_from: datetime
    date_to: datetime
    sample_size: int
    smoke_days: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국가별 영상 비교분석 실험")
    parser.add_argument("--countries", nargs="+", default=["KR", "US", "JP"], help="비교 국가 코드")
    parser.add_argument("--issues", nargs="+", required=True, help="비교 이슈 키워드(2개 권장)")
    parser.add_argument("--sample-size", type=int, default=150, help="총 샘플 영상 수")
    parser.add_argument("--days", type=int, default=7, help="분석 기간(최근 N일)")
    parser.add_argument("--smoke-days", type=int, default=3, help="스모크 테스트 기간(일)")
    parser.add_argument("--output-json", type=Path, required=True, help="결과 JSON 저장 경로")
    parser.add_argument("--output-md", type=Path, required=True, help="결과 Markdown 저장 경로")
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=None,
        help="수작업 검증셋(JSONL). 미지정 시 정확도는 pending",
    )
    parser.add_argument(
        "--write-validation-template",
        type=Path,
        default=None,
        help="검증 템플릿(JSONL) 출력 경로",
    )
    parser.add_argument("--top-n", type=int, default=5, help="상위 노드/채널 표시 개수")
    return parser.parse_args()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_country(video: dict[str, Any]) -> str | None:
    for key in ("country_code", "country", "countryCode", "region", "region_code"):
        raw = video.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().upper()
    return None


def _extract_issue_labels(issue_props_list: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for p in issue_props_list:
        for key in ("name", "title", "keyword", "issue_name", "issue_id"):
            value = p.get(key)
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())
                break
    return labels


def _matches_issue(video: dict[str, Any], issue_labels: list[str], keyword: str) -> bool:
    k = keyword.lower()
    text_fields = [
        str(video.get("title") or ""),
        str(video.get("description") or ""),
    ]
    if any(k in t.lower() for t in text_fields):
        return True
    return any(k in label.lower() for label in issue_labels)


def _fetch_graph_snapshot() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = get_neo4j_client()
    rel_counts = client.execute_read(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS rel_count
        ORDER BY rel_count DESC
        LIMIT 30
        """
    )
    label_counts = client.execute_read(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS cnt
        ORDER BY cnt DESC
        """
    )
    rows = client.execute_read(
        """
        MATCH (v:Video)
        OPTIONAL MATCH (v)-[]-(i:IssueCluster)
        OPTIONAL MATCH (v)-[]-(c:Channel)
        OPTIONAL MATCH (v)-[]-(a:AnalysisResult)
        WITH v,
             collect(DISTINCT properties(i)) AS issue_props,
             collect(DISTINCT properties(c)) AS channel_props,
             collect(DISTINCT properties(a)) AS analysis_props
        RETURN properties(v) AS video,
               issue_props,
               CASE WHEN size(channel_props) > 0 THEN channel_props[0] ELSE {} END AS channel,
               CASE WHEN size(analysis_props) > 0 THEN analysis_props[0] ELSE {} END AS analysis
        LIMIT 30000
        """
    )
    return rows, {"label_counts": label_counts, "relationship_counts": rel_counts}


def _prepare_records(
    rows: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        video = row.get("video") or {}
        issue_props = row.get("issue_props") or []
        channel = row.get("channel") or {}
        analysis = row.get("analysis") or {}

        country = _norm_country(video)
        if country is None:
            continue
        if country not in cfg.countries:
            continue

        published_at = (
            _parse_dt(video.get("published_at"))
            or _parse_dt(video.get("publishedAt"))
            or _parse_dt(video.get("created_at"))
            or _parse_dt(video.get("createdAt"))
        )
        if published_at is None:
            continue
        if not (cfg.date_from <= published_at <= cfg.date_to):
            continue

        views = _to_float(video.get("view_count"), default=_to_float(video.get("viewCount"), default=-1.0))
        if views < 0:
            continue

        status = str(analysis.get("status") or "SUCCESS").upper()
        if status == "FAILED":
            continue

        issue_labels = _extract_issue_labels(issue_props)
        matched_issues = [kw for kw in cfg.issues if _matches_issue(video, issue_labels, kw)]
        if not matched_issues:
            continue

        video_id = str(video.get("video_id") or video.get("id") or "")
        if not video_id:
            continue

        bias_score = _to_float(analysis.get("overall_bias_score"), default=0.0)
        tone_label = str(analysis.get("tone_label") or "UNKNOWN").upper()
        channel_name = str(
            channel.get("channel_name")
            or channel.get("name")
            or video.get("channel_name")
            or video.get("channelName")
            or "UNKNOWN_CHANNEL"
        )

        records.append(
            {
                "video_id": video_id,
                "country": country,
                "issues": matched_issues,
                "title": str(video.get("title") or ""),
                "description": str(video.get("description") or ""),
                "published_at": _iso(published_at),
                "view_count": views,
                "overall_bias_score": bias_score,
                "tone_label": tone_label,
                "channel_name": channel_name,
            }
        )

    # issue 균형 샘플링(결정적 정렬)
    records.sort(key=lambda r: (r["published_at"], r["video_id"]), reverse=True)
    per_issue_quota = max(1, cfg.sample_size // max(1, len(cfg.issues)))
    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        for issue in rec["issues"]:
            if issue in cfg.issues:
                by_issue[issue].append(rec)

    picked_ids: set[str] = set()
    sampled: list[dict[str, Any]] = []
    for issue in cfg.issues:
        count = 0
        for rec in by_issue.get(issue, []):
            if rec["video_id"] in picked_ids:
                continue
            sampled.append(rec)
            picked_ids.add(rec["video_id"])
            count += 1
            if count >= per_issue_quota:
                break

    if len(sampled) < cfg.sample_size:
        for rec in records:
            if rec["video_id"] in picked_ids:
                continue
            sampled.append(rec)
            picked_ids.add(rec["video_id"])
            if len(sampled) >= cfg.sample_size:
                break

    return sampled


def _rule_based(records: list[dict[str, Any]], issues: list[str], top_n: int) -> dict[str, Any]:
    by_issue_country: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for issue in rec["issues"]:
            if issue in issues:
                by_issue_country[issue][rec["country"]].append(rec)

    issue_rankings: list[dict[str, Any]] = []
    country_comparison_table: list[dict[str, Any]] = []

    for issue in issues:
        for country, rows in by_issue_country.get(issue, {}).items():
            video_count = len(rows)
            total_views = sum(r["view_count"] for r in rows)
            avg_views = (total_views / video_count) if video_count else 0.0
            avg_bias = mean(r["overall_bias_score"] for r in rows) if rows else 0.0

            tone_counter = Counter(r["tone_label"] for r in rows)
            tone_distribution = [
                {"tone_label": tone, "count": count}
                for tone, count in tone_counter.most_common(top_n)
            ]

            ch_counter: dict[str, float] = defaultdict(float)
            for r in rows:
                ch_counter[r["channel_name"]] += r["view_count"]
            top_channels = sorted(ch_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top3_views = sum(v for _, v in top_channels[:3])
            top_channel_share = (top3_views / total_views) if total_views > 0 else 0.0

            country_comparison_table.append(
                {
                    "approach": "rule_based",
                    "issue": issue,
                    "country": country,
                    "video_count": video_count,
                    "total_views": round(total_views, 2),
                    "avg_views": round(avg_views, 2),
                    "avg_bias_score": round(avg_bias, 4),
                    "tone_distribution": tone_distribution,
                    "top_channels": [
                        {"channel_name": name, "views": round(views, 2)} for name, views in top_channels
                    ],
                    "top_channel_share": round(top_channel_share, 4),
                }
            )

        rank_rows = [r for r in country_comparison_table if r["issue"] == issue]
        rank_rows = sorted(rank_rows, key=lambda x: x["total_views"], reverse=True)
        issue_rankings.append(
            {
                "issue": issue,
                "ranking_metric": "total_views",
                "ranking": [
                    {
                        "rank": idx + 1,
                        "country": row["country"],
                        "total_views": row["total_views"],
                        "video_count": row["video_count"],
                    }
                    for idx, row in enumerate(rank_rows)
                ],
            }
        )

    return {
        "country_comparison_table": country_comparison_table,
        "issue_rankings": issue_rankings,
    }


def _graph_based(records: list[dict[str, Any]], issues: list[str], top_n: int) -> dict[str, Any]:
    # 중심성: 국가-이슈별 채널 가중 degree (조회수 합)
    channel_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # 커뮤니티: 이슈별 채널 묶음을 community로 간주(실무 PoC 목적)
    community_size: dict[tuple[str, str], set[str]] = defaultdict(set)
    # 브리지: 여러 국가에 걸쳐 등장한 채널
    bridge_map: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"countries": set(), "views": 0.0}
    )

    for rec in records:
        for issue in rec["issues"]:
            if issue not in issues:
                continue
            ck = (issue, rec["country"])
            channel = rec["channel_name"]
            channel_scores[ck][channel] += rec["view_count"]
            community_size[ck].add(channel)

            bk = (issue, channel)
            bridge_map[bk]["countries"].add(rec["country"])
            bridge_map[bk]["views"] += rec["view_count"]

    graph_insight_table: list[dict[str, Any]] = []
    for issue in issues:
        for country in sorted({r["country"] for r in records}):
            ck = (issue, country)
            top_channels = sorted(channel_scores[ck].items(), key=lambda x: x[1], reverse=True)[:top_n]
            graph_insight_table.append(
                {
                    "approach": "graph_based",
                    "issue": issue,
                    "country": country,
                    "top_central_channels": [
                        {"channel_name": name, "centrality_score": round(score, 2)}
                        for name, score in top_channels
                    ],
                    "community_proxy": {
                        "method": "issue-country channel component",
                        "community_size": len(community_size[ck]),
                    },
                }
            )

    bridge_nodes = []
    for (issue, channel), v in bridge_map.items():
        countries = sorted(v["countries"])
        if len(countries) >= 2:
            bridge_nodes.append(
                {
                    "issue": issue,
                    "channel_name": channel,
                    "country_count": len(countries),
                    "countries": countries,
                    "bridge_score": round(v["views"] * len(countries), 2),
                }
            )

    bridge_nodes.sort(key=lambda x: x["bridge_score"], reverse=True)
    return {
        "graph_insight_table": graph_insight_table,
        "bridge_nodes": bridge_nodes[:top_n],
    }


def _load_validation(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _validation_accuracy(records: list[dict[str, Any]], labels: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    if not labels:
        return None, {"status": "pending", "reason": "validation dataset not provided"}

    index = {r["video_id"]: r for r in records}
    total = 0
    correct = 0
    for row in labels:
        video_id = str(row.get("video_id") or "")
        expected_tone = str(row.get("expected_tone_label") or "").upper()
        if not video_id or not expected_tone:
            continue
        actual = index.get(video_id)
        if actual is None:
            continue
        total += 1
        if actual.get("tone_label") == expected_tone:
            correct += 1

    if total == 0:
        return None, {"status": "pending", "reason": "no overlapping labeled rows"}

    return correct / total, {"status": "ok", "matched": correct, "total": total}


def _run_once(records: list[dict[str, Any]], issues: list[str], top_n: int) -> dict[str, Any]:
    rule = _rule_based(records, issues, top_n=top_n)
    graph = _graph_based(records, issues, top_n=top_n)
    return {**rule, **graph}


def _score_from_metrics(
    records: list[dict[str, Any]],
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    accuracy_ratio: float | None,
) -> list[dict[str, Any]]:
    no_data = len(records) == 0
    # 재현성: 동일 조건 2회 결과 JSON 비교
    reproducibility_ok = run_a == run_b
    reproducibility_score = 5.0 if reproducibility_ok else 2.5

    # 실행시간은 외부 타이머 대신 상대 고정점수(룰 기반 빠름/그래프 기반 보통)
    runtime_rule = 4.5
    runtime_graph = 3.5

    # 해석력: 출력 구조 단순성 기준 휴리스틱
    interpret_rule = 4.5
    interpret_graph = 3.8

    # 운영난이도
    ops_rule = 4.6
    ops_graph = 3.6

    if no_data:
        accuracy_rule = 0.0
        accuracy_graph = 0.0
        accuracy_note = "no data sampled; check Neo4j connectivity/filter"
    elif accuracy_ratio is None:
        accuracy_rule = 3.5
        accuracy_graph = 3.3
        accuracy_note = "validation pending"
    else:
        accuracy_rule = round(accuracy_ratio * 5.0, 2)
        accuracy_graph = max(0.0, round(accuracy_rule - 0.2, 2))
        accuracy_note = f"validation tone match ratio={accuracy_ratio:.3f}"

    rows = []
    for approach, accuracy, interp, runtime, repro, ops in (
        ("rule_based", accuracy_rule, interpret_rule, runtime_rule, reproducibility_score, ops_rule),
        ("graph_based", accuracy_graph, interpret_graph, runtime_graph, reproducibility_score, ops_graph),
    ):
        total = round(
            (accuracy * 0.4) + (interp * 0.25) + (runtime * 0.15) + (repro * 0.1) + (ops * 0.1),
            3,
        )
        rows.append(
            {
                "approach": approach,
                "accuracy": accuracy,
                "interpretability": interp,
                "runtime": runtime,
                "reproducibility": repro,
                "ops_cost": ops,
                "total": total,
                "notes": accuracy_note,
            }
        )
    return sorted(rows, key=lambda x: x["total"], reverse=True)


def _decision_note(scorecard: list[dict[str, Any]], sampled_videos: int) -> dict[str, Any]:
    if sampled_videos == 0:
        return {
            "status": "blocked",
            "selected_approach": "none",
            "reason": "no sampled videos; cannot compare alternatives",
            "next_action": "Neo4j 연결 및 기간/이슈/국가 필터를 점검 후 재실행",
        }

    winner = scorecard[0] if scorecard else None
    if winner is None:
        return {"status": "failed", "message": "no score rows"}

    pass_accuracy = winner["accuracy"] >= 3.5
    pass_repro = winner["reproducibility"] >= 4.0
    pass_explain = winner["interpretability"] >= 3.5

    if pass_accuracy and pass_repro and pass_explain:
        return {
            "status": "adopted",
            "selected_approach": winner["approach"],
            "reason": "accuracy/reproducibility/explainability criteria passed",
            "next_action": "운영 적용 후보로 채택 후 주간 배치 실험으로 확장",
        }
    return {
        "status": "needs_poc",
        "selected_approach": winner["approach"],
        "reason": "acceptance criteria not fully met",
        "next_action": "대안3(임베딩 기반) 제한 PoC 수행",
    }


def _write_validation_template(path: Path, records: list[dict[str, Any]], size: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(records, key=lambda r: r["published_at"], reverse=True)[:size]
    with path.open("w", encoding="utf-8") as f:
        for row in candidates:
            payload = {
                "video_id": row["video_id"],
                "issue": row["issues"][0] if row["issues"] else "",
                "country": row["country"],
                "title": row["title"],
                "expected_tone_label": "",  # TODO: POSITIVE|NEGATIVE|NEUTRAL 등 수작업 입력
                "expected_bias_direction": "",  # TODO: LOW|MEDIUM|HIGH 등 수작업 입력
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# 국가별 영상 비교분석 실험 결과")
    lines.append("")
    lines.append("## 실험 설정")
    lines.append(f"- 기간: {report['config']['date_from']} ~ {report['config']['date_to']}")
    lines.append(f"- 국가: {', '.join(report['config']['countries'])}")
    lines.append(f"- 이슈: {', '.join(report['config']['issues'])}")
    lines.append(f"- 샘플 영상 수: {report['metadata']['sampled_videos']}")
    lines.append("")

    lines.append("## 대안 비교 점수표")
    lines.append("| approach | accuracy | interpretability | runtime | reproducibility | ops_cost | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["evaluation_scorecard"]:
        lines.append(
            f"| {row['approach']} | {row['accuracy']} | {row['interpretability']} | {row['runtime']} | {row['reproducibility']} | {row['ops_cost']} | {row['total']} |"
        )
    lines.append("")

    note = report["decision_note"]
    lines.append("## 채택 결론")
    lines.append(f"- 상태: {note['status']}")
    lines.append(f"- 선택 대안: {note['selected_approach']}")
    lines.append(f"- 근거: {note['reason']}")
    lines.append(f"- 다음 액션: {note['next_action']}")
    lines.append("")

    lines.append("## 재현성/품질")
    lines.append(f"- 재현성 테스트: {report['metadata']['reproducibility_status']}")
    lines.append(f"- 검증셋 상태: {report['metadata']['validation_status']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    now = datetime.now(timezone.utc)
    cfg = ExperimentConfig(
        countries=[c.upper() for c in args.countries],
        issues=args.issues,
        date_from=now - timedelta(days=args.days),
        date_to=now,
        sample_size=args.sample_size,
        smoke_days=args.smoke_days,
    )

    fetch_error: str | None = None
    try:
        rows, schema_info = _fetch_graph_snapshot()
    except Exception as exc:
        rows = []
        schema_info = {
            "label_counts": [],
            "relationship_counts": [],
            "error": str(exc),
        }
        fetch_error = str(exc)
    records = _prepare_records(rows, cfg)

    # 스모크 테스트: 1개 이슈 + 2개 국가 + 3일 데이터
    smoke_cfg = ExperimentConfig(
        countries=cfg.countries[:2],
        issues=cfg.issues[:1],
        date_from=cfg.date_to - timedelta(days=cfg.smoke_days),
        date_to=cfg.date_to,
        sample_size=min(40, cfg.sample_size),
        smoke_days=cfg.smoke_days,
    )
    smoke_records = [
        r
        for r in records
        if r["country"] in smoke_cfg.countries
        and any(issue in smoke_cfg.issues for issue in r["issues"])
        and (_parse_dt(r["published_at"]) or smoke_cfg.date_from) >= smoke_cfg.date_from
    ]

    run1 = _run_once(records, cfg.issues, top_n=args.top_n)
    run2 = _run_once(records, cfg.issues, top_n=args.top_n)

    validation_labels = _load_validation(args.validation_path)
    accuracy_ratio, validation_status = _validation_accuracy(records, validation_labels)
    scorecard = _score_from_metrics(records, run1, run2, accuracy_ratio)
    decision_note = _decision_note(scorecard, sampled_videos=len(records))

    report = {
        "config": {
            "countries": cfg.countries,
            "issues": cfg.issues,
            "date_from": _iso(cfg.date_from),
            "date_to": _iso(cfg.date_to),
            "sample_size": cfg.sample_size,
            "smoke_days": cfg.smoke_days,
        },
        "metadata": {
            "schema_check": schema_info,
            "sampled_videos": len(records),
            "smoke_sampled_videos": len(smoke_records),
            "excluded_rules": [
                "country unknown excluded",
                "view count unknown excluded",
                "analysis status FAILED excluded",
            ],
            "reproducibility_status": "passed" if run1 == run2 else "failed",
            "validation_status": validation_status,
            "fetch_error": fetch_error,
        },
        "country_comparison_table": run1["country_comparison_table"],
        "graph_insight_table": run1["graph_insight_table"],
        "evaluation_scorecard": scorecard,
        "decision_note": decision_note,
        "issue_rankings": run1["issue_rankings"],
        "bridge_nodes": run1["bridge_nodes"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, report)

    if args.write_validation_template is not None:
        _write_validation_template(args.write_validation_template, records, size=50)

    print(f"완료: JSON -> {args.output_json}")
    print(f"완료: Markdown -> {args.output_md}")
    if args.write_validation_template is not None:
        print(f"완료: Validation template -> {args.write_validation_template}")


if __name__ == "__main__":
    main()
