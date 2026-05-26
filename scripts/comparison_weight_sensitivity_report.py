"""국가별 비교 추천 가중치 민감도 리포트 생성 스크립트.

목적:
- 비교 추천 baseline 가중치가 수작업 평가셋에서 Top3 품질을 유지하는지 확인
- issue/shared keyword/entity/semantic/meta 신호를 흔들었을 때 순위 안정성을 측정
- 발표 질의응답에서 "가중치가 임의인가?"에 답할 JSON/Markdown 근거 생성

사용 예시:
    .venv/bin/python scripts/comparison_weight_sensitivity_report.py

    .venv/bin/python scripts/comparison_weight_sensitivity_report.py \
      --grid-fields issue_overlap shared_keyword semantic_similarity \
      --grid-values 0,0.5,1,1.5,2
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_graph_bc.comparison_scoring import (  # noqa: E402
    DEFAULT_COMPARISON_SCORING_WEIGHTS,
    ComparisonScoringFeatures,
    ComparisonScoringWeights,
    features_from_mapping,
    score_breakdown,
    score_comparison_features,
)


DEFAULT_DATASET_PATH = Path("knowledge_graph_bc/data/eval/.jsonl")
DEFAULT_OUTPUT_DIR = Path("knowledge_graph_bc/data/eval/reports")
WEIGHT_FIELDS = tuple(asdict(DEFAULT_COMPARISON_SCORING_WEIGHTS).keys())


@dataclass(frozen=True)
class EvalRow:
    source_video_id: str
    source_country: str
    keyword: str
    candidate_video_id: str
    candidate_country: str
    expected_relevance: int
    reason_tags: list[str]
    note: str
    features: ComparisonScoringFeatures


@dataclass(frozen=True)
class WeightVariant:
    name: str
    weights: ComparisonScoringWeights


@dataclass(frozen=True)
class ScoredRow:
    row: EvalRow
    score: float


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _load_eval_rows(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            missing = [
                key
                for key in (
                    "source_video_id",
                    "source_country",
                    "keyword",
                    "candidate_video_id",
                    "candidate_country",
                    "expected_relevance",
                    "reason_tags",
                    "note",
                    "features",
                )
                if key not in payload
            ]
            if missing:
                raise ValueError(f"{path}:{line_no} missing required fields: {missing}")
            relevance = int(payload["expected_relevance"])
            if relevance < 0 or relevance > 3:
                raise ValueError(f"{path}:{line_no} expected_relevance must be 0..3")
            rows.append(
                EvalRow(
                    source_video_id=str(payload["source_video_id"]),
                    source_country=str(payload["source_country"]),
                    keyword=str(payload["keyword"]),
                    candidate_video_id=str(payload["candidate_video_id"]),
                    candidate_country=str(payload["candidate_country"]),
                    expected_relevance=relevance,
                    reason_tags=[str(tag) for tag in payload.get("reason_tags", [])],
                    note=str(payload.get("note", "")),
                    features=features_from_mapping(payload.get("features")),
                )
            )
    if not rows:
        raise ValueError(f"empty evaluation dataset: {path}")
    return rows


def _group_key(row: EvalRow) -> tuple[str, str, str]:
    return (row.source_video_id, row.keyword, row.candidate_country)


def _group_rows(rows: list[EvalRow]) -> dict[tuple[str, str, str], list[EvalRow]]:
    grouped: dict[tuple[str, str, str], list[EvalRow]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    return dict(grouped)


def _rank_rows(rows: list[EvalRow], weights: ComparisonScoringWeights) -> list[ScoredRow]:
    scored = [
        ScoredRow(row=row, score=score_comparison_features(row.features, weights))
        for row in rows
    ]
    return sorted(scored, key=lambda item: (-item.score, item.row.candidate_video_id))


def _dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def _ndcg_at_k(ranked: list[ScoredRow], k: int) -> float:
    actual = [item.row.expected_relevance for item in ranked[:k]]
    ideal = sorted((item.row.expected_relevance for item in ranked), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return 1.0 if ideal_dcg == 0 else _dcg(actual) / ideal_dcg


def _recall_at_k(ranked: list[ScoredRow], k: int, relevance_threshold: int) -> float:
    relevant_total = sum(1 for item in ranked if item.row.expected_relevance >= relevance_threshold)
    if relevant_total == 0:
        return 1.0
    relevant_top = sum(1 for item in ranked[:k] if item.row.expected_relevance >= relevance_threshold)
    return relevant_top / relevant_total


def _mrr(ranked: list[ScoredRow], relevance_threshold: int) -> float:
    for index, item in enumerate(ranked, start=1):
        if item.row.expected_relevance >= relevance_threshold:
            return 1.0 / index
    return 0.0


def _top_relevant_count(ranked: list[ScoredRow], k: int, relevance_threshold: int) -> int:
    return sum(1 for item in ranked[:k] if item.row.expected_relevance >= relevance_threshold)


def _rank_map(ranked: list[ScoredRow]) -> dict[str, int]:
    return {item.row.candidate_video_id: index + 1 for index, item in enumerate(ranked)}


def _spearman_from_rank_maps(base: dict[str, int], other: dict[str, int]) -> float:
    common = sorted(set(base) & set(other))
    n = len(common)
    if n < 2:
        return 1.0
    d2_sum = sum((base[key] - other[key]) ** 2 for key in common)
    return 1.0 - (6.0 * d2_sum / (n * (n * n - 1)))


def _topk_jaccard(left: list[ScoredRow], right: list[ScoredRow], k: int) -> float:
    left_ids = {item.row.candidate_video_id for item in left[:k]}
    right_ids = {item.row.candidate_video_id for item in right[:k]}
    union = left_ids | right_ids
    return 1.0 if not union else len(left_ids & right_ids) / len(union)


def _parse_grid_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--grid-values must contain at least one value")
    if any(value < 0 for value in values):
        raise ValueError("--grid-values must be non-negative")
    return values


def _variant_name_from_weights(prefix: str, weights: ComparisonScoringWeights) -> str:
    parts = [f"{key}={getattr(weights, key):.2f}" for key in WEIGHT_FIELDS]
    return prefix + "__" + "_".join(parts)


def _build_variants(
    *,
    grid_fields: tuple[str, ...],
    grid_values: tuple[float, ...],
) -> list[WeightVariant]:
    baseline = DEFAULT_COMPARISON_SCORING_WEIGHTS
    variants: list[WeightVariant] = [WeightVariant("baseline", baseline)]

    for field in WEIGHT_FIELDS:
        variants.append(WeightVariant(f"ablate_{field}", replace(baseline, **{field: 0.0})))
        variants.append(
            WeightVariant(
                f"double_{field}",
                replace(baseline, **{field: getattr(baseline, field) * 2.0}),
            )
        )

    for field in grid_fields:
        if field not in WEIGHT_FIELDS:
            raise ValueError(f"unknown grid field: {field}")

    for multipliers in itertools.product(grid_values, repeat=len(grid_fields)):
        updates = {
            field: getattr(baseline, field) * multiplier
            for field, multiplier in zip(grid_fields, multipliers, strict=True)
        }
        weights = replace(baseline, **updates)
        name = "grid__" + "_".join(
            f"{field}x{multiplier:.2f}"
            for field, multiplier in zip(grid_fields, multipliers, strict=True)
        )
        variants.append(WeightVariant(name, weights))

    deduped: dict[tuple[float, ...], WeightVariant] = {}
    for variant in variants:
        key = tuple(getattr(variant.weights, field) for field in WEIGHT_FIELDS)
        deduped.setdefault(key, variant)
    return list(deduped.values())


def _evaluate_variant(
    *,
    name: str,
    weights: ComparisonScoringWeights,
    grouped_rows: dict[tuple[str, str, str], list[EvalRow]],
    baseline_ranked: dict[tuple[str, str, str], list[ScoredRow]],
    top_k: int,
    relevance_threshold: int,
) -> dict[str, Any]:
    group_metrics: list[dict[str, Any]] = []
    country_stability: dict[str, list[float]] = defaultdict(list)

    for key, rows in grouped_rows.items():
        source_video_id, keyword, candidate_country = key
        ranked = _rank_rows(rows, weights)
        baseline = baseline_ranked[key]
        top3_ids = [item.row.candidate_video_id for item in ranked[:top_k]]
        jaccard = _topk_jaccard(baseline, ranked, top_k)
        country_stability[candidate_country].append(jaccard)
        group_metrics.append(
            {
                "source_video_id": source_video_id,
                "keyword": keyword,
                "candidate_country": candidate_country,
                "ndcg_at_3": _ndcg_at_k(ranked, top_k),
                "recall_at_3": _recall_at_k(ranked, top_k, relevance_threshold),
                "mrr": _mrr(ranked, relevance_threshold),
                "top3_relevant_count": _top_relevant_count(ranked, top_k, relevance_threshold),
                "top3_jaccard_vs_baseline": jaccard,
                "spearman_vs_baseline": _spearman_from_rank_maps(_rank_map(baseline), _rank_map(ranked)),
                "top3_video_ids": top3_ids,
                "top3_relevances": [item.row.expected_relevance for item in ranked[:top_k]],
            }
        )

    return {
        "variant": name,
        "weights": asdict(weights),
        "mean_ndcg_at_3": _safe_mean([m["ndcg_at_3"] for m in group_metrics]),
        "mean_recall_at_3": _safe_mean([m["recall_at_3"] for m in group_metrics]),
        "mean_mrr": _safe_mean([m["mrr"] for m in group_metrics]),
        "mean_top3_relevant_count": _safe_mean([m["top3_relevant_count"] for m in group_metrics]),
        "mean_top3_jaccard_vs_baseline": _safe_mean([m["top3_jaccard_vs_baseline"] for m in group_metrics]),
        "mean_spearman_vs_baseline": _safe_mean([m["spearman_vs_baseline"] for m in group_metrics]),
        "country_top3_stability": {
            country: _safe_mean(values)
            for country, values in sorted(country_stability.items())
        },
        "groups": group_metrics,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("no variant results")
    return {
        "total_variants": len(results),
        "ndcg_at_3": {
            "min": min(row["mean_ndcg_at_3"] for row in results),
            "median": _safe_median([row["mean_ndcg_at_3"] for row in results]),
            "max": max(row["mean_ndcg_at_3"] for row in results),
            "mean": _safe_mean([row["mean_ndcg_at_3"] for row in results]),
        },
        "recall_at_3": {
            "min": min(row["mean_recall_at_3"] for row in results),
            "median": _safe_median([row["mean_recall_at_3"] for row in results]),
            "max": max(row["mean_recall_at_3"] for row in results),
            "mean": _safe_mean([row["mean_recall_at_3"] for row in results]),
        },
        "spearman_vs_baseline": {
            "min": min(row["mean_spearman_vs_baseline"] for row in results),
            "median": _safe_median([row["mean_spearman_vs_baseline"] for row in results]),
            "max": max(row["mean_spearman_vs_baseline"] for row in results),
            "mean": _safe_mean([row["mean_spearman_vs_baseline"] for row in results]),
        },
        "top3_jaccard_vs_baseline": {
            "min": min(row["mean_top3_jaccard_vs_baseline"] for row in results),
            "median": _safe_median([row["mean_top3_jaccard_vs_baseline"] for row in results]),
            "max": max(row["mean_top3_jaccard_vs_baseline"] for row in results),
            "mean": _safe_mean([row["mean_top3_jaccard_vs_baseline"] for row in results]),
        },
    }


def _build_report_payload(
    *,
    rows: list[EvalRow],
    variants: list[WeightVariant],
    top_k: int,
    relevance_threshold: int,
) -> dict[str, Any]:
    grouped_rows = _group_rows(rows)
    baseline_ranked = {
        key: _rank_rows(group, DEFAULT_COMPARISON_SCORING_WEIGHTS)
        for key, group in grouped_rows.items()
    }
    results = [
        _evaluate_variant(
            name=variant.name,
            weights=variant.weights,
            grouped_rows=grouped_rows,
            baseline_ranked=baseline_ranked,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )
        for variant in variants
    ]
    ranked_stable = sorted(
        results,
        key=lambda row: (
            row["mean_ndcg_at_3"],
            row["mean_recall_at_3"],
            row["mean_top3_jaccard_vs_baseline"],
            row["mean_spearman_vs_baseline"],
        ),
        reverse=True,
    )
    ranked_weak = sorted(
        results,
        key=lambda row: (
            row["mean_ndcg_at_3"],
            row["mean_recall_at_3"],
            row["mean_top3_jaccard_vs_baseline"],
            row["mean_spearman_vs_baseline"],
        ),
    )
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_size": len(rows),
            "group_count": len(grouped_rows),
            "baseline_weights": asdict(DEFAULT_COMPARISON_SCORING_WEIGHTS),
            "top_k": top_k,
            "relevance_threshold": relevance_threshold,
        },
        "aggregate": _aggregate(results),
        "baseline": next(row for row in results if row["variant"] == "baseline"),
        "variants": results,
        "stable_top10": ranked_stable[:10],
        "weak_bottom10": ranked_weak[:10],
        "ablation_focus": {
            row["variant"]: row
            for row in results
            if row["variant"] in {"ablate_issue_overlap", "ablate_semantic_similarity", "ablate_view_score", "ablate_published_at"}
        },
    }


def _format_variant_row(row: dict[str, Any]) -> str:
    return (
        f"| `{row['variant']}` | {row['mean_ndcg_at_3']:.4f} | "
        f"{row['mean_recall_at_3']:.4f} | {row['mean_mrr']:.4f} | "
        f"{row['mean_top3_relevant_count']:.2f} | "
        f"{row['mean_top3_jaccard_vs_baseline']:.4f} | "
        f"{row['mean_spearman_vs_baseline']:.4f} |"
    )


def _build_markdown(payload: dict[str, Any], dataset_path: Path) -> str:
    meta = payload["meta"]
    aggregate = payload["aggregate"]
    baseline = payload["baseline"]
    lines: list[str] = []
    lines.append("# 비교 추천 가중치 민감도 분석 리포트")
    lines.append("")
    lines.append(f"- 생성시각(UTC): `{meta['generated_at']}`")
    lines.append(f"- 평가셋: `{dataset_path}`")
    lines.append(f"- 평가 row 수: `{meta['dataset_size']}`, 그룹 수: `{meta['group_count']}`")
    lines.append(f"- 관련 후보 기준: `expected_relevance >= {meta['relevance_threshold']}`")
    lines.append(f"- Baseline 가중치: `{meta['baseline_weights']}`")
    lines.append("")
    lines.append("## 1) 반박 질문 대응")
    lines.append("- Q. 이 가중치는 임의인가요?")
    lines.append("  - A. KG 신호(issue/entity)를 강하게 두고 텍스트/메타 신호를 보조로 둔 baseline을 수작업 라벨 평가셋에서 검증했습니다.")
    lines.append("- Q. 가중치를 조금 바꾸면 결과가 크게 바뀌나요?")
    lines.append("  - A. grid/ablation sweep에서 nDCG@3, Recall@3, Spearman, Top3 Jaccard를 함께 측정해 안정 구간과 취약 구간을 분리했습니다.")
    lines.append("- Q. 조회수나 최신성이 추천을 왜곡하지 않나요?")
    lines.append("  - A. view/published 가중치는 작게 두었고, ablation에서 해당 신호 제거 시 Top3 변화가 제한적인지 확인합니다.")
    lines.append("")
    lines.append("## 2) Baseline 결과")
    lines.append(
        f"- mean nDCG@3={baseline['mean_ndcg_at_3']:.4f}, "
        f"Recall@3={baseline['mean_recall_at_3']:.4f}, "
        f"MRR={baseline['mean_mrr']:.4f}, "
        f"Top3 relevant count={baseline['mean_top3_relevant_count']:.2f}"
    )
    lines.append("")
    lines.append("## 3) 전체 Sweep 요약")
    lines.append(
        "- nDCG@3: "
        f"min={aggregate['ndcg_at_3']['min']:.4f}, "
        f"median={aggregate['ndcg_at_3']['median']:.4f}, "
        f"max={aggregate['ndcg_at_3']['max']:.4f}"
    )
    lines.append(
        "- Recall@3: "
        f"min={aggregate['recall_at_3']['min']:.4f}, "
        f"median={aggregate['recall_at_3']['median']:.4f}, "
        f"max={aggregate['recall_at_3']['max']:.4f}"
    )
    lines.append(
        "- Baseline 대비 순위 안정성: "
        f"Spearman median={aggregate['spearman_vs_baseline']['median']:.4f}, "
        f"Top3 Jaccard median={aggregate['top3_jaccard_vs_baseline']['median']:.4f}"
    )
    lines.append("")
    lines.append("## 4) 안정 상위 조합")
    lines.append("| variant | nDCG@3 | Recall@3 | MRR | Top3 relevant | Top3 Jaccard | Spearman |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["stable_top10"]:
        lines.append(_format_variant_row(row))
    lines.append("")
    lines.append("## 5) 취약 하위 조합")
    lines.append("| variant | nDCG@3 | Recall@3 | MRR | Top3 relevant | Top3 Jaccard | Spearman |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["weak_bottom10"]:
        lines.append(_format_variant_row(row))
    lines.append("")
    lines.append("## 6) 해석")
    lines.append("- `ablate_issue_overlap`에서 품질이 하락하면 같은 이슈 노드를 강하게 둔 이유를 설명할 수 있습니다.")
    lines.append("- `ablate_view_score`, `ablate_published_at` 변화가 작으면 조회수/최신성은 보조 신호라는 방어 논리가 됩니다.")
    lines.append("- hard negative가 취약 조합에서 Top3에 진입하면, 키워드/semantic만으로는 부족하고 KG 신호가 필요하다는 근거입니다.")
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any], markdown: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_weight_sensitivity_report.json"
    md_path = output_dir / "comparison_weight_sensitivity_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="비교 추천 가중치 민감도 리포트를 생성합니다.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-fields", nargs="+", default=["issue_overlap", "shared_keyword", "semantic_similarity"])
    parser.add_argument("--grid-values", default="0,0.5,1,1.5,2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--relevance-threshold", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _load_eval_rows(args.dataset_path)
    variants = _build_variants(
        grid_fields=tuple(args.grid_fields),
        grid_values=_parse_grid_values(args.grid_values),
    )
    payload = _build_report_payload(
        rows=rows,
        variants=variants,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )
    markdown = _build_markdown(payload, args.dataset_path)
    json_path, md_path = _write_outputs(payload, markdown, args.output_dir)

    baseline = payload["baseline"]
    print("[comparison-weight-sensitivity] baseline")
    print(f"- nDCG@3={baseline['mean_ndcg_at_3']:.4f}")
    print(f"- Recall@3={baseline['mean_recall_at_3']:.4f}")
    print(f"- MRR={baseline['mean_mrr']:.4f}")
    print(f"[comparison-weight-sensitivity] wrote json: {json_path}")
    print(f"[comparison-weight-sensitivity] wrote md:   {md_path}")


if __name__ == "__main__":
    main()
