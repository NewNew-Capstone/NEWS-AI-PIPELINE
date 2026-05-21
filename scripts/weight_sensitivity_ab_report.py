"""가중치 민감도 분석 리포트 생성 스크립트.

목적:
- A/B 극단 테스트의 방향성(A>B) 유지율을 정량화
- 기준 가중치(기본 0.4/0.3/0.3) 대비 영상별 순위 안정성(Spearman) 측정
- 교수님 피드백 대응용 상세 로그/리포트(JSON, Markdown) 생성

사용 예시:
    .venv/bin/python scripts/weight_sensitivity_ab_report.py \
      --snapshot-path analysis_bc/data/eval/ab_pipeline_snapshot.json \
      --grid-divisions 20

    .venv/bin/python scripts/weight_sensitivity_ab_report.py \
      --mode full --grid-divisions 20
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tune_bias_weights_ab import (  # noqa: E402
    SNAPSHOT_PATH,
    _build_snapshot,
    _load_snapshot,
    _score_record,
    _variants,
)


@dataclass(frozen=True)
class WeightPoint:
    variant: str
    w_opinion: float
    w_emotion: float
    w_fact: float
    a_mean: float
    b_mean: float
    a_min: float
    b_max: float
    margin: float
    direction_ok: bool
    direction_strict_ok: bool
    spearman_all: float
    spearman_a: float
    spearman_b: float
    top5_jaccard: float
    top10_jaccard: float


def _rank_map(scores: dict[int, float]) -> dict[int, int]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {target_id: idx + 1 for idx, (target_id, _score) in enumerate(ordered)}


def _spearman_from_ranks(base_rank: dict[int, int], other_rank: dict[int, int]) -> float:
    common = sorted(set(base_rank.keys()) & set(other_rank.keys()))
    n = len(common)
    if n < 2:
        return 1.0

    d2_sum = 0.0
    for key in common:
        d = base_rank[key] - other_rank[key]
        d2_sum += d * d

    denom = n * (n * n - 1)
    if denom == 0:
        return 1.0
    return 1.0 - (6.0 * d2_sum / denom)


def _topk_jaccard(base_scores: dict[int, float], other_scores: dict[int, float], k: int) -> float:
    if k <= 0:
        return 1.0

    base_top = {
        target_id
        for target_id, _score in sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    }
    other_top = {
        target_id
        for target_id, _score in sorted(other_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    }
    union = base_top | other_top
    if not union:
        return 1.0
    return len(base_top & other_top) / len(union)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _safe_min(values: list[float]) -> float:
    return min(values) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _parse_variant_weights(name: str) -> tuple[float, float, float]:
    # grid_0.40_0.30_0.30 형식 우선 파싱
    if name.startswith("grid_"):
        try:
            _prefix, wo, we, wf = name.split("_")
            return float(wo), float(we), float(wf)
        except (ValueError, TypeError):
            pass

    # default_40_30_30 등 대표 variant 지원
    if "40_30_30" in name:
        return 0.4, 0.3, 0.3
    if "50_25_25" in name:
        return 0.5, 0.25, 0.25
    if "35_45_20" in name:
        return 0.35, 0.45, 0.2
    if "35_25_40" in name:
        return 0.35, 0.25, 0.4

    # 알 수 없으면 NaN으로 표기
    return math.nan, math.nan, math.nan


def _collect_scores_by_variant(
    results: list[Any],
) -> dict[str, dict[str, dict[int, float]]]:
    grouped: dict[str, dict[str, dict[int, float]]] = {}
    for row in results:
        grouped.setdefault(row.variant, {"ALL": {}, "A": {}, "B": {}})
        grouped[row.variant]["ALL"][row.target_id] = row.overall
        grouped[row.variant][row.dataset][row.target_id] = row.overall
    return grouped


def _summarize(
    grouped_scores: dict[str, dict[str, dict[int, float]]],
    baseline_variant: str,
) -> list[WeightPoint]:
    if baseline_variant not in grouped_scores:
        raise ValueError(f"baseline variant not found: {baseline_variant}")

    baseline_all = grouped_scores[baseline_variant]["ALL"]
    baseline_a = grouped_scores[baseline_variant]["A"]
    baseline_b = grouped_scores[baseline_variant]["B"]

    baseline_all_rank = _rank_map(baseline_all)
    baseline_a_rank = _rank_map(baseline_a)
    baseline_b_rank = _rank_map(baseline_b)

    rows: list[WeightPoint] = []
    for variant, score_sets in grouped_scores.items():
        a_scores = list(score_sets["A"].values())
        b_scores = list(score_sets["B"].values())

        a_mean = _safe_mean(a_scores)
        b_mean = _safe_mean(b_scores)
        a_min = _safe_min(a_scores)
        b_max = _safe_max(b_scores)
        margin = a_min - b_max

        all_rank = _rank_map(score_sets["ALL"])
        a_rank = _rank_map(score_sets["A"])
        b_rank = _rank_map(score_sets["B"])

        wo, we, wf = _parse_variant_weights(variant)
        rows.append(
            WeightPoint(
                variant=variant,
                w_opinion=wo,
                w_emotion=we,
                w_fact=wf,
                a_mean=a_mean,
                b_mean=b_mean,
                a_min=a_min,
                b_max=b_max,
                margin=margin,
                direction_ok=a_mean > b_mean,
                direction_strict_ok=margin > 0,
                spearman_all=_spearman_from_ranks(baseline_all_rank, all_rank),
                spearman_a=_spearman_from_ranks(baseline_a_rank, a_rank),
                spearman_b=_spearman_from_ranks(baseline_b_rank, b_rank),
                top5_jaccard=_topk_jaccard(baseline_all, score_sets["ALL"], 5),
                top10_jaccard=_topk_jaccard(baseline_all, score_sets["ALL"], 10),
            )
        )

    rows.sort(key=lambda r: (r.w_opinion, r.w_emotion, r.w_fact, r.variant))
    return rows


def _aggregate_metrics(points: list[WeightPoint], baseline_variant: str) -> dict[str, Any]:
    total = len(points)
    if total == 0:
        raise ValueError("no points to summarize")

    direction_ok = [p for p in points if p.direction_ok]
    strict_ok = [p for p in points if p.direction_strict_ok]

    spearman_all = [p.spearman_all for p in points]
    spearman_a = [p.spearman_a for p in points]
    spearman_b = [p.spearman_b for p in points]

    top5 = [p.top5_jaccard for p in points]
    top10 = [p.top10_jaccard for p in points]

    baseline = next((p for p in points if p.variant == baseline_variant), None)

    return {
        "total_variants": total,
        "direction_ok_count": len(direction_ok),
        "direction_ok_ratio": len(direction_ok) / total,
        "direction_strict_ok_count": len(strict_ok),
        "direction_strict_ok_ratio": len(strict_ok) / total,
        "spearman_all": {
            "min": _safe_min(spearman_all),
            "median": _safe_median(spearman_all),
            "max": _safe_max(spearman_all),
            "mean": _safe_mean(spearman_all),
        },
        "spearman_a": {
            "min": _safe_min(spearman_a),
            "median": _safe_median(spearman_a),
            "max": _safe_max(spearman_a),
            "mean": _safe_mean(spearman_a),
        },
        "spearman_b": {
            "min": _safe_min(spearman_b),
            "median": _safe_median(spearman_b),
            "max": _safe_max(spearman_b),
            "mean": _safe_mean(spearman_b),
        },
        "top5_jaccard": {
            "min": _safe_min(top5),
            "median": _safe_median(top5),
            "max": _safe_max(top5),
            "mean": _safe_mean(top5),
        },
        "top10_jaccard": {
            "min": _safe_min(top10),
            "median": _safe_median(top10),
            "max": _safe_max(top10),
            "mean": _safe_mean(top10),
        },
        "baseline": asdict(baseline) if baseline else None,
    }


def _build_baseline_detail(
    grouped_scores: dict[str, dict[str, dict[int, float]]],
    baseline_variant: str,
) -> dict[str, Any]:
    if baseline_variant not in grouped_scores:
        raise ValueError(f"baseline variant not found: {baseline_variant}")

    a_scores = grouped_scores[baseline_variant]["A"]
    b_scores = grouped_scores[baseline_variant]["B"]

    a_rows = [
        {"target_id": target_id, "score": score}
        for target_id, score in sorted(a_scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    b_rows = [
        {"target_id": target_id, "score": score}
        for target_id, score in sorted(b_scores.items(), key=lambda item: (-item[1], item[0]))
    ]

    a_min = _safe_min(list(a_scores.values()))
    b_max = _safe_max(list(b_scores.values()))
    margin = a_min - b_max

    return {
        "variant": baseline_variant,
        "a_scores": a_rows,
        "b_scores": b_rows,
        "a_min": a_min,
        "b_max": b_max,
        "margin": margin,
        "summary_line": f"현재 baseline 분리 강도는 margin={margin:.4f} (A_min={a_min:.4f}, B_max={b_max:.4f}) 입니다.",
    }


def _build_report_markdown(
    *,
    generated_at: str,
    mode: str,
    grid_divisions: int,
    baseline_variant: str,
    baseline_detail: dict[str, Any],
    aggregate: dict[str, Any],
    top_rows: list[WeightPoint],
    worst_rows: list[WeightPoint],
) -> str:
    lines: list[str] = []
    lines.append("# 가중치 민감도 분석 리포트")
    lines.append("")
    lines.append(f"- 생성시각(UTC): {generated_at}")
    lines.append(f"- 실행모드: `{mode}`")
    lines.append(f"- 그리드 해상도: `{grid_divisions}` (step={1/grid_divisions:.4f})")
    lines.append(f"- 기준 가중치 variant: `{baseline_variant}`")
    lines.append("- 데이터셋: 극단 질의 세트 `analysis_bc/data/A.json`, `analysis_bc/data/B.json`")
    lines.append("")

    lines.append("## 1) 교수님 피드백 대응 질문")
    lines.append("- Q1. 가중치를 바꿔도 A>B 방향성이 유지되는가?")
    lines.append("- Q2. 가중치를 바꿔도 영상별 순위가 안정적인가?")
    lines.append("")

    lines.append("## 2) Baseline 결과 (0.4/0.3/0.3)")
    lines.append(f"- {baseline_detail['summary_line']}")
    lines.append("")
    lines.append("### A 점수표 (내림차순)")
    lines.append("| target_id | score |")
    lines.append("|---:|---:|")
    for row in baseline_detail["a_scores"]:
        lines.append(f"| {row['target_id']} | {row['score']:.4f} |")
    lines.append("")
    lines.append("### B 점수표 (내림차순)")
    lines.append("| target_id | score |")
    lines.append("|---:|---:|")
    for row in baseline_detail["b_scores"]:
        lines.append(f"| {row['target_id']} | {row['score']:.4f} |")
    lines.append("")

    lines.append("## 3) 스윕 핵심 요약")
    lines.append(
        f"- 방향성(A_mean > B_mean) 유지율: {aggregate['direction_ok_count']}/{aggregate['total_variants']} "
        f"({aggregate['direction_ok_ratio']:.2%})"
    )
    lines.append(
        f"- 엄격 분리(A_min > B_max) 유지율: {aggregate['direction_strict_ok_count']}/{aggregate['total_variants']} "
        f"({aggregate['direction_strict_ok_ratio']:.2%})"
    )
    lines.append(
        "- 순위 상관(Spearman, 전체): "
        f"min={aggregate['spearman_all']['min']:.4f}, "
        f"median={aggregate['spearman_all']['median']:.4f}, "
        f"max={aggregate['spearman_all']['max']:.4f}"
    )
    lines.append(
        "- Top-K 안정성(Jaccard): "
        f"Top5 median={aggregate['top5_jaccard']['median']:.4f}, "
        f"Top10 median={aggregate['top10_jaccard']['median']:.4f}"
    )
    lines.append("")

    lines.append("## 4) 상관 안정 상위 조합 (Top 10)")
    lines.append("| variant | (wo,we,wf) | direction | strict | spearman_all | top5 | top10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in top_rows:
        lines.append(
            f"| {row.variant} | ({row.w_opinion:.2f},{row.w_emotion:.2f},{row.w_fact:.2f}) | "
            f"{int(row.direction_ok)} | {int(row.direction_strict_ok)} | {row.spearman_all:.4f} | "
            f"{row.top5_jaccard:.4f} | {row.top10_jaccard:.4f} |"
        )
    lines.append("")

    lines.append("## 5) 상관 취약 조합 (Bottom 10)")
    lines.append("| variant | (wo,we,wf) | direction | strict | spearman_all | top5 | top10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in worst_rows:
        lines.append(
            f"| {row.variant} | ({row.w_opinion:.2f},{row.w_emotion:.2f},{row.w_fact:.2f}) | "
            f"{int(row.direction_ok)} | {int(row.direction_strict_ok)} | {row.spearman_all:.4f} | "
            f"{row.top5_jaccard:.4f} | {row.top10_jaccard:.4f} |"
        )
    lines.append("")

    lines.append("## 6) 해석 가이드")
    lines.append("- 기준 조합(0.4/0.3/0.3)이 상위 안정 구간에 포함되고, 방향성 유지율이 높으면 '0.5가 아니라 0.4'에 대한 정량 근거가 됩니다.")
    lines.append("- 일부 조합에서 상관이 낮다면, 해당 구간(예: opinion 과대)에서 순위가 흔들리는 한계를 같이 기술합니다.")

    return "\n".join(lines) + "\n"


def _print_console_summary(aggregate: dict[str, Any]) -> None:
    print("\n[weight-sensitivity] summary")
    print("-" * 80)
    print(
        "direction_ok="
        f"{aggregate['direction_ok_count']}/{aggregate['total_variants']} "
        f"({aggregate['direction_ok_ratio']:.2%})"
    )
    print(
        "direction_strict_ok="
        f"{aggregate['direction_strict_ok_count']}/{aggregate['total_variants']} "
        f"({aggregate['direction_strict_ok_ratio']:.2%})"
    )
    print(
        "spearman_all(min/median/max)="
        f"{aggregate['spearman_all']['min']:.4f}/"
        f"{aggregate['spearman_all']['median']:.4f}/"
        f"{aggregate['spearman_all']['max']:.4f}"
    )
    print(
        "top5_jaccard(min/median/max)="
        f"{aggregate['top5_jaccard']['min']:.4f}/"
        f"{aggregate['top5_jaccard']['median']:.4f}/"
        f"{aggregate['top5_jaccard']['max']:.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B 가중치 민감도 리포트를 생성합니다.",
    )
    parser.add_argument(
        "--mode",
        choices=("snapshot", "full"),
        default="snapshot",
        help="full은 snapshot 재생성, snapshot은 기존 snapshot 사용",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=SNAPSHOT_PATH,
        help="입출력 snapshot 경로",
    )
    parser.add_argument(
        "--grid-divisions",
        type=int,
        default=20,
        help="그리드 해상도. 20이면 0.05 단위",
    )
    parser.add_argument(
        "--baseline-variant",
        default="grid_0.40_0.30_0.30",
        help="순위 상관 비교 기준 variant 이름",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_bc/data/eval/reports"),
        help="리포트 출력 디렉터리",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="리포트에 표시할 상/하위 조합 개수",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.grid_divisions <= 0:
        raise ValueError("--grid-divisions must be positive")

    if args.mode == "full":
        snapshot = _build_snapshot(snapshot_path=args.snapshot_path)
    else:
        snapshot = _load_snapshot(args.snapshot_path)

    # tune script의 variant 생성 로직 재사용
    variant_args = argparse.Namespace(grid=True, grid_divisions=args.grid_divisions)
    variants = _variants(variant_args)

    results = [
        _score_record(record, variant)
        for variant in variants
        for record in snapshot["records"]
    ]
    grouped = _collect_scores_by_variant(results)

    if args.baseline_variant not in grouped:
        raise ValueError(
            f"baseline variant '{args.baseline_variant}' not found. "
            "tip: adjust --grid-divisions or --baseline-variant"
        )

    points = _summarize(grouped, args.baseline_variant)
    baseline_detail = _build_baseline_detail(grouped, args.baseline_variant)
    aggregate = _aggregate_metrics(points, args.baseline_variant)

    ranked = sorted(points, key=lambda p: (p.spearman_all, p.top10_jaccard, p.top5_jaccard), reverse=True)
    top_rows = ranked[: args.top_n]
    worst_rows = ranked[-args.top_n:]

    generated_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "meta": {
            "generated_at": generated_at,
            "mode": args.mode,
            "snapshot_path": str(args.snapshot_path),
            "grid_divisions": args.grid_divisions,
            "baseline_variant": args.baseline_variant,
            "dataset_a_path": "analysis_bc/data/A.json",
            "dataset_b_path": "analysis_bc/data/B.json",
        },
        "baseline_detail": baseline_detail,
        "aggregate": aggregate,
        "points": [asdict(p) for p in points],
        "top_rows": [asdict(p) for p in top_rows],
        "worst_rows": [asdict(p) for p in worst_rows],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "weight_sensitivity_report.json"
    md_path = args.output_dir / "weight_sensitivity_report.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = _build_report_markdown(
        generated_at=generated_at,
        mode=args.mode,
        grid_divisions=args.grid_divisions,
        baseline_variant=args.baseline_variant,
        baseline_detail=baseline_detail,
        aggregate=aggregate,
        top_rows=top_rows,
        worst_rows=worst_rows,
    )
    md_path.write_text(md, encoding="utf-8")

    _print_console_summary(aggregate)
    print(f"[weight-sensitivity] wrote json: {json_path}")
    print(f"[weight-sensitivity] wrote md:   {md_path}")


if __name__ == "__main__":
    main()
