from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from knowledge_graph_bc.comparison_scoring import DEFAULT_COMPARISON_SCORING_WEIGHTS, ComparisonScoringWeights
from scripts.comparison_weight_sensitivity_report import (
    DEFAULT_DATASET_PATH,
    _build_markdown,
    _build_report_payload,
    _build_variants,
    _group_rows,
    _load_eval_rows,
    _parse_grid_values,
    _rank_rows,
    _write_outputs,
)


def test_baseline_ranks_strong_labels_above_hard_negatives() -> None:
    rows = _load_eval_rows(DEFAULT_DATASET_PATH)
    grouped = _group_rows(rows)

    for group_rows in grouped.values():
        ranked = _rank_rows(group_rows, DEFAULT_COMPARISON_SCORING_WEIGHTS)
        top3_relevance = [item.row.expected_relevance for item in ranked[:3]]
        assert top3_relevance == [3, 3, 3]
        assert all("hard_negative" not in item.row.reason_tags for item in ranked[:3])


def test_issue_overlap_ablation_detects_hard_negative_rise() -> None:
    rows = _load_eval_rows(DEFAULT_DATASET_PATH)
    group_rows = [
        row
        for row in rows
        if row.source_video_id == "q0Jo5F8pHbs" and row.candidate_country == "US"
    ]
    ablated = ComparisonScoringWeights(
        issue_overlap=0.0,
        shared_keyword=DEFAULT_COMPARISON_SCORING_WEIGHTS.shared_keyword,
        shared_entity=DEFAULT_COMPARISON_SCORING_WEIGHTS.shared_entity,
        semantic_similarity=DEFAULT_COMPARISON_SCORING_WEIGHTS.semantic_similarity,
        analysis_success=DEFAULT_COMPARISON_SCORING_WEIGHTS.analysis_success,
        view_score=DEFAULT_COMPARISON_SCORING_WEIGHTS.view_score,
        published_at=DEFAULT_COMPARISON_SCORING_WEIGHTS.published_at,
    )

    baseline_top3 = _rank_rows(group_rows, DEFAULT_COMPARISON_SCORING_WEIGHTS)[:3]
    ablated_top3 = _rank_rows(group_rows, ablated)[:3]

    assert all("hard_negative" not in item.row.reason_tags for item in baseline_top3)
    assert any("hard_negative" in item.row.reason_tags for item in ablated_top3)


def test_semantic_overweight_is_reported_as_less_stable() -> None:
    rows = _load_eval_rows(DEFAULT_DATASET_PATH)
    variants = [
        *[
            variant
            for variant in _build_variants(
                grid_fields=("semantic_similarity",),
                grid_values=(1.0, 4.0),
            )
            if variant.name in {"baseline", "grid__semantic_similarityx4.00"}
        ]
    ]
    payload = _build_report_payload(rows=rows, variants=variants, top_k=3, relevance_threshold=3)
    baseline = next(row for row in payload["variants"] if row["variant"] == "baseline")
    semantic_heavy = next(row for row in payload["variants"] if row["variant"] == "grid__semantic_similarityx4.00")

    assert semantic_heavy["mean_top3_jaccard_vs_baseline"] <= baseline["mean_top3_jaccard_vs_baseline"]
    assert semantic_heavy["mean_spearman_vs_baseline"] < baseline["mean_spearman_vs_baseline"]


def test_report_output_contains_required_fields(tmp_path: Path) -> None:
    rows = _load_eval_rows(DEFAULT_DATASET_PATH)
    variants = _build_variants(
        grid_fields=("issue_overlap", "semantic_similarity"),
        grid_values=_parse_grid_values("0,1"),
    )
    payload = _build_report_payload(rows=rows, variants=variants, top_k=3, relevance_threshold=3)
    markdown = _build_markdown(payload, DEFAULT_DATASET_PATH)
    json_path, md_path = _write_outputs(payload, markdown, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["meta"]["dataset_size"] == len(rows)
    assert "baseline" in saved
    assert "aggregate" in saved
    assert "stable_top10" in saved
    assert "weak_bottom10" in saved
    assert "ablation_focus" in saved
    assert "비교 추천 가중치 민감도 분석 리포트" in md_path.read_text(encoding="utf-8")
