from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.weight_sensitivity_ab_report import (
    _aggregate_metrics,
    _build_baseline_detail,
    _rank_map,
    _spearman_from_ranks,
    _summarize,
    _topk_jaccard,
)


def test_rank_map_orders_by_score_desc_then_id() -> None:
    scores = {3: 0.7, 1: 0.9, 2: 0.9}
    rank = _rank_map(scores)
    assert rank == {1: 1, 2: 2, 3: 3}


def test_spearman_perfect_and_reversed() -> None:
    base = {1: 1, 2: 2, 3: 3, 4: 4}
    same = {1: 1, 2: 2, 3: 3, 4: 4}
    rev = {1: 4, 2: 3, 3: 2, 4: 1}

    assert _spearman_from_ranks(base, same) == 1.0
    assert _spearman_from_ranks(base, rev) == -1.0


def test_spearman_tie_case_stable() -> None:
    # tie는 _rank_map에서 target_id 오름차순으로 deterministic 처리한다.
    base_scores = {1: 0.9, 2: 0.9, 3: 0.6, 4: 0.6}
    other_scores = {2: 0.9, 1: 0.9, 4: 0.6, 3: 0.6}
    base_rank = _rank_map(base_scores)
    other_rank = _rank_map(other_scores)
    assert _spearman_from_ranks(base_rank, other_rank) == 1.0


def test_topk_jaccard_basic() -> None:
    base = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
    other = {1: 0.95, 4: 0.85, 2: 0.5, 3: 0.4}
    # base top2={1,2}, other top2={1,4} => intersection=1, union=3
    assert _topk_jaccard(base, other, 2) == (1 / 3)


def test_summarize_and_aggregate_direction_ratio() -> None:
    grouped = {
        "grid_0.40_0.30_0.30": {
            "ALL": {101: 0.9, 102: 0.8, 201: 0.3, 202: 0.2},
            "A": {101: 0.9, 102: 0.8},
            "B": {201: 0.3, 202: 0.2},
        },
        "grid_0.50_0.25_0.25": {
            "ALL": {101: 0.88, 102: 0.81, 201: 0.32, 202: 0.25},
            "A": {101: 0.88, 102: 0.81},
            "B": {201: 0.32, 202: 0.25},
        },
        "grid_0.20_0.40_0.40": {
            "ALL": {101: 0.55, 102: 0.52, 201: 0.58, 202: 0.56},
            "A": {101: 0.55, 102: 0.52},
            "B": {201: 0.58, 202: 0.56},
        },
    }

    points = _summarize(grouped, "grid_0.40_0.30_0.30")
    assert len(points) == 3

    agg = _aggregate_metrics(points, "grid_0.40_0.30_0.30")
    assert agg["total_variants"] == 3
    # 첫 2개 variant만 direction_ok
    assert agg["direction_ok_count"] == 2
    assert agg["direction_ok_ratio"] == (2 / 3)
    assert agg["baseline"] is not None
    assert agg["direction_strict_ok_count"] == 2
    assert agg["direction_strict_ok_ratio"] == (2 / 3)


def test_build_baseline_detail_includes_required_fields() -> None:
    grouped = {
        "grid_0.40_0.30_0.30": {
            "ALL": {101: 0.8, 102: 0.7, 201: 0.4, 202: 0.2},
            "A": {101: 0.8, 102: 0.7},
            "B": {201: 0.4, 202: 0.2},
        }
    }
    detail = _build_baseline_detail(grouped, "grid_0.40_0.30_0.30")
    assert detail["variant"] == "grid_0.40_0.30_0.30"
    assert detail["a_min"] == 0.7
    assert detail["b_max"] == 0.4
    assert detail["margin"] == pytest.approx(0.3)
    assert "summary_line" in detail
    assert len(detail["a_scores"]) == 2
    assert len(detail["b_scores"]) == 2
