from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_emotion_ab import (
    EvalRow,
    _check_release_gate,
    _confirm_patterns,
    _normalize_template,
    build_error_cases,
    dedupe_rows_by_text,
)


def test_build_error_cases_classifies_fp_and_fn() -> None:
    rows = [
        EvalRow(content_sentence_id=1, sentence_text="중립 문장", gold_emotion=False),
        EvalRow(content_sentence_id=2, sentence_text="감정 문장", gold_emotion=True),
    ]
    labels = [
        SimpleNamespace(
            content_sentence_id=1,
            label_type="EMOTIONALLY_LOADED",
            matched_word="사고",
        ),
    ]

    cases = build_error_cases(rows, labels)
    assert len(cases) == 2

    fp = next(c for c in cases if c["error_type"] == "FP")
    fn = next(c for c in cases if c["error_type"] == "FN")

    assert fp["content_sentence_id"] == 1
    assert fp["pred_emotion"] is True
    assert fp["matched_words"] == ["사고"]

    assert fn["content_sentence_id"] == 2
    assert fn["pred_emotion"] is False
    assert fn["matched_words"] == []


def test_dedupe_rows_by_text_is_consistent() -> None:
    rows = [
        EvalRow(content_sentence_id=1, sentence_text="반복 문장", gold_emotion=False),
        EvalRow(content_sentence_id=2, sentence_text="반복 문장", gold_emotion=False),
        EvalRow(content_sentence_id=3, sentence_text="다른 문장", gold_emotion=True),
    ]

    deduped = dedupe_rows_by_text(rows)
    assert len(deduped) == 2
    assert {r.sentence_text for r in deduped} == {"반복 문장", "다른 문장"}


def test_confirm_patterns_balanced_threshold_boundary() -> None:
    # FP: 지원수 4건 -> 미확정
    fp_low = [
        {
            "content_sentence_id": i,
            "sentence_text": f"경찰은 사고 경위를 조사했다 {i}",
            "gold_emotion": False,
            "pred_emotion": True,
            "error_type": "FP",
            "matched_words": ["사고"],
        }
        for i in range(1, 5)
    ]
    # FP: 지원수 5건, 템플릿 2종 -> 확정
    fp_ok = [
        {
            "content_sentence_id": 100 + i,
            "sentence_text": (
                f"경찰은 사고 경위를 조사했다 {i}"
                if i % 2
                else f"당국은 사고 원인을 조사했다 {i}"
            ),
            "gold_emotion": False,
            "pred_emotion": True,
            "error_type": "FP",
            "matched_words": ["사고"],
        }
        for i in range(1, 6)
    ]

    report_low = _confirm_patterns(fp_low)
    report_ok = _confirm_patterns(fp_ok)

    assert "candidate_fp_patterns" in report_low
    assert "candidate_fn_patterns" in report_low
    assert "blacklist_candidates" in report_low

    assert "사고" not in report_low["confirmed_fp_patterns"]
    assert "사고" in report_low["candidate_fp_patterns"]
    assert report_low["candidate_fp_patterns"]["사고"]["status"] == "candidate"
    assert "사고" in report_ok["confirmed_fp_patterns"]
    confirmed = report_ok["confirmed_fp_patterns"]["사고"]
    assert confirmed["support_unique_text"] == 5
    assert confirmed["template_count"] >= 2
    assert confirmed["impact"]["delta_fp"] == 5
    assert confirmed["status"] == "confirmed"
    assert report_ok["blacklist_candidates"][0]["word_root"] == "사고"


def test_normalize_template_numbers_are_merged() -> None:
    t1 = _normalize_template("사고 경위를 조사했다 100")
    t2 = _normalize_template("사고 경위를 조사했다 200")
    assert t1 == t2


def test_release_gate_passes_with_better_or_equal_metrics() -> None:
    baseline = {
        "stages": {
            "raw": {
                "improved": {
                    "fp": 10,
                    "fn": 20,
                    "precision": 0.89,
                    "recall": 0.80,
                    "f1": 0.84,
                }
            }
        }
    }
    current = {
        "stages": {
            "raw": {
                "improved": {
                    "fp": 6,
                    "fn": 19,
                    "precision": 0.92,
                    "recall": 0.79,
                    "f1": 0.85,
                }
            }
        }
    }
    gate = _check_release_gate(baseline, current)
    assert gate["passed"] is True


def test_release_gate_fails_when_precision_drop_too_large() -> None:
    baseline = {
        "stages": {
            "raw": {
                "improved": {
                    "fp": 10,
                    "fn": 20,
                    "precision": 0.95,
                    "recall": 0.85,
                    "f1": 0.89,
                }
            }
        }
    }
    current = {
        "stages": {
            "raw": {
                "improved": {
                    "fp": 8,
                    "fn": 20,
                    "precision": 0.90,
                    "recall": 0.84,
                    "f1": 0.88,
                }
            }
        }
    }
    gate = _check_release_gate(baseline, current)
    assert gate["passed"] is False
    assert gate["checks"]["precision_drop_lte_0_02"] is False
