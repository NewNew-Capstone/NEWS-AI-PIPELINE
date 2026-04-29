"""Emotion span 태깅 A/B 평가 스크립트 (실데이터 라벨셋 기반).

사용 예시:
    .venv/bin/python scripts/eval_emotion_ab.py \
      --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl

그리드서치:
    .venv/bin/python scripts/eval_emotion_ab.py \
      --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl \
      --run-grid
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from kiwipiepy import Kiwi

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
import analysis_bc.tagger.span_tagger as span_mod
from analysis_bc.tagger.span_tagger import SpanTagger


@dataclass(frozen=True)
class EvalRow:
    content_sentence_id: int
    sentence_text: str
    gold_emotion: bool


@dataclass(frozen=True)
class Variant:
    name: str
    threshold: float
    top_k: int
    margin: float


_DEFAULT_ERROR_PATH = Path("analysis_bc/data/eval/reports/error_cases.jsonl")
_DEFAULT_PATTERN_REPORT_PATH = Path("analysis_bc/data/eval/reports/pattern_report.json")
_DEFAULT_ERROR_RAW_PATH = Path("analysis_bc/data/eval/reports/error_cases_raw.jsonl")
_DEFAULT_ERROR_DEDUPED_PATH = Path("analysis_bc/data/eval/reports/error_cases_deduped.jsonl")
_DEFAULT_PATTERN_RAW_PATH = Path("analysis_bc/data/eval/reports/pattern_report_raw.json")
_DEFAULT_PATTERN_DEDUPED_PATH = Path("analysis_bc/data/eval/reports/pattern_report_deduped.json")
_DEFAULT_SUMMARY_PATH = Path("analysis_bc/data/eval/reports/eval_summary.json")
_DEFAULT_HISTORY_PATH = Path("analysis_bc/data/eval/reports/metrics_history.jsonl")
_FN_VALID_POS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}
_CANDIDATE_TOP_N = 20
_CONFIRM_SUPPORT_MIN = 5
_CONFIRM_TEMPLATE_MIN = 2


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"bool 변환 실패: {value!r}")


def load_dataset(path: Path) -> list[EvalRow]:
    if not path.exists():
        raise FileNotFoundError(f"dataset 파일이 없습니다: {path}")

    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            text = str(data["sentence_text"]).strip()
            if not text:
                raise ValueError(f"{path}:{i} sentence_text 비어 있음")
            row = EvalRow(
                content_sentence_id=int(data.get("content_sentence_id", len(rows) + 1)),
                sentence_text=text,
                gold_emotion=_parse_bool(data["gold_emotion"]),
            )
            rows.append(row)
    if not rows:
        raise ValueError("dataset 이 비어 있습니다.")
    return rows


def dedupe_rows_by_text(rows: list[EvalRow]) -> list[EvalRow]:
    deduped: list[EvalRow] = []
    seen: set[str] = set()
    for row in rows:
        if row.sentence_text in seen:
            continue
        seen.add(row.sentence_text)
        deduped.append(row)
    return deduped


def _to_classified(rows: list[EvalRow]) -> list[ClassifiedSentenceDto]:
    return [
        ClassifiedSentenceDto(
            content_sentence_id=r.content_sentence_id,
            sentence_text=r.sentence_text,
            sentence_order=idx + 1,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=0.99,
        )
        for idx, r in enumerate(rows)
    ]


def _set_variant(variant: Variant) -> tuple[float, int, float]:
    old = (
        span_mod.EMOTION_SIMILARITY_THRESHOLD,
        span_mod.EMOTION_TOP_K,
        span_mod.EMOTION_SCORE_MARGIN,
    )
    span_mod.EMOTION_SIMILARITY_THRESHOLD = variant.threshold
    span_mod.EMOTION_TOP_K = variant.top_k
    span_mod.EMOTION_SCORE_MARGIN = variant.margin
    return old


def _restore_variant(old: tuple[float, int, float]) -> None:
    span_mod.EMOTION_SIMILARITY_THRESHOLD = old[0]
    span_mod.EMOTION_TOP_K = old[1]
    span_mod.EMOTION_SCORE_MARGIN = old[2]


def _evaluate_with_labels(rows: list[EvalRow], variant: Variant) -> tuple[dict, list]:
    old = _set_variant(variant)
    try:
        # 과거 구현(anonymous용 SBERT 포함)과 현재 구현(FastText-only) 모두 호환.
        if hasattr(span_mod, "SentenceTransformer"):
            with patch("analysis_bc.tagger.span_tagger.SentenceTransformer", return_value=MagicMock()):
                tagger = SpanTagger()
        else:
            tagger = SpanTagger()
        if not tagger._is_qdrant_healthy():
            raise RuntimeError("Qdrant 연결 실패: emotion 평가를 실행할 수 없습니다.")
        labels = tagger.tag(_to_classified(rows))
    finally:
        _restore_variant(old)

    pred_ids = {
        l.content_sentence_id
        for l in labels
        if l.label_type == SentenceLabelType.EMOTIONALLY_LOADED
    }
    gold_ids = {r.content_sentence_id for r in rows if r.gold_emotion}

    tp_ids = sorted(pred_ids & gold_ids)
    fp_ids = sorted(pred_ids - gold_ids)
    fn_ids = sorted(gold_ids - pred_ids)
    tn = len(rows) - len(tp_ids) - len(fp_ids) - len(fn_ids)

    tp = len(tp_ids)
    fp = len(fp_ids)
    fn = len(fn_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0

    id_to_row = {r.content_sentence_id: r for r in rows}
    fp_examples = [id_to_row[i].sentence_text for i in fp_ids[:5]]
    fn_examples = [id_to_row[i].sentence_text for i in fn_ids[:5]]

    result = {
        "name": variant.name,
        "params": {
            "threshold": variant.threshold,
            "top_k": variant.top_k,
            "margin": variant.margin,
        },
        "count": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }
    return result, labels


def evaluate_variant(rows: list[EvalRow], variant: Variant) -> dict:
    result, _ = _evaluate_with_labels(rows, variant)
    return result


def _normalize_template(text: str) -> str:
    # 숫자/공백 변동을 중화해 템플릿 반복 여부를 계산한다.
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_fn_candidate_words(text: str, kiwi: Kiwi) -> list[str]:
    tokens = kiwi.tokenize(text)
    candidates: list[str] = []
    for t in tokens:
        if t.tag not in _FN_VALID_POS:
            continue
        if len(t.form) < 2:
            continue
        surface = text[t.start:t.start + t.len]
        if surface not in candidates:
            candidates.append(surface)
    return candidates


def build_error_cases(rows: list[EvalRow], labels: list) -> list[dict]:
    label_map: dict[int, list] = {}
    for label in labels:
        if label.label_type != SentenceLabelType.EMOTIONALLY_LOADED:
            continue
        label_map.setdefault(label.content_sentence_id, []).append(label)

    error_cases: list[dict] = []
    for row in rows:
        row_labels = label_map.get(row.content_sentence_id, [])
        pred_emotion = bool(row_labels)
        if pred_emotion == row.gold_emotion:
            continue
        error_type = "FP" if pred_emotion and not row.gold_emotion else "FN"
        matched_words = [label.matched_word for label in row_labels if label.matched_word]
        error_cases.append(
            {
                "content_sentence_id": row.content_sentence_id,
                "sentence_text": row.sentence_text,
                "gold_emotion": row.gold_emotion,
                "pred_emotion": pred_emotion,
                "error_type": error_type,
                "matched_words": matched_words,
            }
        )
    return error_cases


def _export_error_cases(path: Path, error_cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in error_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_pattern_support(error_cases: list[dict]) -> dict[str, dict[str, dict]]:
    fp_bucket: dict[str, dict[str, object]] = {}
    fn_bucket: dict[str, dict[str, object]] = {}
    kiwi = Kiwi()

    for case in error_cases:
        text = case["sentence_text"]
        template = _normalize_template(text)
        if case["error_type"] == "FP":
            for word in case["matched_words"]:
                bucket = fp_bucket.setdefault(word, {"texts": set(), "templates": set()})
                bucket["texts"].add(text)
                bucket["templates"].add(template)
        else:
            for word in _extract_fn_candidate_words(text, kiwi):
                bucket = fn_bucket.setdefault(word, {"texts": set(), "templates": set()})
                bucket["texts"].add(text)
                bucket["templates"].add(template)

    def _to_summary(bucket: dict[str, dict[str, set]]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for key, stats in bucket.items():
            support_unique_text = len(stats["texts"])
            template_count = len(stats["templates"])
            result[key] = {
                "support_unique_text": support_unique_text,
                "template_count": template_count,
            }
        return result

    return {
        "fp": _to_summary(fp_bucket),
        "fn": _to_summary(fn_bucket),
    }


def _build_pattern_section(
    support_map: dict[str, dict],
    *,
    error_type: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    scored: list[tuple[str, dict]] = sorted(
        support_map.items(),
        key=lambda item: (
            item[1]["support_unique_text"],
            item[1]["template_count"],
            item[0],
        ),
        reverse=True,
    )

    confirmed: dict[str, dict] = {}
    candidate: dict[str, dict] = {}

    for word, stats in scored:
        impact = (
            {"delta_fp": stats["support_unique_text"], "delta_fn": 0}
            if error_type == "FP"
            else {"delta_fp": 0, "delta_fn": stats["support_unique_text"]}
        )
        record = {
            **stats,
            "impact": impact,
            "status": "candidate",
        }
        is_confirmed = (
            stats["support_unique_text"] >= _CONFIRM_SUPPORT_MIN
            and stats["template_count"] >= _CONFIRM_TEMPLATE_MIN
        )
        if is_confirmed:
            record["status"] = "confirmed"
            confirmed[word] = record
        elif len(candidate) < _CANDIDATE_TOP_N:
            candidate[word] = record

    return confirmed, candidate


def _extract_blacklist_candidates(confirmed_fp_patterns: dict[str, dict]) -> list[dict]:
    ranked = sorted(
        confirmed_fp_patterns.items(),
        key=lambda item: (
            item[1]["impact"]["delta_fp"],
            item[1]["support_unique_text"],
            item[0],
        ),
        reverse=True,
    )
    return [
        {
            "word_root": word,
            "support_unique_text": stats["support_unique_text"],
            "template_count": stats["template_count"],
            "impact": stats["impact"],
            "status": "confirmed",
        }
        for word, stats in ranked
    ]


def _confirm_patterns(error_cases: list[dict]) -> dict:
    support = _compute_pattern_support(error_cases)
    confirmed_fp, candidate_fp = _build_pattern_section(support["fp"], error_type="FP")
    confirmed_fn, candidate_fn = _build_pattern_section(support["fn"], error_type="FN")

    return {
        "criteria": {
            "support_unique_text_min": _CONFIRM_SUPPORT_MIN,
            "template_count_min": _CONFIRM_TEMPLATE_MIN,
            "impact_required": True,
        },
        "confirmed_fp_patterns": confirmed_fp,
        "confirmed_fn_patterns": confirmed_fn,
        "candidate_fp_patterns": candidate_fp,
        "candidate_fn_patterns": candidate_fn,
        "blacklist_candidates": _extract_blacklist_candidates(confirmed_fp),
    }


def _export_pattern_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _top_candidates(report: dict, key: str, limit: int = 5) -> list[dict]:
    items = sorted(
        report.get(key, {}).items(),
        key=lambda item: (
            item[1]["support_unique_text"],
            item[1]["template_count"],
            item[0],
        ),
        reverse=True,
    )
    return [
        {
            "pattern": pattern,
            "support_unique_text": stats["support_unique_text"],
            "template_count": stats["template_count"],
            "impact": stats["impact"],
            "status": stats["status"],
        }
        for pattern, stats in items[:limit]
    ]


def _build_stage_snapshot(
    stage: str,
    legacy: dict,
    improved: dict,
    report: dict,
) -> dict:
    return {
        "stage": stage,
        "legacy": legacy,
        "improved": improved,
        "delta": {
            "fp": improved["fp"] - legacy["fp"],
            "fn": improved["fn"] - legacy["fn"],
            "precision": round(improved["precision"] - legacy["precision"], 6),
            "recall": round(improved["recall"] - legacy["recall"], 6),
            "f1": round(improved["f1"] - legacy["f1"], 6),
        },
        "confirmed_count": {
            "fp": len(report["confirmed_fp_patterns"]),
            "fn": len(report["confirmed_fn_patterns"]),
        },
        "top_candidates": {
            "fp": _top_candidates(report, "candidate_fp_patterns"),
            "fn": _top_candidates(report, "candidate_fn_patterns"),
        },
    }


def _export_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _append_history(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def _check_release_gate(baseline_summary: dict, current_summary: dict) -> dict:
    base_raw = baseline_summary["stages"]["raw"]["improved"]
    curr_raw = current_summary["stages"]["raw"]["improved"]

    fp_ok = curr_raw["fp"] <= base_raw["fp"]
    precision_drop = base_raw["precision"] - curr_raw["precision"]
    precision_ok = precision_drop <= 0.02
    f1_ok = curr_raw["f1"] >= base_raw["f1"]
    recall_drop = base_raw["recall"] - curr_raw["recall"]
    recall_ok = recall_drop <= 0.02

    passed = fp_ok and precision_ok and f1_ok and recall_ok
    return {
        "passed": passed,
        "checks": {
            "fp_non_increase": fp_ok,
            "precision_drop_lte_0_02": precision_ok,
            "f1_non_decrease": f1_ok,
            "recall_drop_lte_0_02": recall_ok,
        },
        "delta_vs_baseline": {
            "fp": curr_raw["fp"] - base_raw["fp"],
            "fn": curr_raw["fn"] - base_raw["fn"],
            "precision": round(curr_raw["precision"] - base_raw["precision"], 6),
            "recall": round(curr_raw["recall"] - base_raw["recall"], 6),
            "f1": round(curr_raw["f1"] - base_raw["f1"], 6),
        },
    }


def print_result(result: dict) -> None:
    p = result["params"]
    print(f"[{result['name']}] threshold={p['threshold']:.2f} top_k={p['top_k']} margin={p['margin']:.2f}")
    print(
        f"count={result['count']} TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}"
    )
    print(
        "precision={:.4f} recall={:.4f} f1={:.4f} accuracy={:.4f}".format(
            result["precision"], result["recall"], result["f1"], result["accuracy"]
        )
    )
    if result["fp_examples"]:
        print("FP examples:")
        for text in result["fp_examples"]:
            print(f"  - {text}")
    if result["fn_examples"]:
        print("FN examples:")
        for text in result["fn_examples"]:
            print(f"  - {text}")
    print()


def run_grid(rows: list[EvalRow]) -> None:
    thresholds = [0.86, 0.88, 0.90]
    margins = [0.04, 0.06, 0.08]
    top_ks = [2, 3]
    results = []
    for thr in thresholds:
        for margin in margins:
            for top_k in top_ks:
                name = f"GRID thr={thr:.2f} top_k={top_k} m={margin:.2f}"
                res = evaluate_variant(
                    rows,
                    Variant(name=name, threshold=thr, top_k=top_k, margin=margin),
                )
                results.append(res)

    results.sort(key=lambda r: (r["f1"], r["precision"], -r["fp"]), reverse=True)
    print("[Grid Search Top 5]")
    for r in results[:5]:
        p = r["params"]
        print(
            "f1={:.4f} precision={:.4f} recall={:.4f} fp={} (thr={:.2f}, top_k={}, margin={:.2f})".format(
                r["f1"],
                r["precision"],
                r["recall"],
                r["fp"],
                p["threshold"],
                p["top_k"],
                p["margin"],
            )
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("analysis_bc/data/eval/emotion_eval_sample.jsonl"),
        help="JSONL 라벨셋 경로",
    )
    parser.add_argument(
        "--run-grid",
        action="store_true",
        help="threshold/top_k/margin 그리드서치 실행",
    )
    parser.add_argument(
        "--export-errors",
        type=Path,
        default=None,
        help="오차 케이스(JSONL) 출력 경로",
    )
    parser.add_argument(
        "--dedupe-by-text",
        action="store_true",
        help="동일 sentence_text를 1건으로 압축해 평가",
    )
    parser.add_argument(
        "--confirm-patterns",
        action="store_true",
        help="FP/FN 패턴 확정 리포트 생성",
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=_DEFAULT_SUMMARY_PATH,
        help="평가 요약 JSON 저장 경로",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="승인 게이트 비교용 기준선 summary 경로",
    )
    parser.add_argument(
        "--change-reason",
        type=str,
        default="manual update",
        help="변경 이력 기록 사유",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=_DEFAULT_HISTORY_PATH,
        help="변경 이력(JSONL) 저장 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_full = load_dataset(args.dataset)
    rows = dedupe_rows_by_text(rows_full) if args.dedupe_by_text else rows_full
    print(f"Loaded dataset: {args.dataset} ({len(rows)} rows)\n")

    legacy_variant = Variant(name="A - Legacy", threshold=0.80, top_k=1, margin=0.00)
    improved_variant = Variant(name="B - Improved", threshold=0.88, top_k=3, margin=0.06)

    legacy, _ = _evaluate_with_labels(rows, legacy_variant)
    improved, improved_labels = _evaluate_with_labels(rows, improved_variant)

    print_result(legacy)
    print_result(improved)

    print("[Delta B-A]")
    print(
        "FP: {} -> {} ({:+d})".format(
            legacy["fp"], improved["fp"], improved["fp"] - legacy["fp"]
        )
    )
    print(
        "precision: {:.4f} -> {:.4f} ({:+.4f})".format(
            legacy["precision"], improved["precision"], improved["precision"] - legacy["precision"]
        )
    )
    print(
        "recall: {:.4f} -> {:.4f} ({:+.4f})".format(
            legacy["recall"], improved["recall"], improved["recall"] - legacy["recall"]
        )
    )
    print(
        "f1: {:.4f} -> {:.4f} ({:+.4f})".format(
            legacy["f1"], improved["f1"], improved["f1"] - legacy["f1"]
        )
    )
    print()

    if args.confirm_patterns:
        # 계획 기준: 패턴 확정은 raw -> dedupe 두 단계를 고정 실행한다.
        print("[Pattern Pipeline] stage=raw")
        raw_legacy, _ = _evaluate_with_labels(rows_full, legacy_variant)
        raw_improved, raw_labels = _evaluate_with_labels(rows_full, improved_variant)
        print_result(raw_legacy)
        print_result(raw_improved)
        raw_errors = build_error_cases(rows_full, raw_labels)
        raw_report = _confirm_patterns(raw_errors)
        _export_error_cases(_DEFAULT_ERROR_RAW_PATH, raw_errors)
        _export_pattern_report(_DEFAULT_PATTERN_RAW_PATH, raw_report)
        print(f"[Saved] error cases: {_DEFAULT_ERROR_RAW_PATH}")
        print(f"[Saved] pattern report: {_DEFAULT_PATTERN_RAW_PATH}")

        print("[Pattern Pipeline] stage=deduped")
        deduped_rows = dedupe_rows_by_text(rows_full)
        deduped_legacy, _ = _evaluate_with_labels(deduped_rows, legacy_variant)
        deduped_improved, deduped_labels = _evaluate_with_labels(deduped_rows, improved_variant)
        print_result(deduped_legacy)
        print_result(deduped_improved)
        deduped_errors = build_error_cases(deduped_rows, deduped_labels)
        deduped_report = _confirm_patterns(deduped_errors)
        _export_error_cases(_DEFAULT_ERROR_DEDUPED_PATH, deduped_errors)
        _export_pattern_report(_DEFAULT_PATTERN_DEDUPED_PATH, deduped_report)
        print(f"[Saved] error cases: {_DEFAULT_ERROR_DEDUPED_PATH}")
        print(f"[Saved] pattern report: {_DEFAULT_PATTERN_DEDUPED_PATH}")

        # 호환 경로도 유지
        _export_pattern_report(_DEFAULT_PATTERN_REPORT_PATH, deduped_report)
        if args.export_errors:
            # 사용자 지정 경로는 현재 실행 컨텍스트(rows 기준) 결과를 저장
            current_errors = build_error_cases(rows, improved_labels)
            _export_error_cases(args.export_errors, current_errors)
            print(f"[Saved] error cases (custom): {args.export_errors}")
        else:
            _export_error_cases(_DEFAULT_ERROR_PATH, deduped_errors)

        top_blacklist = raw_report["blacklist_candidates"][:10]
        if top_blacklist:
            print("[Blacklist Candidates] word_root (from confirmed_fp_patterns)")
            for item in top_blacklist:
                print(
                    f"  - {item['word_root']} (support={item['support_unique_text']}, templates={item['template_count']})"
                )
        else:
            print("[Blacklist Candidates] none")

        print(
            "[Confirmed/raw] FP={} FN={}".format(
                len(raw_report["confirmed_fp_patterns"]),
                len(raw_report["confirmed_fn_patterns"]),
            )
        )
        print(
            "[Confirmed/deduped] FP={} FN={}".format(
                len(deduped_report["confirmed_fp_patterns"]),
                len(deduped_report["confirmed_fn_patterns"]),
            )
        )

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset),
            "change_reason": args.change_reason,
            "criteria": {
                "support_unique_text_min": _CONFIRM_SUPPORT_MIN,
                "template_count_min": _CONFIRM_TEMPLATE_MIN,
                "release_gate": {
                    "fp_non_increase": True,
                    "precision_drop_lte_0_02": True,
                    "f1_non_decrease": True,
                    "recall_drop_lte_0_02": True,
                },
            },
            "stages": {
                "raw": _build_stage_snapshot("raw", raw_legacy, raw_improved, raw_report),
                "deduped": _build_stage_snapshot("deduped", deduped_legacy, deduped_improved, deduped_report),
            },
            "files": {
                "error_cases_raw": str(_DEFAULT_ERROR_RAW_PATH),
                "error_cases_deduped": str(_DEFAULT_ERROR_DEDUPED_PATH),
                "pattern_report_raw": str(_DEFAULT_PATTERN_RAW_PATH),
                "pattern_report_deduped": str(_DEFAULT_PATTERN_DEDUPED_PATH),
            },
        }

        if args.baseline_summary:
            if not args.baseline_summary.exists():
                raise FileNotFoundError(f"baseline summary 파일이 없습니다: {args.baseline_summary}")
            baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
            gate = _check_release_gate(baseline_summary, summary)
            summary["release_gate"] = gate
            print(
                "[Release Gate] passed={} delta(fp={:+d}, fn={:+d}, precision={:+.4f}, recall={:+.4f}, f1={:+.4f})".format(
                    gate["passed"],
                    gate["delta_vs_baseline"]["fp"],
                    gate["delta_vs_baseline"]["fn"],
                    gate["delta_vs_baseline"]["precision"],
                    gate["delta_vs_baseline"]["recall"],
                    gate["delta_vs_baseline"]["f1"],
                )
            )

        _export_summary(args.save_summary, summary)
        _append_history(args.history_path, summary)
        print(f"[Saved] summary: {args.save_summary}")
        print(f"[Saved] history: {args.history_path}")
        print()
    elif args.export_errors:
        error_path = args.export_errors or _DEFAULT_ERROR_PATH
        error_cases = build_error_cases(rows, improved_labels)
        _export_error_cases(error_path, error_cases)
        print(f"[Saved] error cases: {error_path}")

    if args.run_grid:
        run_grid(rows)


if __name__ == "__main__":
    main()
