from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.tagger.span_tagger import SpanTagger
from scripts.emotion_gate_experiment import ConditionalSpanRunner, EmotionGateClassifier


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_classified(rows: list[dict]) -> list[ClassifiedSentenceDto]:
    return [
        ClassifiedSentenceDto(
            content_sentence_id=int(r.get("content_sentence_id", i + 1)),
            sentence_text=str(r["sentence_text"]),
            sentence_order=i + 1,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=0.99,
        )
        for i, r in enumerate(rows)
    ]


def _metrics(gold_ids: set[int], pred_ids: set[int], total: int) -> dict:
    tp = len(gold_ids & pred_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)
    tn = total - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "count": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _extract_err_tokens(rows: list[dict], pred_ids: set[int], target_words: set[str]) -> dict:
    found = []
    id_to_text = {int(r.get("content_sentence_id", i + 1)): str(r["sentence_text"]) for i, r in enumerate(rows)}
    for cid in sorted(pred_ids):
        text = id_to_text.get(cid, "")
        hit = [w for w in target_words if w in text]
        if hit:
            found.append({"content_sentence_id": cid, "matched_keywords": hit, "sentence_text": text})
    return {"count": len(found), "examples": found[:20]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline span vs gated span on emotion_eval_sample.jsonl")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("analysis_bc/data/eval/emotion_eval_sample.jsonl"),
    )
    parser.add_argument("--gate-threshold", type=float, default=0.35)
    parser.add_argument("--label-threshold", type=float, default=0.2)
    parser.add_argument("--top-k-labels", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("analysis_bc/data/eval/reports/gated_eval_report.json"))
    args = parser.parse_args()

    rows = _load_rows(args.dataset)
    classified = _to_classified(rows)
    gold_ids = {
        int(r.get("content_sentence_id", i + 1))
        for i, r in enumerate(rows)
        if bool(r.get("gold_emotion"))
    }

    baseline_tagger = SpanTagger()
    baseline_spans = baseline_tagger.tag(classified)
    baseline_pred = {
        s.content_sentence_id
        for s in baseline_spans
        if s.label_type in (SentenceLabelType.EMOTIONALLY_LOADED, SentenceLabelType.EMOTIONALLY_LOADED.value)
    }
    baseline_metrics = _metrics(gold_ids, baseline_pred, len(rows))

    gate = EmotionGateClassifier()
    gate_results = gate.predict_batch(
        texts=[r["sentence_text"] for r in rows],
        gate_threshold=args.gate_threshold,
        label_threshold=args.label_threshold,
        top_k=args.top_k_labels,
    )
    runner = ConditionalSpanRunner()
    gated_spans, trace = runner.run(classified, gate_results)
    gated_pred = {s.content_sentence_id for s in gated_spans}
    gated_metrics = _metrics(gold_ids, gated_pred, len(rows))

    keywords = {"중국", "부리", "오히려", "어차피", "결국"}
    report = {
        "dataset": str(args.dataset),
        "params": {
            "gate_threshold": args.gate_threshold,
            "label_threshold": args.label_threshold,
            "top_k_labels": args.top_k_labels,
        },
        "baseline": baseline_metrics,
        "gated": gated_metrics,
        "delta": {
            "fp": gated_metrics["fp"] - baseline_metrics["fp"],
            "fn": gated_metrics["fn"] - baseline_metrics["fn"],
            "precision": gated_metrics["precision"] - baseline_metrics["precision"],
            "recall": gated_metrics["recall"] - baseline_metrics["recall"],
            "f1": gated_metrics["f1"] - baseline_metrics["f1"],
        },
        "gate_passed_count": sum(1 for t in trace if t.gate_passed),
        "gate_total_count": len(trace),
        "gate_pass_rate": (sum(1 for t in trace if t.gate_passed) / len(trace)) if trace else 0.0,
        "target_keyword_hit_baseline": _extract_err_tokens(rows, baseline_pred, keywords),
        "target_keyword_hit_gated": _extract_err_tokens(rows, gated_pred, keywords),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
