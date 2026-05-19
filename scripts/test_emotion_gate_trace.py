from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.classifier import FactOpinionClassifier
from analysis_bc.preprocessor import SentencePreprocessor, split_into_sentences
from analysis_bc.schemas import SentenceInputDto
from scripts.emotion_gate_experiment import (
    ConditionalSpanRunner,
    EmotionGateClassifier,
    GateTraceSentence,
)


def _load_raw_blocks(text: str | None, file_path: str | None) -> list[str]:
    blocks: list[str] = []
    if text:
        blocks.append(text.strip())
    if file_path:
        content = Path(file_path).read_text(encoding="utf-8").strip()
        if content:
            blocks.append(content)
    return blocks


def _to_sentence_inputs(texts: list[str]) -> list[SentenceInputDto]:
    return [
        SentenceInputDto(
            content_sentence_id=i + 1,
            sentence_text=t,
            sentence_order=i,
            start_time_ms=None,
            end_time_ms=None,
        )
        for i, t in enumerate(texts)
    ]


def _build_inputs_via_feature1_preprocess(raw_blocks: list[str], language: str) -> tuple[list[SentenceInputDto], int]:
    # 기능1 raw 분석과 유사하게: raw text -> 문장 분리 -> 전처리
    split_sentences: list[SentenceInputDto] = []
    for block in raw_blocks:
        split_sentences.extend(split_into_sentences(block, language=language))

    if not split_sentences:
        return [], 0

    # id/order 재부여 (후속 모듈 일관성)
    normalized = [
        SentenceInputDto(
            content_sentence_id=i + 1,
            sentence_text=s.sentence_text,
            sentence_order=i,
            start_time_ms=s.start_time_ms,
            end_time_ms=s.end_time_ms,
        )
        for i, s in enumerate(split_sentences)
    ]
    preprocessed = SentencePreprocessor(expected_language=language).preprocess(normalized)
    return preprocessed, len(split_sentences)


def _span_to_dict(span: Any) -> dict[str, Any]:
    label = span.label_type.value if hasattr(span.label_type, "value") else span.label_type
    return {
        "content_sentence_id": span.content_sentence_id,
        "start_offset": span.start_offset,
        "end_offset": span.end_offset,
        "label_type": label,
        "score": span.score,
        "matched_word": span.matched_word,
    }


def _trace_to_dict(trace_sentences: list[GateTraceSentence]) -> dict[str, Any]:
    reasons = Counter()
    payload: list[dict[str, Any]] = []
    gate_passed = 0
    for sent in trace_sentences:
        if sent.gate_passed:
            gate_passed += 1
        token_payload = []
        for token in sent.tokens:
            if token.reject_reason:
                reasons[token.reject_reason] += 1
            token_payload.append(
                {
                    "surface": token.surface,
                    "form": token.form,
                    "tag": token.tag,
                    "start": token.start,
                    "len": token.length,
                    "accepted": token.accepted,
                    "reject_reason": token.reject_reason,
                    "qdrant_scores": token.qdrant_scores,
                    "payload_polarities": token.payload_polarities,
                }
            )
        payload.append(
            {
                "content_sentence_id": sent.content_sentence_id,
                "sentence_text": sent.sentence_text,
                "gate_score": sent.gate_score,
                "gate_passed": sent.gate_passed,
                "top_labels": sent.top_labels,
                "skip_reason": sent.skip_reason,
                "tokens": token_payload,
            }
        )
    return {
        "rejection_stats": dict(reasons),
        "gate_passed_count": gate_passed,
        "gate_total_count": len(trace_sentences),
        "gate_pass_rate": (gate_passed / len(trace_sentences)) if trace_sentences else 0.0,
        "sentences": payload,
    }


def _print_pretty(summary: dict[str, Any]) -> None:
    print("=== Emotion Gate + Conditional Span Trace ===")
    print(
        f"Gate pass rate: {summary['trace'].get('gate_passed_count', 0)}/"
        f"{summary['trace'].get('gate_total_count', 0)}"
    )
    for sent in summary["trace"].get("sentences", []):
        print("")
        print(
            f"[Sentence {sent['content_sentence_id']}] gate={sent['gate_passed']} "
            f"score={sent['gate_score']:.4f} labels={sent['top_labels']}"
        )
        print(sent["sentence_text"])
        if not sent["gate_passed"]:
            print(f" - SKIP({sent['skip_reason']})")
            continue
        for t in sent["tokens"]:
            decision = "ACCEPT" if t["accepted"] else f"REJECT({t['reject_reason']})"
            print(
                f" - {t['surface']} [{t['tag']}] @{t['start']}+{t['len']} "
                f"=> {decision} scores={t['qdrant_scores']} pol={t['payload_polarities']}"
            )
    print("")
    print("Rejection stats:", summary["trace"].get("rejection_stats", {}))
    print("Spans:", len(summary["spans"]))


def _build_compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_count": summary["input_count"],
        "fact_count": summary["fact_count"],
        "opinion_count": summary["opinion_count"],
        "opinion_gate_target_count": summary["opinion_gate_target_count"],
        "span_count": len(summary["spans"]),
        "accepted_spans": summary["spans"],
        "gate_passed_count": summary["trace"]["gate_passed_count"],
        "gate_total_count": summary["trace"]["gate_total_count"],
        "gate_pass_rate": summary["trace"]["gate_pass_rate"],
        "rejection_stats": summary["trace"]["rejection_stats"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace gated emotion tagging decisions.")
    parser.add_argument("--text", type=str, help="Single input sentence")
    parser.add_argument("--file", type=str, help="Text file path (one sentence per line)")
    parser.add_argument("--gate-threshold", type=float, default=0.35)
    parser.add_argument("--label-threshold", type=float, default=0.2)
    parser.add_argument("--top-k-labels", type=int, default=3)
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument(
        "--skip-fact-opinion-classifier",
        action="store_true",
        help="Skip fact/opinion split and treat all input as opinion_like.",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    raw_blocks = _load_raw_blocks(args.text, args.file)
    if not raw_blocks:
        raise SystemExit("No input. Use --text or --file.")
    sentence_inputs, split_count = _build_inputs_via_feature1_preprocess(raw_blocks, args.language)
    if not sentence_inputs:
        raise SystemExit("No usable sentences after split/preprocess.")
    texts = [s.sentence_text for s in sentence_inputs]
    gate_classifier = EmotionGateClassifier()
    runner = ConditionalSpanRunner()

    if args.skip_fact_opinion_classifier:
        classified = [
            ClassifiedSentenceDto(
                content_sentence_id=s.content_sentence_id,
                sentence_text=s.sentence_text,
                sentence_order=s.sentence_order,
                start_time_ms=s.start_time_ms,
                end_time_ms=s.end_time_ms,
                label="opinion_like",
                confidence=1.0,
            )
            for s in sentence_inputs
        ]
    else:
        fact_opinion_classifier = FactOpinionClassifier(model_path="analysis_bc/models/best_model")
        classified = fact_opinion_classifier.classify(sentence_inputs)

    opinion_sentences = [s for s in classified if s.label == "opinion_like"]
    fact_count = len([s for s in classified if s.label == "fact_like"])
    opinion_count = len(opinion_sentences)

    if not opinion_sentences:
        summary = {
            "input_count": len(texts),
            "raw_block_count": len(raw_blocks),
            "split_sentence_count": split_count,
            "preprocessed_count": len(sentence_inputs),
            "fact_count": fact_count,
            "opinion_count": opinion_count,
            "opinion_gate_target_count": 0,
            "params": {
                "gate_threshold": args.gate_threshold,
                "label_threshold": args.label_threshold,
                "top_k_labels": args.top_k_labels,
                "skip_fact_opinion_classifier": args.skip_fact_opinion_classifier,
            },
            "spans": [],
            "trace": {
                "rejection_stats": {},
                "gate_passed_count": 0,
                "gate_total_count": 0,
                "gate_pass_rate": 0.0,
                "sentences": [],
            },
        }
        payload = _build_compact(summary) if args.compact else summary
        if args.pretty:
            _print_pretty(summary)
            if not args.json:
                return
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    gate_results = gate_classifier.predict_batch(
        texts=[s.sentence_text for s in opinion_sentences],
        gate_threshold=args.gate_threshold,
        label_threshold=args.label_threshold,
        top_k=args.top_k_labels,
    )
    for i, g in enumerate(gate_results):
        g.content_sentence_id = opinion_sentences[i].content_sentence_id
        g.sentence_text = opinion_sentences[i].sentence_text

    spans, trace_sentences = runner.run(sentences=opinion_sentences, gate_results=gate_results)

    summary = {
        "input_count": len(texts),
        "raw_block_count": len(raw_blocks),
        "split_sentence_count": split_count,
        "preprocessed_count": len(sentence_inputs),
        "fact_count": fact_count,
        "opinion_count": opinion_count,
        "opinion_gate_target_count": len(opinion_sentences),
        "params": {
            "gate_threshold": args.gate_threshold,
            "label_threshold": args.label_threshold,
            "top_k_labels": args.top_k_labels,
            "skip_fact_opinion_classifier": args.skip_fact_opinion_classifier,
        },
        "spans": [_span_to_dict(s) for s in spans],
        "trace": _trace_to_dict(trace_sentences),
    }

    if args.pretty:
        _print_pretty(summary)
        if not args.json:
            return
    payload = _build_compact(summary) if args.compact else summary
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
