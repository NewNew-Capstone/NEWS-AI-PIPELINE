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
from analysis_bc.tagger.span_tagger import SpanTagger, TagTrace


def _load_texts(text: str | None, file_path: str | None) -> list[str]:
    lines: list[str] = []
    if text:
        lines.append(text.strip())
    if file_path:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def _to_sentences(texts: list[str]) -> list[ClassifiedSentenceDto]:
    return [
        ClassifiedSentenceDto(
            content_sentence_id=i + 1,
            sentence_text=t,
            sentence_order=i,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=1.0,
        )
        for i, t in enumerate(texts)
    ]


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


def _trace_to_dict(trace: TagTrace | None) -> dict[str, Any]:
    if trace is None:
        return {}

    sentences_payload = []
    reasons = Counter()

    for sentence in trace.sentences:
        token_payload = []
        for token in sentence.tokens:
            if token.reject_reason:
                reasons[token.reject_reason] += 1
            token_payload.append(
                {
                    "form": token.form,
                    "surface": token.surface,
                    "tag": token.tag,
                    "start": token.start,
                    "len": token.length,
                    "accepted": token.accepted,
                    "reject_reason": token.reject_reason,
                    "qdrant_scores": token.qdrant_scores,
                    "created_span": _span_to_dict(token.created_span) if token.created_span else None,
                }
            )
        sentences_payload.append(
            {
                "content_sentence_id": sentence.content_sentence_id,
                "sentence_text": sentence.sentence_text,
                "tokens": token_payload,
            }
        )

    return {
        "qdrant_healthy": trace.qdrant_healthy,
        "qdrant_health_reason": trace.qdrant_health_reason,
        "rejection_stats": dict(reasons),
        "sentences": sentences_payload,
    }


def _print_pretty(summary: dict[str, Any]) -> None:
    print("=== Emotion Tag Trace ===")
    print(
        f"Qdrant healthy: {summary.get('trace', {}).get('qdrant_healthy')} "
        f"({summary.get('trace', {}).get('qdrant_health_reason')})"
    )
    for sent in summary.get("trace", {}).get("sentences", []):
        print("")
        print(f"[Sentence {sent['content_sentence_id']}] {sent['sentence_text']}")
        for t in sent["tokens"]:
            decision = "ACCEPT" if t["accepted"] else f"REJECT({t['reject_reason']})"
            print(
                f" - {t['surface']} [{t['tag']}] @{t['start']}+{t['len']} "
                f"=> {decision} scores={t['qdrant_scores']}"
            )
    print("")
    print("Rejection stats:", summary.get("trace", {}).get("rejection_stats", {}))
    print("Spans:", len(summary.get("spans", [])))


def _build_compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    trace = summary.get("trace", {})
    spans = summary.get("spans", [])
    return {
        "input_count": summary.get("input_count", 0),
        "qdrant_healthy": trace.get("qdrant_healthy"),
        "qdrant_health_reason": trace.get("qdrant_health_reason"),
        "rejection_stats": trace.get("rejection_stats", {}),
        "span_count": len(spans),
        "accepted_spans": spans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace emotion tagging decisions for input sentences.")
    parser.add_argument("--text", type=str, help="Single input sentence")
    parser.add_argument("--file", type=str, help="Text file path (one sentence per line)")
    parser.add_argument("--pretty", action="store_true", help="Print human-readable trace logs")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON summary (without full token list)",
    )
    args = parser.parse_args()

    texts = _load_texts(args.text, args.file)
    if not texts:
        raise SystemExit("No input. Use --text or --file.")

    tagger = SpanTagger()
    sentences = _to_sentences(texts)
    spans = tagger.tag(sentences, debug=True)

    summary = {
        "input_count": len(texts),
        "spans": [_span_to_dict(s) for s in spans],
        "trace": _trace_to_dict(tagger.last_debug_trace),
    }

    if args.pretty:
        _print_pretty(summary)
        if not args.json:
            return

    payload = _build_compact_summary(summary) if args.compact else summary
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
