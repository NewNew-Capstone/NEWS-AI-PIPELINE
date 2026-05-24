from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_bc.classifier import FactOpinionClassifier
from analysis_bc.preprocessor import SentencePreprocessor, split_into_sentences
from analysis_bc.schemas import SentenceInputDto
from scripts.emotion_gate_experiment import (
    ConditionalSpanRunner,
    EmotionGateClassifier,
    GateResult,
)


DEFAULT_INPUT = Path("/Users/gimmingyu/youtube_transcript.json")
DEFAULT_OUTPUT = Path(
    "analysis_bc/data/eval/reports/accepted_token_candidates_from_youtube_transcript.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "analysis_bc/data/eval/reports/accepted_token_candidates_from_youtube_transcript_summary.json"
)

_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CAPTION_PREFIX_RE = re.compile(r"^\s*Kind:\s*captions\s+Language:\s*ko\s*", re.IGNORECASE)


def _hangul_count(text: str) -> int:
    return len(_HANGUL_RE.findall(text))


def _latin_count(text: str) -> int:
    return len(_LATIN_RE.findall(text))


def _clean_transcript_text(text: str) -> str:
    text = html.unescape(text)
    text = _CAPTION_PREFIX_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_korean_body(
    row: dict[str, Any],
    *,
    min_hangul_chars: int,
    min_hangul_ratio: float,
) -> bool:
    if str(row.get("language_code", "")).lower() != "ko":
        return False
    text = _clean_transcript_text(str(row.get("transcript_text") or ""))
    hangul = _hangul_count(text)
    if hangul < min_hangul_chars:
        return False
    latin = _latin_count(text)
    denominator = hangul + latin
    if denominator == 0:
        return False
    return (hangul / denominator) >= min_hangul_ratio


def _load_korean_transcripts(
    path: Path,
    *,
    min_hangul_chars: int,
    min_hangul_ratio: float,
) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")

    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _is_korean_body(
            row,
            min_hangul_chars=min_hangul_chars,
            min_hangul_ratio=min_hangul_ratio,
        ):
            continue
        copied = dict(row)
        copied["transcript_text"] = _clean_transcript_text(str(row.get("transcript_text") or ""))
        selected.append(copied)
    return selected


def _build_inputs(raw_text: str, language: str) -> tuple[list[SentenceInputDto], int]:
    split_sentences = split_into_sentences(raw_text, language=language)
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


def _predict_gate(
    gate_classifier: EmotionGateClassifier,
    opinion_sentences: list,
    *,
    gate_threshold: float,
    label_threshold: float,
    top_k_labels: int,
) -> list[GateResult]:
    gate_results = gate_classifier.predict_batch(
        texts=[s.sentence_text for s in opinion_sentences],
        gate_threshold=gate_threshold,
        label_threshold=label_threshold,
        top_k=top_k_labels,
    )
    for i, gate in enumerate(gate_results):
        gate.content_sentence_id = opinion_sentences[i].content_sentence_id
        gate.sentence_text = opinion_sentences[i].sentence_text
    return gate_results


def _accepted_rows_for_transcript(
    row: dict[str, Any],
    *,
    classifier: FactOpinionClassifier,
    gate_classifier: EmotionGateClassifier,
    runner: ConditionalSpanRunner,
    gate_threshold: float,
    label_threshold: float,
    top_k_labels: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sentence_inputs, split_count = _build_inputs(row["transcript_text"], "ko")
    classified = classifier.classify(sentence_inputs)
    opinion_sentences = [s for s in classified if s.label == "opinion_like"]
    fact_count = len([s for s in classified if s.label == "fact_like"])

    stats = {
        "split_sentence_count": split_count,
        "preprocessed_count": len(sentence_inputs),
        "fact_count": fact_count,
        "opinion_count": len(opinion_sentences),
        "gate_target_count": len(opinion_sentences),
    }
    if not opinion_sentences:
        return [], stats

    gate_results = _predict_gate(
        gate_classifier,
        opinion_sentences,
        gate_threshold=gate_threshold,
        label_threshold=label_threshold,
        top_k_labels=top_k_labels,
    )
    gate_by_id = {g.content_sentence_id: g for g in gate_results}
    _, trace_sentences = runner.run(sentences=opinion_sentences, gate_results=gate_results)

    accepted: list[dict[str, Any]] = []
    for sentence in trace_sentences:
        gate = gate_by_id[sentence.content_sentence_id]
        for token in sentence.tokens:
            if not token.accepted:
                continue
            accepted.append(
                {
                    "youtube_video_id": row.get("youtube_video_id"),
                    "transcript_id": row.get("id"),
                    "content_sentence_id": sentence.content_sentence_id,
                    "sentence_text": sentence.sentence_text,
                    "surface": token.surface,
                    "form": token.form,
                    "tag": token.tag,
                    "start": token.start,
                    "len": token.length,
                    "qdrant_scores": token.qdrant_scores,
                    "payload_polarities": token.payload_polarities,
                    "gate_score": gate.gate_score,
                    "top_labels": gate.top_labels,
                }
            )
    return accepted, stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_summary(
    accepted_rows: list[dict[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
    transcript_count: int,
    selected_count: int,
    aggregate_stats: Counter,
) -> dict[str, Any]:
    by_surface: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in accepted_rows:
        surface = str(row["surface"])
        bucket = by_surface.setdefault(
            surface,
            {
                "surface": surface,
                "accept_count": 0,
                "unique_sentence_count": 0,
                "tags": Counter(),
                "forms": Counter(),
                "youtube_video_ids": set(),
                "transcript_ids": set(),
                "max_qdrant_score": None,
                "payload_polarities": Counter(),
            },
        )
        bucket["accept_count"] += 1
        bucket["tags"][row["tag"]] += 1
        bucket["forms"][row["form"]] += 1
        bucket["youtube_video_ids"].add(row["youtube_video_id"])
        bucket["transcript_ids"].add(row["transcript_id"])
        if row["qdrant_scores"]:
            score = float(row["qdrant_scores"][0])
            current = bucket["max_qdrant_score"]
            bucket["max_qdrant_score"] = score if current is None else max(current, score)
        for polarity in row["payload_polarities"]:
            bucket["payload_polarities"][str(polarity)] += 1

        if len(examples[surface]) < 5:
            examples[surface].append(
                {
                    "youtube_video_id": row["youtube_video_id"],
                    "transcript_id": row["transcript_id"],
                    "sentence_text": row["sentence_text"],
                    "qdrant_scores": row["qdrant_scores"],
                    "payload_polarities": row["payload_polarities"],
                    "top_labels": row["top_labels"],
                }
            )

    unique_sentence_keys_by_surface: dict[str, set[tuple[Any, int, str]]] = defaultdict(set)
    for row in accepted_rows:
        unique_sentence_keys_by_surface[str(row["surface"])].add(
            (row["transcript_id"], row["content_sentence_id"], row["sentence_text"])
        )

    candidates = []
    for surface, bucket in by_surface.items():
        unique_sentence_count = len(unique_sentence_keys_by_surface[surface])
        candidates.append(
            {
                "surface": surface,
                "accept_count": bucket["accept_count"],
                "unique_sentence_count": unique_sentence_count,
                "tags": dict(bucket["tags"].most_common()),
                "forms": dict(bucket["forms"].most_common()),
                "youtube_video_count": len(bucket["youtube_video_ids"]),
                "transcript_count": len(bucket["transcript_ids"]),
                "max_qdrant_score": bucket["max_qdrant_score"],
                "payload_polarities": dict(bucket["payload_polarities"].most_common()),
                "examples": examples[surface],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["unique_sentence_count"],
            item["accept_count"],
            item["surface"],
        ),
        reverse=True,
    )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total_transcripts": transcript_count,
        "selected_korean_transcripts": selected_count,
        "accepted_token_count": len(accepted_rows),
        "pipeline_counts": dict(aggregate_stats),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ACCEPT token candidate report from YouTube transcript JSON."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--min-hangul-chars", type=int, default=20)
    parser.add_argument("--min-hangul-ratio", type=float, default=0.30)
    parser.add_argument("--gate-threshold", type=float, default=0.35)
    parser.add_argument("--label-threshold", type=float, default=0.2)
    parser.add_argument("--top-k-labels", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Limit selected transcripts for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected JSON array: {args.input}")

    transcripts = _load_korean_transcripts(
        args.input,
        min_hangul_chars=args.min_hangul_chars,
        min_hangul_ratio=args.min_hangul_ratio,
    )
    if args.limit > 0:
        transcripts = transcripts[: args.limit]

    print(f"Loaded transcripts: total={len(raw_rows)} selected_ko={len(transcripts)}", flush=True)
    classifier = FactOpinionClassifier(model_path="analysis_bc/models/best_model")
    gate_classifier = EmotionGateClassifier()
    runner = ConditionalSpanRunner()

    accepted_rows: list[dict[str, Any]] = []
    aggregate_stats: Counter = Counter()
    for index, transcript in enumerate(transcripts, start=1):
        rows, stats = _accepted_rows_for_transcript(
            transcript,
            classifier=classifier,
            gate_classifier=gate_classifier,
            runner=runner,
            gate_threshold=args.gate_threshold,
            label_threshold=args.label_threshold,
            top_k_labels=args.top_k_labels,
        )
        accepted_rows.extend(rows)
        aggregate_stats.update(stats)
        print(
            "processed {}/{} transcript_id={} youtube_video_id={} accepted={}".format(
                index,
                len(transcripts),
                transcript.get("id"),
                transcript.get("youtube_video_id"),
                len(rows),
            ),
            flush=True,
        )

    _write_jsonl(args.output, accepted_rows)
    summary = _build_summary(
        accepted_rows,
        input_path=args.input,
        output_path=args.output,
        transcript_count=len(raw_rows),
        selected_count=len(transcripts),
        aggregate_stats=aggregate_stats,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote jsonl: {args.output} ({len(accepted_rows)} rows)")
    print(f"wrote summary: {args.summary_output}")


if __name__ == "__main__":
    main()
