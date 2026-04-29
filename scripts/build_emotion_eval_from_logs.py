"""운영 분석 로그(JSONL)에서 emotion 평가셋(JSONL)을 생성한다.

사용 예시:
    .venv/bin/python scripts/build_emotion_eval_from_logs.py \
      --input logs/analyze_requests.jsonl \
      --output analysis_bc/data/eval/emotion_eval_operational_sample.jsonl \
      --sample-size 400 \
      --recent-days 7 \
      --autolabel
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.preprocessor import split_into_sentences
from analysis_bc.tagger.span_tagger import SpanTagger


@dataclass(frozen=True)
class Row:
    sentence_text: str
    created_at: datetime | None
    source: str


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_rows_from_record(record: dict) -> list[Row]:
    created_at = _parse_dt(
        record.get("created_at")
        or record.get("timestamp")
        or record.get("requested_at")
    )
    rows: list[Row] = []

    sentences = record.get("sentences")
    if isinstance(sentences, list):
        for item in sentences:
            if not isinstance(item, dict):
                continue
            text = str(item.get("sentence_text", "")).strip()
            if not text:
                continue
            rows.append(Row(sentence_text=text, created_at=created_at, source="sentences"))
        if rows:
            return rows

    text = str(record.get("sentence_text", "")).strip()
    if text:
        rows.append(Row(sentence_text=text, created_at=created_at, source="sentence_text"))
        return rows

    raw_text = str(record.get("raw_text", "")).strip()
    language = str(record.get("language", "ko")).strip() or "ko"
    if raw_text:
        split_rows = split_into_sentences(raw_text, language=language)
        for s in split_rows:
            text = s.sentence_text.strip()
            if text:
                rows.append(Row(sentence_text=text, created_at=created_at, source="raw_text"))
    return rows


def _to_classified(texts: list[str]) -> list[ClassifiedSentenceDto]:
    return [
        ClassifiedSentenceDto(
            content_sentence_id=i + 1,
            sentence_text=text,
            sentence_order=i + 1,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=0.99,
        )
        for i, text in enumerate(texts)
    ]


def _autolabel(texts: list[str]) -> list[bool]:
    with patch("analysis_bc.tagger.span_tagger.SentenceTransformer", return_value=MagicMock()):
        tagger = SpanTagger()
    if not tagger._is_qdrant_healthy():
        raise RuntimeError("Qdrant 연결 실패: --autolabel 실행 불가")
    labels = tagger.tag(_to_classified(texts))
    pred_ids = {
        l.content_sentence_id
        for l in labels
        if l.label_type == SentenceLabelType.EMOTIONALLY_LOADED
    }
    return [(i + 1) in pred_ids for i in range(len(texts))]


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            rows.extend(_extract_rows_from_record(record))
    if not rows:
        raise ValueError(f"추출된 문장이 없습니다: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="운영 로그에서 emotion 평가셋 생성")
    parser.add_argument("--input", type=Path, required=True, help="운영 로그 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="평가셋 JSONL 출력 경로")
    parser.add_argument("--sample-size", type=int, default=400, help="샘플 문장 수 (기본 400)")
    parser.add_argument("--recent-days", type=int, default=7, help="최근 N일 필터 (기본 7)")
    parser.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    parser.add_argument("--autolabel", action="store_true", help="현재 모델로 임시 라벨 자동 채움")
    args = parser.parse_args()

    rows = load_rows(args.input)

    if args.recent_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.recent_days)
        recent = [r for r in rows if r.created_at is None or r.created_at >= cutoff]
        if recent:
            rows = recent

    if len(rows) > args.sample_size:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample_size]

    texts = [r.sentence_text for r in rows]
    auto = _autolabel(texts) if args.autolabel else [False] * len(texts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            payload = {
                "content_sentence_id": i,
                "sentence_text": row.sentence_text,
                "gold_emotion": bool(auto[i - 1]),
                "label_source": "auto" if args.autolabel else "manual_required",
                "source_type": row.source,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"입력 문장 수: {len(load_rows(args.input))}")
    print(f"샘플링 문장 수: {len(rows)}")
    print(f"출력: {args.output}")
    print(
        "안내: --autolabel 미사용 시 gold_emotion은 기본 false로 채워집니다. "
        "평가 전 반드시 수작업 라벨링으로 교정하세요."
    )


if __name__ == "__main__":
    main()
