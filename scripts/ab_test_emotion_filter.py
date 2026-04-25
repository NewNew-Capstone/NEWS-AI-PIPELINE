"""Emotion 태깅 판정 로직 A/B 비교 스크립트.

실행:
    .venv/bin/python scripts/ab_test_emotion_filter.py

비교 조건:
    A(기존): threshold=0.80, top_k=1, margin=0.00
    B(개선): threshold=0.88, top_k=3, margin=0.06
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.classifier import ClassifiedSentenceDto
import analysis_bc.tagger.span_tagger as span_mod
from analysis_bc.tagger.span_tagger import SpanTagger


@dataclass(frozen=True)
class Case:
    text: str
    token: str
    scores: tuple[float, ...]
    is_emotion: bool


CASES: list[Case] = [
    Case("이번 정책은 정말 최악의 선택이다.", "최악", (0.95, 0.83, 0.81), True),
    Case("폭발적인 분노가 시민들 사이에서 번지고 있다.", "분노", (0.94, 0.84, 0.80), True),
    Case("그의 발언은 매우 충격적이었다.", "충격", (0.91, 0.84, 0.83), True),
    Case("유가 하락으로 시장이 안정세를 보였다.", "시장", (0.86, 0.84, 0.82), False),
    Case("정부는 오늘 새로운 정책을 발표했다.", "정부", (0.84, 0.83, 0.81), False),
    Case("국회는 오늘 예산안을 처리했다.", "국회", (0.82, 0.81, 0.80), False),
    Case("서울 아파트 평균 가격이 10억을 넘었다.", "평균", (0.81, 0.80, 0.79), False),
    Case("업계는 올해 실적 개선을 기대하고 있다.", "기대", (0.90, 0.88, 0.87), False),
    Case("여론은 분개했고 강한 반발이 이어졌다.", "분개", (0.93, 0.84, 0.80), True),
    Case("관계자는 추가 조사를 진행 중이라고 밝혔다.", "조사", (0.85, 0.84, 0.83), False),
]


def _make_sentence(case_id: int, text: str) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=case_id,
        sentence_text=text,
        sentence_order=case_id,
        start_time_ms=None,
        end_time_ms=None,
        label="opinion_like",
        confidence=0.9,
    )


def _fake_qdrant_result(scores: tuple[float, ...], threshold: float, limit: int):
    hits = []
    for score in scores:
        if score < threshold:
            continue
        hits.append(
            SimpleNamespace(
                score=score,
                payload={"word": "dummy", "word_root": "dummy", "polarity": "-1"},
            )
        )
    return SimpleNamespace(points=hits[:limit])


def _build_tagger() -> SpanTagger:
    with (
        patch("analysis_bc.tagger.span_tagger.Kiwi") as mock_kiwi_cls,
        patch("analysis_bc.tagger.span_tagger.SentenceTransformer") as mock_sbert_cls,
        patch("analysis_bc.tagger.span_tagger.fasttext") as mock_ft_module,
        patch("analysis_bc.tagger.span_tagger.QdrantClient") as mock_qdrant_cls,
    ):
        mock_kiwi_cls.return_value = MagicMock()
        mock_sbert_cls.return_value = MagicMock()

        mock_ft = MagicMock()
        mock_ft.get_word_vector.return_value = np.ones(300)
        mock_ft_module.load_model.return_value = mock_ft
        mock_qdrant_cls.return_value = MagicMock()

        tagger = SpanTagger()
        tagger._qdrant_healthy = True
        tagger._health_checked_at = float("inf")
    return tagger


def _evaluate(*, threshold: float, top_k: int, margin: float) -> dict[str, float]:
    old_threshold = span_mod.EMOTION_SIMILARITY_THRESHOLD
    old_top_k = span_mod.EMOTION_TOP_K
    old_margin = span_mod.EMOTION_SCORE_MARGIN

    span_mod.EMOTION_SIMILARITY_THRESHOLD = threshold
    span_mod.EMOTION_TOP_K = top_k
    span_mod.EMOTION_SCORE_MARGIN = margin

    try:
        tagger = _build_tagger()
        tp = fp = fn = tn = 0

        for idx, case in enumerate(CASES):
            sentence = _make_sentence(idx, case.text)
            token_start = case.text.find(case.token)
            token = SimpleNamespace(
                form=case.token,
                tag="NNG",
                start=token_start,
                len=len(case.token),
            )

            def _side_effect(collection_name: str, requests: list) -> list:
                req = requests[0]
                return [_fake_qdrant_result(case.scores, req.score_threshold, req.limit)]

            tagger.qdrant.query_batch_points.side_effect = _side_effect
            spans = tagger._search_emotion(sentence, [token])
            pred = bool(spans)
            gold = case.is_emotion

            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif not pred and gold:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    finally:
        span_mod.EMOTION_SIMILARITY_THRESHOLD = old_threshold
        span_mod.EMOTION_TOP_K = old_top_k
        span_mod.EMOTION_SCORE_MARGIN = old_margin


def _print_result(name: str, result: dict[str, float]) -> None:
    print(f"[{name}]")
    print(
        f"TP={int(result['tp'])} FP={int(result['fp'])} FN={int(result['fn'])} TN={int(result['tn'])}"
    )
    print(
        "precision={:.4f} recall={:.4f} f1={:.4f}".format(
            result["precision"],
            result["recall"],
            result["f1"],
        )
    )
    print()


def main() -> None:
    print("Emotion 판정 A/B 테스트 시작\n")
    a = _evaluate(threshold=0.80, top_k=1, margin=0.00)
    b = _evaluate(threshold=0.88, top_k=3, margin=0.06)

    _print_result("A - Legacy", a)
    _print_result("B - Improved", b)

    print("[Delta B-A]")
    print(
        "FP {} -> {} ({:+d})".format(
            int(a["fp"]), int(b["fp"]), int(b["fp"] - a["fp"])
        )
    )
    print(
        "precision {:.4f} -> {:.4f} ({:+.4f})".format(
            a["precision"], b["precision"], b["precision"] - a["precision"]
        )
    )
    print(
        "recall {:.4f} -> {:.4f} ({:+.4f})".format(
            a["recall"], b["recall"], b["recall"] - a["recall"]
        )
    )
    print(
        "f1 {:.4f} -> {:.4f} ({:+.4f})".format(
            a["f1"], b["f1"], b["f1"] - a["f1"]
        )
    )


if __name__ == "__main__":
    main()
