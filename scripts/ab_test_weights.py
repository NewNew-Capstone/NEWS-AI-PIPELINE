"""
A/B 테스트 스크립트: scorer 가중치 변형 비교

실행: python scripts/ab_test_weights.py
외부 의존성(ML 모델, Qdrant, API) 없이 동작.

A.json — 극단 Opinion (사설 5건): opinion_like fixture + emotion spans
B.json — 극단 Fact  (보도자료 5건): fact_like fixture, emotion spans 없음
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.scorer import BiasScorer, ScorerWeights

# ------------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "analysis_bc" / "data"

_GAP_A = 0.8   # A타입(사설): 제목-본문 의미 갭 높음
_GAP_B = 0.1   # B타입(보도자료): 갭 낮음
_CONFIDENCE = 0.95

# ------------------------------------------------------------------
# weight variants
# ------------------------------------------------------------------
WEIGHT_VARIANTS: dict[str, ScorerWeights] = {
    "default":       ScorerWeights(),
    "emotion_heavy": ScorerWeights(w_opinion=0.4, w_emotion=0.6),
    "position_flat": ScorerWeights(
        w_opinion=0.7,
        w_emotion=0.3,
        position_weight_front=1.0,
        position_weight_mid=1.0,
        position_weight_back=1.0,
    ),
}

# ------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _sentence_count(raw_text: str) -> int:
    """마침표/물음표/느낌표 기준 문장 수 추정."""
    parts = re.split(r"[.!?。]\s*", raw_text.strip())
    return max(len([p for p in parts if p.strip()]), 1)


# ------------------------------------------------------------------
# fixture 생성
# ------------------------------------------------------------------

def _make_opinion_fixture(
    n: int,
) -> tuple[list[ClassifiedSentenceDto], list[SpanLabelDto]]:
    """A타입: 모든 문장 opinion_like + 모든 문장에 감정 span 1개."""
    classified = [
        ClassifiedSentenceDto(
            content_sentence_id=i,
            sentence_text=f"의견문장{i}",
            sentence_order=i,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=_CONFIDENCE,
        )
        for i in range(1, n + 1)
    ]
    spans = [
        SpanLabelDto(
            content_sentence_id=i,
            start_offset=0,
            end_offset=5,
            label_type=SentenceLabelType.EMOTIONALLY_LOADED,
            score=0.95,
        )
        for i in range(1, n + 1)
    ]
    return classified, spans


def _make_fact_fixture(
    n: int,
) -> tuple[list[ClassifiedSentenceDto], list[SpanLabelDto]]:
    """B타입: 모든 문장 fact_like, emotion span 없음."""
    classified = [
        ClassifiedSentenceDto(
            content_sentence_id=i,
            sentence_text=f"사실문장{i}",
            sentence_order=i,
            start_time_ms=None,
            end_time_ms=None,
            label="fact_like",
            confidence=_CONFIDENCE,
        )
        for i in range(1, n + 1)
    ]
    return classified, []


# ------------------------------------------------------------------
# 결과 컨테이너
# ------------------------------------------------------------------

@dataclass
class RunResult:
    dataset: str
    article_id: int
    title_preview: str
    weight_variant: str
    overall: float
    opinion: float
    emotion: float
    subj: float


# ------------------------------------------------------------------
# 실행
# ------------------------------------------------------------------

def _run_all() -> list[RunResult]:
    a_articles = _load(DATA_DIR / "A.json")
    b_articles = _load(DATA_DIR / "B.json")

    results: list[RunResult] = []

    for article in a_articles:
        n = _sentence_count(article.get("raw_text", ""))
        classified, spans = _make_opinion_fixture(n)
        for variant_name, weights in WEIGHT_VARIANTS.items():
            scores = BiasScorer(weights).calculate(
                classified=classified,
                span_labels=spans,
                headline_body_gap=_GAP_A,
            )
            results.append(RunResult(
                dataset="A_opinion",
                article_id=article["target_id"],
                title_preview=article.get("title", "")[:20],
                weight_variant=variant_name,
                overall=scores["overall_bias_score"],
                opinion=scores["opinion_score"],
                emotion=scores["emotion_score"],
                subj=scores["subjectivity_score"],
            ))

    for article in b_articles:
        n = _sentence_count(article.get("raw_text", ""))
        classified, spans = _make_fact_fixture(n)
        for variant_name, weights in WEIGHT_VARIANTS.items():
            scores = BiasScorer(weights).calculate(
                classified=classified,
                span_labels=spans,
                headline_body_gap=_GAP_B,
            )
            results.append(RunResult(
                dataset="B_fact",
                article_id=article["target_id"],
                title_preview=article.get("title", "")[:20],
                weight_variant=variant_name,
                overall=scores["overall_bias_score"],
                opinion=scores["opinion_score"],
                emotion=scores["emotion_score"],
                subj=scores["subjectivity_score"],
            ))

    return results


# ------------------------------------------------------------------
# 리포트 출력
# ------------------------------------------------------------------

def _print_table(results: list[RunResult]) -> None:
    header = (
        f"{'dataset':<12} {'id':>4} {'title':<22} "
        f"{'variant':<14} {'overall':>7} {'opinion':>7} "
        f"{'emotion':>7} {'subj':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    prev_variant = None
    for r in results:
        if prev_variant and prev_variant != r.weight_variant:
            print()
        print(
            f"{r.dataset:<12} {r.article_id:>4} {r.title_preview:<22} "
            f"{r.weight_variant:<14} {r.overall:>7.4f} {r.opinion:>7.4f} "
            f"{r.emotion:>7.4f} {r.subj:>7.2f}"
        )
        prev_variant = r.weight_variant


def _print_summary(results: list[RunResult]) -> None:
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)

    for variant_name in WEIGHT_VARIANTS:
        a_scores = [r.overall for r in results if r.dataset == "A_opinion" and r.weight_variant == variant_name]
        b_scores = [r.overall for r in results if r.dataset == "B_fact"    and r.weight_variant == variant_name]

        mean_a = sum(a_scores) / len(a_scores)
        mean_b = sum(b_scores) / len(b_scores)

        var_a = sum((x - mean_a) ** 2 for x in a_scores) / len(a_scores)
        var_b = sum((x - mean_b) ** 2 for x in b_scores) / len(b_scores)

        denom = var_a + var_b
        fisher = (mean_a - mean_b) ** 2 / denom if denom > 1e-9 else float("inf")

        sep_ok = "O" if mean_a > mean_b else "X"
        print(
            f"  [{variant_name:<14}] "
            f"A평균={mean_a:.4f}  B평균={mean_b:.4f}  "
            f"차이={mean_a - mean_b:.4f}  Fisher={fisher:.2f}  분리={sep_ok}"
        )


def main() -> None:
    print(f"\n{'=' * 60}")
    print("A/B 가중치 비교 리포트")
    print(f"  A.json : 극단 Opinion (사설, gap={_GAP_A})")
    print(f"  B.json : 극단 Fact  (보도자료, gap={_GAP_B})")
    print(f"  가중치 variants: {list(WEIGHT_VARIANTS.keys())}")
    print(f"{'=' * 60}\n")

    results = _run_all()
    _print_table(results)
    _print_summary(results)


if __name__ == "__main__":
    main()
