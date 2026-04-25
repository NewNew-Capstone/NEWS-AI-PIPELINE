"""SpanTagger 실제 태깅 결과 확인 스크립트.

실행:
    python scripts/test_tagger.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.tagger.span_tagger import SpanTagger

# 테스트 문장 (익명/감정/중립 혼합)
TEST_SENTENCES = [
    # 익명 출처 — 정확히 잡혀야 함
    "관계자에 따르면 이번 사태는 매우 심각한 상황이다.",
    "복수의 소식통에 의하면 협상이 결렬될 가능성이 높다.",
    "업계에서는 이번 결정이 시장에 큰 영향을 줄 것으로 보고 있다.",

    # 익명 출처 아님 — 잡히면 false positive
    "전문가들이 이 사안을 면밀히 분석했다.",
    "정부는 오늘 새로운 부동산 정책을 발표했다.",
    "서울 아파트 평균 가격이 10억을 넘었다.",

    # 감정적 표현 — 잡혀야 함
    "이번 정책은 정말 최악의 선택이다.",
    "폭발적인 분노가 시민들 사이에서 번지고 있다.",

    # 감정 표현 아님 — 잡히면 false positive
    "국회는 오늘 예산안을 처리했다.",
    "해당 법안은 내년 1월부터 시행된다.",
]


def _make_sentence(idx: int, text: str) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=idx,
        sentence_text=text,
        sentence_order=idx,
        start_time_ms=None,
        end_time_ms=None,
        label="opinion_like",
        confidence=0.9,
    )


def main() -> None:
    print("SpanTagger 초기화 중...\n")
    tagger = SpanTagger()

    sentences = [_make_sentence(i, t) for i, t in enumerate(TEST_SENTENCES)]
    labels = tagger.tag(sentences)

    label_map: dict[int, list] = {}
    for label in labels:
        label_map.setdefault(label.content_sentence_id, []).append(label)

    print("\n" + "=" * 70)
    print("태깅 결과")
    print("=" * 70)

    for i, text in enumerate(TEST_SENTENCES):
        tags = label_map.get(i, [])
        print(f"\n[{i}] {text}")
        if not tags:
            print("  → 태깅 없음")
        for t in tags:
            print(f"  → {t.label_type} | score={t.score:.4f} | matched={t.matched_word!r}")

    print("\n" + "=" * 70)
    print(f"전체 문장: {len(TEST_SENTENCES)}개 | 태깅된 문장: {len(label_map)}개 | 총 태그: {len(labels)}개")


if __name__ == "__main__":
    main()
