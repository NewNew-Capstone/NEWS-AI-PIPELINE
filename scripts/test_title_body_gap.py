"""TitleBodyGapCalculator 결과 확인 스크립트.

실행:
    python scripts/test_title_body_gap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.schemas import SentenceInputDto
from analysis_bc.tagger.title_body_gap import TitleBodyGapCalculator


def _make_sentences(texts: list[str]) -> list[SentenceInputDto]:
    return [
        SentenceInputDto(
            content_sentence_id=i,
            sentence_text=t,
            sentence_order=i,
        )
        for i, t in enumerate(texts)
    ]


# ── 케이스 정의 ──────────────────────────────────────────────────────────────

CASES = [
    {
        "name": "일치 (gap 낮아야 함)",
        "title": "정부, 부동산 규제 완화 정책 발표",
        "body": [
            "정부가 오늘 부동산 규제를 완화하는 정책을 공식 발표했다.",
            "이번 정책은 취득세 인하와 대출 규제 완화를 주요 내용으로 담고 있다.",
            "국토교통부 장관은 이번 조치가 시장 안정에 기여할 것이라고 밝혔다.",
            "전문가들은 단기적으로 거래량이 증가할 것으로 전망했다.",
            "해당 정책은 다음 달부터 단계적으로 시행될 예정이다.",
        ],
    },
    {
        "name": "낚시성 제목 (gap 높아야 함)",
        "title": "충격! 아이돌 스타, 비밀 결혼 전격 발표",
        "body": [
            "최근 연예계에 다양한 이슈들이 쏟아지고 있다.",
            "전문가들은 엔터테인먼트 산업의 변화를 주목하고 있다.",
            "팬덤 문화가 글로벌하게 확산되면서 K-pop의 영향력이 커지고 있다.",
            "올해 음악 시장 규모는 전년 대비 15% 성장한 것으로 집계됐다.",
            "신인 아이돌 그룹들의 데뷔가 잇따르고 있다.",
        ],
    },
    {
        "name": "리드 일치, 결말 불일치 (buried lede 패턴)",
        "title": "북한, 미사일 발사 시험 단행",
        "body": [
            "북한이 오늘 오전 동해상으로 탄도미사일을 발사했다.",
            "군 당국은 미사일이 약 500km 비행 후 동해에 낙하했다고 밝혔다.",
            "이번 발사는 한미 연합훈련에 대한 반발로 분석된다.",
            "한편 국내 주식시장은 오늘 외국인 매수세에 힘입어 강세를 보였다.",
            "코스피는 전일 대비 1.2% 상승 마감했으며 반도체 업종이 주도했다.",
            "삼성전자와 SK하이닉스의 주가가 나란히 52주 신고가를 경신했다.",
        ],
    },
    {
        "name": "긴 기사 (50문장 — 512토큰 절단 버그 검증)",
        "title": "기후변화 대응 국제 협약 체결",
        "body": [f"기후변화 협약 관련 내용 문장 {i}번: 탄소 중립과 재생에너지 전환이 핵심 의제다." for i in range(50)],
    },
    {
        "name": "빈 본문 (fallback 검증)",
        "title": "제목만 있고 본문이 없는 경우",
        "body": [],
    },
    {
        "name": "문장 1개",
        "title": "삼성전자, 올해 영업이익 역대 최고치 경신",
        "body": ["삼성전자가 올해 3분기 영업이익이 역대 최고치를 기록했다고 발표했다."],
    },
]


# ── 실행 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("TitleBodyGapCalculator 초기화 중 (모델 로딩)...\n")
    calc = TitleBodyGapCalculator()
    print("모델 로딩 완료.\n")

    print("=" * 70)
    print("TitleBodyGap 계산 결과")
    print("=" * 70)

    for case in CASES:
        name: str = case["name"]
        title: str = case["title"]
        sentences = _make_sentences(case["body"])

        result = calc.calculate(title, sentences)

        print(f"\n[케이스] {name}")
        print(f"  제목    : {title}")
        print(f"  문장 수 : {len(sentences)}개")
        diff = result.gap_tail - result.gap_lead
        pattern = (
            "buried_lede ⚠" if diff > 0.4
            else "clickbait ⚠" if result.gap_lead > 0.4
            else "consistent ✓"
        )
        print(f"  gap_score : {result.gap_score:.4f}")
        print(f"  gap_lead  : {result.gap_lead:.4f}  (제목 vs 첫 3문장)")
        print(f"  gap_tail  : {result.gap_tail:.4f}  (제목 vs 마지막 3문장)")
        print(f"  gap_std   : {result.gap_std:.4f}")
        print(f"  pattern   : {pattern}  (tail-lead diff={diff:+.4f})")

    print("\n" + "=" * 70)
    print("완료")


if __name__ == "__main__":
    main()
