"""실제 모델 로드 후 문장 분류 수동 테스트 스크립트."""
from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.classifier import FactOpinionClassifier

TEST_SENTENCES = [
    "정부는 오늘 새로운 부동산 정책을 발표했다.",
    "이번 부동산 정책은 정말 최악의 선택이다.",
    "집값이 너무 올라서 서민들이 너무 힘들다.",
    "서울 아파트 평균 가격이 10억을 넘었다.",
    "한국은행이 기준금리를 0.25%p 인상했다.",
    "이 정도 금리 인상은 경기를 완전히 망칠 것이다.",
]


def main() -> None:
    print("모델 로드 중...", flush=True)
    clf = FactOpinionClassifier(model_path="analysis_bc/models/best_model")
    print("로드 완료\n")

    print(f"{'문장':<40} {'라벨':<15} {'신뢰도':>6}")
    print("-" * 65)
    for text in TEST_SENTENCES:
        label, confidence = clf.predict(text)
        marker = "FACT  " if label == "fact_like" else "OPNION"
        print(f"{text:<40} {marker:<15} {confidence:.4f}")


if __name__ == "__main__":
    main()
