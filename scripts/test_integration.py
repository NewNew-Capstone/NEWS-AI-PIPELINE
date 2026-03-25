# scripts/test_integration.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.schemas import AnalyzeRequestDto, SentenceInputDto
from analysis_bc.service import AnalysisService

service = AnalysisService()

request = AnalyzeRequestDto(
    target_id=1,
    title="낙산공원 무너져 대참사 발생",
    target_type="YOUTUBE_VIDEO",
    transcript_id=1,
    country="KR",
    language="ko",
    sentences=[
        SentenceInputDto(content_sentence_id=1, sentence_text="정부는 오늘 새로운 부동산 정책을 발표했다.", sentence_order=1, start_time_ms=None, end_time_ms=None),
        SentenceInputDto(content_sentence_id=2, sentence_text="이번 부동산 정책은 정말 최악의 선택이다.", sentence_order=2, start_time_ms=None, end_time_ms=None),
        SentenceInputDto(content_sentence_id=3, sentence_text="끔찍한 사건이 발생했다.", sentence_order=3, start_time_ms=None, end_time_ms=None),
        SentenceInputDto(content_sentence_id=4, sentence_text="관계자에 따르면 이번 사태는 심각하다.", sentence_order=4, start_time_ms=None, end_time_ms=None),
        SentenceInputDto(content_sentence_id=5, sentence_text="서울 아파트 평균 가격이 10억을 넘었다.", sentence_order=5, start_time_ms=None, end_time_ms=None),
    ]
)

print("===== 분석 시작 =====")
result = service.analyze(request)

print("\n[1단계 — 전처리]")
print(f"문장 수: {len(request.sentences)}")

print("\n[2단계 — FACT/OPINION 분류]")
classified = service.classifier.classify(
    service.prepare_sentences(request.sentences, request.language)
)
for s in classified:
    print(f"  [{s.label} / {s.confidence:.4f}] {s.sentence_text}")

print("\n[3단계 — 태깅]")
print(f"  span_labels 수: {len(result.sentence_labels)}")
for span in result.sentence_labels:
    print(f"  [{span.label_type}] matched: {span.matched_word} / offset: {span.start_offset}~{span.end_offset}")

print("\n[제목-본문 갭]")
print(f"  headline_body_gap_score: {result.headline_body_gap_score}")

print("\n===== 완료 =====")