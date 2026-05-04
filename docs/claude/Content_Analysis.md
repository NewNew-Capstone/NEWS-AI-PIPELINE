```
[입력]
자막(transcript) + 문장 분리(sentences) + 메타데이터
        |
        v
[1단계] 전처리
- 언어 감지
- 문장 정제/정규화
- 짧은 문장 필터링
        |
        v
[2단계] FACT / OPINION 분류
- 파인튜닝 모델 사용
- 문장별 라벨 + confidence score([FACT: 0.92, OPINION: 0.08]) 반환
        |
        v
[3단계] OPINION 세분화 (LLM)
- OPINION 문장만 대상
- OPINION / EMOTIONAL / UNVERIFIABLE 세분류
- span 단위 원인 구간 추출
  (start_offset, end_offset, label_type)
→ 하이라이팅 데이터 완성
→ 라벨별 비율 집계
  (span 길이 합 / 전체 길이)
        |
        v
[4단계] FACT 문장 필터링
- confidence score 기준 상위 필터링
- 요약 입력용 FACT 문장 추출
        |
        +------------------+------------------+
        |                  |                  |
        v                  v                  |
   [5단계]             [6단계]            [3단계 완료]
	   사실 기반 요약      주관성 점수 계산    span_labels
   생성 (LLM-Claude Sonnet.2026편향분류성능 1위(논문))
	                      ① Frequency Score      → 그래프 데이터
	   FACT 문장 →          frel = OPINION 수      (OPINION %)
	   summary 텍스트              / 전체 문장 수     (EMOTIONAL %)
	                                              (UNVERIFIABLE %)
                      ② Position Weight(논문기반)
                        앞 33%: × 1.3       → 하이라이팅 데이터
                        중 33%: × 1.0
                        뒤 33%: × 0.8

                      ③ Confidence Weight
                        sentence_score(i)
                        = confidence(i)
                          × position_weight(i)

                      ④ 최종 점수
                        subjectivity_score = Σsentence_score
                                / 전체 문장 수
                                × 100

                      ⑤ score_evidence 생성
                        "전체 중 X%가 주관적"
                        "도입부에 집중" 등
        |                  |                  |
        +------------------+------------------+
                           |
                           v
[출력]
┌──────────────────────────────────────────────────┐
│ summary           → 사실 기반 요약 텍스트         │
│ subjectivity_score → 주관성 점수 (0~100)          │
│ score_evidence    → 점수 산출 근거 텍스트         │
│ span_labels       → 하이라이팅 데이터             │
│                     (start_offset, end_offset,    │
│                      label_type)                  │
│ bias_type_scores  → 그래프 데이터                 │
│                     (OPINION % / EMOTIONAL %      │
│                      / UNVERIFIABLE %)            │
└──────────────────────────────────────────────────┘
```

1. 자막(transcript) -> 문장 분리(sentences)

2. 문장이 분리된 후 문장별 파인튜닝 모델을 사용하여 FACT/OPINION으로 분리함 

3. FACT 문장은 객관적 요약을 하기위해 클로드 api키를 활용해서 FACT문장으로 요약문을 생성 (summary 생성)

4. OPINION 문장을 청킹하여 EMOTIONAL, OPINION,UNVERIFIABLE 으로  태깅
5. 청킹 후 태깅된 단어들을 이용하여 subjectivity_score(중립화 점수) 도출 (가중치와 합산식 사용)





