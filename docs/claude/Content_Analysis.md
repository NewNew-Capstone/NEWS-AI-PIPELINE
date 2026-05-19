# 영상 자막 분석 로직 (AS-IS 코드 기준)

이 문서는 `analysis_bc`의 **현재 코드 동작**을 기준으로 작성한 최종 기준 문서다.
과거 설명(예: OPINION 세분화 LLM 태깅, `subjectivity_score` 0~100)은 본 기준에서 제외한다.

## 1) 입력/진입점

### `POST /analyze`
- 입력: `AnalyzeRequestDto`
- 핵심 필드
  - `target_id`, `title`, `language`
  - `sentences: list[SentenceInputDto]`
  - optional: `target_type`, `transcript_id`, `country`

### `POST /analyze/raw`
- 입력: `AnalyzeRawTextRequestDto`
- `raw_text`를 `split_into_sentences(text, language)`로 문장 분리한 뒤,
  내부적으로 `/analyze`와 동일한 분석 파이프라인을 수행한다.

## 2) 전처리

`SentencePreprocessor(expected_language=language).preprocess(sentences)`

- 공백 정규화: 연속 공백을 1개 공백으로 축약
- 길이 필터: `MIN_CHAR_LENGTH = 5` 미만 제거
- 언어 필터: `langdetect` 결과가 요청 `language`와 불일치 시 제거
- 처리 순서: `sentence_order` 기준 정렬 후 필터링

### 한국어 raw 문장 분리 규칙

`split_into_sentences(..., "ko")`는 한국어 전용 정규식/휴리스틱을 사용한다.

- 기본 분리: 종결부호(`.!?`) + 한국어 종결형(다/요/죠/네/까) 뒤 공백
- 긴 문장 분할: 길이 임계(`KO_LONG_CHUNK_THRESHOLD=120`) 초과 시
  - 전환어(예: "하지만", "그러나", "반면" 등) 기준 우선 분할
  - 실패 시 공백 기반 길이 fallback 분할

## 3) 분석 파이프라인

`AnalysisService.analyze()`의 실제 처리 순서:

1. 전처리
2. FACT/OPINION 분류
3. 감정 span 태깅 (OPINION 대상)
4. 제목-본문 갭 계산
5. 점수 계산
6. FACT 상위 문장 추출
7. 요약/점수설명 생성
8. 키워드 추출
9. 근거(evidence) 추출
10. 최종 DTO 반환

### 3.1 FACT/OPINION 분류

- 컴포넌트: `FactOpinionClassifier`
- 모델: `analysis_bc/models/best_model` (Electra sequence classification)
- 출력 라벨: `fact_like`, `opinion_like`
- 각 문장별 `confidence` 포함

### 3.2 감정 span 태깅

- 컴포넌트: `SpanTagger`
- 대상: `opinion_like` 문장만
- 방식
  - Kiwi 형태소 분석
  - 유효 품사(`NNG`, `NNP`, `VV`, `VA`, `MAG`, `XR`) 토큰 선택
  - FastText 임베딩(로컬 모델 또는 FastText 서버)
  - Qdrant `emotion_words` 컬렉션 유사도 검색
- 산출: `SpanLabelDto` 리스트 (`label_type=EMOTIONALLY_LOADED`)
- 예외 처리: Qdrant health check 실패 시 span 결과는 빈 리스트

### 3.3 제목-본문 갭 계산

- 컴포넌트: `TitleBodyGapCalculator`
- 모델: `jhgan/ko-sroberta-multitask`
- 지표
  - `headline_body_gap_score` (`gap_score`)
  - `headline_body_gap_std`
  - `headline_body_gap_lead`
  - `headline_body_gap_tail`
  - `headline_body_gap_label` (`trustworthy`/`neutral`/`clickbait`/`buried_lede`)
- 참고: 이 지표는 최종 편향 점수 산식에 **직접 합산되지 않는 별도 보조 지표**다.

### 3.4 점수 산식 (0~1)

`BiasScorer.calculate()`

- `opinion_score = opinion_like 문장 수 / 전체 문장 수`
- `fact_ratio = fact_like 문장 수 / 전체 문장 수`
- `emotion_score`
  - 감정 span 점수를 문장 단위로 집계(문장당 최대 1.0)
  - 문서 위치 가중(앞 1.3 / 중간 1.0 / 뒤 0.8) 적용
  - 전체 문장 수로 정규화, 최대 1.0
- 최종식

```text
overall_bias_score = 0.4 * opinion_score
                   + 0.3 * emotion_score
                   + 0.3 * (1 - fact_ratio)
```

- 반환 값은 소수 4자리 반올림

### 3.5 요약 및 점수 설명

- FACT 상위 `confidence` 10개 문장을 요약 입력으로 사용
- `BiasSummarizer.summarize()`
  - 출력: `summary_text`
- `BiasSummarizer.summarize_score_reason()`
  - 산식/근거 기반 설명 생성
  - 출력: `score_reason_summary`

### 3.6 키워드

`KeywordExtractor.extract()`

- `EMOTION`: 감정 span 기반
- `FRAME`: opinion_like 문장에서 형태소 추출
- `TOPIC`: fact_like 문장에서 형태소 추출
- 각 타입별 상위 Top-N(기본 5) 반환

### 3.7 근거(evidences)

`EvidenceExtractor.extract()`

- 감정 span → `EvidenceType.EMOTION`
- `opinion_like` 분류 문장 → `EvidenceType.OPINION`
- 결과 DTO: `content_sentence_id`, `title`, `description`, `source_text`, `confidence_score`

## 4) 출력 계약 (Response Schema)

`BiasAnalysisResultDto` 핵심 필드:

- 점수
  - `overall_bias_score`
  - `opinion_score`
  - `emotion_score`
  - `fact_ratio`
- 보조 지표
  - `headline_body_gap_score`
  - `headline_body_gap_std`
  - `headline_body_gap_lead`
  - `headline_body_gap_tail`
  - `headline_body_gap_label`
- 설명/근거
  - `summary_text`
  - `score_reason_summary`
  - `score_evidence`
- 증거 데이터
  - `sentence_labels`
  - `keywords`
  - `emotion_keywords`
  - `evidences`

`/analyze/raw` 응답은 위 필드에 `sentences`를 추가한 `RawAnalysisResultDto`를 반환한다.

## 5) 테스트 기준 (현재 테스트 축)

현재 코드베이스에서 아래 축의 단위 테스트가 존재한다.

1. 전처리/한국어 분리
   - `tests/analysis_bc/test_preprocessor.py`
   - `tests/analysis_bc/test_korean_chunking.py`

2. scorer 산식/가중치/경계
   - `tests/analysis_bc/test_scorer.py`
   - `tests/analysis_bc/test_ab_weights.py`

3. span tagger 정상/장애 경로
   - `tests/analysis_bc/tagger/test_span_tagger.py`

4. title-body gap 라벨 분기
   - `tests/analysis_bc/tagger/test_title_body_gap.py`

5. API/스키마 계약
   - `tests/unit/test_analysis_router_logging.py`
   - `tests/unit/test_analysis_schemas.py`
   - `tests/unit/test_analysis_service_consumer.py`

---

## 운영 가정

- 정리 범위는 **AS-IS 코드 기준**으로 고정한다.
- 외부 의존(Anthropic, Qdrant, FastText 서버/모델 경로)은 현 설정을 유지한다.
- 과거 표현인 `subjectivity_score(0~100)` 체계는 사용하지 않고,
  현재 `overall_bias_score(0~1)` 체계를 기준으로 한다.
