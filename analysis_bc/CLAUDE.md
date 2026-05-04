# analysis_bc/CLAUDE.md

> Analysis BC 전용 규칙. 이 BC는 입력 콘텐츠를 분석하여 분류/세분화/요약/점수화하고, 다음 단계에 전달 가능한 결과를 만든다.

## 1) 이 BC의 책임
- 전처리
- FACT / OPINION 분류
- OPINION 세분화 또는 span 추출
- FACT 필터링
- 요약
- 점수 계산
- 분석 결과 응답 또는 Event payload 조립

---

## 2) 이 BC에서 수정 가능한 파일
- `analysis_bc/router.py`
- `analysis_bc/service.py`
- `analysis_bc/schemas.py`
- `analysis_bc/models.py`
- `analysis_bc/modules/*`
- `tests/unit/analysis_*`
- `tests/integration/analysis_*`

공통 스키마 변경이 필요하면 단독으로 바꾸지 말고 먼저 합의 후 진행한다.

---

## 3) 핵심 원칙
1. `service.py` 는 **순차 orchestration만** 담당
2. 분류 로직, span 추출, 점수 계산은 각각 **모듈 분리**
3. **scorer에 LLM 로직 금지**
4. `segmenter` 와 `summarizer` 는 **서로 다른 세션** 에서 작업
5. 실제 API/모델 호출은 integration 레벨 또는 mock 분리로 관리
6. fallback / timeout / empty input 정책을 코드와 테스트에 명시

---

## 4) 권장 모듈 구조
```text
analysis_bc/
├── CLAUDE.md
├── router.py
├── service.py
├── schemas.py
├── models.py
└── modules/
    ├── preprocessor.py
    ├── classifier.py
    ├── segmenter.py
    ├── fact_filter.py
    ├── summarizer.py
    └── scorer.py
```

### 파일 역할
- `preprocessor.py`: 정제 / 언어 감지 / 기본 normalize
- `classifier.py`: FACT / OPINION 분류만
- `segmenter.py`: OPINION span / label 추출만
- `fact_filter.py`: FACT 제거 또는 선택만
- `summarizer.py`: 요약만
- `scorer.py`: 수식 기반 점수 계산만
- `service.py`: 위 모듈 순차 호출만

---

## 5) 세션 운영 규칙
### 한 세션에 같이 묶어도 되는 예
- `preprocessor.py` + 관련 unit test
- `classifier.py` + fixture + unit test
- `scorer.py` + formula test

### 세션 분리 필수 예
- `segmenter.py` 작업
- `summarizer.py` 작업
- 실제 LLM prompt 수정 작업
- 대량 로그 디버깅 작업

### 권장 순서
1. 스키마 확인
2. 테스트 초안 작성
3. 구현
4. mock 기준 통과
5. integration 확인
6. TODO 업데이트

---

## 6) 입력 / 출력 계약
### 입력 예시
- `ContentPreparedEvent`
- 정규화된 문장 리스트
- 문장 인덱스 / 메타데이터 / source 정보

### 출력 예시
- 분석 완료 DTO
- `AnalysisCompletedEvent`
- sentence-level labels / spans / summary / score evidence

스키마가 아직 확정되지 않았으면 mock payload를 먼저 만들고 진행한다.

---

## 7) 테스트 원칙
- unit test: 분류 / 점수 / span 추출 규칙 검증
- integration test: 전체 분석 흐름 검증
- LLM 호출 모듈: mock 우선
- 실제 외부 호출은 `tests/integration/` 에서만 허용

### 최소 체크리스트
- [ ] empty input 처리
- [ ] 문장 1개 케이스
- [ ] 전부 FACT 케이스
- [ ] 전부 OPINION 케이스
- [ ] span offset 검증
- [ ] summary 없을 때 fallback 검증
- [ ] score evidence 검증

---

## 8) Claude에게 잘 맞는 작업
- preprocessor / classifier wrapper 작성
- span schema 정리
- formula 기반 scorer 구현
- mock test / fixture 생성
- integration test 초안 생성

## 9) 주의할 것
- scorer에 프롬프트 로직 넣지 말 것
- segmenter와 summarizer를 한 세션에서 동시에 건드리지 말 것
- service.py에 계산식 / 정규식 / 장문 prompt를 넣지 말 것
- 다른 BC 구현을 직접 import 하지 말 것

---

## 10) 작업 시작 프롬프트 예시
```text
루트 CLAUDE.md와 analysis_bc/CLAUDE.md를 읽어줘.
오늘 작업은 [preprocessor/classifier/segmenter/summarizer/scorer] 구현이야.
먼저 수정할 파일, 테스트 범위, mock 필요 여부, 완료 기준을 Plan Mode로 보여줘.
service.py는 orchestration만 유지해줘.
```
