# issue_comparison_bc/CLAUDE.md

> Issue Comparison BC 전용 규칙. 이 BC는 분석 결과를 묶고 비교 가능한 이슈 단위 결과로 조립한다.

## 1) 이 BC의 책임
- 분석 결과 수집
- 유사 이슈 클러스터링
- 대표 항목 선정
- 비교 결과 조립
- 필요 시 캐시 키 / 조회 최적화
- 다음 단계 또는 응답용 결과 생성

---

## 2) 이 BC에서 수정 가능한 파일
- `issue_comparison_bc/router.py`
- `issue_comparison_bc/service.py`
- `issue_comparison_bc/schemas.py`
- `issue_comparison_bc/models.py`
- `issue_comparison_bc/modules/*`
- `tests/unit/issue_comparison_*`
- `tests/integration/issue_comparison_*`

벡터 스토어 / 검색엔진 / 캐시 키 정책은 문서와 함께 변경한다.

---

## 3) 핵심 원칙
1. 클러스터링 로직과 대표 선정 로직은 분리한다.
2. 가중치 실험 전에 **표준값을 고정** 한다.
3. 대표 점수 계산은 재현 가능해야 한다.
4. 국가 fallback 정책은 코드/문서 둘 다 명시한다.
5. `service.py` 는 수집 → 클러스터링 → 대표 선정 → 응답 조립만 담당한다.

---

## 4) 권장 모듈 구조
```text
issue_comparison_bc/
├── CLAUDE.md
├── router.py
├── service.py
├── schemas.py
├── models.py
└── modules/
    ├── clusterer.py
    ├── representative_selector.py
    ├── comparison_builder.py
    ├── cache_key.py
    └── event_builder.py
```

### 파일 역할
- `clusterer.py`: 이슈 묶기 / 유사도 기준 적용
- `representative_selector.py`: 대표 항목 점수 계산 / 선택
- `comparison_builder.py`: 최종 비교 응답 DTO 조립
- `cache_key.py`: 캐시 키 생성 규칙
- `event_builder.py`: 필요 시 다음 단계 event payload 생성

---

## 5) 세션 운영 규칙
### 같이 묶을 수 있는 작업
- `cache_key.py` + 관련 test
- `comparison_builder.py` + schema test

### 분리 권장 작업
- `clusterer.py` 가중치 조정
- `representative_selector.py` 점수 공식 수정
- 벡터 검색 / 외부 저장소 연동

---

## 6) 입력 / 출력 계약
### 입력
- `AnalysisCompletedEvent`
- 분석 결과 리스트
- 임베딩 / 유사도 / 메타데이터

### 출력
- issue cluster
- representative item
- comparison response DTO
- 필요 시 `IssueClusterCreatedEvent`

---

## 7) 테스트 원칙
- unit test: 클러스터 점수 / 대표 점수 / fallback 규칙 검증
- integration test: 분석 결과 → 비교 결과 흐름 검증
- fixture 기반 재현 가능성 확보

### 최소 체크리스트
- [ ] 유사도 임계값 경계값 검증
- [ ] 대표 점수 tie-break 검증
- [ ] 국가 fallback 검증
- [ ] 빈 cluster 처리
- [ ] 캐시 키 안정성 검증
- [ ] 동일 입력 시 동일 결과 검증

---

## 8) Claude에게 잘 맞는 작업
- cluster score 함수 작성
- representative score 함수 작성
- comparison response DTO 작성
- cache key 생성 함수 작성
- fixture / mock 데이터 생성

## 9) 주의할 것
- 가중치 실험 전 표준값부터 고정
- fallback 정책을 암묵적으로 두지 말 것
- service.py에 점수 계산식을 직접 넣지 말 것
- 다른 BC 내부 구현을 직접 import 하지 말 것

---

## 10) 작업 시작 프롬프트 예시
```text
루트 CLAUDE.md와 issue_comparison_bc/CLAUDE.md를 읽어줘.
오늘 작업은 [clusterer/representative_selector/comparison_builder] 구현이야.
먼저 입력 스키마, 점수 공식, 테스트 fixture, 완료 기준을 Plan Mode로 보여줘.
service.py는 orchestration만 유지해줘.
```
