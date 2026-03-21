# personalization_bc/CLAUDE.md

> Personalization BC 전용 규칙. 이 BC는 사용자 스크랩, 추천, 마이페이지 조회를 담당한다.

## 1) 이 BC의 책임
- 비교 결과 / 영상 스크랩 저장
- 추천 그룹 / 추천 아이템 생성
- 추천 사유 조립
- 마이페이지용 조회 응답 생성
- 사용자 기준 필터링 / 경계 검증

---

## 2) 이 BC에서 수정 가능한 파일
- `personalization_bc/router.py`
- `personalization_bc/service.py`
- `personalization_bc/schemas.py`
- `personalization_bc/models.py`
- `personalization_bc/modules/*`
- `tests/unit/personalization_*`
- `tests/integration/personalization_*`

사용자 식별 / 접근권한 / 소유권 규칙은 반드시 테스트로 보강한다.

---

## 3) 핵심 원칙
1. 스크랩과 추천은 분리한다.
2. 비교 결과 스크랩과 영상 스크랩은 **type 분리** 한다.
3. 추천 사유는 추적 가능해야 한다.
4. `user_id` 경계 검증은 필수다.
5. `service.py` 는 스크랩 처리 → 추천 생성 → 응답 조립만 담당한다.

---

## 4) 권장 모듈 구조
```text
personalization_bc/
├── CLAUDE.md
├── router.py
├── service.py
├── schemas.py
├── models.py
└── modules/
    ├── scrap_service.py
    ├── recommendation_engine.py
    ├── reason_builder.py
    ├── mypage_query.py
    └── event_consumer.py
```

### 파일 역할
- `scrap_service.py`: 스크랩 CRUD
- `recommendation_engine.py`: 추천 생성
- `reason_builder.py`: 추천 사유 템플릿 / 근거 조립
- `mypage_query.py`: 마이페이지 조회 전용 조립
- `event_consumer.py`: 상위 단계 event 또는 결과 수신 처리

---

## 5) 세션 운영 규칙
### 같이 작업해도 되는 묶음
- `scrap_service.py` + schema + unit test
- `reason_builder.py` + fixture + test

### 분리 권장 작업
- 추천 엔진 로직 수정
- 사용자 경계/권한 로직 수정
- 마이페이지 조회 최적화

---

## 6) 입력 / 출력 계약
### 입력
- 사용자 ID
- issue cluster / representative item / content item
- scrap action event 또는 조회 조건

### 출력
- scrap 상태
- recommendation group / item
- recommendation reason
- mypage response DTO

---

## 7) 테스트 원칙
- unit test: scrap 타입 분리 / 추천 사유 / user scope 검증
- integration test: 사용자 액션 → 추천 결과 흐름 검증
- 권한 / 소유권 / 빈 데이터 응답을 반드시 포함

### 최소 체크리스트
- [ ] user_id 경계 검증
- [ ] ISSUE_CLUSTER / YOUTUBE_VIDEO type 분리
- [ ] 중복 스크랩 처리
- [ ] 추천 사유 비어있지 않음
- [ ] 마이페이지 빈 상태 응답 검증
- [ ] 타 사용자 데이터 노출 차단 검증

---

## 8) Claude에게 잘 맞는 작업
- recommendation item schema 작성
- scrap CRUD 구현
- 추천 사유 템플릿 작성
- 마이페이지 응답 DTO 정리
- fixture / mock 데이터 생성

## 9) 주의할 것
- user_id 경계 검증 누락 금지
- 비교 결과 스크랩과 영상 스크랩을 같은 타입으로 처리하지 말 것
- 추천 사유를 service.py에 직접 하드코딩하지 말 것
- 다른 BC 내부 구현을 직접 import 하지 말 것

---

## 10) 작업 시작 프롬프트 예시
```text
루트 CLAUDE.md와 personalization_bc/CLAUDE.md를 읽어줘.
오늘 작업은 [scrap/recommendation/reason_builder/mypage_query] 구현이야.
먼저 입력 스키마, user_id 경계, 테스트 범위, 완료 기준을 Plan Mode로 보여줘.
service.py는 orchestration만 유지해줘.
```
