# content_bc/CLAUDE.md

> Content BC 전용 규칙. 이 BC는 외부 소스에서 콘텐츠를 수집하고 정규화하여 분석 가능한 입력 단위로 만든다.

## 1) 이 BC의 책임
- 외부 API / 소스 수집
- 원천 데이터 정규화
- 제목 / 설명 / 태그 / 메타데이터 정리
- transcript / comment / sentence 분리
- 키워드 연결
- `ContentPreparedEvent` 발행

---

## 2) 이 BC에서 수정 가능한 파일
- `content_bc/router.py`
- `content_bc/service.py`
- `content_bc/schemas.py`
- `content_bc/models.py`
- `content_bc/modules/*`
- `tests/unit/content_*`
- `tests/integration/content_*`

외부 API quota 정책, credential 처리, secret 값은 코드에 하드코딩하지 않는다.

---

## 3) 핵심 원칙
1. 수집과 정규화는 분리한다.
2. 원천 응답을 바로 다음 BC에 넘기지 말고 **정규화 스키마** 로 바꾼다.
3. transcript fallback 정책은 코드와 문서로 명시한다.
4. sentence split 규칙은 fixture와 함께 테스트한다.
5. `service.py` 는 수집 → 정규화 → 문장 분리 → event 조립 순으로만 orchestration 한다.

---

## 4) 권장 모듈 구조
```text
content_bc/
├── CLAUDE.md
├── router.py
├── service.py
├── schemas.py
├── models.py
└── modules/
    ├── collector.py
    ├── normalizer.py
    ├── transcript_loader.py
    ├── sentence_splitter.py
    ├── keyword_mapper.py
    └── event_builder.py
```

### 파일 역할
- `collector.py`: 외부 API 호출 / paging / retry
- `normalizer.py`: 공통 필드 스키마 변환
- `transcript_loader.py`: transcript source 선택 / fallback
- `sentence_splitter.py`: 문장 분리
- `keyword_mapper.py`: 키워드 정규화 / 연결
- `event_builder.py`: `ContentPreparedEvent` 조립

---

## 5) 세션 운영 규칙
### 같이 작업해도 되는 묶음
- `normalizer.py` + 관련 schema + unit test
- `sentence_splitter.py` + fixture + unit test

### 분리 권장 묶음
- `collector.py` 와 API quota / retry 정책 수정
- transcript fallback 수정
- event schema 연결 작업

---

## 6) 입력 / 출력 계약
### 입력
- 검색어 / 국가 / 주제 / source 설정
- 외부 원천 응답

### 출력
- 정규화된 콘텐츠 레코드
- 문장 단위 레코드
- 키워드 연결 정보
- `ContentPreparedEvent`

---

## 7) 테스트 원칙
- unit test: normalize / split / keyword mapping 규칙 검증
- integration test: source → normalized event 흐름 검증
- 외부 API는 mock / fixture 우선
- quota / empty response / missing transcript 케이스를 반드시 포함

### 최소 체크리스트
- [ ] 빈 응답 처리
- [ ] transcript 없음 fallback
- [ ] 문장 분리 index 유지
- [ ] 국가/언어 메타데이터 유지
- [ ] 중복 콘텐츠 제거 규칙 검증
- [ ] event payload 필수 필드 검증

---

## 8) Claude에게 잘 맞는 작업
- 수집 클라이언트 뼈대 생성
- 응답 정규화 함수 작성
- sentence split 유틸 정리
- event builder 작성
- 예외 처리 및 mock fixture 보강

## 9) 주의할 것
- API quota / 정책 결정은 사람이 최종 검증
- transcript source fallback은 명시적으로 관리
- 외부 응답 포맷을 service.py 안에서 직접 다루지 말 것
- Analysis BC를 직접 import 하지 말 것

---

## 10) 작업 시작 프롬프트 예시
```text
루트 CLAUDE.md와 content_bc/CLAUDE.md를 읽어줘.
오늘 작업은 [collector/normalizer/sentence_splitter/event_builder] 구현이야.
먼저 수정 파일, mock 여부, 외부 API 의존성, 테스트 범위를 Plan Mode로 보여줘.
service.py는 orchestration만 유지해줘.
```
