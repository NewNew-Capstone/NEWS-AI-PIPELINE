# agent-registry.md

> Claude Code 팀 운영용 에이전트 레지스트리.  
> 목적은 **누가 어떤 범위를 책임지고**, **어떤 입력을 받아**, **무엇을 출력하며**, **어디서 handoff 하는지**를 팀 전체가 같은 기준으로 보게 만드는 것이다.  
> 이 문서는 루트 `CLAUDE.md`, 각 BC의 `CLAUDE.md`, `workflow-playbook.md`와 함께 사용한다.

---

## 1. 사용 원칙
- 이 프로젝트는 **3~4인 순차 협업** 기준으로 운영한다.
- 순차 파이프라인은 앞 단계 산출물이 다음 단계 입력이 된다.
- 따라서 에이전트는 "모든 걸 한 번에 처리하는 만능 에이전트"가 아니라, **단계별 책임이 분리된 작업 단위**여야 한다.
- **BC 간 직접 import는 금지**하며, handoff 는 반드시 Event / Schema / DTO / Fixture를 통해 수행한다.
- 막히면 다음 단계는 멈추지 않고 **mock event / fixture** 로 계속 진행한다.

---

## 2. 운영 레벨

### 2.1 레벨 구분
이 프로젝트의 에이전트는 아래 3개 레벨로 본다.

1. **Coordinator Agent**  
   - 작업 범위 확인, 문서 참조, TODO 정렬, handoff 기준 확인
2. **BC Owner Agent**  
   - 특정 BC 내부 구현 담당
3. **Specialist Agent**  
   - 테스트, schema, fixture, LLM prompt, event 연결처럼 세부 작업만 담당

즉, 실제 작업 흐름은 보통 아래 순서다.

```text
Coordinator
  -> BC Owner
      -> Specialist (필요 시)
  -> 다음 BC Owner로 handoff
```

---

## 3. 전체 에이전트 맵

| Agent ID | 종류 | 주 책임 | 주 오너(권장) | 주 입력 | 주 출력 |
|---|---|---|---|---|---|
| `project-coordinator` | Coordinator | 작업 범위 정리, 문서 참조, handoff 기준 관리 | 공통 오너 1명 | TODO, 루트/BC 문서 | 작업 계획, 수정 범위, handoff 체크리스트 |
| `content-owner` | BC Owner | 수집/정규화/문장 분리/Event 조립 | B | 외부 응답, source 설정 | `ContentPreparedEvent` |
| `analysis-owner` | BC Owner | 분류/세분화/요약/점수화 | A | `ContentPreparedEvent` | `AnalysisCompletedEvent` |
| `issue-comparison-owner` | BC Owner | 클러스터링/대표 선정/비교 결과 조립 | C | `AnalysisCompletedEvent` | comparison DTO / cluster event |
| `personalization-owner` | BC Owner | 스크랩/추천/마이페이지 조립 | D 또는 C | cluster/result DTO, user context | recommendation DTO / mypage DTO |
| `schema-guardian` | Specialist | Event / DTO / Pydantic schema 점검 | 공통 오너 | schema 초안 | schema diff, validation checklist |
| `test-writer` | Specialist | unit/integration 테스트 초안 작성 | 각 BC 오너 | 모듈 spec, schema | test file, fixture |
| `fixture-maker` | Specialist | mock event / sample payload / fake data 생성 | 각 BC 오너 | 입력/출력 계약 | fixture, mock payload |
| `llm-module-agent` | Specialist | prompt / wrapper / fallback / timeout 작업 | Analysis 오너 중심 | prompt 목적, 입출력 규약 | prompt 모듈, wrapper, mock test |
| `review-agent` | Specialist | BC 경계 위반, service 과비대화, 테스트 누락 점검 | 리뷰어 | 변경 파일 | 리뷰 체크리스트, 수정 제안 |

---

## 4. Agent 상세 정의

---

### 4.1 `project-coordinator`
**역할**  
작업 시작 전에 이번 세션의 범위를 줄이고, 어떤 BC 문서를 읽어야 하는지, handoff 산출물이 무엇인지 먼저 정리하는 에이전트.

**언제 쓰나**
- 새 작업 시작
- 담당자 교체
- 여러 BC가 걸치는 요청이 들어왔을 때
- 스키마/이벤트 변경이 포함될 때

**입력**
- 루트 `CLAUDE.md`
- 해당 BC의 `CLAUDE.md`
- `TODO.md`
- 현재 작업 요구사항

**출력**
- 수정 파일 후보 1~4개
- 이번 세션 제외 파일 목록
- 테스트 범위
- 완료 기준
- 다음 handoff 대상

**수정 가능 범위**
- 원칙적으로 코드 수정 없음
- 문서/계획/TODO 정리만 권장

**금지**
- 직접 비즈니스 로직 구현 시작
- 여러 BC 코드를 한 세션에서 동시에 수정
- 인터페이스 미합의 상태에서 공통 schema 확정

**완료 조건**
- 작업 범위가 1개 BC 또는 1개 모듈 수준으로 줄어들어 있음
- 다음 작업자가 바로 시작 가능한 상태

**권장 시작 프롬프트**
```text
루트 CLAUDE.md와 관련 BC 문서를 읽고, 이번 작업을 1개 세션 범위로 줄여줘.
수정 파일 후보, 테스트 범위, 완료 기준, 다음 handoff 대상을 먼저 보여줘.
```

---

### 4.2 `content-owner`
**역할**  
외부 source 에서 데이터를 수집하고, 정규화하고, 문장 단위로 분리해서 분석 가능한 입력으로 만든다.

**담당 BC**  
`content_bc/`

**대표 입력**
- 검색어 / 국가 / 주제 / source 옵션
- 외부 API 응답
- raw metadata

**대표 출력**
- normalized content record
- sentence list
- keyword mapping
- `ContentPreparedEvent`

**수정 가능 범위**
- `content_bc/router.py`
- `content_bc/service.py`
- `content_bc/schemas.py`
- `content_bc/models.py`
- `content_bc/modules/*`
- `tests/unit/content_*`
- `tests/integration/content_*`

**핵심 규칙**
- 수집과 정규화 분리
- 원천 응답을 그대로 다음 BC에 넘기지 않음
- sentence split 규칙은 fixture로 재현 가능해야 함
- `service.py`는 orchestration만 담당

**금지**
- Analysis BC 내부 구현 직접 import
- quota 정책을 사람 확인 없이 크게 변경
- secret / credential 하드코딩

**handoff 대상**
- `analysis-owner`

**handoff 산출물**
- `ContentPreparedEvent`
- 필수 필드 설명
- 누락/empty/fallback 정책 메모
- 샘플 fixture 1~2개

**완료 조건**
- 빈 응답, transcript 없음, 문장 분리 인덱스 유지 케이스 테스트 완료
- event payload 필수 필드 검증 완료

---

### 4.3 `analysis-owner`
**역할**  
정규화된 문장 입력을 기반으로 분류, span 추출, 요약, 점수화를 수행한다.

**담당 BC**  
`analysis_bc/`

**대표 입력**
- `ContentPreparedEvent`
- 정규화된 sentence list
- metadata / source info

**대표 출력**
- sentence labels
- opinion spans
- summary
- score evidence
- `AnalysisCompletedEvent`

**수정 가능 범위**
- `analysis_bc/router.py`
- `analysis_bc/service.py`
- `analysis_bc/schemas.py`
- `analysis_bc/models.py`
- `analysis_bc/modules/*`
- `tests/unit/analysis_*`
- `tests/integration/analysis_*`

**핵심 규칙**
- `service.py`는 orchestration만
- classifier / segmenter / summarizer / scorer 분리
- scorer에 LLM 로직 금지
- LLM 호출 모듈은 별도 세션

**금지**
- segmenter와 summarizer를 같은 세션에서 동시에 크게 수정
- service.py에 프롬프트 본문 삽입
- 점수 계산식을 service.py에 직접 작성

**handoff 대상**
- `issue-comparison-owner`

**handoff 산출물**
- `AnalysisCompletedEvent`
- label/span/summary/score 필드 설명
- score evidence 예시
- empty/fallback 규칙

**완료 조건**
- empty / 1문장 / 전부 FACT / 전부 OPINION / summary fallback 테스트 완료
- span offset 검증 완료

---

### 4.4 `issue-comparison-owner`
**역할**  
분석 결과를 유사 이슈끼리 묶고, 대표 항목을 뽑아, 비교 응답으로 조립한다.

**담당 BC**  
`issue_comparison_bc/`

**대표 입력**
- `AnalysisCompletedEvent`
- 분석 결과 리스트
- 유사도 / 임베딩 / 메타데이터

**대표 출력**
- issue cluster
- representative item
- comparison response DTO
- 필요 시 cluster event

**수정 가능 범위**
- `issue_comparison_bc/router.py`
- `issue_comparison_bc/service.py`
- `issue_comparison_bc/schemas.py`
- `issue_comparison_bc/models.py`
- `issue_comparison_bc/modules/*`
- `tests/unit/issue_comparison_*`
- `tests/integration/issue_comparison_*`

**핵심 규칙**
- 클러스터링과 대표 선정 분리
- 가중치 실험 전에 표준값 고정
- fallback 정책 문서화
- 동일 입력이면 동일 결과가 나오도록 재현성 유지

**금지**
- 가중치 변경 후 근거 없이 즉시 merge
- service.py에 점수 계산식 직접 삽입
- vector store 정책을 문서 변경 없이 수정

**handoff 대상**
- `personalization-owner`
- 또는 API 응답 조립 담당자

**handoff 산출물**
- comparison DTO
- cluster 기준 설명
- representative 선정 기준
- fallback / tie-break 규칙

**완료 조건**
- 유사도 경계값, tie-break, empty cluster, cache key 안정성 테스트 완료

---

### 4.5 `personalization-owner`
**역할**  
사용자 기준으로 비교 결과를 저장하고, 추천 그룹/추천 아이템/마이페이지 응답을 조립한다.

**담당 BC**  
`personalization_bc/`

**대표 입력**
- user_id
- comparison result / issue cluster / representative item
- scrap action event 또는 조회 조건

**대표 출력**
- scrap 상태
- recommendation group / item
- recommendation reason
- mypage response DTO

**수정 가능 범위**
- `personalization_bc/router.py`
- `personalization_bc/service.py`
- `personalization_bc/schemas.py`
- `personalization_bc/models.py`
- `personalization_bc/modules/*`
- `tests/unit/personalization_*`
- `tests/integration/personalization_*`

**핵심 규칙**
- 스크랩과 추천 분리
- ISSUE_CLUSTER / YOUTUBE_VIDEO type 분리
- 추천 사유는 추적 가능해야 함
- `user_id` 경계 검증 필수

**금지**
- 타 사용자 데이터 노출 가능성 있는 수정
- 추천 사유를 service.py에 직접 하드코딩
- 서로 다른 scrap type을 같은 enum/흐름으로 뭉개기

**handoff 대상**
- API 응답 조립 담당자
- 마이페이지/스크랩 화면 담당자

**handoff 산출물**
- recommendation DTO
- reason schema
- user scope 제약
- empty state 응답 예시

**완료 조건**
- user_id 경계, type 분리, 중복 스크랩, empty mypage 테스트 완료

---

### 4.6 `schema-guardian`
**역할**  
Event / DTO / Pydantic schema 를 프로젝트 공통 기준으로 지키는 검증 전용 에이전트.

**언제 쓰나**
- Event 필드 추가/삭제
- BC 간 handoff payload 변경
- optional / required 기준 변경
- enum / 타입 변경

**대표 입력**
- schema 초안
- 변경 전/후 diff
- handoff 목적

**대표 출력**
- schema review comment
- required/optional 체크리스트
- backward compatibility 리스크
- fixture 수정 포인트

**금지**
- 비즈니스 로직 구현까지 확장
- 여러 BC에서 독자적으로 다른 이름 사용 허용

**완료 조건**
- 필드명, 타입, nullable, 예시 payload가 일관됨
- 변경된 schema 를 쓰는 fixture/test 수정 포인트가 명확함

---

### 4.7 `test-writer`
**역할**  
각 BC의 unit / integration 테스트 초안을 만드는 테스트 전용 에이전트.

**대표 입력**
- 모듈 spec
- schema
- edge case 목록

**대표 출력**
- test skeleton
- happy path / edge case 테스트
- fixture 참조 포인트

**핵심 규칙**
- 외부 API / LLM / DB 외부 호출은 mock 우선
- 테스트는 모듈 책임 단위로 나눔
- 실패 메시지가 원인을 드러내도록 작성

**금지**
- production 코드 책임을 테스트에서 대신 구현
- 하나의 거대한 integration test로 모든 케이스 처리

**완료 조건**
- 최소 체크리스트가 테스트 파일에 반영됨
- fixture 없이 재현 불가능한 테스트가 남지 않음

---

### 4.8 `fixture-maker`
**역할**  
앞 단계가 아직 안 끝나도 다음 단계 구현이 가능하도록 mock payload 와 fixture 를 만든다.

**대표 입력**
- handoff schema
- 예시 payload 요구사항
- edge case 목록

**대표 출력**
- `tests/fixtures/*.json`
- fake event payload
- empty / fallback / malformed 예시

**핵심 규칙**
- 실제 schema 이름과 필드명을 맞출 것
- 정상 / 빈 값 / 누락 / fallback 케이스를 함께 만들 것
- fixture 파일명만 봐도 목적을 알 수 있게 할 것

**금지**
- 실제 schema 와 다른 임의 이름 사용
- happy path fixture 하나만 만들고 끝내기

**완료 조건**
- 다음 BC가 실제 구현 없이도 mock 기반 개발 가능

---

### 4.9 `llm-module-agent`
**역할**  
LLM 호출이 필요한 모듈을 전담한다. prompt 본문, wrapper, retry, fallback, timeout, mock 테스트를 별도 세션에서 처리한다.

**주 사용 위치**
- `analysis_bc/modules/segmenter.py`
- `analysis_bc/modules/summarizer.py`
- 추후 reasoning / recommendation text generation 모듈

**대표 입력**
- 작업 목적
- 입력 스키마
- 출력 스키마
- timeout / fallback 정책

**대표 출력**
- prompt wrapper
- response parser
- fallback policy
- mock 기반 unit/integration test

**핵심 규칙**
- 반드시 별도 세션
- prompt 수정과 일반 비즈니스 로직 수정을 한 세션에 섞지 않음
- 실패 시 graceful fallback 을 코드와 테스트로 남김

**금지**
- API key / secret 코드 삽입
- prompt와 scorer/cluster 수식 로직 혼합
- 외부 호출 실패를 삼켜버리는 처리

**완료 조건**
- timeout / empty / malformed response fallback 존재
- mock 테스트 존재
- 입출력 schema 가 명확함

---

### 4.10 `review-agent`
**역할**  
PR 직전 또는 merge 전, 변경이 BC 경계 / service 책임 / 테스트 기준 / 문서 기준을 지켰는지 검토한다.

**대표 입력**
- 변경 파일 목록
- diff
- 테스트 결과

**대표 출력**
- 리뷰 체크리스트
- 수정 필요 항목
- merge 가능/보류 판단 근거

**체크 포인트**
- BC 경계 위반 없음
- `service.py` 과비대화 없음
- test/lint/typecheck 결과 있음
- TODO / 문서 업데이트 누락 없음
- mock 이 실제 schema 와 크게 어긋나지 않음

**금지**
- “대충 괜찮아 보임” 식의 추상 피드백
- 로그/실패 케이스 미확인 상태 승인

**완료 조건**
- merge 전 리스크가 문장으로 명확히 정리됨

---

## 5. 권장 handoff 체인

### 5.1 기본 순차 흐름
```text
content-owner
  -> analysis-owner
  -> issue-comparison-owner
  -> personalization-owner
```

### 5.2 스키마 변경이 있을 때
```text
project-coordinator
  -> schema-guardian
  -> 해당 BC Owner
  -> test-writer / fixture-maker
  -> review-agent
```

### 5.3 LLM 모듈이 끼어 있을 때
```text
analysis-owner
  -> llm-module-agent
  -> test-writer
  -> analysis-owner
```

---

## 6. 3인 팀 / 4인 팀 배치안

### 6.1 3인 팀
- **A:** `analysis-owner` + `llm-module-agent`
- **B:** `content-owner` + `fixture-maker`
- **C:** `issue-comparison-owner` + `personalization-owner` + `review-agent`

### 6.2 4인 팀
- **A:** `analysis-owner`
- **B:** `content-owner`
- **C:** `issue-comparison-owner`
- **D:** `personalization-owner` + `review-agent`

### 6.3 공통 보조 역할
아래 역할은 특정 1명이 오너를 잡고, 나머지는 리뷰로 붙는 방식이 좋다.
- `project-coordinator`
- `schema-guardian`
- `.claude/` 운영
- `docs/claude/` 유지보수

---

## 7. 파일별 추천 담당 에이전트

| 파일/폴더 | 1차 담당 | 2차 검토 |
|---|---|---|
| `content_bc/modules/collector.py` | `content-owner` | `test-writer` |
| `content_bc/modules/event_builder.py` | `content-owner` | `schema-guardian` |
| `analysis_bc/modules/classifier.py` | `analysis-owner` | `test-writer` |
| `analysis_bc/modules/segmenter.py` | `llm-module-agent` | `analysis-owner` |
| `analysis_bc/modules/summarizer.py` | `llm-module-agent` | `analysis-owner` |
| `analysis_bc/modules/scorer.py` | `analysis-owner` | `review-agent` |
| `issue_comparison_bc/modules/clusterer.py` | `issue-comparison-owner` | `test-writer` |
| `issue_comparison_bc/modules/comparison_builder.py` | `issue-comparison-owner` | `schema-guardian` |
| `personalization_bc/modules/recommendation_engine.py` | `personalization-owner` | `review-agent` |
| `personalization_bc/modules/reason_builder.py` | `personalization-owner` | `test-writer` |
| `api/`, `db/`, `main.py` | 공통 오너 1명 | 최소 2명 리뷰 |
| `docs/claude/*.md` | `project-coordinator` | 각 BC 오너 |

---

## 8. 에이전트 호출 템플릿

### 8.1 Coordinator 호출
```text
루트 CLAUDE.md와 관련 BC 문서를 읽고, 이번 작업을 1세션 범위로 줄여줘.
수정 파일, 제외 파일, 테스트 범위, 완료 기준, handoff 대상을 먼저 보여줘.
```

### 8.2 BC Owner 호출
```text
루트 CLAUDE.md와 [BC]/CLAUDE.md를 읽어줘.
오늘 작업은 [모듈명] 구현이야.
service.py는 orchestration만 유지하고,
먼저 수정 파일, 테스트 범위, mock 필요 여부, 완료 기준을 Plan Mode로 보여줘.
```

### 8.3 Schema Guardian 호출
```text
이벤트/DTO 스키마 변경안을 검토해줘.
required/optional, nullable, enum, fixture 영향 범위를 체크해서
호환성 리스크와 수정 포인트를 정리해줘.
```

### 8.4 Test Writer 호출
```text
이 모듈의 책임과 스키마를 기준으로
unit test와 필요한 integration test 초안을 작성해줘.
happy path, edge case, empty/fallback 케이스를 포함해줘.
```

### 8.5 Review Agent 호출
```text
이 변경이 BC 경계, service 책임 분리, test/lint/typecheck, 문서 업데이트 기준을 지켰는지 리뷰해줘.
반드시 수정이 필요한 항목만 우선순위와 함께 정리해줘.
```

---

## 9. 최종 체크리스트
- [ ] 지금 세션의 담당 에이전트가 명확한가?
- [ ] 이번 세션이 1개 BC / 1개 모듈 수준으로 줄어들었는가?
- [ ] handoff 입력/출력이 Event / DTO / Fixture로 정리됐는가?
- [ ] mock 으로 다음 단계 진행 가능하게 해두었는가?
- [ ] test / lint / typecheck 기준이 명확한가?
- [ ] merge 전에 `review-agent` 수준의 검토를 했는가?

---

## 10. 한 줄 원칙
**에이전트는 일을 대신 다 해주는 존재가 아니라, 팀의 순차 파이프라인을 흔들림 없이 이어주는 역할 단위다.**
