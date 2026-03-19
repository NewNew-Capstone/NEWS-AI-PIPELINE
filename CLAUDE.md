# NEWS-AI-PIPELINE CLAUDE Guide (Token-Saving)

## 0) 절대 규칙 (Top Priority)
- 답변/설명은 기본적으로 한국어로 작성한다.
- 큰 변경은 항상 `Plan -> Review -> Execute` 순서로 진행한다.
- 한 세션에서는 한 기능(또는 한 버그)만 처리한다.
- 에러 로그는 요약하지 말고 원문 그대로 사용한다.
- `CLAUDE.md`는 짧게 유지하고 상세는 참조 문서로 분리한다.

## 1) 컨텍스트 절약 원칙
- 이 파일은 300줄 이하 유지.
- 구현 상세(스키마, DTO, 알고리즘)는 `docs/claude/*.md`에 작성.
- 루트에서는 "무엇이 중요한지"만 선언하고, "어떻게 할지"는 참조로 분리.
- 새 규칙이 생기면 이 파일에 1~2줄 요약 + 상세 문서 링크만 추가.

## 2) 작업 시작 표준 플로우
1. 요구사항을 3~7단계 계획으로 분해한다.
2. 수정 파일 목록/영향 범위를 먼저 확정한다.
3. 가장 작은 단위 변경 1개를 먼저 완료한다.
4. 테스트/검증 후 다음 단위로 이동한다.
5. 작업 종료 시 변경 요약 + 남은 TODO를 남긴다.

## 3) 품질 게이트
- 작은 단위로 변경하고 자주 검증한다.
- 기존 경계(BC 책임)를 넘는 로직 추가를 금지한다.
- 원본 데이터(Content)와 해석 데이터(Analysis)를 섞지 않는다.
- Issue Comparison은 재분석하지 말고, Analysis 결과를 재조합한다.

## 4) 도메인 기본 구조 (요약)
- `Identity`: 사용자 식별/인증 상태 제공
- `Content`: 원천 데이터 수집/정제/문장화
- `Analysis`: 영상 단위 요약/점수/근거 생성
- `Issue Comparison`: 이슈 클러스터링/국가 대표 선정/비교 결과 생성
- `Personalization`: 스크랩/추천/마이페이지

상세 계약은 아래 문서 참조:
- @docs/claude/project-context.md

## 5) 토큰 절약형 문서 구조
- 워크플로우/운영 원칙: @docs/claude/workflow-playbook.md
- 도메인 경계/DTO 기준: @docs/claude/project-context.md
- 작업 요청 템플릿: @docs/claude/task-template.md

## 6) 트리거 문구 (권장)
- "파이프라인 기준으로 설계해줘"
  - `project-context.md`의 BC 경계/DTO 우선 적용
- "토큰 아끼는 구조로 정리해줘"
  - 루트 파일 최소화 + 상세 분리 + 참조 링크 방식 적용
- "MVP 플로우로 구현해줘"
  - Content -> Analysis -> Comparison -> Personalization 순서 준수

## 7) 금지 패턴
- 루트 `CLAUDE.md`에 긴 API/스키마 전문을 붙여 넣는 행위
- 한 번에 여러 BC를 동시에 대수술하는 변경
- 비교 단계에서 분석을 다시 수행하는 중복 계산
- 상태값/버전 필드 없이 결과를 저장하는 방식
