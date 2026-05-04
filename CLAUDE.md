# CLAUDE.md

> 팀 공용 루트 규칙. 이 파일은 짧게 유지하고, 상세 설명은 `docs/claude/` 와 각 BC의 `CLAUDE.md`를 참조한다.

## 1) 이 프로젝트에서 Claude Code를 쓰는 목적
- BC 내부 구현 가속
- 반복 작업 자동화
- 문서/코드 규칙 일관화
- 테스트/리뷰 준비 보조

아래 용도로는 사용하지 않는다.
- BC 경계를 넘는 무분별한 대량 수정
- 테스트 없이 여러 기능을 한 번에 구현
- 인터페이스 합의 없이 다른 BC 코드 변경
- 시크릿이 포함된 설정 파일 자동 수정

---

## 2) 절대 규칙
1. **BC 간 직접 import 금지**
2. BC 간 통신은 반드시 **공통 Event / Schema** 를 통해서만 진행
3. **한 세션 = 한 피처 또는 한 모듈**
4. **한 세션 = 연관 파일 3~4개 이하**
5. LLM 호출 모듈은 반드시 **별도 세션** 으로 분리
6. 큰 작업은 항상 **Plan Mode 먼저**
7. `service.py` 는 orchestration만 담당
8. 테스트/린트/타입체크 GREEN 전에는 merge 금지
9. 에러가 나면 **해석하지 말고 로그 전체를 그대로 전달**
10. 막히면 다음 단계 담당자는 멈추지 말고 **mock event / fixture** 로 계속 진행

---

## 3) 팀 협업 기본 원칙
- 이 프로젝트는 **3~4인 순차 협업** 기준으로 운영한다.
- 순차 파이프라인은 한 사람이 끝까지 책임진다.
- 병렬 가능한 부분만 Sub-Agent 또는 다른 담당자와 나눈다.
- 공통 영역(`api/`, `db/`, `main.py`, `docs/claude/`, `.claude/`)은 1명 오너 + 나머지 리뷰로 관리한다.

### 권장 담당 구조
- A: `analysis_bc`
- B: `content_bc`
- C: `issue_comparison_bc`
- D(있다면): `personalization_bc` + 공통 운영 보조

---

## 4) 문서 참조 순서
작업 시작 시 아래 순서로 필요한 문서를 읽는다.
1. 현재 파일이 속한 BC의 `CLAUDE.md`
2. `@docs/claude/project-context.md`
3. `@docs/claude/workflow-playbook.md`
4. `@docs/claude/task-template.md`
5. 필요 시 `@docs/claude/agent-registry.md`

루트 `CLAUDE.md` 에는 아래만 유지한다.
- 절대 규칙
- 참조 링크
- 기본 명령
- 짧은 트리거 문장

긴 API 명세, 전체 DB 컬럼, 장문 프롬프트는 넣지 않는다.

---

## 5) 디렉토리 표준
```text
NEWS-AI-PIPELINE/
├── CLAUDE.md
├── TODO.md
├── main.py
├── requirements.txt
├── .env.example
├── docs/
│   └── claude/
│       ├── project-context.md
│       ├── workflow-playbook.md
│       ├── task-template.md
│       └── agent-registry.md
├── .claude/
│   ├── settings.json
│   ├── skills/
│   ├── commands/
│   └── agents/
├── scripts/
├── api/
├── db/
│   └── schemas/
├── analysis_bc/
├── content_bc/
├── issue_comparison_bc/
└── personalization_bc/
```

---

## 6) 공통 작업 시작 프롬프트
```text
CLAUDE.md와 [현재 BC]/CLAUDE.md를 먼저 읽어줘.
오늘 작업: [모듈명]
입력: [입력 스키마]
출력: [출력 스키마]
제약: [사용 라이브러리 / 금지사항]
TODO.md의 첫 번째 관련 항목부터 시작해줘.
먼저 Plan Mode로 수정 파일 / 테스트 범위 / 완료 기준을 보여줘.
```

---

## 7) 세션 운영 규칙
- 새 모듈 시작 전: `/clear`
- 파일 1개 끝나고 테스트 통과: `/compact`
- 세션이 길어짐: `/context` 확인 후 `/compact`
- 오류 3회 이상 반복: `/clear` 후 에러 로그 그대로 전달
- LLM 호출 모듈 작업 시작: 새 세션 강제

---

## 8) Git / PR 규칙
### 브랜치
- 기능: `feat/{담당자}-{bc}-{module}`
- 버그: `fix/{담당자}-{bc}-{issue}`

### PR 기준
- 파일 5개 이하 권장
- 300줄 이하 권장
- 본인 제외 최소 2명 승인 권장
- 아래 3개 모두 GREEN일 때만 merge
  - test
  - lint
  - typecheck

### 커밋 예시
- `feat(analysis): add preprocessor and unit tests`
- `fix(content): handle empty transcript fallback`
- `test(issue-comparison): add cluster score fixtures`

---

## 9) 공통 명령
프로젝트에 맞게 실제 명령으로 치환해 사용한다.
```bash
# 테스트
./scripts/run_tests.sh

# 파이프라인 실행
./scripts/run_pipeline.sh

# 타입체크
python -m mypy .

# 린트
ruff check .
```

---

## 10) Claude에게 항상 요구할 것
- 수정 파일 목록 먼저 제시
- 테스트 먼저 작성 가능 여부 제시
- 완료 기준 명시
- BC 경계 위반 여부 점검
- TODO.md 업데이트 여부 확인

---

## 11) 에러 대응 표준
### 해야 할 것
1. 에러 로그 전체 복사
2. 재현 명령 같이 전달
3. 새 세션 또는 `/clear`
4. 수정 후 test/lint/typecheck 재실행

### 금지
- “대충 이런 오류 같아” 식으로 요약
- 여러 오류를 한 번에 섞어 전달
- failing 상태로 다음 단계 진행

---

## 12) BC별 참조
- `@analysis_bc/CLAUDE.md`
- `@content_bc/CLAUDE.md`
- `@issue_comparison_bc/CLAUDE.md`
- `@personalization_bc/CLAUDE.md`

---

## 13) Plan Mode / Accept Mode 기준

### Plan Mode 필수 작업
- 여러 파일 동시 수정
- 새 모듈/BC 구조 생성
- DB/Event 스키마 변경
- LLM 호출 로직 추가
- 아키텍처 변경

### Accept Mode 직행 가능 작업
- 주석/문자열 수정
- type hint 보강
- 테스트 1개 추가
- 단순 리팩토링 (로직 변경 없음)

---

## 14) Hooks 사용 기준
반복 검증은 Hook으로 자동화한다.

권장 Hook:
- Write 후 → `bash scripts/lint.sh`
- 테스트 파일 변경 후 → `pytest` 자동 실행

설정 위치: `.claude/settings.json`

예시:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [{ "type": "command", "command": "bash scripts/lint.sh" }]
      }
    ]
  }
}
```

> scripts/가 완성된 후 별도 세션에서 팀 합의 후 적용한다.

---

## 15) Skills 사용 기준
반복 작업은 `.claude/skills/`에 등록한다.

팀 공용 Skills 후보:
- `bc-scaffold` : 새 BC 폴더+파일 일괄 생성
- `event-schema` : Event 스키마 템플릿 생성
- `pr-checklist` : PR 전 체크리스트 출력

개인용은 `~/.claude/skills/`, 팀 공용은 `.claude/skills/`에 저장한다.

---

## 16) Sub-Agent 사용 기준

### 써도 되는 경우
- 독립적인 테스트 작성
- 대량 로그 분석
- 문서 업데이트
- 코드 리뷰

### 쓰지 말아야 하는 경우
- BC 경계를 넘는 수정
- 인터페이스가 아직 합의되지 않은 작업
- 메인 세션과 지속 추론을 주고받아야 하는 디버깅
