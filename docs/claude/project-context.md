# Project Context (BC Boundary Contract)

## 1) 파이프라인 개요
`Identity -> Content -> Analysis -> Issue Comparison -> Personalization`

핵심 원칙:
- Analysis는 "개별 영상 분석"에 집중
- Issue Comparison은 "분석 결과 재조합"에 집중
- 수집/정제(Content)와 해석(Analysis)을 분리

## 2) BC 책임 요약

### Identity
- 입력: 로그인/로그아웃/토큰 갱신
- 출력: `user_id`, 인증 상태, 토큰 정보
- 금지: 추천/영상 분석 로직 포함

### Content
- 입력: 검색어, 국가, 기간, 외부 소스
- 출력: `video_id`, 메타데이터, `sentences[]`, `video_keywords[]`
- 금지: 편향/주관성 판단 수행

### Analysis
- 입력: Content 텍스트 번들
- 출력: `summary_text`, `overall_bias_score`, `score_evidence`, `analysis_keywords[]`
- 금지: 외부 수집/검색 수행

### Issue Comparison
- 입력: Content 메타 + Analysis 결과
- 출력: 이슈 클러스터, 국가별 대표, 비교 스냅샷
- 금지: 분석 재실행(중복 계산)

### Personalization
- 입력: `user_id`, cluster/video 기반 feature set
- 출력: 스크랩/추천/마이페이지 데이터
- 금지: 원문 대량 처리 중심 설계

## 3) 경계 간 최소 전달 필드

### Content -> Analysis
- `video_id`, `title`, `description`, `country_code`, `published_at`, `sentences[]`, `video_keywords[]`

### Analysis -> Issue Comparison
- `video_id`, `summary_text`, `overall_bias_score`, `score_evidence`, `analysis_keywords[]`, `status`

### Content -> Issue Comparison
- `video_id`, `country_code`, `published_at`, `title`, `description`, `channel_name`, `video_keywords[]`

### Analysis/Comparison -> Personalization
- `target_type`, `target_id`, `keyword`, `country_set`, `analysis_keywords`, `bias_range`, `representative_video_ids`

## 4) 상태/버전/스냅샷 규칙
- 상태값 표준: `PENDING | RUNNING | SUCCESS | FAILED | SKIPPED`
- 버전 필수: 분석 모델 버전, 클러스터링 규칙 버전, 대표 선정 규칙 버전
- 비교 결과는 `snapshot` 저장을 기본값으로 사용

## 5) MVP 구현 순서
1. Content 수집/정제
2. Analysis 영상 단위 처리
3. Comparison 클러스터/대표 선정
4. Personalization 스크랩/추천
