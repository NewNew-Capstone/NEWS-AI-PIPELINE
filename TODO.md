# TODO — NEWS-AI-PIPELINE

## 공통
- [ ] `db/schemas/` — ContentPreparedEvent, AnalysisCompletedEvent, IssueClusterCreatedEvent 정의
- [ ] `.claude/settings.json` Hook 설정 (scripts/ 완성 후)

## A 담당 (analysis_bc)
- [x] `router.py` 구현 (POST /analyze)
- [ ] `preprocessor.py` 구현
- [ ] `classifier.py` 구현
- [ ] `segmenter.py` 구현 (별도 세션)
- [ ] `fact_filter.py` 구현
- [ ] `summarizer.py` 구현 (별도 세션)
- [ ] `scorer.py` 구현
- [x] 통합 테스트 작성
- [x] `embed_bc` 구현 — POST /embed (싱글톤 모델, 배치 처리, main.py 등록)

## B 담당 (content_bc)
- [ ] `collector.py` (YouTube API 클라이언트)
- [ ] `normalizer.py`
- [ ] `transcript_loader.py`
- [ ] `sentence_splitter.py`
- [ ] `keyword_mapper.py`
- [ ] `event_builder.py` → ContentPreparedEvent 반환

## C 담당 (issue_comparison_bc)
- [ ] `clusterer.py` (Qdrant 연동)
- [ ] `representative_selector.py`
- [ ] `comparison_builder.py`
- [ ] `cache_key.py`
- [ ] `event_builder.py` → IssueClusterCreatedEvent 반환

## D 담당 (personalization_bc)
- [ ] `scrap_service.py`
- [ ] `recommendation_engine.py`
- [ ] `reason_builder.py`
- [ ] `mypage_query.py`
- [ ] `event_consumer.py`
