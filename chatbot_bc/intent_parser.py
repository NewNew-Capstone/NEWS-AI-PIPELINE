from __future__ import annotations

import json
import logging
from typing import Any

from chatbot_bc.schemas import ChatIntentResponse

logger = logging.getLogger(__name__)

ALLOWED_INTENTS = {
    "GENERAL_CHAT",
    "LIST_ANALYZED_VIDEOS",
    "SEARCH_ANALYSIS",
    "SELECT_CANDIDATE",
    "RANK_ANALYSIS",
    "RANK_CURRENT_CANDIDATES",
}
ALLOWED_SORTS = {
    "RECENT",
    "OVERALL_BIAS_SCORE",
    "EMOTION_SCORE",
    "OPINION_SCORE",
}
ALLOWED_TASKS = {
    "SUMMARY",
    "SCORE",
    "DETAIL",
    "COMPARE",
}


def _fallback_intent() -> ChatIntentResponse:
    return ChatIntentResponse(intent="GENERAL_CHAT", confidence=0.0)


def _clean_response(raw: dict[str, Any]) -> ChatIntentResponse:
    intent = str(raw.get("intent") or "GENERAL_CHAT").upper()
    if intent not in ALLOWED_INTENTS:
        intent = "GENERAL_CHAT"

    keywords = raw.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()][:5]

    sort_by = raw.get("sort_by") or raw.get("sortBy")
    sort_by = str(sort_by).upper() if sort_by else None
    if sort_by not in ALLOWED_SORTS:
        sort_by = None

    sort_direction = str(raw.get("sort_direction") or raw.get("sortDirection") or "DESC").upper()
    if sort_direction not in {"ASC", "DESC"}:
        sort_direction = "DESC"

    candidate_index = raw.get("candidate_index") or raw.get("candidateIndex")
    try:
        candidate_index = int(candidate_index) if candidate_index is not None else None
    except (TypeError, ValueError):
        candidate_index = None

    answer_task = raw.get("answer_task") or raw.get("answerTask")
    answer_task = str(answer_task).upper() if answer_task else None
    if answer_task not in ALLOWED_TASKS:
        answer_task = None

    try:
        limit = int(raw.get("limit") or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 10))

    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return ChatIntentResponse(
        intent=intent,
        keywords=keywords,
        sort_by=sort_by,
        sort_direction=sort_direction,
        candidate_index=candidate_index,
        answer_task=answer_task,
        limit=limit,
        confidence=confidence,
    )


class ChatIntentParser:
    def __init__(self) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = "claude-haiku-4-5-20251001"

    def parse(
        self,
        message: str,
        has_pending_candidates: bool = False,
        has_page_video: bool = False,
    ) -> ChatIntentResponse:
        try:
            prompt = self._build_prompt(message, has_pending_candidates, has_page_video)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            raw = json.loads(text[start:end])
            return _clean_response(raw)
        except Exception as exc:
            logger.warning("intent parse failed: %s", exc, exc_info=True)
            return _fallback_intent()

    def _build_prompt(
        self,
        message: str,
        has_pending_candidates: bool,
        has_page_video: bool,
    ) -> str:
        return f"""
너는 뉴스 편향 분석 챗봇의 DB 조회 의도 분류기다.
사용자 질문을 아래 JSON 형식으로만 변환한다. SQL을 만들지 않는다.

허용 intent:
- GENERAL_CHAT: DB 조회가 필요 없는 개념/일반 질문
- LIST_ANALYZED_VIDEOS: 분석 완료 영상 목록 요청
- SEARCH_ANALYSIS: 키워드, 제목, 채널 기반 분석 결과 검색
- SELECT_CANDIDATE: 이전 후보 목록에서 번호 선택
- RANK_ANALYSIS: 전체 분석 결과에서 점수 기준 랭킹 요청
- RANK_CURRENT_CANDIDATES: 이전 후보 목록 안에서 점수 비교/랭킹 요청

sort_by 허용값:
- RECENT
- OVERALL_BIAS_SCORE
- EMOTION_SCORE
- OPINION_SCORE

answer_task 허용값:
- SUMMARY
- SCORE
- DETAIL
- COMPARE

판단 규칙:
- "편향 점수가 뭐야", "감정성 점수 뜻"처럼 개념 설명은 GENERAL_CHAT.
- "분석된 영상 뭐 있어", "저장된 분석 결과 목록"은 LIST_ANALYZED_VIDEOS.
- "전쟁 관련", "YTN 영상", "환율 분석 결과"처럼 주제/채널/제목 검색은 SEARCH_ANALYSIS.
- "1번", "2번도 알려줘"는 SELECT_CANDIDATE.
- "편향 높은 영상", "감정성 높은 뉴스"는 RANK_ANALYSIS.
- 이전 후보가 있고 "이 중에서 가장 높은 것", "그중 감정성 제일 높은 것"은 RANK_CURRENT_CANDIDATES.
- 현재 페이지 영상이 있는 상태에서 "이 영상" 질문은 SEARCH_ANALYSIS가 아니라 GENERAL_CHAT로 둔다. 페이지 context는 서버가 별도 처리한다.

컨텍스트:
- has_pending_candidates: {str(has_pending_candidates).lower()}
- has_page_video: {str(has_page_video).lower()}

사용자 질문:
{message}

반드시 아래 JSON 키만 사용:
{{
  "intent": "GENERAL_CHAT",
  "keywords": [],
  "sort_by": null,
  "sort_direction": "DESC",
  "candidate_index": null,
  "answer_task": null,
  "limit": 5,
  "confidence": 0.0
}}
""".strip()
