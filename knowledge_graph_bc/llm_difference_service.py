from __future__ import annotations

import json
import logging
import os
from typing import Any

from anthropic import Anthropic

from knowledge_graph_bc.schemas import (
    ComparisonInsightRequest,
    ComparisonInsightResponse,
    ComparisonInsightRow,
)

logger = logging.getLogger(__name__)


class ComparisonInsightService:
    def __init__(self) -> None:
        self.api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        self.model = (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5-20251001").strip()
        self.client = Anthropic(timeout=20.0, max_retries=1) if self.api_key else None

    def summarize_difference(self, payload: ComparisonInsightRequest) -> ComparisonInsightResponse:
        fallback = self._build_fallback(payload)

        if self.client is None:
            logger.warning("Comparison LLM disabled: ANTHROPIC_API_KEY is empty")
            return fallback

        raw_text = ""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=900,
                temperature=0.2,
                tools=[self._tool_schema()],
                tool_choice={"type": "tool", "name": "comparison_insight"},
                messages=[{"role": "user", "content": self._build_prompt(payload)}],
            )
            result = self._extract_tool_input(response)
            return self._normalize_response(result, fallback)
        except Exception as exc:
            logger.warning("Comparison LLM failed, fallback enabled: %s", exc, exc_info=True)
            logger.debug("Comparison LLM raw response before failure: %s", raw_text)
            return fallback

    def _build_prompt(self, payload: ComparisonInsightRequest) -> str:
        selected = payload.selected_video
        compared = payload.compared_video
        edge = payload.relation_edge

        return f"""
너는 국가별 뉴스 관점 비교를 설명하는 분석가야.
아래 데이터만 근거로 원본 영상과 비교 영상을 한국어로 비교해줘.
영상 제목, 국가, 키워드, 그래프 연결 근거가 다르면 반드시 다른 요약을 만들어야 해.
모르는 내용은 단정하지 말고 "제목과 키워드 기준" 또는 "제공된 분석 기준"이라고 표현해.
analysisSummary, perspectiveSummary, evidenceSummary가 있으면 제목/키워드보다 우선해서 영상 내용 차이를 설명해.

[원본 영상]
- 국가: {self._country_label(selected)}
- 제목: {self._read_text(selected, "title", fallback="원본 영상")}
- 채널: {self._read_text(selected, "channelName", "channel_name", fallback="채널 정보 없음")}
- 중점 키워드: {self._keyword_text(self._read_list(selected, "focusKeywords", "focus_keywords"))}
- 감정 표현: {self._keyword_text(self._read_list(selected, "emotionKeywords", "emotion_keywords"))}
- 분석 요약: {self._read_text(selected, "analysisSummary", "analysis_summary", "summaryText", "summary_text", fallback="분석 요약 없음")}
- 관점 요약: {self._read_text(selected, "perspectiveSummary", "perspective_summary", fallback="관점 요약 없음")}
- 근거 요약: {self._read_text(selected, "evidenceSummary", "evidence_summary", fallback="근거 요약 없음")}
- 점수 설명: {self._read_text(selected, "scoreReasonSummary", "score_reason_summary", fallback="점수 설명 없음")}
- 표현 톤: {self._read_text(selected, "toneLabel", "tone_label", fallback="톤 정보 없음")}
- 사실 비율: {self._read_text(selected, "factRatio", "fact_ratio", fallback="사실 비율 정보 없음")}
- 중립성 점수: {self._read_text(selected, "neutralityScore", "neutrality_score", fallback="중립성 정보 없음")}

[비교 영상]
- 국가: {self._country_label(compared)}
- 제목: {self._read_text(compared, "title", fallback="비교 영상")}
- 채널: {self._read_text(compared, "channelName", "channel_name", fallback="채널 정보 없음")}
- 중점 키워드: {self._keyword_text(self._read_list(compared, "focusKeywords", "focus_keywords"))}
- 감정 표현: {self._keyword_text(self._read_list(compared, "emotionKeywords", "emotion_keywords"))}
- 분석 요약: {self._read_text(compared, "analysisSummary", "analysis_summary", "summaryText", "summary_text", fallback="분석 요약 없음")}
- 관점 요약: {self._read_text(compared, "perspectiveSummary", "perspective_summary", fallback="관점 요약 없음")}
- 근거 요약: {self._read_text(compared, "evidenceSummary", "evidence_summary", fallback="근거 요약 없음")}
- 점수 설명: {self._read_text(compared, "scoreReasonSummary", "score_reason_summary", fallback="점수 설명 없음")}
- 표현 톤: {self._read_text(compared, "toneLabel", "tone_label", fallback="톤 정보 없음")}
- 사실 비율: {self._read_text(compared, "factRatio", "fact_ratio", fallback="사실 비율 정보 없음")}
- 중립성 점수: {self._read_text(compared, "neutralityScore", "neutrality_score", fallback="중립성 정보 없음")}

[그래프 연결 근거]
- 관계 타입: {self._read_text(edge, "relationType", "relation_type", fallback="관계 정보 없음")}
- 관련도 점수: {self._read_text(edge, "weight", fallback="점수 정보 없음")}
- 내용 유사도: {self._read_text(edge, "similarityScore", "similarity_score", fallback="유사도 정보 없음")}
- 공유 키워드: {self._keyword_text(self._read_list(edge, "keywords"))}
- 추천 이유: {self._keyword_text(self._read_list(edge, "reasons"))}
- 점수 세부항목: {json.dumps(self._read_map(edge, "scoreBreakdown", "score_breakdown"), ensure_ascii=False)}

[전체 비교 맥락]
- 주요 키워드: {self._keyword_text(payload.main_keywords)}
- 공유 키워드: {self._keyword_text(payload.shared_keywords)}
- 공유 엔티티: {self._keyword_text(payload.shared_entities)}
- 국가별 관점 요약: {json.dumps(payload.country_perspectives, ensure_ascii=False)}

comparison_insight 도구를 사용해서 구조화된 결과를 반환해.
comparisonRows는 반드시 4개를 채워줘.
""".strip()

    def _tool_schema(self) -> dict[str, Any]:
        return {
            "name": "comparison_insight",
            "description": "원본 영상과 비교 영상의 국가별 관점 차이를 표와 요약문으로 반환한다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "두 영상의 분석 요약, 관점 요약, 키워드, 연결 근거를 종합해 핵심 차이를 한국어 3~4문장으로 설명",
                    },
                    "comparisonRows": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "selected": {"type": "string"},
                                "compared": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                            "required": ["label", "selected", "compared", "detail"],
                            "additionalProperties": False,
                        },
                    },
                    "points": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                    "recommendationReason": {
                        "type": "string",
                        "description": "이 영상이 비교 대상으로 연결된 이유를 한 문단으로 설명",
                    },
                },
                "required": ["summary", "comparisonRows", "points", "recommendationReason"],
                "additionalProperties": False,
            },
        }

    def _build_fallback(self, payload: ComparisonInsightRequest) -> ComparisonInsightResponse:
        selected = payload.selected_video
        compared = payload.compared_video
        edge = payload.relation_edge

        selected_country = self._country_label(selected)
        compared_country = self._country_label(compared)
        selected_title = self._read_text(selected, "title", fallback="원본 영상")
        compared_title = self._read_text(compared, "title", fallback="비교 영상")
        selected_focus = self._read_text(
            selected,
            "perspectiveSummary",
            "perspective_summary",
            "analysisSummary",
            "analysis_summary",
            "summaryText",
            "summary_text",
            fallback=self._keyword_text(self._read_list(selected, "focusKeywords", "focus_keywords"), "제목과 키워드 기준"),
        )
        compared_focus = self._read_text(
            compared,
            "perspectiveSummary",
            "perspective_summary",
            "analysisSummary",
            "analysis_summary",
            "summaryText",
            "summary_text",
            fallback=self._keyword_text(self._read_list(compared, "focusKeywords", "focus_keywords"), "제목과 키워드 기준"),
        )
        shared_keywords = self._keyword_text(self._read_list(edge, "keywords") or payload.shared_keywords, "공유 키워드 정보 없음")
        similarity = self._read_text(edge, "similarityScore", "similarity_score", fallback="")
        reasons = self._read_list(edge, "reasons")
        reason_text = " ".join(str(reason) for reason in reasons[:3]) or "공유 키워드와 내용 유사도를 기준으로 비교 후보가 되었습니다."

        summary = (
            f"{selected_country} 영상은 \"{selected_title}\"를 통해 이슈를 설명하고, "
            f"{compared_country} 영상은 \"{compared_title}\"를 통해 같은 이슈를 다른 국가 관점에서 다룹니다. "
            f"두 영상은 {shared_keywords}를 공유하지만, 원본 영상은 {selected_focus}에 더 가깝고 "
            f"비교 영상은 {compared_focus}에 더 가깝습니다."
        )
        if similarity:
            summary += f" 제공된 그래프 기준 내용 유사도는 {similarity}입니다."

        rows = [
            ComparisonInsightRow(
                label="이슈 초점",
                selected=selected_focus,
                compared=compared_focus,
                detail=f"{selected_country} 영상은 선택 영상의 문제의식에서 출발하고, {compared_country} 영상은 자국 관점에서 같은 이슈를 해석합니다.",
            ),
            ComparisonInsightRow(
                label="공유 근거",
                selected=shared_keywords,
                compared=shared_keywords,
                detail=reason_text,
            ),
            ComparisonInsightRow(
                label="내용 유사도",
                selected="원본 영상의 제목/키워드/분석 정보",
                compared=f"유사도 {similarity}" if similarity else "비교 영상의 제목/키워드/분석 정보",
                detail="두 영상은 제목과 키워드, 그래프 연결 근거를 함께 비교해 같은 이슈 후보로 연결되었습니다.",
            ),
            ComparisonInsightRow(
                label="관점 차이",
                selected=f"{selected_country} 관점",
                compared=f"{compared_country} 관점",
                detail=f"같은 사건을 다루지만 {selected_country} 영상과 {compared_country} 영상은 강조하는 이해관계와 표현 톤이 다르게 나타납니다.",
            ),
        ]

        return ComparisonInsightResponse(
            summary=summary,
            comparisonRows=rows,
            points=[
                f"{selected_country} 영상의 초점: {selected_focus}",
                f"{compared_country} 영상의 초점: {compared_focus}",
                f"공유 키워드: {shared_keywords}",
            ],
            recommendationReason=reason_text,
            llmUsed=False,
        )

    def _normalize_response(
        self,
        result: dict[str, Any],
        fallback: ComparisonInsightResponse,
    ) -> ComparisonInsightResponse:
        rows = []
        for row in result.get("comparisonRows") or result.get("comparison_rows") or []:
            if not isinstance(row, dict):
                continue
            rows.append(
                ComparisonInsightRow(
                    label=str(row.get("label") or row.get("category") or "비교 항목").strip(),
                    selected=str(row.get("selected") or row.get("source") or row.get("original") or "").strip(),
                    compared=str(row.get("compared") or row.get("target") or row.get("comparison") or "").strip(),
                    detail=str(row.get("detail") or row.get("difference") or row.get("summary") or "").strip(),
                )
            )

        summary = str(result.get("summary") or "").strip()
        points = [str(point).strip() for point in result.get("points", []) if str(point).strip()]
        recommendation_reason = str(result.get("recommendationReason") or result.get("recommendation_reason") or "").strip()

        return ComparisonInsightResponse(
            summary=summary or fallback.summary,
            comparisonRows=rows or fallback.comparison_rows,
            points=points or fallback.points,
            recommendationReason=recommendation_reason or fallback.recommendation_reason,
            llmUsed=True,
        )

    def _extract_tool_input(self, response: Any) -> dict[str, Any]:
        text_blocks: list[str] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use" and getattr(block, "name", "") == "comparison_insight":
                tool_input = getattr(block, "input", None)
                if isinstance(tool_input, dict):
                    return tool_input
            text = getattr(block, "text", "")
            if text:
                text_blocks.append(str(text))

        if text_blocks:
            return self._parse_json("\n".join(text_blocks))

        raise ValueError("LLM response does not contain comparison_insight tool output")

    def _parse_json(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain JSON object")
        parsed = json.loads(text[start:end])
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON is not an object")
        return parsed

    def _country_label(self, video: dict[str, Any]) -> str:
        return self._read_text(video, "countryLocalLabel", "country_local_label", "countryCode", "country_code", fallback="국가 정보 없음")

    def _read_text(self, source: dict[str, Any], *keys: str, fallback: str = "") -> str:
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    def _read_list(self, source: dict[str, Any], *keys: str) -> list[Any]:
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                return value
        return []

    def _read_map(self, source: dict[str, Any], *keys: str) -> dict[str, Any]:
        for key in keys:
            value = source.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _keyword_text(self, values: list[Any], fallback: str = "정보 없음") -> str:
        texts: list[str] = []
        for value in values:
            if isinstance(value, dict):
                text = value.get("keywordText") or value.get("keyword_text") or value.get("keyword") or value.get("text") or value.get("name")
            else:
                text = value
            if text is not None and str(text).strip():
                texts.append(str(text).strip())
        return ", ".join(texts[:6]) if texts else fallback
