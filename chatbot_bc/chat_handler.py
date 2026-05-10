from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chatbot_bc.schemas import MessageDto

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
당신은 뉴스 편향 분석 플랫폼의 AI 상담사 '뉴스봇'입니다.

역할:
- 뉴스 편향 분석 결과를 쉽게 설명해 드립니다.
- 편향 점수(의견성·감정성·제목-본문 괴리)의 의미를 안내합니다.
- 국가별 보도 시각 차이를 중립적으로 설명합니다.
- 미디어 리터러시와 관련된 질문에 답합니다.

원칙:
- 항상 중립적인 입장을 유지합니다.
- 특정 언론사나 정치적 입장을 편들지 않습니다.
- 모르는 내용은 모른다고 솔직하게 답합니다.
- 이 서비스의 편향 분석 지표는 전체 편향 점수, 의견성 점수, 감정성 점수, 제목-본문 괴리 점수입니다.
- 익명출처 점수처럼 이 서비스에서 사용하지 않는 지표는 설명에 포함하지 않습니다.
- 한국어로 간결하고 친절하게 답변합니다.
""".strip()


def _format_list_items(items: Any, key: str) -> str:
    if not isinstance(items, list):
        return ""

    values = []
    for item in items[:8]:
        if isinstance(item, dict):
            value = item.get(key) or item.get("title") or item.get("description") or item.get("source_text")
        else:
            value = item

        if value:
            values.append(str(value))

    return ", ".join(values)


def build_analysis_prompt(analysis_context: dict[str, Any] | None) -> str:
    if not analysis_context:
        return SYSTEM_PROMPT

    keywords = _format_list_items(analysis_context.get("keywords"), "keyword_text")
    evidences = _format_list_items(analysis_context.get("evidences"), "description")

    context_lines = [
        "[분석 대상 영상 데이터]",
        f"- 전체 편향 점수: {analysis_context.get('overall_bias_score', '정보 없음')}",
        f"- 의견성 점수: {analysis_context.get('opinion_score', '정보 없음')}",
        f"- 감정성 점수: {analysis_context.get('emotion_score', '정보 없음')}",
        f"- 제목-본문 괴리 점수: {analysis_context.get('headline_body_gap_score', '정보 없음')}",
        f"- 논조: {analysis_context.get('tone_label', '정보 없음')}",
        f"- 요약: {analysis_context.get('summary_text', '정보 없음')}",
        f"- 관점 요약: {analysis_context.get('perspective_summary', '정보 없음')}",
        f"- 편향 근거 요약: {analysis_context.get('evidence_summary', '정보 없음')}",
    ]

    if keywords:
        context_lines.append(f"- 주요 키워드: {keywords}")

    if evidences:
        context_lines.append(f"- 편향 근거: {evidences}")

    context_lines.append("")
    context_lines.append("위 데이터를 바탕으로 사용자 질문에 답변하세요.")

    return f"{SYSTEM_PROMPT}\n\n" + "\n".join(context_lines)


class ChatHandler:
    def __init__(self) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = "claude-sonnet-4-6"

    def chat(self, messages: list["MessageDto"], analysis_context: dict[str, Any] | None = None) -> str:
        try:
            claude_messages = [
                {
                    "role": "user" if m.role.lower() == "user" else "assistant",
                    "content": m.content,
                }
                for m in messages
            ]

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=build_analysis_prompt(analysis_context),
                messages=claude_messages,
            )

            block = response.content[0]
            from anthropic.types import TextBlock

            if not isinstance(block, TextBlock):
                raise ValueError(f"Unexpected content block type: {type(block)}")

            return block.text.strip()

        except Exception as e:
            logger.error("ChatHandler.chat 실패: %s", e, exc_info=True)
            raise
