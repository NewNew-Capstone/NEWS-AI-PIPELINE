from __future__ import annotations

import logging

from anthropic import Anthropic
from anthropic.types import TextBlock

from chatbot_bc.schemas import MessageDto

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
당신은 뉴스 편향 분석 플랫폼의 AI 상담사 '뉴스봇'입니다.

역할:
- 뉴스 편향 분석 결과를 쉽게 설명해 드립니다.
- 편향 점수(의견성·감정성·익명출처)의 의미를 안내합니다.
- 국가별 보도 시각 차이를 중립적으로 설명합니다.
- 미디어 리터러시와 관련된 질문에 답합니다.

원칙:
- 항상 중립적인 입장을 유지합니다.
- 특정 언론사나 정치적 입장을 편들지 않습니다.
- 모르는 내용은 모른다고 솔직하게 답합니다.
- 한국어로 간결하고 친절하게 답변합니다.
""".strip()


class ChatHandler:
    def __init__(self) -> None:
        self.client = Anthropic()
        self.model = "claude-sonnet-4-6"

    def chat(self, messages: list[MessageDto]) -> str:
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
                system=SYSTEM_PROMPT,
                messages=claude_messages,
            )

            block = response.content[0]
            if not isinstance(block, TextBlock):
                raise ValueError(f"Unexpected content block type: {type(block)}")

            return block.text.strip()

        except Exception as e:
            logger.error("ChatHandler.chat 실패: %s", e, exc_info=True)
            raise
