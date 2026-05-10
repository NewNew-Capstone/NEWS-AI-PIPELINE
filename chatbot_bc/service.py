from __future__ import annotations

import logging

from chatbot_bc.chat_handler import ChatHandler
from chatbot_bc.intent_parser import ChatIntentParser
from chatbot_bc.schemas import ChatIntentRequest, ChatIntentResponse, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatbotService:
    def __init__(self) -> None:
        self.handler = ChatHandler()
        self.intent_parser = ChatIntentParser()

    def chat(self, request: ChatRequest) -> ChatResponse:
        reply = self.handler.chat(request.messages, request.analysis_context)
        return ChatResponse(reply=reply)

    def parse_intent(self, request: ChatIntentRequest) -> ChatIntentResponse:
        return self.intent_parser.parse(
            request.message,
            has_pending_candidates=request.has_pending_candidates,
            has_page_video=request.has_page_video,
        )
