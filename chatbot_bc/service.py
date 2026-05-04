from __future__ import annotations

import logging

from chatbot_bc.chat_handler import ChatHandler
from chatbot_bc.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatbotService:
    def __init__(self) -> None:
        self.handler = ChatHandler()

    def chat(self, request: ChatRequest) -> ChatResponse:
        reply = self.handler.chat(request.messages)
        return ChatResponse(reply=reply)
