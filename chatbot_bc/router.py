from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from chatbot_bc.schemas import ChatIntentRequest, ChatIntentResponse, ChatRequest, ChatResponse
from chatbot_bc.service import ChatbotService

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@lru_cache(maxsize=1)
def get_chatbot_service() -> ChatbotService:
    return ChatbotService()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return get_chatbot_service().chat(request)


@router.post("/intent", response_model=ChatIntentResponse)
def parse_intent(request: ChatIntentRequest) -> ChatIntentResponse:
    return get_chatbot_service().parse_intent(request)
