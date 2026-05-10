from typing import Any

from pydantic import BaseModel


class MessageDto(BaseModel):
    role: str  # "user" | "bot"
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageDto]
    analysis_context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    reply: str


class ChatIntentRequest(BaseModel):
    message: str
    has_pending_candidates: bool = False
    has_page_video: bool = False


class ChatIntentResponse(BaseModel):
    intent: str
    keywords: list[str] = []
    sort_by: str | None = None
    sort_direction: str = "DESC"
    candidate_index: int | None = None
    answer_task: str | None = None
    limit: int = 5
    confidence: float = 0.0
