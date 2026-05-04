from pydantic import BaseModel


class MessageDto(BaseModel):
    role: str  # "user" | "bot"
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageDto]


class ChatResponse(BaseModel):
    reply: str
