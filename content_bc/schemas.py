from pydantic import BaseModel


class TranscriptResponseDto(BaseModel):
    video_id: str
    transcript: str
    transcript_status: str
