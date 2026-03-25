from pydantic import BaseModel

# api 응답 형태 정의 
class TranscriptResponseDto(BaseModel):
    video_id: str
    transcript: str
    transcript_status: str
