from pydantic import BaseModel


# api 응답 형태 정의
class TranscriptResponseDto(BaseModel):
    video_id: str
    transcript: str
    transcript_status: str


# 영상 랭킹 요청/응답 스키마
class VideoItem(BaseModel):
    video_id: str
    title: str
    description: str


class VideoRankRequest(BaseModel):
    keyword: str
    videos: list[VideoItem]
    top_n: int = 20


class VideoRankResponse(BaseModel):
    ranked_video_ids: list[str]
