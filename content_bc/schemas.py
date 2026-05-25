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


class VideoRankItem(BaseModel):
    video_id: str
    score: float


class VideoRankResponse(BaseModel):
    ranked_videos: list[VideoRankItem]


class VideoClusterRequest(BaseModel):
    videos: list[VideoItem]
    n_clusters: int | None = None


class VideoClusterResult(BaseModel):
    cluster_id: int
    video_ids: list[str]


class VideoClusterResponse(BaseModel):
    clusters: list[VideoClusterResult]
