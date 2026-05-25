# GET /content/transcript?video_id=xxx&region_code=KR 엔드포인트 정의. 
# Spring이 여기로 HTTP 요청 보내면 됨.
from fastapi import APIRouter

from content_bc.schemas import (
    TranscriptResponseDto,
    VideoClusterRequest,
    VideoClusterResponse,
    VideoRankRequest,
    VideoRankResponse,
)
from content_bc.service import ContentService

router = APIRouter(prefix="/content", tags=["content"])


@router.get("/transcript", response_model=TranscriptResponseDto)
def get_transcript(video_id: str, region_code: str = "US", priority: bool = False) -> TranscriptResponseDto:
    return ContentService().get_transcript(video_id, region_code, priority=priority)


@router.post("/rank-videos", response_model=VideoRankResponse)
def rank_videos(request: VideoRankRequest) -> VideoRankResponse:
    return ContentService().rank_videos(request)


@router.post("/cluster-videos", response_model=VideoClusterResponse)
def cluster_videos(request: VideoClusterRequest) -> VideoClusterResponse:
    return ContentService().cluster_videos(request)
