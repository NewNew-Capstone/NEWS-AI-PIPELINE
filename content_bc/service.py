# transcript_loader 호출하고, 성공/실패 처리해서 응답 DTO로 만들어줌.
import logging

from content_bc.modules.transcript_loader import load_transcript
from content_bc.modules.video_ranker import rank_by_cosine
from content_bc.schemas import TranscriptResponseDto, VideoRankRequest, VideoRankResponse

logger = logging.getLogger(__name__)


class ContentService:
    def get_transcript(self, video_id: str, region_code: str = "US", priority: bool = False) -> TranscriptResponseDto:
        try:
            text = load_transcript(video_id, region_code, priority=priority)
            if not text.strip():
                logger.warning(
                    "transcript empty after load video_id=%s region=%s",
                    video_id,
                    region_code,
                )
                return TranscriptResponseDto(
                    video_id=video_id,
                    transcript="",
                    transcript_status="failed",
                )
            return TranscriptResponseDto(
                video_id=video_id,
                transcript=text,
                transcript_status="success",
            )
        except Exception:
            logger.warning(
                "transcript load failed video_id=%s region=%s",
                video_id,
                region_code,
                exc_info=True,
            )
            return TranscriptResponseDto(
                video_id=video_id,
                transcript="",
                transcript_status="failed",
            )

    def rank_videos(self, request: VideoRankRequest) -> VideoRankResponse:
        ranked_ids = rank_by_cosine(
            keyword=request.keyword,
            videos=[v.model_dump() for v in request.videos],
            top_n=request.top_n,
        )
        return VideoRankResponse(ranked_video_ids=ranked_ids)
