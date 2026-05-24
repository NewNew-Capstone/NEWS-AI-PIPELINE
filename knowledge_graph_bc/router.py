from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from knowledge_graph_bc.config import Neo4jConfigError
from knowledge_graph_bc.keyword_expander import MultilingualKeywordExpander
from knowledge_graph_bc.neo4j_client import get_neo4j_client
from knowledge_graph_bc.realtime_ingest_service import (
    RealtimeIngestJobNotFoundError,
    RealtimeIngestService,
)
from knowledge_graph_bc.schemas import (
    ComparisonGraphResponse,
    ComparisonHomeResponse,
    ExpandedKeywords,
    MultilingualKeywordExpandRequest,
    MultilingualKeywordExpandResponse,
    RealtimeClickedVideoIngestRequest,
    RealtimeClickedVideoIngestResponse,
    RealtimeIngestJobResponse,
    RealtimeIngestRequest,
    RealtimeIngestResponse,
    SearchVideosResponse,
)
from knowledge_graph_bc.service import KnowledgeGraphComparisonService, VideoNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["knowledge_graph"])


def _service_unavailable(exc: Exception) -> HTTPException:
    logger.warning("Neo4j operation failed: %s", exc, exc_info=True)
    return HTTPException(
        status_code=503,
        detail={
            "status": "failed",
            "message": str(exc),
        },
    )


@router.get("/health")
def health() -> dict:
    try:
        return get_neo4j_client().verify_connectivity()
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.post("/schema/init")
def init_schema() -> dict:
    try:
        return get_neo4j_client().init_schema()
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.get("/comparison-home", response_model=ComparisonHomeResponse)
def comparison_home(limit: int = 5) -> ComparisonHomeResponse:
    try:
        return KnowledgeGraphComparisonService().get_comparison_home(limit=limit)
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.get("/search-videos", response_model=SearchVideosResponse)
def search_videos(keyword: str, limit: int = 5, scope: str = "all") -> SearchVideosResponse:
    try:
        return KnowledgeGraphComparisonService().search_videos(keyword=keyword, limit=limit, scope=scope)
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.get("/videos/{video_id}/comparison-graph", response_model=ComparisonGraphResponse)
def comparison_graph(video_id: str, limit_per_country: int = 5, scope: str = "all") -> ComparisonGraphResponse:
    try:
        return KnowledgeGraphComparisonService().get_comparison_graph(
            video_id=video_id,
            limit_per_country=limit_per_country,
            scope=scope,
        )
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"status": "failed", "message": str(exc)}) from exc
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.post("/realtime-ingest/search-candidates", response_model=RealtimeIngestResponse)
def realtime_ingest_search_candidates(
    request: RealtimeIngestRequest,
    background_tasks: BackgroundTasks,
) -> RealtimeIngestResponse:
    try:
        service = RealtimeIngestService()
        response, queued_candidates = service.prepare_search_candidate_job(request)
        if queued_candidates:
            background_tasks.add_task(
                service.process_search_candidates,
                request_id=response.request_id,
                keyword=request.keyword,
                candidates=queued_candidates,
            )
        return response
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.post("/realtime-ingest/clicked-video", response_model=RealtimeClickedVideoIngestResponse)
def realtime_ingest_clicked_video(
    request: RealtimeClickedVideoIngestRequest,
    background_tasks: BackgroundTasks,
) -> RealtimeClickedVideoIngestResponse:
    try:
        service = RealtimeIngestService()
        response, queued_candidates = service.prepare_clicked_video_job(request)
        if queued_candidates:
            background_tasks.add_task(
                service.process_search_candidates,
                request_id=response.request_id,
                keyword=request.keyword,
                candidates=queued_candidates,
            )
        return response
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.get("/realtime-ingest/jobs/{request_id}", response_model=RealtimeIngestJobResponse)
def realtime_ingest_job(request_id: str) -> RealtimeIngestJobResponse:
    try:
        return RealtimeIngestService().get_job(request_id)
    except RealtimeIngestJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"status": "failed", "message": str(exc)}) from exc
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.post("/expand-multilingual-keywords", response_model=MultilingualKeywordExpandResponse)
def expand_multilingual_keywords(
    payload: MultilingualKeywordExpandRequest,
) -> MultilingualKeywordExpandResponse:
    try:
        expanded = MultilingualKeywordExpander().expand(
            payload.keyword_ko,
            max_terms_per_language=payload.max_terms_per_language,
        )
        return MultilingualKeywordExpandResponse(
            requested_keyword=expanded.requested_keyword,
            expanded_keywords=ExpandedKeywords(
                ko=expanded.ko,
                en=expanded.en,
                zh=expanded.zh,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"status": "failed", "message": str(exc)}) from exc
    except Exception as exc:
        logger.warning("Keyword expansion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "failed",
                "message": str(exc),
            },
        ) from exc
