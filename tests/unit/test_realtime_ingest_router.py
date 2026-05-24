from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

import knowledge_graph_bc.router as router_mod
from knowledge_graph_bc.realtime_ingest_service import RealtimeIngestJobNotFoundError
from knowledge_graph_bc.schemas import (
    RealtimeClickedVideoIngestRequest,
    RealtimeClickedVideoIngestResponse,
    RealtimeIngestCandidate,
    RealtimeIngestJobResponse,
    RealtimeIngestRequest,
    RealtimeIngestResponse,
    SearchVideosResponse,
)


def _request() -> RealtimeIngestRequest:
    return RealtimeIngestRequest(
        keyword="트럼프 대만",
        candidates=[
            RealtimeIngestCandidate(
                video_id="kr1",
                title="트럼프 대만",
                country_code="KR",
                language="ko",
            )
        ],
    )


def test_realtime_ingest_endpoint_returns_immediate_response_and_queues_background_task() -> None:
    request = _request()
    response = RealtimeIngestResponse(
        request_id="rt-test",
        queued_count=1,
        skipped_existing_count=0,
        current_results=SearchVideosResponse(
            applied_cluster_type="ALL_AVAILABLE",
            keyword="트럼프 대만",
            sections=[],
        ),
    )
    service = MagicMock()
    service.prepare_search_candidate_job.return_value = (response, request.candidates)
    background_tasks = BackgroundTasks()

    with patch("knowledge_graph_bc.router.RealtimeIngestService", return_value=service):
        result = router_mod.realtime_ingest_search_candidates(request, background_tasks)

    assert result == response
    service.prepare_search_candidate_job.assert_called_once_with(request)
    assert len(background_tasks.tasks) == 1


def test_realtime_ingest_endpoint_does_not_queue_when_no_candidates_remaining() -> None:
    request = _request()
    response = RealtimeIngestResponse(
        request_id="rt-test",
        queued_count=0,
        skipped_existing_count=1,
        current_results=SearchVideosResponse(
            applied_cluster_type="ALL_AVAILABLE",
            keyword="트럼프 대만",
            sections=[],
        ),
    )
    service = MagicMock()
    service.prepare_search_candidate_job.return_value = (response, [])
    background_tasks = BackgroundTasks()

    with patch("knowledge_graph_bc.router.RealtimeIngestService", return_value=service):
        result = router_mod.realtime_ingest_search_candidates(request, background_tasks)

    assert result == response
    assert background_tasks.tasks == []


def test_realtime_clicked_video_endpoint_returns_immediate_response_and_queues_background_task() -> None:
    request = RealtimeClickedVideoIngestRequest(
        keyword="트럼프 대만",
        selected_video=RealtimeIngestCandidate(
            video_id="kr1",
            title="트럼프 대만",
            country_code="KR",
            language="ko",
        ),
        related_candidates=[],
    )
    response = RealtimeClickedVideoIngestResponse(
        request_id="rt-click",
        selected_video_id="kr1",
        queued_count=1,
        skipped_existing_count=0,
        current_graph=None,
    )
    service = MagicMock()
    service.prepare_clicked_video_job.return_value = (response, [request.selected_video])
    background_tasks = BackgroundTasks()

    with patch("knowledge_graph_bc.router.RealtimeIngestService", return_value=service):
        result = router_mod.realtime_ingest_clicked_video(request, background_tasks)

    assert result == response
    service.prepare_clicked_video_job.assert_called_once_with(request)
    assert len(background_tasks.tasks) == 1


def test_realtime_ingest_job_endpoint_returns_service_response() -> None:
    response = RealtimeIngestJobResponse(
        request_id="rt-test",
        keyword="트럼프 대만",
        status="RUNNING",
        queued_count=1,
    )
    service = MagicMock()
    service.get_job.return_value = response

    with patch("knowledge_graph_bc.router.RealtimeIngestService", return_value=service):
        result = router_mod.realtime_ingest_job("rt-test")

    assert result == response
    service.get_job.assert_called_once_with("rt-test")


def test_realtime_ingest_job_endpoint_maps_missing_job_to_404() -> None:
    service = MagicMock()
    service.get_job.side_effect = RealtimeIngestJobNotFoundError("missing")

    with patch("knowledge_graph_bc.router.RealtimeIngestService", return_value=service):
        with pytest.raises(HTTPException) as exc_info:
            router_mod.realtime_ingest_job("missing")

    assert exc_info.value.status_code == 404
